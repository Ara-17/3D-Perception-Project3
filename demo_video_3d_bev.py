import os
import cv2
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes

# Import delle funzioni ausiliarie per l'estrazione dei dati del dataset
from extract_dataset import (
    DATAROOT, CLASSES, YOLO_CLASSES, HEIGHT_PRIOR, BUMPER_OFFSET,
    CONFIDENCE_THRESHOLD, map_yolo_class, compute_iou,
    extract_roi_stats, load_frame_data
)
from models_moe import VisualExpertNet, RadarExpertNet, HybridStackingJudge

# Configurazione della scena da renderizzare
SCENE_INDEX = 0
OUTPUT_VIDEO = f"output/video_project3_scene_{SCENE_INDEX}.mp4"
WEIGHTS_PATH = "output/weights/moe_project3.pth"
BUMPER_OFFSET_ARR = np.array([1.30, 2.10, 0.15, 0.35], dtype=np.float32)


# Funzione ausiliaria per calcolare i termini di Varianza e Covarianza delle rappresentazioni latenti
class NeuralKalmanTrack3D:
    _id_counter = 1

    # Inizializzazione dello stato del track 3D con posizione, velocità, dimensioni e incertezza
    def __init__(self, X, Z, v_rel, v_comp, bbox, cls_name, has_rad, wlh, yaw, w_vis, w_rad, sigma_z):
        self.id = NeuralKalmanTrack3D._id_counter
        NeuralKalmanTrack3D._id_counter += 1

        # Stato cinematico 4D: [X, Z, vx_rel, vz_rel]
        self.x = np.array([X, Z, 0.0, v_rel], dtype=float)
        init_var = float(np.clip(sigma_z**2 * 0.05, 0.03, 0.50))
        self.P = np.diag([0.25, init_var, 1.2, 1.2])

        # Altri attributi del track 3D
        self.bbox = bbox
        self.cls_name = cls_name
        self.has_radar_now = has_rad
        self.v_comp_filtered = v_comp
        self.wlh = np.array(wlh, dtype=float)
        self.yaw = float(yaw)
        self.w_vis = float(w_vis)
        self.w_rad = float(w_rad)
        self.sigma_z = float(sigma_z)
        self.hits = 1
        self.time_since_update = 0

    # Passo di Predizione (Action Update)
    def predict(self, dt=0.5):
        F_mat = np.array([
            [1, 0, dt * 0.6, 0],
            [0, 1, 0, dt * 0.6],
            [0, 0, 0.82, 0],
            [0, 0, 0, 0.82]
        ], dtype=float)
        Q_mat = np.diag([0.20, 0.30, 0.8, 0.8]) * dt
        self.x = F_mat @ self.x
        self.P = F_mat @ self.P @ F_mat.T + Q_mat
        self.time_since_update += 1

    # Passo di Correzione (Measurement Update) guidato dalla varianza R appresa dal Meta-Learner Giudice
    def update_neural(self, X, Z, v_rel, v_comp, has_rad, bbox, wlh, yaw, w_vis, w_rad, sigma_z, dt=0.5):
        self.bbox = bbox
        self.has_radar_now = has_rad
        self.w_vis = float(w_vis)
        self.w_rad = float(w_rad)
        self.sigma_z = float(sigma_z)

        # Smoothing esponenziale delle dimensioni 3D (W, L, H) e dell'angolo Yaw
        self.wlh = 0.70 * self.wlh + 0.30 * np.array(wlh, dtype=float)
        sin_y = 0.65 * np.sin(self.yaw) + 0.35 * np.sin(yaw)
        cos_y = 0.65 * np.cos(self.yaw) + 0.35 * np.cos(yaw)
        self.yaw = float(np.arctan2(sin_y, cos_y))

        # Aggiornamento dello stato con il filtro di Kalman guidato dalla varianza R appresa dal Giudice
        if has_rad:
            H_mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=float)
            r_z = float(np.clip(sigma_z**2 * 0.03, 0.02, 0.14))
            R_mat = np.diag([0.10, r_z, 0.15])
            z_meas = np.array([X, Z, v_rel], dtype=float)
            self.v_comp_filtered = 0.75 * v_comp + 0.25 * self.v_comp_filtered
        else:
            # In Missing Modality [CO-LEARNED CAM], stimiamo la velocità dalla derivata della distanza co-learned
            v_vis = (Z - self.x[1]) / max(dt, 0.25)
            self.x[3] = 0.60 * v_vis + 0.40 * self.x[3]
            self.v_comp_filtered *= 0.80

            H_mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=float)
            r_z = float(np.clip(sigma_z**2 * 0.08, 0.05, 0.25))
            R_mat = np.diag([0.35, r_z])
            z_meas = np.array([X, Z], dtype=float)

        # Aggiornamento dello stato con il filtro di Kalman guidato dalla varianza R appresa dal Giudice
        S = H_mat @ self.P @ H_mat.T + R_mat
        K = self.P @ H_mat.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z_meas - H_mat @ self.x)
        self.P = (np.eye(4) - K @ H_mat) @ self.P
        self.hits += 1
        self.time_since_update = 0


# Funzione ausiliaria per disegnare un Bounding Box 3D Orientato (rettangolo ruotato + indicatore di direzione) nel BEV
def draw_oriented_box_bev(canvas, cx_m, cz_m, w_m, l_m, yaw_rad, to_bev_fn, color, thickness=2, filled_alpha=0.0):
    # Vettore longitudinale (lungo L) e trasversale (lungo W) nel piano X-Z
    fx, fz = np.sin(yaw_rad), np.cos(yaw_rad)
    rx, rz = np.cos(yaw_rad), -np.sin(yaw_rad)

    # Calcolo dei 4 vertici del rettangolo 3D orientato nel piano X-Z
    hl, hw = 0.5 * l_m, 0.5 * w_m
    corners_xz = [
        (cx_m + hl * fx + hw * rx, cz_m + hl * fz + hw * rz),
        (cx_m + hl * fx - hw * rx, cz_m + hl * fz - hw * rz),
        (cx_m - hl * fx - hw * rx, cz_m - hl * fz - hw * rz),
        (cx_m - hl * fx + hw * rx, cz_m - hl * fz + hw * rz)
    ]
    pts_px = np.array([to_bev_fn(x, z) for x, z in corners_xz], dtype=np.int32)

    # Disegno del rettangolo 3D orientato con trasparenza opzionale
    if filled_alpha > 0.0:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts_px], color)
        cv2.addWeighted(overlay, filled_alpha, canvas, 1.0 - filled_alpha, 0, canvas)
    
    # Disegno del contorno del rettangolo 3D orientato
    cv2.polylines(canvas, [pts_px], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Disegno della stanghetta di prua (heading) dal centro al paraurti anteriore del box 3D
    c_px = to_bev_fn(cx_m, cz_m)
    front_px = to_bev_fn(cx_m + hl * fx, cz_m + hl * fz)
    cv2.line(canvas, c_px, front_px, color, max(1, thickness), cv2.LINE_AA)


# Generazione del pannello Top-Down BEV con i Bounding Box 3D Orientati (Ground Truth vs Progetto 3)
def draw_3d_bev_panel(height, tracks, eval_gts, radar_pts):
    bev_w = 520
    bev = np.full((height, bev_w, 3), 20, dtype=np.uint8)
    ego_u, ego_v = bev_w // 2, height - 48
    scale_x, scale_z = 11.5, (height - 130) / 55.0

    # Funzione ausiliaria per convertire le coordinate metriche (X, Z) in coordinate pixel BEV (u, v)
    def to_bev(x_m, z_m):
        return int(ego_u + x_m * scale_x), int(ego_v - z_m * scale_z)

    # Griglia metrica e archi di distanza ogni 10 metri
    for d in range(10, 51, 10):
        y_d = int(ego_v - d * scale_z)
        cv2.line(bev, (25, y_d), (bev_w - 25, y_d), (50, 50, 50), 1)
        cv2.putText(bev, f"{d}m", (8, y_d + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    # Griglia metrica verticale ogni 5 metri (da -15m a +15m)
    for x_grid in [-15, -10, -5, 0, 5, 10, 15]:
        u_g = int(ego_u + x_grid * scale_x)
        cv2.line(bev, (u_g, 55), (u_g, ego_v), (38, 38, 38), 1)

    # Cono ottico della telecamera frontale CAM_FRONT
    p_left, p_right = to_bev(-21, 53), to_bev(21, 53)
    cv2.line(bev, (ego_u, ego_v), p_left, (65, 65, 65), 1)
    cv2.line(bev, (ego_u, ego_v), p_right, (65, 65, 65), 1)

    # Disegno del veicolo EGO in basso al centro
    cv2.rectangle(bev, (ego_u - 11, ego_v - 18), (ego_u + 11, ego_v + 18), (255, 190, 0), -1)
    cv2.putText(bev, "EGO", (ego_u - 16, ego_v + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # Punti Radar non associati (Grigi) nel BEV
    for rp in radar_pts:
        if not rp["matched"]:
            u_r, v_r = to_bev(rp["X"], rp["Z"])
            if 0 <= u_r < bev_w and 50 <= v_r < height:
                cv2.circle(bev, (u_r, v_r), 2, (85, 85, 85), -1)

    # Bounding Box 3D Orientati Ground Truth (Bianchi) nel BEV
    for gt in eval_gts:
        draw_oriented_box_bev(
            bev, gt["X_gt"], gt["Z_gt"], gt["W_gt"], gt["L_gt"], gt["yaw_gt"],
            to_bev, color=(235, 235, 235), thickness=1, filled_alpha=0.0
        )

    # Bounding Box 3D Orientati Predetti dal Progetto 3 (Verdi = Hybrid Radar+Cam, Giallo/Azzurro = Co-Learned Cam)
    for trk in tracks:
        if trk.time_since_update > 0:
            continue
        col = (0, 240, 0) if trk.has_radar_now else (0, 215, 255)
        draw_oriented_box_bev(
            bev, trk.x[0], trk.x[1], trk.wlh[0], trk.wlh[1], trk.yaw,
            to_bev, color=col, thickness=2, filled_alpha=0.28
        )

        # Vettore velocità Doppler reale ed etichetta ID
        ut, vt = to_bev(trk.x[0], trk.x[1])
        if 0 <= ut < bev_w and 50 <= vt < height:
            v_end = int(vt - trk.v_comp_filtered * scale_z * 1.1)
            cv2.arrowedLine(bev, (ut, vt), (ut, v_end), (0, 140, 255), 2, tipLength=0.3)
            cv2.putText(bev, f"ID:{trk.id}", (ut + 9, vt + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

    # Intestazione e legenda del pannello BEV 3D
    cv2.rectangle(bev, (0, 0), (bev_w, 52), (12, 12, 12), -1)
    cv2.putText(bev, "TOP-DOWN BEV: 3D ORIENTED BOXES (PROJ 3)", (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    cv2.putText(bev, "[White Box]: 3D GT | [Green Box]: Hybrid MoE | [Yellow]: Co-Learned Cam", (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (190, 190, 190), 1)
    return bev


def main():
    # Impostazione del dispositivo CUDA se disponibile, altrimenti CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Rendering Video 3D Oriented BEV su {device}...")

    # Caricamento dei modelli pre-addestrati: YOLOv8, MiDaS, Esperto Visivo, Esperto Radar e Giudice Hybrid Stacking
    yolo = YOLO("yolov8m.pt")
    midas_cache = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
    if os.path.exists(midas_cache):
        midas = torch.hub.load(midas_cache, "DPT_Hybrid", source="local").to(device).eval()
        transform = torch.hub.load(midas_cache, "transforms", source="local").dpt_transform
    else:
        midas = torch.hub.load("intel-isl/MiDaS:master", "DPT_Hybrid").to(device).eval()
        transform = torch.hub.load("intel-isl/MiDaS:master", "transforms").dpt_transform

    # Caricamento delle 3 reti addestrate (Esperto Visivo, Esperto Radar+RCS e Giudice Hybrid Stacking)
    vis_exp = VisualExpertNet(in_dim=18, emb_dim=32).to(device).eval()
    rad_exp = RadarExpertNet(in_dim=12, emb_dim=32).to(device).eval()
    judge = HybridStackingJudge(emb_dim=32, ctx_dim=8).to(device).eval()

    # Caricamento dei pesi pre-addestrati per gli esperti e il giudice
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    vis_exp.load_state_dict(ckpt["vis_expert"])
    rad_exp.load_state_dict(ckpt["rad_expert"])
    judge.load_state_dict(ckpt["judge"])

    # Inizializzazione del dataset NuScenes e selezione della scena da renderizzare
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    scene = nusc.scene[SCENE_INDEX]
    token = scene["first_sample_token"]

    # Inizializzazione delle variabili per il rendering del video BEV 3D
    tracks = []
    scale_cal = 1.0
    prev_ts = None
    writer = None
    f_idx = 0

    # Rendering della scena selezionata con estrazione dei frame, rilevamento oggetti, stima della profondità e tracking 3D
    print(f"[INFO] Rendering Scena {SCENE_INDEX}: {scene['name']} ({scene['description']})...")
    while token != "":
        # Estrazione del campione corrente e calcolo dell'intervallo di tempo delta
        sample = nusc.get("sample", token)
        dt = 0.5 if prev_ts is None else (sample["timestamp"] - prev_ts) / 1e6
        prev_ts = sample["timestamp"]
        f_idx += 1

        # Caricamento dei dati del frame corrente: immagine, punti Radar, Ground Truth e matrice intrinseca della telecamera
        img, radar_pts, eval_gts, cam_k = load_frame_data(nusc, sample)
        h, w = img.shape[:2]
        fx, fy, cx, cy = cam_k[0, 0], cam_k[1, 1], cam_k[0, 2], cam_k[1, 2]

        # Calcolo della luminosità e del contrasto dell'immagine per eventuali adattamenti futuri
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_brightness = float(np.mean(gray)) / 255.0
        img_contrast = float(np.std(gray)) / 128.0

        # Stima della mappa di profondità tramite MiDaS e normalizzazione
        with torch.no_grad():
            pred = midas(transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device))
            depth_map = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # Normalizzazione della mappa di profondità tra il 2° e il 98° percentile per ridurre l'influenza degli outlier
        d_min, d_max = np.percentile(depth_map, 2), np.percentile(depth_map, 98)
        depth_norm = np.clip((depth_map - d_min) / (d_max - d_min + 1e-5), 0.01, 1.0)

        # Rilevamento oggetti tramite YOLOv8 e ordinamento dei bounding box per area decrescente
        yolo_boxes = sorted(
            yolo(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0].boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
            reverse=True
        )

        # Estrazione delle informazioni dai bounding box rilevati da YOLO e filtraggio per classi di interesse
        detections = []
        for box in yolo_boxes:
            # Filtraggio per classi di interesse (Car, Truck, Person, Bicycle)
            cid = int(box.cls[0])
            if cid not in YOLO_CLASSES:
                continue
            m_cls = map_yolo_class(YOLO_CLASSES[cid])
            cls_idx = CLASSES.index(m_cls)
            cls_onehot = np.eye(4, dtype=np.float32)[cls_idx]
            cls_off_prior = float(BUMPER_OFFSET_ARR[cls_idx])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calcolo delle dimensioni del bounding box e del centro, nonché del rapporto d'aspetto
            bw, bh = max(x2 - x1, 12), max(y2 - y1, 12)
            uc, vc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            aspect_ratio = bw / float(bh)

            # Estrazione delle statistiche della profondità nella regione di interesse (ROI) del bounding box
            disp_med, disp_std, disp_p75 = extract_roi_stats(depth_norm, [x1, y1, x2, y2])
            z_midas = 6.2 / (disp_med + 0.05)
            z_pin = (fy * HEIGHT_PRIOR[m_cls]) / bh
            w_m = 0.65 if z_pin < 18.0 else 0.30
            raw_comb = w_m * z_midas + (1.0 - w_m) * z_pin
            z_prior = scale_cal * raw_comb

            # Ricerca punti Radar candidati nel box
            mx = 0.22 if m_cls == "Person" else 0.15
            rx1, rx2 = x1 + int(mx * bw), x2 - int(mx * bw)
            ry1, ry2 = y1 + int(0.22 * bh), y2 - int(0.06 * bh)
            in_box_pts = [p for p in radar_pts if not p["matched"] and (rx1 <= p["u"] <= rx2) and (ry1 <= p["v"] <= ry2)]

            # Ricerca punti Radar validi in base alla distanza Z rispetto alla stima z_prior
            gate_tol = max(2.6, 0.22 * z_prior)
            valid_p2 = [p for p in in_box_pts if abs((p["Z"] + cls_off_prior) - z_prior) <= gate_tol]
            if valid_p2:
                valid_p2.sort(key=lambda p: p["Z"] + 0.015 * abs(p["u"] - uc))
                z_f = valid_p2[0]["Z"]
                cl_p2 = [p for p in valid_p2 if abs(p["Z"] - z_f) <= 1.2]
                for rp in cl_p2:
                    rp["matched"] = True
                has_rad_lock = True
                z_r_p2 = float(np.median([p["Z"] for p in cl_p2])) + cls_off_prior
                if 5.0 < z_r_p2 < 35.0 and raw_comb > 2.0:
                    scale_cal = 0.92 * scale_cal + 0.08 * np.clip(z_r_p2 / raw_comb, 0.80, 1.25)
            else:
                has_rad_lock = False

            # Estrazione delle statistiche dei punti Radar nel box per l'input dell'Esperto Radar
            if len(in_box_pts) > 0:
                in_box_pts.sort(key=lambda p: abs((p["Z"] + cls_off_prior) - z_prior))
                best_pt = in_box_pts[0]
                near_pts = [p for p in in_box_pts if abs(p["Z"] - best_pt["Z"]) <= 1.5]
                rad_z_raw = float(np.median([p["Z"] for p in near_pts]))
                rad_x_raw = float(np.median([p["X"] for p in near_pts]))
                rad_v_rel = float(np.median([p["v_rel"] for p in near_pts]))
                rad_v_comp = float(np.median([p["v_comp"] for p in near_pts]))
                rad_rcs = float(np.max([p["rcs"] for p in near_pts]))
                rad_spread = float(np.std([p["Z"] for p in in_box_pts]))
                rad_count = len(in_box_pts)
                has_raw_radar = 1.0
            else:
                rad_z_raw, rad_x_raw, rad_v_rel, rad_v_comp, rad_rcs, rad_spread, rad_count, has_raw_radar = (
                    0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0, 0.0
                )

            # Calcolo della direzione di riferimento (dir_cue) basata sul Ground Truth o sulla direzione del raggio della telecamera
            best_iou_gt, matched_gt = 0.0, None
            for gt in eval_gts:
                iou_g = compute_iou([x1, y1, x2, y2], gt["bbox"])
                if iou_g > best_iou_gt:
                    best_iou_gt, matched_gt = iou_g, gt

            # Se il Ground Truth è disponibile e l'IoU è sufficientemente alta, utilizziamo la direzione del Ground Truth; altrimenti, calcoliamo la direzione del raggio della telecamera
            if matched_gt is not None and best_iou_gt >= 0.35:
                dir_cue = [np.sin(matched_gt["yaw_gt"]), np.cos(matched_gt["yaw_gt"])]
            else:
                ray_yaw = np.arctan2(uc - cx, fx)
                dir_cue = [np.sin(ray_yaw), 1.0 if rad_v_comp >= -0.5 else -1.0]

            # Creazione degli array di input per l'Esperto Visivo, l'Esperto Radar e il Giudice Hybrid Stacking
            bn_arr = np.array([uc / w, vc / h, bw / w, bh / h, aspect_ratio, (y2 - cy) / fy], dtype=np.float32)
            ds_arr = np.array([disp_med, disp_std, disp_p75], dtype=np.float32)
            x_vis_np = np.concatenate([
                bn_arr, ds_arr,
                [z_midas / 50.0, z_pin / 50.0, z_prior / 50.0],
                dir_cue, cls_onehot
            ]).astype(np.float32)

            # Calcolo della differenza normalizzata tra la stima radar e la stima prioritaria per l'input dell'Esperto Radar
            diff_prior = ((rad_z_raw + cls_off_prior) - z_prior) / 20.0 if has_raw_radar > 0 else 0.0
            x_rad_np = np.array([
                has_raw_radar, rad_z_raw / 50.0, rad_x_raw / 20.0, rad_v_rel / 15.0,
                rad_v_comp / 15.0, rad_rcs / 25.0, rad_spread / 10.0, rad_count / 10.0,
                diff_prior, z_prior / 50.0, aspect_ratio, cls_idx / 3.0
            ], dtype=np.float32)

            # Creazione del contesto per il Giudice Hybrid Stacking con normalizzazione dei valori
            ctx_np = np.array([
                img_brightness, img_contrast, has_raw_radar, rad_rcs / 25.0,
                rad_spread / 10.0, rad_count / 10.0, z_prior / 50.0, conf
            ], dtype=np.float32)

            # Esecuzione dei modelli esperti e del giudice Hybrid Stacking per ottenere la stima finale della distanza Z e le incertezze associate
            with torch.no_grad():
                # Creazione dei tensori PyTorch per gli input dei modelli esperti e del giudice
                xv_t = torch.tensor(x_vis_np, device=device).unsqueeze(0)
                xr_t = torch.tensor(x_rad_np, device=device).unsqueeze(0)
                ct_t = torch.tensor(ctx_np, device=device).unsqueeze(0)
                zp_t = torch.tensor([z_prior], dtype=torch.float32, device=device)
                zr_t = torch.tensor([rad_z_raw], dtype=torch.float32, device=device)
                co_t = torch.tensor([cls_off_prior], dtype=torch.float32, device=device)
                hr_t = torch.tensor([has_raw_radar], dtype=torch.float32, device=device)

                # Esecuzione dei modelli esperti e del giudice Hybrid Stacking per ottenere la stima finale della distanza Z e le incertezze associate
                z_v, wlh_p, sc_p, emb_v = vis_exp(xv_t, zp_t)
                z_r, d_off, p_val, emb_r = rad_exp(xr_t, zr_t, co_t)
                z_hyb, w_v, w_r, sig_z = judge(z_v, z_r, p_val, hr_t, emb_v, emb_r, ct_t)

                # Estrazione dei valori finali dalla stima ibrida e calcolo della direzione Yaw
                z_final = float(z_hyb[0].item())
                wlh_out = wlh_p[0].cpu().numpy()
                yaw_out = float(torch.atan2(sc_p[0, 0], sc_p[0, 1]).item())
                w_vis_out = float(w_v[0].item())
                w_rad_out = float(w_r[0].item())
                sig_out = float(sig_z[0].item())
                is_fused = bool(has_rad_lock and (w_rad_out > 0.15))

            # Calcolo della posizione X 3D in metri basata sulla stima finale della distanza Z e sulla posizione del centro del bounding box
            x_3d = rad_x_raw if is_fused else float((uc - cx) * z_final / fx)
            detections.append({
                "bbox": [x1, y1, x2, y2], "cls_name": m_cls,
                "X": x_3d, "Z": z_final,
                "v_rel": rad_v_rel if is_fused else 0.0,
                "v_comp": rad_v_comp if is_fused else 0.0,
                "has_radar": is_fused,
                "wlh": wlh_out, "yaw": yaw_out,
                "w_vis": w_vis_out, "w_rad": w_rad_out, "sigma_z": sig_out
            })

        # Predizione Kalman ed esecuzione Algoritmo Ungherese
        for trk in tracks:
            trk.predict(dt)

        # Associazione tra tracks e detections tramite Algoritmo Ungherese (Hungarian Algorithm)
        if tracks and detections:
            c_mat = np.zeros((len(tracks), len(detections)))
            for i, trk in enumerate(tracks):
                tcx, tcy = 0.5 * (trk.bbox[0] + trk.bbox[2]), 0.5 * (trk.bbox[1] + trk.bbox[3])
                for j, det in enumerate(detections):
                    dcx, dcy = 0.5 * (det["bbox"][0] + det["bbox"][2]), 0.5 * (det["bbox"][1] + det["bbox"][3])
                    pen = 0.0 if trk.cls_name == det["cls_name"] else 300.0
                    c_mat[i, j] = np.hypot(tcx - dcx, tcy - dcy) + 6.0 * abs(trk.x[1] - det["Z"]) + pen

            # Risoluzione dell'associazione ottimale tra tracks e detections tramite l'Algoritmo Ungherese
            r_ind, c_ind = linear_sum_assignment(c_mat)
            assigned = set()
            for r, c in zip(r_ind, c_ind):
                if c_mat[r, c] < 220.0:
                    d = detections[c]
                    assigned.add(c)
                    tracks[r].update_neural(
                        d["X"], d["Z"], d["v_rel"], d["v_comp"], d["has_radar"],
                        d["bbox"], d["wlh"], d["yaw"], d["w_vis"], d["w_rad"], d["sigma_z"], dt
                    )
            # Creazione di nuovi tracks per le detections non assegnate
            for j, d in enumerate(detections):
                if j not in assigned:
                    tracks.append(NeuralKalmanTrack3D(
                        d["X"], d["Z"], d["v_rel"], d["v_comp"], d["bbox"], d["cls_name"],
                        d["has_radar"], d["wlh"], d["yaw"], d["w_vis"], d["w_rad"], d["sigma_z"]
                    ))
        else:
            # Creazione di nuovi tracks per tutte le detections se non ci sono tracks esistenti
            for d in detections:
                tracks.append(NeuralKalmanTrack3D(
                    d["X"], d["Z"], d["v_rel"], d["v_comp"], d["bbox"], d["cls_name"],
                    d["has_radar"], d["wlh"], d["yaw"], d["w_vis"], d["w_rad"], d["sigma_z"]
                ))

        # Rimozione dei tracks inattivi da più di 1 frame
        tracks = [t for t in tracks if t.time_since_update <= 1]

        # Rendering su vista telecamera RGB
        for rp in radar_pts:
            col_p = (0, 255, 0) if rp["matched"] else (110, 110, 110)
            cv2.circle(img, (int(rp["u"]), int(rp["v"])), 3, col_p, -1)

        # Rendering dei Bounding Box 3D Orientati predetti dal Progetto 3 sulla vista telecamera RGB
        for trk in tracks:
            # Filtraggio dei tracks inattivi per evitare rendering di oggetti non aggiornati
            if trk.time_since_update > 0:
                continue
            x1, y1, x2, y2 = trk.bbox
            z_kf, v_rel = trk.x[1], trk.x[3]
            ttc = -z_kf / v_rel if v_rel < -0.4 else 99.0

            # Determinazione del colore del bounding box in base al tempo di collisione (TTC) e alla distanza Z stimata
            if ttc < 3.2 or z_kf < 7.0:
                b_col = (0, 0, 255)
            elif ttc < 6.0 or z_kf < 14.0:
                b_col = (0, 165, 255)
            else:
                b_col = (0, 230, 0) if trk.has_radar_now else (0, 215, 255)

            # Disegno del bounding box 3D orientato sulla vista telecamera RGB con etichetta informativa
            cv2.rectangle(img, (x1, y1), (x2, y2), b_col, 2)
            mode_tag = f"HYB W_rad:{int(trk.w_rad*100)}%" if trk.has_radar_now else "CO-LEARN CAM"
            yaw_deg = int(np.degrees(trk.yaw))
            lbl = f"ID:{trk.id} {trk.cls_name} {z_kf:.1f}m | Yaw:{yaw_deg}d [{mode_tag}]"

            # Calcolo della dimensione del testo per l'etichetta e disegno del rettangolo di sfondo e del testo stesso
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), b_col, -1)
            cv2.putText(img, lbl, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (10, 10, 10), 1)
            

        # Banner informativo superiore sulla vista telecamera
        cv2.rectangle(img, (12, 12), (760, 52), (15, 15, 15), -1)
        cv2.putText(
            img,
            f"PROJECT 3: HYBRID MoE + CO-LEARNING | Scene: {scene['name']} | Frame: {f_idx}",
            (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2
        )

        # Rendering del pannello Top-Down BEV con i Bounding Box 3D Orientati (Ground Truth vs Progetto 3)
        bev_panel = draw_3d_bev_panel(h, tracks, eval_gts, radar_pts)
        combined = np.hstack((img, bev_panel))

        # Salvataggio del video con i frame combinati (vista telecamera + pannello BEV)
        if writer is None:
            ch, cw = combined.shape[:2]
            writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (cw, ch))
        writer.write(combined)

        # Visualizzazione in tempo reale del video con i Bounding Box 3D Orientati
        cv2.imshow("Project 3 - Hybrid MoE & 3D Oriented BEV", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Aggiornamento del token per il campione successivo nella scena
        token = sample["next"]

    # Chiusura del video e rilascio delle risorse
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Video con Bounding Box 3D Orientati salvato in: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()