import os
import cv2
import torch
import numpy as np
from pyquaternion import Quaternion
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

# Configurazione delle directory locali del progetto
DATAROOT = r"data/v1.0-mini"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definizione delle 4 macro-classi e costanti fisiche di riferimento
CLASSES = ["Car", "Truck/Bus", "Person", "Bike/Moto"]
YOLO_CLASSES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
HEIGHT_PRIOR = {"Car": 1.58, "Truck/Bus": 2.85, "Person": 1.72, "Bike/Moto": 1.45}
BUMPER_OFFSET = {"Car": 1.30, "Truck/Bus": 2.10, "Person": 0.15, "Bike/Moto": 0.35}
CONFIDENCE_THRESHOLD = 0.40


# Funzione di mappatura tra le classi nuScenes e le 4 macro-classi
def map_nusc_class(cat_name):
    if "car" in cat_name:
        return "Car"
    if any(k in cat_name for k in ["truck", "bus", "construction", "trailer"]):
        return "Truck/Bus"
    if "pedestrian" in cat_name:
        return "Person"
    if any(k in cat_name for k in ["bicycle", "motorcycle"]):
        return "Bike/Moto"
    return None


# Funzione di mappatura tra le classi YOLO e le 4 macro-classi
def map_yolo_class(label):
    if label == "Car":
        return "Car"
    if label in ["Truck", "Bus"]:
        return "Truck/Bus"
    if label == "Person":
        return "Person"
    if label in ["Bicycle", "Motorcycle"]:
        return "Bike/Moto"
    return None


# Funzione per calcolare l'Intersection over Union (IoU) 2D tra due bounding box
def compute_iou(boxA, boxB):
    ix1, iy1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    ix2, iy2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-5)


# Estrazione di mediana, deviazione standard e 75° percentile della disparità MiDaS nella ROI centrale
def extract_roi_stats(depth_map, box, crop_ratio=0.50):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    cx1 = int(x1 + w * (1 - crop_ratio) / 2)
    cy1 = int(y1 + h * (1 - crop_ratio) / 2)
    cx2 = int(x2 - w * (1 - crop_ratio) / 2)
    cy2 = int(y2 - h * (1 - crop_ratio) / 2)
    roi = depth_map[max(0, cy1):max(1, cy2), max(0, cx1):max(1, cx2)]
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.median(roi)), float(np.std(roi)), float(np.percentile(roi, 75))


# Caricamento frame, punti Radar (con Doppler e RCS) e Ground Truth 3D (con dimensioni e angolo Yaw)
def load_frame_data(nusc, sample):
    # Estrazione dei token dei dati della telecamera frontale e del radar frontale
    cam_token = sample['data']['CAM_FRONT']
    radar_token = sample['data']['RADAR_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    radar_data = nusc.get('sample_data', radar_token)

    # Estrazione dei punti Radar 3D (X, Y, Z) e delle feature fisiche (v_rel, v_comp, RCS)
    pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, radar_data['filename']))
    r_dist = np.sqrt(pc.points[0, :]**2 + pc.points[1, :]**2) + 1e-5
    v_rel = (pc.points[0, :] * pc.points[6, :] + pc.points[1, :] * pc.points[7, :]) / r_dist
    v_comp = (pc.points[0, :] * pc.points[8, :] + pc.points[1, :] * pc.points[9, :]) / r_dist
    rcs_db = pc.points[5, :]  # Riga 5: Radar Cross-Section (riflettività metallica in dBsm)

    # Allineamento 3D dal paraurti (Radar) al parabrezza (Camera)
    cs_rad = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_rad['rotation']).rotation_matrix)
    pc.translate(np.array(cs_rad['translation']))

    # Allineamento 3D dal sistema di riferimento del veicolo (Ego) al sistema di riferimento della telecamera
    pos_rad = nusc.get('ego_pose', radar_data['ego_pose_token'])
    pc.rotate(Quaternion(pos_rad['rotation']).rotation_matrix)
    pc.translate(np.array(pos_rad['translation']))

    # Allineamento 3D dal sistema di riferimento della telecamera al sistema di riferimento del veicolo (Ego)
    pos_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
    rot_ego = Quaternion(pos_cam['rotation']).rotation_matrix
    pc.translate(-np.array(pos_cam['translation']))
    pc.rotate(rot_ego.T)

    # Allineamento 3D dal sistema di riferimento della telecamera al sistema di riferimento della telecamera (Calibrated Sensor)
    cs_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    rot_cam = Quaternion(cs_cam['rotation']).rotation_matrix
    pc.translate(-np.array(cs_cam['translation']))
    pc.rotate(rot_cam.T)

    # Proiezione dei punti Radar 3D sul piano dell'immagine della telecamera frontale
    cam_k = np.array(cs_cam['camera_intrinsic'])
    pts_2d = view_points(pc.points[:3, :], cam_k, normalize=True)

    # Filtraggio dei punti Radar proiettati fuori dall'immagine e con profondità non valida
    img = cv2.imread(os.path.join(nusc.dataroot, cam_data['filename']))
    h, w = img.shape[:2]
    x_3d, z_3d = pc.points[0, :], pc.points[2, :]

    # Creazione di un mascheramento booleano per selezionare solo i punti Radar validi
    mask = (z_3d > 2.0) & (z_3d < 55.0) & (pts_2d[0, :] >= 0) & (pts_2d[0, :] < w) & (pts_2d[1, :] >= 0) & (pts_2d[1, :] < h)
    radar_pts = [{
        "u": float(pts_2d[0, i]), "v": float(pts_2d[1, i]),
        "X": float(x_3d[i]), "Z": float(z_3d[i]),
        "v_rel": float(v_rel[i]), "v_comp": float(v_comp[i]),
        "rcs": float(rcs_db[i]), "matched": False
    } for i in np.where(mask)[0]]

    # Estrazione Ground Truth 3D certificato in coordinate della telecamera
    _, boxes_gt, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    eval_gts = []

    # Iterazione sui box 3D del Ground Truth per filtrare e salvare le informazioni rilevanti
    for box in boxes_gt:
        ann = nusc.get('sample_annotation', box.token)
        cls_name = map_nusc_class(box.name)
        z_gt, x_gt = float(box.center[2]), float(box.center[0])
        if cls_name is None or z_gt <= 1.5 or z_gt > 50.0 or int(ann['visibility_token']) < 2:
            continue

        # Proiezione dei vertici del box 3D sul piano dell'immagine della telecamera per calcolare il bounding box 2D
        c2d = view_points(box.corners(), cam_k, normalize=True)[:2, :]
        x1, y1 = int(np.clip(np.min(c2d[0]), 0, w - 1)), int(np.clip(np.min(c2d[1]), 0, h - 1))
        x2, y2 = int(np.clip(np.max(c2d[0]), 0, w - 1)), int(np.clip(np.max(c2d[1]), 0, h - 1))
        if (x2 - x1) < 22 or (y2 - y1) < 22:
            continue

        # Dimensioni 3D reali (Width, Length, Height) e angolo Yaw nel piano X-Z della telecamera
        w_3d, l_3d, h_3d = float(box.wlh[0]), float(box.wlh[1]), float(box.wlh[2])
        front_vec = box.orientation.rotation_matrix @ np.array([1.0, 0.0, 0.0])
        yaw_cam = float(np.arctan2(front_vec[0], front_vec[2]))

        # Calcolo della velocità relativa del box 3D rispetto alla telecamera (in m/s)
        vel_g = nusc.box_velocity(box.token)
        if not np.any(np.isnan(vel_g)):
            vel_c = rot_cam.T @ (rot_ego.T @ vel_g)
            v_gt = float((x_gt * vel_c[0] + z_gt * vel_c[2]) / (np.hypot(x_gt, z_gt) + 1e-5))
        else:
            v_gt = 0.0

        # Salvataggio delle informazioni del Ground Truth 3D filtrato nella lista eval_gts
        eval_gts.append({
            "instance_token": ann["instance_token"],
            "cls": cls_name,
            "cls_id": CLASSES.index(cls_name),
            "bbox": [x1, y1, x2, y2],
            "X_gt": x_gt, "Z_gt": z_gt, "v_gt": v_gt,
            "W_gt": w_3d, "L_gt": l_3d, "H_gt": h_3d,
            "yaw_gt": yaw_cam,
            "matched": False
        })

    return img, radar_pts, eval_gts, cam_k


def main():
    # Configurazione del dispositivo di calcolo (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Estrazione Feature Multi-Modali su {device}...")

    # Caricamento del modello YOLOv8 pre-addestrato per il rilevamento degli oggetti
    yolo = YOLO("yolov8m.pt")

    # Caricamento MiDaS dalla cache locale di sistema per evitare errori HTTP 504
    midas_cache = os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master")
    if os.path.exists(midas_cache):
        midas = torch.hub.load(midas_cache, "DPT_Hybrid", source="local").to(device).eval()
        transform = torch.hub.load(midas_cache, "transforms", source="local").dpt_transform
    else:
        midas = torch.hub.load("intel-isl/MiDaS:master", "DPT_Hybrid").to(device).eval()
        transform = torch.hub.load("intel-isl/MiDaS:master", "transforms").dpt_transform

    # Inizializzazione del dataset NuScenes (versione mini) per l'elaborazione dei primi 10 scenari
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)

    # Iterazione sui primi 10 scenari del dataset NuScenes per l'estrazione delle feature multi-modali
    records = []
    for s_idx in range(10):
        scene = nusc.scene[s_idx]
        token = scene["first_sample_token"]
        f_idx = 0
        scale_cal_p2 = 1.0
        print(f" -> Elaborazione Scena {s_idx + 1}/10: {scene['name']}...")

        # Iterazione sui frame della scena corrente fino a quando non ci sono più campioni
        while token != "":
            sample = nusc.get("sample", token)
            f_idx += 1
            img, radar_pts, eval_gts, cam_k = load_frame_data(nusc, sample)
            h, w = img.shape[:2]
            fx, fy, cx, cy = cam_k[0, 0], cam_k[1, 1], cam_k[0, 2], cam_k[1, 2]

            # Indicatori di contesto ambientale dell'immagine (luminosità e contrasto)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_brightness = float(np.mean(gray)) / 255.0
            img_contrast = float(np.std(gray)) / 128.0

            # Estrazione della mappa di profondità MiDaS e normalizzazione tra 0.01 e 1.0
            with torch.no_grad():
                pred = midas(transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device))
                depth_map = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()

            # Normalizzazione della mappa di profondità MiDaS tra il 2° e il 98° percentile per ridurre l'influenza degli outlier
            d_min, d_max = np.percentile(depth_map, 2), np.percentile(depth_map, 98)
            depth_norm = np.clip((depth_map - d_min) / (d_max - d_min + 1e-5), 0.01, 1.0)

            # Rilevamento degli oggetti con YOLOv8 e ordinamento dei box per area decrescente
            yolo_boxes = sorted(
                yolo(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0].boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
                reverse=True
            )

            # Iterazione sui box rilevati da YOLO per l'estrazione delle feature multi-modali e l'abbinamento con il Ground Truth 3D
            for box in yolo_boxes:
                cid = int(box.cls[0])
                if cid not in YOLO_CLASSES:
                    continue
                m_cls = map_yolo_class(YOLO_CLASSES[cid])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calcolo delle dimensioni del box e del centroide, nonché del rapporto d'aspetto
                bw, bh = max(x2 - x1, 12), max(y2 - y1, 12)
                uc, vc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                aspect_ratio = bw / float(bh)

                # Stime baseline del Progetto 1 (MiDaS) e del Progetto 2 (Pinhole + Gating Fisso)
                disp_med, disp_std, disp_p75 = extract_roi_stats(depth_norm, [x1, y1, x2, y2])
                z_midas_proj1 = 6.2 / (disp_med + 0.05)
                z_pinhole = (fy * HEIGHT_PRIOR[m_cls]) / bh
                w_m = 0.65 if z_pinhole < 18.0 else 0.30
                raw_comb = w_m * z_midas_proj1 + (1.0 - w_m) * z_pinhole
                z_prior_proj2 = scale_cal_p2 * raw_comb

                # Ricerca punti Radar dentro il box (escludendo i bordi esterni)
                mx = 0.22 if m_cls == "Person" else 0.15
                rx1, rx2 = x1 + int(mx * bw), x2 - int(mx * bw)
                ry1, ry2 = y1 + int(0.22 * bh), y2 - int(0.06 * bh)
                in_box_pts = [p for p in radar_pts if not p["matched"] and (rx1 <= p["u"] <= rx2) and (ry1 <= p["v"] <= ry2)]

                # Regola fissa del Progetto 2 (±22% gate + offset fisso)
                offset_p2 = BUMPER_OFFSET[m_cls]
                gate_tol = max(2.6, 0.22 * z_prior_proj2)
                valid_p2 = [p for p in in_box_pts if abs((p["Z"] + offset_p2) - z_prior_proj2) <= gate_tol]
                if valid_p2:
                    valid_p2.sort(key=lambda p: p["Z"] + 0.015 * abs(p["u"] - uc))
                    z_f = valid_p2[0]["Z"]
                    cl_p2 = [p for p in valid_p2 if abs(p["Z"] - z_f) <= 1.2]
                    for rp in cl_p2:
                        rp["matched"] = True
                    z_rad_proj2 = float(np.median([p["Z"] for p in cl_p2])) + offset_p2
                    has_rad_proj2 = 1
                    if 5.0 < z_rad_proj2 < 35.0 and raw_comb > 2.0:
                        scale_cal_p2 = 0.92 * scale_cal_p2 + 0.08 * np.clip(z_rad_proj2 / raw_comb, 0.80, 1.25)
                else:
                    z_rad_proj2 = z_prior_proj2
                    has_rad_proj2 = 0

                # Estrazione feature fisiche Radar + RCS per i modelli Learning-Based del Progetto 3
                if len(in_box_pts) > 0:
                    in_box_pts.sort(key=lambda p: abs((p["Z"] + offset_p2) - z_prior_proj2))
                    best_pt = in_box_pts[0]
                    near_pts = [p for p in in_box_pts if abs(p["Z"] - best_pt["Z"]) <= 1.5]
                    rad_z_raw = float(np.median([p["Z"] for p in near_pts]))
                    rad_x_raw = float(np.median([p["X"] for p in near_pts]))
                    rad_v_rel = float(np.median([p["v_rel"] for p in near_pts]))
                    rad_v_comp = float(np.median([p["v_comp"] for p in near_pts]))
                    rad_rcs = float(np.max([p["rcs"] for p in near_pts]))
                    rad_spread = float(np.std([p["Z"] for p in in_box_pts]))
                    rad_count = len(in_box_pts)
                    has_raw_radar = 1
                else:
                    rad_z_raw, rad_x_raw, rad_v_rel, rad_v_comp, rad_rcs, rad_spread, rad_count, has_raw_radar = (
                        0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0, 0
                    )

                # Abbinamento con il Ground Truth 3D certificato
                best_iou, best_gt = 0.0, None
                for gt in eval_gts:
                    if not gt["matched"] and gt["cls"] == m_cls:
                        iou = compute_iou([x1, y1, x2, y2], gt["bbox"])
                        if iou > best_iou:
                            best_iou, best_gt = iou, gt

                # Se non c'è un Ground Truth 3D valido o l'IoU è inferiore a 0.45, si salta il campione
                if best_gt is None or best_iou < 0.45:
                    continue
                best_gt["matched"] = True

                # Salvataggio delle informazioni del campione multi-modale estratto nella lista records
                records.append({
                    "scene_idx": s_idx,
                    "frame_idx": f_idx,
                    "instance_token": best_gt["instance_token"],
                    "cls_id": best_gt["cls_id"],
                    "cls_name": m_cls,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "cam_intrinsics": [fx, fy, cx, cy, w, h],
                    "box_norm": [uc / w, vc / h, bw / w, bh / h, aspect_ratio, (y2 - cy) / fy],
                    "disp_stats": [disp_med, disp_std, disp_p75],
                    "z_midas_proj1": z_midas_proj1,
                    "z_pinhole": z_pinhole,
                    "z_prior_proj2": z_prior_proj2,
                    "z_rad_proj2": z_rad_proj2,
                    "has_rad_proj2": has_rad_proj2,
                    "env_stats": [img_brightness, img_contrast],
                    "radar_stats": [has_raw_radar, rad_z_raw, rad_x_raw, rad_v_rel, rad_v_comp, rad_rcs, rad_spread, rad_count],
                    "gt_3d": [best_gt["X_gt"], best_gt["Z_gt"], best_gt["v_gt"], best_gt["W_gt"], best_gt["L_gt"], best_gt["H_gt"], best_gt["yaw_gt"]]
                })

            token = sample["next"]

    # Salvataggio dei campioni multi-modali estratti in un file compresso .npz
    out_file = os.path.join(OUTPUT_DIR, "features_nuscenes.npz")
    np.savez_compressed(out_file, records=records)

    # Stampa del riepilogo dei campioni estratti, con conteggio di quelli con e senza punto Radar valido
    tot = len(records)
    with_rad = sum(1 for r in records if r["has_rad_proj2"] == 1)
    no_rad = tot - with_rad
    print("\n" + "="*85)
    print(f"[SUCCESS] Salvati {tot} campioni multi-modali in {out_file}")
    print("-" * 85)
    print(f" -> Oggetti con punto Radar valido (Modalità Completa) : {with_rad} ({with_rad/max(tot,1)*100:.1f}%)")
    print(f" -> Oggetti senza punto Radar (Missing Modality [CAM]) : {no_rad} ({no_rad/max(tot,1)*100:.1f}%)")
    for c_idx, c_name in enumerate(CLASSES):
        c_sub = [r for r in records if r["cls_id"] == c_idx]
        c_rad = sum(1 for r in c_sub if r["has_rad_proj2"] == 1)
        print(f"    * {c_name:<10}: {len(c_sub):<4} campioni (con Radar: {c_rad:<4} | senza Radar: {len(c_sub)-c_rad:<4})")
    print("="*85)


if __name__ == "__main__":
    main()