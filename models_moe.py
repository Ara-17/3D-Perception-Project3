import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss Function -> VICReg (Variance-Invariance-Covariance Regularization) per il Co-Learning tra Esperto Visivo e Esperto Radar
def vicreg_loss(z_vis, z_rad, valid_mask):
    # Calcolo dei termini di Varianza (evita il collasso a costante) e Covarianza (decorrela le feature)
    def var_cov_terms(z):
        # Calcolo della varianza e della covarianza delle feature latenti
        z_cent = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z_cent.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))
        cov = (z_cent.T @ z_cent) / max(z.shape[0] - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / z.shape[1]
        return var_loss, cov_loss

    # Calcolo dei termini di Varianza e Covarianza per le rappresentazioni visive e radar
    v_vis, c_vis = var_cov_terms(z_vis)
    v_rad, c_rad = var_cov_terms(z_rad)

    # Calcolo del termine di Invarianza (MSE tra le rappresentazioni latenti visive e radar)
    if valid_mask.sum() > 1:
        inv_loss = F.mse_loss(z_vis[valid_mask], z_rad[valid_mask].detach()) + \
                   0.5 * F.mse_loss(z_rad[valid_mask], z_vis[valid_mask].detach())
    else:
        inv_loss = torch.tensor(0.0, device=z_vis.device)

    return 1.2 * inv_loss + 0.5 * (v_vis + v_rad) + 0.05 * (c_vis + c_rad)


# Versione 1 (v1) -> Rete Monolitica Early Fusion (Concatenazione Grezza [Camera || Radar])
class EarlyFusionMLP(nn.Module):
    # Rete MLP che riceve in ingresso il vettore concatenato grezzo [Camera || Radar] e produce la stima della profondità 3D
    def __init__(self, in_dim=28):
        super().__init__()
        # Rete monolitica che riceve in ingresso il vettore concatenato grezzo [Camera || Radar]
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    # Forward pass che concatena le feature visive e radar e produce la stima della profondità 3D
    def forward(self, x_vis, x_rad):
        x_joint = torch.cat([x_vis, x_rad], dim=-1)
        return F.softplus(self.net(x_joint)).squeeze(-1) * 12.0 + 1.5


# Versione 2 (v2 - Ramo A) -> Esperto Visivo 3D (Profondità, Dimensioni WLH e Angolo Yaw)
class VisualExpertNet(nn.Module):
    # Rete MLP che riceve in ingresso le feature visive e produce la stima della profondità 3D, delle dimensioni WLH e dell'angolo Yaw
    def __init__(self, in_dim=18, emb_dim=32):
        super().__init__()
        # Encoder visivo che proietta le feature 2D e di disparità nello spazio latente coordinato
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU()
        )
        # Testa di correzione della profondità visiva (apprende il fattore correttivo sul prior ottico)
        self.depth_head = nn.Sequential(
            nn.Linear(emb_dim + in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 2)
        )
        # Testa di regressione del Bounding Box 3D Orientato: dimensioni (W, L, H) e rotazione (sin_yaw, cos_yaw)
        self.box3d_head = nn.Sequential(
            nn.Linear(emb_dim + in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 5)
        )
    # Forward pass che elabora le feature visive e produce la stima della profondità 3D, delle dimensioni WLH e dell'angolo Yaw
    def forward(self, x_vis, z_prior):
        # Estrazione della rappresentazione latente coordinata tramite l'encoder visivo
        emb_vis = self.encoder(x_vis)
        feat_cat = torch.cat([emb_vis, x_vis], dim=-1)

        # Calcolo della profondità visiva corretta tramite Co-Learning
        d_out = self.depth_head(feat_cat)
        scale_corr = 1.0 + 0.45 * torch.tanh(d_out[:, 0])
        shift_corr = 4.5 * torch.tanh(d_out[:, 1])
        z_vis_pred = torch.clamp(z_prior * scale_corr + shift_corr, 1.5, 55.0)

        # Predizione delle dimensioni 3D (Width, Length, Height) e dell'angolo Yaw normalizzato
        b_out = self.box3d_head(feat_cat)
        wlh_pred = F.softplus(b_out[:, :3]) + torch.tensor([0.4, 0.4, 0.8], device=x_vis.device)
        yaw_sincos = F.normalize(b_out[:, 3:5], p=2, dim=-1)

        return z_vis_pred, wlh_pred, yaw_sincos, emb_vis


# Versione 2 (v2 - Ramo B) -> Esperto Radar e Geometrico 3D (Profondità, Offset Dinamico e Validità Fisica)
class RadarExpertNet(nn.Module):
    def __init__(self, in_dim=12, emb_dim=32):
        super().__init__()
        # Encoder fisico che elabora distanza, velocità Doppler e potenza di riflessione metallica RCS
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.SiLU()
        )
        # Testa per la stima dell'offset dinamico Paraurti -> Centro 3D (sostituisce il +1.30m fisso)
        self.offset_head = nn.Sequential(
            nn.Linear(emb_dim + in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        # Testa di classificazione della validità fisica del punto Radar (filtra il multipath)
        self.validity_head = nn.Sequential(
            nn.Linear(emb_dim + in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_rad, z_rad_raw, cls_offset_prior):
        emb_rad = self.encoder(x_rad)
        feat_cat = torch.cat([emb_rad, x_rad], dim=-1)

        # Correzione dinamica dell'offset geometrico in funzione dell'orientamento e della RCS
        delta_off = 1.3 * torch.tanh(self.offset_head(feat_cat).squeeze(-1))
        dyn_offset = torch.clamp(cls_offset_prior + delta_off, 0.05, 3.80)
        z_rad_corr = z_rad_raw + dyn_offset

        # Probabilità che il riflesso Radar appartenga realmente al target e non allo sfondo (multipath)
        p_valid = torch.sigmoid(self.validity_head(feat_cat).squeeze(-1))
        return z_rad_corr, dyn_offset, p_valid, emb_rad


# Versione 3 (v3): Hybrid Stacking Meta-Learner ("Il Giudice") sopra gli Esperti Congelati
class HybridStackingJudge(nn.Module):
    # Rete MLP che riceve in ingresso le stime degli esperti visivo e radar, le feature latenti congiunte e il contesto, e produce la stima finale della profondità 3D e l'incertezza sigma_z
    def __init__(self, emb_dim=32, ctx_dim=8):
        super().__init__()
        # Rete di Gating (Meta-Learner) che assegna i pesi di fiducia dinamici agli esperti del primo livello
        self.gating_net = nn.Sequential(
            nn.Linear(ctx_dim + 4, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 2)
        )
        # Ramo Hybrid Fusion: combina le rappresentazioni latenti congiunte e stima l'incertezza sigma_z
        self.hybrid_head = nn.Sequential(
            nn.Linear(emb_dim * 2 + ctx_dim + 2, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Linear(48, 2)
        )

    # Forward pass che riceve le stime degli esperti visivo e radar, le feature latenti congiunte e il contesto, e produce la stima finale della profondità 3D e l'incertezza sigma_z
    def forward(self, z_vis, z_rad_corr, p_valid, has_raw_rad, emb_vis, emb_rad, ctx_feat):
        # Discrepanza relativa tra l'Esperto Visivo e l'Esperto Radar
        rel_diff = torch.abs(z_vis - z_rad_corr) / (z_vis + 1.0)
        gate_in = torch.cat([
            ctx_feat,
            z_vis.unsqueeze(-1) / 50.0,
            z_rad_corr.unsqueeze(-1) / 50.0,
            p_valid.unsqueeze(-1),
            rel_diff.unsqueeze(-1)
        ], dim=-1)

        # Calcolo dei pesi di fiducia per l'Esperto Visivo e l'Esperto Radar, limitando il peso del Radar a 0.96 per evitare collasso
        valid_gate = has_raw_rad * (p_valid > 0.45).float() * p_valid
        gate_logits = self.gating_net(gate_in)
        soft_w = F.softmax(gate_logits, dim=-1)

        # Calcolo dei pesi di fiducia per l'Esperto Visivo e l'Esperto Radar, limitando il peso del Radar a 0.96 per evitare collasso
        w_rad = torch.clamp(soft_w[:, 1] * valid_gate, 0.0, 0.96)
        w_vis = 1.0 - w_rad

        # Fusione decisionale pesata (Late Stacking Prediction)
        z_stack = w_vis * z_vis + w_rad * z_rad_corr

        # Raffinamento Hybrid Fusion dalle feature latenti congiunte + stima dell'incertezza per Kalman
        hyb_in = torch.cat([
            emb_vis,
            emb_rad * valid_gate.unsqueeze(-1),
            ctx_feat,
            w_vis.unsqueeze(-1),
            w_rad.unsqueeze(-1)
        ], dim=-1)
        hyb_out = self.hybrid_head(hyb_in)

        # Calcolo della stima finale della profondità 3D tramite Hybrid Fusion e stima dell'incertezza sigma_z
        residual_hyb = 0.85 * torch.tanh(hyb_out[:, 0])
        z_hybrid = torch.clamp(z_stack + residual_hyb, 1.5, 55.0)

        # Deviazione standard dell'incertezza appresa (usata per calibrare R_t nel Filtro di Kalman)
        sigma_z = F.softplus(hyb_out[:, 1]) + 0.05
        return z_hybrid, w_vis, w_rad, sigma_z