# Learning-Based Camera-Radar Data Fusion and 3D Oriented Bounding Box Estimation

**Course:** 3D Perception & Learning-Based Data Fusion (A.Y. 2026/2027) - University of Parma  
**Student:** Arashpreet Singh  
**Dataset:** [nuScenes v1.0-mini](https://www.nuscenes.org/nuscenes) (`CAM_FRONT` + `RADAR_FRONT` + 3D LiDAR Ground Truth)
**Media, Demo Videos & Outputs (OneDrive):** [Open OneDrive Folder](https://univpr-my.sharepoint.com/:f:/g/personal/arashpreet_singh_studenti_unipr_it/IgAP8D1HTET2QIrnTexATodmAd8hQxoXYwRSKbClBzFQRrE?e=RjOth6)

---

## 1. Project Overview

This repository contains the implementation of **Project 3**, developed as a direct continuation of Project 1 (Monocular Camera 3D Perception) and Project 2 (Rule-Based Camera + Doppler Radar Fusion).

In Project 2, combining YOLOv8m, MiDaS (`DPT_Hybrid`), and 77 GHz Doppler Radar measurements through the Hungarian Algorithm and a 4D Linear Kalman Filter reduced the global 3D distance RMSE from $6.92\text{ m}$ (Project 1) to $3.94\text{ m}$ ($2.19\text{ m}$ on targets with an active Radar lock). However, that traditional rule-based pipeline had three practical limitations:

1. **Missing Radar reflections on $53.2\%$ of road objects:** Non-metallic targets such as pedestrians ($85.6\%$ missing Radar) and bicycles ($78.9\%$ missing Radar) have a very low Radar Cross-Section (RCS). When falling back to the uncalibrated Pinhole formula (`[CAM]` mode), Project 2 had an average distance error of $5.00\text{ m}$.
2. **Fixed bumper-to-center offset ($+1.30\text{ m}$) without 3D vehicle orientation ($\theta_{\text{yaw}}$):** Adding a constant $+1.30\text{ m}$ offset to Radar points fails when vehicles turn sideways at intersections. Furthermore, objects could only be drawn as circular dots in the Bird's-Eye View (BEV) map because their 3D dimensions and rotation angle were unknown.
3. **Static Kalman measurement noise matrices:** The Kalman Filter used fixed measurement variances ($R_{\text{radar}} = 0.05$, $R_{\text{cam}} = 2.20$) regardless of lighting conditions (day, night, heavy rain), target distance, or Radar signal quality.

To solve these issues without overfitting on `nuScenes v1.0-mini` ($1,373$ matched objects), Project 3 implements a modular **Learning-Based Multi-Modal Data Fusion** architecture in PyTorch based on **Coordinated Representations, VICReg Regularization, Parallel Co-Learning, and Hybrid Model Stacking**.

---

## 2. Architecture & Fusion Paradigms Evaluated

We implement and compare three progressive learning architectures using **5-Fold Stratified Cross-Validation**:

* **Version 1 (`v1` — Naive Early Fusion MLP):** Directly concatenates the $18$ visual features and $12$ Radar features ($[x_{\text{vis}} \parallel x_{\text{rad}}]$) into a single monolithic MLP. Because $53.2\%$ of samples have zero-padded Radar inputs, it suffers from modality imbalance.
* **Version 2 (`v2` — Coordinated Space & Parallel Co-Learning with VICReg):** Decouples perception into two specialized micro-experts mapped into a 32-dimensional Coordinated Latent Space:
  * **Visual & 3D Geometry Expert (`VisualExpertNet`):** Takes 2D box geometry, ground-contact coordinates, and MiDaS ROI disparity statistics to predict corrected metric depth, 3D dimensions ($W, L, H$), and continuous 3D orientation $[\sin(\theta_{\text{yaw}}), \cos(\theta_{\text{yaw}})]$.
  * **Radar Physics & RCS Expert (`RadarExpertNet`):** Takes raw Radar range, Doppler radial velocities ($v_{\text{rel}}, v_{\text{comp}}$), and **Radar Cross-Section (RCS reflectivity in $\text{dBsm}$)** to predict a dynamic 3D surface-to-center offset $\Delta Z_{\text{dyn}}$ and a physical validity score $p_{\text{valid}} \in [0, 1]$ that rejects background multipath reflections.
  * **Parallel Co-Learning:** During training, on the $643$ parallel samples where Radar is present, the Radar Expert acts as a teacher for the Visual Expert via VICReg alignment and co-learning distillation. At test time, even when Radar is completely missing ($730$ objects), the Co-Learned Visual Expert predicts accurate 3D depth from the camera alone.
* **Version 3 (`v3` — Hybrid Stacking Judge + Learned 4D Kalman Filter):** Freezes the first-layer experts and trains a meta-learner (`HybridStackingJudge`) that takes the experts' predictions, joint embeddings, and environmental context (image brightness/contrast for night and rain, range, RCS strength) to output dynamic trust weights ($w_{\text{vis}}, w_{\text{rad}}$), a hybrid residual correction, and learned measurement uncertainty $\sigma_Z^2$ for the 4D Kalman Filter (`NeuralKalmanTrack3D`).

---

## 3. Repository Structure

```text
3d_perception_project3/
├── data/
│   └── v1.0-mini/               # Local nuScenes v1.0-mini dataset (ignored by Git)
├── output/
│   ├── features_nuscenes.npz    # Extracted multi-modal dataset (1,373 objects)
│   ├── predictions_project3.npz # Out-of-fold predictions across v1, v2, v3
│   ├── weights/
│   │   └── moe_project3.pth     # Trained PyTorch weights (Visual Expert, Radar Expert, Judge)
│   └── plots_en/                # High-resolution benchmark plots (PNG) and metrics CSV
├── extract_dataset.py           # STEP 1: Multi-modal feature extraction (Camera + Doppler + RCS + 3D GT)
├── models_moe.py                # STEP 2: PyTorch architectures (Early MLP, Experts, VICReg, Hybrid Judge)
├── train_and_benchmark.py       # STEP 3: 5-Fold training, ablation study, missing-modality & XAI plots
├── demo_video_3d_bev.py         # STEP 4: Real-time tracking & 3D Oriented Bounding Box BEV video renderer
├── yolov8m.pt                   # Pre-trained YOLOv8m weights (ignored by Git)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

> **Note on Heavy Files (Videos, Dataset & Weights):** Rendered 3D BEV demo videos, high-resolution plots, and project materials are available on [OneDrive](https://univpr-my.sharepoint.com/:f:/g/personal/arashpreet_singh_studenti_unipr_it/IgAP8D1HTET2QIrnTexATodmAd8hQxoXYwRSKbClBzFQRrE?e=RjOth6).

---

## 4. Installation & Setup

1. **Create and activate a dedicated Python virtual environment (`venv`):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. **Install the required packages (with CUDA 12.4 support for PyTorch):**
   ```powershell
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
   ```
3. **Dataset placement:**  
   Ensure the `nuScenes v1.0-mini` dataset is located inside `data/v1.0-mini/` and `yolov8m.pt` is in the root folder `3d_perception_project3/`.

---

## 5. How to Run the Pipeline

Run the scripts in the following order:

### Step 1: Extract Multi-Modal Camera & Radar Features
Runs YOLOv8m, MiDaS (`DPT_Hybrid`), 3D Radar projection (including Doppler and RCS reflectivity), and 3D LiDAR Ground Truth matching across all 10 scenes (404 keyframes), saving the compact dataset to `output/features_nuscenes.npz`:
```powershell
python extract_dataset.py
```

### Step 2 & 3: Train Neural Architectures & Generate Benchmark Plots
Trains Version 1 (`EarlyFusionMLP`), Version 2 (`VisualExpertNet` + `RadarExpertNet` with VICReg Co-Learning), and Version 3 (`HybridStackingJudge` + Learned Kalman Filter) via 5-Fold Cross-Validation, prints the quantitative comparison tables, saves model weights to `output/weights/moe_project3.pth`, and generates all 7 evaluation plots in `output/plots_en/`:
```powershell
python train_and_benchmark.py
```

### Step 4: Render 3D Oriented Bounding Box BEV Video
Loads the trained neural weights and renders the side-by-side tracking video (`output/video_project3_scene_0.mp4`) showing live Judge trust weights on the RGB camera view and **3D Oriented Bounding Boxes** rotating in the Top-Down BEV map:
```powershell
python demo_video_3d_bev.py
```
*(You can change `SCENE_INDEX` from `0` to `9` inside `demo_video_3d_bev.py` to render any of the 10 nuScenes scenes).*

---

## 6. Quantitative Results Summary

### A. Dataset Missing-Modality Distribution (1,373 Matched Objects)

| Class | Total Matched | With Valid Radar (Complete Modality) | Without Radar (Missing Modality `[CAM]`) | Missing Radar (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Car** | 933 | 505 | 428 | $45.9\%$ |
| **Truck/Bus** | 137 | 93 | 44 | $32.1\%$ |
| **Person** | 284 | 41 | 243 | $85.6\%$ |
| **Bike/Moto** | 19 | 4 | 15 | $78.9\%$ |
| **TOTAL** | **1,373** | **643 ($46.8\%$)** | **730 ($53.2\%$)** | **$53.2\%$** |

### B. 3D Distance RMSE Comparison: Project 1 vs. Project 2 vs. Project 3

| Class | N | Proj 1 (Cam Only) | Proj 2 (All Tracks) | Proj 2 (Radar Lock) | Proj 3 `v1` (Early MLP) | Proj 3 `v2` (Co-Learning) | Proj 3 `v3` (All Tracks) | Proj 3 `v3` (Radar Lock) | Proj 3 `v3` (No-Radar) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Car** | 933 | $6.96\text{ m}$ | $3.80\text{ m}$ | $1.71\text{ m}$ | $2.66\text{ m}$ | $2.10\text{ m}$ | **$1.94\text{ m}$ ($-48.9\%$)** | **$1.24\text{ m}$ ($-82.2\%$)** | $2.52\text{ m}$ |
| **Truck/Bus** | 137 | $8.20\text{ m}$ | $4.18\text{ m}$ | $3.60\text{ m}$ | $4.13\text{ m}$ | $3.37\text{ m}$ | **$3.10\text{ m}$ ($-25.8\%$)** | **$3.21\text{ m}$ ($-60.9\%$)** | $2.87\text{ m}$ |
| **Person** | 284 | $6.19\text{ m}$ | $4.16\text{ m}$ | $2.83\text{ m}$ | $1.92\text{ m}$ | $2.04\text{ m}$ | **$1.91\text{ m}$ ($-54.1\%$)** | $2.20\text{ m}$ ($-64.5\%$) | **$1.86\text{ m}$** |
| **Bike/Moto** | 19 | $5.23\text{ m}$ | $5.49\text{ m}$ | $3.92\text{ m}$ | $1.31\text{ m}$ | $1.43\text{ m}$ | **$1.09\text{ m}$ ($-80.1\%$)** | **$1.23\text{ m}$ ($-76.5\%$)** | **$1.05\text{ m}$** |
| **TOTAL** | **1,373** | **$6.92\text{ m}$** | **$3.94\text{ m}$** | **$2.19\text{ m}$** | **$2.70\text{ m}$** | **$2.24\text{ m}$** | **$2.07\text{ m}$ ($-47.5\%$)** | **$1.74\text{ m}$ ($-74.9\%$)** | **$2.33\text{ m}$ ($-53.4\%$)** |

### C. 3D Oriented Bounding Box & Judge Performance

* **3D Dimension Estimation Error (MAE):** Width $W = 0.14\text{ m}$ | Length $L = 0.36\text{ m}$ | Height $H = 0.15\text{ m}$
* **3D Orientation Angle Error ($\theta_{\text{yaw}}$ MAE):** $2.4^\circ$ ($0.04\text{ rad}$)
* **Dynamic Judge Trust Allocation:** Assigns $80.0\%$ to $87.1\%$ trust to the Radar+RCS Expert at close/medium ranges ($0\text{--}35\text{ m}$) and increases Visual Expert trust to $28.6\%$ at long range ($35\text{--}50\text{ m}$).