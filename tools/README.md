# LinkSight AI Training Tools & Execution Instructions

All scripts required to generate synthetic datasets, train the PyTorch models, and export optimized ONNX edge binaries are fully implemented in this directory.

---

## Quick Execution Guide

### Model 1: TinyBeaconNet (Core 1 — Glint Rejection CNN)
1. **Generate Dataset:**
   ```bash
   python tools/generate_beacon_data.py
   ```
   *Generates 20,000 synthetic patches + 8D features $\rightarrow$ `datasets/dataset_beacon.npz` (~5 MB)*

2. **Train & Export ONNX:**
   ```bash
   python tools/train_beacon_colab.py
   ```
   *(Or upload `dataset_beacon.npz` and `train_beacon_colab.py` to Google Colab with free T4 GPU)*  
   *Output: `models/tinybeaconnet.onnx` (~140 KB)*

---

### Model 2: MicroGRU-Coast (Core 2 — 200ms Blind Trajectory Predictor)
1. **Generate Trajectory Data:**
   ```bash
   python tools/generate_trajectory_data.py
   ```
   *Generates ~100,000 flight steps at 60 Hz across diverse maneuvers $\rightarrow$ `datasets/dataset_trajectory.npz` (~4 MB)*

2. **Train & Export ONNX:**
   ```bash
   python tools/train_microgru.py
   ```
   *Runs locally on laptop CPU in ~15-20 mins (or on Colab in ~2 mins)*  
   *Output: `models/microgru_coast.onnx` (~38 KB)*

---

### Model 3: TinyDDPG-Dampener (Core 3 — Turbulence RL Damping Agent)
1. **Train & Export ONNX:**
   ```bash
   python tools/train_ddpg.py
   ```
   *Runs the Actor-Critic agent inside `tools/fsoc_gym_env.py` under dynamic Dabiri PSD aerodynamic turbulence*  
   *Output: `models/tinyddpg_dampener.onnx` (~28 KB)*

---

## Resulting Model Directory (`FSOC/models/`):
```
models/
├── tinybeaconnet.onnx       # ~140 KB (Glint Rejection Vision AI)
├── microgru_coast.onnx      #  ~38 KB (Occlusion Trajectory AI)
└── tinyddpg_dampener.onnx   #  ~28 KB (Vibration Dampener RL AI)
```
*(Combined total size: < 210 KB — ready for sub-15ms edge inference)*
