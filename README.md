# LinkSight // Autonomous FSOC ATP Coarse Tracking Terminal

<div align="center">

### 🇮🇳 SMART INDIA HACKATHON 2026
**Problem Statement ID:** `PS-26169`  
**Problem Statement Title:** Development of an AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile Free Space Optical Communication (FSOC) Terminals  
**Host Agency:** Indian Space Research Organisation (ISRO) / Department of Space (DOS)  
**Theme:** Smart Automation | **Team:** Guardians of the Galaxy  

---

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GUI](https://img.shields.io/badge/GUI-PySide6%20Qt6-41CD52.svg?logo=qt&logoColor=white)](https://pyside.org/)
[![AI Engine](https://img.shields.io/badge/AI%20Inference-ONNX%20Runtime-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Test Suite](https://img.shields.io/badge/Unit%20Tests-15%2F15%20Passing-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-gray.svg)]()

</div>

---

## 🛰️ 1. Project Overview

**LinkSight** is a high-performance, edge-deployable software prototype for **Stage-1 Coarse Alignment** in Mobile Free-Space Optical Communication (FSOC) terminals.

In free-space laser communications (inter-satellite, airborne UAV, and satellite-to-ground downlinks), optical beam divergence is extremely narrow (milliradians). Before high-speed Fast Steering Mirrors (FSM) can execute microradian fine pointing, the **coarse alignment subsystem** must rapidly acquire the incoming beacon spot across a wide spatial region and keep it centered inside the optical detector's field-of-view under severe dynamic disturbances.

LinkSight achieves this with a **zero-cloud, hybrid classical-vision and sleep-wake neural pipeline** running at **35–60 FPS** on standard edge computing hardware.

---

## 🏗️ 2. Core Architecture & Sleep-Wake Pipeline

LinkSight operates on a **dual-tier, sleep-wake architecture** designed for deterministic safety and high computational efficiency:

```
                  ┌────────────────────────────────────────────────────────┐
                  │          FOCAL PLANE ARRAY CAMERA (640x480)            │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Raw Frames (30-60 Hz)
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │          STAGE 1: OPTICAL DETECTION ENGINE             │
                  │  - 5x5 Median Filter (Impulse noise suppression)       │
                  │  - Morphological Top-Hat Filter (DC radiance strip)    │
                  │  - Dual CFAR Adaptive Thresholding (μ + 3.2σ)          │
                  │  - Intensity-Weighted Sub-Pixel Centroiding            │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Centroid (u, v) + Confidence
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │          STAGE 2: HYBRID KALMAN & AI TRACKER           │
                  │  - Constant-Velocity Discrete Kalman Filter            │
                  │  - Confidence-Weighted Covariance Scaling (R_k)        │
                  │  - Chi-Squared (χ²) Innovation Gating                  │
                  │  - MicroGRU Coaster (Wakes on sensor dropouts/gaps)    │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Filtered State (Pos, Vel)
                                              ▼
        ┌─────────────────────────────────────┴─────────────────────────────────────┐
        ▼                                                                           ▼
 ┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
 │        CLOSED-LOOP TRACKING (LOCKED)         │    │      RE-ACQUISITION ENGINE (LOST/FADE)       │
 │  - PID Rate Controller + Velocity Feedforward│    │  - Cut-Hexagonal Expanding Spiral Search     │
 │  - TinyDDPG Neural Dampener (Wakes on high-G)│    │  - Bounded by Kalman Covariance Ellipse      │
 │  - Steady-State Error: 0.0 - 4.2 px          │    │  - Sub-Second Target Re-Lock (< 0.75s)       │
 └──────────────────────┬───────────────────────┘    └──────────────────────┬───────────────────────┘
                        │                                                   │
                        └─────────────────────┬─────────────────────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              VIRTUAL PTZ GIMBAL ACTUATOR               │
                  │  - Slew Rate Limiter (5.0 °/s default, 1-25 °/s max)   │
                  │  - RS-422 / UDP / MIPI Physical Interface Abstractions │
                  └────────────────────────────────────────────────────────┘
```

* **Nominal Flight (Sleep Mode):** 100% deterministic classical CV, morphological Top-Hat background isolation, and Bayesian Kalman filtering execute at $<15\text{ ms/frame}$ CPU latency.
* **Degraded Flight (Wake Mode):** Lightweight ONNX neural co-processors (**TinyBeaconNet**, **MicroGRU Coaster**, **TinyDDPG Dampener**) wake up strictly during cloud occlusions, extreme solar glints, or high-G aerodynamic shear.

---

## 📂 3. Repository Layout 

```
SIH2026_ISRO_LinkSight_FSOC_ATP/
├── 📄 main.py                          # Application entry point (PySide6 GUI)
├── 📄 requirements.txt                  # Python dependencies (PySide6, OpenCV, NumPy, ONNXRuntime)
├── 📄 Launch_LinkSight.bat              # 1-Click launcher script
├── 📄 Compile_Executable.bat            # 1-Click PyInstaller build script
│
├── 📁 fsoc/                            # Core Algorithmic Package
│   ├── detector.py                     # Optical detection: Median, Top-Hat, CFAR, Centroiding, TinyBeaconNet
│   ├── tracker.py                      # State estimation: Kalman Filter + MicroGRU Neural Coaster
│   ├── controller.py                   # Actuation: 2-Axis PID Gimbal Controller + TinyDDPG Dampener
│   ├── reacquisition.py                # Re-acquisition: Cut-Hexagonal Spiral Search Engine
│   ├── frame_source.py                 # Virtual camera simulator (2000x2000) & Video frame grabber
│   ├── disturbance.py                  # Channel noise: S&P, Gaussian, Poisson, Dabiri PSD jitter, Weather
│   ├── engine.py                       # Master tracking state machine & pipeline orchestrator
│   ├── logger.py                       # Central telemetry recorder, audit trail, CSV/JSON exporter
│   └── 📁 gui/                         # PySide6 Qt6 GUI Components
│       ├── main_window.py              # 3-column Mission Control HUD window
│       ├── worker.py                   # High-rate QThread worker isolating compute from GUI
│       ├── theme.py                    # Dark telemetry palette & styling tokens
│       └── 📁 widgets/                 # Viewport, Control Panel, Minimap, Error Chart, Stat Cards
│
├── 📁 models/                          # Pre-trained Lightweight Edge AI Models (ONNX)
│   ├── microgru_coast.onnx             # MicroGRU trajectory extrapolation (12.4k params)
│   ├── tinybeaconnet.onnx              # TinyBeaconNet patch classifier (3.1k params)
│   └── tinyddpg_dampener.onnx          # DDPG non-linear rate dampener (1.8k params)
│
├── 📁 tests/                           # Comprehensive Automated Unit Test Suite (15/15 passing)
│   ├── test_pipeline.py                # End-to-end headless pipeline convergence & metric verification
│   ├── test_detector.py                # Optical detection, SNR, and false alarm rate tests
│   ├── test_tracker.py                 # Kalman filtering, covariance growth, and occlusion coasting
│   ├── test_controller.py              # PID rate limiter, anti-windup, and deadband validation
│   └── test_reacquisition.py           # Cut-hexagonal search lattice geometry & dwell timing
│
├── 📁 logs/                            # Technical Report & Empirical Benchmark Data
│   ├── LinkSight_Technical_Report.tex  # Complete LaTeX Technical Report Source
│   ├── LinkSight_FSOC_ATP_Technical_Report.pdf # Compiled Technical Report PDF
│   ├── SCENARIO1_BASELINE.json         # Scenario 1 empirical benchmark logs
│   ├── SCENARIO2_FOG.json              # Scenario 2 heavy fog stress-test logs
│   └── SCENARIO3_RAIN.json             # Scenario 3 dynamic rain & wind shear logs
│
├── 📁 tools/                           # Model Training & Benchmark Utility Scripts
│   ├── train_microgru.py               # PyTorch training script for MicroGRU Coaster
│   ├── train_ddpg.py                   # PyTorch/Gym training script for TinyDDPG Dampener
│   └── generate_report_data.py         # Automated multi-scenario test harness
│
├── 📄 PROTOTYPE_DEMO_SCRIPT.txt         # 5-Minute Technical Demonstration Video Script
├── 📄 USER_MANUAL.md                   # Operational GUI user manual & control guide
└── 📄 TECHNICAL_REPORT.md              # Compact markdown technical specification
```

---

## 🚀 4. Quick Start Guide

### Prerequisites
* **Python 3.10 or 3.11** (recommended)
* Windows, Linux, or macOS

### Option A: 1-Click Launch (Windows)
Double-click `Launch_LinkSight.bat` in the root directory.

### Option B: Run via Terminal
```bash
# 1. Clone repository
git clone https://github.com/bhutanisatyam-droid/LINK-SIGHT.git
cd LINK-SIGHT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch prototype GUI
python main.py
```

### Option C: Run Automated Test Suite
```bash
python -m unittest discover tests
```

---

## 📊 5. ISRO Benchmark Performance Summary

All five core technical criteria defined in ISRO Problem Statement **PS-26169** are satisfied with safety margins:

| Metric | ISRO Specification | LinkSight Measured Performance | Status |
| :--- | :---: | :---: | :---: |
| **Initial Acquisition Time** | $\le 2.0\,\text{s}$ | **$0.20 - 1.20\,\text{s}$** | ✅ Passed ($40\%$ margin) |
| **Steady-State Pointing Error** | $\le 10.0\,\text{px}$ | **$0.0 - 4.2\,\text{px}$** | ✅ Passed ($58\%$ margin) |
| **Target Lock Retention** | $> 95.0\%$ ($<5\%$ loss) | **$96.8 - 99.2\%$** | ✅ Passed |
| **Re-Acquisition Time** | $\le 1.0\,\text{s}$ | **$0.15 - 0.65\,\text{s}$** | ✅ Passed ($35\%$ margin) |
| **Loop Processing Rate** | $\ge 20.0\,\text{Hz}$ | **$30.0 - 60.0\,\text{Hz}$** | ✅ Passed ($100\%$ above spec) |

---

## 📄 6. Documentation & Deliverables

* **Technical Report (LaTeX & PDF):** [LinkSight_Technical_Report.tex](logs/LinkSight_Technical_Report.tex)
* **Operational User Manual:** [USER_MANUAL.md](USER_MANUAL.md)
* **5-Minute Video Walkthrough Script:** [PROTOTYPE_DEMO_SCRIPT.txt](PROTOTYPE_DEMO_SCRIPT.txt)

---

<div align="center">

**Smart India Hackathon 2026** | **ISRO Department of Space**  
*Engineered by Team Guardians of the Galaxy*

</div>
