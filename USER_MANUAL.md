# LinkSight // User Manual & Operational Guide

### 🛰️ Smart India Hackathon 2026 | Problem Statement: 26169 (ISRO / DOS)

---

## 1. System Requirements & Installation

### Option 1: Standalone Executable (.EXE) — No Python Needed
1. Navigate to: `dist/LinkSight_FSOC_ATP_Terminal/`
2. Double-click `LinkSight_FSOC_ATP_Terminal.exe`.
3. The Mission Control GUI opens immediately.

### Option 2: Running from Source
* **Operating System:** Windows 10/11 (64-bit), Linux, or macOS.
* **Python Version:** Python 3.10 or 3.11.
* **Installation:**
```bash
cd SIH2026_ISRO_LinkSight_FSOC_ATP
pip install -r requirements.txt
python main.py
```
*(Or simply double-click `Launch_LinkSight.bat` on Windows)*.

---

## 2. Graphical User Interface (GUI) Layout

The interface is structured into four functional zones:

1. **Top Header & Status Bar:**
   * Displays the active subsystem state, Mission Execution timer, and real-time Loop Rate (FPS).
2. **Left Control Panel:**
   * **Mission Execution:** `RUN`, `PAUSE`, `RESET`, and `⚡ SIMULATE BEAM BREAK (1s)`.
   * **Trajectory & Optical Specs:** Motion Model selector (`Circular`, `Straight`, `Figure-8`, `Random Walk`), Spot Size ($5-20\text{ px}$), PTZ Max Speed ($1-25^\circ/\text{s}$).
   * **Atmospheric Channel:** Dropdown for `Clear`, `Haze`, `Fog`, `Rain`, and `Low-Light`.
   * **Disturbance Engine:** Salt & Pepper noise, Gaussian thermal noise ($\sigma=0-50$), Poisson shot noise, Camera Jitter ($\pm 20\text{ px}$), Platform Motion ($\pm 20\text{ px}$).
3. **Center Main Viewport:**
   * Displays the $640 \times 480$ optical camera feed.
   * Real-time HUD overlay: Gimbal Az/El angles, FOV, tracking crosshairs, confidence score, coasting counters, and active AI status (`KALMAN (MATH)` vs `MicroGRU WAKE-UP`).
   * Hexagonal spiral re-acquisition waypoints overlay during beam breaks.
4. **Right & Bottom Telemetry Dashboard:**
   * **Top-Down 2D Scene Radar:** Shows target beacon position $(X, Y)$ relative to the camera FOV rectangle across the $2000 \times 2000\text{ px}$ arena.
   * **Telemetry Audit Cards:** Live Loop Rate, Tracking Error, Lock Retention %, Acquisition Time, Re-Acquisition Stopwatch, Loss Event Counter.
   * **Real-time Error History Strip Chart:** Rolling 15-second tracking error graph with the ISRO $10.0\text{ px}$ specification threshold line.
   * **Chronological Event Log:** Real-time timestamped event stream.

---

## 3. Operational Workflows & Demonstrations

### Workflow A: Nominal Closed-Loop Tracking
1. Click **`RUN`**.
2. Select **`Circular Orbit`** or **`Linear Flight Path`**.
3. Observe steady-state tracking error converging to **$\le 4.5\text{ px}$** (well within ISRO's $10.0\text{ px}$ limit).

### Workflow B: Testing Beam Occlusion & Sub-Second Re-Acquisition
1. While tracking is locked, click **`⚡ SIMULATE BEAM BREAK (1s)`**.
2. Observe the FSM transition:
   * **Frame 1–20:** Enters `DEGRADED [COASTING]` where Kalman + MicroGRU coast the gimbal blindly along the velocity vector.
   * **Frame 20+:** Enters `LOST [SPIRAL SEARCH]`, initiating the **Cut-Hexagonal Spiral Search**.
   * **Re-Lock:** The gimbal intercepts the target in **$< 1.0\text{s}$** (`RE-ACQUISITION: NOMINAL`).

### Workflow C: Severe Environmental Disturbance Stress Test
1. Set **Sky Radiance** to `100% (High Noon)`.
2. Set **Atmospheric Channel** to `Dense Fog` or `Koschmieder Haze`.
3. Enable **Salt & Pepper (10%)**, **Gaussian Thermal ($\sigma=15$)**, and **Camera Jitter ($\pm 20\text{ px}$)**.
4. Observe the Morphological Top-Hat and CFAR filter isolating the beacon through heavy noise.

---

## 4. Telemetry Export & Analysis

To export performance data for post-mission analysis or evaluation:
* Click **`EXPORT TELEMETRY (CSV / JSON)`** in the UI.
* Exports detailed per-frame logs containing: `frame_idx`, `tracking_error_px`, `confidence`, `pan_deg`, `tilt_deg`, `status`, and `timestamp`.
