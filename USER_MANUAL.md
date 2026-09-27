# LinkSight // User Manual & Operational Guide

<div align="center">

### 🇮🇳 SMART INDIA HACKATHON 2026
**Problem Statement ID:** `PS-26169`  
**Problem Statement Title:** Development of an AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile Free Space Optical Communication (FSOC) Terminals  
**Host Agency:** Indian Space Research Organisation (ISRO) / Department of Space (DOS)  
**Theme:** Smart Automation | **Team:** Guardians of the Galaxy  

---

</div>

---

## 📑 Table of Contents
1. [System Overview & Prerequisites](#1-system-overview--prerequisites)
2. [Quick Installation & Launch Guide](#2-quick-installation--launch-guide)
3. [Graphical User Interface (GUI) Reference](#3-graphical-user-interface-gui-reference)
   - [3.1 Left Panel: Control & Disturbance Configuration](#31-left-panel-control--disturbance-configuration)
   - [3.2 Center Panel: Primary Tracking HUD & Rolling Error Chart](#32-center-panel-primary-tracking-hud--rolling-error-chart)
   - [3.3 Right Panel: 2D Arena Radar & ISRO Telemetry Audit](#33-right-panel-2d-arena-radar--isro-telemetry-audit)
4. [Step-by-Step Operating Workflows](#4-step-by-step-operating-workflows)
   - [Workflow 1: Nominal Closed-Loop Tracking](#workflow-1-nominal-closed-loop-tracking)
   - [Workflow 2: RF Side-Link Slew & Optical Handoff](#workflow-2-rf-side-link-slew--optical-handoff)
   - [Workflow 3: Beam Occlusion & Sub-Second Re-Acquisition](#workflow-3-beam-occlusion--sub-second-re-acquisition)
   - [Workflow 4: Severe Weather & Channel Stress-Testing](#workflow-4-severe-weather--channel-stress-testing)
   - [Workflow 5: Raw MP4/AVI Video Benchmark Evaluation](#workflow-5-raw-mp4avi-video-benchmark-evaluation)
5. [Telemetry Audit Logs & Data Export](#5-telemetry-audit-logs--data-export)
6. [Automated Test Suite Verification](#6-automated-test-suite-verification)
7. [Troubleshooting & FAQ](#7-troubleshooting--faq)

---

## 1. System Overview & Prerequisites

**LinkSight** is a high-performance desktop application designed for **Stage-1 Coarse Alignment** in Mobile Free-Space Optical Communication (FSOC) systems. It features a complete simulation of a $2000 \times 2000\,\text{px}$ operational space arena with a $640 \times 480\,\text{px}$ gimbaled optical camera, hardware-realistic channel noise models, and a hybrid classical-vision / sleep-wake AI tracking pipeline.

### Hardware & Software Prerequisites
* **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04+, or macOS 12+
* **Python Runtime:** Python `3.10` or `3.11` (recommended)
* **Processor:** Any modern dual-core CPU ($>2.0\,\text{GHz}$)
* **RAM:** Minimum $4\,\text{GB}$ ($8\,\text{GB}$ recommended)
* **Display Resolution:** $1920 \times 1080$ (Full HD) or higher recommended

---

## 2. Quick Installation & Launch Guide

### Method A: 1-Click Launch (Windows)
1. Open the project root folder: `SIH2026_ISRO_LinkSight_FSOC_ATP/`
2. Double-click **`Launch_LinkSight.bat`**.
3. The virtual terminal window opens and automatically launches the PySide6 Qt6 GUI.

### Method B: Manual Python Launch (Windows / Linux / macOS)
```bash
# 1. Navigate to project root
cd SIH2026_ISRO_LinkSight_FSOC_ATP

# 2. Create and activate a virtual environment (optional)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Launch the application
python main.py
```

### Method C: Standalone Executable (.EXE)
If built via PyInstaller (`Compile_Executable.bat`):
1. Navigate to: `dist/LinkSight_FSOC_ATP_Terminal/`
2. Double-click `LinkSight_FSOC_ATP_Terminal.exe`.

---

## 3. Graphical User Interface (GUI) Reference

The LinkSight interface is divided into **three functional columns**:

```
┌─────────────────────────┬───────────────────────────────┬─────────────────────────┐
│      CONTROL PANEL      │       PRIMARY HUD VIEWPORT    │    2D RADAR & TELEMETRY │
│ ─────────────────────── │ ───────────────────────────── │ ─────────────────────── │
│ • Mission Execution     │ • 640x480 Sensor Feed         │ • 2000x2000 Minimap     │
│ • System Parameters     │ • Target Reticle & Ellipse    │ • 5 ISRO Audit Cards    │
│ • RF Side-Link (Noise)  │ • Telemetry Status Overlay    │ • Chronological Logs    │
│ • Channel & Weather     │ ───────────────────────────── │ ─────────────────────── │
│ • Noise & Jitter Sliders│ • 15s Rolling Error Chart     │ • Export CSV / JSON     │
└─────────────────────────┴───────────────────────────────┴─────────────────────────┘
```

---

### 3.1 Left Panel: Control & Disturbance Configuration

The Left Panel allows configuring all physical parameters of the target, camera, and atmospheric channel:

#### 1. Mission Execution Controls
* **`RUN` (Green):** Starts the real-time simulation loop at $35\,\text{FPS}$.
* **`PAUSE` (Yellow):** Freezes the simulation kinematics for detailed state inspection.
* **`RESET` (Red):** Resets camera gimbal to center $(0^\circ, 0^\circ)$, clears telemetry history, and spawns the target according to the selected initial position mode.
* **`⚡ SIMULATE BEAM BREAK (1s)`:** Instantly blocks optical beacon transmission for 15–30 frames to test blind Kalman coasting and re-acquisition.

#### 2. System & Target Parameters
* **Camera FOV:** Displays current horizontal and vertical Field of View ($4.0^\circ \times 3.0^\circ$, corresponding to $160\,\text{px/deg}$).
* **Spot Shape:** Dropdown selector for `Square (Default)`, `Circle`, and `Crosshair`.
* **Spot Size:** Slider adjusting beacon aperture size from $5\,\text{px}$ to $20\,\text{px}$ (Default: $10\,\text{px}$).
* **Trajectory Motion Models:**
  * `Circular Orbit:` Target orbits in a continuous $r=220\,\text{px}$ circle.
  * `Linear Flight Path:` Straight-line pass with wall boundary reflections.
  * `Figure of 8:` Complex Lissajous non-linear looping trajectory.
  * `Random Walk:` Gauss-Markov continuous stochastic acceleration flight.
* **Initial Spawn Coordinates:**
  * `Random Mode (Default):` Target spawns anywhere across the $2000 \times 2000\,\text{px}$ scene.
  * `Custom Mode:` User-specified $(X, Y)$ coordinates for repeatable testing.
* **PTZ Slew Rates:** User-adjustable maximum angular gimbal rate ($1.0^\circ/\text{s}$ to $25.0^\circ/\text{s}$, default: $5.0^\circ/\text{s}$).

#### 3. RF Side-Link (Simulated Radio Coordinates)
* **`RF Coordinate (Sim)` Checkbox:** Default **ON**.
* **RF Noise Slider ($\sigma$):** Adjusts initial RF broadcast uncertainty from $\sigma=0\,\text{px}$ to $\sigma=200\,\text{px}$ (Default: $\sigma=80\,\text{px} \text{ (noise)}$).
* *Behavior:* On startup or reset, the gimbal uses coarse RF coordinates to slew rapidly towards the target region before handing over to optical lock.

#### 4. Atmospheric Channel & Weather Effects
* **Sky Radiance Slider:** Simulates ambient daylight background levels:
  * `Night (0%):` Deep space / night environment (zero background pedestal).
  * `Dusk (30%):` Twilight transition with moderate background glow.
  * `Overcast (60%):` Diffuse cloud cover with uniform DC offset.
  * `Noon (100%):` Maximum solar background radiance and glint potential.
* **Atmospheric Conditions Dropdown:**
  * `Clear:` Optimal transmission (nominal optical propagation).
  * `Haze:` Koschmieder scattering with mild beam attenuation.
  * `Fog:` Heavy Mie scattering causing strong contrast attenuation and halo blooming.
  * `Rain:` Dynamic moving streaks introducing localized intensity dropouts.
  * `Low Light:` Star-field illumination with reduced target signal amplitude.

#### 5. Sensor Noise & Structural Disturbance Generators
* **Salt & Pepper Noise (0–30%):** Injects random dead/saturated sensor pixels.
* **Gaussian Thermal Noise ($\sigma = 0\text{–}50$):** Simulates detector readout and thermal Johnson noise.
* **Poisson Shot Noise:** Quantum photon arrival noise.
* **Camera Jitter ($\pm 20\,\text{px}$):**
  * `Dynamic PSD Mode:` Structural jitter coupled to instantaneous target velocity and gimbal slew rate (Dabiri IEEE JSAC 2018 aerodynamic model).
  * `Steady Mode:` Fixed-amplitude high-frequency micro-vibrations.
* **Platform Motion ($\pm 20\,\text{px}$):** Simulates base vehicle drift in Linear, Circular, or Random walk modes.

---

### 3.2 Center Panel: Primary Tracking HUD & Rolling Error Chart

#### 1. Real-Time $640 \times 480$ Sensor HUD Viewport
* **Optical Sensor Crop:** Shows the active $4^\circ \times 3^\circ$ camera frame.
* **Target Reticle & Bounding Box:**
  * 🟢 **Green Reticle (`TRACKING [LOCKED]`):** Target acquired; error $\le 10.0\,\text{px}$.
  * 🟡 **Yellow Reticle (`DEGRADED [COASTING]`):** Beam occluded; Kalman + MicroGRU coasting.
  * 🔴 **Red Reticle / Blue Hexagons (`ACQUIRING [SPIRAL SEARCH]`):** Target lost; cut-hexagonal search lattice active.
* **Kalman Covariance Ellipse:** Live 3-sigma positional uncertainty ellipse rendered around target.
* **Live OSD Overlays:**
  * Upper-Left: `GIMBAL: AZ ±X.XX° | EL ±X.XX° | FOV 4.0°x3.0°`
  * Upper-Right: `TRACKING [LOCKED] | ESTIMATOR: KALMAN FILTER (MATH) | AI: SLEEP MODE`
  * Lower-Left: `CONF: 0.99 | SNR: 18.2 dB | APERTURE: 10.0 px`

#### 2. Rolling Tracking Error History Chart (15 Seconds)
* Displays real-time Euclidean tracking error in pixels ($e = \sqrt{\Delta x^2 + \Delta y^2}$).
* **Green Dashed Line (`SPEC LIMIT: 10.0 px`):** ISRO Requirement #17 threshold limit.
* Color-coded trajectory curve showing nominal tracking ($< 4.5\,\text{px}$), degraded coasting, and beam-break recoveries.

---

### 3.3 Right Panel: 2D Arena Radar & ISRO Telemetry Audit

#### 1. Top-Down 2D Space Arena Radar ($2000 \times 2000\,\text{px}$)
* **Black Canvas:** Full $2000 \times 2000\,\text{px}$ spatial domain.
* **Cyan Rectangle:** Current $640 \times 480\,\text{px}$ camera field of view in world coordinates.
* **Cyan Dot & Trail:** True beacon position and 200-frame motion history.
* **Center Crosshair:** Physical boresight $(1000, 1000)$.

#### 2. Five-Metric ISRO Performance Audit Cards
| Card | Metric | ISRO Limit | Pass / Fail Condition |
| :--- | :--- | :---: | :---: |
| **LOOP RATE** | Processing Throughput | $\ge 20.0\,\text{Hz}$ | 🟢 **PASS** ($\ge 30\,\text{Hz}$) |
| **TRACKING ERROR** | Steady-State Accuracy | $\le 10.0\,\text{px}$ | 🟢 **PASS** ($\le 4.5\,\text{px}$) |
| **LOCK RETENTION** | Link Availability | $> 95.0\%$ | 🟢 **PASS** ($96\text{–}99\%$) |
| **ACQUISITION TIME** | Initial Lock Latency | $\le 2.0\,\text{s}$ | 🟢 **PASS** ($0.2\text{–}1.2\,\text{s}$) |
| **RE-ACQUISITION** | Recovery Latency | $\le 1.0\,\text{s}$ | 🟢 **PASS** ($0.15\text{–}0.65\,\text{s}$) |

#### 3. Chronological System Event Log
* Real-time stream of timestamped state transitions (`ACQUIRED`, `BEAM_BREAK`, `COASTING`, `REACQUIRED`, `CONFIG_CHANGE`).

---

## 4. Step-by-Step Operating Workflows

### Workflow 1: Nominal Closed-Loop Tracking
1. Launch the application (`python main.py` or `Launch_LinkSight.bat`).
2. Verify **Motion Model** is set to `Circular Orbit`.
3. Click **`RUN`**.
4. Observe:
   * Target is acquired within $< 1.0\,\text{s}$.
   * Gimbal centers the beacon in the center reticle.
   * Tracking error settles to **$0.5\text{–}3.5\,\text{px}$** (well below the $10\,\text{px}$ spec limit).
   * All 5 ISRO audit cards display green `NORMAL / PASS` badges.

---

### Workflow 2: RF Side-Link Slew & Optical Handoff
1. Set **Initial Position Mode** to `Random`.
2. Ensure **`RF Coordinate (Sim)`** is **Checked** with noise $\sigma=80\,\text{px}$.
3. Click **`RESET`** then **`RUN`**.
4. Observe the two-tier handoff sequence:
   * **Phase 1 (RF Slew):** Status reads `RF COARSE POINTING [SLEW]`. The gimbal executes high-rate slew ($5^\circ/\text{s}$) directly toward the noisy RF broadcast coordinate.
   * **Phase 2 (Optical Lock):** Once within the optical FOV basket, the classical detector locks the beacon and transitions immediately to `TRACKING [LOCKED]`.

---

### Workflow 3: Beam Occlusion & Sub-Second Re-Acquisition
1. While the system is actively tracking in `RUN` mode, click **`⚡ SIMULATE BEAM BREAK (1s)`**.
2. Observe the automatic failover lifecycle:
   * **Frames 1–20 ($0.0\text{–}0.67\,\text{s}$):** Status enters `DEGRADED [COASTING]`. The optical detection is lost, but the Kalman filter + MicroGRU neural coaster maintain smooth gimbal tracking along the extrapolated trajectory.
   * **Frames 21+ ($>0.67\,\text{s}$):** If the dropout persists beyond the coasting budget, status transitions to `ACQUIRING [SPIRAL SEARCH]`. The cut-hexagonal search lattice expands outward.
   * **Recovery:** When the laser unblocks, the beacon is recaptured within **$0.2\text{–}0.6\,\text{s}$**, logging a `REACQUIRED` event.

---

### Workflow 4: Severe Weather & Channel Stress-Testing
1. In the Left Control Panel, adjust:
   * **Atmospheric Channel:** Select `Fog` or `Rain`.
   * **Sky Radiance:** Drag to `80% (Overcast/Daylight)`.
   * **Salt & Pepper Noise:** Set to `10%`.
   * **Gaussian Noise:** Set slider to $\sigma = 15\,\text{px}$.
   * **Camera Jitter:** Enable with `Dynamic PSD` mode.
2. Click **`RUN`**.
3. Observe how the Morphological Top-Hat filter strips the ambient background luminance and the Dual CFAR threshold cleanly isolates the laser spot with zero false alarms.

---

### Workflow 5: Raw MP4/AVI Video Benchmark Evaluation
1. In the Control Panel, click **`LOAD MP4 / AVI VIDEO`**.
2. Select the included benchmark video: `assets/benchmark_sample.mp4`.
3. Click **`RUN`**.
4. The simulator switches to video playback mode, processing real optical recorded frames through the full detector and Kalman state estimator.

---

## 5. Telemetry Audit Logs & Data Export

LinkSight maintains a high-precision per-frame telemetry log of every mission.

### Exporting Logs
1. Click **`EXPORT TELEMETRY (CSV / JSON)`** at the bottom of the right panel.
2. Choose your destination directory.
3. The application writes two synchronized files:
   * **`telemetry_<timestamp>.csv`** (Tabular time-series data for Excel, Python, or MATLAB).
   * **`telemetry_<timestamp>.json`** (Structured JSON containing mission metadata, event logs, and per-frame records).

### CSV Data Schema
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `time_s` | `float` | Elapsed mission time in seconds ($0.001\,\text{s}$ precision) |
| `frame` | `int` | Sequential frame counter |
| `fps` | `float` | Instantaneous processing frame rate |
| `error_px` | `float` | Euclidean tracking error from reticle center |
| `status` | `string` | System state (`TRACKING`, `DEGRADED`, `LOST`, `ACQUIRING`) |
| `confidence` | `float` | Optical detection confidence ($0.0\text{–}1.0$) |
| `pan_deg` | `float` | Current gimbal Azimuth angle in degrees |
| `tilt_deg` | `float` | Current gimbal Elevation angle in degrees |
| `loss_count` | `int` | Cumulative target loss event counter |
| `lock_retention` | `float` | Cumulative percentage of frames in locked state |
| `tracking_mode` | `string` | Active subsystem (`MATHEMATICS`, `CNN (AI)`, `MicroGRU (AI)`, `SPIRAL SCAN`) |

---

## 6. Automated Test Suite Verification

LinkSight includes a comprehensive 15-test automated verification suite:

```bash
# Run all unit tests
python -m unittest discover tests

# Output:
# ...............
# ----------------------------------------------------------------------
# Ran 15 tests in 3.12s
# OK
```

### Test Coverage Breakdown
1. **`test_detector.py`:** Morphological Top-Hat filtering, SNR calculation, and false alarm rate under salt-and-pepper noise.
2. **`test_tracker.py`:** Kalman 2-hit confirmation, dynamic covariance scaling, occlusion coasting, and $\chi^2$ gating.
3. **`test_controller.py`:** PID anti-windup clamping, slew rate limiter ($5^\circ/\text{s}$), and deadband.
4. **`test_reacquisition.py`:** Hexagonal lattice coordinates, step sizes, and dwell timing.
5. **`test_pipeline.py`:** End-to-end headless pipeline integration, convergence, and CSV/JSON export.

---

## 7. Troubleshooting & FAQ

#### Q1: The GUI opens but shows a black screen or low FPS.
* **Fix:** Ensure you are running on Python 3.10 or 3.11. Update your graphics drivers if OpenGL hardware acceleration is supported.

#### Q2: `ModuleNotFoundError: No module named 'PySide6'`
* **Fix:** Install dependencies within your active Python environment:
  ```bash
  pip install -r requirements.txt
  ```

#### Q3: How do I test the prototype without ONNX runtime?
* **Answer:** LinkSight includes built-in fallback mathematical approximations. If `onnxruntime` is not installed, the classical Kalman coaster and PID controllers operate transparently with zero errors.

#### Q4: Why does the beacon start outside the camera viewport on Reset?
* **Answer:** This is intentional! When `Initial Position Mode = Random`, the target spawns randomly across the $2000 \times 2000\,\text{px}$ arena to demonstrate the RF side-link coarse slew and cut-hexagonal re-acquisition search capabilities.

---

<div align="center">

**Smart India Hackathon 2026** | **ISRO Problem Statement PS-26169**  
*Developed by Team Guardians of the Galaxy*

</div>
