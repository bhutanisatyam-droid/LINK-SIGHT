"""Left-side Control Panel widget for FSOC coarse-pointing system.

Provides telemetry control inputs:
- Mission execution controls (Run / Pause / Reset)
- Motion model selector (Circular, Straight, Figure-8, Random)
- Disturbance injection toggles and intensity sliders (Salt-and-Pepper, Gaussian, Poisson, Jitter, Platform Drift)
- Atmospheric condition selector (Clear, Haze, Fog, Rain, Low-Light)
- Physical optical parameters (Beacon size, Max PTZ speed)
- Visually distinct Benchmark-2 video file loader section
- Session log export (CSV / JSON)
"""

import datetime
import os
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fsoc.disturbance import AtmosphericCondition, DisturbanceConfig, JitterMode
from fsoc.frame_source import MotionModel
from fsoc.gui.theme import (
    COLOR_BORDER,
    COLOR_BORDER_LIGHT,
    COLOR_DEGRADED,
    COLOR_HUD_ACCENT,
    COLOR_LOCKED,
    COLOR_LOST,
    COLOR_SURFACE_1,
    COLOR_SURFACE_2,
    COLOR_SURFACE_INPUT,
    COLOR_TEXT_DIM,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    get_label_font,
    get_mono_font,
)


class ControlPanelWidget(QWidget):
    """Dense mission control dashboard input panel."""

    # Control Signals
    motion_model_changed = Signal(object) # MotionModel
    target_size_changed = Signal(int)
    ptz_speed_changed = Signal(float)
    run_clicked = Signal()
    pause_clicked = Signal()
    reset_clicked = Signal()
    occlude_clicked = Signal()
    video_file_selected = Signal(str)
    switch_to_sim_clicked = Signal()
    export_logs_clicked = Signal()

    def __init__(self, disturbance_config: DisturbanceConfig, parent=None):
        super().__init__(parent)
        self.config = disturbance_config
        self.setFixedWidth(310)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for dense parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # --- SECTION 1: EXECUTION CONTROLS ---
        exec_frame = QFrame()
        exec_frame.setObjectName("SectionFrame")
        exec_layout = QVBoxLayout(exec_frame)
        exec_layout.setContentsMargins(6, 6, 6, 6)
        exec_layout.setSpacing(6)

        hdr_exec = QLabel("MISSION EXECUTION")
        hdr_exec.setObjectName("SectionHeader")
        exec_layout.addWidget(hdr_exec)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.btn_run = QPushButton("RUN")
        self.btn_run.setObjectName("BtnRun")
        self.btn_run.clicked.connect(self.run_clicked.emit)

        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.clicked.connect(self.pause_clicked.emit)

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setObjectName("BtnReset")
        self.btn_reset.clicked.connect(self.reset_clicked.emit)

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_reset)
        exec_layout.addLayout(btn_row)

        self.btn_occlude = QPushButton("⚡ SIMULATE BEAM BREAK (1s)")
        self.btn_occlude.setObjectName("BtnOcclude")
        self.btn_occlude.setFont(get_mono_font(size_pt=8, bold=True))
        self.btn_occlude.setStyleSheet(f"""
            QPushButton#BtnOcclude {{
                background-color: #2D1A1A;
                color: #FF7B72;
                border: 1px solid #5A2A2A;
                border-radius: 2px;
                padding: 4px;
            }}
            QPushButton#BtnOcclude:hover {{
                background-color: #3D2222;
                border: 1px solid #8A3A3A;
            }}
            QPushButton#BtnOcclude:pressed {{
                background-color: #5A2A2A;
            }}
        """)
        self.btn_occlude.clicked.connect(self.occlude_clicked.emit)
        exec_layout.addWidget(self.btn_occlude)
        layout.addWidget(exec_frame)

        # --- SECTION 2: MOTION MODEL & OPTICS ---
        motion_frame = QFrame()
        motion_frame.setObjectName("SectionFrame")
        motion_layout = QVBoxLayout(motion_frame)
        motion_layout.setContentsMargins(6, 6, 6, 6)
        motion_layout.setSpacing(6)

        hdr_motion = QLabel("TRAJECTORY & OPTICAL SPECS")
        hdr_motion.setObjectName("SectionHeader")
        motion_layout.addWidget(hdr_motion)

        # Motion model dropdown
        lbl_model = QLabel("Beacon Trajectory Model:")
        lbl_model.setFont(get_label_font(size_pt=8))
        lbl_model.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        motion_layout.addWidget(lbl_model)

        self.combo_motion = QComboBox()
        self.combo_motion.addItem("Circular Orbit (Default)", MotionModel.CIRCULAR)
        self.combo_motion.addItem("Linear Flight Path", MotionModel.STRAIGHT)
        self.combo_motion.addItem("Figure-8 (Lemniscate)", MotionModel.FIGURE_8)
        self.combo_motion.addItem("Gauss-Markov Random Walk", MotionModel.RANDOM)
        self.combo_motion.currentIndexChanged.connect(self._on_motion_changed)
        motion_layout.addWidget(self.combo_motion)

        # Target size and PTZ speed
        params_grid = QGridLayout()
        params_grid.setContentsMargins(0, 4, 0, 0)
        params_grid.setHorizontalSpacing(8)
        params_grid.setVerticalSpacing(4)

        lbl_size = QLabel("Target Size:")
        lbl_size.setFont(get_label_font(size_pt=8))
        self.spin_size = QSpinBox()
        self.spin_size.setRange(5, 20)
        self.spin_size.setValue(10)
        self.spin_size.setSuffix(" px")
        self.spin_size.valueChanged.connect(self.target_size_changed.emit)

        lbl_speed = QLabel("PTZ Max Speed:")
        lbl_speed.setFont(get_label_font(size_pt=8))
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(2.0, 20.0)
        self.spin_speed.setValue(8.0)
        self.spin_speed.setSingleStep(0.5)
        self.spin_speed.setSuffix(" °/s")
        self.spin_speed.valueChanged.connect(self.ptz_speed_changed.emit)

        params_grid.addWidget(lbl_size, 0, 0)
        params_grid.addWidget(self.spin_size, 0, 1)
        params_grid.addWidget(lbl_speed, 1, 0)
        params_grid.addWidget(self.spin_speed, 1, 1)

        motion_layout.addLayout(params_grid)
        layout.addWidget(motion_frame)

        # --- SECTION 3: ATMOSPHERIC CONDITIONS ---
        atmo_frame = QFrame()
        atmo_frame.setObjectName("SectionFrame")
        atmo_layout = QVBoxLayout(atmo_frame)
        atmo_layout.setContentsMargins(6, 6, 6, 6)
        atmo_layout.setSpacing(6)

        hdr_atmo = QLabel("ATMOSPHERIC CHANNEL EFFECTS")
        hdr_atmo.setObjectName("SectionHeader")
        atmo_layout.addWidget(hdr_atmo)

        self.combo_atmo = QComboBox()
        self.combo_atmo.addItem("Clear Channel (Nominal Space)", AtmosphericCondition.CLEAR)
        self.combo_atmo.addItem("Haze (Koschmieder Scattering)", AtmosphericCondition.HAZE)
        self.combo_atmo.addItem("Dense Fog (Attenuation & Blur)", AtmosphericCondition.FOG)
        self.combo_atmo.addItem("Dynamic Rain (Angled Streaks)", AtmosphericCondition.RAIN)
        self.combo_atmo.addItem("Low-Light (Photon Starvation)", AtmosphericCondition.LOW_LIGHT)
        self.combo_atmo.currentIndexChanged.connect(self._on_atmo_changed)
        atmo_layout.addWidget(self.combo_atmo)
        layout.addWidget(atmo_frame)

        # --- SECTION 4: DISTURBANCE INJECTION ---
        dist_frame = QFrame()
        dist_frame.setObjectName("SectionFrame")
        dist_layout = QVBoxLayout(dist_frame)
        dist_layout.setContentsMargins(6, 6, 6, 6)
        dist_layout.setSpacing(6)

        hdr_dist = QLabel("DISTURBANCE INJECTION ENGINE")
        hdr_dist.setObjectName("SectionHeader")
        dist_layout.addWidget(hdr_dist)

        # --- Sky Brightness / Day-Night Ambient Radiance ---
        sky_row = QHBoxLayout()
        lbl_sky = QLabel("Sky Radiance (Day/Night):")
        lbl_sky.setFont(get_label_font(size_pt=8, bold=True))
        self.lbl_sky_val = QLabel(f"{int(self.config.sky_radiance * 100)}% (Night)")
        self.lbl_sky_val.setFont(get_mono_font(size_pt=8))
        sky_row.addWidget(lbl_sky)
        sky_row.addStretch()
        sky_row.addWidget(self.lbl_sky_val)
        dist_layout.addLayout(sky_row)

        self.slider_sky = QSlider(Qt.Horizontal)
        self.slider_sky.setRange(0, 100)
        self.slider_sky.setValue(int(self.config.sky_radiance * 100))
        self.slider_sky.valueChanged.connect(self._on_sky_slider)
        dist_layout.addWidget(self.slider_sky)

        # Quick Presets Row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for label, val in [("Night", 0), ("Dusk", 30), ("Overcast", 60), ("Noon", 100)]:
            btn = QPushButton(label)
            btn.setFixedHeight(20)
            btn.setFont(get_mono_font(size_pt=7))
            btn.setStyleSheet(f"background-color: {COLOR_SURFACE_INPUT}; border: 1px solid {COLOR_BORDER};")
            btn.clicked.connect(lambda _, v=val, l=label: self._set_sky_preset(v, l))
            preset_row.addWidget(btn)
        dist_layout.addLayout(preset_row)

        # Salt and Pepper Noise (10% default)
        sp_row = QHBoxLayout()
        self.chk_sp = QCheckBox("Salt & Pepper Noise:")
        self.chk_sp.setChecked(self.config.enable_salt_pepper)
        self.chk_sp.toggled.connect(self._on_sp_toggled)
        self.lbl_sp_val = QLabel(f"{int(self.config.salt_pepper_ratio * 100)}%")
        self.lbl_sp_val.setFont(get_mono_font(size_pt=8))
        sp_row.addWidget(self.chk_sp)
        sp_row.addStretch()
        sp_row.addWidget(self.lbl_sp_val)
        dist_layout.addLayout(sp_row)

        self.slider_sp = QSlider(Qt.Horizontal)
        self.slider_sp.setRange(0, 30) # 0 to 30%
        self.slider_sp.setValue(int(self.config.salt_pepper_ratio * 100))
        self.slider_sp.valueChanged.connect(self._on_sp_slider)
        dist_layout.addWidget(self.slider_sp)

        # Gaussian Noise
        gauss_row = QHBoxLayout()
        self.chk_gauss = QCheckBox("Gaussian Thermal Noise:")
        self.chk_gauss.setChecked(self.config.enable_gaussian)
        self.chk_gauss.toggled.connect(self._on_gauss_toggled)
        self.lbl_gauss_val = QLabel(f"σ={int(self.config.gaussian_sigma)}")
        self.lbl_gauss_val.setFont(get_mono_font(size_pt=8))
        gauss_row.addWidget(self.chk_gauss)
        gauss_row.addStretch()
        gauss_row.addWidget(self.lbl_gauss_val)
        dist_layout.addLayout(gauss_row)

        self.slider_gauss = QSlider(Qt.Horizontal)
        self.slider_gauss.setRange(5, 50)
        self.slider_gauss.setValue(int(self.config.gaussian_sigma))
        self.slider_gauss.valueChanged.connect(self._on_gauss_slider)
        dist_layout.addWidget(self.slider_gauss)

        # Poisson Noise
        self.chk_poisson = QCheckBox("Poisson Shot Noise (Photon Fluctuation)")
        self.chk_poisson.setChecked(self.config.enable_poisson)
        self.chk_poisson.toggled.connect(lambda v: setattr(self.config, "enable_poisson", v))
        dist_layout.addWidget(self.chk_poisson)

        # Camera Jitter (±20px default)
        jit_row = QHBoxLayout()
        self.chk_jitter = QCheckBox("Camera Jitter:")
        self.chk_jitter.setChecked(self.config.enable_camera_jitter)
        self.chk_jitter.toggled.connect(self._on_jitter_toggled)
        self.lbl_jit_val = QLabel(f"±{int(self.config.camera_jitter_max_px)}px")
        self.lbl_jit_val.setFont(get_mono_font(size_pt=8))
        jit_row.addWidget(self.chk_jitter)
        jit_row.addStretch()
        jit_row.addWidget(self.lbl_jit_val)
        dist_layout.addLayout(jit_row)

        # Jitter Mode Selector (Steady vs. Dynamic PSD)
        jit_mode_row = QHBoxLayout()
        lbl_jmode = QLabel("Jitter Physics:")
        lbl_jmode.setFont(get_label_font(size_pt=8))
        lbl_jmode.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        self.combo_jitter_mode = QComboBox()
        self.combo_jitter_mode.addItem("Dynamic PSD (Dabiri)", JitterMode.DYNAMIC_PSD)
        self.combo_jitter_mode.addItem("Steady (Fixed ±px)", JitterMode.STEADY)
        self.combo_jitter_mode.setCurrentIndex(0 if self.config.jitter_mode == JitterMode.DYNAMIC_PSD else 1)
        self.combo_jitter_mode.currentIndexChanged.connect(self._on_jitter_mode_changed)
        jit_mode_row.addWidget(lbl_jmode)
        jit_mode_row.addWidget(self.combo_jitter_mode)
        dist_layout.addLayout(jit_mode_row)

        self.slider_jitter = QSlider(Qt.Horizontal)
        self.slider_jitter.setRange(2, 30)
        self.slider_jitter.setValue(int(self.config.camera_jitter_max_px))
        self.slider_jitter.valueChanged.connect(self._on_jitter_slider)
        dist_layout.addWidget(self.slider_jitter)

        # Platform Motion (±20px default)
        plat_row = QHBoxLayout()
        self.chk_plat = QCheckBox("Platform Motion:")
        self.chk_plat.setChecked(self.config.enable_platform_motion)
        self.chk_plat.toggled.connect(self._on_plat_toggled)
        self.lbl_plat_val = QLabel(f"±{int(self.config.platform_motion_max_px)}px")
        self.lbl_plat_val.setFont(get_mono_font(size_pt=8))
        plat_row.addWidget(self.chk_plat)
        plat_row.addStretch()
        plat_row.addWidget(self.lbl_plat_val)
        dist_layout.addLayout(plat_row)

        self.slider_plat = QSlider(Qt.Horizontal)
        self.slider_plat.setRange(2, 30)
        self.slider_plat.setValue(int(self.config.platform_motion_max_px))
        self.slider_plat.valueChanged.connect(self._on_plat_slider)
        dist_layout.addWidget(self.slider_plat)

        layout.addWidget(dist_frame)

        # --- SECTION 5: BENCHMARK-2 VIDEO MODE (VISUALLY SEPARATED) ---
        # Strictly separated visual identity per specification
        bench_frame = QFrame()
        bench_frame.setObjectName("BenchmarkCard")
        bench_frame.setStyleSheet(f"""
            QFrame#BenchmarkCard {{
                background-color: #0E1520;
                border: 1px solid #1C3352;
                border-radius: 2px;
                padding: 6px;
            }}
        """)
        bench_layout = QVBoxLayout(bench_frame)
        bench_layout.setContentsMargins(6, 6, 6, 6)
        bench_layout.setSpacing(6)

        hdr_bench = QLabel("BENCHMARK-2: RAW VIDEO VALIDATION")
        hdr_bench.setObjectName("SectionHeader")
        hdr_bench.setStyleSheet(f"color: {COLOR_HUD_ACCENT}; border-bottom-color: #1C3352;")
        bench_layout.addWidget(hdr_bench)

        self.lbl_mode_status = QLabel("ACTIVE: SIMULATOR (SYNTHETIC)")
        self.lbl_mode_status.setFont(get_mono_font(size_pt=8, bold=True))
        self.lbl_mode_status.setStyleSheet(f"color: {COLOR_LOCKED};")
        bench_layout.addWidget(self.lbl_mode_status)

        desc_bench = QLabel(
            "Validates detector/tracker on un-simulated raw MP4 footage. PTZ commands are logged as no-ops."
        )
        desc_bench.setWordWrap(True)
        desc_bench.setFont(get_label_font(size_pt=8))
        desc_bench.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        bench_layout.addWidget(desc_bench)

        self.btn_load_video = QPushButton("LOAD VIDEO FILE (.MP4)...")
        self.btn_load_video.setStyleSheet(f"""
            background-color: #162438;
            color: {COLOR_HUD_ACCENT};
            border: 1px solid #29456B;
        """)
        self.btn_load_video.clicked.connect(self._on_browse_video)
        bench_layout.addWidget(self.btn_load_video)

        self.btn_switch_sim = QPushButton("RETURN TO SIMULATOR")
        self.btn_switch_sim.setEnabled(False)
        self.btn_switch_sim.clicked.connect(self._on_return_sim)
        bench_layout.addWidget(self.btn_switch_sim)

        layout.addWidget(bench_frame)

        # --- SECTION 6: TELEMETRY EXPORT ---
        export_frame = QFrame()
        export_frame.setObjectName("SectionFrame")
        export_layout = QVBoxLayout(export_frame)
        export_layout.setContentsMargins(6, 6, 6, 6)
        export_layout.setSpacing(6)

        hdr_exp = QLabel("AUDIT LOG GENERATION")
        hdr_exp.setObjectName("SectionHeader")
        export_layout.addWidget(hdr_exp)

        self.btn_export = QPushButton("EXPORT SESSION LOG (CSV/JSON)")
        self.btn_export.clicked.connect(self.export_logs_clicked.emit)
        export_layout.addWidget(self.btn_export)

        self.lbl_export_status = QLabel("Ready for session export")
        self.lbl_export_status.setFont(get_mono_font(size_pt=7))
        self.lbl_export_status.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        export_layout.addWidget(self.lbl_export_status)

        layout.addWidget(export_frame)
        layout.addStretch()

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _on_motion_changed(self, index: int) -> None:
        model = self.combo_motion.itemData(index)
        if model is not None:
            self.motion_model_changed.emit(model)

    def _on_atmo_changed(self, index: int) -> None:
        cond = self.combo_atmo.itemData(index)
        if cond is not None:
            self.config.atmospheric_condition = cond

    def _on_sky_slider(self, val: int) -> None:
        rad = val / 100.0
        self.config.sky_radiance = rad
        tag = "Night" if val < 20 else ("Dusk" if val < 50 else ("Overcast" if val < 80 else "High Noon"))
        self.lbl_sky_val.setText(f"{val}% ({tag})")

    def _set_sky_preset(self, val: int, label: str) -> None:
        self.slider_sky.setValue(val)
        self._on_sky_slider(val)

    def _on_sp_toggled(self, checked: bool) -> None:
        self.config.enable_salt_pepper = checked

    def _on_sp_slider(self, val: int) -> None:
        ratio = val / 100.0
        self.config.salt_pepper_ratio = ratio
        self.lbl_sp_val.setText(f"{val}%")

    def _on_gauss_toggled(self, checked: bool) -> None:
        self.config.enable_gaussian = checked

    def _on_gauss_slider(self, val: int) -> None:
        self.config.gaussian_sigma = float(val)
        self.lbl_gauss_val.setText(f"σ={val}")

    def _on_jitter_toggled(self, checked: bool) -> None:
        self.config.enable_camera_jitter = checked

    def _on_jitter_mode_changed(self, index: int) -> None:
        mode = self.combo_jitter_mode.itemData(index)
        if mode is not None:
            self.config.jitter_mode = mode

    def _on_jitter_slider(self, val: int) -> None:
        self.config.camera_jitter_max_px = float(val)
        self.lbl_jit_val.setText(f"±{val}px")

    def _on_plat_toggled(self, checked: bool) -> None:
        self.config.enable_platform_motion = checked

    def _on_plat_slider(self, val: int) -> None:
        self.config.platform_motion_max_px = float(val)
        self.lbl_plat_val.setText(f"±{val}px")

    def _on_browse_video(self) -> None:
        default_dir = os.path.abspath("assets")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Benchmark-2 Video File",
            default_dir,
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
        )
        if file_path:
            self.video_file_selected.emit(file_path)
            self.lbl_mode_status.setText("ACTIVE: BENCHMARK-2 (RAW FOOTAGE)")
            self.lbl_mode_status.setStyleSheet(f"color: {COLOR_HUD_ACCENT};")
            self.btn_switch_sim.setEnabled(True)

    def _on_return_sim(self) -> None:
        self.switch_to_sim_clicked.emit()
        self.lbl_mode_status.setText("ACTIVE: SIMULATOR (SYNTHETIC)")
        self.lbl_mode_status.setStyleSheet(f"color: {COLOR_LOCKED};")
        self.btn_switch_sim.setEnabled(False)

    def set_export_status(self, text: str, is_success: bool = True) -> None:
        color = COLOR_LOCKED if is_success else COLOR_LOST
        self.lbl_export_status.setText(text)
        self.lbl_export_status.setStyleSheet(f"color: {color};")
