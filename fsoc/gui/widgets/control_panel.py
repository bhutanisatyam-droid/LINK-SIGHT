"""Left-side Control Panel widget for FSOC coarse-pointing system.

Compact mission control dashboard — all sections fit without excessive scrolling.
Provides:
- Mission execution controls (Run / Pause / Reset / Beam Break)
- ISRO System Parameters (FOV, Shape, Trajectory, Init Pos, RF Link, Speeds)
- Atmospheric condition selector
- Disturbance injection toggles + sliders
- Benchmark-2 video file loader
- Session log export
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
from fsoc.frame_source import MotionModel, TargetShape
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


# ── helpers ──────────────────────────────────────────────────────────────────

def _inline_row(label_text: str, widget: QWidget, label_w: int = 72) -> QHBoxLayout:
    """Return an HBoxLayout with a fixed-width label + widget on the same line."""
    row = QHBoxLayout()
    row.setSpacing(4)
    row.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFont(get_label_font(size_pt=8))
    lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
    lbl.setFixedWidth(label_w)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


def _section(title: str) -> tuple[QFrame, QVBoxLayout]:
    """Return a styled section frame + its inner layout."""
    frame = QFrame()
    frame.setObjectName("SectionFrame")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(5, 4, 5, 4)
    lay.setSpacing(3)
    hdr = QLabel(title)
    hdr.setObjectName("SectionHeader")
    lay.addWidget(hdr)
    return frame, lay


# ── widget ────────────────────────────────────────────────────────────────────

class ControlPanelWidget(QWidget):
    """Compact mission control dashboard input panel."""

    # Signals
    motion_model_changed  = Signal(object)          # MotionModel
    target_size_changed   = Signal(int)
    ptz_speed_changed     = Signal(float)
    fov_changed           = Signal(float, float)    # (pan_deg, tilt_deg)
    target_shape_changed  = Signal(object)          # TargetShape
    initial_pos_changed   = Signal(str, float, float)  # (mode, x, y)
    pan_speed_changed     = Signal(float)
    tilt_speed_changed    = Signal(float)
    rf_link_changed       = Signal(bool, float)     # (active, uncertainty_px)
    run_clicked           = Signal()
    pause_clicked         = Signal()
    reset_clicked         = Signal()
    occlude_clicked       = Signal()
    video_file_selected   = Signal(str)
    switch_to_sim_clicked = Signal()
    export_logs_clicked   = Signal()

    def __init__(self, disturbance_config: DisturbanceConfig, parent=None):
        super().__init__(parent)
        self.config = disturbance_config
        self.setFixedWidth(320)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── SECTION 1: MISSION EXECUTION ─────────────────────────────────────
        exec_frame, exec_layout = _section("MISSION EXECUTION")

        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)

        self.btn_run = QPushButton("RUN")
        self.btn_run.setObjectName("BtnRun")
        self.btn_run.setFixedHeight(24)
        self.btn_run.clicked.connect(self.run_clicked.emit)

        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_pause.setFixedHeight(24)
        self.btn_pause.clicked.connect(self.pause_clicked.emit)

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setObjectName("BtnReset")
        self.btn_reset.setFixedHeight(24)
        self.btn_reset.clicked.connect(self.reset_clicked.emit)

        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_reset)
        exec_layout.addLayout(btn_row)

        self.btn_occlude = QPushButton("⚡ SIMULATE BEAM BREAK (1s)")
        self.btn_occlude.setObjectName("BtnOcclude")
        self.btn_occlude.setFixedHeight(22)
        self.btn_occlude.setFont(get_mono_font(size_pt=8, bold=True))
        self.btn_occlude.setStyleSheet(f"""
            QPushButton#BtnOcclude {{
                background-color: #2D1A1A; color: #FF7B72;
                border: 1px solid #5A2A2A; border-radius: 2px; padding: 2px;
            }}
            QPushButton#BtnOcclude:hover {{ background-color: #3D2222; border: 1px solid #8A3A3A; }}
            QPushButton#BtnOcclude:pressed {{ background-color: #5A2A2A; }}
        """)
        self.btn_occlude.clicked.connect(self.occlude_clicked.emit)
        exec_layout.addWidget(self.btn_occlude)
        layout.addWidget(exec_frame)

        # ── SECTION 2: ISRO SYSTEM PARAMETERS ────────────────────────────────
        isro_frame, isro_layout = _section("ISRO SYSTEM PARAMETERS")

        # Camera FOV: Pan & Tilt side-by-side on 1 row
        fov_row = QHBoxLayout()
        fov_row.setSpacing(4)
        fov_lbl = QLabel("Camera FOV:")
        fov_lbl.setFont(get_label_font(size_pt=8))
        fov_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        fov_lbl.setFixedWidth(72)
        fov_row.addWidget(fov_lbl)

        self.spin_fov_pan = QDoubleSpinBox()
        self.spin_fov_pan.setRange(1.0, 12.0)
        self.spin_fov_pan.setValue(4.0)
        self.spin_fov_pan.setSingleStep(0.5)
        self.spin_fov_pan.setPrefix("Pan ")
        self.spin_fov_pan.setSuffix("°")
        self.spin_fov_pan.valueChanged.connect(self._on_fov_changed)

        self.spin_fov_tilt = QDoubleSpinBox()
        self.spin_fov_tilt.setRange(1.0, 9.0)
        self.spin_fov_tilt.setValue(3.0)
        self.spin_fov_tilt.setSingleStep(0.5)
        self.spin_fov_tilt.setPrefix("Tilt ")
        self.spin_fov_tilt.setSuffix("°")
        self.spin_fov_tilt.valueChanged.connect(self._on_fov_changed)

        fov_row.addWidget(self.spin_fov_pan, 1)
        fov_row.addWidget(self.spin_fov_tilt, 1)
        isro_layout.addLayout(fov_row)

        # Shape
        self.combo_shape = QComboBox()
        self.combo_shape.addItem("Square (Default)", TargetShape.SQUARE)
        self.combo_shape.addItem("Circle", TargetShape.CIRCLE)
        self.combo_shape.addItem("Cross (+)", TargetShape.CROSS)
        self.combo_shape.currentIndexChanged.connect(self._on_shape_changed)
        isro_layout.addLayout(_inline_row("Spot Shape:", self.combo_shape))

        # Trajectory
        self.combo_motion = QComboBox()
        self.combo_motion.addItem("Circular Orbit", MotionModel.CIRCULAR)
        self.combo_motion.addItem("Linear Flight", MotionModel.STRAIGHT)
        self.combo_motion.addItem("Figure-8", MotionModel.FIGURE_8)
        self.combo_motion.addItem("Random Walk", MotionModel.RANDOM)
        self.combo_motion.currentIndexChanged.connect(self._on_motion_changed)
        isro_layout.addLayout(_inline_row("Trajectory:", self.combo_motion))

        # Initial Target Location
        self.combo_initpos = QComboBox()
        self.combo_initpos.addItem("Random (ISRO Default)", "random")
        self.combo_initpos.addItem("Custom Coordinates", "custom")
        self.combo_initpos.currentIndexChanged.connect(self._on_initpos_changed)
        isro_layout.addLayout(_inline_row("Init. Pos:", self.combo_initpos))

        # Coordinates X & Y side-by-side on 1 row
        xy_row = QHBoxLayout()
        xy_row.setSpacing(4)
        xy_lbl = QLabel("Spawn (X,Y):")
        xy_lbl.setFont(get_label_font(size_pt=8))
        xy_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        xy_lbl.setFixedWidth(72)
        xy_row.addWidget(xy_lbl)

        self.spin_custom_x = QDoubleSpinBox()
        self.spin_custom_x.setRange(100.0, 1900.0)
        self.spin_custom_x.setValue(1000.0)
        self.spin_custom_x.setSingleStep(50.0)
        self.spin_custom_x.setPrefix("X ")
        self.spin_custom_x.setSuffix("px")
        self.spin_custom_x.setEnabled(False)
        self.spin_custom_x.valueChanged.connect(self._on_initpos_changed)

        self.spin_custom_y = QDoubleSpinBox()
        self.spin_custom_y.setRange(100.0, 1900.0)
        self.spin_custom_y.setValue(1000.0)
        self.spin_custom_y.setSingleStep(50.0)
        self.spin_custom_y.setPrefix("Y ")
        self.spin_custom_y.setSuffix("px")
        self.spin_custom_y.setEnabled(False)
        self.spin_custom_y.valueChanged.connect(self._on_initpos_changed)

        xy_row.addWidget(self.spin_custom_x, 1)
        xy_row.addWidget(self.spin_custom_y, 1)
        isro_layout.addLayout(xy_row)

        # ── RF SIDE-LINK ──────────────────────────────────────────────────
        rf_top = QHBoxLayout()
        rf_top.setSpacing(4)
        self.chk_rf = QCheckBox("RF Side-Link (Sim):")
        self.chk_rf.setFont(get_label_font(size_pt=8, bold=True))
        self.chk_rf.setStyleSheet(f"color: {COLOR_HUD_ACCENT};")
        self.chk_rf.setChecked(False)
        self.chk_rf.toggled.connect(self._on_rf_changed)
        self.lbl_rf_val = QLabel("σ=80px")
        self.lbl_rf_val.setFont(get_mono_font(size_pt=8))
        self.lbl_rf_val.setStyleSheet(f"color: {COLOR_LOCKED};")
        self.lbl_rf_val.setEnabled(False)
        rf_top.addWidget(self.chk_rf, 1)
        rf_top.addWidget(self.lbl_rf_val)
        isro_layout.addLayout(rf_top)

        self.slider_rf = QSlider(Qt.Horizontal)
        self.slider_rf.setRange(10, 200)
        self.slider_rf.setValue(80)
        self.slider_rf.setEnabled(False)
        self.slider_rf.setFixedHeight(16)
        self.slider_rf.valueChanged.connect(self._on_rf_slider)
        isro_layout.addWidget(self.slider_rf)

        # Pan & Tilt Speeds side-by-side on 1 row
        speed_row = QHBoxLayout()
        speed_row.setSpacing(4)
        spd_lbl = QLabel("Slew Rate:")
        spd_lbl.setFont(get_label_font(size_pt=8))
        spd_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED};")
        spd_lbl.setFixedWidth(72)
        speed_row.addWidget(spd_lbl)

        self.spin_pan_speed = QDoubleSpinBox()
        self.spin_pan_speed.setRange(1.0, 10.0)
        self.spin_pan_speed.setValue(5.0)
        self.spin_pan_speed.setSingleStep(0.5)
        self.spin_pan_speed.setPrefix("Pan ")
        self.spin_pan_speed.setSuffix("°/s")
        self.spin_pan_speed.valueChanged.connect(self.pan_speed_changed.emit)

        self.spin_tilt_speed = QDoubleSpinBox()
        self.spin_tilt_speed.setRange(1.0, 10.0)
        self.spin_tilt_speed.setValue(5.0)
        self.spin_tilt_speed.setSingleStep(0.5)
        self.spin_tilt_speed.setPrefix("Tilt ")
        self.spin_tilt_speed.setSuffix("°/s")
        self.spin_tilt_speed.valueChanged.connect(self.tilt_speed_changed.emit)

        speed_row.addWidget(self.spin_pan_speed, 1)
        speed_row.addWidget(self.spin_tilt_speed, 1)
        isro_layout.addLayout(speed_row)

        # Target Size
        self.spin_size = QSpinBox()
        self.spin_size.setRange(5, 20)
        self.spin_size.setValue(10)
        self.spin_size.setSuffix(" px")
        self.spin_size.valueChanged.connect(self.target_size_changed.emit)
        isro_layout.addLayout(_inline_row("Target Size:", self.spin_size))

        layout.addWidget(isro_frame)

        # ── SECTION 3: ATMOSPHERIC CONDITIONS ────────────────────────────────
        atmo_frame, atmo_layout = _section("ATMOSPHERIC CHANNEL")

        self.combo_atmo = QComboBox()
        self.combo_atmo.addItem("Clear (Nominal Space)",      AtmosphericCondition.CLEAR)
        self.combo_atmo.addItem("Haze (Koschmieder)",         AtmosphericCondition.HAZE)
        self.combo_atmo.addItem("Dense Fog",                  AtmosphericCondition.FOG)
        self.combo_atmo.addItem("Dynamic Rain (Streaks)",     AtmosphericCondition.RAIN)
        self.combo_atmo.addItem("Low-Light (Photon Starved)", AtmosphericCondition.LOW_LIGHT)
        self.combo_atmo.currentIndexChanged.connect(self._on_atmo_changed)
        atmo_layout.addWidget(self.combo_atmo)
        layout.addWidget(atmo_frame)

        # ── SECTION 4: DISTURBANCE INJECTION ─────────────────────────────────
        dist_frame, dist_layout = _section("DISTURBANCE INJECTION")

        # Sky Radiance
        sky_row = QHBoxLayout()
        sky_lbl = QLabel("Sky Radiance:")
        sky_lbl.setFont(get_label_font(size_pt=8, bold=True))
        sky_lbl.setFixedWidth(80)
        self.lbl_sky_val = QLabel(f"{int(self.config.sky_radiance * 100)}% (Night)")
        self.lbl_sky_val.setFont(get_mono_font(size_pt=8))
        sky_row.addWidget(sky_lbl)
        sky_row.addStretch()
        sky_row.addWidget(self.lbl_sky_val)
        dist_layout.addLayout(sky_row)

        self.slider_sky = QSlider(Qt.Horizontal)
        self.slider_sky.setRange(0, 100)
        self.slider_sky.setValue(int(self.config.sky_radiance * 100))
        self.slider_sky.setFixedHeight(16)
        self.slider_sky.valueChanged.connect(self._on_sky_slider)
        dist_layout.addWidget(self.slider_sky)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(2)
        for label, val in [("Night", 0), ("Dusk", 30), ("Overcast", 60), ("Noon", 100)]:
            btn = QPushButton(label)
            btn.setFixedHeight(16)
            btn.setFont(get_mono_font(size_pt=7))
            btn.setStyleSheet(f"background-color: {COLOR_SURFACE_INPUT}; border: 1px solid {COLOR_BORDER}; padding: 0px;")
            btn.clicked.connect(lambda _, v=val, l=label: self._set_sky_preset(v, l))
            preset_row.addWidget(btn)
        dist_layout.addLayout(preset_row)

        # Salt & Pepper
        sp_row = QHBoxLayout()
        self.chk_sp = QCheckBox("S&P Noise:")
        self.chk_sp.setFont(get_label_font(size_pt=8))
        self.chk_sp.setChecked(self.config.enable_salt_pepper)
        self.chk_sp.toggled.connect(self._on_sp_toggled)
        self.lbl_sp_val = QLabel(f"{int(self.config.salt_pepper_ratio * 100)}%")
        self.lbl_sp_val.setFont(get_mono_font(size_pt=8))
        sp_row.addWidget(self.chk_sp, 1)
        sp_row.addWidget(self.lbl_sp_val)
        dist_layout.addLayout(sp_row)
        self.slider_sp = QSlider(Qt.Horizontal)
        self.slider_sp.setRange(0, 30)
        self.slider_sp.setValue(int(self.config.salt_pepper_ratio * 100))
        self.slider_sp.setFixedHeight(16)
        self.slider_sp.valueChanged.connect(self._on_sp_slider)
        dist_layout.addWidget(self.slider_sp)

        # Gaussian Noise
        gauss_row = QHBoxLayout()
        self.chk_gauss = QCheckBox("Gaussian Noise:")
        self.chk_gauss.setFont(get_label_font(size_pt=8))
        self.chk_gauss.setChecked(self.config.enable_gaussian)
        self.chk_gauss.toggled.connect(self._on_gauss_toggled)
        self.lbl_gauss_val = QLabel(f"σ={int(self.config.gaussian_sigma)}")
        self.lbl_gauss_val.setFont(get_mono_font(size_pt=8))
        gauss_row.addWidget(self.chk_gauss, 1)
        gauss_row.addWidget(self.lbl_gauss_val)
        dist_layout.addLayout(gauss_row)
        self.slider_gauss = QSlider(Qt.Horizontal)
        self.slider_gauss.setRange(1, 20)  # ISRO cap σ ≤ 20
        self.slider_gauss.setValue(min(20, int(self.config.gaussian_sigma)))
        self.slider_gauss.setFixedHeight(16)
        self.slider_gauss.valueChanged.connect(self._on_gauss_slider)
        dist_layout.addWidget(self.slider_gauss)

        # Poisson
        self.chk_poisson = QCheckBox("Poisson Shot Noise")
        self.chk_poisson.setFont(get_label_font(size_pt=8))
        self.chk_poisson.setChecked(self.config.enable_poisson)
        self.chk_poisson.toggled.connect(lambda v: setattr(self.config, "enable_poisson", v))
        dist_layout.addWidget(self.chk_poisson)

        # Camera Jitter
        jit_row = QHBoxLayout()
        self.chk_jitter = QCheckBox("Camera Jitter:")
        self.chk_jitter.setFont(get_label_font(size_pt=8))
        self.chk_jitter.setChecked(self.config.enable_camera_jitter)
        self.chk_jitter.toggled.connect(self._on_jitter_toggled)
        self.lbl_jit_val = QLabel(f"±{int(self.config.camera_jitter_max_px)}px")
        self.lbl_jit_val.setFont(get_mono_font(size_pt=8))
        jit_row.addWidget(self.chk_jitter, 1)
        jit_row.addWidget(self.lbl_jit_val)
        dist_layout.addLayout(jit_row)

        jit_mode_row = QHBoxLayout()
        lbl_jmode = QLabel("Jitter Model:")
        lbl_jmode.setFont(get_label_font(size_pt=7))
        lbl_jmode.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        lbl_jmode.setFixedWidth(56)
        self.combo_jitter_mode = QComboBox()
        self.combo_jitter_mode.addItem("Dynamic PSD (Dabiri)", JitterMode.DYNAMIC_PSD)
        self.combo_jitter_mode.addItem("Steady (Fixed ±px)", JitterMode.STEADY)
        self.combo_jitter_mode.setCurrentIndex(0 if self.config.jitter_mode == JitterMode.DYNAMIC_PSD else 1)
        self.combo_jitter_mode.currentIndexChanged.connect(self._on_jitter_mode_changed)
        jit_mode_row.addWidget(lbl_jmode)
        jit_mode_row.addWidget(self.combo_jitter_mode, 1)
        dist_layout.addLayout(jit_mode_row)

        self.slider_jitter = QSlider(Qt.Horizontal)
        self.slider_jitter.setRange(1, 20)  # ISRO cap ≤ ±20px
        self.slider_jitter.setValue(min(20, int(self.config.camera_jitter_max_px)))
        self.slider_jitter.setFixedHeight(16)
        self.slider_jitter.valueChanged.connect(self._on_jitter_slider)
        dist_layout.addWidget(self.slider_jitter)

        # Platform Motion
        plat_row = QHBoxLayout()
        self.chk_plat = QCheckBox("Platform Motion:")
        self.chk_plat.setFont(get_label_font(size_pt=8))
        self.chk_plat.setChecked(self.config.enable_platform_motion)
        self.chk_plat.toggled.connect(self._on_plat_toggled)
        self.lbl_plat_val = QLabel(f"±{int(self.config.platform_motion_max_px)}px")
        self.lbl_plat_val.setFont(get_mono_font(size_pt=8))
        plat_row.addWidget(self.chk_plat, 1)
        plat_row.addWidget(self.lbl_plat_val)
        dist_layout.addLayout(plat_row)
        self.slider_plat = QSlider(Qt.Horizontal)
        self.slider_plat.setRange(1, 20)  # ISRO cap ≤ ±20px
        self.slider_plat.setValue(min(20, int(self.config.platform_motion_max_px)))
        self.slider_plat.setFixedHeight(16)
        self.slider_plat.valueChanged.connect(self._on_plat_slider)
        dist_layout.addWidget(self.slider_plat)

        layout.addWidget(dist_frame)

        # ── SECTION 5: BENCHMARK-2 ────────────────────────────────────────────
        bench_frame = QFrame()
        bench_frame.setObjectName("BenchmarkCard")
        bench_frame.setStyleSheet(f"""
            QFrame#BenchmarkCard {{
                background-color: #0E1520; border: 1px solid #1C3352;
                border-radius: 2px; padding: 2px;
            }}
        """)
        bench_layout = QVBoxLayout(bench_frame)
        bench_layout.setContentsMargins(5, 4, 5, 4)
        bench_layout.setSpacing(3)

        hdr_bench = QLabel("BENCHMARK-2: RAW VIDEO")
        hdr_bench.setObjectName("SectionHeader")
        hdr_bench.setStyleSheet(f"color: {COLOR_HUD_ACCENT}; border-bottom-color: #1C3352;")
        bench_layout.addWidget(hdr_bench)

        self.lbl_mode_status = QLabel("ACTIVE: SIMULATOR (SYNTHETIC)")
        self.lbl_mode_status.setFont(get_mono_font(size_pt=7, bold=True))
        self.lbl_mode_status.setStyleSheet(f"color: {COLOR_LOCKED};")
        bench_layout.addWidget(self.lbl_mode_status)

        self.btn_load_video = QPushButton("LOAD VIDEO FILE (.MP4)...")
        self.btn_load_video.setFixedHeight(22)
        self.btn_load_video.setStyleSheet(f"""
            background-color: #162438; color: {COLOR_HUD_ACCENT}; border: 1px solid #29456B;
        """)
        self.btn_load_video.clicked.connect(self._on_browse_video)
        bench_layout.addWidget(self.btn_load_video)

        self.btn_switch_sim = QPushButton("RETURN TO SIMULATOR")
        self.btn_switch_sim.setFixedHeight(22)
        self.btn_switch_sim.setEnabled(False)
        self.btn_switch_sim.clicked.connect(self._on_return_sim)
        bench_layout.addWidget(self.btn_switch_sim)
        layout.addWidget(bench_frame)

        # ── SECTION 6: TELEMETRY EXPORT ──────────────────────────────────────
        export_frame, export_layout = _section("AUDIT LOG")

        self.btn_export = QPushButton("EXPORT SESSION LOG (CSV/JSON)")
        self.btn_export.setFixedHeight(22)
        self.btn_export.clicked.connect(self.export_logs_clicked.emit)
        export_layout.addWidget(self.btn_export)

        self.lbl_export_status = QLabel("Ready for session export")
        self.lbl_export_status.setFont(get_mono_font(size_pt=7))
        self.lbl_export_status.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        export_layout.addWidget(self.lbl_export_status)

        layout.addWidget(export_frame)

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_motion_changed(self, index: int) -> None:
        model = self.combo_motion.itemData(index)
        if model is not None:
            self.motion_model_changed.emit(model)

    def _on_fov_changed(self) -> None:
        self.fov_changed.emit(self.spin_fov_pan.value(), self.spin_fov_tilt.value())

    def _on_shape_changed(self, index: int) -> None:
        shape = self.combo_shape.itemData(index)
        if shape is not None:
            self.target_shape_changed.emit(shape)

    def _on_initpos_changed(self, *_) -> None:
        mode = self.combo_initpos.currentData()
        is_custom = (mode == "custom")
        self.spin_custom_x.setEnabled(is_custom)
        self.spin_custom_y.setEnabled(is_custom)
        self.initial_pos_changed.emit(mode, self.spin_custom_x.value(), self.spin_custom_y.value())

    def _on_rf_changed(self, checked: bool) -> None:
        self.slider_rf.setEnabled(checked)
        self.lbl_rf_val.setEnabled(checked)
        self.rf_link_changed.emit(checked, float(self.slider_rf.value()))

    def _on_rf_slider(self, val: int) -> None:
        self.lbl_rf_val.setText(f"σ={val}px")
        self.rf_link_changed.emit(self.chk_rf.isChecked(), float(val))

    def _on_atmo_changed(self, index: int) -> None:
        cond = self.combo_atmo.itemData(index)
        if cond is not None:
            self.config.atmospheric_condition = cond

    def _on_sky_slider(self, val: int) -> None:
        self.config.sky_radiance = val / 100.0
        tag = "Night" if val < 20 else ("Dusk" if val < 50 else ("Overcast" if val < 80 else "Noon"))
        self.lbl_sky_val.setText(f"{val}% ({tag})")

    def _set_sky_preset(self, val: int, label: str) -> None:
        self.slider_sky.setValue(val)

    def _on_sp_toggled(self, checked: bool) -> None:
        self.config.enable_salt_pepper = checked

    def _on_sp_slider(self, val: int) -> None:
        self.config.salt_pepper_ratio = val / 100.0
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
            self, "Select Benchmark-2 Video File", default_dir,
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
