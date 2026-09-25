"""Main Telemetry Window for FSOC Coarse-Alignment Virtual Tracking System.

Assembles:
- Top Header: Mission clock, Loop rate monitor, System identity
- Left Panel: Motion, Disturbance, Atmospheric, PTZ parameters, Benchmark-2 Video mode
- Center Panel: Primary HUD Video Display + Real-time Rolling Tracking Error Chart
- Right Panel: Top-Down 2D Scene Radar Minimap + Performance Metric Cards & Event Console
"""

import datetime
import os

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fsoc.engine import EngineOutput
from fsoc.frame_source import MotionModel
from fsoc.gui.theme import (
    COLOR_BG_DARK,
    COLOR_BORDER,
    COLOR_BORDER_LIGHT,
    COLOR_DEGRADED,
    COLOR_HUD_ACCENT,
    COLOR_LOCKED,
    COLOR_LOST,
    COLOR_SURFACE_1,
    COLOR_SURFACE_2,
    COLOR_TEXT_DIM,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    MISSION_CONTROL_STYLESHEET,
    get_label_font,
    get_mono_font,
)
from fsoc.gui.widgets.control_panel import ControlPanelWidget
from fsoc.gui.widgets.error_chart import TrackingErrorChartWidget
from fsoc.gui.widgets.minimap import TopDownMinimapWidget
from fsoc.gui.widgets.stat_cards import StatCardsWidget
from fsoc.gui.widgets.video_display import VideoDisplayWidget
from fsoc.gui.worker import TrackingThread


class MainWindow(QMainWindow):
    """Primary telemetry and mission-control console window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "ISRO / DOS ATP COARSE-ALIGNMENT VIRTUAL TRACKING SYSTEM — RESEARCH DEMO [SIH-2026]"
        )
        self.resize(1380, 880)
        self.setMinimumSize(1100, 720)

        # Apply global mission-control dark stylesheet
        self.setStyleSheet(MISSION_CONTROL_STYLESHEET)

        # Instantiate tracking worker thread
        self.worker = TrackingThread(target_rate_hz=35.0)

        # Build UI layout
        self._init_ui()

        # Connect signals
        self._connect_signals()

        # Start tracking thread
        self.worker.start()

        # UI heartbeat timer for clock
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(500)

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        main_vbox.setContentsMargins(6, 6, 6, 6)
        main_vbox.setSpacing(6)

        # --- 1. TOP MISSION HEADER BAR ---
        header_frame = QFrame()
        header_frame.setObjectName("PanelFrame")
        header_frame.setFixedHeight(38)
        hdr_layout = QHBoxLayout(header_frame)
        hdr_layout.setContentsMargins(12, 0, 12, 0)
        hdr_layout.setSpacing(16)

        # Title / Subsystem Tag
        title_lbl = QLabel("ISRO-DOS // FSOC ATP COARSE TRACKING TERMINAL")
        title_lbl.setObjectName("HeaderTitle")
        hdr_layout.addWidget(title_lbl)

        # Sub-stage chip
        stage_chip = QLabel("STAGE-1: COARSE GIMBAL POINTING (PRE-FSM)")
        stage_chip.setFont(get_mono_font(size_pt=8, bold=True))
        stage_chip.setStyleSheet(f"""
            color: {COLOR_HUD_ACCENT};
            background-color: #121F2D;
            border: 1px solid #1D3550;
            border-radius: 2px;
            padding: 2px 6px;
        """)
        hdr_layout.addWidget(stage_chip)

        hdr_layout.addStretch()

        # Telemetry Rate Monitor
        self.lbl_rate_monitor = QLabel("RATE: 35.0 Hz")
        self.lbl_rate_monitor.setFont(get_mono_font(size_pt=9, bold=True))
        self.lbl_rate_monitor.setStyleSheet(f"color: {COLOR_LOCKED};")
        hdr_layout.addWidget(self.lbl_rate_monitor)

        # UTC Clock
        self.lbl_clock = QLabel("UTC: --:--:--")
        self.lbl_clock.setFont(get_mono_font(size_pt=9, bold=False))
        self.lbl_clock.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        hdr_layout.addWidget(self.lbl_clock)

        main_vbox.addWidget(header_frame)

        # --- 2. THREE-COLUMN DATA-DENSE SPLITTER ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(6)
        body_layout.setContentsMargins(0, 0, 0, 0)

        # Left Column: Control Panel
        self.control_panel = ControlPanelWidget(
            disturbance_config=self.worker.pipeline.disturbance_injector.config
        )
        body_layout.addWidget(self.control_panel)

        # Center Column: Main HUD Video Display + Rolling Error Chart
        center_frame = QFrame()
        center_frame.setObjectName("PanelFrame")
        center_vbox = QVBoxLayout(center_frame)
        center_vbox.setContentsMargins(4, 4, 4, 4)
        center_vbox.setSpacing(4)

        # HUD Video Viewport
        self.video_display = VideoDisplayWidget()
        center_vbox.addWidget(self.video_display, stretch=5)

        # Real-time Rolling Error Chart
        self.error_chart = TrackingErrorChartWidget(time_window_sec=15.0)
        center_vbox.addWidget(self.error_chart, stretch=2)

        body_layout.addWidget(center_frame, stretch=1)

        # Right Column: 2D Minimap Radar + Stat Cards & Event Log
        right_frame = QFrame()
        right_frame.setObjectName("PanelFrame")
        right_frame.setFixedWidth(295)
        right_vbox = QVBoxLayout(right_frame)
        right_vbox.setContentsMargins(4, 4, 4, 4)
        right_vbox.setSpacing(6)

        # Top-Down 2D Radar Minimap
        self.minimap = TopDownMinimapWidget()
        right_vbox.addWidget(self.minimap, stretch=1)

        # Live Stat Cards & System Event Log
        self.stat_cards = StatCardsWidget()
        right_vbox.addWidget(self.stat_cards, stretch=2)

        body_layout.addWidget(right_frame)
        main_vbox.addLayout(body_layout)

        # --- 3. BOTTOM AUDIT FOOTER ---
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(8, 0, 8, 2)

        foot_left = QLabel(
            "COMPLIANCE TARGETS: Acq ≤2.0s | Error ≤10.0px | Loss <5% | Re-acq ≤1.0s | Rate ≥20 Hz"
        )
        foot_left.setFont(get_mono_font(size_pt=8))
        foot_left.setStyleSheet(f"color: {COLOR_TEXT_DIM};")
        footer_layout.addWidget(foot_left)

        footer_layout.addStretch()

        foot_right = QLabel("ARCH: DECOUPLED QTHREAD PIPELINE // STRICT GROUND-TRUTH ISOLATION")
        foot_right.setFont(get_mono_font(size_pt=8))
        foot_right.setStyleSheet(f"color: {COLOR_HUD_ACCENT};")
        footer_layout.addWidget(foot_right)

        main_vbox.addLayout(footer_layout)

    def _connect_signals(self) -> None:
        """Connect UI interactions and worker data streams."""
        # Worker frame output stream
        self.worker.frame_ready.connect(self._on_frame_ready)

        # Execution controls
        self.control_panel.run_clicked.connect(lambda: self.worker.set_paused(False))
        self.control_panel.pause_clicked.connect(lambda: self.worker.set_paused(True))
        self.control_panel.reset_clicked.connect(self.worker.request_reset)
        self.control_panel.occlude_clicked.connect(lambda: self.worker.request_occlusion(35))

        # Parameter controls
        self.control_panel.motion_model_changed.connect(self.worker.request_motion_model)
        self.control_panel.target_size_changed.connect(self.worker.request_target_size)
        self.control_panel.ptz_speed_changed.connect(self.worker.request_ptz_speed)

        # Benchmark-2 Video source loading
        self.control_panel.video_file_selected.connect(self.worker.request_load_video)
        self.control_panel.switch_to_sim_clicked.connect(self.worker.request_switch_simulator)

        # Session Log Export
        self.control_panel.export_logs_clicked.connect(self._export_session_logs)

    def _on_frame_ready(self, output: EngineOutput) -> None:
        """Handle incoming frame telemetry snapshot."""
        # 1. Update live HUD video
        self.video_display.update_frame(output)

        # 2. Update Minimap radar (or grey out in video mode)
        self.minimap.update_ground_truth(output.ground_truth)

        # 3. Update Stat Cards & Event Log
        events = self.worker.pipeline.logger.get_recent_events(count=25)
        self.stat_cards.update_telemetry(output.telemetry, events)

        # 4. Update Rolling Error Chart
        chart_data = self.worker.pipeline.logger.get_chart_data()
        self.error_chart.update_data(chart_data)

        # 5. Header loop rate monitor
        fps_val = output.telemetry.fps
        fps_color = COLOR_LOCKED if fps_val >= 20.0 else COLOR_DEGRADED
        self.lbl_rate_monitor.setText(f"LOOP: {fps_val:.1f} Hz")
        self.lbl_rate_monitor.setStyleSheet(f"color: {fps_color};")

    def _update_clock(self) -> None:
        """Update UTC header timestamp."""
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S UTC")
        self.lbl_clock.setText(now_utc)

    def _export_session_logs(self) -> None:
        """Trigger CSV and JSON log export to disk."""
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = os.path.abspath("logs")
        os.makedirs(logs_dir, exist_ok=True)

        csv_path = os.path.join(logs_dir, f"telemetry_{now_str}.csv")
        json_path = os.path.join(logs_dir, f"telemetry_{now_str}.json")

        csv_ok = self.worker.pipeline.logger.export_csv(csv_path)
        json_ok = self.worker.pipeline.logger.export_json(json_path)

        if csv_ok and json_ok:
            self.control_panel.set_export_status(
                f"Exported: telemetry_{now_str}.csv / .json", is_success=True
            )
            QMessageBox.information(
                self,
                "Telemetry Export Completed",
                f"Session performance logs successfully exported to:\n\n"
                f"• {csv_path}\n"
                f"• {json_path}\n\n"
                f"Total Records: {len(self.worker.pipeline.logger.records)}\n"
                f"System Events: {len(self.worker.pipeline.logger.events)}",
            )
        else:
            self.control_panel.set_export_status("Export failed", is_success=False)

    def closeEvent(self, event) -> None:
        """Ensure clean shutdown of background tracking thread."""
        self.worker.stop()
        event.accept()
