"""Telemetry Stat Cards and Live Event Console widget.

Displays large monospace readouts with live compliance indicators against ISRO targets:
- Processing Rate (>= 20 FPS)
- Tracking Error (<= 10.0 px)
- Lock Retention (> 95 %)
- Acquisition Time (<= 2.0 s)
- Re-acquisition Time (<= 1.0 s)
- Loss Event Count
"""

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from fsoc.gui.theme import (
    COLOR_BORDER,
    COLOR_BORDER_LIGHT,
    COLOR_DEGRADED,
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
from fsoc.logger import SystemEvent, TelemetrySnapshot


class StatCard(QFrame):
    """Single telemetry metric card with title, large monospace value, unit, and compliance chip."""

    def __init__(self, title: str, unit: str, spec_label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setStyleSheet(f"""
            QFrame#StatCard {{
                background-color: {COLOR_SURFACE_2};
                border: 1px solid {COLOR_BORDER};
                border-radius: 2px;
                padding: 6px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Header line: Title + Spec
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setFont(get_label_font(size_pt=8, bold=True))
        self.title_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; letter-spacing: 0.5px;")

        self.spec_lbl = QLabel(f"SPEC: {spec_label}")
        self.spec_lbl.setFont(get_mono_font(size_pt=7, bold=False))
        self.spec_lbl.setStyleSheet(f"color: {COLOR_TEXT_DIM};")

        top_layout.addWidget(self.title_lbl)
        top_layout.addStretch()
        top_layout.addWidget(self.spec_lbl)

        # Main numeric line
        num_layout = QHBoxLayout()
        num_layout.setContentsMargins(0, 2, 0, 0)
        num_layout.setSpacing(4)

        self.val_lbl = QLabel("--")
        self.val_lbl.setFont(get_mono_font(size_pt=14, bold=True))
        self.val_lbl.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setFont(get_mono_font(size_pt=9, bold=False))
        self.unit_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; margin-bottom: 2px;")

        self.status_chip = QLabel("NOMINAL")
        self.status_chip.setFont(get_mono_font(size_pt=7, bold=True))
        self.status_chip.setStyleSheet(f"""
            color: {COLOR_LOCKED};
            background-color: #12281E;
            border: 1px solid #1E4632;
            border-radius: 2px;
            padding: 1px 4px;
        """)

        num_layout.addWidget(self.val_lbl)
        num_layout.addWidget(self.unit_lbl)
        num_layout.addStretch()
        num_layout.addWidget(self.status_chip)

        layout.addLayout(top_layout)
        layout.addLayout(num_layout)

    def set_value(self, val_str: str, is_nominal: bool, status_text: str = None) -> None:
        """Update displayed readout and compliance styling."""
        self.val_lbl.setText(val_str)
        if status_text is None:
            status_text = "NOMINAL" if is_nominal else "DEGRADED"

        self.status_chip.setText(status_text)
        if is_nominal:
            self.status_chip.setStyleSheet(f"""
                color: {COLOR_LOCKED};
                background-color: #12281E;
                border: 1px solid #1E4632;
                border-radius: 2px;
                padding: 1px 4px;
            """)
        else:
            self.status_chip.setStyleSheet(f"""
                color: {COLOR_DEGRADED};
                background-color: #2B2313;
                border: 1px solid #4D3B18;
                border-radius: 2px;
                padding: 1px 4px;
            """)


class StatCardsWidget(QWidget):
    """Panel containing the performance metric cards and scrolling system event log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatCardsPanel")
        self.setFixedWidth(290)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Section Header
        hdr = QLabel("TELEMETRY & PERFORMANCE AUDIT")
        hdr.setObjectName("SectionHeader")
        layout.addWidget(hdr)

        # 1. Processing FPS Card (Target >= 20 FPS)
        self.card_fps = StatCard("Loop Rate", "Hz", "≥20 Hz")
        layout.addWidget(self.card_fps)

        # 2. Tracking Error Card (Target <= 10.0 px)
        self.card_err = StatCard("Tracking Error", "px", "≤10 px")
        layout.addWidget(self.card_err)

        # 3. Lock Retention Card (Target > 95 %)
        self.card_lock = StatCard("Lock Retention", "%", ">95%")
        layout.addWidget(self.card_lock)

        # 4. Acquisition Time Card (Target <= 2.0 s)
        self.card_acq = StatCard("Acquisition Time", "s", "≤2.0 s")
        layout.addWidget(self.card_acq)

        # 5. Re-acquisition Time Card (Target <= 1.0 s)
        self.card_reacq = StatCard("Re-Acquisition", "s", "≤1.0 s")
        layout.addWidget(self.card_reacq)

        # 6. Target Loss Event Count
        self.card_loss = StatCard("Loss Events", "cnt", "<5%")
        layout.addWidget(self.card_loss)

        # Section Header for Event Log
        log_hdr = QLabel("CHRONOLOGICAL EVENT LOG")
        log_hdr.setObjectName("SectionHeader")
        layout.addWidget(log_hdr)

        # System Event Log Console (Read-only monospace text area)
        self.event_console = QPlainTextEdit()
        self.event_console.setReadOnly(True)
        self.event_console.setFont(get_mono_font(size_pt=8, bold=False))
        self.event_console.setStyleSheet(f"""
            background-color: {COLOR_SURFACE_INPUT};
            color: #9BA7B6;
            border: 1px solid {COLOR_BORDER};
            border-radius: 2px;
            padding: 4px;
        """)
        self.event_console.setMaximumHeight(140)
        layout.addWidget(self.event_console)

        self._last_event_count: int = 0

    def update_telemetry(self, snap: TelemetrySnapshot, events: List[SystemEvent]) -> None:
        """Update all telemetry cards and append new system events."""
        # 1. FPS
        fps_nominal = snap.fps >= 20.0
        self.card_fps.set_value(f"{snap.fps:.1f}", fps_nominal)

        # 2. Tracking Error (px)
        err_nominal = snap.tracking_error_px <= 10.0
        self.card_err.set_value(f"{snap.tracking_error_px:.1f}", err_nominal)

        # 3. Lock Retention (%)
        lock_nominal = snap.lock_retention_pct >= 90.0
        self.card_lock.set_value(f"{snap.lock_retention_pct:.1f}", lock_nominal)

        # 4. Acquisition Time (s)
        acq_nominal = snap.acquisition_time_s <= 2.0 or snap.acquisition_time_s == 0.0
        acq_str = f"{snap.acquisition_time_s:.2f}" if snap.acquisition_time_s > 0 else "--"
        self.card_acq.set_value(acq_str, acq_nominal)

        # 5. Re-Acquisition Time (s)
        reacq_nominal = snap.reacquisition_time_s <= 1.0 or snap.reacquisition_time_s == 0.0
        reacq_str = f"{snap.reacquisition_time_s:.2f}" if snap.reacquisition_time_s > 0 else "--"
        self.card_reacq.set_value(reacq_str, reacq_nominal)

        # 6. Loss Events
        loss_nominal = snap.target_loss_count <= 2
        self.card_loss.set_value(str(snap.target_loss_count), loss_nominal)

        # Update Event Console if new events arrived
        if len(events) != self._last_event_count:
            self._last_event_count = len(events)
            lines = [f"[{e.timestamp_str}] {e.event_type:<10} {e.message}" for e in events]
            self.event_console.setPlainText("\n".join(lines))
            self.event_console.verticalScrollBar().setValue(
                self.event_console.verticalScrollBar().maximum()
            )
