"""Telemetry Logger & Performance Engine for FSOC tracking system.

Acts as the central event bus and telemetry aggregator:
- Evaluates live performance against ISRO targets:
  * Acquisition <= 2.0 s
  * Tracking error <= 10.0 px
  * Target loss < 5.0 %
  * Re-acquisition <= 1.0 s
  * Processing >= 20.0 FPS
- Maintains rolling history for the live tracking error chart
- Provides thread-safe snapshot queries for the GUI
- Supports on-demand CSV and JSON export of complete flight/bench telemetry
"""

from collections import deque
from dataclasses import asdict, dataclass, field
import datetime
import json
import math
import os
import threading
import time
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class TelemetrySnapshot:
    """Immutable state snapshot consumed by the GUI and logging exports."""
    timestamp: float = 0.0
    frame_index: int = 0
    fps: float = 0.0
    tracking_error_px: float = 0.0
    mean_error_px: float = 0.0
    rms_error_px: float = 0.0
    max_error_px: float = 0.0

    acquisition_time_s: float = 0.0
    reacquisition_time_s: float = 0.0
    lock_retention_pct: float = 100.0
    target_loss_count: int = 0

    status: str = "INITIALIZING"
    confidence: float = 0.0
    pan_deg: float = 0.0
    tilt_deg: float = 0.0

    is_locked: bool = False
    is_searching: bool = False
    is_video_mode: bool = False


@dataclass
class LogRecord:
    """Row in persistent time-series audit log."""
    time_s: float
    frame: int
    fps: float
    error_px: float
    status: str
    confidence: float
    pan_deg: float
    tilt_deg: float
    loss_count: int
    lock_retention: float


@dataclass
class SystemEvent:
    """Discrete system event (e.g. state transition, parameter change)."""
    timestamp_str: str
    event_type: str
    message: str


class TelemetryLogger:
    """Central telemetry collector and performance metrics computer."""

    def __init__(self, rolling_window_seconds: float = 15.0):
        self.lock = threading.RLock()
        self.rolling_sec = rolling_window_seconds

        # FPS calculation state
        self._fps_timestamps: Deque[float] = deque(maxlen=40)
        self.current_fps: float = 0.0

        # Run lifecycle timers
        self.start_time: float = time.perf_counter()
        self.first_lock_time: Optional[float] = None
        self.acquisition_duration_s: float = 0.0

        # Lock retention statistics
        self.total_frames: int = 0
        self.locked_frames: int = 0

        # Loss & Re-acquisition tracking
        self.loss_event_count: int = 0
        self.loss_start_time: Optional[float] = None
        self.last_reacquisition_duration_s: float = 0.0
        self._consecutive_locked: int = 0

        # Tracking error statistics
        self._error_history: Deque[float] = deque(maxlen=2000)

        # Rolling buffer for real-time chart: (relative_time_s, error_px, status_code)
        # status_code: 0 = LOCKED, 1 = DEGRADED, 2 = LOST/SEARCHING
        self.chart_buffer: Deque[Tuple[float, float, int]] = deque(maxlen=600)

        # Complete session logs for disk export
        self.records: List[LogRecord] = []
        self.events: List[SystemEvent] = []

        # Current cached snapshot
        self._snapshot = TelemetrySnapshot()

        self.log_event("SYSTEM_INIT", "FSOC Tracking Telemetry Engine online")

    def reset(self) -> None:
        """Reset run session statistics."""
        with self.lock:
            self.start_time = time.perf_counter()
            self.first_lock_time = None
            self.acquisition_duration_s = 0.0
            self.total_frames = 0
            self.locked_frames = 0
            self.loss_event_count = 0
            self.loss_start_time = None
            self.last_reacquisition_duration_s = 0.0
            self._consecutive_locked = 0
            self._error_history.clear()
            self.chart_buffer.clear()
            self.records.clear()
            self.events.clear()
            self._fps_timestamps.clear()
            self._snapshot = TelemetrySnapshot()
            self.log_event("SESSION_RESET", "Telemetry session counters cleared")

    def log_event(self, event_type: str, message: str) -> None:
        """Log a notable system event with high-precision timestamp."""
        now_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        event = SystemEvent(timestamp_str=now_str, event_type=event_type, message=message)
        with self.lock:
            self.events.append(event)
            if len(self.events) > 300:
                self.events.pop(0)

    def record_step(
        self,
        frame_idx: int,
        tracking_error_px: float,
        status: str,
        confidence: float,
        pan_deg: float,
        tilt_deg: float,
        is_searching: bool,
        is_video_mode: bool,
    ) -> None:
        """Process tracking loop iteration and update metrics."""
        now = time.perf_counter()

        with self.lock:
            # 1. Update FPS
            self._fps_timestamps.append(now)
            if len(self._fps_timestamps) >= 2:
                time_span = self._fps_timestamps[-1] - self._fps_timestamps[0]
                if time_span > 0:
                    self.current_fps = (len(self._fps_timestamps) - 1) / time_span

            self.total_frames += 1
            rel_time = now - self.start_time

            # Status classification
            # ISRO Locked tracking specification: status == "TRACKING" with tracking error <= 15.0px
            is_locked = (status == "TRACKING") and (tracking_error_px <= 15.0)

            if is_locked:
                self._consecutive_locked += 1
                self.locked_frames += 1
                if self.first_lock_time is None and self._consecutive_locked >= 2:
                    self.first_lock_time = now
                    self.acquisition_duration_s = now - self.start_time
                    self.log_event(
                        "ACQUIRED",
                        f"Initial lock in {self.acquisition_duration_s:.2f}s (Error: {tracking_error_px:.1f}px)"
                    )

                # Re-acquisition completed only when target is centered within ISRO spec for >= 2 frames
                if self.loss_start_time is not None and self._consecutive_locked >= 2:
                    reacq_time = now - self.loss_start_time
                    self.last_reacquisition_duration_s = reacq_time
                    self.loss_start_time = None
                    self.log_event(
                        "REACQUIRED",
                        f"Target re-locked in {reacq_time:.2f}s (Error: {tracking_error_px:.1f}px)"
                    )

            else:
                self._consecutive_locked = 0
                if status in ("LOST", "SEARCHING", "DEGRADED") or tracking_error_px > 25.0:
                    if self.loss_start_time is None:
                        self.loss_start_time = now
                        self.loss_event_count += 1
                        self.log_event("TARGET_LOST", f"Track lost (Event #{self.loss_event_count})")

            # Lock retention rate
            lock_retention = (
                (self.locked_frames / self.total_frames * 100.0)
                if self.total_frames > 0 else 100.0
            )

            # Error metrics calculation
            self._error_history.append(tracking_error_px)
            mean_err = float(np.mean(self._error_history)) if self._error_history else 0.0
            rms_err = float(np.sqrt(np.mean(np.square(self._error_history)))) if self._error_history else 0.0
            max_err = float(np.max(self._error_history)) if self._error_history else 0.0

            # Chart buffer encoding: 0 = TRACKING, 1 = DEGRADED, 2 = LOST/SEARCHING
            if status == "TRACKING":
                s_code = 0
            elif status == "DEGRADED":
                s_code = 1
            else:
                s_code = 2
            self.chart_buffer.append((rel_time, tracking_error_px, s_code))

            # Store in session log record
            record = LogRecord(
                time_s=round(rel_time, 3),
                frame=frame_idx,
                fps=round(self.current_fps, 1),
                error_px=round(tracking_error_px, 2),
                status=status,
                confidence=round(confidence, 3),
                pan_deg=round(pan_deg, 3),
                tilt_deg=round(tilt_deg, 3),
                loss_count=self.loss_event_count,
                lock_retention=round(lock_retention, 2),
            )
            self.records.append(record)

            # Update cached snapshot
            self._snapshot = TelemetrySnapshot(
                timestamp=now,
                frame_index=frame_idx,
                fps=self.current_fps,
                tracking_error_px=tracking_error_px,
                mean_error_px=mean_err,
                rms_error_px=rms_err,
                max_error_px=max_err,
                acquisition_time_s=self.acquisition_duration_s,
                reacquisition_time_s=self.last_reacquisition_duration_s,
                lock_retention_pct=lock_retention,
                target_loss_count=self.loss_event_count,
                status=status,
                confidence=confidence,
                pan_deg=pan_deg,
                tilt_deg=tilt_deg,
                is_locked=is_locked,
                is_searching=is_searching,
                is_video_mode=is_video_mode,
            )

    def get_snapshot(self) -> TelemetrySnapshot:
        """Thread-safe query of the current telemetry state."""
        with self.lock:
            return self._snapshot

    def get_chart_data(self) -> List[Tuple[float, float, int]]:
        """Return rolling chart points for live GUI rendering."""
        with self.lock:
            return list(self.chart_buffer)

    def get_recent_events(self, count: int = 20) -> List[SystemEvent]:
        """Return the most recent system events for the log console."""
        with self.lock:
            return list(self.events[-count:])

    def export_csv(self, filepath: str) -> bool:
        """Export session time-series records to CSV format."""
        with self.lock:
            records = list(self.records)

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                # Header
                f.write("time_s,frame,fps,error_px,status,confidence,pan_deg,tilt_deg,loss_count,lock_retention_pct\n")
                for r in records:
                    f.write(
                        f"{r.time_s:.3f},{r.frame},{r.fps:.1f},{r.error_px:.2f},{r.status},"
                        f"{r.confidence:.3f},{r.pan_deg:.3f},{r.tilt_deg:.3f},{r.loss_count},{r.lock_retention:.2f}\n"
                    )
            return True
        except Exception as e:
            self.log_event("EXPORT_ERROR", f"Failed to export CSV: {e}")
            return False

    def export_json(self, filepath: str) -> bool:
        """Export full session metadata and telemetry points to JSON."""
        with self.lock:
            snap = asdict(self._snapshot)
            records = [asdict(r) for r in self.records]
            events = [asdict(e) for e in self.events]

        payload = {
            "metadata": {
                "system": "ISRO FSOC Virtual Tracking System (Coarse ATP Stage)",
                "export_time": datetime.datetime.now().isoformat(),
                "summary": snap,
            },
            "events": events,
            "telemetry_points": records,
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception as e:
            self.log_event("EXPORT_ERROR", f"Failed to export JSON: {e}")
            return False
