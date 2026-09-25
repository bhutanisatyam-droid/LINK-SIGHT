"""Background Tracking Thread worker for decoupled ATP control loop.

Runs at >= 30Hz decoupled from PySide6 GUI paint/event loop.
A slow Qt paint event or UI interaction will never block the tracking loop.
"""

import threading
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from fsoc.engine import EngineOutput, TrackingPipeline
from fsoc.frame_source import MotionModel


class TrackingThread(QThread):
    """Dedicated background worker running the real-time tracking pipeline."""

    # Signal emitted when a new frame is processed
    frame_ready = Signal(object) # Carries EngineOutput
    status_changed = Signal(str)
    fps_updated = Signal(float)

    def __init__(self, target_rate_hz: float = 35.0, parent=None):
        super().__init__(parent)
        self.target_rate = target_rate_hz
        self.pipeline = TrackingPipeline()

        self._running = False
        self._paused = False
        self._mutex = threading.Lock()

        # Commanded pending configuration changes
        self._pending_video_path: Optional[str] = None
        self._pending_switch_sim: bool = False
        self._pending_motion_model: Optional[MotionModel] = None
        self._pending_target_size: Optional[int] = None
        self._pending_ptz_speed: Optional[float] = None
        self._pending_reset: bool = False
        self._pending_occlusion_frames: Optional[int] = None

    def run(self) -> None:
        """Main real-time tracking thread loop."""
        self._running = True
        frame_interval = 1.0 / self.target_rate

        while self._running:
            t_loop_start = time.perf_counter()

            # Handle queued commands under lock
            with self._mutex:
                if self._pending_reset:
                    self.pipeline.reset()
                    self._pending_reset = False

                if self._pending_occlusion_frames is not None:
                    self.pipeline.trigger_occlusion(self._pending_occlusion_frames)
                    self._pending_occlusion_frames = None

                if self._pending_video_path is not None:
                    self.pipeline.load_video_source(self._pending_video_path)
                    self._pending_video_path = None

                if self._pending_switch_sim:
                    self.pipeline.load_simulator_source()
                    self._pending_switch_sim = False

                if self._pending_motion_model is not None:
                    self.pipeline.set_motion_model(self._pending_motion_model)
                    self._pending_motion_model = None

                if self._pending_target_size is not None:
                    self.pipeline.set_target_size(self._pending_target_size)
                    self._pending_target_size = None

                if self._pending_ptz_speed is not None:
                    self.pipeline.set_max_ptz_speed(self._pending_ptz_speed)
                    self._pending_ptz_speed = None

                is_paused = self._paused

            if not is_paused:
                try:
                    output: EngineOutput = self.pipeline.process_frame()
                    self.frame_ready.emit(output)
                except Exception as e:
                    print(f"[TrackingThread Error]: {e}")

            # Precise timing regulation for loop rate
            elapsed = time.perf_counter() - t_loop_start
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def stop(self) -> None:
        """Gracefully terminate thread."""
        self._running = False
        self.wait(2000)

    def set_paused(self, paused: bool) -> None:
        """Pause or unpause tracking loop."""
        with self._mutex:
            self._paused = paused

    def is_paused(self) -> bool:
        with self._mutex:
            return self._paused

    def request_reset(self) -> None:
        """Queue a pipeline reset."""
        with self._mutex:
            self._pending_reset = True

    def request_occlusion(self, duration_frames: int = 35) -> None:
        """Queue a simulated beam break / occlusion."""
        with self._mutex:
            self._pending_occlusion_frames = duration_frames

    def request_motion_model(self, model: MotionModel) -> None:
        """Queue motion model switch."""
        with self._mutex:
            self._pending_motion_model = model

    def request_target_size(self, size_px: int) -> None:
        """Queue beacon size update."""
        with self._mutex:
            self._pending_target_size = size_px

    def request_ptz_speed(self, speed_deg_s: float) -> None:
        """Queue max gimbal speed update."""
        with self._mutex:
            self._pending_ptz_speed = speed_deg_s

    def request_load_video(self, video_path: str) -> None:
        """Queue switch to video file benchmark."""
        with self._mutex:
            self._pending_video_path = video_path

    def request_switch_simulator(self) -> None:
        """Queue switch back to simulator."""
        with self._mutex:
            self._pending_switch_sim = True
