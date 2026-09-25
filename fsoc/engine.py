"""Tracking Pipeline Engine for FSOC coarse-pointing system.

Orchestrates the frame source, disturbance injector, classical detector,
Kalman filter state estimator, cut hexagonal spiral re-acquisition engine,
PID gimbal rate controller, and central telemetry logger.

Enforces strict architectural boundaries:
- Detector, Tracker, and Controller receive ONLY camera frame data.
- Ground truth is isolated and routed directly to minimap telemetry.
"""

from dataclasses import dataclass
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from fsoc.controller import PTZPIDController
from fsoc.detector import (
    BaseDetector,
    ClassicalBeaconDetector,
    DetectionResult,
    SpatiotemporalBeaconDetector,
)
from fsoc.disturbance import DisturbanceConfig, DisturbanceInjector
from fsoc.frame_source import BaseFrameSource, GroundTruthState, MotionModel, SimulatorFrameSource, TargetShape, VideoFileFrameSource
from fsoc.logger import TelemetryLogger, TelemetrySnapshot
from fsoc.reacquisition import CutHexagonalSpiralSearch
from fsoc.tracker import BaseTracker, KalmanBeaconTracker, TrackEstimate, TrackStatus


@dataclass
class EngineOutput:
    """Consolidated state emitted by the tracking pipeline for every processed frame."""
    frame: np.ndarray                          # 640x480 uint8 camera frame
    detection: DetectionResult                 # Raw optical detection result
    track: TrackEstimate                       # Kalman filtered state estimate
    is_searching: bool                         # Whether cut hexagonal spiral search is active
    search_waypoints: List[Tuple[float, float]]# Waypoint list during search
    active_waypoint_idx: int                   # Index of active search point
    error_vector_px: Tuple[float, float]       # (dx, dy) from center (320, 240)
    tracking_error_px: float                   # Euclidean norm in pixels
    status_label: str                          # ACQUIRING / TRACKING / DEGRADED / LOST
    telemetry: TelemetrySnapshot               # Performance statistics snapshot
    ground_truth: Optional[GroundTruthState]   # Isolated ground truth (None in video mode)
    is_ai_dampener_active: bool = False        # Core 3 DDPG Dampener wake status


class TrackingPipeline:
    """Master ATP coarse tracking pipeline."""

    def __init__(
        self,
        frame_source: Optional[BaseFrameSource] = None,
        detector: Optional[BaseDetector] = None,
        tracker: Optional[BaseTracker] = None,
        controller: Optional[PTZPIDController] = None,
        logger: Optional[TelemetryLogger] = None,
    ):
        # Instantiate default modular components if not injected
        self.disturbance_injector = DisturbanceInjector()
        self.frame_source = frame_source or SimulatorFrameSource()

        # Wire disturbance injector into simulator source
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.disturbance_injector = self.disturbance_injector

        self.detector = detector or ClassicalBeaconDetector()
        self.tracker = tracker or KalmanBeaconTracker()
        self.reacquisition = CutHexagonalSpiralSearch()
        self.controller = controller or PTZPIDController()
        self.logger = logger or TelemetryLogger()

        self.frame_index: int = 0
        self.forced_occlusion_frames: int = 0
        self.last_step_time: float = time.perf_counter()
        self.reticle_center: Tuple[float, float] = (320.0, 240.0)
        self._lost_frames: int = 0   # Counts consecutive LOST frames for hard-reset watchdog
        self._unacquired_frames: int = 0  # Counts initial frames without lock before triggering search
        self._last_target_pan_deg: float = 0.0
        self._last_target_tilt_deg: float = 0.0

    def set_motion_model(self, model: MotionModel) -> None:
        """Change trajectory of simulated beacon."""
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_motion_model(model)
            self.logger.log_event("CONFIG_CHANGE", f"Beacon motion model set to {model.value.upper()}")

    def set_target_size(self, size_px: int) -> None:
        """Update optical beacon size in pixels (5-20px)."""
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_target_size(size_px)
            self.logger.log_event("CONFIG_CHANGE", f"Beacon aperture size set to {size_px}px")

    def set_fov(self, fov_pan_deg: float, fov_tilt_deg: float) -> None:
        """Update virtual camera field of view (ISRO Parameter #4)."""
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_fov(fov_pan_deg, fov_tilt_deg)
            self.logger.log_event(
                "CONFIG_CHANGE",
                f"Camera FOV set to {fov_pan_deg:.1f}° pan × {fov_tilt_deg:.1f}° tilt"
            )

    def set_target_shape(self, shape: TargetShape) -> None:
        """Update rendered beacon spot shape (ISRO Parameter #9)."""
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_target_shape(shape)
            self.logger.log_event("CONFIG_CHANGE", f"Target shape set to {shape.value.upper()}")

    def set_initial_position(self, mode: str, custom_x: float = 1000.0, custom_y: float = 1000.0) -> None:
        """Set spawn location mode ('random' or 'custom') (ISRO Parameter #11)."""
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_initial_pos_mode(mode, custom_x, custom_y)
            self.logger.log_event(
                "CONFIG_CHANGE",
                f"Initial target position: {mode.upper()}"
                + (f" ({custom_x:.0f}, {custom_y:.0f})" if mode == "custom" else "")
            )

    def set_pan_speed(self, speed_deg_s: float) -> None:
        """Update Pan axis slew rate (ISRO Parameter #13, range 1–10 °/s)."""
        speed_deg_s = max(1.0, min(10.0, speed_deg_s))
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_max_pan_speed(speed_deg_s)
        self.controller.set_max_speed(speed_deg_s)
        self.logger.log_event("CONFIG_CHANGE", f"Pan slew speed set to {speed_deg_s:.1f} deg/s")

    def set_tilt_speed(self, speed_deg_s: float) -> None:
        """Update Tilt axis slew rate (ISRO Parameter #14, range 1–10 °/s)."""
        speed_deg_s = max(1.0, min(10.0, speed_deg_s))
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_max_tilt_speed(speed_deg_s)
        self.logger.log_event("CONFIG_CHANGE", f"Tilt slew speed set to {speed_deg_s:.1f} deg/s")

    def set_max_ptz_speed(self, speed_deg_s: float) -> None:
        """Update gimbal maximum slew rate limit (unified pan + tilt)."""
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_max_ptz_speed(speed_deg_s)
        self.controller.set_max_speed(speed_deg_s)
        self.logger.log_event("CONFIG_CHANGE", f"Max PTZ slew speed set to {speed_deg_s:.1f} deg/s")

    def set_disturbance_bounds(self, gaussian_sigma: float, jitter_max_px: float, platform_max_px: float) -> None:
        """Clamp disturbance parameters to ISRO PS-26169 hard limits (σ≤20, ±20px)."""
        self.disturbance_injector.config.gaussian_sigma = max(0.0, min(20.0, gaussian_sigma))
        self.disturbance_injector.config.camera_jitter_max_px = max(0.0, min(20.0, jitter_max_px))
        self.disturbance_injector.config.platform_motion_max_px = max(0.0, min(20.0, platform_max_px))
        self.logger.log_event(
            "CONFIG_CHANGE",
            f"Disturbance bounds: Gauss σ={gaussian_sigma:.0f}, Jitter ±{jitter_max_px:.0f}px, Platform ±{platform_max_px:.0f}px"
        )

    def set_rf_link(self, active: bool, uncertainty_px: float) -> None:
        """Configure RF side-link simulation (ISRO coarse handoff model)."""
        if isinstance(self.frame_source, SimulatorFrameSource):
            self.frame_source.set_rf_link(active, uncertainty_px)
            status = f"ACTIVE (σ={uncertainty_px:.0f}px)" if active else "DISABLED"
            self.logger.log_event("CONFIG_CHANGE", f"RF Side-Link: {status}")


    def load_video_source(self, video_path: str) -> bool:
        """Switch to VideoFileFrameSource (Benchmark-2 mode on un-simulated footage)."""
        try:
            video_src = VideoFileFrameSource(video_path)
            self.frame_source = video_src
            self.reset()
            self.logger.log_event("MODE_SWITCH", f"Loaded raw video source: {video_path}")
            return True
        except Exception as e:
            self.logger.log_event("ERROR", f"Failed to open video source: {e}")
            return False

    def load_simulator_source(self) -> None:
        """Switch back to SimulatorFrameSource."""
        self.frame_source = SimulatorFrameSource()
        self.frame_source.disturbance_injector = self.disturbance_injector
        self.reset()
        self.logger.log_event("MODE_SWITCH", "Switched back to synthetic dynamic space simulator")

    def reset(self) -> None:
        """Reset all pipeline stages to initial state."""
        self.frame_source.reset()
        self.detector.reset()
        self.tracker.reset()
        self.reacquisition.stop_search()
        self.controller.reset()
        self.logger.reset()
        self.forced_occlusion_frames = 0
        self._lost_frames = 0
        self._unacquired_frames = 0
        self._last_target_pan_deg = 0.0
        self._last_target_tilt_deg = 0.0
        self.last_step_time = time.perf_counter()

    def trigger_occlusion(self, duration_frames: int = 35) -> None:
        """Inject complete beam blockage / dark frame for testing re-acquisition."""
        self.forced_occlusion_frames = duration_frames
        self.logger.log_event("BEAM_BREAK", f"Injected simulated beam break ({duration_frames} frames)")

    def process_frame(self) -> EngineOutput:
        """Execute one complete tracking cycle at >= 30Hz."""
        now = time.perf_counter()
        dt = max(0.001, min(0.1, now - self.last_step_time))
        self.last_step_time = now
        self.frame_index += 1

        # 1. Acquire raw camera frame (isolated boundary)
        frame = self.frame_source.get_frame()
        if self.forced_occlusion_frames > 0:
            self.forced_occlusion_frames -= 1
            frame = np.full_like(frame, 14)
            if self.forced_occlusion_frames == 0:
                self.logger.loss_start_time = time.perf_counter()
                self.logger.log_event("BEAM_RESTORED", "Beam blockage cleared, measuring re-acquisition lock time")

        # 2. Detect optical beacon using classical CV
        detection = self.detector.detect(frame)

        # 3. State Estimation & Re-acquisition Logic
        search_waypoints: List[Tuple[float, float]] = []
        active_wp_idx: int = 0
        status_label: str = "INITIALIZING"

        if self.reacquisition.is_active:
            # Cut hexagonal spiral search is currently executing
            active_wp_idx = self.reacquisition.get_current_waypoint_index()

            # Advance search step: returns target gimbal angles (target_pan_deg, target_tilt_deg)
            cam_p = getattr(self.frame_source, 'pan_deg', 0.0)
            cam_t = getattr(self.frame_source, 'tilt_deg', 0.0)
            wp = self.reacquisition.step(detection.confidence, cam_p, cam_t)

            if wp is not None:
                target_pan_deg, target_tilt_deg = wp
                if hasattr(self.frame_source, 'pan_deg') and hasattr(self.frame_source, 'tilt_deg'):
                    err_pan = target_pan_deg - self.frame_source.pan_deg
                    err_tilt = target_tilt_deg - self.frame_source.tilt_deg
                    max_pan_spd = getattr(self.frame_source, 'max_pan_speed_deg_s', getattr(self.frame_source, 'max_ptz_speed_deg_s', 8.0))
                    max_tilt_spd = getattr(self.frame_source, 'max_tilt_speed_deg_s', getattr(self.frame_source, 'max_ptz_speed_deg_s', 8.0))
                    pan_rate = max(-max_pan_spd, min(max_pan_spd, err_pan * 8.0))
                    tilt_rate = max(-max_tilt_spd, min(max_tilt_spd, err_tilt * 8.0))
                    self.frame_source.apply_ptz_velocity(pan_rate, tilt_rate, dt)
                    self.controller.integral_x = 0.0
                    self.controller.integral_y = 0.0

                # Update tracker with candidate measurement
                track = self.tracker.update(detection, dt)
                status_label = "ACQUIRING [SPIRAL SEARCH]"
            else:
                # Re-acquisition succeeded or search pattern finished!
                track = self.tracker.update(detection, dt)
                status_label = "RE-ACQUIRED"
                self._unacquired_frames = 0

            # Convert angular waypoints to pixel coordinates on current frame for HUD
            pxd_x = getattr(self.frame_source, 'px_per_deg_x', 160.0)
            pxd_y = getattr(self.frame_source, 'px_per_deg_y', 160.0)
            cam_p = getattr(self.frame_source, 'pan_deg', 0.0)
            cam_t = getattr(self.frame_source, 'tilt_deg', 0.0)
            search_waypoints = [
                (320.0 + (wp_p - cam_p) * pxd_x, 240.0 + (wp_t - cam_t) * pxd_y)
                for wp_p, wp_t in self.reacquisition.get_waypoints_deg()
            ]
        else:
            # Normal state estimation step
            track = self.tracker.update(detection, dt)

            if track.status == TrackStatus.TRACKING:
                self._lost_frames = 0
                self._unacquired_frames = 0
                status_label = "TRACKING [LOCKED]"

                # Record last known target gimbal angles
                cam_p = getattr(self.frame_source, 'pan_deg', 0.0)
                cam_t = getattr(self.frame_source, 'tilt_deg', 0.0)
                pxd_x = getattr(self.frame_source, 'px_per_deg_x', 160.0)
                pxd_y = getattr(self.frame_source, 'px_per_deg_y', 160.0)
                self._last_target_pan_deg = cam_p + (track.pos[0] - 320.0) / pxd_x
                self._last_target_tilt_deg = cam_t + (track.pos[1] - 240.0) / pxd_y

                # Close the loop using Kalman filtered state + velocity feedforward
                pan_rate, tilt_rate = self.controller.compute_command(
                    target_pos=track.pos, dt=dt, target_vel=track.vel
                )
                self.frame_source.apply_ptz_velocity(pan_rate, tilt_rate, dt)

            elif track.status == TrackStatus.DEGRADED:
                status_label = "DEGRADED [COASTING]"
                # Coast gimbal with damped velocity, clamp if predicted position is off-screen
                if 0 <= track.pos[0] <= 640 and 0 <= track.pos[1] <= 480:
                    pan_rate, tilt_rate = self.controller.compute_command(
                        target_pos=track.pos, dt=dt, target_vel=(track.vel[0] * 0.4, track.vel[1] * 0.4)
                    )
                else:
                    pan_rate, tilt_rate = 0.0, 0.0
                self.frame_source.apply_ptz_velocity(pan_rate, tilt_rate, dt)

            elif track.status == TrackStatus.LOST:
                self._lost_frames += 1
                if self._lost_frames > 60:
                    self._lost_frames = 0
                    self.tracker.x = np.array([[320.0], [240.0], [0.0], [0.0]], dtype=np.float64)
                    self.tracker.P = np.diag([2500.0, 2500.0, 10000.0, 10000.0]).astype(np.float64)
                    self.tracker.consecutive_misses = 0

                status_label = "LOST [INIT SEARCH]"
                cam_p = getattr(self.frame_source, 'pan_deg', 0.0)
                cam_t = getattr(self.frame_source, 'tilt_deg', 0.0)
                pxd_x = getattr(self.frame_source, 'px_per_deg_x', 160.0)
                pxd_y = getattr(self.frame_source, 'px_per_deg_y', 160.0)

                # Initiate cut hexagonal spiral search starting at last known position or current camera center
                start_center = (self._last_target_pan_deg, self._last_target_tilt_deg)
                self.reacquisition.start_search(center_deg=start_center, max_radius_deg=5.8, initial_radius_deg=1.6)
                search_waypoints = [
                    (320.0 + (wp_p - cam_p) * pxd_x, 240.0 + (wp_t - cam_t) * pxd_y)
                    for wp_p, wp_t in self.reacquisition.get_waypoints_deg()
                ]
                active_wp_idx = self.reacquisition.get_current_waypoint_index()
            else:
                # TrackStatus.INITIALIZING: Target not locked in FOV
                self._unacquired_frames += 1
                if self._unacquired_frames >= 4:
                    # Target starts off-center -> Autonomously sweep uncertainty zone with Cut Hexagonal Spiral
                    cam_p = getattr(self.frame_source, 'pan_deg', 0.0)
                    cam_t = getattr(self.frame_source, 'tilt_deg', 0.0)
                    pxd_x = getattr(self.frame_source, 'px_per_deg_x', 160.0)
                    pxd_y = getattr(self.frame_source, 'px_per_deg_y', 160.0)
                    start_center = (cam_p, cam_t)
                    self.reacquisition.start_search(center_deg=start_center, max_radius_deg=5.8, initial_radius_deg=1.6)
                    search_waypoints = [
                        (320.0 + (wp_p - cam_p) * pxd_x, 240.0 + (wp_t - cam_t) * pxd_y)
                        for wp_p, wp_t in self.reacquisition.get_waypoints_deg()
                    ]
                    active_wp_idx = self.reacquisition.get_current_waypoint_index()
                    status_label = "ACQUIRING [SPIRAL SEARCH]"
                else:
                    status_label = "ACQUIRING"

        # 4. Tracking Error Vector Calculation (from reticle dead-center)
        if track.is_valid:
            tx, ty = track.pos
        elif detection.detected:
            tx, ty = detection.centroid
        else:
            tx, ty = self.reticle_center

        err_x = tx - self.reticle_center[0]
        err_y = ty - self.reticle_center[1]
        err_norm = math.hypot(err_x, err_y)

        # 5. Telemetry & Performance Logging
        ground_truth = self.frame_source.get_ground_truth_state()
        pan_deg = ground_truth.cam_angles_deg[0] if ground_truth else 0.0
        tilt_deg = ground_truth.cam_angles_deg[1] if ground_truth else 0.0

        self.logger.record_step(
            frame_idx=self.frame_index,
            tracking_error_px=err_norm,
            status="TRACKING" if track.status == TrackStatus.TRACKING else (
                "DEGRADED" if track.status == TrackStatus.DEGRADED else "LOST"
            ),
            confidence=detection.confidence,
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            is_searching=self.reacquisition.is_active,
            is_video_mode=self.frame_source.is_video_mode,
        )

        snapshot = self.logger.get_snapshot()

        return EngineOutput(
            frame=frame,
            detection=detection,
            track=track,
            is_searching=self.reacquisition.is_active,
            search_waypoints=search_waypoints,
            active_waypoint_idx=active_wp_idx,
            error_vector_px=(err_x, err_y),
            tracking_error_px=err_norm,
            status_label=status_label,
            telemetry=snapshot,
            ground_truth=ground_truth,
            is_ai_dampener_active=self.controller.is_ai_dampener_active,
        )
