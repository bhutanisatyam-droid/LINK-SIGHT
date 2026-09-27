"""FrameSource module for the FSOC Coarse-Alignment Virtual Tracking System.

Provides an abstract interface and two decoupled implementations:
- SimulatorFrameSource: Renders a 2000x2000 scene with moving optical beacon and virtual PTZ camera.
- VideoFileFrameSource: Reads a user-provided .mp4 video for benchmark validation on raw footage.

Architectural Rule:
Downstream modules (detector, tracker, controller) access ONLY get_frame().
Ground truth is isolated and strictly routed to telemetry/minimap only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


class MotionModel(Enum):
    """Supported beacon trajectory patterns."""
    CIRCULAR = "circular"
    STRAIGHT = "straight"
    FIGURE_8 = "figure_8"
    RANDOM = "random"


class TargetShape(Enum):
    """Supported optical beacon spot shapes (ISRO Parameter #9)."""
    SQUARE = "square"      # Default flat-top square core + Gaussian halo
    CIRCLE = "circle"      # Circular disc + radial halo
    CROSS = "cross"        # Anamorphic crosshair / optical plus pattern


@dataclass
class GroundTruthState:
    """Telemetry data isolated for minimap / truth-validation logging only.
    NEVER to be consumed by detector, tracker, or controller.
    """
    target_pos: Tuple[float, float]     # (x, y) in 2000x2000 scene
    cam_pos: Tuple[float, float]        # (x, y) center of 640x480 crop in 2000x2000 scene
    cam_angles_deg: Tuple[float, float] # (pan_deg, tilt_deg)
    cam_fov_deg: Tuple[float, float]    # (fov_pan, fov_tilt) = (4.0, 3.0)
    scene_size: Tuple[int, int]         # (2000, 2000)
    target_trail: List[Tuple[float, float]] = field(default_factory=list)
    is_target_in_fov: bool = True


class BaseFrameSource(ABC):
    """Abstract interface for all frame acquisition sources."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Fetch the next 640x480 frame.
        Returns:
            np.ndarray: uint8 image of shape (480, 640) or (480, 640, 3).
        """
        pass

    @abstractmethod
    def apply_ptz_velocity(self, pan_vel_deg_s: float, tilt_vel_deg_s: float, dt: float) -> None:
        """Apply commanded pan/tilt angular rates over time interval dt."""
        pass

    @abstractmethod
    def apply_ptz_offset(self, d_pan_deg: float, d_tilt_deg: float) -> None:
        """Apply discrete pan/tilt displacement (e.g. for re-acquisition search steps)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal kinematics, gimbal positions, or video streams."""
        pass

    @property
    @abstractmethod
    def is_video_mode(self) -> bool:
        """Whether this source is reading recorded video footage (PTZ commands are no-ops)."""
        pass

    @abstractmethod
    def get_ground_truth_state(self) -> Optional[GroundTruthState]:
        """Telemetry hook strictly for minimap visualization and benchmark logging.
        Returns None for VideoFileFrameSource.
        """
        pass


class SimulatorFrameSource(BaseFrameSource):
    """Simulates a 2000x2000 px space scene with an optical beacon and virtual PTZ camera."""

    def __init__(
        self,
        scene_width: int = 2000,
        scene_height: int = 2000,
        cam_width: int = 640,
        cam_height: int = 480,
        fov_pan_deg: float = 4.0,
        fov_tilt_deg: float = 3.0,
        target_size_px: int = 10,
        motion_model: MotionModel = MotionModel.CIRCULAR,
        max_pan_speed_deg_s: float = 5.0,
        max_tilt_speed_deg_s: float = 5.0,
        target_shape: TargetShape = TargetShape.SQUARE,
        initial_pos_mode: str = "random",
        custom_initial_x: float = 1000.0,
        custom_initial_y: float = 1000.0,
    ):
        self.scene_w = scene_width
        self.scene_h = scene_height
        self.cam_w = cam_width
        self.cam_h = cam_height
        self.fov_pan_deg = fov_pan_deg
        self.fov_tilt_deg = fov_tilt_deg
        self.target_size = target_size_px
        self.motion_model = motion_model
        self.max_pan_speed_deg_s = max(1.0, min(10.0, max_pan_speed_deg_s))
        self.max_tilt_speed_deg_s = max(1.0, min(10.0, max_tilt_speed_deg_s))
        self.target_shape = target_shape
        self.initial_pos_mode = initial_pos_mode
        self.custom_initial_x = custom_initial_x
        self.custom_initial_y = custom_initial_y
        # RF Link simulation
        self.rf_link_active: bool = True
        self.rf_uncertainty_px: float = 80.0  # Gaussian sigma in scene pixels
        self.rf_target_angles_deg: Optional[Tuple[float, float]] = None

        # Angular scale: pixels per degree — dynamically computed from FOV
        self._recompute_fov_scaling()

        # Virtual PTZ state
        self.pan_deg: float = 0.0
        self.tilt_deg: float = 0.0

        # Beacon trajectory state
        self.sim_time: float = 0.0
        self.trail: List[Tuple[float, float]] = []
        self.max_trail_len: int = 200

        # Target trajectory parameters
        self.target_x: float = self.scene_w / 2.0
        self.target_y: float = self.scene_h / 2.0
        self.prev_target_x: float = self.target_x
        self.prev_target_y: float = self.target_y
        self.target_velocity_px_s: float = 0.0
        self.last_commanded_omega_deg_s: float = 0.0
        self.target_vx: float = 40.0 # px/s
        self.target_vy: float = 30.0 # px/s

        # Random walk state
        self.random_heading: float = np.random.uniform(0, 2 * math.pi)
        self.random_speed: float = 50.0 # px/s

        # Optional external disturbance injector hook
        self.disturbance_injector = None
        self.beam_occluded_frames: int = 0

        self.last_update_time: float = time.perf_counter()
        self.reset()

    @property
    def is_video_mode(self) -> bool:
        return False

    def _recompute_fov_scaling(self) -> None:
        """Recompute angular scale and gimbal limits based on current FOV settings."""
        self.px_per_deg_x = self.cam_w / self.fov_pan_deg
        self.px_per_deg_y = self.cam_h / self.fov_tilt_deg
        # Gimbal hard limits: 40px margin from scene boundary
        self.max_pan_deg = (self.scene_w / 2.0 - 40.0) / self.px_per_deg_x
        self.min_pan_deg = -self.max_pan_deg
        self.max_tilt_deg = (self.scene_h / 2.0 - 40.0) / self.px_per_deg_y
        self.min_tilt_deg = -self.max_tilt_deg

    def reset(self) -> None:
        """Reset kinematics, gimbal pointing, and target trajectory."""
        self.sim_time = 0.0
        self.trail.clear()
        self.last_update_time = time.perf_counter()
        self.last_commanded_omega_deg_s = 0.0
        self.target_velocity_px_s = 0.0

        # Determine initial spawn coordinates (ISRO Parameter #11: Default = Random)
        # Margin ensures full orbit / trajectory stays strictly inside 2000x2000 scene space
        if self.motion_model == MotionModel.CIRCULAR:
            margin = 250.0
        elif self.motion_model == MotionModel.FIGURE_8:
            margin = 400.0
        else:
            margin = 200.0

        if self.initial_pos_mode == "random":
            spawn_x = float(np.random.uniform(margin, self.scene_w - margin))
            spawn_y = float(np.random.uniform(margin, self.scene_h - margin))
        else:
            spawn_x = float(np.clip(self.custom_initial_x, margin, self.scene_w - margin))
            spawn_y = float(np.clip(self.custom_initial_y, margin, self.scene_h - margin))

        self.orbit_center_x = spawn_x
        self.orbit_center_y = spawn_y

        if self.motion_model == MotionModel.CIRCULAR:
            r = 220.0
            self.target_x = spawn_x + r
            self.target_y = spawn_y
        elif self.motion_model == MotionModel.STRAIGHT:
            self.target_x = spawn_x
            self.target_y = spawn_y
            self.target_vx = 60.0
            self.target_vy = 35.0
        elif self.motion_model == MotionModel.FIGURE_8:
            self.target_x = spawn_x
            self.target_y = spawn_y
        elif self.motion_model == MotionModel.RANDOM:
            self.target_x = spawn_x
            self.target_y = spawn_y
            self.random_heading = np.random.uniform(0, 2 * math.pi)

        self.prev_target_x = self.target_x
        self.prev_target_y = self.target_y

        # Gimbal ALWAYS starts physically boresighted at center (0.0, 0.0)
        self.pan_deg = 0.0
        self.tilt_deg = 0.0

        # --- RF LINK SIMULATION ---
        # If RF Side-Link is active: assign RF target pointing angles (perturbed by Gaussian error).
        # The camera gimbal will dynamically slew across the sky at max PTZ rate toward this coordinate.
        # If RF Side-Link is OFF: no RF target is given; camera remains at boresight (0, 0) and performs spiral search outward.
        scene_cx = self.scene_w / 2.0
        scene_cy = self.scene_h / 2.0
        if self.rf_link_active:
            max_err = 1.8 * self.rf_uncertainty_px
            noise_x = float(np.clip(np.random.normal(0.0, self.rf_uncertainty_px), -max_err, max_err))
            noise_y = float(np.clip(np.random.normal(0.0, self.rf_uncertainty_px), -max_err, max_err))
            rf_pan_deg = (self.target_x + noise_x - scene_cx) / self.px_per_deg_x
            rf_tilt_deg = (self.target_y + noise_y - scene_cy) / self.px_per_deg_y
            self.rf_target_angles_deg = (
                max(self.min_pan_deg, min(self.max_pan_deg, rf_pan_deg)),
                max(self.min_tilt_deg, min(self.max_tilt_deg, rf_tilt_deg)),
            )
        else:
            self.rf_target_angles_deg = None


    def get_rf_target_angles(self) -> Optional[Tuple[float, float]]:
        """Return dynamic RF broadcast coordinates with Gaussian pointing error."""
        if not self.rf_link_active:
            return None
        scene_cx = self.scene_w / 2.0
        scene_cy = self.scene_h / 2.0
        max_err = 1.8 * self.rf_uncertainty_px
        noise_x = float(np.clip(np.random.normal(0.0, self.rf_uncertainty_px), -max_err, max_err))
        noise_y = float(np.clip(np.random.normal(0.0, self.rf_uncertainty_px), -max_err, max_err))
        rf_pan_deg = (self.target_x + noise_x - scene_cx) / self.px_per_deg_x
        rf_tilt_deg = (self.target_y + noise_y - scene_cy) / self.px_per_deg_y
        return (
            max(self.min_pan_deg, min(self.max_pan_deg, rf_pan_deg)),
            max(self.min_tilt_deg, min(self.max_tilt_deg, rf_tilt_deg)),
        )

    def set_beam_occluded(self, duration_frames: int) -> None:
        """Inject physical beam blockage / optical path occlusion for N frames."""
        self.beam_occluded_frames = max(0, int(duration_frames))

    def set_rf_link(self, active: bool, uncertainty_px: float) -> None:
        """Configure RF side-link simulation (active flag + Gaussian pointing error)."""
        self.rf_link_active = active
        self.rf_uncertainty_px = max(10.0, min(300.0, uncertainty_px))
        self.rf_target_angles_deg = self.get_rf_target_angles()

    def set_motion_model(self, model: MotionModel) -> None:
        """Dynamically update beacon motion model."""
        self.motion_model = model
        self.reset()

    def set_target_size(self, size_px: int) -> None:
        """Update beacon physical size in pixels (5-20px)."""
        self.target_size = max(5, min(25, size_px))

    def set_fov(self, fov_pan_deg: float, fov_tilt_deg: float) -> None:
        """Dynamically update camera Field of View (ISRO Parameter #4)."""
        self.fov_pan_deg = max(1.0, min(12.0, fov_pan_deg))
        self.fov_tilt_deg = max(1.0, min(9.0, fov_tilt_deg))
        self._recompute_fov_scaling()

    def set_target_shape(self, shape: TargetShape) -> None:
        """Set the optical beacon rendering shape (ISRO Parameter #9)."""
        self.target_shape = shape

    def set_initial_pos_mode(self, mode: str, custom_x: float = 1000.0, custom_y: float = 1000.0) -> None:
        """Set spawn mode: 'random' (ISRO default) or 'custom' with explicit coordinates."""
        self.initial_pos_mode = mode
        self.custom_initial_x = custom_x
        self.custom_initial_y = custom_y
        self.reset()

    def set_max_pan_speed(self, max_speed_deg_s: float) -> None:
        """Update maximum Pan axis slew rate (ISRO Parameter #13)."""
        self.max_pan_speed_deg_s = max(1.0, min(10.0, max_speed_deg_s))

    def set_max_tilt_speed(self, max_speed_deg_s: float) -> None:
        """Update maximum Tilt axis slew rate (ISRO Parameter #14)."""
        self.max_tilt_speed_deg_s = max(1.0, min(10.0, max_speed_deg_s))

    def set_max_ptz_speed(self, max_speed_deg_s: float) -> None:
        """Legacy unified setter — sets both pan and tilt speeds equally."""
        self.set_max_pan_speed(max_speed_deg_s)
        self.set_max_tilt_speed(max_speed_deg_s)

    def apply_ptz_velocity(self, pan_vel_deg_s: float, tilt_vel_deg_s: float, dt: float) -> None:
        """Slew virtual PTZ camera with independent Pan & Tilt slew rate limiting (ISRO #13 & #14)."""
        # Independent axis slew rate clamping
        limited_pan_vel = max(-self.max_pan_speed_deg_s, min(self.max_pan_speed_deg_s, pan_vel_deg_s))
        limited_tilt_vel = max(-self.max_tilt_speed_deg_s, min(self.max_tilt_speed_deg_s, tilt_vel_deg_s))

        self.last_commanded_omega_deg_s = math.sqrt(limited_pan_vel ** 2 + limited_tilt_vel ** 2)

        self.pan_deg += limited_pan_vel * dt
        self.tilt_deg += limited_tilt_vel * dt

        # Enforce gimbal hard stops
        self.pan_deg = max(self.min_pan_deg, min(self.max_pan_deg, self.pan_deg))
        self.tilt_deg = max(self.min_tilt_deg, min(self.max_tilt_deg, self.tilt_deg))

    def apply_ptz_offset(self, d_pan_deg: float, d_tilt_deg: float) -> None:
        """Apply direct angular offset (e.g. for re-acquisition search steps)."""
        self.pan_deg += d_pan_deg
        self.tilt_deg += d_tilt_deg
        self.pan_deg = max(self.min_pan_deg, min(self.max_pan_deg, self.pan_deg))
        self.tilt_deg = max(self.min_tilt_deg, min(self.max_tilt_deg, self.tilt_deg))

    def update_physics(self, dt: float) -> None:
        """Step beacon position forward in time according to active motion model."""
        self.sim_time += dt
        cx = getattr(self, 'orbit_center_x', self.scene_w / 2.0)
        cy = getattr(self, 'orbit_center_y', self.scene_h / 2.0)

        if self.motion_model == MotionModel.CIRCULAR:
            # Orbital beacon trajectory centered at spawn position
            radius = 220.0 # pixels
            angular_velocity = 0.22 # rad/s (~28s full orbit)
            theta = angular_velocity * self.sim_time
            self.target_x = cx + radius * math.cos(theta)
            self.target_y = cy + radius * math.sin(theta)

        elif self.motion_model == MotionModel.STRAIGHT:
            # Linear trajectory with boundary bounce
            self.target_x += self.target_vx * dt
            self.target_y += self.target_vy * dt

            margin = 250.0
            if self.target_x < margin or self.target_x > self.scene_w - margin:
                self.target_vx = -self.target_vx
                self.target_x = max(margin, min(self.scene_w - margin, self.target_x))
            if self.target_y < margin or self.target_y > self.scene_h - margin:
                self.target_vy = -self.target_vy
                self.target_y = max(margin, min(self.scene_h - margin, self.target_y))

        elif self.motion_model == MotionModel.FIGURE_8:
            # Lemniscate of Gerono centered at spawn position
            scale_x = 380.0
            scale_y = 260.0
            omega = 0.30
            t = omega * self.sim_time
            self.target_x = cx + scale_x * math.sin(t)
            self.target_y = cy + scale_y * math.sin(t) * math.cos(t)

        elif self.motion_model == MotionModel.RANDOM:
            # Gauss-Markov smooth random walk
            heading_change = np.random.normal(0, 1.2) * dt
            self.random_heading += heading_change
            self.random_speed = np.clip(self.random_speed + np.random.normal(0, 15.0) * dt, 20.0, 90.0)

            dx = self.random_speed * math.cos(self.random_heading) * dt
            dy = self.random_speed * math.sin(self.random_heading) * dt
            self.target_x += dx
            self.target_y += dy

            # Boundary repulsion
            margin = 200.0
            if self.target_x < margin:
                self.random_heading = 0.0
                self.target_x = margin
            elif self.target_x > self.scene_w - margin:
                self.random_heading = math.pi
                self.target_x = self.scene_w - margin
            if self.target_y < margin:
                self.random_heading = math.pi / 2
                self.target_y = margin
            elif self.target_y > self.scene_h - margin:
                self.random_heading = -math.pi / 2
                self.target_y = self.scene_h - margin

        # Hard constraint: target beacon remains strictly within 2000x2000 space boundary
        self.target_x = max(50.0, min(self.scene_w - 50.0, self.target_x))
        self.target_y = max(50.0, min(self.scene_h - 50.0, self.target_y))

        # Estimate instantaneous velocity
        inst_vx = (self.target_x - self.prev_target_x) / max(0.001, dt)
        inst_vy = (self.target_y - self.prev_target_y) / max(0.001, dt)
        self.target_velocity_px_s = math.sqrt(inst_vx ** 2 + inst_vy ** 2)
        self.prev_target_x = self.target_x
        self.prev_target_y = self.target_y

        # Append to historical trail for telemetry minimap
        self.trail.append((float(self.target_x), float(self.target_y)))
        if len(self.trail) > self.max_trail_len:
            self.trail.pop(0)

    def get_frame(self) -> np.ndarray:
        """Render the 640x480 virtual camera crop of the scene."""
        now = time.perf_counter()
        dt = max(0.001, min(0.1, now - self.last_update_time))
        self.last_update_time = now

        self.update_physics(dt)

        # ---------------------------------------------------------------------
        # Step 1: World-to-Camera Coordinate Transformation
        # ---------------------------------------------------------------------
        # Camera center in 2000x2000 arena space (pixels) based on gimbal pan/tilt angles:
        # X_cam = W/2 + theta_pan * px_per_deg_x
        # Y_cam = H/2 + theta_tilt * px_per_deg_y
        cam_center_x = self.scene_w / 2.0 + self.pan_deg * self.px_per_deg_x
        cam_center_y = self.scene_h / 2.0 + self.tilt_deg * self.px_per_deg_y

        # Camera sensor crop bounding box (640x480) in world scene coordinates
        x_min = int(cam_center_x - self.cam_w / 2)
        y_min = int(cam_center_y - self.cam_h / 2)

        # Initialize dark sensor background floor (ambient thermal offset)
        frame = np.full((self.cam_h, self.cam_w), 14, dtype=np.uint8)

        # Transform target world coordinates (X_tgt, Y_tgt) into camera pixel coordinates (u, v):
        # u = X_tgt - x_min,   v = Y_tgt - y_min
        u_target = int(self.target_x - x_min)
        v_target = int(self.target_y - y_min)

        half_s = self.target_size // 2

        # Check if optical beam is currently occluded by an atmospheric obstacle
        is_occluded = False
        if self.beam_occluded_frames > 0:
            self.beam_occluded_frames -= 1
            is_occluded = True

        # ---------------------------------------------------------------------
        # Step 2: Optical Laser Spot & Gaussian PSF Rendering
        # ---------------------------------------------------------------------
        # Render beacon if inside camera Field-of-View and beam is unblocked
        if not is_occluded and (-half_s <= u_target < self.cam_w + half_s and -half_s <= v_target < self.cam_h + half_s):

            if self.target_shape == TargetShape.SQUARE:
                # Flat-top square core (255 intensity) + Gaussian halo flare
                b_x1 = max(0, u_target - half_s)
                b_x2 = min(self.cam_w, u_target + half_s)
                b_y1 = max(0, v_target - half_s)
                b_y2 = min(self.cam_h, v_target + half_s)
                if b_x2 > b_x1 and b_y2 > b_y1:
                    frame[b_y1:b_y2, b_x1:b_x2] = 255

            elif self.target_shape == TargetShape.CIRCLE:
                # Circular saturated disc + radial halo
                radius = half_s
                for yy in range(max(0, v_target - radius - 2), min(self.cam_h, v_target + radius + 3)):
                    for xx in range(max(0, u_target - radius - 2), min(self.cam_w, u_target + radius + 3)):
                        dist_sq = (xx - u_target) ** 2 + (yy - v_target) ** 2
                        if dist_sq <= radius * radius:
                            frame[yy, xx] = 255

            elif self.target_shape == TargetShape.CROSS:
                # Anamorphic crosshair / optical plus pattern
                arm = half_s + 2
                thick = max(1, half_s // 3)
                # Horizontal arm
                hx1 = max(0, u_target - arm)
                hx2 = min(self.cam_w, u_target + arm)
                hy1 = max(0, v_target - thick)
                hy2 = min(self.cam_h, v_target + thick)
                if hx2 > hx1 and hy2 > hy1:
                    frame[hy1:hy2, hx1:hx2] = 255
                # Vertical arm
                vx1 = max(0, u_target - thick)
                vx2 = min(self.cam_w, u_target + thick)
                vy1 = max(0, v_target - arm)
                vy2 = min(self.cam_h, v_target + arm)
                if vx2 > vx1 and vy2 > vy1:
                    frame[vy1:vy2, vx1:vx2] = 255

            # Common optical PSF Gaussian halo flare for all shapes
            halo_radius = self.target_size + 4
            h_x1 = max(0, u_target - halo_radius)
            h_x2 = min(self.cam_w, u_target + halo_radius)
            h_y1 = max(0, v_target - halo_radius)
            h_y2 = min(self.cam_h, v_target + halo_radius)
            sigma_sq = 2.0 * (half_s * 1.5) ** 2
            for yy in range(h_y1, h_y2):
                for xx in range(h_x1, h_x2):
                    dist_sq = (xx - u_target) ** 2 + (yy - v_target) ** 2
                    flare = int(80 * math.exp(-dist_sq / sigma_sq))
                    if flare > 0 and frame[yy, xx] < 255:
                        frame[yy, xx] = min(255, frame[yy, xx] + flare)

        # Apply disturbance injector if attached (with real-time flight dynamics feedback)
        if self.disturbance_injector is not None:
            self.disturbance_injector.set_flight_dynamics(
                velocity_px_s=self.target_velocity_px_s,
                omega_deg_s=self.last_commanded_omega_deg_s
            )
            frame = self.disturbance_injector.apply(frame)

        return frame

    def get_ground_truth_state(self) -> Optional[GroundTruthState]:
        """Expose simulator ground truth strictly for minimap and performance validation."""
        cam_center_x = self.scene_w / 2.0 + self.pan_deg * self.px_per_deg_x
        cam_center_y = self.scene_h / 2.0 + self.tilt_deg * self.px_per_deg_y

        u_target = self.target_x - (cam_center_x - self.cam_w / 2)
        v_target = self.target_y - (cam_center_y - self.cam_h / 2)
        in_fov = (0 <= u_target <= self.cam_w) and (0 <= v_target <= self.cam_h)

        return GroundTruthState(
            target_pos=(self.target_x, self.target_y),
            cam_pos=(cam_center_x, cam_center_y),
            cam_angles_deg=(self.pan_deg, self.tilt_deg),
            cam_fov_deg=(self.fov_pan_deg, self.fov_tilt_deg),
            scene_size=(self.scene_w, self.scene_h),
            target_trail=list(self.trail),
            is_target_in_fov=in_fov,
        )


class VideoFileFrameSource(BaseFrameSource):
    """Reads recorded video (.mp4) frame-by-frame for Benchmark-2 mode.
    Proves detector & tracker operate on raw, un-simulated camera footage.
    PTZ commands are logged as no-ops.
    """

    def __init__(self, video_path: str, target_width: int = 640, target_height: int = 480):
        self.video_path = video_path
        self.target_w = target_width
        self.target_h = target_height

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {video_path}")

        self.last_log_time: float = 0.0
        self.logged_ptz_commands: int = 0

    @property
    def is_video_mode(self) -> bool:
        return True

    def reset(self) -> None:
        """Rewind video stream to start."""
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def apply_ptz_velocity(self, pan_vel_deg_s: float, tilt_vel_deg_s: float, dt: float) -> None:
        """No-op on recorded video (logs commanded rates for audit)."""
        self.logged_ptz_commands += 1

    def apply_ptz_offset(self, d_pan_deg: float, d_tilt_deg: float) -> None:
        """No-op on recorded video."""
        self.logged_ptz_commands += 1

    def get_frame(self) -> np.ndarray:
        """Read and resize next frame to 640x480."""
        ret, frame = self.cap.read()
        if not ret:
            # Loop video playback seamlessly
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                # Return empty frame if file read failed completely
                return np.zeros((self.target_h, self.target_w), dtype=np.uint8)

        # Convert to grayscale if color
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Resize to standard 640x480 if needed
        if gray.shape[1] != self.target_w or gray.shape[0] != self.target_h:
            gray = cv2.resize(gray, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        return gray

    def get_ground_truth_state(self) -> Optional[GroundTruthState]:
        """No ground truth exists for un-simulated video footage."""
        return None

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
