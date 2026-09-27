"""2-Axis PID Gimbal Controller for FSOC coarse-pointing tracking.

Converts pixel tracking error from frame reticle center (320, 240) into pan/tilt
angular rate commands (deg/s):
- Proportional, Integral (with anti-windup clamp), Derivative with low-pass filtering
- Gimbal slew rate limiter enforcing physical speed limits (5-10 deg/s)
- Deadband to eliminate steady-state actuator jitter
"""

from dataclasses import dataclass
import math
import os
from typing import Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


@dataclass
class PIDGains:
    """Tunable PID gains for gimbal rate control."""
    kp: float = 14.0         # Proportional gain (deg/s per deg error)
    ki: float = 10.0         # Integral gain (deg/s per deg*s)
    kd: float = 0.30         # Derivative gain (deg/s per deg/s)
    i_max: float = 15.0      # Maximum anti-windup integral contribution (deg/s)
    deadband_px: float = 0.0 # Actuator deadband in pixels


class TinyDDPGDampener:
    """Wake-on-Degradation Reinforcement Learning Actuation Dampener (Core 3 AI).
    
    Sleeps during nominal flight (error <= 15px).
    Wakes up when tracking error or derivative spikes due to high-G maneuvers or aerodynamic shear.
    Outputs non-linear high-frequency rate adjustments (delta u) to suppress overshoot and jitter.
    """

    def __init__(
        self,
        model_path: str = "models/tinyddpg_dampener.onnx",
        wake_error_threshold_px: float = 18.0,
        px_per_deg: float = 160.0,
        max_damping_deg_s: float = 1.5,
    ):
        self.model_path = model_path
        self.wake_thresh_px = wake_error_threshold_px
        self.px_per_deg = px_per_deg
        self.max_damping = max_damping_deg_s
        self.is_active: bool = False

        self.prev_err_px_x: float = 0.0
        self.prev_err_px_y: float = 0.0
        self.filtered_d_err_x: float = 0.0
        self.filtered_d_err_y: float = 0.0
        self.has_prev_error: bool = False

        # Load ONNX actor session if model exists
        self.onnx_session = None
        if _ONNX_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.onnx_session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            except Exception:
                self.onnx_session = None

    def reset(self) -> None:
        """Reset internal dampener memory."""
        self.is_active = False
        self.prev_err_px_x = 0.0
        self.prev_err_px_y = 0.0
        self.filtered_d_err_x = 0.0
        self.filtered_d_err_y = 0.0
        self.has_prev_error = False

    def compute_damping(
        self,
        err_px_x: float,
        err_px_y: float,
        pid_cmd_x: float,
        pid_cmd_y: float,
        dt: float,
        target_vel_px_s: float = 0.0,
        current_omega_deg_s: float = 0.0,
    ) -> Tuple[float, float, bool]:
        """Compute rate dampening delta (deg/s) for pan and tilt axes."""
        dt = max(0.001, min(0.1, dt))
        err_mag = math.hypot(err_px_x, err_px_y)

        # Calculate rate of error change with low-pass filtering
        if self.has_prev_error:
            raw_d_err_x = (err_px_x - self.prev_err_px_x) / dt
            raw_d_err_y = (err_px_y - self.prev_err_px_y) / dt
            alpha = 0.65
            self.filtered_d_err_x = alpha * raw_d_err_x + (1.0 - alpha) * self.filtered_d_err_x
            self.filtered_d_err_y = alpha * raw_d_err_y + (1.0 - alpha) * self.filtered_d_err_y
        else:
            self.filtered_d_err_x = 0.0
            self.filtered_d_err_y = 0.0
            self.has_prev_error = True

        self.prev_err_px_x = err_px_x
        self.prev_err_px_y = err_px_y
        d_err_mag = math.hypot(self.filtered_d_err_x, self.filtered_d_err_y)

        # Wake-on-Degradation trigger condition: error > 18px or high derivative spike
        if err_mag >= self.wake_thresh_px or d_err_mag >= 150.0:
            self.is_active = True

            # If ONNX trained actor model is available, evaluate policy
            if self.onnx_session is not None:
                try:
                    state_vec = np.array(
                        [
                            err_px_x / 100.0,
                            err_px_y / 100.0,
                            self.filtered_d_err_x / 200.0,
                            self.filtered_d_err_y / 200.0,
                            pid_cmd_x / 10.0,
                            pid_cmd_y / 10.0,
                            target_vel_px_s / 300.0,
                            current_omega_deg_s / 20.0,
                        ],
                        dtype=np.float32,
                    )[np.newaxis, :] # shape (1, 8)
                    action = self.onnx_session.run(None, {"state_vector": state_vec})[0][0]
                    delta_x = float(action[0]) * self.max_damping
                    delta_y = float(action[1]) * self.max_damping
                except Exception:
                    # Fallback to analytical rate dampener (properly scaled in deg/s)
                    d_err_deg_x = self.filtered_d_err_x / self.px_per_deg
                    d_err_deg_y = self.filtered_d_err_y / self.px_per_deg
                    delta_x = -0.12 * d_err_deg_x
                    delta_y = -0.12 * d_err_deg_y
            else:
                # Analytical rate dampener (convert pixel rate to deg/s)
                d_err_deg_x = self.filtered_d_err_x / self.px_per_deg
                d_err_deg_y = self.filtered_d_err_y / self.px_per_deg
                scale = min(err_mag / 20.0, 1.8)
                delta_x = -0.15 * d_err_deg_x * scale
                delta_y = -0.15 * d_err_deg_y * scale

            # Clamp damping effort
            delta_x = max(-self.max_damping, min(self.max_damping, delta_x))
            delta_y = max(-self.max_damping, min(self.max_damping, delta_y))
            return (delta_x, delta_y, True)

        else:
            self.is_active = False
            return (0.0, 0.0, False)


class PTZPIDController:
    """Closed-loop 2-axis (pan/tilt) rate controller with Wake-on-Degradation DDPG dampener."""

    def __init__(
        self,
        gains: PIDGains = None,
        px_per_deg_x: float = 160.0,    # 640px / 4.0 deg
        px_per_deg_y: float = 160.0,    # 480px / 3.0 deg
        max_speed_deg_s: float = 8.0,   # Physical gimbal angular speed limit
        frame_center: Tuple[float, float] = (320.0, 240.0),
    ):
        self.gains = gains or PIDGains()
        self.px_per_deg_x = px_per_deg_x
        self.px_per_deg_y = px_per_deg_y
        self.max_speed = max_speed_deg_s
        self.cx, self.cy = frame_center

        # Wake-on-Degradation DDPG Dampener (Core 3 AI)
        self.dampener = TinyDDPGDampener()
        self.is_ai_dampener_active: bool = False

        # Integrator states
        self.integral_x: float = 0.0
        self.integral_y: float = 0.0

        # Previous error states for derivative computation
        self.prev_error_deg_x: float = 0.0
        self.prev_error_deg_y: float = 0.0
        self.prev_deriv_x: float = 0.0
        self.prev_deriv_y: float = 0.0

        self.has_prev: bool = False

    def reset(self) -> None:
        """Reset PID integrators and derivative memory."""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_deg_x = 0.0
        self.prev_error_deg_y = 0.0
        self.prev_deriv_x = 0.0
        self.prev_deriv_y = 0.0
        self.has_prev = False
        self.dampener.reset()
        self.is_ai_dampener_active = False

    def set_max_speed(self, max_speed_deg_s: float) -> None:
        """Update maximum gimbal slew rate limit."""
        self.max_speed = max(1.0, min(30.0, max_speed_deg_s))

    def compute_command(
        self,
        target_pos: Tuple[float, float],
        dt: float,
        target_vel: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """Compute commanded pan/tilt angular velocity in deg/s.

        Args:
            target_pos: (x, y) target position in camera coordinates.
            dt: elapsed time in seconds.
            target_vel: Optional (vx, vy) estimated velocity in px/s for feedforward.

        Returns:
            Tuple[float, float]: (pan_rate_deg_s, tilt_rate_deg_s)
        """
        dt = max(0.001, min(0.1, dt))
        tx, ty = target_pos

        # Pixel error from reticle dead-center
        err_px_x = tx - self.cx
        err_px_y = ty - self.cy

        # Deadband check to avoid micro-chatter
        if abs(err_px_x) < self.gains.deadband_px:
            err_px_x = 0.0
        if abs(err_px_y) < self.gains.deadband_px:
            err_px_y = 0.0

        # ---------------------------------------------------------------------
        # Step 1: Angular Error Computation & Actuator Deadband
        # ---------------------------------------------------------------------
        # Convert pixel tracking error to gimbal angular space (degrees):
        # theta_deg = pixel_error / (pixels_per_degree)
        err_deg_x = err_px_x / self.px_per_deg_x
        err_deg_y = err_px_y / self.px_per_deg_y

        # ---------------------------------------------------------------------
        # Step 2: Proportional (P) Term
        # ---------------------------------------------------------------------
        # Commands gimbal velocity proportional to instantaneous pointing displacement
        p_term_x = self.gains.kp * err_deg_x
        p_term_y = self.gains.kp * err_deg_y

        # ---------------------------------------------------------------------
        # Step 3: Integral (I) Term with Anti-Windup Clamping
        # ---------------------------------------------------------------------
        # Eliminates steady-state tracking offset while clamping accumulated error
        # to prevent integrator windup during high-slew maneuvers or occlusions.
        self.integral_x += err_deg_x * dt
        self.integral_y += err_deg_y * dt

        max_i = self.gains.i_max / max(self.gains.ki, 1e-4)
        self.integral_x = max(-max_i, min(max_i, self.integral_x))
        self.integral_y = max(-max_i, min(max_i, self.integral_y))

        i_term_x = self.gains.ki * self.integral_x
        i_term_y = self.gains.ki * self.integral_y

        # ---------------------------------------------------------------------
        # Step 4: Filtered Derivative (D) Term (Lead Compensation)
        # ---------------------------------------------------------------------
        # First-order low-pass filtered numerical derivative provides damping
        # without amplifying high-frequency pixel centroid jitter.
        if self.has_prev:
            raw_deriv_x = (err_deg_x - self.prev_error_deg_x) / dt
            raw_deriv_y = (err_deg_y - self.prev_error_deg_y) / dt
            alpha = 0.70 # LPF smoothing cutoff factor
            filtered_deriv_x = alpha * raw_deriv_x + (1.0 - alpha) * self.prev_deriv_x
            filtered_deriv_y = alpha * raw_deriv_y + (1.0 - alpha) * self.prev_deriv_y
        else:
            filtered_deriv_x = 0.0
            filtered_deriv_y = 0.0
            self.has_prev = True

        self.prev_error_deg_x = err_deg_x
        self.prev_error_deg_y = err_deg_y
        self.prev_deriv_x = filtered_deriv_x
        self.prev_deriv_y = filtered_deriv_y

        d_term_x = self.gains.kd * filtered_deriv_x
        d_term_y = self.gains.kd * filtered_deriv_y

        # Commanded PID rate before DDPG damping and mechanical speed limits
        cmd_x = p_term_x + i_term_x + d_term_x
        cmd_y = p_term_y + i_term_y + d_term_y

        # Core 3 AI: Wake-on-Degradation DDPG Dampener Evaluation
        target_v_mag = math.hypot(target_vel[0], target_vel[1]) if target_vel else 0.0
        cur_omega = math.hypot(cmd_x, cmd_y)
        damp_x, damp_y, is_active = self.dampener.compute_damping(
            err_px_x=err_px_x,
            err_px_y=err_px_y,
            pid_cmd_x=cmd_x,
            pid_cmd_y=cmd_y,
            dt=dt,
            target_vel_px_s=target_v_mag,
            current_omega_deg_s=cur_omega,
        )
        self.is_ai_dampener_active = is_active

        # Blend DDPG damping with PID rate command
        cmd_x += damp_x
        cmd_y += damp_y

        # Slew rate limiter: enforce physical max gimbal velocity
        cmd_x = max(-self.max_speed, min(self.max_speed, cmd_x))
        cmd_y = max(-self.max_speed, min(self.max_speed, cmd_y))

        return (cmd_x, cmd_y)

