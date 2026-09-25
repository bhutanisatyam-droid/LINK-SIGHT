"""Gymnasium Environment for FSOC Coarse Tracking Control & RL Training.

Simulates a virtual PTZ tracking loop with:
- Dynamic target motion
- Velocity-coupled Dabiri PSD aerodynamic turbulence
- PID baseline control + RL additive damping action
"""

import math
from typing import Optional, Tuple
import numpy as np


class FSOCTrackingEnv:
    """Gym-compatible environment for training TinyDDPG-Dampener."""

    def __init__(self, max_steps: int = 500, dt: float = 1.0 / 60.0):
        self.max_steps = max_steps
        self.dt = dt

        # Camera dimensions
        self.cam_w = 640.0
        self.cam_h = 480.0
        self.center_x = 320.0
        self.center_y = 240.0
        self.deg_to_px = 640.0 / 4.0 # 160 px/deg

        # Action limit: Damping angular velocity in deg/s
        self.action_max = 2.5 # deg/s

        # PID Baseline Gains
        self.kp = 0.015
        self.ki = 0.0002
        self.kd = 0.004

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset tracking episode to randomized initial condition."""
        self.step_count = 0
        
        # Target position in virtual camera FOV
        self.target_x = self.center_x + np.random.uniform(-40.0, 40.0)
        self.target_y = self.center_y + np.random.uniform(-40.0, 40.0)
        
        # Target velocity
        self.drone_speed = np.random.uniform(20.0, 50.0)
        heading = np.random.uniform(0, 2 * math.pi)
        self.target_vx = self.drone_speed * math.cos(heading)
        self.target_vy = self.drone_speed * math.sin(heading)

        # Gimbal states
        self.pan_rate = 0.0
        self.tilt_rate = 0.0
        self.prev_error_x = self.target_x - self.center_x
        self.prev_error_y = self.target_y - self.center_y
        self.integral_x = 0.0
        self.integral_y = 0.0

        # Command history buffer
        self.cmd_hist = np.zeros(4, dtype=np.float32) # [wx_1, wy_1, wx_2, wy_2]

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Construct 8-dimensional observation vector."""
        ex = (self.target_x - self.center_x) / 100.0  # Normalize
        ey = (self.target_y - self.center_y) / 100.0
        dex = ((self.target_x - self.center_x) - self.prev_error_x) / (self.dt * 100.0)
        dey = ((self.target_y - self.center_y) - self.prev_error_y) / (self.dt * 100.0)
        
        obs = np.array([
            ex, ey, dex, dey,
            self.cmd_hist[0] / 5.0, self.cmd_hist[1] / 5.0,
            self.cmd_hist[2] / 5.0, self.cmd_hist[3] / 5.0
        ], dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one 60 Hz control step with RL damping action."""
        self.step_count += 1
        
        # Clamp action to [-action_max, +action_max]
        action = np.clip(action, -self.action_max, self.action_max)
        delta_wx, delta_wy = float(action[0]), float(action[1])

        # 1. Classical PID nominal command
        err_x = self.target_x - self.center_x
        err_y = self.target_y - self.center_y

        self.integral_x += err_x * self.dt
        self.integral_y += err_y * self.dt
        deriv_x = (err_x - self.prev_error_x) / self.dt
        deriv_y = (err_y - self.prev_error_y) / self.dt

        pid_pan = self.kp * err_x + self.ki * self.integral_x + self.kd * deriv_x
        pid_tilt = self.kp * err_y + self.ki * self.integral_y + self.kd * deriv_y

        # 2. Add RL damping action
        total_pan = pid_pan + delta_wx
        total_tilt = pid_tilt + delta_wy

        # 3. Dynamic PSD turbulence (Dabiri model)
        v = self.drone_speed
        omega = math.sqrt(total_pan ** 2 + total_tilt ** 2)
        sigma_jitter = math.sqrt(2.0 ** 2 + 0.005 * (v ** 2) + 0.15 * (omega ** 2))
        jitter_x = np.random.normal(0, sigma_jitter)
        jitter_y = np.random.normal(0, sigma_jitter)

        # 4. Update kinematics
        self.target_x += (self.target_vx - total_pan * self.deg_to_px) * self.dt + jitter_x
        self.target_y += (self.target_vy - total_tilt * self.deg_to_px) * self.dt + jitter_y

        # Random gentle drone heading turns
        self.target_vx += np.random.normal(0, 15.0) * self.dt
        self.target_vy += np.random.normal(0, 15.0) * self.dt

        # Shift command history
        self.cmd_hist[2:] = self.cmd_hist[:2]
        self.cmd_hist[:2] = [total_pan, total_tilt]
        self.prev_error_x = err_x
        self.prev_error_y = err_y

        # 5. Reward Calculation
        dist_sq = (self.target_x - self.center_x) ** 2 + (self.target_y - self.center_y) ** 2
        act_penalty = 0.02 * (delta_wx ** 2 + delta_wy ** 2)
        reward = -dist_sq / 100.0 - act_penalty

        # Termination check
        out_of_fov = (self.target_x < 0 or self.target_x > self.cam_w or self.target_y < 0 or self.target_y > self.cam_h)
        if out_of_fov:
            reward -= 100.0
            done = True
        else:
            done = (self.step_count >= self.max_steps)

        return self._get_obs(), float(reward), done, {"error_px": math.sqrt(dist_sq)}
