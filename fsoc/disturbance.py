"""Disturbance Injector module for FSOC tracking simulation.

Injects physical space/atmospheric channel disturbances into the camera frame:
- Salt-and-pepper noise (~10% default)
- Gaussian thermal noise
- Poisson shot noise
- High-frequency camera jitter (+/-20px/frame)
- Low-frequency platform motion drift (+/-20px/frame)
- Atmospheric channel effects: Clear, Haze, Fog, Rain, Low-Light
"""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Tuple

import cv2
import numpy as np


class JitterMode(Enum):
    """Camera structural jitter models."""
    STEADY = "steady"              # Fixed-amplitude uniform/Gaussian noise
    DYNAMIC_PSD = "dynamic_psd"    # Velocity & slew-coupled aerodynamic model (Dabiri et al., IEEE JSAC 2018)


class AtmosphericCondition(Enum):
    """Atmospheric transmission models."""
    CLEAR = "clear"
    HAZE = "haze"
    FOG = "fog"
    RAIN = "rain"
    LOW_LIGHT = "low_light"


@dataclass
class DisturbanceConfig:
    """Configurable parameters for the disturbance injection engine."""
    enable_salt_pepper: bool = True
    salt_pepper_ratio: float = 0.10     # 0.0 to 0.30 (10% default)

    enable_gaussian: bool = False
    gaussian_sigma: float = 15.0        # 0.0 to 50.0

    enable_poisson: bool = False
    poisson_scale: float = 0.8          # Shot noise scale

    enable_camera_jitter: bool = False
    camera_jitter_max_px: float = 20.0  # +/- 20px default (master amplitude/scale)
    jitter_mode: JitterMode = JitterMode.DYNAMIC_PSD  # Steady vs. Dynamic PSD
    kv_drag: float = 0.005              # Aerodynamic dynamic pressure coefficient
    k_omega: float = 0.15               # Gimbal slew rate reaction coefficient
    sigma_base: float = 2.0             # Baseline hover micro-vibration in px

    enable_platform_motion: bool = False
    platform_motion_max_px: float = 20.0
    platform_motion_mode: str = "linear" # "linear", "circular", "random"

    # Ambient Sky Radiance / Day-Night Simulation (0.0=Night, 0.3=Dusk, 0.6=Overcast, 1.0=Noon)
    sky_radiance: float = 0.0

    atmospheric_condition: AtmosphericCondition = AtmosphericCondition.CLEAR


class DisturbanceInjector:
    """Applies realistic optical, sensor, mechanical, and channel noise to simulated frames."""

    def __init__(self, config: DisturbanceConfig = None):
        self.config = config or DisturbanceConfig()
        self.t_start = time.perf_counter()

        # Dynamic flight feedback variables (Dabiri model)
        self.current_velocity: float = 0.0 # Instantaneous target velocity in px/s
        self.current_omega: float = 0.0    # Commanded PTZ slew rate in deg/s

        # Platform motion internal state
        self.platform_phase_x: float = 0.0
        self.platform_phase_y: float = 0.0
        self.platform_rand_x: float = 0.0
        self.platform_rand_y: float = 0.0

        # Precomputed rain streaks for speed
        self._rain_streak_count = 45

    def set_flight_dynamics(self, velocity_px_s: float, omega_deg_s: float) -> None:
        """Update instantaneous target velocity and gimbal slew rate for Dynamic PSD jitter."""
        self.current_velocity = velocity_px_s
        self.current_omega = omega_deg_s

    def set_sky_radiance(self, radiance: float) -> None:
        """Set ambient sky background radiance (0.0=Night, 0.3=Dusk, 0.6=Overcast, 1.0=High Noon)."""
        self.config.sky_radiance = max(0.0, min(1.0, float(radiance)))

    def update_config(self, **kwargs) -> None:
        """Dynamically update any disturbance parameters."""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all active disturbances in physical sequence:
        1. Ambient sky background radiance (Day/Night simulation)
        2. Atmospheric channel attenuation/dispersion
        3. Mechanical motion (platform drift + high-freq jitter)
        4. Sensor noise (Poisson shot noise, Gaussian thermal, Salt-and-Pepper dead/hot pixels)
        """
        output = frame.copy()
        h, w = int(output.shape[0]), int(output.shape[1])
        t = time.perf_counter() - self.t_start

        # --- 0. Ambient Sky Radiance (Day / Night Simulation) ---
        if self.config.sky_radiance > 0.005:
            # Scale background DC luminance floor (up to ~180 DN at 1.0 radiance)
            bg_offset = int(255.0 * self.config.sky_radiance * 0.72)
            if bg_offset > 0:
                output = cv2.add(output, np.full((h, w), bg_offset, dtype=np.uint8))
                # Inject solar photon arrival shot noise proportional to sqrt(bg_offset)
                solar_shot_sigma = math.sqrt(float(bg_offset)) * 1.1
                solar_noise = np.random.normal(0, solar_shot_sigma, (h, w))
                output = np.clip(output.astype(np.float32) + solar_noise, 0, 255).astype(np.uint8)

        # --- 1. Atmospheric Channel Effects ---
        cond = self.config.atmospheric_condition
        if cond == AtmosphericCondition.HAZE:
            # Koschmieder scattering: attenuation + airlight glow
            # I_out = I_in * 0.60 + 50
            output = cv2.convertScaleAbs(output, alpha=0.60, beta=50)

        elif cond == AtmosphericCondition.FOG:
            # Dense aerosol scattering: severe contrast attenuation + atmospheric glow
            output = cv2.convertScaleAbs(output, alpha=0.55, beta=45)
            output = cv2.GaussianBlur(output, (3, 3), sigmaX=0.9)

        elif cond == AtmosphericCondition.RAIN:
            # Rain attenuation + dynamic directional streaks
            output = cv2.convertScaleAbs(output, alpha=0.72, beta=15)
            # Draw pseudo-random rain streaks
            streak_img = np.zeros((h, w), dtype=np.uint8)
            np.random.seed(int(t * 100) % 10000)
            xs = np.random.randint(0, w, size=self._rain_streak_count)
            ys = np.random.randint(0, h, size=self._rain_streak_count)
            lengths = np.random.randint(15, 30, size=self._rain_streak_count)
            for x, y, l in zip(xs, ys, lengths):
                x2 = int(x + l * 0.35)
                y2 = int(y + l)
                cv2.line(streak_img, (x, y), (x2, y2), 35, 1)
            output = cv2.add(output, streak_img)

        elif cond == AtmosphericCondition.LOW_LIGHT:
            # Photon starvation regime (link path loss margin)
            output = cv2.convertScaleAbs(output, alpha=0.48, beta=6)

        # --- 2. Mechanical Motion (Jitter & Platform Drift) ---
        dx, dy = 0.0, 0.0

        if self.config.enable_platform_motion:
            p_mode = self.config.platform_motion_mode
            p_amp = self.config.platform_motion_max_px
            if p_mode == "linear":
                # Low-frequency linear sweep
                dx += p_amp * math.sin(0.8 * t)
                dy += (p_amp * 0.6) * math.cos(0.5 * t)
            elif p_mode == "circular":
                dx += p_amp * math.cos(1.2 * t)
                dy += p_amp * math.sin(1.2 * t)
            elif p_mode == "random":
                self.platform_rand_x += np.random.normal(0, 1.5)
                self.platform_rand_y += np.random.normal(0, 1.5)
                self.platform_rand_x = np.clip(self.platform_rand_x, -p_amp, p_amp)
                self.platform_rand_y = np.clip(self.platform_rand_y, -p_amp, p_amp)
                dx += self.platform_rand_x
                dy += self.platform_rand_y

        if self.config.enable_camera_jitter:
            if self.config.jitter_mode == JitterMode.STEADY:
                # Steady Jitter Mode (Legacy/Fixed Uniform)
                j_amp = self.config.camera_jitter_max_px
                dx += np.random.uniform(-j_amp, j_amp)
                dy += np.random.uniform(-j_amp, j_amp)
            else:
                # Dynamic PSD Jitter Mode (Dabiri et al., IEEE JSAC 2018)
                # Master severity scale factor S_user from user slider
                s_user = self.config.camera_jitter_max_px / 20.0
                v = self.current_velocity
                omega_val = self.current_omega
                sigma = s_user * math.sqrt(
                    self.config.sigma_base ** 2 +
                    self.config.kv_drag * (v ** 2) +
                    self.config.k_omega * (omega_val ** 2)
                )
                dx += np.random.normal(0, sigma)
                dy += np.random.normal(0, sigma)

        if abs(dx) > 0.1 or abs(dy) > 0.1:
            # Affine translation for frame shift with border replication (avoids artificial step discontinuities)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            output = cv2.warpAffine(output, M, (int(w), int(h)), borderMode=cv2.BORDER_REPLICATE)

        # --- 3. Sensor Noise Injection ---
        if self.config.enable_poisson:
            # Physical sensor photon shot noise: variance proportional to local intensity sqrt(I)
            shot_sigma = np.sqrt(np.maximum(output.astype(np.float32), 1.0)) * (self.config.poisson_scale * 0.8)
            shot_noise = np.random.normal(0, shot_sigma)
            output = np.clip(output.astype(np.float32) + shot_noise, 0, 255).astype(np.uint8)

        if self.config.enable_gaussian:
            # Thermal readout noise
            sigma = self.config.gaussian_sigma
            gauss = np.random.normal(0, sigma, (h, w))
            noisy = output.astype(np.float32) + gauss
            output = np.clip(noisy, 0, 255).astype(np.uint8)

        if self.config.enable_salt_pepper:
            # Impulsive sensor defects (dead/hot pixels)
            ratio = self.config.salt_pepper_ratio
            if ratio > 0:
                rand_mask = np.random.uniform(0.0, 1.0, size=(h, w))
                half_r = ratio / 2.0
                output[rand_mask < half_r] = 0        # Pepper (dead pixels)
                output[rand_mask > 1.0 - half_r] = 255 # Salt (hot pixels)

        return output
