"""Phase 1 Verification Script: Dynamic PSD Jitter Engine.

Tests and verifies:
1. JitterMode.STEADY vs JitterMode.DYNAMIC_PSD.
2. Aerodynamic Dabiri scaling with flight velocity (v) and gimbal slew rate (omega).
3. DisturbanceInjector frame distortion output.
"""

import os
import sys

# Ensure FSOC root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import cv2
import numpy as np
from fsoc.disturbance import DisturbanceConfig, DisturbanceInjector, JitterMode


def test_phase_1_dynamic_jitter():
    print("=" * 60)
    print("PHASE 1 VERIFICATION: Dynamic PSD Jitter Engine")
    print("=" * 60)

    # Base configuration with only camera jitter enabled
    config = DisturbanceConfig(
        enable_camera_jitter=True,
        camera_jitter_max_px=20.0,
        enable_salt_pepper=False,
        enable_gaussian=False,
        enable_poisson=False,
        enable_platform_motion=False,
        sky_radiance=0.0,
    )

    injector = DisturbanceInjector(config)

    # 1. Test Steady Mode (Legacy fixed uniform noise)
    config.jitter_mode = JitterMode.STEADY
    print("\n[Test 1] Testing JitterMode.STEADY:")
    # Calculate sample jitter displacement
    displacements_steady = []
    dummy_frame = np.zeros((480, 640), dtype=np.uint8)
    dummy_frame[240, 320] = 255 # Point source at center

    for _ in range(100):
        # Even with high velocity, steady mode should remain fixed amplitude
        injector.set_flight_dynamics(velocity_px_s=500.0, omega_deg_s=20.0)
        out = injector.apply(dummy_frame)
        # Measure sub-pixel centroid shift using moments
        # Subtract background border value (14)
        fg = np.clip(out.astype(np.float32) - 14.0, 0, 255)
        M = cv2.moments(fg)
        if M["m00"] > 1e-3:
            cx = float(M["m10"] / M["m00"]) - 320.0
            cy = float(M["m01"] / M["m00"]) - 240.0
            displacements_steady.append(math.sqrt(cx**2 + cy**2))

    mean_disp_steady = np.mean(displacements_steady) if displacements_steady else 0.0
    print(f"  -> Steady Mode Mean Displacement: {mean_disp_steady:.2f} px (Uniform +/-20px -> theoretical ~15.3px)")

    # 2. Test Dynamic PSD Mode (Dabiri Aerodynamic Model)
    config.jitter_mode = JitterMode.DYNAMIC_PSD
    print("\n[Test 2] Testing JitterMode.DYNAMIC_PSD (Aerodynamic Scaling):")

    test_cases = [
        ("Hover (v=0 px/s, omega=0 deg/s)", 0.0, 0.0),
        ("Cruising (v=150 px/s, omega=0 deg/s)", 150.0, 0.0),
        ("High-Speed Dash (v=350 px/s, omega=0 deg/s)", 350.0, 0.0),
        ("Hard Gimbal Slew (v=0 px/s, omega=15 deg/s)", 0.0, 15.0),
        ("High-G Turn + Dash (v=350 px/s, omega=15 deg/s)", 350.0, 15.0),
    ]

    for name, v, omega in test_cases:
        injector.set_flight_dynamics(velocity_px_s=v, omega_deg_s=omega)
        s_user = config.camera_jitter_max_px / 20.0
        expected_sigma = s_user * math.sqrt(
            config.sigma_base**2 + config.kv_drag * (v**2) + config.k_omega * (omega**2)
        )

        # Measure empirical shift across 300 samples
        shifts = []
        for _ in range(300):
            out = injector.apply(dummy_frame)
            fg = np.clip(out.astype(np.float32) - 14.0, 0, 255)
            M = cv2.moments(fg)
            if M["m00"] > 1e-3:
                cx = float(M["m10"] / M["m00"]) - 320.0
                shifts.append(cx)

        measured_std = np.std(shifts) if shifts else 0.0
        print(f"  [{name}]")
        print(f"     Theoretical 1-Sigma: {expected_sigma:.2f} px")
        print(f"     Measured Empirical Std: {measured_std:.2f} px")
        assert abs(measured_std - expected_sigma) < 2.5, "Empirical jitter deviates from Dabiri formula!"

    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION SUCCESSFUL: All Dabiri aerodynamic scaling tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase_1_dynamic_jitter()
