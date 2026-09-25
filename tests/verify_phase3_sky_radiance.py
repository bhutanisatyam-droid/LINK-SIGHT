"""Phase 3 Verification Script: Continuous Sky Radiance & Solar DC Noise.

Tests and verifies:
1. Dynamic background DC offset scaling with sky radiance (0.0 to 1.0).
2. Photon Poisson shot noise scaling with solar intensity.
3. Proper day/night optical contrast behavior.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from fsoc.disturbance import DisturbanceConfig, DisturbanceInjector


def test_phase_3_sky_radiance():
    print("=" * 60)
    print("PHASE 3 VERIFICATION: Sky Radiance & Solar Shot Noise Engine")
    print("=" * 60)

    config = DisturbanceConfig(
        enable_camera_jitter=False,
        enable_salt_pepper=False,
        enable_gaussian=False,
        enable_poisson=False,
        enable_platform_motion=False,
        sky_radiance=0.0,
    )

    injector = DisturbanceInjector(config)
    blank_frame = np.full((480, 640), 14, dtype=np.uint8) # Dark space baseline

    presets = [
        ("Night (Radiance = 0.0)", 0.0, 14.0, 0.0),
        ("Dusk / Dawn (Radiance = 0.3)", 0.3, 69.0, 7.5),
        ("Overcast (Radiance = 0.6)", 0.6, 124.0, 11.5),
        ("High Noon (Radiance = 1.0)", 1.0, 197.0, 14.8),
    ]

    for name, rad, expected_mean, min_std in presets:
        injector.set_sky_radiance(rad)
        out = injector.apply(blank_frame)

        mean_val = float(np.mean(out))
        std_val = float(np.std(out))

        print(f"\n[{name}]")
        print(f"  -> Mean Background Pixel Value: {mean_val:.1f} DN (Expected ~{expected_mean:.1f})")
        print(f"  -> Solar Shot Noise Std Dev:    {std_val:.2f} DN")

        if rad == 0.0:
            assert abs(mean_val - 14.0) < 1.0, "Night baseline shifted incorrectly!"
            assert std_val < 0.1, "Night should have zero solar shot noise"
        else:
            assert abs(mean_val - expected_mean) < 15.0, f"Mean background {mean_val} deviates from expected {expected_mean}"
            assert std_val >= min_std - 3.0, "Poisson shot noise is not scaling with solar photons!"

    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION SUCCESSFUL: Continuous Sky Radiance & Shot Noise Verified!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase_3_sky_radiance()
