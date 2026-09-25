"""Diagnostic script for extreme noise detector response."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from fsoc.detector import ClassicalBeaconDetector, SpatiotemporalBeaconDetector
from fsoc.disturbance import DisturbanceConfig, DisturbanceInjector


def test_extreme_noise():
    config = DisturbanceConfig(
        sky_radiance=1.0,           # High Noon
        enable_poisson=True,        # Poisson shot noise
        poisson_scale=0.8,
        enable_gaussian=True,       # Thermal noise
        gaussian_sigma=15.0,
        enable_salt_pepper=True,    # 10% S&P
        salt_pepper_ratio=0.10,
        enable_camera_jitter=True,
    )
    injector = DisturbanceInjector(config)

    # Frame with beacon at center (320, 240)
    frame = np.full((480, 640), 14, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 5, 255, -1)

    noisy_frame = injector.apply(frame)

    classical = ClassicalBeaconDetector()
    res_c = classical.detect(noisy_frame)

    spatio = SpatiotemporalBeaconDetector()
    res_s = spatio.detect(noisy_frame)

    print(f"Classical Detection: {res_c.detected} | Centroid: ({res_c.centroid[0]:.1f}, {res_c.centroid[1]:.1f}) | Conf: {res_c.confidence:.3f}")
    print(f"Spatio Detection:    {res_s.detected} | Centroid: ({res_s.centroid[0]:.1f}, {res_s.centroid[1]:.1f}) | Conf: {res_s.confidence:.3f}")


if __name__ == "__main__":
    test_extreme_noise()
