"""Unit tests for the Classical CV Beacon Detector.
Tests detection accuracy on clean frames, frames with 10% salt-and-pepper noise,
and frames with Gaussian/atmospheric disturbances.
"""

import unittest
import numpy as np
import cv2

from fsoc.detector import ClassicalBeaconDetector
from fsoc.disturbance import DisturbanceConfig, DisturbanceInjector, AtmosphericCondition


class TestClassicalBeaconDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ClassicalBeaconDetector()

    def _create_synthetic_frame(self, beacon_pos=(320, 240), beacon_size=10):
        frame = np.full((480, 640), 14, dtype=np.uint8)
        bx, by = beacon_pos
        half = beacon_size // 2
        frame[by - half : by + half, bx - half : bx + half] = 255
        return frame

    def test_clean_frame_detection(self):
        frame = self._create_synthetic_frame(beacon_pos=(350, 200), beacon_size=12)
        res = self.detector.detect(frame)

        self.assertTrue(res.detected)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertAlmostEqual(res.centroid[0], 350.0, delta=2.0)
        self.assertAlmostEqual(res.centroid[1], 200.0, delta=2.0)

    def test_salt_pepper_noise_robustness(self):
        """Verify detector reliably detects beacon under 10% salt-and-pepper noise."""
        frame = self._create_synthetic_frame(beacon_pos=(300, 260), beacon_size=10)
        injector = DisturbanceInjector(
            DisturbanceConfig(enable_salt_pepper=True, salt_pepper_ratio=0.10)
        )
        noisy_frame = injector.apply(frame)

        res = self.detector.detect(noisy_frame)
        self.assertTrue(res.detected)
        self.assertGreaterEqual(res.confidence, 0.40)
        self.assertAlmostEqual(res.centroid[0], 300.0, delta=3.5)
        self.assertAlmostEqual(res.centroid[1], 260.0, delta=3.5)

    def test_haze_atmospheric_detection(self):
        """Verify detector operates under atmospheric haze."""
        frame = self._create_synthetic_frame(beacon_pos=(320, 240), beacon_size=14)
        injector = DisturbanceInjector(
            DisturbanceConfig(atmospheric_condition=AtmosphericCondition.HAZE)
        )
        hazy_frame = injector.apply(frame)

        res = self.detector.detect(hazy_frame)
        self.assertTrue(res.detected)
        self.assertAlmostEqual(res.centroid[0], 320.0, delta=2.0)

    def test_empty_black_frame(self):
        """Verify graceful non-detection on empty frame."""
        empty_frame = np.full((480, 640), 14, dtype=np.uint8)
        res = self.detector.detect(empty_frame)
        self.assertFalse(res.detected)
        self.assertEqual(res.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
