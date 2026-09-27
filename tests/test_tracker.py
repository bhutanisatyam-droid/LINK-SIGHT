"""Unit tests for KalmanBeaconTracker.
Tests state initialization, measurement fusion, coasting through occlusions,
and sustained loss status transitions.
"""

import unittest
import numpy as np

from fsoc.detector import DetectionResult
from fsoc.tracker import KalmanBeaconTracker, TrackStatus


class TestKalmanBeaconTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = KalmanBeaconTracker(max_coast_frames=8)

    def _make_detection(self, x, y, conf=0.95):
        return DetectionResult(
            detected=True,
            centroid=(x, y),
            bbox=(int(x - 5), int(y - 5), 10, 10),
            confidence=conf,
        )

    def _make_empty_detection(self):
        return DetectionResult(
            detected=False,
            centroid=(320.0, 240.0),
            bbox=(0, 0, 0, 0),
            confidence=0.0,
        )

    def test_initialization_and_lock(self):
        # Frame 1: Tentative first hit
        det1 = self._make_detection(330.0, 250.0, conf=0.90)
        track1 = self.tracker.update(det1, dt=0.033)
        self.assertEqual(track1.status, TrackStatus.INITIALIZING)

        # Frame 2: Confirmed track (2-hit M-out-of-N confirmation)
        det2 = self._make_detection(330.0, 250.0, conf=0.90)
        track2 = self.tracker.update(det2, dt=0.033)
        self.assertEqual(track2.status, TrackStatus.TRACKING)
        self.assertTrue(track2.is_valid)
        self.assertAlmostEqual(track2.pos[0], 330.0, delta=1.0)
        self.assertAlmostEqual(track2.pos[1], 250.0, delta=1.0)

    def test_occlusion_coasting(self):
        """Verify tracker coasts on prediction for brief dropouts without declaring loss."""
        # 1. Acquire and track for 5 frames
        for i in range(5):
            det = self._make_detection(300.0 + i * 2.0, 240.0, conf=0.95)
            track = self.tracker.update(det, dt=0.033)

        self.assertEqual(track.status, TrackStatus.TRACKING)

        # 2. Simulate 3-frame dropout (occlusion/gap)
        for i in range(3):
            empty_det = self._make_empty_detection()
            track = self.tracker.update(empty_det, dt=0.033)
            # Should coast in DEGRADED status, NOT LOST
            self.assertEqual(track.status, TrackStatus.DEGRADED)
            self.assertTrue(track.is_valid)

        # 3. Target re-appears
        resume_det = self._make_detection(318.0, 240.0, conf=0.90)
        track = self.tracker.update(resume_det, dt=0.033)
        self.assertEqual(track.status, TrackStatus.TRACKING)

    def test_sustained_loss_transition(self):
        """Verify sustained dropout (>8 frames) transitions track to LOST."""
        # Acquire with 2-hit confirmation
        self.tracker.update(self._make_detection(320.0, 240.0, conf=0.95), dt=0.033)
        self.tracker.update(self._make_detection(320.0, 240.0, conf=0.95), dt=0.033)

        # Drop for 15 consecutive frames (exceeding max_coast_frames=8)
        for _ in range(15):
            track = self.tracker.update(self._make_empty_detection(), dt=0.033)

        self.assertEqual(track.status, TrackStatus.LOST)
        self.assertFalse(track.is_valid)

    def test_uncertainty_ellipse_growth(self):
        """Verify positional uncertainty covariance grows during occlusion."""
        # Acquire
        t0 = self.tracker.update(self._make_detection(320.0, 240.0, conf=0.95), dt=0.033)
        uncert_locked = t0.pos_uncertainty_px

        # Miss 4 frames
        for _ in range(4):
            t_coast = self.tracker.update(self._make_empty_detection(), dt=0.033)

        self.assertGreater(t_coast.pos_uncertainty_px, uncert_locked)
        self.assertGreater(t_coast.covariance_ellipse[0], 0.0)


if __name__ == "__main__":
    unittest.main()
