"""Unit tests for CutHexagonalSpiralSearch re-acquisition engine."""

import math
import unittest

from fsoc.reacquisition import CutHexagonalSpiralSearch


class TestCutHexagonalSpiralSearch(unittest.TestCase):

    def setUp(self):
        self.search = CutHexagonalSpiralSearch(step_size_deg=0.5, confidence_threshold=0.60)

    def test_cut_hexagonal_geometry(self):
        """Verify all generated waypoints lie strictly within the cut uncertainty radius."""
        center = (0.0, 0.0)
        uncertainty = 1.5
        wps = self.search.generate_angular_hex_pattern(center, uncertainty)

        self.assertGreater(len(wps), 5)
        # Center is first point
        self.assertEqual(wps[0], center)

        # Verify distance constraint
        for x, y in wps:
            dist = math.hypot(x - center[0], y - center[1])
            self.assertLessEqual(dist, uncertainty + 1e-4)

    def test_early_termination_on_detection(self):
        """Verify search immediately aborts when detection confidence crosses threshold."""
        self.search.start_search(center_deg=(0.0, 0.0), max_radius_deg=1.5)
        self.assertTrue(self.search.is_active)

        # Low confidence: search keeps advancing
        wp = self.search.step(detection_confidence=0.30, cam_pan_deg=0.0, cam_tilt_deg=0.0)
        self.assertTrue(self.search.is_active)
        self.assertIsNotNone(wp)

        # High confidence (beacon re-enters sensor window): search halts after consecutive confirmations
        _ = self.search.step(detection_confidence=0.85, cam_pan_deg=0.0, cam_tilt_deg=0.0)
        wp_end = self.search.step(detection_confidence=0.85, cam_pan_deg=0.0, cam_tilt_deg=0.0)
        self.assertFalse(self.search.is_active)
        self.assertIsNone(wp_end)


if __name__ == "__main__":
    unittest.main()
