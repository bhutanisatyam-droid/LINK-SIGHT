"""Unit tests for PTZPIDController."""

import unittest

from fsoc.controller import PIDGains, PTZPIDController


class TestPTZPIDController(unittest.TestCase):

    def setUp(self):
        self.controller = PTZPIDController(
            gains=PIDGains(kp=5.0, ki=1.0, kd=0.1, i_max=4.0, deadband_px=1.0),
            max_speed_deg_s=8.0,
        )

    def test_centered_deadband(self):
        """Zero error inside deadband produces 0 rate command."""
        cmd_x, cmd_y = self.controller.compute_command((320.0, 240.0), dt=0.033)
        self.assertEqual(cmd_x, 0.0)
        self.assertEqual(cmd_y, 0.0)

    def test_directional_restoring_rate(self):
        """Target to right (+x) produces positive pan command; below (+y) produces positive tilt."""
        cmd_x, cmd_y = self.controller.compute_command((360.0, 280.0), dt=0.033)
        self.assertGreater(cmd_x, 0.0)
        self.assertGreater(cmd_y, 0.0)

    def test_slew_rate_saturation(self):
        """Large step error is clamped strictly to max_speed_deg_s."""
        cmd_x, cmd_y = self.controller.compute_command((600.0, 400.0), dt=0.033)
        self.assertLessEqual(abs(cmd_x), 8.0)
        self.assertLessEqual(abs(cmd_y), 8.0)

    def test_anti_windup_clamping(self):
        """Integrator does not wind up past i_max."""
        for _ in range(50):
            self.controller.compute_command((400.0, 240.0), dt=0.033)
        max_i_deg = self.controller.gains.i_max / self.controller.gains.ki
        self.assertLessEqual(abs(self.controller.integral_x), max_i_deg + 1e-4)


if __name__ == "__main__":
    unittest.main()
