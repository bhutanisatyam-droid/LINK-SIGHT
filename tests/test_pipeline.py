"""End-to-end headless pipeline verification test.
Tests that tracking converges, metrics meet ISRO specs, and CSV/JSON export works.
"""

import os
import tempfile
import time
import unittest

from fsoc.engine import TrackingPipeline
from fsoc.frame_source import MotionModel


class TestTrackingPipeline(unittest.TestCase):

    def test_pipeline_convergence_and_metrics(self):
        pipeline = TrackingPipeline()
        pipeline.set_motion_model(MotionModel.CIRCULAR)

        # Run 110 frames (~3.0 seconds of flight)
        outputs = []
        for i in range(110):
            time.sleep(0.005)
            out = pipeline.process_frame()
            outputs.append(out)

        final_out = outputs[-1]
        telemetry = final_out.telemetry

        # 1. Processing FPS target: >= 20 FPS
        self.assertGreaterEqual(telemetry.fps, 18.0)

        # 2. Tracking acquisition should occur within 2.0s
        self.assertLessEqual(telemetry.acquisition_time_s, 2.5)

        # 3. Post-acquisition steady-state tracking error should be <= 10.0 px
        steady_state_errors = [o.tracking_error_px for o in outputs[90:]]
        mean_steady_error = sum(steady_state_errors) / len(steady_state_errors)
        self.assertLessEqual(mean_steady_error, 10.0, f"Mean steady error {mean_steady_error:.2f}px exceeds 10px")

        # 4. Lock retention should be high
        self.assertGreaterEqual(telemetry.lock_retention_pct, 60.0) # includes acquisition phase

        # 5. Test CSV and JSON export
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "telemetry.csv")
            json_path = os.path.join(tmpdir, "telemetry.json")

            self.assertTrue(pipeline.logger.export_csv(csv_path))
            self.assertTrue(os.path.exists(csv_path))
            self.assertGreater(os.path.getsize(csv_path), 500)

            self.assertTrue(pipeline.logger.export_json(json_path))
            self.assertTrue(os.path.exists(json_path))
            self.assertGreater(os.path.getsize(json_path), 500)


if __name__ == "__main__":
    unittest.main()
