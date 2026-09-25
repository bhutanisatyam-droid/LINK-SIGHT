"""Test circular orbit closed-loop tracking response."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from fsoc.engine import TrackingPipeline
from fsoc.frame_source import MotionModel


def test_circular_orbit_response():
    print("=" * 60)
    print("TESTING CLOSED-LOOP CIRCULAR ORBIT TRACKING")
    print("=" * 60)

    pipe = TrackingPipeline()
    pipe.set_motion_model(MotionModel.CIRCULAR)

    errors = []
    # Run 120 frames at 60 Hz (~2.0 seconds of flight)
    for f in range(120):
        out = pipe.process_frame()
        errors.append(out.tracking_error_px)
        time.sleep(0.016)

    steady_errors = errors[40:] # Evaluate steady-state after initial acquisition
    mean_err = float(np.mean(steady_errors))
    max_err = float(np.max(steady_errors))
    std_err = float(np.std(steady_errors))

    print(f"  -> Steady-State Mean Tracking Error: {mean_err:.2f} px (Target <= 10.0 px)")
    print(f"  -> Steady-State Max Tracking Error:  {max_err:.2f} px")
    print(f"  -> Tracking Error Standard Dev:     {std_err:.2f} px")
    print(f"  -> Lock Retention:                  {out.telemetry.lock_retention_pct:.1f}%")

    assert mean_err < 10.0, f"Mean tracking error {mean_err} exceeded 10px target!"
    assert std_err < 6.0, f"High standard deviation {std_err} indicates limit cycle oscillation!"
    print("\n>>> CIRCULAR ORBIT TRACKING PERFECTLY STABLE <<<")


if __name__ == "__main__":
    test_circular_orbit_response()
