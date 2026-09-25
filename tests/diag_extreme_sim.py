"""Simulate exact GUI scenario with extreme noise and 20 deg/s PTZ."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fsoc.disturbance import DisturbanceConfig
from fsoc.engine import TrackingPipeline
from fsoc.frame_source import MotionModel, SimulatorFrameSource


def test_extreme_sim():
    pipe = TrackingPipeline()
    pipe.set_motion_model(MotionModel.CIRCULAR)
    pipe.set_max_ptz_speed(20.0)

    # Inject exact user disturbance settings
    config = DisturbanceConfig(
        sky_radiance=1.0,
        enable_poisson=True,
        poisson_scale=0.8,
        enable_gaussian=True,
        gaussian_sigma=15.0,
        enable_salt_pepper=True,
        salt_pepper_ratio=0.10,
        enable_camera_jitter=True,
    )
    pipe.disturbance_injector.config = config

    print("Running 60 frames under extreme noise...")
    for f in range(1, 61):
        out = pipe.process_frame()
        time.sleep(0.016)
        if f % 10 == 0 or out.status_label.startswith("LOST"):
            print(f"Frame {f:02d}: Status={out.status_label:20s} | Err={out.tracking_error_px:5.1f}px | Det={out.detection.detected} (Conf: {out.detection.confidence:.2f}) | Cam=({out.ground_truth.cam_pos[0]:.0f}, {out.ground_truth.cam_pos[1]:.0f})")


if __name__ == "__main__":
    test_extreme_sim()
