"""
LinkSight FSOC ATP - Headless Benchmark Data Generator v3
Runs 3 fully-instrumented simulation scenarios with rate-limited timing
to match real GUI behavior and writes report_data.json.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fsoc.engine import TrackingPipeline
from fsoc.disturbance import AtmosphericCondition, JitterMode
from fsoc.frame_source import MotionModel

TARGET_FPS = 30.0
FRAME_DT = 1.0 / TARGET_FPS

SCENARIOS = [
    {
        "id": "circular_clear",
        "label": "Scenario A - Circular Orbit / Clear Atmosphere (Baseline Nominal)",
        "motion": MotionModel.CIRCULAR,
        "atmo": AtmosphericCondition.CLEAR,
        "salt_pepper": True,
        "sp_ratio": 0.05,
        "gaussian": False,
        "jitter": False,
        "platform": False,
        "sky_radiance": 0.05,
        "beam_break_at": 200,
        "frames": 600,
    },
    {
        "id": "figure8_fog",
        "label": "Scenario B - Figure-8 / Dense Fog + Camera Jitter (Stress Test)",
        "motion": MotionModel.FIGURE_8,
        "atmo": AtmosphericCondition.FOG,
        "salt_pepper": True,
        "sp_ratio": 0.10,
        "gaussian": True,
        "jitter": True,
        "platform": False,
        "sky_radiance": 0.30,
        "beam_break_at": 300,
        "frames": 600,
    },
    {
        "id": "random_rain",
        "label": "Scenario C - Random Walk / Dynamic Rain (Edge Case)",
        "motion": MotionModel.RANDOM,
        "atmo": AtmosphericCondition.RAIN,
        "salt_pepper": True,
        "sp_ratio": 0.08,
        "gaussian": False,
        "jitter": True,
        "platform": True,
        "sky_radiance": 0.20,
        "beam_break_at": 280,
        "frames": 600,
    },
]


def run_scenario(cfg):
    print("\n" + "=" * 60)
    print("  " + cfg["label"])
    print("=" * 60)

    pipeline = TrackingPipeline()

    d = pipeline.disturbance_injector.config
    d.enable_salt_pepper = cfg["salt_pepper"]
    d.salt_pepper_ratio = cfg["sp_ratio"]
    d.enable_gaussian = cfg["gaussian"]
    d.gaussian_sigma = 12.0
    d.enable_poisson = False
    d.enable_camera_jitter = cfg["jitter"]
    d.camera_jitter_max_px = 8.0
    d.jitter_mode = JitterMode.DYNAMIC_PSD
    d.enable_platform_motion = cfg["platform"]
    d.platform_motion_max_px = 6.0
    d.atmospheric_condition = cfg["atmo"]
    d.sky_radiance = cfg["sky_radiance"]

    pipeline.set_motion_model(cfg["motion"])

    times_s = []
    errors_px = []
    statuses = []
    fps_series = []
    confidences = []
    first_lock_t = None
    beam_break_done = False

    t0 = time.perf_counter()
    next_frame_t = t0

    for frame_idx in range(cfg["frames"]):
        # Rate-limit to TARGET_FPS so gimbal slew simulation is realistic
        now = time.perf_counter()
        sleep_needed = next_frame_t - now
        if sleep_needed > 0:
            time.sleep(sleep_needed)
        next_frame_t = time.perf_counter() + FRAME_DT

        if frame_idx == cfg["beam_break_at"] and not beam_break_done:
            pipeline.trigger_occlusion(duration_frames=18)
            beam_break_done = True
            print("  [Frame %03d] Beam break injected." % frame_idx)

        out = pipeline.process_frame()

        if out.track.status.value == "TRACKING" and first_lock_t is None:
            first_lock_t = time.perf_counter() - t0
            print("  [Frame %03d] First LOCK in %.3fs" % (frame_idx, first_lock_t))

        t_now = time.perf_counter() - t0
        times_s.append(round(t_now, 4))
        errors_px.append(round(out.tracking_error_px, 3))
        statuses.append(out.status_label)
        fps_series.append(round(out.telemetry.fps, 2))
        confidences.append(round(out.detection.confidence, 3))

        if frame_idx % 100 == 0:
            print("  Frame %03d | Err=%.1fpx | FPS=%.1f | %s" % (
                frame_idx, out.tracking_error_px, out.telemetry.fps, out.status_label))

    snap = pipeline.logger.get_snapshot()

    locked_frames = sum(1 for s in statuses if "TRACKING" in s)
    degraded_frames = sum(1 for s in statuses if "DEGRADED" in s)
    lost_frames = sum(1 for s in statuses if "LOST" in s)
    n = len(errors_px)

    mean_err = sum(errors_px) / max(1, n)
    rms_err = math.sqrt(sum(e**2 for e in errors_px) / max(1, n))
    max_err = max(errors_px) if errors_px else 0.0
    mean_fps = sum(fps_series) / max(1, len(fps_series))
    lock_pct = 100.0 * locked_frames / max(1, n)

    tracking_errs = [errors_px[i] for i, s in enumerate(statuses) if "TRACKING" in s]
    steady_mean = sum(tracking_errs) / max(1, len(tracking_errs)) if tracking_errs else 0.0
    steady_rms = math.sqrt(sum(e**2 for e in tracking_errs) / max(1, len(tracking_errs))) if tracking_errs else 0.0

    result = {
        "id": cfg["id"],
        "label": cfg["label"],
        "frames": n,
        "first_acquisition_s": round(first_lock_t, 4) if first_lock_t else None,
        "reacquisition_s": round(snap.reacquisition_time_s, 4),
        "mean_error_px": round(mean_err, 3),
        "rms_error_px": round(rms_err, 3),
        "max_error_px": round(max_err, 3),
        "steady_state_mean_px": round(steady_mean, 3),
        "steady_state_rms_px": round(steady_rms, 3),
        "mean_fps": round(mean_fps, 2),
        "lock_retention_pct": round(lock_pct, 2),
        "loss_event_count": snap.target_loss_count,
        "locked_frames": locked_frames,
        "degraded_frames": degraded_frames,
        "lost_frames": lost_frames,
        "times_s": times_s,
        "errors_px": errors_px,
        "statuses": statuses,
        "fps_series": fps_series,
        "confidences": confidences,
    }

    acq = result["first_acquisition_s"]
    print("\n  RESULTS:")
    print("  Acquisition Time: %s s   (spec <=2.0s)" % (("%.3f" % acq) if acq else "N/A"))
    print("  Re-Acquisition:   %.3f s   (spec <=1.0s)" % result["reacquisition_s"])
    print("  Steady-State Err: %.2f px mean / %.2f px RMS   (spec <=10px)" % (
        result["steady_state_mean_px"], result["steady_state_rms_px"]))
    print("  Lock Retention:   %.1f%%   (spec >95%%)" % result["lock_retention_pct"])
    print("  Mean Loop Rate:   %.1f Hz   (spec >=20Hz)" % result["mean_fps"])
    return result


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report_data.json")

    all_results = []
    for cfg in SCENARIOS:
        r = run_scenario(cfg)
        all_results.append(r)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("  Data written -> " + out_path)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
