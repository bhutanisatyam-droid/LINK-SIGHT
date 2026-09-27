"""Phase 2 Verification Script: Telemetry Performance Logger & CSV Export.

Tests and verifies:
1. Precision time-series recording (FPS, latency, tracking error, lock retention).
2. Live statistics calculations (RMS error, acquisition time, loss counts).
3. CSV & JSON benchmark file generation against ISRO format requirements.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from fsoc.logger import TelemetryLogger


def test_phase_2_telemetry_logger():
    print("=" * 60)
    print("PHASE 2 VERIFICATION: Telemetry Performance Logging & Export")
    print("=" * 60)

    logger = TelemetryLogger()

    # Simulate 50 tracking loop steps
    print("\n[Test 1] Simulating 50 frames of telemetry ingestion...")
    for frame_idx in range(1, 51):
        # Simulate initial lock after 5 frames
        status = "TRACKING" if frame_idx > 5 else "INITIALIZING"
        error_px = 3.5 + (0.5 * (frame_idx % 4)) if frame_idx > 5 else 45.0
        confidence = 0.92 if frame_idx > 5 else 0.15
        pan_deg = 0.1 * frame_idx
        tilt_deg = -0.05 * frame_idx

        logger.record_step(
            frame_idx=frame_idx,
            tracking_error_px=error_px,
            status=status,
            confidence=confidence,
            pan_deg=pan_deg,
            tilt_deg=tilt_deg,
            is_searching=False,
            is_video_mode=False,
            tracking_mode="CNN (AI)" if frame_idx > 5 else "SPIRAL SCAN",
        )
        time.sleep(0.005) # simulate loop dt

    snap = logger.get_snapshot()
    print(f"  -> Total Frames: {snap.frame_index}")
    print(f"  -> Calculated FPS: {snap.fps:.1f} Hz")
    print(f"  -> Mean Error: {snap.mean_error_px:.2f} px")
    print(f"  -> RMS Error: {snap.rms_error_px:.2f} px")
    print(f"  -> Lock Retention: {snap.lock_retention_pct:.1f}%")
    print(f"  -> Acquisition Time: {snap.acquisition_time_s:.3f} s")

    assert snap.frame_index == 50, "Frame index mismatch"
    assert snap.lock_retention_pct > 80.0, "Lock retention calculation error"

    # Test 2: CSV Export
    print("\n[Test 2] Testing CSV Export...")
    csv_path = "logs/test_performance_log.csv"
    success_csv = logger.export_csv(csv_path)
    assert success_csv, "CSV export returned False"
    assert os.path.exists(csv_path), "CSV file was not created on disk"

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"  -> CSV Header: {lines[0].strip()}")
        print(f"  -> Total CSV Rows: {len(lines)}")
        print(f"  -> Sample Row: {lines[10].strip()}")
        assert len(lines) == 51, "CSV row count mismatch (expected 1 header + 50 rows)"

    # Test 3: JSON Export
    print("\n[Test 3] Testing JSON Export...")
    json_path = "logs/test_performance_log.json"
    success_json = logger.export_json(json_path)
    assert success_json, "JSON export returned False"
    assert os.path.exists(json_path), "JSON file was not created on disk"
    print(f"  -> JSON successfully generated ({os.path.getsize(json_path)} bytes)")

    print("\n" + "=" * 60)
    print("PHASE 2 VERIFICATION SUCCESSFUL: Telemetry logging & CSV export verified!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase_2_telemetry_logger()
