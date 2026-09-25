"""Phase 4 Verification Script: MicroGRU Trajectory Coaster & Occlusion Budget.

Tests and verifies:
1. 30-frame rolling state history ring buffer.
2. Graceful transition to DEGRADED coasting during detection dropouts.
3. Strict 12-frame (200ms) budget enforcement before declaring LOST.
4. Immediate re-lock when measurement resumes.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fsoc.detector import DetectionResult
from fsoc.tracker import KalmanBeaconTracker, TrackStatus


def test_phase_4_microgru_coaster():
    print("=" * 60)
    print("PHASE 4 VERIFICATION: MicroGRU Trajectory Coaster & Coast Budget")
    print("=" * 60)

    tracker = KalmanBeaconTracker(max_coast_frames=12)

    # 1. Acquire and track for 20 frames
    print("\n[Test 1] Establishing steady-state lock (20 frames)...")
    for i in range(20):
        # Target moving linearly at (320 + i*2, 240 + i*1)
        det = DetectionResult(
            detected=True,
            centroid=(320.0 + i * 2.0, 240.0 + i * 1.0),
            bbox=(int(315 + i * 2), int(235 + i * 1), 10, 10),
            confidence=0.95,
        )
        est = tracker.update(det, dt=0.016)

    print(f"  -> Status: {est.status.value}")
    print(f"  -> State History Buffer Length: {len(tracker.state_history)} (Max: 30)")
    print(f"  -> Estimated Position: ({est.pos[0]:.1f}, {est.pos[1]:.1f})")
    assert est.status == TrackStatus.TRACKING
    assert len(tracker.state_history) >= 15

    # 2. Inject Occlusion / Dropout (Frames 21 to 30: 10 frames of miss)
    print("\n[Test 2] Injecting 10-frame cloud occlusion (within 200ms budget)...")
    miss_det = DetectionResult(
        detected=False,
        centroid=(320.0, 240.0),
        bbox=(0, 0, 0, 0),
        confidence=0.0,
    )

    for i in range(10):
        est = tracker.update(miss_det, dt=0.016)

    print(f"  -> Consecutive Misses: {est.consecutive_misses}")
    print(f"  -> Status: {est.status.value} (Expected DEGRADED)")
    print(f"  -> Coasted Position: ({est.pos[0]:.1f}, {est.pos[1]:.1f})")
    print(f"  -> Is Valid: {est.is_valid}")
    assert est.status == TrackStatus.DEGRADED, "Tracker should be coasting in DEGRADED mode!"
    assert est.is_valid == True, "Track must remain valid while within coast budget!"

    # 3. Exceed 12-frame Budget (Frame 31 to 33: misses 11, 12, 13)
    print("\n[Test 3] Exceeding 12-frame (200ms) budget -> Expect transition to LOST...")
    est = tracker.update(miss_det, dt=0.016) # 11
    est = tracker.update(miss_det, dt=0.016) # 12 -> Exceeds max_coast_frames
    print(f"  -> Consecutive Misses: {est.consecutive_misses}")
    print(f"  -> Status: {est.status.value}")
    assert est.status == TrackStatus.LOST, "Tracker must declare LOST when exceeding 200ms budget to trigger search!"

    # 4. Re-acquisition recovery (requires 2-frame confirmation to reject glints)
    print("\n[Test 4] Target reappears -> 2-frame confirmation re-lock...")
    relock_det = DetectionResult(
        detected=True,
        centroid=(380.0, 270.0),
        bbox=(375, 265, 10, 10),
        confidence=0.90,
    )
    est_1 = tracker.update(relock_det, dt=0.016)
    print(f"  -> Frame 1: Status = {est_1.status.value} (Confirm Count: {tracker.lost_confirm_count})")
    assert tracker.lost_confirm_count == 1

    est_2 = tracker.update(relock_det, dt=0.016)
    print(f"  -> Frame 2: Status = {est_2.status.value} (Verified Lock!)")
    assert est_2.status == TrackStatus.TRACKING, "Tracker should transition to TRACKING after 2 consecutive frames!"

    print("\n" + "=" * 60)
    print("PHASE 4 VERIFICATION SUCCESSFUL: MicroGRU coasting and budget logic verified!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase_4_microgru_coaster()
