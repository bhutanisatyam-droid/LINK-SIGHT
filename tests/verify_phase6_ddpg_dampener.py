"""Phase 6 Verification Script: TinyDDPG Actuation Dampener (Core 3 AI).

Tests and verifies:
1. Sleep mode during nominal tracking (error <= 10px).
2. Wake-on-Degradation trigger on error spikes / wind shear (error >= 11px).
3. Non-linear lead rate damping execution.
4. Ultra-low latency execution (<15ms budget).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from fsoc.controller import PTZPIDController, TinyDDPGDampener


def test_phase_6_ddpg_dampener():
    print("=" * 60)
    print("PHASE 6 VERIFICATION: TinyDDPG Actuation Dampener (Core 3 AI)")
    print("=" * 60)

    controller = PTZPIDController()

    # -------------------------------------------------------------
    # Test 1: Sleep Mode during Nominal Locked Tracking (Error = 3.0px)
    # -------------------------------------------------------------
    print("\n[Test 1] Testing Nominal Tracking (Error = 3.0 px, Nominal Flight)...")
    cmd_x, cmd_y = controller.compute_command(target_pos=(323.0, 240.0), dt=0.016)

    print(f"  -> Commanded Rate: ({cmd_x:.2f}, {cmd_y:.2f}) deg/s")
    print(f"  -> DDPG Dampener Active: {controller.is_ai_dampener_active}")
    assert controller.is_ai_dampener_active == False, "DDPG dampener must sleep during nominal tracking (error <= 10px)!"

    # -------------------------------------------------------------
    # Test 2: Wake-on-Degradation Trigger on Error Spike (Error = 28.0px)
    # -------------------------------------------------------------
    print("\n[Test 2] Injecting Aerodynamic Turbulence Spike (Error = 28.0 px)...")
    cmd_x_spike, cmd_y_spike = controller.compute_command(target_pos=(348.0, 240.0), dt=0.016)

    print(f"  -> Commanded Rate: ({cmd_x_spike:.2f}, {cmd_y_spike:.2f}) deg/s")
    print(f"  -> DDPG Dampener Active: {controller.is_ai_dampener_active}")
    assert controller.is_ai_dampener_active == True, "DDPG dampener must wake up on degradation (error > 11px)!"

    # -------------------------------------------------------------
    # Test 3: Standalone Dampener Lead Compensation
    # -------------------------------------------------------------
    print("\n[Test 3] Testing Standalone Non-linear Lead Damping...")
    dampener = TinyDDPGDampener()
    # Step 1: establish initial error
    dampener.compute_damping(err_px_x=15.0, err_px_y=0.0, pid_cmd_x=1.0, pid_cmd_y=0.0, dt=0.016)
    # Step 2: rapid accelerating error spike (+20px in 16ms -> ~1250 px/s rate)
    dx, dy, active = dampener.compute_damping(
        err_px_x=35.0, err_px_y=0.0, pid_cmd_x=2.5, pid_cmd_y=0.0, dt=0.016
    )

    print(f"  -> Damping Correction Delta: dx = {dx:.2f} deg/s, dy = {dy:.2f} deg/s")
    print(f"  -> Dampener Woken: {active}")
    assert active == True
    assert abs(dx) > 0.0, "DDPG dampener must output non-zero rate correction!"

    # -------------------------------------------------------------
    # Test 4: Latency Benchmark (<15ms Budget)
    # -------------------------------------------------------------
    print("\n[Test 4] Measuring Controller + DDPG Execution Latency...")
    t0 = time.perf_counter()
    for _ in range(1000):
        controller.compute_command(target_pos=(345.0, 255.0), dt=0.016)
    avg_latency_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0

    print(f"  -> Average Execution Latency: {avg_latency_ms:.4f} ms per step (Spec Limit: < 15.0 ms)")
    assert avg_latency_ms < 2.0, "Control loop latency exceeded edge hardware budget!"

    print("\n" + "=" * 60)
    print("PHASE 6 VERIFICATION SUCCESSFUL: TinyDDPG Dampener passed all tests!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase_6_ddpg_dampener()
