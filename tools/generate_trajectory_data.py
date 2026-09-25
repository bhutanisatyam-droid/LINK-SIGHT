"""Synthetic Trajectory Data Generator for MicroGRU-Coast (Core 2 State AI).

Generates ~100,000 steps of 60 Hz UAV flight telemetry across diverse realistic flight regimes:
- Smooth Circular Orbits (varying radii & angular rates)
- High-G Figure-8 Maneuvers
- Linear Cruising with Turbulent Wind Shear (Brownian random walk)
- Sharp Decelerations and Banked Turns

Slices continuous flight into sliding window sequences:
- Input: Past 30 frames [Δx, Δy, vx, vy] (500ms history)
- Target: Future displacement [Δx_next, Δy_next] (autoregressively rolled out for 200ms)

Output:
Saves to `datasets/dataset_trajectory.npz`.
"""

import math
import os
import numpy as np


def generate_flight_session(num_steps: int = 100000, dt: float = 1.0 / 60.0) -> np.ndarray:
    """Simulate continuous multi-regime UAV flight coordinates (x, y)."""
    positions = np.zeros((num_steps, 2), dtype=np.float32)

    # Initial drone state in 2000x2000 scene
    x, y = 1000.0, 1000.0
    vx, vy = 15.0, 0.0
    t = 0.0

    current_mode = "circular"
    mode_timer = 0
    mode_duration = 1200 # Change regime every 20 seconds at 60 Hz

    center_x, center_y = 1000.0, 1000.0
    orbit_radius = 350.0
    orbit_omega = 0.4 # rad/s

    print(f"[*] Simulating {num_steps} flight steps ({num_steps * dt / 60.0:.1f} minutes of flight)...")

    for i in range(num_steps):
        mode_timer += 1
        t += dt

        if mode_timer >= mode_duration:
            mode_timer = 0
            current_mode = np.random.choice(["circular", "figure_8", "linear_gusts", "sharp_turn"])
            orbit_radius = np.random.uniform(150.0, 500.0)
            orbit_omega = np.random.uniform(0.2, 0.7) * np.random.choice([-1, 1])

        if current_mode == "circular":
            angle = orbit_omega * t
            x = center_x + orbit_radius * math.cos(angle)
            y = center_y + orbit_radius * math.sin(angle)

        elif current_mode == "figure_8":
            freq = 0.3
            x = center_x + orbit_radius * math.sin(freq * t)
            y = center_y + (orbit_radius * 0.6) * math.sin(2 * freq * t)

        elif current_mode == "linear_gusts":
            # Linear cruising with Brownian turbulent acceleration
            ax = np.random.normal(0, 12.0)
            ay = np.random.normal(0, 12.0)
            vx += ax * dt
            vy += ay * dt
            # Speed clamping
            speed = math.sqrt(vx ** 2 + vy ** 2)
            if speed > 60.0:
                vx = (vx / speed) * 60.0
                vy = (vy / speed) * 60.0
            x += vx * dt
            y += vy * dt

            # Keep inside bounds
            if x < 300 or x > 1700: vx *= -1
            if y < 300 or y > 1700: vy *= -1

        elif current_mode == "sharp_turn":
            # High-G turning maneuver
            heading = math.atan2(vy, vx) + 2.5 * dt
            speed = 45.0
            vx = speed * math.cos(heading)
            vy = speed * math.sin(heading)
            x += vx * dt
            y += vy * dt

        positions[i] = [x, y]

    return positions


def main():
    print("=" * 60)
    print("LinkSight: MicroGRU Trajectory Dataset Generator")
    print("=" * 60)

    num_steps = 110000
    positions = generate_flight_session(num_steps)

    # Compute step-to-step deltas and estimated velocities
    # dx[t] = x[t] - x[t-1], vx[t] = dx[t] / dt
    dt = 1.0 / 60.0
    deltas = np.diff(positions, axis=0) # shape: (N-1, 2)
    velocities = deltas / dt            # shape: (N-1, 2)

    # State vector at each frame: [dx, dy, vx, vy]
    states = np.hstack([deltas, velocities]) # shape: (N-1, 4)

    # Slice into sequences of 30 past steps (Input) -> next 1 step (Target)
    seq_len = 30
    total_samples = len(states) - seq_len - 12

    X = np.zeros((total_samples, seq_len, 4), dtype=np.float32)
    Y = np.zeros((total_samples, 2), dtype=np.float32)

    print(f"[*] Extracting {total_samples:,} sliding window training sequences...")
    for i in range(total_samples):
        X[i] = states[i : i + seq_len]
        Y[i] = states[i + seq_len, :2] # Next step delta (dx, dy)

    # Calculate normalization parameters
    x_mean = np.mean(X, axis=(0, 1))
    x_std = np.std(X, axis=(0, 1)) + 1e-5
    X_norm = (X - x_mean) / x_std

    os.makedirs("datasets", exist_ok=True)
    out_file = "datasets/dataset_trajectory.npz"
    np.savez_compressed(out_file, X=X_norm, Y=Y, mean=x_mean, std=x_std)

    print(f"\n[SUCCESS] Trajectory dataset generated!")
    print(f"[*] Saved to: {out_file}")
    print(f"[*] Total Sequences: {total_samples:,} (Shape: {X.shape})")
    print(f"[*] File Size: {os.path.getsize(out_file) / (1024 * 1024):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
