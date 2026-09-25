"""Utility script to generate a sample benchmark MP4 video file.

Used to test Benchmark-2 mode (VideoFileFrameSource) out-of-the-box on raw,
recorded camera footage without requiring external video downloads.
"""

import math
import os
import sys

import cv2
import numpy as np


def generate_benchmark_video(
    output_path: str = "assets/benchmark_sample.mp4",
    num_frames: int = 300,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> str:
    """Render a 10-second synthetic test video simulating raw camera recording."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Use mp4v or XVID codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    if not writer.isOpened():
        # Fallback to avi/MJPG if mp4v is not supported on platform
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    print(f"Generating {num_frames} frames benchmark video at: {output_path}...")

    cx, cy = width / 2.0, height / 2.0
    radius = 160.0

    for i in range(num_frames):
        t = i / float(fps)

        # Orbiting spot with occasional jitter
        theta = 0.55 * t
        bx = int(cx + radius * math.cos(theta) + np.random.uniform(-4, 4))
        by = int(cy + (radius * 0.7) * math.sin(theta) + np.random.uniform(-4, 4))

        # Base space background
        frame = np.full((height, width), 16, dtype=np.uint8)

        # Draw optical beacon (12x12 spot + flare)
        half = 6
        if 0 <= bx < width and 0 <= by < height:
            x1, x2 = max(0, bx - half), min(width, bx + half)
            y1, y2 = max(0, by - half), min(height, by + half)
            frame[y1:y2, x1:x2] = 255

            # Flare
            cv2.circle(frame, (bx, by), 12, 90, 1)

        # 5% sensor salt-and-pepper noise
        rand_mask = np.random.rand(height, width)
        frame[rand_mask < 0.025] = 0
        frame[rand_mask > 0.975] = 255

        # Light Gaussian noise
        noise = np.random.normal(0, 10.0, (height, width))
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    print(f"Benchmark video successfully written ({os.path.getsize(output_path)} bytes).")
    return output_path


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "assets/benchmark_sample.mp4"
    generate_benchmark_video(out_file)
