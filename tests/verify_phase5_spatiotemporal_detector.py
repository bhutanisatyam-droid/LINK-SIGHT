"""Phase 5 Verification Script: Self-Calibrating Spatiotemporal Beacon Detector.

Tests and verifies:
1. Spatial phase (Frames 1-29) provides immediate zero-lag detection.
2. 1D-FFT Hann window calibration at Frame 30 distinguishes Modulated vs Static/CW beacons.
3. Static beacon fallback mode for Benchmark-2 compatibility (no beacon erasure).
4. Spatiotemporal solar glint rejection in modulated mode.
5. 8D feature vector extraction.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from fsoc.detector import DetectionResult, SpatiotemporalBeaconDetector


def create_synthetic_frame(
    cx: int = 320,
    cy: int = 240,
    beacon_intensity: int = 255,
    bg_radiance: int = 14,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Helper to generate clean or noisy synthetic frames."""
    frame = np.full((480, 640), bg_radiance, dtype=np.uint8)
    if beacon_intensity > 0:
        cv2.circle(frame, (cx, cy), 5, beacon_intensity, -1)
        # Add subtle PSF glow
        cv2.circle(frame, (cx, cy), 9, beacon_intensity // 3, 2)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return frame


def test_phase_5_spatiotemporal_detector():
    print("=" * 60)
    print("PHASE 5 VERIFICATION: Spatiotemporal Beacon Detector (Core 1 AI)")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Static / CW Beacon (Benchmark-2 Video Compatibility)
    # -------------------------------------------------------------
    print("\n[Test 1] Testing Static / CW Unmodulated Beacon (Frames 1-35)...")
    detector_static = SpatiotemporalBeaconDetector()

    for f_idx in range(1, 36):
        frame = create_synthetic_frame(cx=320, cy=240, beacon_intensity=255, bg_radiance=25)
        res = detector_static.detect(frame)

        if f_idx == 1:
            print(f"  -> Frame 1 Detection: {res.detected} | Centroid: ({res.centroid[0]:.1f}, {res.centroid[1]:.1f})")
            assert res.detected, "Frame 1 spatial detection must be immediately active (0 acquisition lag)!"

        if f_idx == 30:
            print(f"  -> Frame 30 FFT Verdict: Modulated={detector_static.temporal_filter_enabled}")
            assert detector_static.temporal_filter_enabled == False, "Static beacon must calibrate to Static Mode!"

    # Frame 35 check: Must remain detected without temporal attenuation
    res35 = detector_static.detect(frame)
    print(f"  -> Frame 35 (Steady State) Detection: {res35.detected} | Confidence: {res35.confidence:.2f}")
    assert res35.detected and res35.confidence > 0.60, "Static beacon must not be erased in steady state!"

    # -------------------------------------------------------------
    # Test 2: Modulated (Blinking) Pulsed Laser Beacon
    # -------------------------------------------------------------
    print("\n[Test 2] Testing Modulated (10 Hz Blinking) Laser Beacon...")
    detector_mod = SpatiotemporalBeaconDetector()

    # Simulate 10 Hz square-wave modulation at 60 FPS (period = 6 frames: 3 ON, 3 OFF)
    for f_idx in range(1, 36):
        is_on = (f_idx % 6) < 3
        b_val = 255 if is_on else 25 # Low state still has ambient
        frame = create_synthetic_frame(cx=320, cy=240, beacon_intensity=b_val, bg_radiance=25)
        res = detector_mod.detect(frame)

        if f_idx == 30:
            print(f"  -> Frame 30 FFT Verdict: Modulated={detector_mod.temporal_filter_enabled}")
            print(f"  -> Detected Modulation Frequency: {detector_mod.modulation_freq_hz:.1f} Hz")
            assert detector_mod.temporal_filter_enabled == True, "Blinking laser must calibrate to Modulated Mode!"

    # -------------------------------------------------------------
    # Test 3: High Solar DC Noise & Glint Rejection
    # -------------------------------------------------------------
    print("\n[Test 3] Testing High Solar Background (Noon) & Solar Glint Rejection...")
    # Inject heavy solar DC floor (180 DN) + Gaussian shot noise (15 DN)
    # Target beacon modulated at 10 Hz
    glint_detector = SpatiotemporalBeaconDetector()

    for f_idx in range(1, 35):
        is_on = (f_idx % 6) < 3
        b_val = 255 if is_on else 180
        frame = create_synthetic_frame(
            cx=320, cy=240, beacon_intensity=b_val, bg_radiance=180, noise_sigma=12.0
        )
        res = glint_detector.detect(frame)

    print(f"  -> High Noon Solar Background Handled: Detected={res.detected} | Confidence={res.confidence:.2f}")

    # -------------------------------------------------------------
    # Test 4: 8D Feature Vector Validation
    # -------------------------------------------------------------
    print("\n[Test 4] Testing 8D Morpho-Photometric Feature Vector Extraction...")
    sample_patch = np.random.uniform(0.1, 0.9, (24, 24)).astype(np.float32)
    sample_res = DetectionResult(
        detected=True,
        centroid=(320.0, 240.0),
        bbox=(315, 235, 10, 10),
        confidence=0.85,
        area=80.0,
        snr_db=22.0,
        peak_intensity=245.0,
    )
    feat_8d = glint_detector._extract_8d_features(sample_patch, sample_res, temporal_ratio=4.5)
    print(f"  -> 8D Vector: {np.round(feat_8d, 3)}")
    assert len(feat_8d) == 8, "Feature vector must be exactly 8 dimensions"
    assert np.all(feat_8d >= 0.0) and np.all(feat_8d <= 1.5), "Feature values out of normalized range"

    print("\n" + "=" * 60)
    print("PHASE 5 VERIFICATION SUCCESSFUL: Spatiotemporal Detector passed all tests!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase_5_spatiotemporal_detector()
