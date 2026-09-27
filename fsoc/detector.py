"""Optical Beacon Detection module for FSOC tracking.

Implements classical computer vision:
- Pre-filtering (median filter for salt-and-pepper noise suppression)
- Morphological Top-Hat filter for background haze isolation
- Dynamic/adaptive thresholding
- Contour/blob feature analysis (area, solidity, aspect ratio)
- Sub-pixel intensity-weighted centroid calculation
- Confidence scoring based on peak SNR and morphological regularity

Structured with BaseDetector ABC for seamless drop-in swapping of future learned detectors.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import math
import os
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


@dataclass
class DetectionResult:
    """Standardized detection result output."""
    detected: bool
    centroid: Tuple[float, float]          # (x, y) in camera coordinates (sub-pixel)
    bbox: Tuple[int, int, int, int]        # (x, y, w, h)
    confidence: float                      # 0.0 to 1.0
    area: float = 0.0                      # Contour pixel area
    snr_db: float = 0.0                    # Signal-to-noise ratio in decibels
    peak_intensity: float = 0.0            # Peak pixel value in detection window
    is_temporal_modulated: bool = False    # True if 1D-FFT confirms active beacon modulation
    ai_beacon_net_active: bool = False     # True if TinyBeaconNet ONNX inference evaluated


class BaseDetector(ABC):
    """Abstract interface for optical beacon detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Process incoming 640x480 frame and return detection state."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal temporal filters or detector state."""
        pass


class ClassicalBeaconDetector(BaseDetector):
    """Classical CV detector combining morphological top-hat filtering,
    adaptive thresholding, and moment-based sub-pixel centroiding.
    """

    def __init__(
        self,
        min_area: float = 2.0,
        max_area: float = 2000.0,
        min_aspect_ratio: float = 0.20,
        max_aspect_ratio: float = 4.50,
        min_solidity: float = 0.15,
        median_kernel_size: int = 5,
        tophat_kernel_size: int = 21,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        self.median_ksize = median_kernel_size

        # Structuring element for morphological top-hat (isolates compact bright spots < 21px)
        self.tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (tophat_kernel_size, tophat_kernel_size)
        )

        # Pre-allocated null result
        self._null_result = DetectionResult(
            detected=False,
            centroid=(320.0, 240.0),
            bbox=(0, 0, 0, 0),
            confidence=0.0,
            area=0.0,
            snr_db=0.0,
            peak_intensity=0.0,
        )

    def reset(self) -> None:
        """No persistent state required for memoryless frame detector."""
        pass

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Execute robust multi-stage detection pipeline with adaptive local background discrimination.
        
        Pipeline Execution Stages:
        1. Pre-filtering: 5x5 spatial median blur for salt-and-pepper shot noise suppression.
        2. Global noise floor estimation: Image-wide mean and standard deviation.
        3. Morphological Top-Hat filtering: Strips low-frequency solar gradients and haze pedestals.
        4. Statistical thresholding: Dynamic CFAR threshold based on Top-Hat noise distribution.
        5. Connected component filtering: Morphological gating on bounding box, area, and aspect ratio.
        6. Local annulus contrast discrimination: Signal-to-Clutter Ratio (SCR) against local surround.
        7. Sub-pixel centroiding: 1st-order spatial intensity moments for sub-pixel accuracy.
        """
        if frame is None or frame.size == 0:
            return self._null_result

        # Convert 3-channel BGR video/sensor frames to 8-bit single-channel grayscale
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h_img, w_img = gray.shape[:2]

        # -------------------------------------------------------------------------
        # Stage 1: 5x5 Median Filter (Impulse Noise Suppression)
        # -------------------------------------------------------------------------
        # Salt-and-pepper noise generates high-frequency isolated single-pixel spikes
        # that can spoof simple thresholders. A 5x5 non-linear median filter strips
        # isolated impulse pixels while preserving compact beacon spot morphology.
        filtered = cv2.medianBlur(gray, 5)

        # -------------------------------------------------------------------------
        # Stage 2: Global Background Statistics & Noise Floor Estimation
        # -------------------------------------------------------------------------
        raw_mean, raw_std = cv2.meanStdDev(filtered)
        bg_mu = float(raw_mean[0][0])
        bg_sigma = float(raw_std[0][0])

        # -------------------------------------------------------------------------
        # Stage 3: Morphological Top-Hat Transform (Daytime Solar Haze Isolation)
        # -------------------------------------------------------------------------
        # Top-Hat Transform: T(I) = I - (I ∘ K)
        # where (I ∘ K) is morphological opening with a rectangular/disk structuring element.
        # This isolates bright compact structures smaller than the kernel size (21x21 px)
        # while completely eliminating uniform DC pedestals and non-uniform solar glare.
        tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, self.tophat_kernel)

        # Zero-out frame perimeter (18px) to suppress boundary interpolation artifacts
        # from camera jitter, vibration warpAffine translations, or sensor margin readout.
        tophat[:18, :] = 0
        tophat[-18:, :] = 0
        tophat[:, :18] = 0
        tophat[:, -18:] = 0

        # -------------------------------------------------------------------------
        # Stage 4: Statistical Top-Hat Noise Floor & Dynamic Thresholding
        # -------------------------------------------------------------------------
        top_mean, top_std = cv2.meanStdDev(tophat)
        t_mu = float(top_mean[0][0])
        t_sigma = float(top_std[0][0])

        # Constant False Alarm Rate (CFAR) threshold rule:
        # Threshold scales dynamically with residual background variance: T = max(μ + 2.5σ, 16.0)
        thresh_val = max(t_mu + 2.5 * max(t_sigma, 1.0), 16.0)
        _, thresh = cv2.threshold(tophat, int(thresh_val), 255, cv2.THRESH_BINARY)

        # -------------------------------------------------------------------------
        # Stage 5: Connected Component & Contour Extraction
        # -------------------------------------------------------------------------
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._null_result

        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 5 or h < 5:
                continue

            # Reject detections on the extreme perimeter
            if x <= 18 or y <= 18 or x + w >= w_img - 18 or y + h >= h_img - 18:
                continue

            bounding_area = float(w * h)
            if bounding_area < 28.0 or bounding_area > self.max_area:
                continue

            # Aspect ratio check: optical spots must be approximately compact (0.38 <= w/h <= 2.65)
            aspect = float(w) / float(h)
            if aspect < 0.38 or aspect > 2.65:
                continue

            contour_area = cv2.contourArea(c)
            if contour_area < 16.0:
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / max(hull_area, 1e-4) if hull_area > 0 else 0.5
            if solidity < 0.22:
                continue

            effective_area = max(contour_area, bounding_area * 0.4)

            # Sub-window peak intensity and energy
            roi_tophat = tophat[y : y + h, x : x + w]
            roi_raw = filtered[y : y + h, x : x + w]
            total_energy = float(np.sum(roi_tophat)) if roi_tophat.size > 0 else 0.0

            if total_energy < 320.0:
                continue

            # ---------------------------------------------------------------------
            # Stage 6: Local Annulus Background Contrast Discrimination (SCR)
            # ---------------------------------------------------------------------
            # To reject bright clutter blobs (clouds, sun glints), we construct a 
            # local rectangular annulus around the candidate bounding box.
            # We compute the local mean (local_mu) and standard deviation (local_sigma)
            # of the surrounding ring, excluding the core beacon pixels.
            pad = 14
            bx1 = max(0, x - pad)
            by1 = max(0, y - pad)
            bx2 = min(w_img, x + w + pad)
            by2 = min(h_img, y + h + pad)

            surround_raw = filtered[by1:by2, bx1:bx2].astype(np.float32)
            core_mask = np.zeros_like(surround_raw, dtype=bool)
            cx1, cy1 = x - bx1, y - by1
            core_mask[cy1 : cy1 + h, cx1 : cx1 + w] = True
            annulus_pixels = surround_raw[~core_mask]
            core_pixels = surround_raw[core_mask]

            if annulus_pixels.size > 0:
                local_mu = float(np.mean(annulus_pixels))
                local_sigma = float(np.std(annulus_pixels))
            else:
                local_mu = bg_mu
                local_sigma = bg_sigma

            core_mean = float(np.mean(core_pixels)) if core_pixels.size > 0 else float(np.max(roi_raw))
            local_contrast = core_mean - local_mu
            global_contrast = core_mean - bg_mu

            # Dual-Gate CFAR Statistical Criteria:
            # 1. Global Contrast Gate: Must exceed full-frame background by >= 3.5 sigma
            # 2. Local Contrast Gate: Must exceed local surrounding annulus by >= 3.0 sigma
            min_global_contrast = max(3.5 * bg_sigma, 10.0)
            min_local_contrast = max(3.0 * local_sigma, 8.0)

            if global_contrast < min_global_contrast or local_contrast < min_local_contrast:
                continue

            # Signal-to-Noise Ratio (dB): SNR = 20 * log10(local_contrast / sigma)
            local_snr = local_contrast / max(local_sigma, 1.0)
            snr_db = 20.0 * np.log10(max(local_snr, 1.0))

            # ---------------------------------------------------------------------
            # Stage 7: Sub-Pixel Intensity-Weighted Centroiding
            # -------------------------------------------------------------------------
            # Centroid formula using 0th (m00) and 1st order (m10, m01) spatial moments:
            # cx = x + m10 / m00,   cy = y + m01 / m00
            # Computed on the Top-Hat transformed intensity map to prevent background bias.
            M = cv2.moments(roi_tophat)
            if M["m00"] > 1e-4:
                cx = float(x + M["m10"] / M["m00"])
                cy = float(y + M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            # Analytical Quality / Confidence Metric [0.0 to 1.0]
            # Weighted combination of SNR, global contrast ratio, contour solidity, and aspect symmetry
            aspect_sym = 1.0 - min(abs(1.0 - aspect), 0.6)
            confidence = float(
                0.40 * min(local_snr / 4.0, 1.0)
                + 0.30 * min(global_contrast / (5.0 * max(bg_sigma, 1.0)), 1.0)
                + 0.15 * solidity
                + 0.15 * aspect_sym
            )
            confidence = max(0.0, min(1.0, confidence))

            # Composite ranking score favoring high total energy flux and peak SNR
            rank_score = total_energy * local_snr * confidence * aspect_sym
            candidates.append(
                (
                    rank_score,
                    DetectionResult(
                        detected=True,
                        centroid=(cx, cy),
                        bbox=(x, y, w, h),
                        confidence=confidence,
                        area=effective_area,
                        snr_db=snr_db,
                        peak_intensity=float(np.max(roi_raw)) if roi_raw.size > 0 else 0.0,
                    ),
                )
            )

        if not candidates:
            return self._null_result

        # Return candidate with highest integrated flux energy & confidence
        return max(candidates, key=lambda pair: pair[0])[1]


class SpatiotemporalBeaconDetector(BaseDetector):
    """Dual-Mode Self-Calibrating Spatiotemporal Beacon Detector (Core 1 AI).
    
    1. Spatial Phase (Frames 1-29): Instantaneous tracking via Top-Hat morphology (0 acquisition lag).
    2. 1D-FFT Auto-Calibration (Frame 30): Applies Hann window to 30-frame temporal history.
       Detects modulated blinking frequency vs static/CW beacon without manual configuration.
    3. Spatiotemporal Solar Rejection (Frame 31+): Rejects 0 Hz DC solar background + shot noise.
    4. TinyBeaconNet AI Classification: Evaluates 8D morpho-photometric features via ONNX runtime.
    """

    def __init__(
        self,
        model_path: str = "models/tinybeaconnet.onnx",
        roi_size: int = 24,
        history_len: int = 30,
        modulation_snr_threshold: float = 3.0,
    ):
        self.spatial_detector = ClassicalBeaconDetector()
        self.roi_size = roi_size
        self.history_len = history_len
        self.mod_snr_thresh = modulation_snr_threshold
        self.model_path = model_path

        # 30-frame rolling ROI buffer for spatiotemporal analysis (24x24 px float32)
        self.roi_buffer: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self.frame_count: int = 0
        self.temporal_filter_enabled: Optional[bool] = None
        self.modulation_freq_hz: float = 0.0

        # Precomputed Hann Window (30 samples) to prevent DC spectral leakage
        n = np.arange(self.history_len)
        self.hann_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (self.history_len - 1))
        self.hann_window = self.hann_window.astype(np.float32)

        # ONNX inference session
        self.onnx_session = None
        if _ONNX_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.onnx_session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            except Exception:
                self.onnx_session = None

        self.last_centroid: Tuple[float, float] = (320.0, 240.0)

    def reset(self) -> None:
        """Reset temporal buffer and recalibrate modulation state."""
        self.spatial_detector.reset()
        self.roi_buffer.clear()
        self.frame_count = 0
        self.temporal_filter_enabled = None
        self.modulation_freq_hz = 0.0
        self.last_centroid = (320.0, 240.0)

    def _crop_roi(self, gray_frame: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Extract a pixel-registered 24x24 ROI centered at (cx, cy)."""
        h, w = gray_frame.shape[:2]
        half = self.roi_size // 2
        ix, iy = int(round(cx)), int(round(cy))

        x1 = max(0, ix - half)
        x2 = min(w, ix + half)
        y1 = max(0, iy - half)
        y2 = min(h, iy + half)

        patch = np.zeros((self.roi_size, self.roi_size), dtype=np.float32)
        roi = gray_frame[y1:y2, x1:x2].astype(np.float32) / 255.0

        ph, pw = roi.shape[:2]
        if ph > 0 and pw > 0:
            patch[:ph, :pw] = roi
        return patch

    def _extract_8d_features(
        self,
        patch: np.ndarray,
        res: DetectionResult,
        temporal_ratio: float,
    ) -> np.ndarray:
        """Extract standardized 8D morpho-photometric feature vector for TinyBeaconNet."""
        area_norm = min(res.area / 400.0, 1.0)
        aspect = res.bbox[2] / max(res.bbox[3], 1)
        aspect_score = 1.0 - min(abs(1.0 - aspect), 0.8)
        peak_norm = min(res.peak_intensity / 255.0, 1.0)
        snr_norm = min(res.snr_db / 30.0, 1.0)

        # Compactness / radial symmetry
        cy, cx = patch.shape[0] / 2.0, patch.shape[1] / 2.0
        y_grid, x_grid = np.indices(patch.shape)
        r_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        weighted_r = np.sum(r_sq * patch) / max(np.sum(patch), 1e-4)
        compactness = float(math.exp(-weighted_r / 25.0))

        # Gradient sharpness (Sobel edge energy)
        sobel_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sharpness = float(min(np.mean(grad_mag) * 3.0, 1.0))

        feat = np.array(
            [
                area_norm,
                aspect_score,
                min(res.confidence, 1.0),
                peak_norm,
                snr_norm,
                compactness,
                sharpness,
                min(temporal_ratio, 1.0),
            ],
            dtype=np.float32,
        )
        return feat

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Execute full spatiotemporal detection pipeline."""
        if frame is None or frame.size == 0:
            return self.spatial_detector._null_result

        # Convert to grayscale if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        self.frame_count += 1

        # 1. Spatial Candidate Proposal
        spatial_res = self.spatial_detector.detect(gray)

        # Determine ROI center (use spatial detection or previous locked position)
        if spatial_res.detected:
            cx, cy = spatial_res.centroid
            self.last_centroid = (cx, cy)
        else:
            cx, cy = self.last_centroid

        # 2. Extract & Buffer Registered 24x24 ROI Patch from 5x5 filtered frame (strips salt-and-pepper)
        filtered_gray = cv2.medianBlur(gray, 5)
        patch = self._crop_roi(filtered_gray, cx, cy)
        self.roi_buffer.append(patch)

        # 3. 1D-FFT Temporal Analysis with Hann Window
        temporal_peak_ratio = 0.0
        is_modulated = False

        if len(self.roi_buffer) == self.history_len:
            time_cube = np.array(self.roi_buffer, dtype=np.float32) # shape (30, 24, 24)

            # Measure total temporal variance across the patch
            temporal_var = float(np.mean(np.var(time_cube, axis=0)))

            # Subtract temporal DC mean per pixel to isolate pure AC fluctuation
            cube_mean = np.mean(time_cube, axis=0, keepdims=True)
            ac_cube = time_cube - cube_mean

            # Apply Hann Window along temporal axis
            windowed_cube = ac_cube * self.hann_window[:, np.newaxis, np.newaxis]

            # 1D-FFT along temporal axis
            fft_res = np.fft.rfft(windowed_cube, axis=0)
            power_spec = np.abs(fft_res) ** 2

            # Spatial mean of AC power spectrum across patch (exclude bin 0 DC)
            ac_power = np.mean(power_spec[1:], axis=(1, 2))
            peak_ac = float(np.max(ac_power)) if len(ac_power) > 0 else 0.0
            noise_floor = float(np.mean(ac_power)) + 1e-6

            # Peak-to-average spectral ratio (PASR)
            if temporal_var > 0.015:
                temporal_peak_ratio = peak_ac / noise_floor
            else:
                temporal_peak_ratio = 0.0

            # Calibration Verdict at Frame 30
            if self.temporal_filter_enabled is None:
                if temporal_peak_ratio >= 4.5 and temporal_var > 0.025:
                    self.temporal_filter_enabled = True
                    peak_bin = int(np.argmax(ac_power)) + 1
                    self.modulation_freq_hz = peak_bin * (60.0 / self.history_len)
                else:
                    self.temporal_filter_enabled = False

            is_modulated = bool(self.temporal_filter_enabled and temporal_peak_ratio >= self.mod_snr_thresh)

        # 4. Feature Extraction & AI / Fusion Scoring
        ai_active = False
        final_confidence = spatial_res.confidence

        if spatial_res.detected:
            feat_8d = self._extract_8d_features(patch, spatial_res, temporal_peak_ratio)

            if self.onnx_session is not None:
                try:
                    patch_in = patch[np.newaxis, np.newaxis, :, :]
                    feat_in = feat_8d[np.newaxis, :]
                    onnx_out = self.onnx_session.run(
                        None,
                        {"patch": patch_in, "features": feat_in}
                    )[0]
                    prob_beacon = float(onnx_out[0, 0])
                    final_confidence = 0.60 * spatial_res.confidence + 0.40 * prob_beacon
                    ai_active = True
                except Exception:
                    ai_active = False

            # Modulated beacon confirmed: boost confidence
            if self.temporal_filter_enabled is True and is_modulated:
                final_confidence = min(1.0, final_confidence * 1.15)

            final_confidence = max(0.0, min(1.0, float(final_confidence)))

        return DetectionResult(
            detected=spatial_res.detected,
            centroid=spatial_res.centroid,
            bbox=spatial_res.bbox,
            confidence=final_confidence if spatial_res.detected else 0.0,
            area=spatial_res.area,
            snr_db=spatial_res.snr_db,
            peak_intensity=spatial_res.peak_intensity,
            is_temporal_modulated=is_modulated,
            ai_beacon_net_active=ai_active,
        )

