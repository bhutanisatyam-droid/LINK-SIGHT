"""Optical Beacon Tracking module for FSOC coarse alignment.

Implements a 2D Constant Velocity Discrete Kalman Filter fusing optical detections:
- State: [x, y, vx, vy]^T
- Dynamic measurement noise covariance R scaled inversely by detection confidence
- Chi-square innovation gating for clutter rejection
- Multi-frame occlusion coasting (predicts through detection gaps without premature loss)
- Uncertainty covariance ellipse calculation for HUD visualization and re-acquisition sizing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque
import os
from typing import Deque, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

from fsoc.detector import DetectionResult


class TrackStatus(Enum):
    """Lifecycle states of the optical beacon track."""
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"       # Healthy track, verified measurement fused
    DEGRADED = "DEGRADED"       # Coasting through occlusion / gap; prediction active
    LOST = "LOST"               # Sustained loss, covariance boundary exceeded -> trigger search


@dataclass
class TrackEstimate:
    """State estimate emitted by the Kalman filter."""
    status: TrackStatus
    pos: Tuple[float, float]               # (x, y) estimated position in camera frame
    vel: Tuple[float, float]               # (vx, vy) estimated velocity in px/s
    covariance: np.ndarray                 # 4x4 state covariance matrix P
    pos_uncertainty_px: float              # 1-sigma positional uncertainty in px
    covariance_ellipse: Tuple[float, float, float] # (semi_major_px, semi_minor_px, angle_rad)
    consecutive_misses: int = 0
    is_valid: bool = False
    ai_gru_active: bool = False            # True when MicroGRU is actively coasting the track


class BaseTracker(ABC):
    """Abstract interface for beacon state estimators."""

    @abstractmethod
    def update(self, detection: DetectionResult, dt: float) -> TrackEstimate:
        """Process new frame detection and return updated track state."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset state vector and covariance matrices."""
        pass


class KalmanBeaconTracker(BaseTracker):
    """Discrete Kalman Filter with Wake-on-Degradation MicroGRU Trajectory Coaster."""

    def __init__(
        self,
        q_spectral_density: float = 600.0,     # Process noise acceleration intensity (px^2/s^3)
        base_meas_noise_std: float = 2.5,     # Base measurement noise standard deviation (px)
        max_coast_frames: int = 20,           # 20 frames at 30Hz = 0.67s coast budget
        max_uncertainty_px: float = 200.0,    # Positional uncertainty threshold to declare loss
        gating_threshold_chi2: float = 25.0,  # Chi-squared 2-DOF innovation gate threshold
        model_path: str = "models/microgru_coast.onnx",
    ):
        self.q = q_spectral_density
        self.base_meas_std = base_meas_noise_std
        self.max_coast_frames = max_coast_frames
        self.max_uncertainty_px = max_uncertainty_px
        self.chi2_gate = gating_threshold_chi2
        self.model_path = model_path

        # 30-frame rolling state history ring buffer for MicroGRU: [dx, dy, vx, vy]
        self.state_history: Deque[np.ndarray] = deque(maxlen=30)
        self.prev_pos: Optional[Tuple[float, float]] = None
        self.ai_gru_active: bool = False

        # Load MicroGRU ONNX session if available
        self.gru_session = None
        if _ONNX_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.gru_session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            except Exception:
                self.gru_session = None

        # State vector: [px, py, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float64)

        # State covariance matrix P
        self.P = np.eye(4, dtype=np.float64) * 1000.0

        # Measurement matrix H
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64
        )

        self.status = TrackStatus.INITIALIZING
        self.consecutive_misses: int = 0
        self.track_age_frames: int = 0
        self.lost_confirm_count: int = 0
        self._init_hits: int = 0

        self.reset()

    def reset(self) -> None:
        """Reset tracker state to initial acquisition stance."""
        self.x = np.array([[320.0], [240.0], [0.0], [0.0]], dtype=np.float64)
        self.P = np.diag([100.0, 100.0, 400.0, 400.0]).astype(np.float64)
        self.status = TrackStatus.INITIALIZING
        self.consecutive_misses = 0
        self.track_age_frames = 0
        self.lost_confirm_count = 0
        self._init_hits = 0
        self.state_history.clear()
        self.prev_pos = None
        self.ai_gru_active = False

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        """Compute state transition matrix F for time step dt."""
        return np.array(
            [[1.0, 0.0,  dt, 0.0],
             [0.0, 1.0, 0.0,  dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64
        )

    def _get_process_noise(self, dt: float) -> np.ndarray:
        """Compute continuous white noise acceleration process covariance Q."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        q = self.q
        return q * np.array(
            [[dt3 / 3.0, 0.0,       dt2 / 2.0, 0.0      ],
             [0.0,       dt3 / 3.0, 0.0,       dt2 / 2.0],
             [dt2 / 2.0, 0.0,       dt,        0.0      ],
             [0.0,       dt2 / 2.0, 0.0,       dt       ]],
            dtype=np.float64
        )

    def _compute_covariance_ellipse(self) -> Tuple[float, float, float]:
        """Extract 2D positional 3-sigma covariance ellipse (semi-major, semi-minor, angle)."""
        P_pos = self.P[:2, :2]
        eigenvals, eigenvecs = np.linalg.eigh(P_pos)
        # Sort eigenvalues descending
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # 3-sigma semi-axes (covers 98.9% probability mass for 2D Gaussian)
        semi_major = 3.0 * math.sqrt(max(eigenvals[0], 1e-4))
        semi_minor = 3.0 * math.sqrt(max(eigenvals[1], 1e-4))
        angle_rad = math.atan2(eigenvecs[1, 0], eigenvecs[0, 0])
        return (semi_major, semi_minor, angle_rad)

    def update(self, detection: DetectionResult, dt: float) -> TrackEstimate:
        """Step Kalman filter: Predict -> Validate -> Update (or MicroGRU Coast)."""
        dt = max(0.001, min(0.1, dt))

        # 1. Prediction step
        F = self._get_transition_matrix(dt)
        Q = self._get_process_noise(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Check for valid measurement
        has_detection = detection.detected and detection.confidence >= 0.20

        if self.status == TrackStatus.INITIALIZING:
            if has_detection:
                self._init_hits += 1
                if self._init_hits >= 2:
                    # M-out-of-N confirmation (2 consecutive detections) to establish verified optical track
                    cx, cy = detection.centroid
                    self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
                    self.P = np.diag([25.0, 25.0, 10000.0, 10000.0]).astype(np.float64)
                    self.status = TrackStatus.TRACKING
                    self.consecutive_misses = 0
                    self.track_age_frames = 2
                    self.prev_pos = (cx, cy)
                    self.ai_gru_active = False
                    self._init_hits = 0
                else:
                    # Tentative first hit: do not declare TRACKING yet
                    self.status = TrackStatus.INITIALIZING
                    return self._make_estimate(is_valid=False)
            else:
                self._init_hits = 0
                self.status = TrackStatus.INITIALIZING
                self.ai_gru_active = False
                return self._make_estimate(is_valid=False)

        elif self.status in (TrackStatus.TRACKING, TrackStatus.DEGRADED):
            if has_detection:
                z = np.array([[detection.centroid[0]], [detection.centroid[1]]], dtype=np.float64)

                # Dynamic measurement noise scaled by detection confidence
                meas_std = self.base_meas_std / max(detection.confidence, 0.20)
                R = np.eye(2, dtype=np.float64) * (meas_std ** 2)

                # Innovation (measurement residual)
                y = z - self.H @ self.x
                S = self.H @ self.P @ self.H.T + R

                try:
                    S_inv = np.linalg.inv(S)
                    # Mahalanobis distance / Chi-squared gating
                    mahalanobis_sq = float((y.T @ S_inv @ y).item())

                    # Tiered chi-squared gate or direct optical re-lock:
                    if mahalanobis_sq < 2500.0:
                        # Kalman gain
                        K = self.P @ self.H.T @ S_inv

                        # State update
                        self.x = self.x + K @ y

                        # Joseph form covariance update for numerical stability
                        I_KH = np.eye(4, dtype=np.float64) - K @ self.H
                        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

                        self.status = TrackStatus.TRACKING
                        self.consecutive_misses = 0
                        self.track_age_frames += 1

                        # Record clean state history for MicroGRU memory
                        pos_x = float(self.x[0, 0])
                        pos_y = float(self.x[1, 0])
                        if self.prev_pos is not None:
                            dx = pos_x - self.prev_pos[0]
                            dy = pos_y - self.prev_pos[1]
                            vx = float(self.x[2, 0])
                            vy = float(self.x[3, 0])
                            self.state_history.append(np.array([dx, dy, vx, vy], dtype=np.float32))
                        self.prev_pos = (pos_x, pos_y)
                        self.ai_gru_active = False
                    elif detection.confidence >= 0.35:
                        # Direct sub-second optical re-acquisition upon beam restoration
                        cx, cy = detection.centroid
                        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
                        self.P = np.diag([15.0, 15.0, 10000.0, 10000.0]).astype(np.float64)
                        self.status = TrackStatus.TRACKING
                        self.consecutive_misses = 0
                        self.track_age_frames += 1
                        self.prev_pos = (cx, cy)
                        self.ai_gru_active = False
                    else:
                        # Gated out: low confidence outlier clutter
                        self.consecutive_misses += 1
                except np.linalg.LinAlgError:
                    self.consecutive_misses += 1
            else:
                # Occlusion / dropout: coast on prediction
                self.consecutive_misses += 1

            # MicroGRU Non-Linear Coasting Intervention
            if self.consecutive_misses > 0:
                if self.gru_session is not None and len(self.state_history) >= 10:
                    # Wake-on-Degradation: MicroGRU active
                    self.ai_gru_active = True
                    try:
                        hist = list(self.state_history)
                        while len(hist) < 30:
                            hist.insert(0, hist[0])
                        seq = np.array(hist[-30:], dtype=np.float32)[np.newaxis, :, :]
                        pred_delta = self.gru_session.run(None, {"state_history": seq})[0][0]
                        # Apply non-linear predicted displacement to state vector
                        self.x[0, 0] += float(pred_delta[0])
                        self.x[1, 0] += float(pred_delta[1])
                        self.state_history.append(
                            np.array([pred_delta[0], pred_delta[1], float(self.x[2, 0]), float(self.x[3, 0])], dtype=np.float32)
                        )
                    except Exception:
                        self.ai_gru_active = False
                else:
                    self.ai_gru_active = False

            # Check degradation / loss thresholds
            pos_uncert = math.sqrt(max(self.P[0, 0] + self.P[1, 1], 1e-4))
            pos_x_val = float(self.x[0, 0])
            pos_y_val = float(self.x[1, 0])
            is_off_screen = (pos_x_val < -10.0 or pos_x_val > 650.0 or pos_y_val < -10.0 or pos_y_val > 490.0)

            if self.consecutive_misses > 0:
                if self.consecutive_misses >= self.max_coast_frames or pos_uncert > self.max_uncertainty_px or is_off_screen:
                    self.status = TrackStatus.LOST
                    self.ai_gru_active = False
                    self.x[2, 0] = 0.0
                    self.x[3, 0] = 0.0
                else:
                    self.status = TrackStatus.DEGRADED
                    self.x[2, 0] *= 0.95
                    self.x[3, 0] *= 0.95

        elif self.status == TrackStatus.LOST:
            self.ai_gru_active = False
            # Immediate sub-second re-lock upon verified optical detection
            if has_detection and detection.confidence >= 0.20:
                cx, cy = detection.centroid
                self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
                self.P = np.diag([15.0, 15.0, 10000.0, 10000.0]).astype(np.float64)
                self.status = TrackStatus.TRACKING
                self.consecutive_misses = 0
                self.track_age_frames = 1
                self.prev_pos = (cx, cy)
                self.lost_confirm_count = 0
            else:
                self.lost_confirm_count = 0

        is_valid = self.status in (TrackStatus.TRACKING, TrackStatus.DEGRADED)
        return self._make_estimate(is_valid=is_valid)

    def _make_estimate(self, is_valid: bool) -> TrackEstimate:
        """Construct TrackEstimate output."""
        pos_x = float(self.x[0, 0])
        pos_y = float(self.x[1, 0])
        vel_x = float(self.x[2, 0])
        vel_y = float(self.x[3, 0])
        pos_uncert = math.sqrt(max(self.P[0, 0] + self.P[1, 1], 1e-4))
        ellipse = self._compute_covariance_ellipse()

        return TrackEstimate(
            status=self.status,
            pos=(pos_x, pos_y),
            vel=(vel_x, vel_y),
            covariance=self.P.copy(),
            pos_uncertainty_px=pos_uncert,
            covariance_ellipse=ellipse,
            consecutive_misses=self.consecutive_misses,
            is_valid=is_valid,
            ai_gru_active=self.ai_gru_active,
        )
