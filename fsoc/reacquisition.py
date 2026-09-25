"""Cut Hexagonal Spiral Re-acquisition Engine for FSOC ATP coarse pointing.

Implements a cut hexagonal spiral search:
- Concentric hexagonal rings with diagonal step traversal
- Bounded and pruned by current Kalman positional uncertainty radius (cut hex pattern)
- Centered on the Kalman filter's last predicted target position
- Immediate termination upon verified detection (sub-second re-acquisition target <= 1s)
- Waypoint telemetry export for real-time HUD visualization
"""

import math
from typing import List, Optional, Tuple


class CutHexagonalSpiralSearch:
    """Executes an angular cut hexagonal spiral search pattern centered on last known target position or boresight."""

    def __init__(
        self,
        step_size_deg: float = 0.8,            # Optimal step size covering camera FOV and orbit
        confidence_threshold: float = 0.60,    # Immediate abort threshold on verified optical detection
        dwell_frames: int = 1,                 # Frames to hold pointing per waypoint
    ):
        self.step_size_deg = step_size_deg
        self.conf_threshold = confidence_threshold
        self.dwell_frames = dwell_frames

        self.waypoints_deg: List[Tuple[float, float]] = []
        self.current_idx: int = 0
        self.current_dwell: int = 0
        self.consecutive_confirms: int = 0
        self.is_active: bool = False
        self.search_center: Tuple[float, float] = (0.0, 0.0)
        self.current_radius_deg: float = 1.6
        self.max_radius_deg: float = 5.0

    def generate_angular_hex_pattern(
        self,
        center_deg: Tuple[float, float] = (0.0, 0.0),
        max_radius_deg: float = 1.5,
    ) -> List[Tuple[float, float]]:
        """Generate ordered hexagonal spiral waypoints in gimbal angular space (degrees)."""
        cx, cy = center_deg
        # Clamp center within physical gimbal limits
        cx = max(-4.5, min(4.5, cx))
        cy = max(-4.5, min(4.5, cy))
        pts: List[Tuple[float, float]] = [(cx, cy)]  # Center point is always first

        s = self.step_size_deg
        max_rings = max(1, int(math.ceil(max_radius_deg / s)))

        # 6 unit directions for hexagonal lattice (60-degree increments)
        hex_dirs = [
            (s * math.cos(i * math.pi / 3.0), s * math.sin(i * math.pi / 3.0))
            for i in range(6)
        ]

        for ring in range(1, max_rings + 1):
            curr_x = cx + ring * hex_dirs[4][0]
            curr_y = cy + ring * hex_dirs[4][1]

            for side in range(6):
                step_dir = hex_dirs[(side + 2) % 6]
                for _ in range(ring):
                    dist = math.hypot(curr_x - cx, curr_y - cy)
                    if dist <= max_radius_deg + 1e-4:
                        clamped_x = max(-5.5, min(5.5, curr_x))
                        clamped_y = max(-5.5, min(5.5, curr_y))
                        pts.append((clamped_x, clamped_y))
                    curr_x += step_dir[0]
                    curr_y += step_dir[1]

        return pts

    def start_search(
        self,
        center_deg: Tuple[float, float] = (0.0, 0.0),
        max_radius_deg: float = 5.0,
        initial_radius_deg: Optional[float] = None,
    ) -> None:
        """Initiate cut hexagonal spiral re-acquisition search centered at last known position or home."""
        self.search_center = center_deg
        self.max_radius_deg = max_radius_deg
        self.current_radius_deg = initial_radius_deg if initial_radius_deg is not None else min(max_radius_deg, 1.6)
        self.waypoints_deg = self.generate_angular_hex_pattern(self.search_center, self.current_radius_deg)
        self.current_idx = 0
        self.current_dwell = 0
        self.consecutive_confirms = 0
        self.is_active = True

    def stop_search(self) -> None:
        """Deactivate search pattern."""
        self.is_active = False
        self.waypoints_deg.clear()
        self.current_idx = 0
        self.consecutive_confirms = 0

    def step(
        self,
        detection_confidence: float,
        cam_pan_deg: float = 0.0,
        cam_tilt_deg: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """Advance search pattern by one frame.

        Returns:
            Optional[Tuple[float, float]]: Current waypoint (target_pan_deg, target_tilt_deg)
            or None if target re-acquired or search finished.
        """
        if not self.is_active:
            return None

        # Two consecutive verified detections required to abort search (prevents single-frame noise glitches)
        if detection_confidence >= self.conf_threshold:
            self.consecutive_confirms += 1
            if self.consecutive_confirms >= 2:
                self.stop_search()
                return None
        else:
            self.consecutive_confirms = 0

        if not self.waypoints_deg:
            self.start_search(self.search_center, self.max_radius_deg, self.current_radius_deg)
            return self.search_center

        if self.current_idx >= len(self.waypoints_deg):
            # Expand search radius hierarchically
            if self.current_radius_deg < self.max_radius_deg:
                self.current_radius_deg = min(self.max_radius_deg, self.current_radius_deg + 1.6)
                self.waypoints_deg = self.generate_angular_hex_pattern(self.search_center, self.current_radius_deg)
                self.current_idx = 0
            else:
                self.current_idx = 0

        wp = self.waypoints_deg[self.current_idx]
        err_dist = math.hypot(wp[0] - cam_pan_deg, wp[1] - cam_tilt_deg)

        self.current_dwell += 1
        # Advance to next waypoint if reached within 0.35 deg or dwelled for 4 frames (~0.13s)
        if err_dist < 0.35 or self.current_dwell >= 4:
            self.current_dwell = 0
            self.current_idx += 1
            if self.current_idx >= len(self.waypoints_deg):
                if self.current_radius_deg < self.max_radius_deg:
                    self.current_radius_deg = min(self.max_radius_deg, self.current_radius_deg + 1.6)
                    self.waypoints_deg = self.generate_angular_hex_pattern(self.search_center, self.current_radius_deg)
                    self.current_idx = 0
                else:
                    self.current_idx = 0

        if self.current_idx < len(self.waypoints_deg):
            return self.waypoints_deg[self.current_idx]
        return wp

    def get_waypoints_deg(self) -> List[Tuple[float, float]]:
        """Return full list of waypoints in gimbal angular space (degrees)."""
        return list(self.waypoints_deg)

    def get_waypoints(self) -> List[Tuple[float, float]]:
        """Legacy compatibility method returning waypoints."""
        return list(self.waypoints_deg)

    def get_current_waypoint_index(self) -> int:
        """Return index of the active search point."""
        return self.current_idx

    def get_search_radius(self) -> float:
        """Return active search radius."""
        return self.current_radius_deg
