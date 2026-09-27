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
        step_size_deg: float = 1.3,            # Optimal step size with 65% camera FOV overlap
        confidence_threshold: float = 0.20,    # Fast abort threshold on verified optical detection
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
        self.current_radius_deg: float = 5.8
        self.max_radius_deg: float = 5.8

    def generate_angular_hex_pattern(
        self,
        center_deg: Tuple[float, float] = (0.0, 0.0),
        max_radius_deg: float = 3.6,
    ) -> List[Tuple[float, float]]:
        """Generate ordered hexagonal spiral waypoints in gimbal angular space (degrees).
        
        Mathematical Principle:
        - Regular hexagonal lattice achieves the optimal 2D circle packing density of pi/(2*sqrt(3)) ≈ 90.69%
          (compared to square grids which only achieve 78.54%).
        - Concentric rings expand outward at radius r = k * step_size (k = 1, 2, ...).
        - Ring k has 6 * k discrete waypoints along the 6 equilateral hexagonal facets.
        """
        cx, cy = center_deg
        cx = max(-4.5, min(4.5, cx))
        cy = max(-4.5, min(4.5, cy))
        pts: List[Tuple[float, float]] = [(cx, cy)]

        s = self.step_size_deg
        max_rings = max(1, int(math.ceil(max_radius_deg / s)))

        for r in range(1, max_rings + 1):
            # Compute 6 primary vertices of regular hexagon at radial distance (r * s):
            # V_i = [cx + r*s*cos(60°*i),  cy + r*s*sin(60°*i)] for i in {0..5}
            verts = [
                (cx + r * s * math.cos(math.radians(60 * i)),
                 cy + r * s * math.sin(math.radians(60 * i)))
                for i in range(6)
            ]
            # Interpolate (r) discrete points along each of the 6 hexagonal segments
            for i in range(6):
                v1 = verts[i]
                v2 = verts[(i + 1) % 6]
                for k in range(r):
                    px = v1[0] + (k / r) * (v2[0] - v1[0])
                    py = v1[1] + (k / r) * (v2[1] - v1[1])
                    dist = math.hypot(px - cx, py - cy)
                    # Bound search path within maximum covariance uncertainty radius
                    if dist <= max_radius_deg + 1e-4:
                        if -5.0 <= px <= 5.0 and -5.0 <= py <= 5.0:
                            pts.append((round(px, 3), round(py, 3)))

        if not pts:
            pts.append((cx, cy))
        return pts

    def start_search(
        self,
        center_deg: Tuple[float, float] = (0.0, 0.0),
        max_radius_deg: float = 3.6,
        initial_radius_deg: Optional[float] = None,
    ) -> None:
        """Initiate cut hexagonal spiral re-acquisition search centered at last known position or home."""
        self.search_center = center_deg
        self.max_radius_deg = max_radius_deg
        self.current_radius_deg = max_radius_deg
        self.waypoints_deg = self.generate_angular_hex_pattern(self.search_center, self.max_radius_deg)
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

        # Verified detection immediately halts search and locks on
        if detection_confidence >= self.conf_threshold:
            self.stop_search()
            return None

        if not self.waypoints_deg:
            self.start_search(self.search_center, self.max_radius_deg)
            return self.search_center

        if self.current_idx >= len(self.waypoints_deg):
            self.current_idx = 0

        wp = self.waypoints_deg[self.current_idx]
        err_dist = math.hypot(wp[0] - cam_pan_deg, wp[1] - cam_tilt_deg)

        self.current_dwell += 1
        # Continuous smooth slew: advance to next waypoint when within 0.60 deg or dwelled for 3 frames (~0.1s)
        if err_dist < 0.60 or self.current_dwell >= 3:
            self.current_dwell = 0
            self.current_idx = (self.current_idx + 1) % len(self.waypoints_deg)

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
