"""Top-Down 2D Ground-Truth Mini-Map Radar widget.

Displays:
- Full 2000x2000 scene radar view
- Ground truth beacon position and flight trail
- Virtual camera gimbal position, pointing vector, and 4°x3° FOV coverage wedge
- Disambiguates "camera panned" vs "target moved"
- Automatically greys out with technical watermark in VideoFileFrameSource mode
"""

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from fsoc.frame_source import GroundTruthState
from fsoc.gui.theme import (
    COLOR_BG_DARK,
    COLOR_BORDER,
    COLOR_BORDER_LIGHT,
    COLOR_DEGRADED,
    COLOR_HUD_ACCENT,
    COLOR_HUD_RETICLE,
    COLOR_LOCKED,
    COLOR_LOST,
    COLOR_PREDICTED,
    COLOR_SURFACE_1,
    COLOR_SURFACE_2,
    COLOR_TEXT_DIM,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    get_mono_font,
)


class TopDownMinimapWidget(QWidget):
    """2D World Scene Radar showing true beacon position and camera pointing cone."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.ground_truth: Optional[GroundTruthState] = None
        self.font_mono = get_mono_font(size_pt=8, bold=False)
        self.font_watermark = get_mono_font(size_pt=9, bold=True)

    def update_ground_truth(self, gt: Optional[GroundTruthState]) -> None:
        """Update telemetry ground truth and trigger repaint."""
        self.ground_truth = gt
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Background fill
        painter.fillRect(self.rect(), QColor(COLOR_SURFACE_2))

        side = min(self.width(), self.height()) - 20
        offset_x = (self.width() - side) / 2.0
        offset_y = (self.height() - side) / 2.0
        map_rect = QRectF(offset_x, offset_y, side, side)

        # Scene is 2000x2000 px
        scene_dim = 2000.0

        def to_map(x: float, y: float) -> QPointF:
            nx = x / scene_dim
            ny = y / scene_dim
            return QPointF(offset_x + nx * side, offset_y + ny * side)

        # 1. Radar Background Grid & Concentric Range Rings
        painter.setPen(QPen(QColor(COLOR_BORDER), 1.0))
        painter.setBrush(QBrush(QColor(COLOR_BG_DARK)))
        painter.drawRect(map_rect)

        center_map = to_map(scene_dim / 2.0, scene_dim / 2.0)
        painter.setBrush(Qt.NoBrush)

        for r_pct in (0.25, 0.50, 0.75):
            radius = (side / 2.0) * r_pct
            painter.setPen(QPen(QColor("#182230"), 1.0, Qt.DotLine))
            painter.drawEllipse(center_map, radius, radius)

        # Center axes
        painter.setPen(QPen(QColor("#1E2A3C"), 1.0))
        painter.drawLine(QPointF(offset_x, center_map.y()), QPointF(offset_x + side, center_map.y()))
        painter.drawLine(QPointF(center_map.x(), offset_y), QPointF(center_map.x(), offset_y + side))

        # Title
        painter.setFont(self.font_mono)
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        painter.drawText(
            QRectF(offset_x + 4.0, offset_y + 4.0, 180.0, 14.0),
            Qt.AlignLeft,
            "TOP-DOWN 2D SCENE RADAR",
        )

        gt = self.ground_truth

        # 2. Check if Running Against VideoFileFrameSource (No Ground Truth)
        if gt is None:
            # Grey out and draw clear technical watermark
            painter.fillRect(map_rect, QColor(9, 12, 16, 210))
            painter.setPen(QPen(QColor(COLOR_DEGRADED), 1.0, Qt.DashLine))
            painter.drawRect(map_rect)

            painter.setFont(self.font_watermark)
            painter.setPen(QColor(COLOR_DEGRADED))
            painter.drawText(
                map_rect,
                Qt.AlignCenter,
                "[ GROUND TRUTH N/A ]\nBENCHMARK-2 VIDEO MODE\nUN-SIMULATED FOOTAGE",
            )
            return

        # 3. Draw Historical Target Trail
        if len(gt.target_trail) >= 2:
            trail_pen = QPen(QColor("#1A4835"), 1.2)
            painter.setPen(trail_pen)
            trail_poly = QPolygonF([to_map(px, py) for px, py in gt.target_trail[-80:]])
            painter.drawPolyline(trail_poly)

        # 4. Draw Camera Field-of-View (FOV) Footprint & Pointing Vector
        # Camera window is 640x480 centered at gt.cam_pos
        cam_x, cam_y = gt.cam_pos
        cam_pt = to_map(cam_x, cam_y)

        # 640x480 box in 2000x2000 scene
        cam_w_scene = 640.0
        cam_h_scene = 480.0
        c_left = to_map(cam_x - cam_w_scene / 2.0, cam_y - cam_h_scene / 2.0)
        c_right = to_map(cam_x + cam_w_scene / 2.0, cam_y + cam_h_scene / 2.0)
        cam_box_rect = QRectF(c_left, c_right)

        # FOV Footprint Rectangle in Cyan
        painter.setPen(QPen(QColor(COLOR_PREDICTED), 1.2))
        painter.setBrush(QBrush(QColor(0, 210, 255, 20)))
        painter.drawRect(cam_box_rect)

        # Camera Center Marker
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(COLOR_HUD_ACCENT)))
        painter.drawEllipse(cam_pt, 3.0, 3.0)

        # Pointing lines from scene center (gimbal mount) to camera center
        painter.setPen(QPen(QColor("#253D5C"), 1.0, Qt.DashLine))
        painter.drawLine(center_map, cam_pt)

        # 5. Draw Optical Beacon True Position Marker
        tgt_x, tgt_y = gt.target_pos
        tgt_pt = to_map(tgt_x, tgt_y)

        # Outer ring if in FOV, red if out of FOV
        if gt.is_target_in_fov:
            painter.setPen(QPen(QColor(COLOR_LOCKED), 1.2))
            painter.setBrush(QBrush(QColor(COLOR_LOCKED)))
        else:
            painter.setPen(QPen(QColor(COLOR_LOST), 1.2))
            painter.setBrush(QBrush(QColor(COLOR_LOST)))

        painter.drawEllipse(tgt_pt, 3.5, 3.5)

        # 6. Bottom Coordinates Callout with background pill
        coord_font = get_mono_font(size_pt=7, bold=False)
        painter.setFont(coord_font)
        
        coord_bg = QRectF(offset_x, offset_y + side - 16.0, side, 16.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(9, 12, 16, 220)))
        painter.drawRect(coord_bg)
        
        painter.setPen(QColor(COLOR_TEXT_PRIMARY))
        coord_str = f"TGT:({int(tgt_x)},{int(tgt_y)}) | CAM:({int(cam_x)},{int(cam_y)})"
        painter.drawText(
            QRectF(offset_x + 4.0, offset_y + side - 15.0, side - 8.0, 14.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            coord_str,
        )

        # Outer border
        painter.setPen(QPen(QColor(COLOR_BORDER_LIGHT), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(map_rect)

