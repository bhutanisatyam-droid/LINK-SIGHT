"""HUD Video Display Widget for FSOC coarse tracking camera.

Renders:
- Native 640x480 aspect-ratio video feed
- Fixed dead-center crosshairs with mil-dot/azimuth-elevation reticle
- Detection bounding box: Green (locked), Amber (degraded), Red dashed (lost/search)
- Kalman filter predicted position & 3-sigma uncertainty ellipse in Cyan (distinct from detection box)
- Thin error vector line from center crosshairs to beacon centroid
- Cut hexagonal spiral search waypoints as dots with active search point highlighted
- Mission-control status chip in corner: ACQUIRING / TRACKING / DEGRADED / LOST
"""

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from fsoc.engine import EngineOutput
from fsoc.gui.theme import (
    COLOR_BG_DARK,
    COLOR_DEGRADED,
    COLOR_HUD_RETICLE,
    COLOR_LOCKED,
    COLOR_LOST,
    COLOR_PREDICTED,
    get_mono_font,
)


class VideoDisplayWidget(QWidget):
    """High-performance HUD video viewport widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.output: Optional[EngineOutput] = None
        self.qimage: Optional[QImage] = None

        # Fixed native source resolution
        self.src_w = 640
        self.src_h = 480

        # Monospace fonts for HUD text
        self.hud_font_large = get_mono_font(size_pt=10, bold=True)
        self.hud_font_small = get_mono_font(size_pt=8, bold=False)

    def update_frame(self, output: EngineOutput) -> None:
        """Receive new tracking engine output and trigger repaint."""
        self.output = output
        frame = output.frame

        # Convert numpy uint8 array to QImage (Format_Grayscale8)
        h, w = frame.shape[:2]
        if len(frame.shape) == 2:
            self.qimage = QImage(
                frame.data, w, h, w, QImage.Format.Format_Grayscale8
            ).copy()
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            self.qimage = QImage(
                frame.data, w, h, 3 * w, QImage.Format.Format_BGR888
            ).copy()

        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Background fill
        painter.fillRect(self.rect(), QColor(COLOR_BG_DARK))

        # Calculate aspect-ratio preserved destination rect
        w_widget = self.width()
        h_widget = self.height()
        aspect = float(self.src_w) / float(self.src_h)

        if w_widget / h_widget > aspect:
            # Pillarbox (black bars on left/right)
            render_h = h_widget
            render_w = int(h_widget * aspect)
            offset_x = (w_widget - render_w) // 2
            offset_y = 0
        else:
            # Letterbox (black bars top/bottom)
            render_w = w_widget
            render_h = int(w_widget / aspect)
            offset_x = 0
            offset_y = (h_widget - render_h) // 2

        target_rect = QRect(offset_x, offset_y, render_w, render_h)

        # 1. Draw camera video frame
        if self.qimage is not None:
            painter.drawImage(target_rect, self.qimage)
        else:
            painter.fillRect(target_rect, QColor("#0D1117"))
            painter.setPen(QColor(COLOR_HUD_RETICLE))
            painter.drawText(target_rect, Qt.AlignCenter, "[ WAITING FOR FRAME SOURCE ]")
            return

        # Coordinate transformation helper from 640x480 to widget coordinates
        scale_x = float(render_w) / float(self.src_w)
        scale_y = float(render_h) / float(self.src_h)

        def to_widget(x: float, y: float) -> QPointF:
            return QPointF(offset_x + x * scale_x, offset_y + y * scale_y)

        # 2. Draw Subtle HUD Reticle & Crosshairs at Dead-Center (320, 240)
        center_pt = to_widget(320.0, 240.0)
        cx, cy = center_pt.x(), center_pt.y()

        reticle_pen = QPen(QColor(COLOR_HUD_RETICLE), 1.0)
        painter.setPen(reticle_pen)

        # Inner crosshairs with center gap
        gap = 12.0
        line_len = 45.0
        painter.drawLine(QPointF(cx - line_len, cy), QPointF(cx - gap, cy))
        painter.drawLine(QPointF(cx + gap, cy), QPointF(cx + line_len, cy))
        painter.drawLine(QPointF(cx, cy - line_len), QPointF(cx, cy - gap))
        painter.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + line_len))

        # Outer mil-spec aiming rings
        r_inner = 25.0 * scale_x
        r_outer = 65.0 * scale_x
        reticle_pen_subtle = QPen(QColor("#243144"), 1.0, Qt.DashLine)
        painter.setPen(reticle_pen_subtle)
        painter.drawEllipse(center_pt, r_inner, r_inner)
        painter.drawEllipse(center_pt, r_outer, r_outer)

        # Dead center dot
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(COLOR_HUD_RETICLE)))
        painter.drawEllipse(center_pt, 2.0, 2.0)

        if self.output is None:
            return

        out = self.output
        det = out.detection
        trk = out.track

        # 3. Draw Cut Hexagonal Spiral Search Waypoints (if re-acquisition active)
        if out.is_searching and out.search_waypoints:
            painter.setPen(Qt.NoPen)
            for i, (wx, wy) in enumerate(out.search_waypoints):
                wpt = to_widget(wx, wy)
                if i == out.active_waypoint_idx:
                    # Active search point highlighted in Amber
                    painter.setBrush(QBrush(QColor(COLOR_DEGRADED)))
                    painter.drawEllipse(wpt, 5.0, 5.0)
                    painter.setPen(QPen(QColor(COLOR_DEGRADED), 1.0, Qt.DashLine))
                    painter.drawEllipse(wpt, 10.0, 10.0)
                    painter.setPen(Qt.NoPen)
                elif i < out.active_waypoint_idx:
                    # Visited waypoints (dim blue)
                    painter.setBrush(QBrush(QColor("#1F3A52")))
                    painter.drawEllipse(wpt, 2.0, 2.0)
                else:
                    # Upcoming waypoints (cyan dots)
                    painter.setBrush(QBrush(QColor(COLOR_PREDICTED)))
                    painter.drawEllipse(wpt, 2.5, 2.5)

        # 4. Draw Raw Detection Bounding Box
        if det.detected:
            dx, dy, dw, dh = det.bbox
            top_left = to_widget(dx, dy)
            box_w = dw * scale_x
            box_h = dh * scale_y

            # Determine box color based on status
            if out.is_searching:
                box_pen = QPen(QColor(COLOR_LOST), 1.5, Qt.DashLine)
            elif trk.status.value == "TRACKING":
                box_pen = QPen(QColor(COLOR_LOCKED), 1.5)
            else:
                box_pen = QPen(QColor(COLOR_DEGRADED), 1.5)

            painter.setPen(box_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(top_left.x(), top_left.y(), box_w, box_h))

            # Centroid crosshair
            c_pt = to_widget(det.centroid[0], det.centroid[1])
            painter.drawLine(QPointF(c_pt.x() - 4, c_pt.y()), QPointF(c_pt.x() + 4, c_pt.y()))
            painter.drawLine(QPointF(c_pt.x(), c_pt.y() - 4), QPointF(c_pt.x(), c_pt.y() + 4))

        # 5. Draw Kalman Predicted Position & Uncertainty Ellipse (in Cyan)
        # Distinct from raw detection box so they visibly coincide when tracking is good
        # and diverge / persist alone during brief occlusion!
        if trk.is_valid or out.is_searching:
            pred_pt = to_widget(trk.pos[0], trk.pos[1])
            semi_maj, semi_min, angle = trk.covariance_ellipse
            r_maj = max(6.0, semi_maj * scale_x)
            r_min = max(6.0, semi_min * scale_y)

            # Dashed Cyan ellipse showing Kalman predicted state & covariance
            painter.save()
            painter.translate(pred_pt)
            painter.rotate(math.degrees(angle))
            pred_pen = QPen(QColor(COLOR_PREDICTED), 1.2, Qt.DashLine)
            painter.setPen(pred_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(0, 0), r_maj, r_min)
            # Center diamond for predicted position
            painter.drawRect(QRectF(-2.5, -2.5, 5.0, 5.0))
            painter.restore()

        # 6. Thin Error Vector Line from Center Reticle to Target
        if trk.is_valid:
            target_pt = to_widget(trk.pos[0], trk.pos[1])
        elif det.detected:
            target_pt = to_widget(det.centroid[0], det.centroid[1])
        else:
            target_pt = None

        if target_pt is not None:
            err_px = out.tracking_error_px
            if err_px > 3.0: # Only draw if above micro-threshold
                # Line color: green if <= 10px (ISRO spec), amber if <= 25px, red if > 25px
                if err_px <= 10.0:
                    vec_color = QColor(COLOR_LOCKED)
                elif err_px <= 25.0:
                    vec_color = QColor(COLOR_DEGRADED)
                else:
                    vec_color = QColor(COLOR_LOST)

                vec_pen = QPen(vec_color, 1.2)
                painter.setPen(vec_pen)
                painter.drawLine(center_pt, target_pt)

                # Error distance callout near midpoint
                mid_x = (cx + target_pt.x()) / 2.0 + 8.0
                mid_y = (cy + target_pt.y()) / 2.0 - 6.0
                painter.setFont(self.hud_font_small)
                painter.setPen(vec_color)
                painter.drawText(QPointF(mid_x, mid_y), f"ERR: {err_px:.1f}px")

        # 7. Corner Status & AI Subsystem HUD Card (Top-Right)
        status_str = out.status_label
        if "TRACKING" in status_str:
            badge_color = QColor(COLOR_LOCKED)
            badge_bg = QColor(14, 32, 24, 235)
            mode_desc = "ESTIMATOR: KALMAN FILTER (MATH)"
            if getattr(out, 'is_ai_dampener_active', False):
                ai_status = "DDPG DAMPENER [AI WAKE-UP ACTIVE]"
            else:
                ai_status = "AI CO-PROCESSORS [SLEEP MODE]"
        elif "DEGRADED" in status_str:
            badge_color = QColor(COLOR_DEGRADED)
            badge_bg = QColor(36, 28, 14, 235)
            if trk.ai_gru_active:
                mode_desc = "ESTIMATOR: MicroGRU AI (WAKE-UP)"
                ai_status = "NON-LINEAR RECURRENT COASTING"
            else:
                mode_desc = "ESTIMATOR: KALMAN COAST (MATH)"
                ai_status = f"DROPOUT COAST: {trk.consecutive_misses}/20 FRAMES"
        elif "LOST" in status_str:
            badge_color = QColor(COLOR_LOST)
            badge_bg = QColor(38, 18, 18, 235)
            mode_desc = "RE-ACQUISITION: CUT HEX SPIRAL"
            ai_status = "EXPANDING UNCERTAINTY SWEEP"
        else:
            badge_color = QColor(COLOR_DEGRADED)
            badge_bg = QColor(24, 28, 36, 235)
            mode_desc = "INITIALIZING OPTICAL LINK"
            ai_status = "ACQUIRING BEACON CENTROID"

        badge_w = 265.0
        badge_h = 52.0
        badge_x = offset_x + render_w - badge_w - 10.0
        badge_y = offset_y + 10.0

        painter.setPen(QPen(badge_color, 1.2))
        painter.setBrush(QBrush(badge_bg))
        painter.drawRect(QRectF(badge_x, badge_y, badge_w, badge_h))

        # Status glowing dot + primary status header
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(badge_color))
        painter.drawEllipse(QPointF(badge_x + 12.0, badge_y + 13.0), 4.0, 4.0)

        painter.setFont(self.hud_font_large)
        painter.setPen(badge_color)
        painter.drawText(
            QRectF(badge_x + 22.0, badge_y + 3.0, badge_w - 26.0, 20.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            status_str,
        )

        # Line 2: Estimator Engine (e.g. KALMAN FILTER [MATH] / MicroGRU AI [WAKE-UP])
        painter.setFont(self.hud_font_small)
        painter.setPen(QColor("#E2E8F0"))
        painter.drawText(
            QRectF(badge_x + 10.0, badge_y + 22.0, badge_w - 20.0, 14.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            mode_desc,
        )

        # Line 3: AI Co-Processor Sleep/Wake Status
        is_ai_woken = trk.ai_gru_active or getattr(out, 'is_ai_dampener_active', False)
        painter.setPen(QColor(COLOR_DEGRADED) if is_ai_woken else QColor("#8C9BAE"))
        painter.drawText(
            QRectF(badge_x + 10.0, badge_y + 36.0, badge_w - 20.0, 13.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"⚡ {ai_status}",
        )

        # 8. Top-Left HUD Telemetry Overlay (Gimbal Coordinates + Benchmark Mode)
        tele_x = offset_x + 12.0
        tele_y = offset_y + 18.0
        painter.setFont(self.hud_font_small)
        painter.setPen(QColor("#A0AEC0"))

        if out.ground_truth is not None:
            pan_deg, tilt_deg = out.ground_truth.cam_angles_deg
            gimbal_text = f"GIMBAL: AZ {pan_deg:+.2f}° | EL {tilt_deg:+.2f}° | FOV 4.0°x3.0°"
        else:
            gimbal_text = "MODE: BENCHMARK-2 (RAW FOOTAGE) | PTZ NO-OP"

        painter.drawText(QPointF(tele_x, tele_y), gimbal_text)

        # 9. Bottom-Left Target Metrics Overlay
        bot_y = offset_y + render_h - 12.0
        if det.detected:
            det_text = f"CONF: {det.confidence:.2f} | SNR: {det.snr_db:.1f} dB | APERTURE: {math.sqrt(det.area):.1f}px"
        else:
            det_text = f"CONF: 0.00 | COASTING FRAMES: {trk.consecutive_misses}"

        painter.drawText(QPointF(tele_x, bot_y), det_text)

        # Hairline outer border around video viewport
        painter.setPen(QPen(QColor("#2B3648"), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(target_rect)
