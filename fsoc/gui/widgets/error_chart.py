"""Real-time Rolling Tracking Error Chart widget.

Renders:
- High-speed rolling time-series graph of tracking error (px) over time (15s window)
- Horizontal dashed line marking the ISRO coarse alignment threshold (<= 10.0 px)
- Vertical shaded background regions marking target-loss and re-acquisition events
- Pure QPainter implementation for optimal frame rate and crisp telemetry styling
"""

from typing import List, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from fsoc.gui.theme import (
    COLOR_BG_DARK,
    COLOR_BORDER,
    COLOR_BORDER_LIGHT,
    COLOR_DEGRADED,
    COLOR_LOCKED,
    COLOR_LOST,
    COLOR_SURFACE_1,
    COLOR_TEXT_DIM,
    COLOR_TEXT_MUTED,
    get_mono_font,
)


class TrackingErrorChartWidget(QWidget):
    """Rolling line chart displaying live tracking error and loss intervals."""

    def __init__(self, time_window_sec: float = 15.0, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.time_window = time_window_sec
        self.data_points: List[Tuple[float, float, int]] = [] # (time_s, error_px, status_code)
        self.font_mono = get_mono_font(size_pt=8, bold=False)

    def update_data(self, points: List[Tuple[float, float, int]]) -> None:
        """Update chart history points and trigger repaint."""
        self.data_points = points
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Background fill
        painter.fillRect(self.rect(), QColor(COLOR_SURFACE_1))

        margin_l = 45.0
        margin_r = 15.0
        margin_t = 15.0
        margin_b = 22.0

        plot_w = self.width() - margin_l - margin_r
        plot_h = self.height() - margin_t - margin_b

        if plot_w <= 10 or plot_h <= 10:
            return

        plot_rect = QRectF(margin_l, margin_t, plot_w, plot_h)

        # 1. Determine dynamic X and Y scale
        if self.data_points:
            t_now = self.data_points[-1][0]
            t_min = max(0.0, t_now - self.time_window)
            t_max = max(t_min + self.time_window, t_now)

            # Filter visible points
            vis_points = [p for p in self.data_points if p[0] >= t_min - 0.5]
            max_val = max([p[1] for p in vis_points] + [25.0])
            # Round up max_val to nice number
            y_max = max(25.0, min(150.0, max_val * 1.25))
        else:
            t_min = 0.0
            t_max = self.time_window
            y_max = 30.0
            vis_points = []

        def to_screen(t: float, err: float) -> QPointF:
            nx = (t - t_min) / max(t_max - t_min, 1e-4)
            ny = 1.0 - (err / max(y_max, 1e-4))
            sx = margin_l + nx * plot_w
            sy = margin_t + ny * plot_h
            return QPointF(sx, sy)

        # 2. Draw Hairline Grid Lines & Y-Axis Labels
        painter.setFont(self.font_mono)
        painter.setPen(QPen(QColor(COLOR_BORDER), 1.0))

        y_ticks = [0.0, 10.0, y_max * 0.5, y_max]
        for y_val in y_ticks:
            py = to_screen(t_min, y_val).y()
            if margin_t <= py <= margin_t + plot_h:
                painter.setPen(QPen(QColor(COLOR_BORDER), 1.0))
                painter.drawLine(QPointF(margin_l, py), QPointF(margin_l + plot_w, py))

                # Label
                painter.setPen(QColor(COLOR_TEXT_DIM))
                painter.drawText(
                    QRectF(0, py - 6.0, margin_l - 6.0, 14.0),
                    Qt.AlignRight | Qt.AlignVCenter,
                    f"{int(y_val)}px",
                )

        # 3. Draw Vertical Shaded Regions for Loss and Degraded intervals
        if len(vis_points) >= 2:
            painter.setPen(Qt.NoPen)
            for i in range(len(vis_points) - 1):
                t1, _, s1 = vis_points[i]
                t2, _, _ = vis_points[i + 1]
                x1 = to_screen(t1, 0).x()
                x2 = to_screen(t2, 0).x()

                if s1 == 2:
                    # LOST / SEARCHING interval -> Red shaded band
                    painter.fillRect(
                        QRectF(x1, margin_t, max(1.0, x2 - x1), plot_h),
                        QColor(231, 76, 60, 40),
                    )
                elif s1 == 1:
                    # DEGRADED / COASTING interval -> Amber shaded band
                    painter.fillRect(
                        QRectF(x1, margin_t, max(1.0, x2 - x1), plot_h),
                        QColor(243, 156, 18, 30),
                    )

        # 4. Draw Horizontal ISRO Spec Target Line (10.0 px threshold)
        spec_y = to_screen(t_min, 10.0).y()
        if margin_t <= spec_y <= margin_t + plot_h:
            spec_pen = QPen(QColor("#D29922"), 1.0, Qt.DashLine)
            painter.setPen(spec_pen)
            painter.drawLine(QPointF(margin_l, spec_y), QPointF(margin_l + plot_w, spec_y))

            # Spec callout tag
            painter.setFont(self.font_mono)
            painter.setPen(QColor("#D29922"))
            painter.drawText(
                QRectF(margin_l + 8.0, spec_y - 14.0, 180.0, 12.0),
                Qt.AlignLeft,
                "SPEC LIMIT: 10.0 px",
            )

        # 5. Draw Live Tracking Error Polyline
        if len(vis_points) >= 2:
            curve_pen = QPen(QColor(COLOR_LOCKED), 1.8)
            painter.setPen(curve_pen)

            poly = QPolygonF()
            for t_val, err_val, _ in vis_points:
                pt = to_screen(t_val, err_val)
                # Clamp Y to chart bounds
                pt.setY(max(margin_t, min(margin_t + plot_h, pt.y())))
                poly.append(pt)

            painter.drawPolyline(poly)

            # Draw current value point dot
            last_pt = poly.last()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(COLOR_LOCKED)))
            painter.drawEllipse(last_pt, 3.5, 3.5)

        # 6. Chart Border & Title
        painter.setPen(QPen(QColor(COLOR_BORDER_LIGHT), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(plot_rect)

        # Title
        painter.setFont(self.font_mono)
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        painter.drawText(
            QRectF(margin_l + 6.0, margin_t + 4.0, 250.0, 14.0),
            Qt.AlignLeft,
            "REAL-TIME TRACKING ERROR HISTORY (15s)",
        )

        # X-Axis Time Label
        painter.drawText(
            QRectF(margin_l, margin_t + plot_h + 4.0, plot_w, 14.0),
            Qt.AlignRight,
            "TIME (T-15s → NOW)",
        )
