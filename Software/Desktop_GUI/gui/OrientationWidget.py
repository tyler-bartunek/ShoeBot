import math
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt6.QtCore import Qt, QPointF, pyqtSignal


class _CompassCanvas(QWidget):
    """
    Draws a circular compass with a robot rectangle that rotates
    to reflect current yaw (dead reckoning).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self._yaw_deg = 0.0

    def set_yaw(self, degrees: float):
        self._yaw_deg = degrees
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        r  = min(cx, cy) - 3

        # Outer ring
        p.setPen(QPen(QColor("#2a2a2a"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Crosshairs
        p.setPen(QPen(QColor("#2a2a2a"), 0.5))
        p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
        p.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))

        # Robot rectangle — rotates with yaw
        p.save()
        p.translate(cx, cy)
        p.rotate(self._yaw_deg)

        rect_w, rect_h = 16, 26
        p.setPen(QPen(QColor("#378ADD"), 1.5))
        p.setBrush(QBrush(QColor("#1e2a38")))
        p.drawRoundedRect(
            int(-rect_w / 2), int(-rect_h / 2),
            rect_w, rect_h, 2, 2
        )

        # Forward arrow
        arrow_tip    = QPointF(0, -rect_h / 2 - 4)
        arrow_left   = QPointF(-4, -rect_h / 2 + 2)
        arrow_right  = QPointF( 4, -rect_h / 2 + 2)
        poly = QPolygonF([arrow_tip, arrow_left, arrow_right])
        p.setBrush(QBrush(QColor("#378ADD")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(poly)

        p.restore()
        p.end()


class OrientationWidget(QWidget):
    """
    Dead-reckoning orientation display.
    Shows compass canvas + yaw angle + X/Y position + zero button.
    """

    zeroed = pyqtSignal()   # emitted when user presses Zero

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OrientationWidget")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        header = QLabel("TELEMETRY: VELOCITY & ORIENTATION")
        header.setObjectName("PanelSectionHeader")
        outer.addWidget(header)

        row = QHBoxLayout()
        row.setSpacing(10)

        self._canvas = _CompassCanvas()
        row.addWidget(self._canvas)

        stats = QVBoxLayout()
        stats.setSpacing(4)

        self._yaw_lbl = self._stat_row("yaw", "0°")
        self._x_lbl   = self._stat_row("x",   "0.00 m")
        self._y_lbl   = self._stat_row("y",   "0.00 m")

        stats.addLayout(self._yaw_lbl["layout"])
        stats.addLayout(self._x_lbl["layout"])
        stats.addLayout(self._y_lbl["layout"])

        zero_btn = QPushButton("zero")
        zero_btn.setObjectName("SmallButton")
        zero_btn.setFixedHeight(24)
        zero_btn.clicked.connect(self._on_zero)
        stats.addWidget(zero_btn)
        stats.addStretch()

        row.addLayout(stats)
        outer.addLayout(row)

    def _stat_row(self, key: str, default: str) -> dict:
        layout = QHBoxLayout()
        layout.setSpacing(6)
        key_lbl = QLabel(key)
        key_lbl.setObjectName("OrientStatKey")
        val_lbl = QLabel(default)
        val_lbl.setObjectName("OrientStatValue")
        layout.addWidget(key_lbl)
        layout.addWidget(val_lbl)
        layout.addStretch()
        return {"layout": layout, "value": val_lbl}

    def _on_zero(self):
        self.zeroed.emit()
        self.update_pose(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_pose(self, x: float, y: float, yaw_deg: float):
        self._canvas.set_yaw(yaw_deg)
        self._yaw_lbl["value"].setText(f"{yaw_deg:.1f}°")
        self._x_lbl["value"].setText(f"{x:.2f} m")
        self._y_lbl["value"].setText(f"{y:.2f} m")
