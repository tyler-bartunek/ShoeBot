from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class _StatCard(QWidget):
    def __init__(self, label: str, value: str = "---", parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(1)
        self._label = QLabel(label)
        self._label.setObjectName("StatCardLabel")
        self._value = QLabel(value)
        self._value.setObjectName("StatCardValue_normal")
        layout.addWidget(self._label)
        layout.addWidget(self._value)

    def set_value(self, text: str, state: str = "normal"):
        self._value.setText(text)
        self._value.setObjectName(f"StatCardValue_{state}")
        self._value.style().unpolish(self._value)
        self._value.style().polish(self._value)


class _BatteryCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        
        self._percent = 0.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(2)
        
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("battery")
        
        self._label.setObjectName("StatCardLabel")
        self._pct = QLabel("---")
        self._pct.setObjectName("StatCardValue_ok")
        self._pct.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        top_row.addWidget(self._label)
        top_row.addStretch()
        top_row.addWidget(self._pct)
        
        self._bar_bg = QWidget()
        self._bar_bg.setObjectName("BatBarBg")
        self._bar_bg.setFixedHeight(5)
        
        self._bar_fill = QWidget(self._bar_bg)
        self._bar_fill.setObjectName("BatBarFill")
        self._bar_fill.setFixedHeight(5)
        
        self._voltage = QLabel("--- V avg")
        self._voltage.setObjectName("StatCardLabel")
        
        layout.addLayout(top_row)
        layout.addWidget(self._bar_bg)
        layout.addWidget(self._voltage)

    def set_battery(self, voltage: float):
        
        self._percent = (voltage / 3.3) * 100.
        
        self._pct.setText(f"{self._percent:.0f}%")
        self._voltage.setText(f"{voltage:.2f} V avg")
        
        self._update_fill()
        state = "ok" if self._percent > 40 else ("warn" if self._percent > 20 else "error")
        self._pct.setObjectName(f"StatCardValue_{state}")
        self._pct.style().unpolish(self._pct)
        self._pct.style().polish(self._pct)

    def _update_fill(self):
        fill_w = int(self._bar_bg.width() * self._percent / 100)
        self._bar_fill.setFixedWidth(max(fill_w, 0))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_fill()


class StatusStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatusStrip")
        self.setFixedHeight(52)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(6)
        self._bus     = _StatCard("bus devices", "0/6")
        self._cmdvel  = _StatCard("cmd_vel", "inactive")
        self._loop    = _StatCard("loop", "--- Hz")
        self._battery = _BatteryCard()
        for card in (self._bus, self._cmdvel, self._loop):
            layout.addWidget(card, stretch=1)
        layout.addWidget(self._battery, stretch=1)

    def update_bus(self, devices:list):
        connected = len(devices)
        self._bus.set_value(f"{connected} / {6}",
                            "ok" if ((connected <= 6) and (connected > 0)) else "warn")

    def update_cmdvel(self, publishing: bool):
        self._cmdvel.set_value("publishing" if publishing else "inactive",
                               "ok" if publishing else "normal")

    def update_loop(self, hz: float):
        if hz != 0.0:
            self._loop.set_value(f"{hz:.1f} Hz")
        else:
            self._loop.set_value("--- Hz")

    def update_battery(self, voltage: float):
        if voltage != 0.0:
            self._battery.set_battery(voltage)
        else:
            self._battery._pct.setText("---")
            self._battery._voltage.setText("--- V avg")
