from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt


device_types = {"0x00":None, "0x01":"Echo", "0x02":"Wheels L", "0x03":"Wheels R", "0x04":"Quad Leg A", "0x05":"Quad Leg B"}

class _SectionHeader(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text.upper(), parent)
        self.setObjectName("PanelSectionHeader")
        
    def _on_device_change(self):
        
        pass


class _ModuleLibraryItem(QWidget):
    """Draggable chip representing a module type in the library."""

    def __init__(self, label: str, dot_color: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ModuleLibraryItem")
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)

        row = QWidget()
        from PyQt6.QtWidgets import QHBoxLayout
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(7)

        dot = QLabel("  ")
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(
            f"background-color: {dot_color}; border-radius: 2px;"
        )

        lbl = QLabel(label)
        lbl.setObjectName("ModuleLibraryLabel")

        hl.addWidget(dot)
        hl.addWidget(lbl)
        hl.addStretch()
        layout.addWidget(row)


class _DeviceItem(QWidget):
    """Row showing a detected SPI device with status dot."""

    def __init__(self, address: str, position: str,
                 module_type: str, parent=None):
        super().__init__(parent)
        self.setObjectName("DeviceItem")

        from PyQt6.QtWidgets import QHBoxLayout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(7)

        # dot = QLabel("●")
        # dot.setObjectName("DeviceDotFault" if fault else "DeviceDotOnline")
        # dot.setFixedWidth(14)

        text = QLabel(f"{address}  ·  pos {position}  ·  {module_type}")
        text.setObjectName("DeviceItemLabel")

        # layout.addWidget(dot)
        layout.addWidget(text)
        layout.addStretch()


class RightPanel(QWidget):
    """
    Right sidebar: Module Library (static catalog) on top,
    Detected Devices (live from bus_state) below.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("RightPanel")
        self.setFixedWidth(190)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Module Library ---
        lib_section = QWidget()
        lib_section.setObjectName("PanelSection")
        lib_layout = QVBoxLayout(lib_section)
        lib_layout.setContentsMargins(12, 10, 12, 10)
        lib_layout.setSpacing(4)
        lib_layout.addWidget(_SectionHeader("Module Library"))
        lib_layout.addWidget(_ModuleLibraryItem("Echo",       "#5BA8A0"))  # muted teal — sensor/comms feel
        lib_layout.addWidget(_ModuleLibraryItem("Wheels L",   "#7B68EE"))  # slate purple
        lib_layout.addWidget(_ModuleLibraryItem("Wheels R",   "#9B88FF"))  # lighter purple — related to L
        lib_layout.addWidget(_ModuleLibraryItem("Quad Leg A", "#E07B54"))  # terracotta
        lib_layout.addWidget(_ModuleLibraryItem("Quad Leg B", "#C45A35"))

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("PanelDivider")

        # --- Detected Devices ---
        dev_section = QWidget()
        dev_section.setObjectName("PanelSection")
        dev_layout = QVBoxLayout(dev_section)
        dev_layout.setContentsMargins(12, 10, 12, 10)
        dev_layout.setSpacing(2)
        dev_layout.addWidget(_SectionHeader("Detected Devices"))

        self._device_list_layout = dev_layout   # store ref for live updates

        # Scroll area so the list can grow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("DeviceScroll")
        scroll.setWidget(dev_section)

        layout.addWidget(lib_section)
        layout.addWidget(sep)
        layout.addWidget(scroll, stretch=1)


    # ------------------------------------------------------------------
    # Public API — call from roslibpy bus_state callback
    # ------------------------------------------------------------------

    def refresh_devices(self, devices: list[dict]):
        """
        Replace device list with fresh data.
        Each dict: { "address": str, "position": str,
                     "type": str}
        """
        # Clear existing items (keep header)
        self.clear_devices()

        for d in devices:
            self._device_list_layout.addWidget(
                _DeviceItem(
                    d["address"], d["position"],
                    d["type"]
                )
            )
        self._device_list_layout.addStretch()
        
    def clear_devices(self):
        """Clear the device list (keep header)."""
        while self._device_list_layout.count() > 1:
            item = self._device_list_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
