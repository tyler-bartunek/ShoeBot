
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PyQt6.QtGui import QColor


class FaultLogWidget(QWidget):

    LEVEL_COLORS = {
        "debug": QColor("#9E9E9E"),   # grey
        "info":  QColor("#4CAF50"),   # green -- matches existing FaultMessage_ok convention
        "warn":  QColor("#FFA726"),   # amber
        "error": QColor("#EF5350"),   # red
        "fatal": QColor("#B71C1C"),   # darker red -- escalation, not a new hue
    }

    MAX_ENTRIES = 500  # cap retained history, same reasoning as the SensorConfigWidget leak fix

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FaultLogWidget")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QLabel("LOGGING / FAULTS")
        header.setObjectName("PanelSectionHeader")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setObjectName("FaultLogList")
        layout.addWidget(self._list)

        # "No Faults Detected" placeholder -- same remove-on-first-real-entry
        # pattern as RobotSelector's "No robots found" state.
        self._placeholder = QListWidgetItem("No Faults Detected")
        self._placeholder.setForeground(self.LEVEL_COLORS["info"])
        self._list.addItem(self._placeholder)

    def update_faults(self, message: str, level: str = "info"):
        """Append a new fault/log entry, colored by severity level.

        Kept as `update_faults` (not renamed to `append`) since this is the
        method name already wired to signals elsewhere -- e.g.
        `ros_worker.log_message.connect(fault_log.update_faults)` and
        `connection_failed.connect(lambda r: fault_log.update_faults(f"Pi: {r}", level="error"))`.
        """
        if self._placeholder is not None:
            self._list.takeItem(self._list.row(self._placeholder))
            self._placeholder = None

        item = QListWidgetItem(message)
        item.setForeground(self.LEVEL_COLORS.get(level.lower(), self.LEVEL_COLORS["info"]))
        self._list.addItem(item)
        self._list.scrollToBottom()

        while self._list.count() > self.MAX_ENTRIES:
            self._list.takeItem(0)