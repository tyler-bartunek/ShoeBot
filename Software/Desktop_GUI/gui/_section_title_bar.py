
#File navigation functionality
from pathlib import Path

#PyQt Functionality
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QToolButton, QMenu, QWidgetAction
)
from PyQt6.QtGui import QStandardItem
from PyQt6.QtCore import pyqtSignal

from connection import RobotProfileManager, RobotProfile


class TitleBar(QWidget):
    
    def __init__(self, img:Path, parent = None):
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self.setFixedHeight(40)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(12)

        
        title = QLabel()
        title.setText(
                        f"<h1><img src='{img}' width='30' height='30'> KICK Robot Desktop </h1>"
                        )
        title.setObjectName("TitleLabel")
        layout.addWidget(title)
        layout.addStretch()
        

        # Robot selector — populated by discovery signals
        self.robot_combo = RobotSelector()
        self.robot_combo.setObjectName("RobotCombo")
        self.robot_combo.setMinimumWidth(180)
        layout.addWidget(self.robot_combo)

        # Connection status dot + label
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("StatusDotDisconnected")
        layout.addWidget(self.status_dot)

        self.status_label = QLabel("disconnected")
        self.status_label.setObjectName("StatusLabel")
        layout.addWidget(self.status_label)
        
    
class RobotItem(QWidget):
    
    connect_robot = pyqtSignal(str)
    disconnect_robot = pyqtSignal(str)
    remove_robot = pyqtSignal(str)
    
    """A simple data object to hold robot information for the combo box."""
    def __init__(self, name:str, profile:RobotProfile, parent = None):
        super().__init__(parent)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.name = name
        self.profile = profile
        
        self.label = QLabel(name)
        self.connect_button = QPushButton("Connect")
        self.remove_button = QPushButton("Remove")
        
        self.connect_button.setVisible(False)
        
        self.connect_button.clicked.connect(self._on_connect_clicked)
        self.remove_button.clicked.connect(self._on_remove_clicked)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.connect_button)
        self.layout.addWidget(self.remove_button)
        
    def set_available(self, avail:bool):
        
        self.connect_button.setVisible(avail)
        
    def _on_connect_clicked(self):
        """Emit a signal to connect to this robot."""
        if not self.profile.has_focus:
            self.connect_robot.emit(self.name)
            #Change label to disconnect and connect this button to the disconnect signal
            self.connect_button.setText("Disconnect")
            self.connect_button.clicked.disconnect(self._on_connect_clicked)
            self.connect_button.clicked.connect(self._on_disconnect_clicked)
        
    def _on_disconnect_clicked(self):
        """Emit a signal to disconnect from this robot."""
        self.disconnect_robot.emit(self.name)
        #Change label to connect and connect this button to the connect signal
        self.connect_button.setText("Connect")
        self.connect_button.clicked.disconnect(self._on_disconnect_clicked)
        self.connect_button.clicked.connect(self._on_connect_clicked)
        
        
    def _on_remove_clicked(self):
        """Remove this robot from the list."""
        self.setParent(None)
        self.deleteLater()
        self.remove_robot.emit(self.name)
        
        
class RobotSelector(QToolButton):
    """A custom QToolButton that acts as a dropdown menu for selecting robots."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Select Robot ▾")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        self.menu = QMenu(self)
        self.setMenu(self.menu)
        
        self.count = 0

        self._placeholder = self.menu.addAction("No robots found")
        self._placeholder.setEnabled(False)

    def add_robot(self, robot_item: "RobotItem"):
        if self._placeholder is not None:
            self.menu.removeAction(self._placeholder)
            self._placeholder = None

        action = QWidgetAction(self.menu)
        action.setDefaultWidget(robot_item)
        self.menu.addAction(action)
        self.count += 1
        
    def remove_robot(self, robot_name: str):
        for action in self.menu.actions():
            widget = action.defaultWidget()
            if isinstance(widget, RobotItem) and widget.name == robot_name:
                self.menu.removeAction(action)
                widget.setParent(None)
                widget.deleteLater()
                self.count -= 1
                break
            if self.count == 0:
                self._placeholder = self.menu.addAction("No robots found")
                
                
