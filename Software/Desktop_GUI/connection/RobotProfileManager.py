
"""
Manages saved robot profiles (alias, hostname, username, workspace).
Profiles are stored in %APPDATA%/KICKRobot/robots.json on Windows.
Passwords are stored separately in the OS keychain via keyring.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from ros_bridge import ROS_StreamWorker
 


@dataclass
class RobotProfile:
    
    hostname:  str
    port:  int
    ip_address: str
    workspace: str = ""   # filled in after first successful connect + detect
    has_focus: bool = False
    bridge_available: bool = False
    hardware_connection_points: Optional[dict[int, int]] = None  # {board_connection_point: offset from default for board_connection_point)}

    def display(self) -> str:
        return f"{self.hostname}:{self.port}"

class RobotProfileManager:
    """
    Manage the list of connected robots, including saving and loading settings from disk.
    """

    def __init__(self):
        self._path = self._config_path()
        self._bridges: dict[str, ROS_StreamWorker] = {} #TODO: Update load to also update this list
        self._profiles: list[RobotProfile] = self._load_profiles()
        
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def profiles(self) -> list[RobotProfile]:
        return list(self._profiles)

    def add_or_update(self, profile: RobotProfile,
                      ros_worker: ROS_StreamWorker):
        """
        Save or update a profile. If remember=True and password is given,
        store the password in the OS keychain.
        """
  
        self._profiles = [p for p in self._profiles if p.hostname != profile.hostname]
        self._profiles.append(profile)
        self._bridges[profile.hostname] = ros_worker
        self._save()

            
    def change_focus(self, hostname: str) -> None:
        
        for p in self._profiles:
            if p.hostname != hostname:
                p.has_focus = False
                #Disconnect signals from the ROS_StreamWorker for this profile, with exception of the logger
                try:
                    self._bridges[p.hostname].bot_state_updated.disconnect()
                    self._bridges[p.hostname].message_speed.disconnect()
                    self._bridges[p.hostname].battery_updated.disconnect()
                    self._bridges[p.hostname].cmd_vel_active.disconnect()
                except Exception:
                    pass
            else:
                p.has_focus = True


    def remove(self, hostname: str):
        self._profiles = [p for p in self._profiles
                          if p.hostname != hostname]
        try:
            self._bridges.pop(hostname) #Remove the stream_worker, hopefully flagging it for garbage collection
        except KeyError:
            pass
        self._save()

    def update_workspace(self, hostname: str, workspace: str):
        """Update workspace path after successful auto-detect."""
        for p in self._profiles:
            if p.hostname == hostname:
                p.workspace = workspace
                break
        self._save()

    def get(self, hostname: str) -> RobotProfile | None:
        return next((p for p in self._profiles
                     if p.hostname == hostname), None)
        
    def get_bridge(self, hostname: str) -> ROS_StreamWorker:
        try:
            bridge = self._bridges[hostname]
        except KeyError:
            self._bridges[hostname] = ROS_StreamWorker()
            bridge = self._bridges[hostname]
            
        return bridge
            
    
    def get_address(self, hostname: str):
        
        for p in self._profiles:
            if p.hostname == hostname:
                return p.ip_address

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_profiles(self) -> list[RobotProfile]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            #Ensure no robot has focus to start upon load
            profiles = [RobotProfile(**d) for d in data]
            for p in profiles:
                p.has_focus = False
                p.bridge_available = False
            return profiles
        except Exception:
            return []
        
    def _load_ros_workers(self) -> dict[str, ROS_StreamWorker]:
        
        for p in self._profiles:
            self._bridges[p.hostname] = ROS_StreamWorker()
            self._bridges[p.hostname].connect()
        

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(p) for p in self._profiles], indent=2),
            encoding="utf-8"
        )

    @staticmethod
    def _config_path() -> Path:
        # %APPDATA% on Windows, ~/.config on Linux/Mac
        base = Path(os.environ.get("APPDATA", Path.home() / ".config"))
        return base / "KICKRobot" / "robots.json"
