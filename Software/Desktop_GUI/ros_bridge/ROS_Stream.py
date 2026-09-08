
from PyQt6.QtCore import QObject, pyqtSignal
import roslibpy

#Import utility to time how fast bot state messages are coming in
from time import perf_counter

_MODULE_TYPE_MAP: dict[int, str] = {
    0x00: "NA",   # Not applicable, no connection
    0x01: "ECH",  # Echo- Debug case
    0x02: "MW-L",   # Left Mecanum Wheel
    0x03: "MW-R",   # Right Mechanum Wheel
    0x04: "QL-A",   # Quadruped leg A
    0x05: "QL-B",   # Quadruped leg B
}
 
def _hw_id_to_type(hw_id: int) -> str:
    return _MODULE_TYPE_MAP.get(hw_id, f"UNK({hw_id:#04x})")



class ROS_StreamWorker(QObject):
    """
    Manages the ROS connection and subscriptions, and emits signals to update the GUI.
    """
    # Define signals to communicate with the GUI
    connection_failed = pyqtSignal(str)
    connection_lost = pyqtSignal(str)
    
    bot_state_updated = pyqtSignal(list)  # Emitted when a new bot state message is received
    message_speed = pyqtSignal(float)  # Emitted to indicate the speed of incoming messages
    battery_updated = pyqtSignal(float) #Emitted when a new battery state is received
    last_vel_updated = pyqtSignal(dict)
    cmd_vel_active = pyqtSignal(bool) #Emitted when the cmd_vel publisher is advertised or unadvertised
    
    last_pose = pyqtSignal(list) #Emitted when a new pose is computed from the last velocity message
    
    log_message = pyqtSignal(str, str) #Emitted for logging messages to the GUI

    def __init__(self, parent=None):
        
        super().__init__(parent)
        self.client = None
        self.bot_state_subscriber = None
        self.rosout_subscriber = None
        self.cmd_vel_publisher = None

    def connect(self, host='localhost', port=9090):
        
        #Initialize the ROS client
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()
        print("ROS connection established")

        # Subscribe to bot_state topic
        self.bot_state_subscriber = roslibpy.Topic(self.client, '/bot_state', 'kickbot_interfaces/msg/BotState')
        self.bot_state_subscriber.subscribe(self._bot_state_callback)
        self.receive_time = perf_counter()  # Initialize the receive time for bot_state messages
        
        #Subscribe to the rosout topic for FaultLog widget
        self.rosout_subscriber = roslibpy.Topic(self.client, '/rosout', 'rcl_interfaces/msg/Log')
        self.rosout_subscriber.subscribe(self.log_callback)
        
        # Set to publish to cmd_vel topic
        self.cmd_vel_publisher = roslibpy.Topic(self.client, "/cmd_vel", 'geometry_msgs/Twist')
        self.cmd_vel_publisher.advertise()
        
        self.cmd_vel_active.emit(self.cmd_vel_publisher.is_advertised)
        

    def _bot_state_callback(self, message: dict):
        """
        Parse BotState and emit bot_state_updated(list[dict]).
 
        Expected ROS message fields:
            active_devices : bool[6]  — True = slot occupied
            device_ids     : int[6]   — hardware ID per slot
            voltage        : float    — voltage readout from ADC for battery monitoring
            velocity       : Twist    — estimated COM velocity of the robot
 
        Emitted list item format (matches RightPanel.refresh_devices):
            {
                "address":  "0xNN",
                "position": "N",
                "type":     "MW" | "SJ" | ...,
                "fault":    bool,
            }
        """
        self.dt = perf_counter() - self.receive_time
        self.message_speed.emit( 1.0 / (self.dt) )  # Calculate and emit the message speed in Hz
        
        active = message.get('active_paths', [False] * 6)
        ids    = message.get('device_ids',     [0]     * 6)
        voltage = message.get('voltage', 3.3)
        
        directions = ['x', 'y', 'z']
        vel_type = ['linear', 'angular']
        vel_default = {vel:{basis:0.0 for basis in directions} for vel in vel_type}
        velocity = message.get('velocity', vel_default)
        
        # print("Emitting voltage and velocity updates")
        self.battery_updated.emit(voltage)
        self.last_vel_updated.emit(velocity)
        self.compute_pose(velocity)  # Update the pose based on the received velocity
        pose_simplified = [self.pose['linear']['x'], self.pose['linear']['y'], self.pose['angular']['z']]
        self.last_pose.emit(pose_simplified)  # Emit the simplified pose for GUI updates
 
        devices = []
        for slot in range(6):
            if not active[slot]:
                continue
 
            hw_id      = ids[slot]
 
            devices.append({
                "address":  f"0x{hw_id:02X}",
                "position": str(slot),
                "type":     _hw_id_to_type(hw_id),
            })
 
        self.bot_state_updated.emit(devices)
        self.receive_time = perf_counter()  # Update the receive time for bot_state messages
        
    def log_callback(self, message):
        """Callback function triggered every time a new log enters /rosout."""
        # Map numeric ROS 2 severity levels to readable strings
        levels = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "FATAL"}
        
        level_num = message.get('level', 0)
        level_name = levels.get(level_num, f"UNKNOWN({level_num})")
        node_name = message.get('name', 'unknown_node')
        log_text = message.get('msg', '')
    
        self.log_message.emit(f"[{node_name}]: {log_text}", level_name)
        
    def compute_pose(self, velocity: dict[str, dict[str, float]]) -> None:
        """
        Compute the new pose based on the current velocity and time delta.
        This is a simple dead reckoning calculation.
        """
        # Initialize pose if not already present
        if not hasattr(self, 'pose'):
            pose = {vel: {basis: 0.0 for basis in ['x', 'y', 'z']} for vel in ['linear', 'angular']}
        
        # Update pose based on velocity and time delta
        for vel_type in ['linear', 'angular']:
            for basis in ['x', 'y', 'z']:
                pose[vel_type][basis] += velocity.get(vel_type, {}).get(basis, 0.0) * self.dt

    def _velocity_msg_callback(self, velocity:dict[str,dict[str,float]]):
        #Send the new velocity command to the device 
        self.cmd_vel_publisher.publish(velocity)
        
        
    def disconnect(self):
        """Clean shutdown — unsubscribe, unadvertise, close connection."""
        if self.bot_state_subscriber:
            self.bot_state_subscriber.unsubscribe()
        if self.rosout_subscriber:
            self.rosout_subscriber.unsubscribe()
        if self.cmd_vel_publisher:
            self.cmd_vel_publisher.unadvertise()
        if self.client and self.client.is_connected:
            self.client.terminate()
            self.client = None
            
        