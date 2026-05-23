"""UDP packet structs and lockstep bridge for Betaflight SITL.

FDM packets leave on port 9003, RC packets leave on 9004, and normalized motor
packets return on 9002. See ARCHITECTURE.md for the coordinate-frame details.
"""

import struct
import socket
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# Port definitions matching Betaflight SITL (sitl.c)
PORT_PWM_RAW = 9001  # Betaflight -> Simulator (raw PWM)
PORT_PWM = 9002  # Betaflight -> Simulator (normalized)
PORT_STATE = 9003  # Simulator -> Betaflight (FDM/sensor data)
PORT_RC = 9004  # Simulator -> Betaflight (RC channels)

# Default host
DEFAULT_HOST = "127.0.0.1"

# Conversion constants (from sitl.c)
ACC_SCALE = 256.0 / 9.80665  # Convert m/s² to Betaflight LSB
GYRO_SCALE = 16.4  # Convert deg/s to Betaflight LSB
RAD_TO_DEG = 180.0 / np.pi

# Packet sizes
MAX_RC_CHANNELS = 16
MAX_PWM_CHANNELS = 16


@dataclass
class FDMPacket:
    """Flight Dynamics Model packet, sim -> BF.

    Total size: 144 bytes (18 packed doubles, see _FORMAT). The size is also
    asserted in `tests/test_packets.py`.
    """

    timestamp: float = 0.0  # seconds
    imu_angular_velocity_rpy: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # rad/s, body frame
    imu_linear_acceleration_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # m/s², NED body frame
    imu_orientation_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )  # w, x, y, z
    velocity_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s, ENU earth frame
    position_xyz: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # meters, ENU (lon, lat, alt for GPS)
    pressure: float = 101325.0  # Pa (sea level default)

    # Layout: timestamp(1) + gyro(3) + accel(3) + quat(4) + vel(3) + pos(3) + pressure(1) = 18 doubles.
    _FORMAT = "<18d"
    SIZE = struct.calcsize(_FORMAT)

    def pack(self) -> bytes:
        """Pack the FDM packet into bytes for UDP transmission."""
        return struct.pack(
            self._FORMAT,
            self.timestamp,
            self.imu_angular_velocity_rpy[0],
            self.imu_angular_velocity_rpy[1],
            self.imu_angular_velocity_rpy[2],
            self.imu_linear_acceleration_xyz[0],
            self.imu_linear_acceleration_xyz[1],
            self.imu_linear_acceleration_xyz[2],
            self.imu_orientation_quat[0],  # w
            self.imu_orientation_quat[1],  # x
            self.imu_orientation_quat[2],  # y
            self.imu_orientation_quat[3],  # z
            self.velocity_xyz[0],
            self.velocity_xyz[1],
            self.velocity_xyz[2],
            self.position_xyz[0],
            self.position_xyz[1],
            self.position_xyz[2],
            self.pressure,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "FDMPacket":
        """Unpack FDM packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[: cls.SIZE])
        return cls(
            timestamp=values[0],
            imu_angular_velocity_rpy=np.array(values[1:4]),
            imu_linear_acceleration_xyz=np.array(values[4:7]),
            imu_orientation_quat=np.array(values[7:11]),
            velocity_xyz=np.array(values[11:14]),
            position_xyz=np.array(values[14:17]),
            pressure=values[17],
        )


@dataclass
class RCPacket:
    """16-channel RC packet (PWM µs, typically 1000-2000), sim -> BF.

    Standard channel mapping: 0=Roll, 1=Pitch, 2=Throttle, 3=Yaw, 4-15=Aux.
    """

    timestamp: float = 0.0
    channels: np.ndarray = field(
        default_factory=lambda: np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
    )

    # Format: 1 double + 16 uint16 = 8 + 32 = 40 bytes
    _FORMAT = f"<d{MAX_RC_CHANNELS}H"
    SIZE = struct.calcsize(_FORMAT)

    def pack(self) -> bytes:
        """Pack the RC packet into bytes for UDP transmission."""
        return struct.pack(self._FORMAT, self.timestamp, *self.channels[:MAX_RC_CHANNELS])

    @classmethod
    def from_bytes(cls, data: bytes) -> "RCPacket":
        """Unpack RC packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[: cls.SIZE])
        return cls(
            timestamp=values[0],
            channels=np.array(values[1:], dtype=np.uint16),
        )


@dataclass
class ServoPacket:
    """Normalized 4-motor output, BF -> sim. [0,1] in normal mode, [-1,1] in 3D mode."""

    motor_speed: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Format: 4 floats = 16 bytes
    _FORMAT = "<4f"
    SIZE = struct.calcsize(_FORMAT)

    def pack(self) -> bytes:
        """Pack servo packet into bytes."""
        return struct.pack(self._FORMAT, *self.motor_speed[:4])

    @classmethod
    def from_bytes(cls, data: bytes) -> "ServoPacket":
        """Unpack servo packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[: cls.SIZE])
        return cls(motor_speed=np.array(values))


@dataclass
class ServoPacketRaw:
    """Raw 16-channel PWM motor output (µs), BF -> sim. Unused in the lockstep path."""

    motor_count: int = 4
    pwm_output: np.ndarray = field(default_factory=lambda: np.full(MAX_PWM_CHANNELS, 1000.0))

    # Format: 1 uint16 + 2 bytes padding + 16 floats = 68 bytes
    # C struct has uint16_t motorCount (2 bytes) followed by padding for
    # 4-byte float alignment, then float[16] array
    _FORMAT = f"<Hxx{MAX_PWM_CHANNELS}f"  # xx = 2 padding bytes
    SIZE = struct.calcsize(_FORMAT)  # 68 bytes

    def pack(self) -> bytes:
        """Pack raw servo packet into bytes."""
        # Note: struct.pack with 'xx' padding bytes doesn't require values for them
        return struct.pack(self._FORMAT, self.motor_count, *self.pwm_output[:MAX_PWM_CHANNELS])

    @classmethod
    def from_bytes(cls, data: bytes) -> "ServoPacketRaw":
        """Unpack raw servo packet from bytes."""
        if len(data) < cls.SIZE:
            raise ValueError(f"Data too short: {len(data)} < {cls.SIZE}")
        values = struct.unpack(cls._FORMAT, data[: cls.SIZE])
        return cls(
            motor_count=values[0],
            pwm_output=np.array(values[1:]),
        )


class BetaflightSyncBridge:
    """Blocking step(fdm, rc) -> motors path for Betaflight SITL lockstep.

    Pairs with Betaflight built with `SIMULATOR_GYROPID_SYNC`: each call sends
    FDM + RC, blocks on the motor reply, and returns 4 normalized motor values.
    One step here corresponds to exactly one Betaflight PID iteration.

    Usage:
        bridge = BetaflightSyncBridge()
        bridge.start()
        motors = bridge.step(fdm_packet, rc_packet)  # inside post_step
        bridge.stop()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        state_port: int = PORT_STATE,
        rc_port: int = PORT_RC,
        pwm_port: int = PORT_PWM,
        timeout_ms: int = 100,
    ):
        self.host = host
        self.state_port = state_port
        self.rc_port = rc_port
        self.pwm_port = pwm_port
        self.timeout_ms = timeout_ms

        # Sockets
        self._state_socket: Optional[socket.socket] = None
        self._rc_socket: Optional[socket.socket] = None
        self._pwm_socket: Optional[socket.socket] = None

        # State
        self._started = False
        self._step_count = 0
        self._last_motors = np.zeros(4)
        self._current_timeout_ms = timeout_ms  # Cache current timeout to avoid redundant syscalls

    def start(self) -> None:
        """Start the bridge and prepare sockets."""
        if self._started:
            return

        # Create UDP sockets for sending
        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Create and bind receiving socket with timeout
        self._pwm_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._pwm_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pwm_socket.bind(("0.0.0.0", self.pwm_port))
        self._pwm_socket.settimeout(self.timeout_ms / 1000.0)

        # Drain any stale packets from previous runs
        self._pwm_socket.setblocking(False)
        drained = 0
        try:
            while True:
                self._pwm_socket.recv(1024)
                drained += 1
        except BlockingIOError:
            pass  # No more data to drain
        self._pwm_socket.setblocking(True)
        self._pwm_socket.settimeout(self.timeout_ms / 1000.0)
        if drained > 0:
            print(f"[BetaflightSyncBridge] Drained {drained} stale packet(s)")

        self._started = True
        self._step_count = 0

        print("[BetaflightSyncBridge] Started in lockstep mode")
        print(f"  FDM -> {self.host}:{self.state_port}")
        print(f"  RC  -> {self.host}:{self.rc_port}")
        print(f"  PWM <- 0.0.0.0:{self.pwm_port} (timeout={self.timeout_ms}ms)")

    def stop(self) -> None:
        if not self._started:
            return

        # Close sockets
        for sock in [self._state_socket, self._rc_socket, self._pwm_socket]:
            if sock:
                sock.close()

        self._state_socket = None
        self._rc_socket = None
        self._pwm_socket = None
        self._started = False

        print(f"[BetaflightSyncBridge] Stopped after {self._step_count} steps")

    def step(
        self,
        fdm: FDMPacket,
        rc: RCPacket,
        timeout_ms: Optional[int] = None,
    ) -> np.ndarray:
        """Send FDM + RC, block on the motor reply, return 4 motors in [0.0, 1.0].

        Sending the FDM packet is what unblocks Betaflight's GYROPID_SYNC loop.
        Raises TimeoutError if no motor packet arrives within `timeout_ms`
        (or the bridge's default), except on the very first call where we
        return zeros to ride out Betaflight's own startup.
        """
        if not self._started:
            raise RuntimeError("Bridge not started - call start() first")

        # Send FDM packet (this unblocks Betaflight's GYROPID_SYNC)
        fdm_data = fdm.pack()
        self._state_socket.sendto(fdm_data, (self.host, self.state_port))

        # Send RC packet
        rc_data = rc.pack()
        self._rc_socket.sendto(rc_data, (self.host, self.rc_port))

        # Wait for motor response (blocking)
        # Only update timeout if explicitly changed (avoid syscall every step)
        effective_timeout_ms = timeout_ms or self.timeout_ms
        if effective_timeout_ms != self._current_timeout_ms:
            self._pwm_socket.settimeout(effective_timeout_ms / 1000.0)
            self._current_timeout_ms = effective_timeout_ms

        try:
            data, addr = self._pwm_socket.recvfrom(ServoPacket.SIZE)
            packet = ServoPacket.from_bytes(data)
            self._last_motors = packet.motor_speed.copy()
            self._step_count += 1
            return self._last_motors

        except socket.timeout:
            # Return last known motors on timeout (first few steps may timeout
            # before Betaflight is fully initialized)
            if self._step_count == 0:
                # During initialization, just return zeros
                return np.zeros(4)
            raise TimeoutError(
                f"No motor response from Betaflight within {self._current_timeout_ms:.0f}ms "
                f"(step {self._step_count})"
            )

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_motors(self) -> np.ndarray:
        return self._last_motors.copy()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
