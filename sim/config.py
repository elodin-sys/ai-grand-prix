"""Configuration presets for the AI Grand Prix practice quad.

The default is a conservative 5-inch Betaflight Quad-X model with motor order
matching current SITL output: BR, FR, BL, FL. See ARCHITECTURE.md for frame and
motor-layout details.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Self
import numpy as np
from numpy.typing import NDArray


class _classproperty:
    """Descriptor that works like @property but on the class itself (Python 3.13+)."""

    def __init__(self, func):
        self.fget = func

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.fget(cls)


@dataclass
class MotorConfig:
    """Configuration for a single motor."""

    # Motor position relative to CoM [x, y, z] in meters
    position: NDArray[np.float64]

    # Thrust direction (unit vector, typically [0, 0, 1] for up)
    thrust_direction: NDArray[np.float64]

    # Spin direction: 1 for CCW (positive torque), -1 for CW (negative torque)
    spin_direction: float

    # Maximum thrust in Newtons
    max_thrust: float = 10.0

    # Motor time constant (how fast motor responds) in seconds
    time_constant: float = 0.02

    # Torque coefficient (torque = k * thrust)
    torque_coefficient: float = 0.01


@dataclass
class DroneConfig:
    """
    Complete drone configuration for Betaflight SITL simulation.

    Physical parameters are chosen to match a typical 5" racing quadcopter.
    """

    _GLOBAL: ClassVar[Optional[Self]] = None

    # Total mass in kg (typical 5" quad with battery)
    mass: float = 0.8

    # Moment of inertia diagonal [Ixx, Iyy, Izz] in kg*m^2 for a 5" quad.
    inertia_diagonal: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0025, 0.0025, 0.004])
    )

    # Arm length from center to motor in meters (half of motor-to-motor distance)
    arm_length: float = 0.12

    # Conservative per-motor max thrust (N). Betaflight SITL often drives mixed
    # outputs hard; keeping this modest stops the smoke-test airframe from
    # rocketing away while still allowing hover.
    motor_max_thrust: float = 8.6

    # Motor time constant in seconds (response time)
    motor_time_constant: float = 0.02

    # Torque coefficient: reaction_torque = k * thrust. Sets yaw authority.
    motor_torque_coeff: float = 0.012

    # Linear drag coefficient [drag_x, drag_y, drag_z] in N/(m/s)
    linear_drag: NDArray[np.float64] = field(default_factory=lambda: np.array([0.2, 0.2, 40.0]))

    # Rotational drag coefficient [drag_roll, drag_pitch, drag_yaw] in N*m/(rad/s)
    angular_drag: NDArray[np.float64] = field(default_factory=lambda: np.array([0.01, 0.01, 0.015]))

    # Initial position [x, y, z] in meters (ENU)
    initial_position: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 0.1]))

    # Initial velocity [vx, vy, vz] in m/s
    initial_velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    # Initial attitude as quaternion [x, y, z, w] (Elodin scalar-last). Identity.
    initial_quaternion: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )

    # Initial angular velocity [wx, wy, wz] in rad/s
    initial_angular_velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    # Physics/PID lockstep rate in Hz. 1 kHz keeps the editor responsive while
    # preserving enough timing margin for stock Betaflight SITL PID/filter gains.
    simulation_rate: float = 1000.0

    # Total simulation time in seconds
    simulation_time: float = 15.0

    # Enable sensor noise simulation.
    sensor_noise: bool = True

    # Sensor update rates labelled at Elodin Aleph hardware rates. Sensors slower
    # than pid_rate are decimated via *_tick_interval in sim/sensors.py.

    gyro_rate: float = 4000.0
    # BMI270: 1.6 kHz x 3 IMUs.
    accel_rate: float = 4800.0
    # BMP581 continuous mode.
    baro_rate: float = 480.0
    # BMM350.
    mag_rate: float = 200.0

    # Forward FPV camera render rate. The solver runs every physics tick and
    # receives frame_fresh=True only when a new frame has been collected.
    fpv_rate: float = 30.0

    # Gravity in m/s^2 (positive in ENU, which means upward).
    gravity: float = 9.81

    # Sea-level air density in kg/m^3.
    air_density: float = 1.225

    # Ground level in meters.
    ground_level: float = 0.0

    @property
    def dt(self) -> float:
        """Physics time step."""
        return 1.0 / self.simulation_rate

    @property
    def pid_rate(self) -> float:
        """PID loop rate in Hz (inverse of time step)."""
        return self.simulation_rate

    @property
    def total_sim_ticks(self) -> int:
        """Total number of simulation ticks."""
        return int(self.simulation_time / self.dt)

    @property
    def gyro_tick_interval(self) -> int:
        """Ticks between gyro updates (1 = every tick)."""
        return max(1, round(self.pid_rate / self.gyro_rate))

    @property
    def accel_tick_interval(self) -> int:
        """Ticks between accelerometer updates."""
        return max(1, round(self.pid_rate / self.accel_rate))

    @property
    def baro_tick_interval(self) -> int:
        """Ticks between barometer updates."""
        return max(1, round(self.pid_rate / self.baro_rate))

    @property
    def mag_tick_interval(self) -> int:
        """Ticks between magnetometer updates."""
        return max(1, round(self.pid_rate / self.mag_rate))

    @property
    def fpv_tick_interval(self) -> int:
        """Ticks between FPV render requests."""
        return max(1, round(self.pid_rate / self.fpv_rate))

    @property
    def motor_positions(self) -> NDArray[np.float64]:
        """Motor positions in body FLU, ordered [BR, FR, BL, FL] to match the
        raw Betaflight SITL motor output order. See ARCHITECTURE.md for the
        full Quad-X layout and the matching spin/torque sign tables."""
        d = self.arm_length * np.sqrt(2) / 2

        return np.array(
            [
                [-d, -d, 0.0],  # BR
                [d, -d, 0.0],   # FR
                [-d, d, 0.0],   # BL
                [d, d, 0.0],    # FL
            ]
        )

    @property
    def motor_thrust_directions(self) -> NDArray[np.float64]:
        """All four thrust vectors point up in body FLU."""
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def motor_spin_directions(self) -> NDArray[np.float64]:
        """Yaw reaction torque sign per motor in [BR, FR, BL, FL] order.

        +1 produces +Z torque, -1 produces -Z. Pattern follows Betaflight Quad-X
        "props out".
        """
        return np.array([1.0, -1.0, -1.0, 1.0])

    @property
    def motor_torque_axes(self) -> NDArray[np.float64]:
        """Per-motor roll/pitch torque arms (position x thrust_direction)."""
        return np.cross(self.motor_positions, self.motor_thrust_directions)

    @property
    def hover_throttle(self) -> float:
        """Throttle fraction (mass*g / 4*max_thrust) needed to hover the airframe."""
        hover_thrust = self.mass * self.gravity
        return hover_thrust / (4 * self.motor_max_thrust)

    def get_motor_config(self, index: int) -> MotorConfig:
        return MotorConfig(
            position=self.motor_positions[index],
            thrust_direction=self.motor_thrust_directions[index],
            spin_direction=self.motor_spin_directions[index],
            max_thrust=self.motor_max_thrust,
            time_constant=self.motor_time_constant,
            torque_coefficient=self.motor_torque_coeff,
        )

    @_classproperty
    def GLOBAL(cls) -> Self:
        """Get the global configuration instance."""
        if cls._GLOBAL is None:
            raise ValueError("No global config set. Call set_as_global() first.")
        return cls._GLOBAL

    def set_as_global(self) -> None:
        """Set this configuration as the global instance."""
        DroneConfig._GLOBAL = self


# Pre-configured drone types
def create_5inch_racing_quad() -> DroneConfig:
    """Create configuration for a typical 5" racing quadcopter."""
    return DroneConfig(
        mass=0.65,
        inertia_diagonal=np.array([0.0020, 0.0020, 0.0035]),
        arm_length=0.11,
        motor_max_thrust=14.0,
        motor_time_constant=0.015,
        motor_torque_coeff=0.010,
        linear_drag=np.array([0.15, 0.15, 0.25]),
        angular_drag=np.array([0.008, 0.008, 0.012]),
    )


def create_3inch_cinewhoop() -> DroneConfig:
    """Create configuration for a 3" cinewhoop style quad."""
    return DroneConfig(
        mass=0.35,
        inertia_diagonal=np.array([0.0008, 0.0008, 0.0015]),
        arm_length=0.08,
        motor_max_thrust=6.0,
        motor_time_constant=0.025,
        motor_torque_coeff=0.008,
        linear_drag=np.array([0.3, 0.3, 0.4]),  # Higher drag from ducts
        angular_drag=np.array([0.015, 0.015, 0.020]),
    )


def create_7inch_long_range() -> DroneConfig:
    """Create configuration for a 7" long range quadcopter."""
    return DroneConfig(
        mass=1.2,
        inertia_diagonal=np.array([0.0045, 0.0045, 0.008]),
        arm_length=0.16,
        motor_max_thrust=18.0,
        motor_time_constant=0.030,
        motor_torque_coeff=0.015,
        linear_drag=np.array([0.2, 0.2, 0.3]),
        angular_drag=np.array([0.012, 0.012, 0.018]),
    )


DEFAULT_CONFIG = DroneConfig()
