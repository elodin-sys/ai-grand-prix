"""Multi-rate sensor systems for the Betaflight SITL practice rig.

This module computes gyro, accel, baro, mag, and body-velocity components with
deterministic noise. Betaflight packet conversion lives in
sim.betaflight_bridge; see ARCHITECTURE.md for the frame conventions.
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp
import jax.random as rng
import numpy as np

from sim.config import DroneConfig
from sim.betaflight_bridge import FDMPacket


class Noise:
    """
    Sensor noise model with Gaussian noise and bias drift.

    Uses JAX random keys seeded deterministically via tick counter
    for reproducible noise across simulation runs.
    """

    def __init__(
        self,
        seed: int,
        device: int,
        noise_covariance: float,
        bias_drift_covariance: float,
    ):
        """
        Initialize noise model.

        Args:
            seed: Random seed for reproducibility
            device: Device index (for different noise streams per sensor)
            noise_covariance: Variance of measurement noise
            bias_drift_covariance: Variance of bias drift per timestep
        """
        self.noise_covariance = noise_covariance
        self.bias_drift_covariance = bias_drift_covariance
        self.key = rng.fold_in(rng.key(seed), device)

    def drift_bias(self, bias: jax.Array, tick: jax.Array, dt: float) -> jax.Array:
        """Apply random walk to bias (bias drift over time)."""
        # Fold in tick, then fold in 0 to differentiate from sample() stream
        key = rng.fold_in(rng.fold_in(self.key, tick), 0)
        std_dev = jnp.sqrt(self.bias_drift_covariance)
        drift = std_dev * rng.normal(key, shape=bias.shape, dtype=bias.dtype) * dt
        return bias + drift

    def sample(self, m: jax.Array, bias: jax.Array, tick: jax.Array) -> jax.Array:
        """Add measurement noise and bias to a measurement."""
        # Fold in tick, then fold in 1 to differentiate from drift_bias() stream
        key = rng.fold_in(rng.fold_in(self.key, tick), 1)
        std_dev = jnp.sqrt(self.noise_covariance)
        noise = std_dev * rng.normal(key, shape=m.shape, dtype=m.dtype)
        return m + noise + bias


# Noise instances - tuned for Betaflight SITL testing
#
# Note: Betaflight's attitude estimator is sensitive to noise during the
# bootgrace/calibration period. High noise causes attitude drift and
# motor imbalance at liftoff.
gyro_noise = Noise(0, 0, 0.01, 0.001)  # Gyro noise + bias drift
accel_noise = Noise(0, 1, 0.01, 0.001)  # Accel noise (no drift)
baro_noise = Noise(0, 2, 0.01, 0.001)  # ~0.03m std dev
mag_noise = Noise(0, 3, 0.01, 0.001)  # Magnetometer noise (very low)


# Initial gyro bias (set to zero for SITL - avoids consistent drift direction)
# In a real sensor, this would be calibrated out during Betaflight's startup
init_gyro_bias = jnp.array([0.0, 0.0, 0.0])


# --- Sensor Component Types ---

# Sensor tick counter for deterministic RNG
SensorTick = ty.Annotated[jax.Array, el.Component("sensor_tick", el.ComponentType.U64)]

# IMU accelerometer reading in body frame [ax, ay, az] m/s^2
Accel = ty.Annotated[
    jax.Array,
    el.Component(
        "accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 150},
    ),
]

# Accelerometer bias [bx, by, bz] m/s^2
AccelBias = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

# IMU gyroscope reading in body frame [wx, wy, wz] rad/s
Gyro = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 151},
    ),
]

# Gyroscope bias [bx, by, bz] rad/s
GyroBias = ty.Annotated[
    jax.Array,
    el.Component(
        "gyro_bias",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

# Barometer altitude reading in meters
Baro = ty.Annotated[
    jax.Array,
    el.Component(
        "baro",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"priority": 152},
    ),
]

# Body-frame linear velocity (for reference/debugging)
BodyVel = ty.Annotated[
    jax.Array,
    el.Component(
        "body_vel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 153},
    ),
]

# Magnetometer reading in body frame [mx, my, mz] (normalized)
Mag = ty.Annotated[
    jax.Array,
    el.Component(
        "mag",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 154},
    ),
]

# Previous sensor readings (for multi-rate simulation - hold values between updates)
PrevAccel = ty.Annotated[
    jax.Array,
    el.Component(
        "prev_accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

PrevBaro = ty.Annotated[
    jax.Array,
    el.Component(
        "prev_baro",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
    ),
]

PrevMag = ty.Annotated[
    jax.Array,
    el.Component(
        "prev_mag",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]


@dataclass
class IMU(el.Archetype):
    """
    IMU sensor archetype with noise state and multi-rate support.

    Stores computed sensor values and bias state for the drone entity.
    No filtering state - Betaflight handles its own filtering.

    Multi-rate simulation: Sensors update at different frequencies matching
    Aleph hardware (BMI270, BMP581, BMM350). Previous values are held between
    updates via prev_* components.
    """

    sensor_tick: SensorTick = field(default_factory=lambda: jnp.array(0, dtype=jnp.uint64))

    # Current sensor readings (updated at sensor-specific rates)
    gyro: Gyro = field(default_factory=lambda: jnp.zeros(3))
    gyro_bias: GyroBias = field(default_factory=lambda: jnp.array(init_gyro_bias))
    accel: Accel = field(default_factory=lambda: jnp.array([0.0, 0.0, 9.80665]))  # 1g at rest
    accel_bias: AccelBias = field(default_factory=lambda: jnp.zeros(3))
    baro: Baro = field(default_factory=lambda: jnp.zeros(1))
    mag: Mag = field(default_factory=lambda: jnp.array([1.0, 0.0, 0.0]))  # North
    body_vel: BodyVel = field(default_factory=lambda: jnp.zeros(3))

    # Previous readings for multi-rate hold (sensors slower than PID loop)
    prev_accel: PrevAccel = field(default_factory=lambda: jnp.array([0.0, 0.0, 9.80665]))
    prev_baro: PrevBaro = field(default_factory=lambda: jnp.zeros(1))
    prev_mag: PrevMag = field(default_factory=lambda: jnp.array([1.0, 0.0, 0.0]))


# --- Sensor Computation Systems ---


@el.map
def advance_sensor_tick(tick: SensorTick) -> SensorTick:
    """Advance the sensor tick counter for deterministic RNG."""
    return tick + 1


def create_gyro_bias_drift_system(config: DroneConfig):
    """Create system to drift the gyro bias over time."""
    dt = config.dt

    @el.map
    def update_gyro_bias(tick: SensorTick, bias: GyroBias) -> GyroBias:
        """Apply random walk to gyro bias."""
        if config.sensor_noise:
            return gyro_noise.drift_bias(bias, tick, dt)
        return bias

    return update_gyro_bias


def create_gyro_system(config: DroneConfig):
    """
    Create the gyroscope sensor computation system.

    The gyroscope measures angular velocity in body frame.
    Applies noise and bias when enabled. No filtering - Betaflight handles that.
    """

    @el.map
    def compute_gyro(
        tick: SensorTick,
        pos: el.WorldPos,
        vel: el.WorldVel,
        bias: GyroBias,
    ) -> Gyro:
        """Compute gyroscope reading from physics state."""
        # Angular velocity is already in body frame in Elodin
        body_v = pos.angular().inverse() @ vel.angular()

        # Add noise and bias if enabled
        if config.sensor_noise:
            body_v = gyro_noise.sample(body_v, bias, tick)

        return body_v

    return compute_gyro


def create_accel_system(config: DroneConfig):
    """
    Create the accelerometer sensor computation system with multi-rate support.

    The accelerometer measures specific force (acceleration minus gravity)
    in body frame. Applies noise and bias when enabled.
    No filtering - Betaflight handles that.

    Multi-rate: Updates at ~4.8kHz (BMI270 1.6kHz × 3 IMUs). Between updates,
    the previous reading is held.

    Detects ground contact to correctly report +g when at rest (the ground
    constraint zeros velocity but doesn't add normal force to the Force component).
    """
    gravity = config.gravity
    ground_level = config.ground_level
    tick_interval = config.accel_tick_interval

    def _compute_accel_reading(
        tick: jax.Array,
        pos: el.SpatialTransform,
        vel: el.SpatialMotion,
        force: el.SpatialForce,
        inertia: el.SpatialInertia,
        bias: jax.Array,
    ) -> jax.Array:
        """Internal: Compute fresh accelerometer reading."""
        # Get orientation quaternion for frame transformation
        quat = pos.angular()
        quat_inv = quat.inverse()

        # Check if on ground: position at ground level and not moving upward
        z = pos.linear()[2]
        vz = vel.linear()[2]
        on_ground = (z <= ground_level + 0.01) & (vz <= 0.01)

        # Compute acceleration from forces
        mass = inertia.mass()
        total_accel_from_force = force.linear() / mass

        # Gravity vector in world frame (ENU: +Z is up)
        gravity_world = jnp.array([0.0, 0.0, -gravity])

        # When on ground, effective acceleration is 0 (velocity is clamped)
        # Otherwise use force-based acceleration
        total_accel_world = jnp.where(
            on_ground,
            jnp.zeros(3),  # On ground: no acceleration (constrained)
            total_accel_from_force,  # In air: normal physics
        )

        # Specific force in world frame (what accelerometer measures)
        specific_force_world = total_accel_world - gravity_world

        # Transform to body frame
        body_a = quat_inv @ specific_force_world

        # Add noise and bias if enabled
        if config.sensor_noise:
            body_a = accel_noise.sample(body_a, bias, tick)

        return body_a

    @el.map
    def compute_accel(
        tick: SensorTick,
        pos: el.WorldPos,
        vel: el.WorldVel,
        force: el.Force,
        inertia: el.Inertia,
        bias: AccelBias,
        prev_accel: PrevAccel,
    ) -> tuple[Accel, PrevAccel]:
        """
        Compute accelerometer reading with multi-rate decimation.

        Updates at accel_tick_interval (1 tick by default at the 1 kHz sim rate).
        Returns previous reading when not updating.
        """
        new_reading = _compute_accel_reading(tick, pos, vel, force, inertia, bias)

        # Update only on tick intervals; otherwise hold previous value
        accel_out = jax.lax.cond(
            tick % tick_interval == 0,
            lambda _: new_reading,
            lambda _: prev_accel,
            None,
        )

        return accel_out, accel_out

    return compute_accel


def create_body_vel_system(config: DroneConfig):
    """Create system to compute body-frame velocity (for reference/debugging)."""

    @el.map
    def compute_body_vel(pos: el.WorldPos, vel: el.WorldVel) -> BodyVel:
        """Transform world velocity to body frame."""
        quat_inv = pos.angular().inverse()
        return quat_inv @ vel.linear()

    return compute_body_vel


def create_baro_system(config: DroneConfig):
    """
    Create barometer sensor system with multi-rate support.

    Simulates barometric altitude measurement based on height.
    Applies noise when enabled (~0.3m std dev typical for consumer barometers).

    Multi-rate: Updates at 480Hz (BMP581 continuous mode). Between updates,
    the previous reading is held.
    """
    tick_interval = config.baro_tick_interval

    def _compute_baro_reading(tick: jax.Array, pos: el.SpatialTransform) -> jax.Array:
        """Internal: Compute fresh barometer reading."""
        # Simple model: altitude = z position
        altitude = pos.linear()[2]
        baro_reading = jnp.array([altitude])

        # Add noise if enabled
        if config.sensor_noise:
            baro_reading = baro_noise.sample(baro_reading, jnp.zeros(1), tick)

        return baro_reading

    @el.map
    def compute_baro(
        tick: SensorTick,
        pos: el.WorldPos,
        prev_baro: PrevBaro,
    ) -> tuple[Baro, PrevBaro]:
        """
        Compute barometer reading with multi-rate decimation.

        Updates at baro_tick_interval (about every 2 ticks for 480 Hz at 1 kHz).
        Returns previous reading when not updating.
        """
        new_reading = _compute_baro_reading(tick, pos)

        # Update only on tick intervals; otherwise hold previous value
        baro_out = jax.lax.cond(
            tick % tick_interval == 0,
            lambda _: new_reading,
            lambda _: prev_baro,
            None,
        )

        return baro_out, baro_out

    return compute_baro


def create_mag_system(config: DroneConfig):
    """
    Create magnetometer sensor system with multi-rate support.

    Simulates magnetometer reading for heading reference.
    Applies noise when enabled.

    Multi-rate: Updates at 200Hz (BMM350). Between updates,
    the previous reading is held.
    """
    tick_interval = config.mag_tick_interval

    # Earth's magnetic field reference (normalized, pointing North in world frame)
    # In ENU: North is +Y direction
    world_mag_ref = jnp.array([0.0, 1.0, 0.0])

    def _compute_mag_reading(tick: jax.Array, pos: el.SpatialTransform) -> jax.Array:
        """Internal: Compute fresh magnetometer reading."""
        # Transform world magnetic field to body frame
        quat_inv = pos.angular().inverse()
        body_mag = quat_inv @ world_mag_ref

        # Add noise if enabled
        if config.sensor_noise:
            body_mag = mag_noise.sample(body_mag, jnp.zeros(3), tick)

        return body_mag

    @el.map
    def compute_mag(
        tick: SensorTick,
        pos: el.WorldPos,
        prev_mag: PrevMag,
    ) -> tuple[Mag, PrevMag]:
        """
        Compute magnetometer reading with multi-rate decimation.

        Updates at mag_tick_interval (every 5 ticks for 200 Hz at 1 kHz).
        Returns previous reading when not updating.
        """
        new_reading = _compute_mag_reading(tick, pos)

        # Update only on tick intervals; otherwise hold previous value
        mag_out = jax.lax.cond(
            tick % tick_interval == 0,
            lambda _: new_reading,
            lambda _: prev_mag,
            None,
        )

        return mag_out, mag_out

    return compute_mag


def create_imu_system(config: DroneConfig) -> el.System:
    """
    Create the complete IMU sensor system with noise model.

    Combines:
    - Sensor tick advancement
    - Gyro bias drift
    - Gyroscope computation with noise
    - Accelerometer computation with noise
    - Body velocity computation

    Note: No filtering is applied - Betaflight handles its own filtering.
    This sends realistic noisy sensor data for SITL testing.

    Args:
        config: Drone configuration

    Returns:
        Combined IMU system
    """
    return (
        advance_sensor_tick
        | create_gyro_bias_drift_system(config)
        | create_gyro_system(config)
        | create_accel_system(config)
        | create_body_vel_system(config)
    )


def create_sensor_system(config: DroneConfig) -> el.System:
    """
    Create the complete sensor system with multi-rate simulation.

    Combines sensors at their realistic hardware rates:
    - Gyroscope: 4 kHz nominal (updates every tick at the default 1 kHz sim rate)
    - Accelerometer: 4.8 kHz (3x BMI270 @ 1.6 kHz each)
    - Barometer: 480 Hz (BMP581 continuous mode)
    - Magnetometer: 200 Hz (BMM350)

    Args:
        config: Drone configuration

    Returns:
        Combined sensor system
    """
    imu = create_imu_system(config)
    baro = create_baro_system(config)
    mag = create_mag_system(config)

    return imu | baro | mag


def build_fdm_from_components(
    world_pos: np.ndarray,
    world_vel: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    timestamp: float,
    gravity: float = 9.80665,
) -> FDMPacket:
    """
    Build an FDM packet directly from Elodin component data.

    This is the primary function for extracting sensor data from the simulation
    and packaging it for Betaflight. It handles:
    - Quaternion extraction from world_pos (Elodin scalar-last to Betaflight scalar-first)
    - Velocity extraction from world_vel
    - ENU to NED coordinate conversion for gyro/accel
    - Pressure calculation from altitude

    Args:
        world_pos: Position array [qx, qy, qz, qw, x, y, z] (Elodin scalar-last format)
        world_vel: Velocity array [wx, wy, wz, vx, vy, vz] (angular first in Elodin)
        accel: Accelerometer [ax, ay, az] in body frame (ENU)
        gyro: Gyroscope [wx, wy, wz] in body frame (ENU)
        timestamp: Simulation time in seconds
        gravity: Gravity constant (for reference)

    Returns:
        FDMPacket ready for transmission to Betaflight
    """
    # Import here to avoid circular dependency
    from sim.betaflight_bridge import FDMPacket

    # Extract quaternion from Elodin format [qx, qy, qz, qw] and convert to [qw, qx, qy, qz]
    quat_xyzw = np.array(world_pos[:4])
    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]

    # Extract position [x, y, z] (ENU)
    position = np.array(world_pos[4:7])

    # Elodin stores world_vel as [wx, wy, wz, vx, vy, vz]
    linear_vel = np.array(world_vel[3:6])

    # Use provided sensor readings or compute from velocity
    accel_enu = np.array(accel) if accel is not None else np.array([0.0, 0.0, gravity])
    gyro_enu = np.array(gyro) if gyro is not None else np.array(world_vel[:3])

    # Convert from Elodin FLU body frame to Betaflight FRD body frame
    #
    # Betaflight SITL (sitl.c) applies internal sign conversions to incoming data:
    #   accel: negates all axes (-X, -Y, -Z)
    #   gyro:  keeps X, negates Y and Z (X, -Y, -Z)
    #
    # We pre-compensate so that AFTER BF's conversion, correct FRD values result.
    #
    # FLU→FRD conversion (conceptually):
    #   FRD_x = FLU_x   (forward stays forward)
    #   FRD_y = -FLU_y  (right = -left)
    #   FRD_z = -FLU_z  (down = -up)
    #
    # Accelerometer: We want [FLU_x, -FLU_y, -FLU_z] after BF negates all.
    #   Send [-FLU_x, FLU_y, FLU_z] → BF gets [FLU_x, -FLU_y, -FLU_z] ✓
    accel_ned = np.array(
        [
            -accel_enu[0],  # BF: -(-X) = X
            accel_enu[1],  # BF: -Y
            accel_enu[2],  # BF: -Z
        ]
    )

    # Gyroscope: latest Betaflight's Gazebo bridge negates packet pitch
    # internally (virtualGyro Y = -packet Y). Pre-negate Elodin pitch so BF's
    # internal rate matches the physical pitch-rate sign for damping.
    gyro_ned = np.array(
        [
            gyro_enu[0],  # Roll: correct sign
            -gyro_enu[1],  # Pitch: BF negates internally
            gyro_enu[2],  # Yaw: BF negates to get -Z
        ]
    )

    # Calculate pressure from altitude (simplified atmosphere model)
    altitude = position[2]
    pressure = 101325.0 - 12.0 * altitude

    return FDMPacket(
        timestamp=timestamp,
        imu_angular_velocity_rpy=gyro_ned,
        imu_linear_acceleration_xyz=accel_ned,
        imu_orientation_quat=quat,
        velocity_xyz=linear_vel,  # ENU world velocity for Betaflight SITL
        position_xyz=position,  # ENU world position for Betaflight SITL
        pressure=pressure,
    )


class SensorDataBuffer:
    """
    Buffer for accumulating sensor data during simulation.

    This class provides a convenient way to store and retrieve sensor data
    for use in the post_step callback. It's updated during simulation and
    read by the Betaflight bridge.
    """

    def __init__(self):
        # Elodin format: [qx, qy, qz, qw, x, y, z] - identity quaternion is [0,0,0,1]
        self.world_pos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.world_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.accel = np.array([0.0, 0.0, 9.80665])  # 1g upward at rest
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.baro = np.array([0.0])
        self.timestamp = 0.0

    def update(
        self,
        world_pos: np.ndarray = None,
        world_vel: np.ndarray = None,
        accel: np.ndarray = None,
        gyro: np.ndarray = None,
        baro: np.ndarray = None,
        timestamp: float = None,
    ):
        """Update sensor buffer with new values."""
        if world_pos is not None:
            self.world_pos = np.array(world_pos)
        if world_vel is not None:
            self.world_vel = np.array(world_vel)
        if accel is not None:
            self.accel = np.array(accel)
        if gyro is not None:
            self.gyro = np.array(gyro)
        if baro is not None:
            self.baro = np.array(baro)
        if timestamp is not None:
            self.timestamp = timestamp

    def build_fdm(self) -> FDMPacket:
        """Build FDM packet from current buffer state."""
        return build_fdm_from_components(
            self.world_pos,
            self.world_vel,
            self.accel,
            self.gyro,
            self.timestamp,
        )
