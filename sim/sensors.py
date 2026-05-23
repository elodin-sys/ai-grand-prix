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
    """Gaussian measurement noise + random-walk bias, keyed on (seed, device).

    The key is folded with the current tick on every call so a re-run with the
    same code reproduces the same noise stream bit-for-bit.
    """

    def __init__(
        self,
        seed: int,
        device: int,
        noise_covariance: float,
        bias_drift_covariance: float,
    ):
        self.noise_covariance = noise_covariance
        self.bias_drift_covariance = bias_drift_covariance
        self.key = rng.fold_in(rng.key(seed), device)

    def drift_bias(self, bias: jax.Array, tick: jax.Array, dt: float) -> jax.Array:
        """Apply random walk to bias (bias drift over time)."""
        # Fold in tick, then 0, so this stream is distinct from sample()'s.
        key = rng.fold_in(rng.fold_in(self.key, tick), 0)
        std_dev = jnp.sqrt(self.bias_drift_covariance)
        drift = std_dev * rng.normal(key, shape=bias.shape, dtype=bias.dtype) * dt
        return bias + drift

    def sample(self, m: jax.Array, bias: jax.Array, tick: jax.Array) -> jax.Array:
        """Add measurement noise and bias to a measurement."""
        # Fold in tick, then 1, so this stream is distinct from drift_bias()'s.
        key = rng.fold_in(rng.fold_in(self.key, tick), 1)
        std_dev = jnp.sqrt(self.noise_covariance)
        noise = std_dev * rng.normal(key, shape=m.shape, dtype=m.dtype)
        return m + noise + bias


# Betaflight's attitude estimator is sensitive to noise during the
# bootgrace/calibration period. High noise here causes attitude drift and
# motor imbalance at liftoff, so the variances are deliberately modest.
gyro_noise = Noise(0, 0, 0.01, 0.001)
accel_noise = Noise(0, 1, 0.01, 0.001)
# sqrt(0.01) ~= 0.1 m one-sigma on barometric altitude.
baro_noise = Noise(0, 2, 0.01, 0.001)
mag_noise = Noise(0, 3, 0.01, 0.001)


# In a real sensor this would be calibrated out during Betaflight's startup;
# zeroing it for SITL avoids a consistent drift direction.
init_gyro_bias = jnp.array([0.0, 0.0, 0.0])


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
    """Per-drone IMU + baro + mag state. Each sensor advances at its own rate
    (see `create_sensor_system`); `prev_*` fields hold the last reading between
    updates so the bus always has a value to publish."""

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


@el.map
def advance_sensor_tick(tick: SensorTick) -> SensorTick:
    """Advance the sensor tick counter for deterministic RNG."""
    return tick + 1


def create_gyro_bias_drift_system(config: DroneConfig):
    """Random-walk drift of gyro bias, applied every tick."""
    dt = config.dt

    @el.map
    def update_gyro_bias(tick: SensorTick, bias: GyroBias) -> GyroBias:
        if config.sensor_noise:
            return gyro_noise.drift_bias(bias, tick, dt)
        return bias

    return update_gyro_bias


def create_gyro_system(config: DroneConfig):
    """Body-frame angular velocity readout with optional noise + bias."""

    @el.map
    def compute_gyro(
        tick: SensorTick,
        pos: el.WorldPos,
        vel: el.WorldVel,
        bias: GyroBias,
    ) -> Gyro:
        # Elodin's angular velocity vector is already body-frame.
        body_v = pos.angular().inverse() @ vel.angular()
        if config.sensor_noise:
            body_v = gyro_noise.sample(body_v, bias, tick)
        return body_v

    return compute_gyro


def create_accel_system(config: DroneConfig):
    """Body-frame specific force, decimated to `accel_tick_interval`.

    Detects ground contact so the resting-on-ground value comes out as +g: the
    ground constraint zeros velocity but does not feed a normal force back into
    the Force component, so the naive `F/m` reading would be zero there.
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
        quat_inv = pos.angular().inverse()

        z = pos.linear()[2]
        vz = vel.linear()[2]
        on_ground = (z <= ground_level + 0.01) & (vz <= 0.01)

        mass = inertia.mass()
        total_accel_from_force = force.linear() / mass

        # ENU: +Z is up.
        gravity_world = jnp.array([0.0, 0.0, -gravity])

        total_accel_world = jnp.where(
            on_ground,
            jnp.zeros(3),
            total_accel_from_force,
        )

        specific_force_world = total_accel_world - gravity_world
        body_a = quat_inv @ specific_force_world

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
        new_reading = _compute_accel_reading(tick, pos, vel, force, inertia, bias)
        accel_out = jax.lax.cond(
            tick % tick_interval == 0,
            lambda _: new_reading,
            lambda _: prev_accel,
            None,
        )
        return accel_out, accel_out

    return compute_accel


def create_body_vel_system(config: DroneConfig):
    """Body-frame linear velocity, derived from WorldVel for debug/telemetry."""

    @el.map
    def compute_body_vel(pos: el.WorldPos, vel: el.WorldVel) -> BodyVel:
        return pos.angular().inverse() @ vel.linear()

    return compute_body_vel


def create_baro_system(config: DroneConfig):
    """Altitude-as-baro readout, decimated to `baro_tick_interval`."""
    tick_interval = config.baro_tick_interval

    def _compute_baro_reading(tick: jax.Array, pos: el.SpatialTransform) -> jax.Array:
        altitude = pos.linear()[2]
        baro_reading = jnp.array([altitude])
        if config.sensor_noise:
            baro_reading = baro_noise.sample(baro_reading, jnp.zeros(1), tick)
        return baro_reading

    @el.map
    def compute_baro(
        tick: SensorTick,
        pos: el.WorldPos,
        prev_baro: PrevBaro,
    ) -> tuple[Baro, PrevBaro]:
        new_reading = _compute_baro_reading(tick, pos)
        baro_out = jax.lax.cond(
            tick % tick_interval == 0,
            lambda _: new_reading,
            lambda _: prev_baro,
            None,
        )
        return baro_out, baro_out

    return compute_baro


def create_mag_system(config: DroneConfig):
    """Magnetometer aligned to ENU +Y (North), decimated to `mag_tick_interval`."""
    tick_interval = config.mag_tick_interval

    world_mag_ref = jnp.array([0.0, 1.0, 0.0])

    def _compute_mag_reading(tick: jax.Array, pos: el.SpatialTransform) -> jax.Array:
        body_mag = pos.angular().inverse() @ world_mag_ref
        if config.sensor_noise:
            body_mag = mag_noise.sample(body_mag, jnp.zeros(3), tick)
        return body_mag

    @el.map
    def compute_mag(
        tick: SensorTick,
        pos: el.WorldPos,
        prev_mag: PrevMag,
    ) -> tuple[Mag, PrevMag]:
        new_reading = _compute_mag_reading(tick, pos)
        mag_out = jax.lax.cond(
            tick % tick_interval == 0,
            lambda _: new_reading,
            lambda _: prev_mag,
            None,
        )
        return mag_out, mag_out

    return compute_mag


def create_imu_system(config: DroneConfig) -> el.System:
    """Gyro + accel + body-velocity pipeline. No filtering; Betaflight does its own."""
    return (
        advance_sensor_tick
        | create_gyro_bias_drift_system(config)
        | create_gyro_system(config)
        | create_accel_system(config)
        | create_body_vel_system(config)
    )


def create_sensor_system(config: DroneConfig) -> el.System:
    """Full multi-rate sensor stack matching Elodin Aleph hardware:

    - Gyroscope: 4 kHz nominal (every tick at the default 1 kHz sim rate)
    - Accelerometer: 4.8 kHz (3x BMI270 @ 1.6 kHz each)
    - Barometer: 480 Hz (BMP581 continuous mode)
    - Magnetometer: 200 Hz (BMM350)
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
    """Pack the latest Elodin component values into the FDM wire format that
    Betaflight SITL expects. The interesting conversions (scalar-last to
    scalar-first quat, ENU to NED frame flips, gyro pitch pre-negation) are
    commented inline below."""
    # Import locally to dodge the circular import between this module and the
    # bridge module.
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
    """Last-seen sensor snapshot for the post_step callback, with build_fdm()."""

    def __init__(self):
        # Elodin world_pos layout: [qx, qy, qz, qw, x, y, z]. Identity quat first.
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
        return build_fdm_from_components(
            self.world_pos,
            self.world_vel,
            self.accel,
            self.gyro,
            self.timestamp,
        )
