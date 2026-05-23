"""Elodin physics graph for the AI Grand Prix practice quad.

The inner-loop controller lives in Betaflight SITL. This module owns only the
plant: motor dynamics, body thrust/torque, drag, ground constraint, and the
six-DOF integrator. See ARCHITECTURE.md for the full simulator data flow.
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp

from sim.config import DroneConfig


# Motor commands from Betaflight (normalized 0..1). external_control lets the
# bridge write to it via Elodin-DB.
#
# Latest Betaflight SITL motor output order is BR=0, FR=1, BL=2, FL=3. See
# config.py for the matching motor positions and spin directions.
MotorCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_command",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "element_names": "BR,FR,BL,FL",
            "priority": 100,
            "external_control": "true",
        },
    ),
]

MotorThrust = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "BR,FR,BL,FL", "priority": 99},
    ),
]

BodyThrust = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "body_thrust",
        metadata={"priority": 98, "element_names": "τx,τy,τz,fx,fy,fz"},
    ),
]

BodyDrag = ty.Annotated[
    jax.Array,
    el.Component(
        "body_drag",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "fx,fy,fz"},
    ),
]

SimTime = ty.Annotated[
    jax.Array,
    el.Component(
        "sim_time",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"priority": 200},
    ),
]


@dataclass
class Drone(el.Archetype):
    """Physics-state archetype attached to each simulated drone entity."""

    motor_command: MotorCommand = field(default_factory=lambda: jnp.zeros(4))
    motor_thrust: MotorThrust = field(default_factory=lambda: jnp.zeros(4))
    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    sim_time: SimTime = field(default_factory=lambda: jnp.zeros(1))


def create_motor_dynamics(config: DroneConfig):
    """First-order motor lag: `thrust' = (commanded - thrust) / time_constant`."""
    dt = config.dt
    tau = config.motor_time_constant
    max_thrust = config.motor_max_thrust
    alpha = dt / (dt + tau)

    @el.map
    def motor_dynamics(cmd: MotorCommand, thrust: MotorThrust) -> MotorThrust:
        cmd_clamped = jnp.clip(cmd, 0.0, 1.0)
        target_thrust = cmd_clamped * max_thrust
        return thrust + alpha * (target_thrust - thrust)

    return motor_dynamics


def create_body_thrust_system(config: DroneConfig):
    """Sum per-motor thrusts into the body-frame force + torque pair."""
    motor_positions = jnp.array(config.motor_positions)
    thrust_directions = jnp.array(config.motor_thrust_directions)
    spin_directions = jnp.array(config.motor_spin_directions)
    torque_coeff = config.motor_torque_coeff

    torque_arms = jnp.cross(motor_positions, thrust_directions)

    @el.map
    def compute_body_thrust(thrust: MotorThrust) -> BodyThrust:
        total_force = jnp.sum(thrust[:, None] * thrust_directions, axis=0)
        diff_torque = jnp.sum(thrust[:, None] * torque_arms, axis=0)
        # Yaw reaction torque from each motor's spin direction.
        yaw_torque = jnp.sum(thrust * spin_directions) * torque_coeff
        total_torque = diff_torque + jnp.array([0.0, 0.0, yaw_torque])
        return el.SpatialForce(linear=total_force, torque=total_torque)

    return compute_body_thrust


def create_drag_system(config: DroneConfig):
    """Quadratic drag `F = -k * |v| * v` on world velocity."""
    linear_drag = jnp.array(config.linear_drag)

    @el.map
    def compute_drag(vel: el.WorldVel) -> BodyDrag:
        v = vel.linear()
        v_mag = jnp.linalg.norm(v)
        return -linear_drag * v_mag * v

    return compute_drag


def create_apply_forces_system(config: DroneConfig):
    """Rotate body thrust to world frame, add gravity and drag, write el.Force."""
    gravity_vec = jnp.array([0.0, 0.0, -config.gravity])
    angular_drag = jnp.array(config.angular_drag)

    @el.map
    def apply_forces(
        thrust: BodyThrust,
        drag: BodyDrag,
        pos: el.WorldPos,
        vel: el.WorldVel,
        inertia: el.Inertia,
        force: el.Force,
    ) -> el.Force:
        quat = pos.angular()
        world_thrust = quat @ thrust
        gravity_force = el.SpatialForce(linear=gravity_vec * inertia.mass())
        drag_force = el.SpatialForce(linear=drag)

        omega = vel.angular()
        omega_mag = jnp.linalg.norm(omega)
        angular_drag_torque = -angular_drag * omega_mag * omega
        angular_drag_force = el.SpatialForce(torque=angular_drag_torque)

        return force + world_thrust + gravity_force + drag_force + angular_drag_force

    return apply_forces


def create_ground_constraint_system(config: DroneConfig):
    """Clamp z >= ground_level and bleed angular velocity near the ground.

    The angular bleed is an integrator-stability aid for takeoff, not a real
    ground-effect or contact-physics model. Damping ramps from 0.95 at ground
    contact to 0 by 0.5 m altitude so the drone doesn't tip on lift-off.
    """
    ground_level = config.ground_level
    max_damping = 0.95
    damping_start = ground_level + 0.01
    damping_end = ground_level + 0.5

    @el.map
    def ground_constraint(pos: el.WorldPos, vel: el.WorldVel) -> tuple[el.WorldPos, el.WorldVel]:
        p = pos.linear()
        v = vel.linear()
        omega = vel.angular()

        below_ground = p[2] < ground_level
        new_z = jnp.where(below_ground, ground_level, p[2])
        new_vz = jnp.where(below_ground & (v[2] < 0), 0.0, v[2])

        damping_ratio = jnp.clip((damping_end - p[2]) / (damping_end - damping_start), 0.0, 1.0)
        damping_factor = max_damping * damping_ratio
        new_omega = omega * (1.0 - damping_factor)

        new_pos = el.SpatialTransform(
            linear=jnp.array([p[0], p[1], new_z]),
            angular=pos.angular(),
        )
        new_vel = el.SpatialMotion(
            linear=jnp.array([v[0], v[1], new_vz]),
            angular=new_omega,
        )

        return new_pos, new_vel

    return ground_constraint


def create_time_update_system(config: DroneConfig):
    """Walk SimTime forward by one dt each tick so the editor has a clock."""
    dt = config.dt

    @el.map
    def update_time(t: SimTime) -> SimTime:
        return t + dt

    return update_time


def create_world(config: DroneConfig) -> tuple[el.World, el.EntityId]:
    """Build a fresh world with one drone entity seeded from `config`."""
    world = el.World()

    initial_pos = el.SpatialTransform(
        linear=jnp.array(config.initial_position),
        angular=el.Quaternion(jnp.array(config.initial_quaternion)),
    )
    initial_vel = el.SpatialMotion(
        linear=jnp.array(config.initial_velocity),
        angular=jnp.array(config.initial_angular_velocity),
    )
    inertia = el.SpatialInertia(
        mass=config.mass,
        inertia=jnp.array(config.inertia_diagonal),
    )

    drone = world.spawn(
        [
            el.Body(
                world_pos=initial_pos,
                world_vel=initial_vel,
                inertia=inertia,
            ),
            Drone(sim_time=jnp.array([0.0])),
        ],
        name="drone",
    )

    return world, drone


def create_physics_system(config: DroneConfig) -> el.System:
    """Compose the effector pipeline with the six-DOF integrator and post-steps."""
    motor_dynamics = create_motor_dynamics(config)
    body_thrust = create_body_thrust_system(config)
    drag = create_drag_system(config)
    apply_forces = create_apply_forces_system(config)
    ground = create_ground_constraint_system(config)
    time_update = create_time_update_system(config)

    effectors = motor_dynamics | body_thrust | drag | apply_forces

    physics = el.six_dof(
        config.dt,
        effectors,
        integrator=el.Integrator.SemiImplicit,
    )

    post_systems = ground | time_update

    return physics | post_systems

