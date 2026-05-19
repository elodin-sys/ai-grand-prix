"""Editor-only visualization components for the drone model.

These systems do not feed back into physics or Betaflight. They derive
propeller animation angles and thrust-arrow vectors from the existing motor
thrust component so the Elodin editor reflects the live vehicle state.
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp

from sim.config import DroneConfig
from sim.physics import MotorThrust


MAX_VIS_RPM = 25_000.0
MAX_ARROW_LEN = 0.4
MIN_ARROW_LEN = 0.001


PropellerAngle = ty.Annotated[
    jax.Array,
    el.Component(
        "propeller_angle",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "BR,FR,BL,FL", "priority": 94},
    ),
]

ThrustVizM0 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m0", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
ThrustVizM1 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m1", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
ThrustVizM2 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m2", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
ThrustVizM3 = ty.Annotated[
    jax.Array,
    el.Component("thrust_viz_m3", el.ComponentType(el.PrimitiveType.F64, (3,))),
]


@dataclass
class DroneViz(el.Archetype):
    """Visualization-only state for editor widgets and GLB animation."""

    propeller_angle: PropellerAngle = field(default_factory=lambda: jnp.zeros(4))
    thrust_viz_m0: ThrustVizM0 = field(default_factory=lambda: jnp.array([0.0, 0.0, -MIN_ARROW_LEN]))
    thrust_viz_m1: ThrustVizM1 = field(default_factory=lambda: jnp.array([0.0, 0.0, -MIN_ARROW_LEN]))
    thrust_viz_m2: ThrustVizM2 = field(default_factory=lambda: jnp.array([0.0, 0.0, -MIN_ARROW_LEN]))
    thrust_viz_m3: ThrustVizM3 = field(default_factory=lambda: jnp.array([0.0, 0.0, -MIN_ARROW_LEN]))


def create_visualization_system(config: DroneConfig) -> el.System:
    """Create propeller animation and thrust-arrow visualization systems."""

    dt = config.dt
    max_thrust = config.motor_max_thrust

    # Motor order is BR, FR, BL, FL. These are propeller spin directions, which
    # are opposite the reaction-torque signs used by the physics model.
    spin_direction = jnp.array([-1.0, 1.0, 1.0, -1.0])

    @el.map
    def propeller_animation(thrust: MotorThrust, prev_angle: PropellerAngle) -> PropellerAngle:
        """Accumulate propeller angles in degrees for GLB joint animation."""
        normalized = jnp.clip(thrust / max_thrust, 0.0, 1.0)
        omega_deg_per_second = normalized * MAX_VIS_RPM * 6.0 * spin_direction
        angle = prev_angle + omega_deg_per_second * dt
        return jnp.mod(angle + 180.0, 360.0) - 180.0

    @el.map
    def thrust_visualization(
        thrust: MotorThrust,
    ) -> tuple[ThrustVizM0, ThrustVizM1, ThrustVizM2, ThrustVizM3]:
        """Scale body-frame downward arrows by live per-motor thrust."""

        def arrow(t: jax.Array) -> jax.Array:
            normalized = jnp.clip(t / max_thrust, 0.0, 1.0)
            length = MIN_ARROW_LEN + normalized * (MAX_ARROW_LEN - MIN_ARROW_LEN)
            return jnp.array([0.0, 0.0, -length])

        return (
            arrow(thrust[0]),
            arrow(thrust[1]),
            arrow(thrust[2]),
            arrow(thrust[3]),
        )

    return propeller_animation | thrust_visualization
