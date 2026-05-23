"""
AI Grand Prix race course: gates, ordering, and pass-time tracking.

Gate dimensions follow VADR-TS-002 §3.7:
  - Outer:  2700 x 2700 mm (square frame)
  - Inner (flyable hole): 1500 x 1500 mm
  - Depth:  260 mm

Gates are placed as static visual entities. Pass detection runs as a regular
@el.map system that watches the drone's `world_pos`, advances `last_gate_passed`
when the drone crosses the next-in-order gate's plane within the inner square,
and stamps the crossing time in `gate_pass_time`.

The drone starts at the ENU origin. The default course places vertical gates
straight ahead along +X (East) at hover altitude.
"""

import typing as ty
from dataclasses import dataclass, field
from typing import Optional, Tuple

import elodin as el
import jax
import jax.numpy as jnp


# Gate dimensions, meters.
GATE_OUTER_W = 2.7
GATE_OUTER_H = 2.7
GATE_DEPTH = 0.26
GATE_INNER_W = 1.5
GATE_INNER_H = 1.5

# Frame thickness around the inner hole, per side. Matches AGP outer/inner spec.
GATE_FRAME = (GATE_OUTER_W - GATE_INNER_W) / 2.0  # 0.6 m


@dataclass(frozen=True)
class Gate:
    """A single race gate.

    `index` is the order in which it must be passed (0 = first).
    `center` is the world-frame (ENU) position of the gate's center.
    `yaw_deg` is reserved for future rotated courses. Today all gates are
        vertical hoops whose plane is perpendicular to world X (opening faces +X).
    """

    index: int
    center: Tuple[float, float, float]
    yaw_deg: float = 0.0


# 3 AGP-style vertical gates straight ahead at hover altitude. Elodin uses
# ENU world coordinates, so +X is East and +Z is Up.
EASY_COURSE: Tuple[Gate, ...] = (
    Gate(0, (10.0, 0.0, 1.8)),
    Gate(1, (20.0, 0.0, 1.8)),
    Gate(2, (30.0, 0.0, 1.8)),
)

# Index of the LAST gate the drone has crossed; -1 before any pass.
# external_control so the post_step gate-tracker can write to it.
LastGatePassed = ty.Annotated[
    jax.Array,
    el.Component(
        "last_gate_passed",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={
            "priority": 200,
            "external_control": "true",
        },
    ),
]

# Wall-clock-equivalent (sim seconds) when each gate was first crossed.
# `MAX_GATES` slots; -1.0 = "not yet passed".
MAX_GATES = 32

GatePassTimes = ty.Annotated[
    jax.Array,
    el.Component(
        "gate_pass_times",
        el.ComponentType(el.PrimitiveType.F64, (MAX_GATES,)),
        metadata={
            "priority": 199,
            "external_control": "true",
        },
    ),
]


@dataclass
class GateProgress(el.Archetype):
    """Per-drone race progress state."""

    last_gate_passed: LastGatePassed = field(
        default_factory=lambda: jnp.array([-1.0])
    )
    gate_pass_times: GatePassTimes = field(
        default_factory=lambda: jnp.full(MAX_GATES, -1.0)
    )


# Visual asset bound to each gate entity. The Elodin editor resolves GLB
# paths the same way it does for `crazyflie.glb` (the drone model), so the
# file just lives in `assets/` and is referenced by name.
GATE_ASSET = "gate.glb"

# Base rotation applied to every gate GLB. The model is authored facing
# along its native Y axis; rotating 90° about Y orients the opening to face
# +X (East) in our ENU world. Per-gate `Gate.yaw_deg` is currently always 0;
# when non-zero yaw enters the courses, it must be composed with this base
# rotation inside `schematic_for`.
GATE_BASE_ROTATE = "(0.0, 90.0, 0.0)"


def gate_name(index: int) -> str:
    """Stable Elodin entity name for the gate at `index` in the course."""
    return f"gate_{index}"


def spawn_gates(world: el.World, course: Tuple[Gate, ...]) -> list:
    """Spawn one static `el.Body` per gate so the schematic can bind a GLB
    to `gate_N.world_pos`.

    Gates carry no `Drone` archetype, so `apply_forces` (the only system
    that injects gravity / thrust / drag) never fires on them. With zero
    initial velocity and no force, `el.six_dof` integrates them in place.
    """
    ids = []
    for g in course:
        ent = world.spawn(
            [
                el.Body(
                    world_pos=el.SpatialTransform(
                        linear=jnp.array(g.center),
                        angular=el.Quaternion(
                            jnp.array([0.0, 0.0, 0.0, 1.0])
                        ),
                    ),
                    world_vel=el.SpatialMotion(
                        linear=jnp.zeros(3),
                        angular=jnp.zeros(3),
                    ),
                    inertia=el.SpatialInertia(
                        mass=1.0,
                        inertia=jnp.array([1.0, 1.0, 1.0]),
                    ),
                ),
            ],
            name=gate_name(g.index),
        )
        ids.append(ent)
    return ids


def schematic_for(course: Tuple[Gate, ...]) -> str:
    """KDL gate visuals to splice into the world schematic.

    Each gate is one `object_3d` bound to the spawned `gate_N` entity's
    `world_pos`, loading the spec-accurate `assets/gate.glb` model.
    """
    blocks: list[str] = []
    for g in course:
        blocks.append(
            f"    object_3d {gate_name(g.index)}.world_pos {{\n"
            f'        glb path="{GATE_ASSET}" rotate="{GATE_BASE_ROTATE}" translate="(0.0, 0.0, 0.0)"\n'
            f"    }}"
        )
    return "\n".join(blocks)


def detect_gate_pass(
    course: Tuple[Gate, ...],
    last_gate_passed: int,
    prev_pos: Tuple[float, float, float],
    curr_pos: Tuple[float, float, float],
) -> Optional[int]:
    """Return the index of the next gate that was just crossed, else None.

    A pass requires:
      - Drone crossed the next-in-order gate's X plane moving forward.
      - Crossing point was inside the inner 1.5 x 1.5 square in Y/Z.
    """
    next_idx = last_gate_passed + 1
    if next_idx >= len(course):
        return None
    g = course[next_idx]
    gx, gy, gz = g.center
    px, py, pz = prev_pos
    cx, cy, cz = curr_pos
    half = GATE_INNER_W / 2.0
    if not (px < gx <= cx):
        return None
    if abs(cy - gy) > half or abs(cz - gz) > GATE_INNER_H / 2.0:
        return None
    return next_idx


def print_summary(
    course: Tuple[Gate, ...],
    last_gate_passed: int,
    pass_times: list,
    final_t: float,
) -> str:
    """Build and print a single-line `[RACE]` summary. Returns the line."""
    n = len(course)
    n_passed = max(0, last_gate_passed + 1)
    if n_passed == n:
        status = "COMPLETE"
        lap_time = pass_times[n - 1] if n > 0 else final_t
    else:
        status = "DNF"
        lap_time = final_t

    laps_str = ",".join(
        f"{pass_times[i]:.2f}" if i < n_passed else "--" for i in range(n)
    )
    line = (
        f"[RACE] course=easy gates_passed={n_passed}/{n} "
        f"lap_time={lap_time:.2f}s status={status} pass_times=[{laps_str}]"
    )
    print(line)
    return line
