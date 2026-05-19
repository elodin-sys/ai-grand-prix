"""
Forward FPV sensor camera matching the AI Grand Prix spec.

Per VADR-TS-002 §3.8 (Issue 00.02, 2026-05-08):
  - Resolution: 640 x 360
  - Principal point (cx, cy): 320, 180
  - Focal lengths (fx, fy): 320, 320
  - Camera tilt: +20° upward in body frame (NED on the spec, ENU/FLU here)
  - 30 Hz stream rate

The spec also lists "VFoV = 90°" in prose, which is inconsistent with the
intrinsics: fx = fy = 320 and cy = 180 imply

    VFoV = 2 * atan(cy / fy) = 2 * atan(180 / 320) ≈ 58.72°
    HFoV = 2 * atan(cx / fx) = 2 * atan(320 / 320) = 90°

The stated 90° matches the *horizontal* FoV computed from the same numbers,
so the prose almost certainly mislabels HFoV as VFoV. We honor the
unambiguous intrinsics rather than the prose, since (a) they define a
single self-consistent pinhole model and (b) the official simulator's
renderer will produce frames consistent with these intrinsics. See
context/agp-spec-reference.md §5 for the full reasoning.
"""

from __future__ import annotations

import math

import elodin as el
import numpy as np

CAM_WIDTH = 640
CAM_HEIGHT = 360
CAM_FX = 320.0
CAM_FY = 320.0
CAM_CX = 320.0
CAM_CY = 180.0
CAM_FOV_VERT_DEG = 2.0 * math.degrees(math.atan(CAM_CY / CAM_FY))
CAM_FOV_HORIZ_DEG = 2.0 * math.degrees(math.atan(CAM_CX / CAM_FX))
CAM_TILT_UP_DEG = 20.0
TARGET_FPS = 30.0

CAMERA_NAME = "fpv"
DRONE_ENTITY_NAME = "drone"
DB_NAME = f"{DRONE_ENTITY_NAME}.{CAMERA_NAME}"


def register(world: el.World, drone: el.EntityId) -> str:
    """Attach a forward-facing FPV camera to the drone.

    Returns the full DB component name (`drone.fpv`) for use with
    `request_render()` / `collect_frame()` in the post_step callback.
    """
    tilt_rad = math.radians(CAM_TILT_UP_DEG)
    look_at = [math.cos(tilt_rad) + 0.3, 0.0, math.sin(tilt_rad)]

    world.sensor_camera(
        entity=drone,
        name=CAMERA_NAME,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        fov=CAM_FOV_VERT_DEG,
        near=0.02,
        far=0.65,
        pos_offset=[0.3,0,0],
        look_at_offset=look_at,
        format="rgba",
        create_frustum=True,
        frustums_color=[0.0, 1.0, 0.4, 0.4],
        projection_color=[0.0, 1.0, 0.4, 0.1],
        frustums_thickness=0.008,
    )
    return DB_NAME


def render_every_n_ticks(pid_rate_hz: float) -> int:
    """How often (in physics ticks) should we render to hit ~TARGET_FPS?"""
    return max(1, round(pid_rate_hz / TARGET_FPS))


def request_render(ctx: el.StepContext, name: str) -> None:
    """Trigger a render request for a sensor camera.

    Elodin's current render API is synchronous inside this call. Keeping it
    isolated here lets the simulation call it only on camera-rate ticks while
    the solver and Betaflight RC path continue to run every physics tick.
    """
    ctx.render_cameras([name])


def collect_frame(ctx: el.StepContext, name: str) -> np.ndarray | None:
    """Read the latest rendered frame from the message log without rendering."""
    frame = ctx.read_msg(name)
    if frame is None or len(frame) == 0:
        return None
    return np.asarray(frame).reshape(CAM_HEIGHT, CAM_WIDTH, 4)
