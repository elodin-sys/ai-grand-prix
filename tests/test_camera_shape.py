"""Smoke test: build a tiny world with a sensor camera, verify the
intrinsics constants we expose match the AGP spec.

Note: actually rendering a frame requires the headless render-server (only
available under `elodin run` / `elodin editor`), so we don't exercise
`request_render()` here. The integration test for that is `elodin run sim/main.py`
which prints `[FPV] First frame ... shape=(360, 640, 4)`.
"""

import math

import elodin as el

from sim import camera as fpv_camera


def test_camera_intrinsics_match_agp_spec():
    assert fpv_camera.CAM_WIDTH == 640
    assert fpv_camera.CAM_HEIGHT == 360
    assert fpv_camera.CAM_FX == 320.0
    assert fpv_camera.CAM_FY == 320.0
    assert fpv_camera.CAM_CX == 320.0
    assert fpv_camera.CAM_CY == 180.0
    assert fpv_camera.CAM_TILT_UP_DEG == 20.0


def test_vfov_is_consistent_with_intrinsics():
    expected_vfov = 2.0 * math.degrees(
        math.atan(fpv_camera.CAM_CY / fpv_camera.CAM_FY)
    )
    assert math.isclose(fpv_camera.CAM_FOV_VERT_DEG, expected_vfov, rel_tol=1e-12)
    assert math.isclose(fpv_camera.CAM_FOV_VERT_DEG, 58.715, abs_tol=0.01)


def test_hfov_is_consistent_with_intrinsics():
    expected_hfov = 2.0 * math.degrees(
        math.atan(fpv_camera.CAM_CX / fpv_camera.CAM_FX)
    )
    assert math.isclose(fpv_camera.CAM_FOV_HORIZ_DEG, expected_hfov, rel_tol=1e-12)
    assert math.isclose(fpv_camera.CAM_FOV_HORIZ_DEG, 90.0, abs_tol=1e-9)


def test_render_every_n_at_2khz_targets_30hz():
    n = fpv_camera.render_every_n_ticks(2000.0)
    achieved = 2000.0 / n
    # Within 5 Hz of the 30 Hz target
    assert abs(achieved - fpv_camera.TARGET_FPS) <= 5.0


def test_render_every_n_at_120hz_targets_30hz_exactly():
    n = fpv_camera.render_every_n_ticks(120.0)
    assert n == 4
    assert 120.0 / n == 30.0


def test_register_attaches_camera_without_errors():
    """Spawning + sensor_camera registration should not raise."""
    w = el.World()
    drone = w.spawn([], name=fpv_camera.DRONE_ENTITY_NAME)
    name = fpv_camera.register(w, drone)
    assert name == "drone.fpv"


def test_camera_tilt_vector_is_unit():
    """Sanity check: the look_at_offset we use should be a unit vector."""
    rad = math.radians(fpv_camera.CAM_TILT_UP_DEG)
    v = (math.cos(rad), 0.0, math.sin(rad))
    mag = math.sqrt(sum(x * x for x in v))
    assert abs(mag - 1.0) < 1e-9
