"""Baseline autopilot: track known gate centers as position goals.

This is intentionally small and deterministic. It proves that the simulator can
stream sparse sensors into a solver every tick while Betaflight keeps receiving
a continuous RC stream.

Replace this file (or set RACE_SOLVER=mymodule) with a real
perception/planning stack to compete.
"""

from __future__ import annotations

from .api import RCCommand, SensorUpdate


# Phase boundaries (seconds since sim start). Betaflight's own power-on arming
# grace is disabled by scripts/configure_betaflight.py; the bridge warmup has
# already handled gyro calibration before t=0.
T_DISARMED_END = 0.50
T_ARM_IDLE_END = 0.75
T_LAND_END = 14.00

# Demo cheat: hardcoded from EASY_COURSE in sim/course.py. A real contestant
# solver would derive these goal positions from the FPV camera frame.
GATE_CENTERS = (
    (10.0, 0.0, 1.8),
    (20.0, 0.0, 1.8),
    (30.0, 0.0, 1.8),
)

MIN_ALT_FOR_TRANSLATION_M = 1.0
BASE_HOVER_PWM = 1135
TAKEOFF_PWM = 1300

KP_X = 70.0
KD_X = 30.0
KP_Y = 35.0
KD_Y = 80.0
KP_Z = 140.0
KD_Z = 45.0

_state = {
    "i_term": 0.0,
    "last_t": 0.0,
    "last_baro": 0.0,
}


def reset_state() -> None:
    """Reset controller integrator state for tests or repeated module use."""
    _state["i_term"] = 0.0
    _state["last_t"] = 0.0
    _state["last_baro"] = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _current_goal(update: SensorUpdate) -> tuple[float, float, float]:
    if update.next_gate_index == -1:
        return GATE_CENTERS[-1]

    index = int(_clamp(update.next_gate_index, 0, len(GATE_CENTERS) - 1))
    return GATE_CENTERS[index]


def _altitude_throttle(update: SensorUpdate, target_alt_m: float) -> int:
    """Altitude PI controller targeting the current goal's center height."""
    t = update.t
    if update.baro_fresh:
        _state["last_baro"] = update.baro

    # The baseline uses the world z channel as its primary altitude reference
    # so the smoke test is deterministic even with noisy baro samples.
    altitude = float(update.world_pos[6])
    vertical_speed = float(update.world_vel[5]) if update.world_vel.size > 5 else 0.0
    dt = max(1e-3, t - _state["last_t"])
    err = target_alt_m - altitude

    if update.baro_fresh:
        _state["i_term"] = _clamp(_state["i_term"] + err * dt * 8.0, -80.0, 80.0)

    if altitude < MIN_ALT_FOR_TRANSLATION_M and vertical_speed < 0.7:
        throttle = TAKEOFF_PWM
    else:
        throttle = BASE_HOVER_PWM + KP_Z * err - KD_Z * vertical_speed + _state["i_term"]

    _state["last_t"] = t
    return int(round(_clamp(throttle, 1000, 1600)))


def _track_goal(update: SensorUpdate, goal: tuple[float, float, float]) -> RCCommand:
    x = float(update.world_pos[4])
    y = float(update.world_pos[5])
    z = float(update.world_pos[6])
    vx = float(update.world_vel[3]) if update.world_vel.size > 3 else 0.0
    vy = float(update.world_vel[4]) if update.world_vel.size > 4 else 0.0

    throttle = _altitude_throttle(update, goal[2])
    pitch = 1500
    roll = 1500

    if z >= MIN_ALT_FOR_TRANSLATION_M:
        dx = goal[0] - x
        dy = goal[1] - y
        pitch = int(round(_clamp(1500.0 + KP_X * dx - KD_X * vx, 1450, 1550)))
        roll = int(round(_clamp(1500.0 - KP_Y * dy + KD_Y * vy, 1450, 1550)))

    return RCCommand(
        arm=1800,
        throttle=throttle,
        roll=roll,
        pitch=pitch,
    )


def autopilot(update: SensorUpdate) -> RCCommand:
    t = update.t

    if t < T_DISARMED_END:
        # Let Betaflight see the arm switch low after RX is live.
        return RCCommand(arm=1000, throttle=1000)

    if t < T_ARM_IDLE_END:
        # Arm at idle briefly so Betaflight sees a clean switch transition.
        return RCCommand(arm=1800, throttle=1000)

    if t >= T_LAND_END:
        return RCCommand(arm=1000, throttle=1000)

    return _track_goal(update, _current_goal(update))
