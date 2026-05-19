"""Baseline solver schedule and hover-controller smoke tests."""

import numpy as np

from solver import baseline
from solver.api import SensorUpdate


def update_at(
    t: float,
    x: float = 0.0,
    vx: float = 0.0,
    z: float = 0.0,
    vz: float = 0.0,
    y: float = 0.0,
    vy: float = 0.0,
    baro_fresh: bool = True,
    next_gate_index: int = 0,
) -> SensorUpdate:
    return SensorUpdate(
        t=t,
        tick=int(t * 1000),
        world_pos=np.array([0.0, 0.0, 0.0, 1.0, x, y, z]),
        world_vel=np.array([0.0, 0.0, 0.0, vx, vy, vz]),
        gyro=np.zeros(3),
        accel=np.array([0.0, 0.0, 9.81]),
        baro=z,
        baro_fresh=baro_fresh,
        mag=np.array([0.0, 1.0, 0.0]),
        mag_fresh=True,
        next_gate_index=next_gate_index,
    )


def test_baseline_starts_disarmed():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(0.1))
    assert rc.arm == 1000
    assert rc.throttle == 1000


def test_baseline_arms_at_idle_before_takeoff():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(0.6))
    assert rc.arm == 1800
    assert rc.throttle == 1000
    assert rc.aux2 == 1500


def test_baseline_commands_takeoff_when_low():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(1.0, z=0.1))
    assert rc.arm == 1800
    assert rc.throttle > 1100


def test_throttle_increases_when_below_goal_altitude():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(2.0, z=0.5, vz=0.0))
    assert rc.throttle > baseline.BASE_HOVER_PWM


def test_throttle_decreases_when_above_goal_altitude():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(2.0, z=3.0, vz=0.0))
    assert rc.throttle < baseline.BASE_HOVER_PWM


def test_pitch_centers_during_takeoff():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(2.0, z=0.2, next_gate_index=0))
    assert rc.arm == 1800
    assert rc.pitch == 1500


def test_pitch_forward_when_behind_first_gate():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(2.0, x=0.0, z=1.8, next_gate_index=0))
    assert rc.arm == 1800
    assert rc.pitch > 1500


def test_pitch_back_when_overshooting_first_gate():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(2.0, x=12.0, vx=0.0, z=1.8, next_gate_index=0))
    assert rc.arm == 1800
    assert rc.pitch < 1500


def test_roll_left_when_drone_right_of_centerline():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(2.0, x=2.0, y=0.8, z=1.8, next_gate_index=0))
    assert rc.arm == 1800
    assert rc.roll > 1500


def test_roll_right_when_drone_left_of_centerline():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(2.0, x=2.0, y=-0.8, z=1.8, next_gate_index=0))
    assert rc.arm == 1800
    assert rc.roll < 1500


def test_goal_holds_at_last_gate_after_completion():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(10.0, x=30.0, y=0.0, z=1.8, next_gate_index=-1))
    assert rc.arm == 1800
    assert abs(rc.pitch - 1500) < 5
    assert abs(rc.roll - 1500) < 5


def test_baseline_disarms_after_land_end():
    baseline.reset_state()
    rc = baseline.autopilot(update_at(baseline.T_LAND_END + 0.1, z=1.8))
    assert rc.arm == 1000
    assert rc.throttle == 1000
