"""Roundtrip pack/unpack tests for the Betaflight SITL UDP wire format."""

import numpy as np
import pytest

from sim.betaflight_bridge import (
    FDMPacket,
    RCPacket,
    ServoPacket,
    ServoPacketRaw,
    MAX_RC_CHANNELS,
    MAX_PWM_CHANNELS,
)


def test_fdm_size_is_144_bytes():
    # 18 doubles per the upstream BF SITL definition
    assert FDMPacket.SIZE == 18 * 8


def test_fdm_roundtrip():
    fdm = FDMPacket(
        timestamp=1.234,
        imu_angular_velocity_rpy=np.array([0.1, -0.2, 0.3]),
        imu_linear_acceleration_xyz=np.array([0.0, 0.5, 9.81]),
        imu_orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity_xyz=np.array([1.0, 2.0, 3.0]),
        position_xyz=np.array([10.0, 20.0, 30.0]),
        pressure=101325.0,
    )
    out = FDMPacket.from_bytes(fdm.pack())
    assert out.timestamp == pytest.approx(fdm.timestamp)
    assert np.allclose(out.imu_angular_velocity_rpy, fdm.imu_angular_velocity_rpy)
    assert np.allclose(out.imu_linear_acceleration_xyz, fdm.imu_linear_acceleration_xyz)
    assert np.allclose(out.imu_orientation_quat, fdm.imu_orientation_quat)
    assert np.allclose(out.velocity_xyz, fdm.velocity_xyz)
    assert np.allclose(out.position_xyz, fdm.position_xyz)
    assert out.pressure == pytest.approx(fdm.pressure)


def test_rc_size_is_40_bytes():
    # 1 double timestamp + 16 uint16 channels
    assert RCPacket.SIZE == 8 + 16 * 2
    assert MAX_RC_CHANNELS == 16


def test_rc_roundtrip():
    chans = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
    chans[0] = 1234
    chans[2] = 1800
    chans[4] = 1700
    rc = RCPacket(timestamp=2.5, channels=chans)
    out = RCPacket.from_bytes(rc.pack())
    assert out.timestamp == pytest.approx(rc.timestamp)
    assert np.array_equal(out.channels, chans)


def test_servo_size_and_roundtrip():
    assert ServoPacket.SIZE == 16
    motors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out = ServoPacket.from_bytes(ServoPacket(motor_speed=motors).pack())
    assert np.allclose(out.motor_speed, motors)


def test_servo_raw_size_and_roundtrip():
    assert ServoPacketRaw.SIZE == 2 + 2 + MAX_PWM_CHANNELS * 4
    pwm = np.full(MAX_PWM_CHANNELS, 1500.0, dtype=np.float32)
    pwm[0] = 1234.0
    out = ServoPacketRaw.from_bytes(
        ServoPacketRaw(motor_count=4, pwm_output=pwm).pack()
    )
    assert out.motor_count == 4
    assert np.allclose(out.pwm_output, pwm)
