"""Contract between the simulator and a contestant autonomy stack.

The simulator calls `autopilot(update: SensorUpdate) -> RCCommand` every
physics tick. Each update carries the latest available value for every sensor
plus per-sensor freshness flags so solvers can consume sparse streams without
blocking Betaflight's RC packet cadence.

Channel values are PWM microseconds in the standard Betaflight range:
  - 1000 = min (idle / disarmed)
  - 1500 = center (no input)
  - 2000 = max
  - For AUX1 (arm): >= 1700 = armed.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SensorUpdate:
    """Per-tick sensor bundle passed to the autopilot.

    Mirrors what the official AI Grand Prix sim is expected to expose
    while making each slower stream explicitly optional/fresh.
    """

    t: float
    tick: int

    # World-frame state from Elodin. Always present.
    world_pos: np.ndarray  # [qx, qy, qz, qw, x, y, z]
    world_vel: np.ndarray  # [wx, wy, wz, vx, vy, vz]

    # IMU readings. At current defaults these refresh every physics tick.
    gyro: np.ndarray
    accel: np.ndarray
    gyro_fresh: bool = True
    accel_fresh: bool = True

    # Slower sensors.
    baro: float = 0.0
    baro_fresh: bool = False
    mag: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag_fresh: bool = False

    # Latest camera frame; frame_fresh is true only once per rendered frame.
    frame_rgba: Optional[np.ndarray] = None
    frame_fresh: bool = False

    # Race context.
    last_gate_passed: int = -1
    next_gate_index: int = -1


@dataclass
class RCCommand:
    """RC channel outputs sent to Betaflight."""

    throttle: int = 1000
    roll: int = 1500
    pitch: int = 1500
    yaw: int = 1500
    arm: int = 1000
    aux2: int = 1500
    aux3: int = 1500
    aux4: int = 1500


def fill_rc_channels(
    cmd: RCCommand,
    out: np.ndarray,
) -> np.ndarray:
    """Write `cmd` into a 16-channel buffer (in-place) and return it.

    Channel mapping matches Betaflight's default AETR:
      [0]=Roll, [1]=Pitch, [2]=Throttle, [3]=Yaw, [4]=AUX1 (ARM), ...
    """
    out[0] = int(cmd.roll)
    out[1] = int(cmd.pitch)
    out[2] = int(cmd.throttle)
    out[3] = int(cmd.yaw)
    out[4] = int(cmd.arm)
    out[5] = int(cmd.aux2)
    out[6] = int(cmd.aux3)
    out[7] = int(cmd.aux4)
    return out
