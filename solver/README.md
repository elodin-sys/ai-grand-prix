# `solver/` — your autonomy stack

This is the only directory you should need to edit to compete.

## Contract

The simulator calls one function:

```python
from solver.api import SensorUpdate, RCCommand

def autopilot(update: SensorUpdate) -> RCCommand:
    ...
```

- **`update`** — see [`api.py`](api.py). Includes `t`, `tick`, world pose/velocity, gyro, accel, baro, mag, optional `frame_rgba`, and `*_fresh` flags for slower streams. **No GPS, no depth, no motor RPM** (matching the official AI Grand Prix sim).

Return an `RCCommand` with PWM channel values:

| Channel | PWM range | Meaning |
|---------|-----------|---------|
| `throttle` | 1000-2000 | Up |
| `roll` | 1000-2000 (1500=center) | Lean left/right |
| `pitch` | 1000-2000 (1500=center; **<1500 = forward**) | Lean forward/back |
| `yaw` | 1000-2000 (1500=center) | Spin left/right |
| `arm` | 1000 / 1800 | Disarm / Arm (AUX1) |

The simulator calls your autopilot every physics tick. The most recent return value is sent to Betaflight on the next tick, so keep per-tick work cheap or self-decimate heavier perception.

## Running your code

The default solver is [`solver.baseline`](baseline.py). To use a different module, set the environment variable:

```bash
RACE_SOLVER=my_team.my_solver elodin editor sim/main.py
```

(Module must be importable from the repo root and expose an `autopilot` function with the signature above.)

## Tips

- Camera intrinsics in [`sim/camera.py`](../sim/camera.py) match the published [AI Grand Prix](https://www.theaigrandprix.com/) VADR-TS-002 sensor spec.
- Gate inner opening is **1.5 m × 1.5 m** — use this as your PnP scale anchor.
- Camera tilts **20° upward** in body frame; account for that when projecting gate centers from image to body.
- The sim is **deterministic**: same seed + same code = same lap time. Lean on this for regression testing.
- AGP runs are capped at 8 minutes, but this practice sim defaults to a 15 s run so smoke tests stay quick. Change `simulation_time` in [`sim/config.py`](../sim/config.py) for longer attempts.
- If you rebuild Betaflight against a clean EEPROM, re-run `uv run python scripts/configure_betaflight.py` before launching the sim.
- Coordinate-frame and lockstep details live in [`ARCHITECTURE.md`](../ARCHITECTURE.md).
