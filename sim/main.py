#!/usr/bin/env python3
"""
AI Grand Prix practice simulator entry point.

This wires Elodin physics, Betaflight SITL, the FPV camera, gate tracking, and
the contestant solver hook into one lockstep run.

Usage:
    elodin editor sim/main.py  # interactive viewport
    elodin run sim/main.py     # headless, s10-managed

Prerequisites:
    1. Build Betaflight SITL: bash scripts/build_betaflight.sh
    2. Optional, only if eeprom.bin is missing or stale:
       uv run python scripts/configure_betaflight.py
"""

import importlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make the repo root importable when invoked as `elodin run sim/main.py`
# (the Elodin CLI doesn't auto-add the project root to sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import elodin as el
import jax.numpy as jnp
import numpy as np

from sim.config import DEFAULT_CONFIG
from sim.physics import Drone, create_physics_system
from sim.sensors import IMU, create_sensor_system, SensorDataBuffer
from sim.visualization import DroneViz, create_visualization_system
from sim.betaflight_bridge import (
    BetaflightSyncBridge,
    RCPacket,
    MAX_RC_CHANNELS,
)
from sim import camera as fpv_camera
from sim import course as race_course
from solver.api import RCCommand, SensorUpdate, fill_rc_channels


try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


# --- Configuration ---
config = DEFAULT_CONFIG
config.set_as_global()


# --- Betaflight Binary Path ---
REPO_ROOT = _REPO_ROOT
BETAFLIGHT_PATH = REPO_ROOT / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"

if not BETAFLIGHT_PATH.exists():
    print(f"ERROR: Betaflight SITL not found at {BETAFLIGHT_PATH}")
    print("Run:")
    print("  bash scripts/fetch_betaflight.sh")
    print("  bash scripts/build_betaflight.sh")
    sys.exit(1)


# --- Clean up stale processes from previous runs ---
# This runs BEFORE s10 starts, so it only affects leftover processes from
# previous interrupted simulations, not the current run's Betaflight.
def cleanup_stale_betaflight():
    """Kill any stale Betaflight SITL processes from previous runs."""
    try:
        subprocess.run(["pkill", "-f", "betaflight_SITL"], capture_output=True, timeout=5)
        time.sleep(0.1)  # Brief pause to let the process terminate
    except Exception:
        pass


# Only cleanup when running with s10 (without --no-s10 flag)
# s10 will start a fresh Betaflight after world.run() begins
if "--no-s10" not in sys.argv:
    cleanup_stale_betaflight()


# --- World Creation ---
world = el.World()

# Active race course: straight ahead along ENU +X at hover altitude.
ACTIVE_COURSE = race_course.EASY_COURSE

drone = world.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(
                linear=jnp.array(config.initial_position),
                angular=el.Quaternion(jnp.array(config.initial_quaternion)),
            ),
            world_vel=el.SpatialMotion(
                linear=jnp.array(config.initial_velocity),
                angular=jnp.array(config.initial_angular_velocity),
            ),
            inertia=el.SpatialInertia(
                mass=config.mass,
                inertia=jnp.array(config.inertia_diagonal),
            ),
        ),
        Drone(),
        DroneViz(),
        IMU(),
        race_course.GateProgress(),
    ],
    name="drone",
)

# Spawn one static entity per gate so the schematic can bind `gate.glb` to
# `gate_N.world_pos`. Must run before `world.schematic(...)` so the entity
# names resolve when the schematic is registered.
gate_ids = race_course.spawn_gates(world, ACTIVE_COURSE)

# Attach a forward FPV camera matching the AI Grand Prix VADR-TS-002 spec.
FPV_CAM_NAME = fpv_camera.register(world, drone)
RENDER_EVERY = config.fpv_tick_interval

# Editor schematic for visualization
world.schematic(
    """
    timeline follow_latest=#true

    tabs {{
        hsplit name="Race" {{
            viewport name=Viewport pos="drone.world_pos + (0,0,0,0, -3.5,0,1.5)" look_at="drone.world_pos" show_grid=#true show_frustums=#true active=#true
            vsplit share=0.4 {{
                sensor_view "drone.fpv" name="FPV Camera (640x360)"
                graph "drone.motor_command" name="Motor Commands"
                graph "drone.world_pos.linear()" name="Position (ENU)"
            }}
        }}
        vsplit name="Motors" {{
            graph "drone.motor_thrust" name="Motor Thrust (N)"
            graph "drone.motor_command" name="Motor Command (from BF)"
            graph "drone.propeller_angle" name="Propeller Angle"
            graph "drone.world_vel.linear()" name="World Velocity"
        }}
    }}
    object_3d drone.world_pos {{
        glb path="crazyflie.glb" rotate="(0.0, 0.0, 0.0)" translate="(0.0, 0.0, 0.0)" scale=2.7
        animate joint="Root.Propeller_0" rotation_vector="(0, drone.propeller_angle[1], 0)"
        animate joint="Root.Propeller_1" rotation_vector="(0, drone.propeller_angle[3], 0)"
        animate joint="Root.Propeller_2" rotation_vector="(0, drone.propeller_angle[2], 0)"
        animate joint="Root.Propeller_3" rotation_vector="(0, drone.propeller_angle[0], 0)"
    }}
    vector_arrow "drone.thrust_viz_m0" origin="drone.world_pos + (0,0,0,0, -0.0848,-0.0848,0.05)" body_frame=#true {{
        color red 30
    }}
    vector_arrow "drone.thrust_viz_m1" origin="drone.world_pos + (0,0,0,0, 0.0848,-0.0848,0.05)" body_frame=#true {{
        color cyan 30
    }}
    vector_arrow "drone.thrust_viz_m2" origin="drone.world_pos + (0,0,0,0, -0.0848,0.0848,0.05)" body_frame=#true {{
        color cyan 30
    }}
    vector_arrow "drone.thrust_viz_m3" origin="drone.world_pos + (0,0,0,0, 0.0848,0.0848,0.05)" body_frame=#true {{
        color red 30
    }}
    line_3d frame="ENU" drone.world_pos line_width=2.5 {{
        color 255 196 0
    }}
    object_3d "(0,0,0,1, 0,0,0)" {{
        plane width=80 depth=80 {{
            color 60 80 60
        }}
    }}
{gates}
    """.format(gates=race_course.schematic_for(ACTIVE_COURSE)),
    "ai-grand-prix.kdl",
)


# --- System ---
physics = create_physics_system(config)
sensors = create_sensor_system(config)
visualization = create_visualization_system(config)
system = physics | sensors | visualization


# --- Betaflight Process Management via s10 ---
# Register Betaflight SITL as an s10 process recipe
# s10 will manage the process lifecycle (start/stop) in all execution contexts
betaflight_recipe = el.s10.PyRecipe.process(
    name="Betaflight SITL",
    cmd=str(BETAFLIGHT_PATH),
    cwd=str(REPO_ROOT),
)
world.recipe(betaflight_recipe)

print(f"[CFG] Betaflight SITL: {BETAFLIGHT_PATH.name}")
print(f"[CFG] Simulation: {config.simulation_time}s at {config.pid_rate:.0f}Hz PID loop")
print(
    f"[CFG] Sensor rates: gyro={config.gyro_rate:.0f}Hz, accel={config.accel_rate:.0f}Hz, baro={config.baro_rate:.0f}Hz, mag={config.mag_rate:.0f}Hz"
)


# --- SITL State ---
@dataclass
class SITLState:
    """State for SITL synchronization."""

    throttle: int = 1000
    arm: int = 1000
    tick: int = 0
    sim_time: float = 0.0
    motors: np.ndarray = None
    max_motor: float = 0.0

    def __post_init__(self):
        if self.motors is None:
            self.motors = np.zeros(4)


# Calculate max ticks for completion detection
MAX_TICKS = int(config.simulation_time / config.dt)

# Shared state (using lists for mutable closure)
bridge = [None]
sensor_buf = [None]
state = [None]
start_time = [None]
last_print = [0.0]
_completed = [False]

# Pre-allocated buffers to avoid allocation in hot loop
_rc_channels_buffer = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)

# Camera-frame counters
_fpv_frames = [0]
_fpv_first_logged = [False]
_last_render_tick = [-1]
_latest_frame_tick = [-1]
_last_consumed_frame_tick = [-1]
_warmup_done_tick = [-1]

# Race state: tracked host-side across post_step calls
_race_prev_pos = [None]          # previous tick's drone (x,y,z) for plane-crossing
_race_last_gate = [-1]           # index of last gate passed
_race_pass_times: list = [-1.0] * race_course.MAX_GATES

# Load the solver module (default: solver.baseline; override with RACE_SOLVER env var)
_SOLVER_MODULE_NAME = os.environ.get("RACE_SOLVER", "solver.baseline")
print(f"[SOLVER] using module: {_SOLVER_MODULE_NAME}")
_solver_module = importlib.import_module(_SOLVER_MODULE_NAME)
_current_rc = [RCCommand()]      # most recent solver output, held between calls
_latest_frame = [None]           # last RGBA frame returned by render_camera


def sitl_post_step(tick: int, ctx: el.StepContext):
    """
    Post-step callback for lockstep SITL synchronization.

    This implements the two-phase synchronization pattern:
    1. Send sensor data (FDM) and RC inputs to Betaflight
    2. Wait for motor response (blocking - this is the lockstep sync point)
    3. Write motor commands back to Elodin-DB via ctx.write_component()

    See ARCHITECTURE.md ("Lockstep cycle") for the full sequence diagram.
    """
    if _completed[0]:
        return

    # Lazy initialization - only start bridge when first tick runs
    if bridge[0] is None or not getattr(bridge[0], "_started", False):
        print("[SITL] Initializing bridge...")
        pending_bridge = BetaflightSyncBridge(timeout_ms=100)
        sensor_buf[0] = SensorDataBuffer()
        state[0] = SITLState()
        pending_bridge.start()
        bridge[0] = pending_bridge
        # Give Betaflight (started by s10) time to complete gyro calibration
        # and internal setup before the first real physics tick.
        print("[SITL] Waiting for Betaflight to initialize...")
        time.sleep(2)

        print("[SITL] Sending warmup packets...")
        warmup_buf = SensorDataBuffer()
        warmup_fdm = warmup_buf.build_fdm()
        warmup_channels = np.full(MAX_RC_CHANNELS, 1500, dtype=np.uint16)
        warmup_channels[2] = 1000  # Low throttle
        warmup_channels[4] = 1000  # Disarmed
        warmup_rc = RCPacket(timestamp=0.0, channels=warmup_channels)

        warmup_count = 0
        warmup_packets = int(0.5 / config.dt)
        for i in range(warmup_packets):
            try:
                warmup_fdm.timestamp = i * config.dt
                warmup_rc.timestamp = i * config.dt
                bridge[0].step(warmup_fdm, warmup_rc, timeout_ms=5)
                warmup_count += 1
            except TimeoutError:
                pass
        print(f"[SITL] Warmup complete ({warmup_count} responses at {config.pid_rate:.0f}Hz)")
        print("[SITL] Bridge ready")
        _warmup_done_tick[0] = tick

    if start_time[0] is None:
        start_time[0] = time.time()

    s = state[0]
    b = bridge[0]
    buf = sensor_buf[0]

    # Update timing
    s.tick = tick
    s.sim_time = tick * config.dt
    t = s.sim_time

    # Read actual sensor data from physics simulation using batch operation.
    # This acquires the DB lock once for all reads, improving performance at high tick rates
    accel = np.zeros(3)
    gyro = np.zeros(3)
    baro = np.zeros(1)
    mag = np.zeros(3)
    world_pos = np.array([0.0, 0.0, 0.0, 1.0, *config.initial_position])
    world_vel = np.zeros(6)
    try:
        sensor_data = ctx.component_batch_operation(
            reads=[
                "drone.accel",
                "drone.gyro",
                "drone.baro",
                "drone.mag",
                "drone.world_pos",
                "drone.world_vel",
            ]
        )
        accel = np.array(sensor_data["drone.accel"])  # Body-frame accelerometer
        gyro = np.array(sensor_data["drone.gyro"])  # Body-frame gyroscope
        baro = np.array(sensor_data["drone.baro"])  # Barometric altitude
        mag = np.array(sensor_data["drone.mag"])  # Body-frame magnetometer
        world_pos = np.array(sensor_data["drone.world_pos"])  # simulator-side world state
        world_vel = np.array(sensor_data["drone.world_vel"])  # simulator-side world velocity

        # Update sensor buffer with real physics data
        buf.update(
            world_pos=world_pos,
            world_vel=world_vel,
            accel=accel,
            gyro=gyro,
            baro=baro,
            timestamp=t,
        )
    except RuntimeError as e:
        # First few ticks may not have data yet
        if tick > 5:
            print(f"[SITL] Warning: Could not read sensor data: {e}")
        buf.timestamp = t

    # Send Betaflight the most recent solver output before doing any optional
    # camera work. This keeps the FC's RC/FDM stream continuous even if render
    # is slow or unavailable.
    rc_cmd = _current_rc[0]
    s.arm = rc_cmd.arm
    s.throttle = rc_cmd.throttle
    phase = (
        "disarm" if rc_cmd.arm < 1700
        else ("arm-idle" if rc_cmd.throttle < 1100 else "fly")
    )
    channels = _rc_channels_buffer
    channels[:] = 1500
    fill_rc_channels(rc_cmd, channels)

    # Build FDM packet with sensor data
    fdm = buf.build_fdm()
    rc = RCPacket(timestamp=t, channels=channels)

    try:
        # Synchronous lockstep: send FDM+RC, wait for motor response
        # Motor order matches current Betaflight SITL: BR(0), FR(1), BL(2), FL(3)
        # The physics simulation (config.py) uses the same motor layout
        s.motors = b.step(fdm, rc)
        s.max_motor = max(s.max_motor, np.max(s.motors))

        # Write motor commands back to Elodin-DB for physics simulation
        # This uses the StepContext for direct DB access (no TCP overhead)
        ctx.write_component("drone.motor_command", s.motors)
    except TimeoutError:
        pass

    # Request a camera frame at the configured FPV rate, then collect the latest
    # frame from the message log. The solver still runs every tick and can tell
    # whether the frame is new using frame_fresh.
    if tick % RENDER_EVERY == 0 and tick > _warmup_done_tick[0]:
        try:
            fpv_camera.request_render(ctx, FPV_CAM_NAME)
            _last_render_tick[0] = tick
        except Exception as e:
            if tick % max(1, int(config.pid_rate)) == 0:
                print(f"[FPV] render trigger failed at tick {tick}: {e}")

    try:
        frame = fpv_camera.collect_frame(ctx, FPV_CAM_NAME)
        if frame is not None and tick == _last_render_tick[0]:
            _latest_frame[0] = frame
            _latest_frame_tick[0] = tick
            _fpv_frames[0] += 1
            if not _fpv_first_logged[0]:
                print(
                    f"[FPV] First frame at tick {tick}: "
                    f"shape={frame.shape}, dtype={frame.dtype}, "
                    f"nonzero={int(np.count_nonzero(frame))}"
                )
                _fpv_first_logged[0] = True
    except Exception as e:
        if tick > 200 and _fpv_frames[0] == 0:
            print(f"[FPV] collect error at tick {tick}: {e}")

    next_gate_index = (
        _race_last_gate[0] + 1
        if _race_last_gate[0] + 1 < len(ACTIVE_COURSE)
        else -1
    )
    solver_update = SensorUpdate(
        t=t,
        tick=tick,
        world_pos=np.asarray(world_pos),
        world_vel=np.asarray(world_vel),
        gyro=np.asarray(gyro),
        accel=np.asarray(accel),
        gyro_fresh=(tick % config.gyro_tick_interval == 0),
        accel_fresh=(tick % config.accel_tick_interval == 0),
        baro=float(baro[0]) if baro.size else 0.0,
        baro_fresh=(tick % config.baro_tick_interval == 0),
        mag=np.asarray(mag),
        mag_fresh=(tick % config.mag_tick_interval == 0),
        frame_rgba=_latest_frame[0],
        frame_fresh=(_latest_frame_tick[0] > _last_consumed_frame_tick[0]),
        last_gate_passed=_race_last_gate[0],
        next_gate_index=next_gate_index,
    )
    try:
        rc_out = _solver_module.autopilot(solver_update)
        if isinstance(rc_out, RCCommand):
            _current_rc[0] = rc_out
        if solver_update.frame_fresh:
            _last_consumed_frame_tick[0] = _latest_frame_tick[0]
    except Exception as e:
        if tick % max(1, int(config.pid_rate)) == 0:
            print(f"[SOLVER] error: {e}")

    # --- Gate-pass detection (runs every tick) ---
    # world_pos layout (Elodin scalar-last quat): [qx, qy, qz, qw, x, y, z]
    try:
        wp = np.asarray(world_pos)
        curr_pos = (float(wp[4]), float(wp[5]), float(wp[6]))
    except (IndexError, TypeError):
        curr_pos = (0.0, 0.0, 0.0)

    if _race_prev_pos[0] is not None:
        passed = race_course.detect_gate_pass(
            ACTIVE_COURSE,
            _race_last_gate[0],
            _race_prev_pos[0],
            curr_pos,
        )
        if passed is not None:
            _race_last_gate[0] = passed
            t_pass = tick * config.dt
            _race_pass_times[passed] = t_pass
            print(
                f"[GATE] passed gate {passed} at t={t_pass:.2f}s "
                f"pos=({curr_pos[0]:.2f},{curr_pos[1]:.2f},{curr_pos[2]:.2f})"
            )
            try:
                pass_times = np.full(race_course.MAX_GATES, -1.0)
                pass_times[: len(_race_pass_times)] = _race_pass_times
                ctx.write_component(
                    "drone.last_gate_passed",
                    np.array([float(passed)]),
                )
                ctx.write_component("drone.gate_pass_times", pass_times)
            except Exception:
                pass
    _race_prev_pos[0] = curr_pos

    # Print status every second
    if t - last_print[0] >= 1.0:
        armed = "ARMED" if np.any(s.motors > 0.02) else "DISARMED"
        elapsed = time.time() - start_time[0]
        rate = t / elapsed if elapsed > 0 else 0

        # Get current position for debug output
        try:
            pos = np.array(ctx.read_component("drone.world_pos"))
            x_pos = pos[4] if len(pos) > 6 else 0.0
            y_pos = pos[5] if len(pos) > 6 else 0.0
            z_pos = pos[6] if len(pos) > 6 else pos[2]
            vel = np.array(ctx.read_component("drone.world_vel"))
            z_vel = vel[5] if len(vel) > 5 else vel[2]
            pos_str = f"pos=({x_pos:+5.2f},{y_pos:+5.2f},{z_pos:+5.2f})m vz={z_vel:+.2f}m/s"
        except Exception:
            pos_str = "pos=?,?,?"

        print(
            f"  t={t:5.1f}s | {phase:8} | {armed:8} | "
            f"motors=[{s.motors[0]:.3f},{s.motors[1]:.3f},{s.motors[2]:.3f},{s.motors[3]:.3f}] | "
            f"{pos_str} | {rate:.1f}x realtime"
        )
        last_print[0] = t

    # Check if simulation is complete - print summary and exit
    if tick >= MAX_TICKS - 1:
        _completed[0] = True
        b.stop()
        elapsed = time.time() - start_time[0]

        # Read final position
        try:
            final_pos = np.array(ctx.read_component("drone.world_pos"))
            final_z = final_pos[6] if len(final_pos) > 6 else final_pos[2]
            final_vel = np.array(ctx.read_component("drone.world_vel"))
            final_vz = final_vel[5] if len(final_vel) > 5 else final_vel[2]
        except Exception:
            final_z = 0.0
            final_vz = 0.0

        print()
        print("=" * 50)
        print("Simulation complete!")
        print(
            f"  Simulated: {s.sim_time:.1f}s in {elapsed:.1f}s "
            f"({s.sim_time / elapsed if elapsed > 0 else 0:.1f}x realtime)"
        )
        print(f"  Total ticks: {s.tick}")
        print(f"  Sync steps: {b.step_count}")
        print(f"  FPV frames: {_fpv_frames[0]} (target ~{int(s.sim_time * fpv_camera.TARGET_FPS)})")
        print(f"  Max motor: {s.max_motor:.3f}")
        print(f"  Final position: z={final_z:.2f}m, vz={final_vz:.2f}m/s")
        print()

        # Success criteria: motors responded AND drone moved
        took_off = final_z > config.initial_position[2] + 0.1  # More than 10cm above start

        if b.step_count > 0 and s.max_motor > 0.06 and took_off:
            print("SUCCESS: SITL integration working! Drone took off!")
        elif b.step_count > 0 and s.max_motor > 0.06:
            print("WARNING: Motors responded but drone did not take off.")
            print("  Check physics pipeline: motor_command -> thrust -> force")
        elif b.step_count > 0 and s.max_motor > 0.02:
            print("WARNING: Motors armed but no throttle response.")
        else:
            print("WARNING: No motor response. Check Betaflight configuration.")

        race_course.print_summary(
            ACTIVE_COURSE,
            _race_last_gate[0],
            _race_pass_times,
            s.sim_time,
        )


# Return the next non-existent filename with auto-incremented
# number if the pattern ends in Xs.
#
# e.g., `next_filename("sim_sitlXXX") -> "sim_sitl001"`
# `next_filename("sim_sitl_mine") -> "sim_sitl_mine"`
def next_filename(pattern: str) -> str:
    match = re.search(r"(X+)$", pattern)
    if not match:
        return pattern

    width = len(match.group(1))
    prefix = pattern[:-width]

    i = 0
    while True:
        fname = f"{prefix}{i:0{width}d}"
        if not os.path.exists(fname):
            return fname
        i += 1


# --- Run Simulation ---
db_filename = next_filename("betaflight_dbXXX")
print(f"Writing database to: {db_filename}")
world.run(
    system,
    simulation_rate=config.pid_rate,
    generate_real_time=True,
    post_step=sitl_post_step,
    db_path=db_filename,
    start_timestamp=0,
    max_ticks=MAX_TICKS,
    interactive=True,
)

if not bridge[0]:
    print("\nNo simulation ticks executed.")
    print("Usage: elodin editor sim/main.py")
