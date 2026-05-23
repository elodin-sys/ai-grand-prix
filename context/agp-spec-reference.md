# AI Grand Prix: spec reference and explainer

> Companion explainer to `260508_Technical_Spec_0002.pdf` ("Virtual AI Drone Race
> Technical Specification"), for both human readers (contestants, contributors)
> and AI coding agents working in this repo. Every section pairs the **spec
> text** with **what it means for us** and **where it lives in code**, and ends
> with a section calling out **deviations, ambiguities, and open questions**.
>
> This document is intentionally redundant with `ARCHITECTURE.md` in places.
> ARCHITECTURE.md describes _our practice rig_; this file describes _the spec
> we're targeting_ and how faithfully we currently mirror it.

---

## 1. Source document metadata

| Field | Value |
|---|---|
| Document ID | `VADR-TS-002` |
| Title | "AI Grand Prix — Virtual Qualifier Technical Specification" |
| Issue | `00.02` |
| Date | `2026-05-08` |
| File in repo | [`260508_Technical_Spec_0002.pdf`](../260508_Technical_Spec_0002.pdf) |
| Pages | 11 |
| Authors of record | Issue 00.01 (2026-03-09) by `KH`; Issue 00.02 (2026-05-04) by `NT`, summary "camera" |
| Audience (spec §1.2) | competition participants ("Teams") |
| Programme | [Anduril's $500K AI Grand Prix](https://www.theaigrandprix.com/) (Virtual Qualifier 1) |

> The spec is a **contract** between contestants and the as-yet-unreleased DCL
> Simulator. It is normative on protocol, sensors, gates, and timing. It is
> _not_ a description of the simulator's internals; those are explicitly out
> of scope (§2.3).

## 2. The challenge in one paragraph

Contestants ship Python (or any other language, see §5.1) autopilot software that
controls a simulated drone through a sequenced gate course, MAVLink over UDP,
with a 640×360 forward camera and IMU/attitude telemetry. The course is
deterministic, identical for every team, and must be flown within an 8-minute
window with no human in the loop. Round One ("Qualification Phase") is a
pass/fail check that your software can navigate the course at all.

## 3. Simulation environment (spec §3)

### 3.1 Physics & visual model

| Spec item | Value | Notes |
|---|---|---|
| Physics model | rigid-body, with thrust, aero drag, gravity, collisions | §3.2 |
| Physics update rate | **120 Hz** | §3.2 / §4.4 |
| Spatial reference | local Cartesian only, **no GPS, no global position** | §3.3 |
| Visual environment | forward FPV camera + gates + scene + dynamic lighting | §3.4 |
| Determinism | course geometry, physics params, environment are identical for every team | §3.5 |

What it means for us:

- Solvers must localize from **vision + IMU + attitude** alone. No
  drone-position oracle is available at race time.
- 120 Hz physics is the _spec_; our practice rig runs Betaflight in lockstep at
  a much higher tick rate (`simulation_rate = 1000 Hz` by default in
  [`sim/config.py`](../sim/config.py)) so that PID iteration matches a real
  Betaflight flight controller. Solvers see a much higher tick stream, but
  should not rely on tick rates above 120 Hz for portability.
- Determinism means a fixed seed + fixed code → bit-identical replay. Use this
  for regression testing.

### 3.2 Drone chassis (spec §3.6)

| Dimension | Value |
|---|---|
| Width | **280 mm** |
| Length | **280 mm** |
| Height | **160 mm** |

This is the official AGP airframe envelope (roughly a 5-inch racing quad).
Mass, motor count, motor mapping, prop diameter, and inertia are **not
specified**. Treat any specific numbers in our practice rig as our best guess.

Code: [`sim/config.py`](../sim/config.py) (`DroneConfig`,
`create_5inch_racing_quad`). Our `DEFAULT_CONFIG` is a generic 5″ racer, not a
verified match for the AGP airframe.

### 3.3 Gate dimensions (spec §3.7)

| Region | Width | Height | Depth |
|---|---|---|---|
| Outer gate frame | **2700 mm** | **2700 mm** | **260 mm** |
| Inner flyable opening | **1500 mm** | **1500 mm** | **260 mm** |

Therefore the frame is `(2.7 − 1.5) / 2 = 0.6 m` thick on every side, in a
0.26 m-deep face. The **1.5 m × 1.5 m inner square** is the right scale anchor
for PnP / vision pipelines.

Code: [`sim/course.py`](../sim/course.py) (`GATE_OUTER_W`, `GATE_OUTER_H`,
`GATE_INNER_W`, `GATE_INNER_H`, `GATE_DEPTH`). Pass detection in
`detect_gate_pass()` uses the inner 1.5 m × 1.5 m square.

Gate visual style is _not_ in the spec beyond "visually distinctive to the
environment, but consistent throughout the Virtual Qualifier 1 track" (§3.1).
Our gate uses high-contrast magenta + white accents purely so it reads well in
the FPV stream during development.

## 4. Coordinate frames (spec §3.8)

The spec is **NED-everywhere** for MAVLink-side coordinates, with a small set
of body / camera / IMU transforms.

| Frame | Origin | Axes | Used for |
|---|---|---|---|
| `MAV_FRAME_LOCAL_NED` | fixed ground point (≈ where the drone armed) | X=North, Y=East, Z=Down | World position |
| `MAV_FRAME_BODY_NED` | vehicle | X=forward, Y=right, Z=down | Body-frame commands |
| Camera frame | same origin as body | body rotated **+20° upward (pitch up)** | Vision data |
| IMU frame | same origin as body | **identity** to body | `HIGHRES_IMU` |

> "Be aware that all coordinates are NED and you might need to rotate the
> camera frame into the camera coordinate convention of your specific image
> processing library." — spec §3.8

What it means for us:

- The official sim publishes **NED** values. OpenCV, PyTorch, and most CV
  libraries assume a camera frame with Z forward, X right, Y down (or similar
  RDF / RUB conventions). Contestants must rotate explicitly.
- The camera's +20° upward tilt means the optical axis is _not_ aligned with
  body-forward. PnP / image-to-body projection must include this static
  rotation.

Where it lives in our code:

- We run Elodin internally in **ENU** (East-North-Up) and convert at the
  Betaflight bridge; see [`sim/betaflight_bridge.py`](../sim/betaflight_bridge.py)
  and `ARCHITECTURE.md` §"Coordinate frames and unit conventions".
- Our solver API (`SensorUpdate` in [`solver/api.py`](../solver/api.py)) hands
  out **ENU** world state today. Once we adopt MAVLink (see §11) this should
  switch to NED to match the spec.

## 5. Camera intrinsics (spec §3.8)

Standard pinhole, **no lens distortion**.

| Parameter | Value |
|---|---|
| Image resolution | **640 × 360** |
| Principal point `[cx, cy]` | `[320 px, 180 px]` |
| Focal lengths `[fx, fy]` | `[320, 320]` |
| Stated FoV | **VFoV = 90°** |
| Stream rate | **30 Hz** (§4.6) |
| Tilt | **+20° upward** in body frame (§3.8) |

> ⚠️ **Spec inconsistency, flagged on first read:** The intrinsics imply
> `VFoV = 2·atan(180/320) ≈ 58.72°` and `HFoV = 2·atan(320/320) = 90°`. The
> stated `VFoV = 90°` matches the _horizontal_ FoV computed from the same
> intrinsics. Most likely the spec mislabels HFoV as VFoV.
>
> We honor the intrinsics rather than the prose: `CAM_FOV_VERT_DEG` in
> [`sim/camera.py`](../sim/camera.py) is derived as
> `2·atan(cy / fy) ≈ 58.72°`. The intrinsics define a single self-consistent
> pinhole model, and the official simulator's renderer is far more likely to
> produce frames consistent with `fx, fy, cx, cy` than with a one-sentence
> FoV statement that contradicts them.

## 6. Communication protocol: MAVLink (spec §4)

The official simulator speaks MAVLink 2 over UDP, using
[c_library_v2](https://github.com/mavlink/c_library_v2) /
MAVSDK-compatible interfaces (§4.1, §4.2).

### 6.1 Message catalog (spec §4.3)

| Message | Direction | Purpose |
|---|---|---|
| `HEARTBEAT` | Simulator → Client | Connection status |
| `ATTITUDE` | Simulator → Client | Vehicle attitude |
| `HIGHRES_IMU` | Simulator → Client | Vehicle status / measurements |
| `TIMESYNC` | Simulator → Client | Timing |
| `SET_POSITION_TARGET_LOCAL_NED` | Client → Simulator | Control input |
| `SET_ATTITUDE_TARGET` | Client → Simulator | Control input |

The spec lists `HIGHRES_IMU` twice ("Vehicle status" and "Measurements"). Read
that as: high-rate IMU is the primary inertial telemetry stream.

### 6.2 Timing constraints (spec §4.4)

| Constraint | Value |
|---|---|
| Physics simulation rate | 120 Hz |
| Command rate (Client → Sim) | **< 100 Hz** |
| Minimum heartbeat rate | **2 Hz** |

Implications:

- Solvers must emit at least 2 Hz `HEARTBEAT` or the link will be considered
  dropped (standard MAVLink behavior; link timeout is typically ~5 s without
  heartbeats).
- Commands above ~100 Hz will be rate-limited / dropped. Most racing pilots
  send at 50 Hz, which is plenty.

### 6.3 Telemetry payload (spec §4.5)

`ATTITUDE` + `HIGHRES_IMU` between them give you:

- Vehicle attitude (quaternion or Euler)
- Orientation
- Linear velocities
- System status flags

Notably absent: **GPS**, **depth**, **motor RPM**, **battery state of charge**.
A perception system that depends on any of those will not transfer.

## 7. Vision stream (spec §4.6)

The vision stream is a **separate UDP channel** from MAVLink, on port `5600`,
sending JPEG-compressed 640×360 frames at 30 Hz. Frames are chunked because a
JPEG can exceed a single UDP MTU.

### 7.1 Transport summary

| Field | Value |
|---|---|
| Protocol | UDP |
| Default port | **5600** |
| Byte order | Little-Endian (`<`) |
| Header size | 24 bytes (fixed) |
| Payload | variable-length JPEG slice |

### 7.2 Per-packet header

Each datagram = `[24-byte header][JPEG slice]`. Header layout:

| Offset | Field | Type | Size | Description |
|--:|---|---|--:|---|
| 0 | `frame_id` | `uint32` | 4 B | Unique sequence ID for the image frame |
| 4 | `chunk_id` | `uint16` | 2 B | Packet index within the frame (`0..total_chunks-1`) |
| 6 | `total_chunks` | `uint16` | 2 B | Total packets that make up this frame |
| 8 | `jpeg_size` | `uint32` | 4 B | Final reassembled JPEG size, in bytes |
| 12 | `payload_size` | `uint32` | 4 B | JPEG bytes in _this_ packet |
| 16 | `sim_time_ns` | `uint64` | 8 B | Simulation epoch timestamp, nanoseconds |

To consume the stream:

1. Bind a UDP socket on `5600`.
2. Read datagrams; group by `frame_id`.
3. Reassemble in `chunk_id` order; verify `sum(payload_size) == jpeg_size`.
4. JPEG-decode (cv2 / PIL / turbojpeg) into a `(360, 640, 3)` uint8 image.
5. Pair with `sim_time_ns` for time-alignment to telemetry.

> Frames _will_ drop or arrive out of order on lossy UDP. A robust receiver
> uses `frame_id` to keep at most N partial frames in flight and drops any
> still incomplete after a small timeout.

### 7.3 Where it lives in our practice rig

We do **not** ship the UDP/JPEG pipeline today. We hand contestants a raw RGBA
NumPy array directly (`SensorUpdate.frame_rgba` in
[`solver/api.py`](../solver/api.py)) at our internal `fpv_rate` (default 30 Hz
via `fpv_tick_interval`). The shape matches: `(360, 640, 4)` RGBA.

Porting from practice to the official sim:

- Convert `frame_rgba[:, :, :3]` from RGB to BGR if the receiver expects BGR.
- Skip JPEG decode in practice; in the official sim, you _must_ JPEG-decode.
- Use `update.t` (sim seconds) where the official protocol uses
  `sim_time_ns / 1e9`.

## 8. Software-in-the-loop bridge (spec §4.7)

> "The simulator provides a low-latency UDP SITL bridge enabling external AI
> controllers to exchange telemetry and control commands."

This is the same UDP MAVLink channel from §4, surfaced specifically so
external (non-Windows, non-DCL) controllers can plug in. The bridge is the
intended attachment point for contestant code.

In our practice rig the equivalent role is filled by the lockstep
post-step callback in [`sim/main.py`](../sim/main.py), which hosts the solver
in-process. The wire-level protocol differs (Betaflight RC/FDM, not MAVLink)
but the architectural role is the same.

## 9. Contestant software environment (spec §5)

### 9.1 Runtime (spec §5.1)

| Item | Value |
|---|---|
| Reference Python | **3.14.2** ("known to operate correctly") |
| Other runtimes | allowed |
| Simulator host OS | **Windows 11** |
| Simulator GPU requirement | "decent GPU with 8 GB VRAM" |
| Linux simulator support | **not currently supported** |

> "Currently we do not support Linux OS." — spec §5.1

What it means:

- Contestants can develop on any OS, in any language, but the **official
  qualifier sim only runs on Windows 11**. UDP-over-loopback to a Windows
  host with the sim is the canonical setup.
- 8 GB VRAM is the GPU target. Solvers using heavy CNNs / VLMs should
  budget memory accordingly; the host's VRAM is shared with the sim itself.
- Python 3.14.2 is the _stated_ reference. If you target it explicitly,
  prefer language features that gracefully degrade on 3.11–3.13.

### 9.2 Client responsibilities (spec §5.2)

The client must:

- Establish MAVLink communication
- Maintain `HEARTBEAT` messages (≥ 2 Hz, §4.4)
- Send control commands (`SET_POSITION_TARGET_LOCAL_NED` or
  `SET_ATTITUDE_TARGET`, < 100 Hz)
- Process telemetry data (`ATTITUDE`, `HIGHRES_IMU`, `TIMESYNC`)
- Process vision stream data (port 5600 JPEG chunks)

### 9.3 Intended control architecture (spec §5.3)

Spec's reference pipeline:

```
Vision + Telemetry  →  Perception  →  Planning  →  Control  →  Pilot Commands  →  Stabilized Controller
```

Concretely:

- **Perception** turns image + IMU into _what's around me_ (gate detection,
  pose estimation, depth cues).
- **Planning** turns _what's around me_ + _where the next gate is_ into a
  short-horizon trajectory or attitude target.
- **Control** turns the trajectory into pilot commands
  (`SET_ATTITUDE_TARGET` or position target).
- The **Stabilized Controller** (inside the sim) is the inner-loop
  Betaflight-equivalent that turns pilot commands into motor PWMs.

> The "Stabilized Controller" is _not_ the contestant's responsibility. The
> sim runs the inner loop. Contestants compete on perception + planning +
> outer-loop control.

In our practice rig, the inner-loop equivalent is the real
[Betaflight SITL](https://betaflight.com/docs/development/SITL) build, so
solvers can tune against the same stabilization plant a real airframe would
use.

## 10. Example control session (spec §6)

```
1. Client initializes MAVSDK
2. Client connects to simulator endpoint
3. Simulator transmits HEARTBEAT
4. Client streams control commands
5. Simulator applies commands
6. Telemetry and vision streams returned
```

Note the order: MAVLink dialog opens, **simulator** sends `HEARTBEAT` first,
then the client may begin commanding. In MAVLink parlance the client should
also send its own `HEARTBEAT`s once it's online.

## 11. Compliance (spec §7)

> "Participants must ensure their implementation conforms to this
> specification. Specifically, human interaction during the flight which the
> participants submit as a timed run is grounds for **immediate
> disqualification**."

What it means:

- **No human in the loop** during a timed run. No manual transponder, no
  click-to-restart, no joystick. The autopilot starts the run and finishes
  the run.
- Submitted runs are presumed to be entirely software-driven.
- Pre-run setup (loading weights, calibrating filters) is fine; only
  in-flight intervention is disqualifying.

## 12. Round One: qualification phase (spec §8)

| Item | Value |
|---|---|
| Objective (§8.1) | Verify the contestant's software can navigate the racecourse |
| Course structure (§8.2) | start gate → intermediate gates → finish gate |
| Maximum run duration (§8.3) | **8 minutes** |

Round One is pass/fail. The course shape, number of intermediate gates, and
scoring details beyond "did you finish in 8 minutes" are **not in the spec**
and will presumably be announced separately.

Code: our practice course is a 3-gate straight line along +X at hover altitude
(`EASY_COURSE` in [`sim/course.py`](../sim/course.py)). `simulation_time` in
[`sim/config.py`](../sim/config.py) governs the headless run length but is far
shorter than 8 minutes by default.

---

## 13. Where the spec lives in our codebase

Quick lookup table for contributors and AI agents. The spec column quotes the
section the value comes from; the code column is the canonical location to
edit if it ever changes.

| Spec topic | Spec § | In this repo |
|---|---|---|
| Physics 120 Hz | §3.2, §4.4 | `pid_rate` in [`sim/config.py`](../sim/config.py) (our internal `simulation_rate` is higher; that's the Betaflight lockstep tick) |
| Drone 280 × 280 × 160 mm | §3.6 | [`sim/config.py`](../sim/config.py) (`DroneConfig`; not literally enforced, presets approximate) |
| Gate 2.7 m outer / 1.5 m inner / 0.26 m depth | §3.7 | `GATE_OUTER_W/H`, `GATE_INNER_W/H`, `GATE_DEPTH` in [`sim/course.py`](../sim/course.py) |
| MAVLink coordinate frames (NED) | §3.8 | NED ↔ ENU conversion in [`sim/betaflight_bridge.py`](../sim/betaflight_bridge.py); `SensorUpdate` is still ENU |
| Camera 640 × 360, intrinsics, +20° tilt | §3.8 | [`sim/camera.py`](../sim/camera.py) (`CAM_WIDTH`, `CAM_HEIGHT`, `CAM_FX`, `CAM_FY`, `CAM_CX`, `CAM_CY`, `CAM_FOV_VERT_DEG` (derived), `CAM_TILT_UP_DEG`) |
| Vision 30 Hz | §4.6 | `fpv_rate` in [`sim/config.py`](../sim/config.py), `TARGET_FPS` in [`sim/camera.py`](../sim/camera.py) |
| MAVLink messages over UDP | §4 | **Not implemented.** We use Betaflight RC + FDM + PWM packets over UDP, then re-expose telemetry to the solver via in-process `SensorUpdate` |
| JPEG-chunked vision UDP | §4.6 | **Not implemented.** Frames handed to solver as raw RGBA arrays |
| No GPS, no depth, no motor RPM, no battery | §3.3 + §4.5 (by omission) | [`solver/api.py`](../solver/api.py) (`SensorUpdate` deliberately omits these) |
| 8-minute max run | §8.3 | `simulation_time` in [`sim/config.py`](../sim/config.py) |
| Time-trial scoring & gate ordering | §8 (implied) | `detect_gate_pass` and `print_summary` in [`sim/course.py`](../sim/course.py) |
| ≥ 2 Hz heartbeat, < 100 Hz commands | §4.4 | **Not implemented.** Solver returns an `RCCommand` _every tick_; rate-limiting belongs in a future MAVLink shim |
| Windows-only simulator | §5.1 | This rig runs on macOS and Linux. We are intentionally OS-broader than the official sim |

## 14. Deviations from the spec in this practice rig

These are deliberate, called out so they aren't mistaken for bugs. None of
them are correctness issues today; all should be reconsidered when the
official sim ships and parity becomes the priority.

1. **Wire protocol.** We use Betaflight SITL UDP (RC + FDM + PWM) instead of
   MAVLink 2. Solvers see a Python `SensorUpdate` dataclass, not parsed
   MAVLink messages. A future MAVLink shim around the same `SensorUpdate`
   data should be straightforward.
2. **Coordinate frame at the solver boundary.** We pass ENU to the solver;
   spec says NED. The conversion is mechanical (flip Y and Z signs) but
   today _our solver lives in ENU_, and porting code will need that flip.
3. **Camera FoV.** We use the value implied by the intrinsics
   (`fx = fy = 320`, `cy = 180` → VFoV ≈ 58.72°) rather than the spec's
   prose-stated 90°, on the basis that the prose almost certainly mislabels
   HFoV as VFoV. See §5 above. This is a deviation from the spec text but
   matches the spec intrinsics, and we expect it to match the official sim's
   actual rendering.
4. **Inner-loop controller.** We run real Betaflight SITL; the official sim
   advertises a generic "stabilized controller" whose internals are explicitly
   out of scope (§2.3). Tunings that win in Betaflight may need re-tuning.
5. **OS support.** Spec is Windows 11; we run on macOS and Linux.
6. **Tick rate exposed to the solver.** Spec says physics at 120 Hz; our
   default `simulation_rate` is 1 kHz so Betaflight's PID loop runs at a
   lockstep-friendly cadence. Solvers should be _at least_ time-step-aware
   (`update.t`, `update.tick`) rather than counting ticks.
7. **Vision frame format.** Raw RGBA NumPy, not chunked JPEG. JPEG decode
   adds ~1–3 ms; that cost is hidden today.
8. **No `HEARTBEAT` cadence enforcement.** Anything you can return from
   `autopilot()` counts; in the real sim a missed heartbeat will time out
   the link.

## 15. Ambiguities and open questions

Things the spec does not nail down. Until the AGP team publishes more, treat
all of these as TBD.

- **Course geometry.** §8.2 says "start gate, intermediate gates, finish
  gate" with no count, geometry, banking, turns, vertical movement, or
  obstacle map. We ship `EASY_COURSE` and an "Opportunities for improvement"
  list in `ARCHITECTURE.md` includes a procedural-course / curriculum item.
- **Scoring beyond pass/fail.** Round One is qualification; later rounds
  presumably use lap time as the primary score and likely add penalties
  (clipped gates, crashes). Not specified.
- **Penalty model.** Is a clipped gate-edge OK? Does a wall collision DNF
  you? Does a missed gate force a re-fly? Unspecified.
- **Crash physics.** Rigid-body with collisions is mentioned (§3.2). It is
  not specified whether the drone recovers after a crash, despawns, or ends
  the run.
- **Camera VFoV labelling.** §5 above: 90° stated vs ≈58.72° implied by
  the intrinsics. We treat this as a labelling error in the spec text and
  follow the intrinsics, but if the official sim ever ships with an
  actual 90° vertical FoV we will need to revisit.
- **MAVLink dialect details.** §4.3 lists messages by name but not the
  specific dialect, message-ID set, or extension behavior.
- **Vehicle / drone mass and inertia.** §3.6 gives the chassis envelope
  only. Mass, motor count, motor positions, motor max thrust, and motor
  time constant are all unspecified.
- **Wind, turbulence, ground effect.** §3.2 names "thrust, aero drag,
  gravity, collision physics" only. No atmospheric model is described.
- **Network jitter / packet-loss model.** Whether the bridge introduces
  artificial latency or loss is unspecified.
- **TIMESYNC semantics.** Listed but no exchange format / cadence detailed.
- **Camera shutter / exposure / motion blur.** Pinhole + no distortion is
  spelled out; nothing about per-frame motion blur, rolling shutter, or
  exposure / auto-gain.
- **Are intrinsics fixed for the whole event** or per-track? Currently
  reads as fixed.

When updated documents arrive, this file should be revised alongside
[`ARCHITECTURE.md`](../ARCHITECTURE.md) and the cross-reference table in §13.

## 16. Recommended reading order

For a new contestant or contributor coming in cold:

1. The spec PDF itself: [`260508_Technical_Spec_0002.pdf`](../260508_Technical_Spec_0002.pdf), 11 pages, ~20 min.
2. This file (`context/agp-spec-reference.md`).
3. [`README.md`](../README.md): get the practice rig running locally.
4. [`solver/README.md`](../solver/README.md): the autopilot contract you'll actually be writing against.
5. [`ARCHITECTURE.md`](../ARCHITECTURE.md): how the practice rig is built; useful both for hacking on the sim and for understanding why the on-the-wire surface differs from the spec.

## Appendix A. Glossary

| Term | Meaning |
|---|---|
| **AGP** | AI Grand Prix; the Anduril-sponsored autonomous drone racing competition |
| **VADR-TS-002** | Spec document ID; "Virtual AI Drone Race, Technical Specification, doc 002" |
| **DCL Simulator** | The official simulator the AGP team is building (the one we're emulating) |
| **MAVLink 2** | Lightweight binary message protocol used by ArduPilot/PX4/etc. for vehicle telemetry & control |
| **MAVSDK** | High-level MAVLink client SDK |
| **SITL** | Software-in-the-loop: running a flight controller as a process with simulated inputs |
| **NED** | North-East-Down coordinate frame, MAVLink default for world/body |
| **ENU** | East-North-Up coordinate frame, Elodin (and ROS) default |
| **FoV** | Field of view; in a pinhole model, derived from focal length and image size |
| **PnP** | Perspective-n-Point: estimating camera pose from N known 3D points |
| **`SensorUpdate`** | Our practice-rig surrogate for the MAVLink telemetry bundle; see [`solver/api.py`](../solver/api.py) |
| **`RCCommand`** | Our practice-rig surrogate for `SET_ATTITUDE_TARGET` / `SET_POSITION_TARGET_LOCAL_NED` |
