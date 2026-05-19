#!/usr/bin/env python3
"""
Configure a fresh Betaflight SITL via its CLI on TCP port 5761.

Sets up the minimum config required for the AI Grand Prix sim:
  - AUX1 = ARM switch (1700-2100)
  - lockstep-friendly gyro/PID loop settings
  - Disable Betaflight's default 5s power-on arming grace
  - Disable arming-disable angle / runaway / takeoff prevention checks
    (we want a sim that arms cleanly from a known-good initial state)
  - Save to eeprom.bin

Run this once after building Betaflight SITL. The resulting eeprom.bin
gets committed so other contributors don't have to repeat this step.

Usage: uv run python scripts/configure_betaflight.py
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BF_BINARY = REPO_ROOT / "betaflight" / "obj" / "main" / "betaflight_SITL.elf"
EEPROM = REPO_ROOT / "eeprom.bin"

CLI_HOST = "127.0.0.1"
CLI_PORT = 5761

# Send these AFTER entering CLI mode. Each line is one command.
CLI_COMMANDS = [
    # Map AUX1 to the ARM mode (mode 0). Trigger when channel value is 1700-2100.
    "aux 0 0 0 1700 2100 0 0",
    # Map AUX2 to ANGLE mode (mode 1) — auto-leveling so the solver can send
    # stable angle-target inputs instead of fighting raw rate commands.
    "aux 1 1 1 1700 2100 0 0",
    # 1:1 PID denom for lockstep SITL
    "set gyro_hardware_lpf = NORMAL",
    "set pid_process_denom = 1",
    # The simulator already does bridge warmup before t=0; don't make users
    # wait through Betaflight's default 5 second power-on arming grace.
    "set pwr_on_arm_grace = 0",
    # Stop arming from being blocked when the drone is sitting on the ground.
    "set runaway_takeoff_prevention = OFF",
    "set small_angle = 180",
    # Don't fail on RX loss before we've sent RC packets
    "set failsafe_delay = 200",
    # Persist
    "save",
]


class CLIClient:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = bytearray()
        self.lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        self.sock.settimeout(0.1)
        while not self._stop:
            try:
                data = self.sock.recv(8192)
                if not data:
                    return
                with self.lock:
                    self.buf.extend(data)
            except socket.timeout:
                continue
            except OSError:
                return

    def drain(self, seconds: float = 0.5) -> str:
        """Wait `seconds`, then snapshot the buffer."""
        time.sleep(seconds)
        with self.lock:
            out = bytes(self.buf).decode(errors="replace")
            self.buf.clear()
        return out

    def send(self, data: bytes) -> None:
        self.sock.sendall(data)

    def close(self) -> None:
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass


def main() -> int:
    if not BF_BINARY.exists():
        print(f"ERROR: BF binary not found at {BF_BINARY}", file=sys.stderr)
        return 1

    subprocess.run(["pkill", "-9", "-f", "betaflight_SITL"], capture_output=True)
    time.sleep(0.5)

    # Keep any existing EEPROM as the base config. A completely fresh SITL
    # EEPROM can take a long time to initialize on some Betaflight builds, and
    # this script's job is to update/persist the few settings this simulator
    # needs rather than force a full flash-format cycle every run.

    print(f"Starting {BF_BINARY.name}...")
    bf_log = open("/tmp/bf-configure.log", "w")
    bf = subprocess.Popen(
        [str(BF_BINARY)],
        stdout=bf_log,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )

    cli: CLIClient | None = None
    try:
        # First boot after a clean build can spend a while initializing EEPROM
        # before the MSP/CLI TCP listener appears.
        deadline = time.time() + 120.0
        while time.time() < deadline:
            try:
                cli = CLIClient(CLI_HOST, CLI_PORT)
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.2)
        if cli is None:
            print("ERROR: CLI port did not open before the deadline", file=sys.stderr)
            return 2

        # 1) wake the line discipline; 2) enter CLI mode; 3) issue each command
        print("Entering CLI mode...")
        cli.send(b"\r")
        time.sleep(0.5)
        cli.send(b"#")
        banner = cli.drain(2.0)
        if "Entering CLI Mode" not in banner:
            print("WARNING: did not see CLI banner. Got:")
            print(repr(banner))
        else:
            print(banner.split("\r\n")[0])

        for cmd in CLI_COMMANDS:
            print(f"\n> {cmd}")
            cli.send((cmd + "\r\n").encode())
            wait = 5.0 if cmd == "save" else 0.5
            print(cli.drain(wait), end="")

        cli.close()

        # save() reboots BF — give it time to flush
        time.sleep(2.0)

    finally:
        if bf.poll() is None:
            try:
                os.kill(bf.pid, signal.SIGTERM)
                bf.wait(timeout=3.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.kill(bf.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        bf_log.close()

    if EEPROM.exists():
        sz = EEPROM.stat().st_size
        print(f"\nOK: wrote {EEPROM} ({sz} bytes)")
        return 0
    print(f"ERROR: {EEPROM} was not created", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
