#!/usr/bin/env python3
"""Connect to a running BF SITL CLI on TCP 5761 and probe it interactively.

Usage:
    # In one terminal:
    ./betaflight/obj/main/betaflight_SITL.elf

    # In another:
    uv run python scripts/bf_cli_probe.py [cmd]

Without an arg it sends `#`, then `status`, then `aux`, with long reads.
"""

from __future__ import annotations

import socket
import sys
import time

HOST, PORT = "127.0.0.1", 5761


def drain(sock: socket.socket, seconds: float, label: str = "") -> str:
    """Read everything available within `seconds`, regardless of pauses."""
    sock.settimeout(0.5)
    buf = bytearray()
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf.extend(chunk)
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[{label}] OSError: {e}")
            break
    return buf.decode(errors="replace")


def main() -> int:
    cmds = sys.argv[1:] if len(sys.argv) > 1 else ["#", "status", "aux"]

    print(f"Connecting to {HOST}:{PORT}...")
    s = socket.create_connection((HOST, PORT), timeout=5.0)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    banner = drain(s, 1.0, "banner")
    print(f"--- banner ({len(banner)} bytes) ---\n{banner!r}")

    for c in cmds:
        print(f"\n>>> {c}")
        s.sendall((c + "\r\n").encode())
        out = drain(s, 3.0, c)
        print(f"--- response ({len(out)} bytes) ---\n{out}")

    s.close()
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
