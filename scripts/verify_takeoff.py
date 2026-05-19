#!/usr/bin/env python3
"""Export the latest Elodin DB and verify a sustained hover."""

from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_RE = re.compile(r"^betaflight_db(\d+)$")


def latest_db() -> Path:
    candidates: list[tuple[int, float, Path]] = []
    for path in REPO_ROOT.iterdir():
        if not path.is_dir():
            continue
        match = DB_RE.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        raise SystemExit("No betaflight_dbNNN directory found. Run elodin first.")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def export_db(db_path: Path) -> Path:
    out_dir = REPO_ROOT / "dbs" / f"{db_path.name}-csv"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(exist_ok=True)
    subprocess.run(
        [
            "elodin-db",
            "export",
            "-o",
            str(out_dir),
            str(db_path),
            "--format",
            "csv",
            "--flatten",
            "--join",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return out_dir


def _as_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_drone_rows(export_dir: Path) -> list[dict[str, str]]:
    csv_path = export_dir / "drone.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing exported drone.csv at {csv_path}")
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def longest_hover_span(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    best_start = best_end = current_start = 0.0
    in_span = False
    best_duration = 0.0

    for row in rows:
        t = _as_float(row, "sim_time_0", _as_float(row, "time"))
        z = _as_float(row, "world_pos_z", math.nan)
        hovering = 1.0 <= z <= 2.5
        if hovering and not in_span:
            current_start = t
            in_span = True
        elif not hovering and in_span:
            duration = t - current_start
            if duration > best_duration:
                best_duration = duration
                best_start = current_start
                best_end = t
            in_span = False

    if in_span and rows:
        t = _as_float(rows[-1], "sim_time_0", _as_float(rows[-1], "time"))
        duration = t - current_start
        if duration > best_duration:
            best_duration = duration
            best_start = current_start
            best_end = t
    return best_duration, best_start, best_end


def main() -> int:
    db = latest_db()
    export_dir = export_db(db)
    rows = load_drone_rows(export_dir)
    if len(rows) < 2:
        raise SystemExit(f"Not enough rows in {export_dir / 'drone.csv'}")

    z_values = [_as_float(row, "world_pos_z", math.nan) for row in rows]
    max_z = max(z for z in z_values if not math.isnan(z))
    hover_duration, hover_start, hover_end = longest_hover_span(rows)

    thrust_cols = [
        "motor_thrust_br",
        "motor_thrust_fr",
        "motor_thrust_bl",
        "motor_thrust_fl",
    ]
    max_thrust_sum = max(sum(_as_float(row, col) for col in thrust_cols) for row in rows)

    print(f"Verified DB: {db.name}")
    print(f"Export: {export_dir}")
    print(f"Rows: {len(rows)}")
    print(f"Max altitude: {max_z:.2f} m")
    print(f"Longest hover span [1.0, 2.5] m: {hover_duration:.2f}s ({hover_start:.2f}-{hover_end:.2f}s)")
    print(f"Max motor thrust sum: {max_thrust_sum:.2f} N")

    failures = []
    if max_z <= 1.0:
        failures.append("max altitude never exceeded 1.0 m")
    if hover_duration < 3.0:
        failures.append("hover span in [1.0, 2.5] m was shorter than 3.0 s")
    if max_thrust_sum <= 4.0:
        failures.append("motor thrust never exceeded 4 N")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: drone took off and hovered for at least 3 seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
