"""Unit tests for the gate-pass detection geometry and schematic rendering."""

from sim.course import (
    Gate,
    GATE_ASSET,
    GATE_INNER_W,
    detect_gate_pass,
    gate_name,
    schematic_for,
    EASY_COURSE,
)


# Helper: a single gate at +X with default inner box
gate_x = (Gate(0, (5.0, 0.0, 1.5)),)


def test_x_gate_pass_centered():
    # Drone moves from x=4 to x=6, dead-center on Y/Z
    assert detect_gate_pass(gate_x, -1, (4.0, 0.0, 1.5), (6.0, 0.0, 1.5)) == 0


def test_x_gate_no_pass_when_outside_inner_y():
    # Crosses plane but Y is too far off-center
    assert detect_gate_pass(gate_x, -1, (4.0, 1.0, 1.5), (6.0, 1.0, 1.5)) is None


def test_x_gate_no_pass_when_outside_inner_z():
    assert detect_gate_pass(
        gate_x, -1, (4.0, 0.0, 5.0), (6.0, 0.0, 5.0)
    ) is None


def test_x_gate_no_pass_going_backwards():
    # x is decreasing (drone flying backwards)
    assert detect_gate_pass(gate_x, -1, (6.0, 0.0, 1.5), (4.0, 0.0, 1.5)) is None


def test_x_gate_no_pass_when_already_done():
    # last_gate_passed = 0 means gate 0 already done; next is 1 which doesn't exist
    assert detect_gate_pass(gate_x, 0, (4.0, 0.0, 1.5), (6.0, 0.0, 1.5)) is None


def test_pass_on_inner_boundary_y():
    half = GATE_INNER_W / 2.0
    # Exactly on the Y boundary should still count (inclusive on the half)
    assert detect_gate_pass(
        gate_x, -1, (4.0, half - 1e-6, 1.5), (6.0, half - 1e-6, 1.5)
    ) == 0


def test_pass_just_outside_inner_y():
    half = GATE_INNER_W / 2.0
    assert detect_gate_pass(
        gate_x, -1, (4.0, half + 0.01, 1.5), (6.0, half + 0.01, 1.5)
    ) is None


def test_easy_course_is_three_x_gates_ahead_of_origin():
    assert len(EASY_COURSE) == 3
    for g in EASY_COURSE:
        assert g.center[1] == 0.0
        assert g.center[2] == 1.8
    xs = [g.center[0] for g in EASY_COURSE]
    assert xs == sorted(xs)
    assert xs == [10.0, 20.0, 30.0]


def test_gate_name_stable():
    assert gate_name(0) == "gate_0"
    assert gate_name(2) == "gate_2"
    assert gate_name(31) == "gate_31"


def test_schematic_renders_one_glb_per_gate():
    s = schematic_for(EASY_COURSE)
    assert s.count("object_3d") == len(EASY_COURSE)
    assert s.count(f'glb path="{GATE_ASSET}"') == len(EASY_COURSE)
    for g in EASY_COURSE:
        ref = f"{gate_name(g.index)}.world_pos"
        assert s.count(ref) == 1, f"{ref} should appear exactly once"
