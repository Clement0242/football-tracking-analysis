"""Tests on synthetic trajectories with known answers (no data download needed)."""

import numpy as np
import pandas as pd
import pytest

from pitchtrack.io import FPS
from pitchtrack.kinematics import add_velocities, detect_efforts, speed_zone_distances
from pitchtrack.pitch_control import PitchControlParams, integrate_control
from pitchtrack.profiles import acceleration_speed_profile


def make_track(x: np.ndarray, y: np.ndarray | None = None, player: str = "Home_1") -> pd.DataFrame:
    n = len(x)
    y = np.zeros(n) if y is None else y
    return pd.DataFrame({
        "period": 1, "frame": np.arange(n), "time": np.arange(n) / FPS,
        f"{player}_x": x, f"{player}_y": y, "ball_x": 0.0, "ball_y": 0.0,
    })


def test_constant_speed_is_recovered():
    t = np.arange(10 * FPS) / FPS
    df = add_velocities(make_track(5.0 * t))
    speed = df["Home_1_speed"].dropna()
    assert speed.median() == pytest.approx(5.0, abs=1e-6)
    assert df["Home_1_acc"].dropna().abs().max() < 1e-6


def test_distance_in_speed_zones_sums_to_path_length():
    t = np.arange(20 * FPS) / FPS
    df = add_velocities(make_track(6.0 * t))  # 6 m/s = high-speed running zone
    zones = speed_zone_distances(df, "Home_1")
    assert zones["hsr"] == pytest.approx(6.0 * 20, rel=0.05)
    assert zones["walk"] == pytest.approx(0, abs=1)


def test_one_hard_acceleration_is_detected():
    # Stand still 3 s, accelerate at 4 m/s^2 for 1.5 s, then run steadily.
    dt = 1 / FPS
    acc = np.r_[np.zeros(3 * FPS), np.full(int(1.5 * FPS), 4.0), np.zeros(5 * FPS)]
    v = np.cumsum(acc) * dt
    x = np.cumsum(v) * dt
    df = add_velocities(make_track(x))
    efforts = detect_efforts(df, "Home_1", threshold=2.5, min_duration=0.5)
    assert [e.kind for e in efforts] == ["acceleration"]
    # Speed change is measured only while acceleration stays above the
    # threshold; smoothing spreads the 6 m/s gain over a longer ramp.
    assert 4.0 < efforts[0].speed_change <= 6.0
    assert 3.5 < efforts[0].peak < 4.5


def test_tracking_glitch_is_removed():
    t = np.arange(10 * FPS) / FPS
    x = 3.0 * t
    x[100] += 8.0  # 8 m teleport in one frame = 200 m/s
    df = add_velocities(make_track(x))
    assert df["Home_1_speed"].max() < 6.0


def test_as_profile_recovers_linear_relation():
    rng = np.random.default_rng(0)
    speed = rng.uniform(0, 9, 20000)
    a0, s0 = 7.0, 9.5
    ceiling = a0 * (1 - speed / s0)
    acc = ceiling * rng.uniform(0, 1, speed.size) ** 6   # most efforts are sub-maximal
    acc[::50] = ceiling[::50]                            # a few maximal ones
    prof = acceleration_speed_profile(speed, acc)
    assert prof.a0 == pytest.approx(a0, rel=0.05)
    assert prof.s0 == pytest.approx(s0, rel=0.05)


def test_pitch_control_is_symmetric_and_bounded():
    targets = np.array([[-10.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    att = np.array([[-10.0, 0.0]])
    dfn = np.array([[10.0, 0.0]])
    zeros = np.zeros((1, 2))
    pc = integrate_control(targets, np.array([0.0, 0.0]), att, zeros, dfn, zeros,
                           att_gk=None, def_gk=None, params=PitchControlParams())
    assert np.all((pc >= 0) & (pc <= 1))
    assert pc[0] > 0.95 and pc[2] < 0.05          # each player owns his own spot
    assert pc[1] == pytest.approx(0.5, abs=1e-6)  # equidistant point is a coin flip


def test_pitch_control_favours_the_faster_arrival():
    targets = np.array([[0.0, 0.0]])
    att = np.array([[-5.0, 0.0]])
    dfn = np.array([[8.0, 0.0]])
    zeros = np.zeros((1, 2))
    pc = integrate_control(targets, np.array([0.0, 0.0]), att, zeros, dfn, zeros, None, None)
    assert pc[0] > 0.7
