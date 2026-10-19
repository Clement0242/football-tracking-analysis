"""Power models checked against numbers printed in the source papers."""

import numpy as np
import pandas as pd
import pytest

from pitchtrack import power as P
from pitchtrack.io import FPS
from pitchtrack.kinematics import add_velocities


def test_walking_cost_reproduces_worked_example():
    # di Prampero & Osgnach (2018), p. 585: v = 1 m/s, ES = 0.24 -> 10.296 J/kg/m
    # (before multiplying by EM, so EM is set to 1 here).
    ec = P.walking_cost(np.array([1.0]), np.array([0.24]), np.array([1.0]))
    assert ec[0] == pytest.approx(10.296, abs=0.01)


@pytest.mark.parametrize("es, expected", [(0.20, 8.54), (0.30, 12.93)])
def test_walking_table_rows(es, expected):
    ec = P.walking_cost(np.array([1.0]), np.array([es]), np.array([1.0]))
    assert ec[0] == pytest.approx(expected, abs=0.02)


def test_equivalent_slope_and_mass():
    a, v = np.array([3.0]), np.array([5.0])
    assert P.equivalent_slope(a, v)[0] == pytest.approx(3 / 9.81 + 0.0037 * 25 / 9.81)
    assert P.equivalent_mass(a)[0] == pytest.approx(np.sqrt(1 + (3 / 9.81) ** 2))
    # Decelerating: negative slope, but the body still "weighs" more.
    assert P.equivalent_slope(-a, v)[0] < 0
    assert P.equivalent_mass(-a)[0] > 1


def test_constant_speed_running_cost_is_ec0_plus_air():
    v = np.array([4.0])
    m = P.metabolic_power(v, np.zeros(1), walking=False)
    es = 0.0037 * 16 / 9.81
    expected_ec = 155.4 * es**5 - 30.4 * es**4 - 43.3 * es**3 + 46.3 * es**2 + 19.5 * es + 3.6
    assert m["ec"][0] == pytest.approx(expected_ec)
    assert m["p_met"][0] == pytest.approx(expected_ec * 4.0)


def test_acceleration_multiplies_metabolic_power():
    # 2018 paper: at 9 km/h, ~10 W/kg at constant speed, ~5.5 times more at 4 m/s^2.
    v = np.array([2.5])
    steady = P.metabolic_power(v, np.zeros(1))["p_met"][0]
    hard = P.metabolic_power(v, np.array([4.0]))["p_met"][0]
    assert 8 < steady < 11
    assert 4.5 < hard / steady < 6.0


def test_bolt_peak_metabolic_power_order_of_magnitude():
    # di Prampero et al. (2023): peak ~140 W/kg at v ~ 5 m/s, which needs ES ~ 0.53,
    # i.e. a ~ 5.2 m/s^2. Capping ES at Minetti's 0.45 would make this unreachable.
    p = P.metabolic_power(np.array([5.0]), np.array([5.2]), walking=False)["p_met"][0]
    assert 125 < p < 155


def test_external_power_components():
    v, a = np.array([5.0]), np.array([2.0])
    m = P.mechanical_power(v, a)
    assert m["p_const"][0] == pytest.approx((1.3 - 0.4 / 5) * 5)
    assert m["p_acc"][0] == pytest.approx(10.0)
    assert m["p_ext"][0] == pytest.approx(m["p_const"][0] + 10.0)
    braking = P.mechanical_power(v, -a)
    assert braking["p_brake"][0] == pytest.approx(10.0)
    assert braking["p_ext"][0] == pytest.approx(braking["p_const"][0])


def test_max_power_dome_ends_at_constant_speed_cost():
    a0, s0 = 6.5, 9.5
    assert P.max_external_power(np.array([0.0]), a0, s0)[0] == pytest.approx(0.0)
    top = P.max_external_power(np.array([s0]), a0, s0)[0]
    assert top == pytest.approx(P.constant_speed_work(np.array([s0]))[0] * s0)


def test_vo2_reaches_steady_state_and_respects_vo2max():
    p = np.full(10 * 60 * FPS, 12.0)
    vo2 = P.estimate_vo2(p)
    assert vo2[-1] == pytest.approx(12.0, rel=1e-3)
    assert P.estimate_vo2(np.full(60 * FPS, 40.0)).max() <= P.VO2MAX + 1e-9


def test_one_sprint_gives_one_mpe():
    p = np.full(120 * FPS, 6.0)
    p[60 * FPS: 66 * FPS] = 45.0
    vo2 = P.estimate_vo2(p)
    events = P.detect_mpe(p, vo2, np.arange(len(p)))
    assert len(events) == 1
    assert events[0].duration == pytest.approx(6.0, abs=0.1)


def test_centripetal_acceleration_on_a_circle():
    # Constant 5 m/s on a 10 m radius circle: a_c = v^2 / r = 2.5 m/s^2, a_t = 0.
    t = np.arange(30 * FPS) / FPS
    theta = 5.0 / 10.0 * t
    df = pd.DataFrame({"period": 1, "frame": np.arange(len(t)), "time": t,
                       "Home_1_x": 10 * np.cos(theta), "Home_1_y": 10 * np.sin(theta),
                       "ball_x": 0.0, "ball_y": 0.0})
    d = add_velocities(df)
    assert d["Home_1_acc_c"].dropna().median() == pytest.approx(2.5, rel=0.03)
    assert d["Home_1_acc"].dropna().abs().median() < 0.05


def test_centripetal_noise_floor_is_low_for_straight_running():
    rng = np.random.default_rng(1)
    t = np.arange(60 * FPS) / FPS
    df = pd.DataFrame({"period": 1, "frame": np.arange(len(t)), "time": t,
                       "Home_1_x": 4.0 * t + rng.normal(0, 0.02, len(t)),
                       "Home_1_y": rng.normal(0, 0.02, len(t)),
                       "ball_x": 0.0, "ball_y": 0.0})
    d = add_velocities(df)
    assert d["Home_1_acc_c"].dropna().mean() < 0.15
