"""Metabolic and mechanical power from tracking data, without a black box.

Commercial systems report "metabolic power" or "mechanical power" without
showing the calculation. This module writes every step out, one equation
per function, each referenced to the paper it comes from, so any number can
be traced back to speed and acceleration.

Two families of models are implemented.

A. Metabolic power, di Prampero & Osgnach (2018), updating Osgnach et al. (2010)
   ------------------------------------------------------------------------------
   Idea: accelerating on flat ground is biomechanically equivalent to running
   uphill at constant speed. The slope is set by the acceleration; decelerating
   is running downhill. The energy cost of uphill/downhill running is known
   from Minetti et al. (2002), so the cost of accelerated running follows.

     ES  = a/g + k v^2 / g                          equivalent slope (eq. 4, 6, 7)
     EM  = sqrt(a^2/g^2 + 1)                        equivalent mass (eq. 8)
     EC  = (155.4 ES^5 - 30.4 ES^4 - 43.3 ES^3
            + 46.3 ES^2 + 19.5 ES + EC0) * EM       running cost, J/kg/m (eq. 10)
     If v < v_trans(ES), the walking cost (eq. 12-14, Table 1) is used instead.
     P_met = EC * v                                 W/kg (eq. 1)

   EC0 = 3.6 J/kg/m on compact flat ground (Minetti 2002). Osgnach et al.
   (2010) multiply by KT = 1.29 for grass; that is a parameter here.

B. External mechanical power, di Prampero et al. (2023) and Osgnach et al. (2023)
   -------------------------------------------------------------------------------
     W_const = 1.3 - 0.4 / v                        J/kg/m to run at constant speed (Part I eq. 2)
     P_ext   = W_const * v + a * v                  W/kg (Part I eq. 1)

   ``a * v`` is the power to change kinetic energy. During a deceleration it
   is negative: energy is absorbed (eccentric work), not produced. That
   braking component is reported separately, because Buchheit et al. (GPS
   3.0) stress that eccentric load is what classic metrics miss.

   Individual threshold ("burst", Osgnach et al. 2023, Part II):
     the in-situ acceleration-speed profile a_max(v) = A0 (1 - v/S0) gives the
     highest power the player can produce at each speed,
       EP_ASP(v) = W_const v + a_max(v) v
     A burst is an effort where P_ext > 80 % of EP_ASP(v). Work above that
     threshold (high EW) is split into force (0-1/3 S0), power (1/3-2/3 S0)
     and speed (2/3 S0-S0) zones.

C. Direction-sensitive extension (Buchheit et al., GPS 3.0)
   ---------------------------------------------------------
   Speed is a scalar; velocity is a vector. Turning at constant speed is an
   acceleration (centripetal, v^2/r) that needs force. The tracking gives the
   full acceleration vector, so we also report

     a_total = sqrt(a_t^2 + a_c^2)                  complete acceleration
     P_turn  = a_c * v                              W/kg, "turning power"

   P_turn is not a work rate in the strict physical sense: a centripetal
   force does no work on the centre of mass. It is used here, as in GPS 3.0,
   as an *index* of the force a player must produce to change direction,
   expressed on the same scale as P_ext. The ADI algorithm behind Buchheit's
   own figures is proprietary; this is an open approximation of the idea,
   not a reproduction.

   Buchheit's split of mechanical work, made explicit:
     MW_stride = integral of P_const               stride-dominated running
     MW_thigh  = integral of (|a v| + P_turn)      accelerations, braking, turns
   and P_mech = P_const + |a v| + P_turn as the intensity (load) signal.
   P_mech adds absolute values: it is a *load index*, not net work.

D. Oxygen uptake, anaerobic energy and metabolic power events
   ----------------------------------------------------------
   Metabolic power is the energy *requirement*; VO2 can only follow it with
   a delay. VO2 is modelled as a first-order response to P_met, capped at
   VO2max:  dVO2/dt = (min(P_met, VO2max) - VO2) / tau.
   What VO2 does not cover comes from anaerobic stores (phosphocreatine,
   lactate):  AnE = integral of max(P_met - VO2, 0).
   A metabolic power event (MPE, Osgnach & di Prampero 2018; Osgnach et al.
   2023) is a spell where P_met exceeds the estimated VO2. Here an MPE must
   last at least 1 s and peak above 20 W/kg; tau = 20 s and VO2max = 18 W/kg
   (about 52 ml O2/kg/min above rest, the value drawn in Osgnach et al. 2023
   Fig. 3) are parameters. These operational choices are mine: the 2018
   Part 2 paper that defines the original procedure was not available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitchtrack.io import FPS, player_ids

G = 9.81          # m/s^2
K_AIR = 0.0037    # N s^2 kg^-1 m^-2, drag constant (di Prampero & Osgnach 2018)
EC0 = 3.6         # J/kg/m, net cost of running on flat compact terrain (Minetti 2002)
KT_GRASS = 1.29   # grass-terrain multiplier (Osgnach et al. 2010)

# Table 1 of di Prampero & Osgnach (2018): walking cost polynomial coefficients
# ECw = a v^4 + b v^3 + c v^2 + d v + e, for each equivalent slope.
WALK_TABLE = pd.DataFrame(
    [[+0.40, 94.62, -213.94, 184.43, -68.49, 25.04],
     [+0.30, 44.95, -122.88, 126.94, -57.46, 21.39],
     [+0.20, 3.80, -14.91, 22.94, -14.53, 11.24],
     [+0.10, 0.68, -4.17, 10.17, -10.31, 8.66],
     [0.00, 1.25, -6.57, 13.14, -11.15, 5.35],
     [-0.10, 0.69, -3.21, 5.94, -5.07, 2.79],
     [-0.20, 0.03, -0.15, 0.98, -2.25, 3.14],
     [-0.30, 0.28, -1.66, 3.81, -3.96, 4.01]],
    columns=["es", "a", "b", "c", "d", "e"]).sort_values("es").reset_index(drop=True)


# --------------------------------------------------------------------------
# A. Metabolic power (di Prampero & Osgnach 2018)
# --------------------------------------------------------------------------
def equivalent_slope(acc: np.ndarray, speed: np.ndarray, k_air: float = K_AIR) -> np.ndarray:
    """ES = a/g + k v^2/g (eq. 7). Air resistance adds a small uphill slope."""
    return acc / G + k_air * speed ** 2 / G


def equivalent_mass(acc: np.ndarray) -> np.ndarray:
    """EM = sqrt(a^2/g^2 + 1) (eq. 8). Always >= 1, also when decelerating."""
    return np.sqrt(acc ** 2 / G ** 2 + 1)


def running_cost(es: np.ndarray, em: np.ndarray, ec0: float = EC0) -> np.ndarray:
    """Energy cost of running, J/kg/m (eq. 10, Minetti 2002 polynomial).

    Minetti measured slopes from -0.45 to +0.45. Hard accelerations go beyond
    (a = 5 m/s^2 gives ES = 0.51); like the original papers, the polynomial
    is extrapolated rather than capped. Capping would flatten every peak:
    di Prampero et al. (2023) need ES = 0.53 to reach Bolt's 140 W/kg.
    """
    return (155.4 * es ** 5 - 30.4 * es ** 4 - 43.3 * es ** 3 + 46.3 * es ** 2
            + 19.5 * es + ec0) * em


def transition_speed(es: np.ndarray) -> np.ndarray:
    """Walk-run transition speed, m/s (eq. 12)."""
    return (-107.05 * es ** 5 + 113.13 * es ** 4 - 1.13 * es ** 3
            - 15.84 * es ** 2 - 1.70 * es + 2.27)


def walking_cost(speed: np.ndarray, es: np.ndarray, em: np.ndarray) -> np.ndarray:
    """Energy cost of walking, J/kg/m (eq. 13 and linear interpolation of eq. 14)."""
    es_c = np.clip(es, WALK_TABLE["es"].min(), WALK_TABLE["es"].max())
    grid = WALK_TABLE["es"].to_numpy()
    hi = np.clip(np.searchsorted(grid, es_c), 1, len(grid) - 1)
    lo = hi - 1

    def poly(row_idx, v):
        c = WALK_TABLE.iloc[row_idx][["a", "b", "c", "d", "e"]].to_numpy().T
        return c[0] * v ** 4 + c[1] * v ** 3 + c[2] * v ** 2 + c[3] * v + c[4]

    ec_lo, ec_hi = poly(lo, speed), poly(hi, speed)
    frac = (es_c - grid[lo]) / (grid[hi] - grid[lo])
    return (ec_lo + frac * (ec_hi - ec_lo)) * em


def metabolic_power(speed: np.ndarray, acc: np.ndarray, terrain: float = 1.0,
                    walking: bool = True) -> dict[str, np.ndarray]:
    """Instantaneous metabolic power, W/kg, plus every intermediate quantity.

    ``terrain``: 1.0 = compact flat ground (2018 model), 1.29 = grass as in
    Osgnach et al. (2010).
    """
    speed = np.asarray(speed, float)
    acc = np.asarray(acc, float)
    es = equivalent_slope(acc, speed)
    em = equivalent_mass(acc)
    ec_run = running_cost(es, em) * terrain
    if walking:
        is_walk = speed < transition_speed(es)
        ec_walk = walking_cost(speed, es, em) * terrain
        ec = np.where(is_walk, ec_walk, ec_run)
    else:
        is_walk = np.zeros_like(speed, dtype=bool)
        ec = ec_run
    return {"es": es, "em": em, "ec": ec, "walking": is_walk, "p_met": ec * speed}


# --------------------------------------------------------------------------
# B. External mechanical power (di Prampero et al. 2023, Part I)
# --------------------------------------------------------------------------
def constant_speed_work(speed: np.ndarray) -> np.ndarray:
    """W_ext-const = 1.3 - 0.4/v, J/kg/m (Part I eq. 2), valid while running.

    Below ~1 m/s the formula diverges, so it is held at its 1 m/s value.
    """
    return 1.3 - 0.4 / np.maximum(np.asarray(speed, float), 1.0)


def mechanical_power(speed: np.ndarray, acc: np.ndarray) -> dict[str, np.ndarray]:
    """External mechanical power and its components, W/kg (Part I eq. 1).

    - ``p_const``: cost of moving at that speed (lifting / re-accelerating
      the centre of mass at each stride);
    - ``p_acc``: a*v, change of kinetic energy, positive when speeding up;
    - ``p_ext``: p_const + max(p_acc, 0), concentric (positive) power;
    - ``p_brake``: -min(p_acc, 0), eccentric power absorbed when braking.
    """
    speed = np.asarray(speed, float)
    acc = np.asarray(acc, float)
    p_const = constant_speed_work(speed) * speed
    p_acc = acc * speed
    return {"p_const": p_const, "p_acc": p_acc,
            "p_ext": p_const + np.maximum(p_acc, 0), "p_brake": -np.minimum(p_acc, 0)}


def max_external_power(speed: np.ndarray, a0: float, s0: float) -> np.ndarray:
    """EP_ASP(v): highest external power reachable at speed v (Part II)."""
    speed = np.asarray(speed, float)
    a_max = np.clip(a0 * (1 - speed / s0), 0, None)
    return constant_speed_work(speed) * speed + a_max * speed


# --------------------------------------------------------------------------
# Bursts and high-intensity work (Osgnach et al. 2023, Part II)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Burst:
    start_frame: int
    end_frame: int
    duration: float
    peak_power: float        # W/kg
    start_speed: float       # m/s
    zone: str                # force / power / speed, by start speed vs S0


def detect_bursts(speed: np.ndarray, p_ext: np.ndarray, frames: np.ndarray,
                  a0: float, s0: float, fraction: float = 0.8,
                  min_duration: float = 0.3) -> tuple[list[Burst], np.ndarray]:
    """Events with P_ext above ``fraction`` x EP_ASP(v); returns (bursts, above-mask)."""
    thr = fraction * max_external_power(speed, a0, s0)
    # A burst needs a real acceleration: exclude constant-speed running,
    # where P_ext is close to the threshold only at top speed.
    above = np.nan_to_num(p_ext > thr) & (p_ext > constant_speed_work(speed) * speed + 1.0)
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    min_frames = int(round(min_duration * FPS))
    bursts = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_frames:
            continue
        v0 = speed[a]
        zone = "force" if v0 < s0 / 3 else ("power" if v0 < 2 * s0 / 3 else "speed")
        bursts.append(Burst(int(frames[a]), int(frames[b - 1]), (b - a) / FPS,
                            float(np.nanmax(p_ext[a:b])), float(v0), zone))
    return bursts, above


# --------------------------------------------------------------------------
# D. VO2 kinetics, anaerobic energy, metabolic power events
# --------------------------------------------------------------------------
TAU_VO2 = 20.0     # s
VO2MAX = 18.0      # W/kg above rest (~52 ml O2/kg/min, 20.9 J per ml O2)
MPE_MIN_PEAK = 20.0
MPE_MIN_DURATION = 1.0


def estimate_vo2(p_met: np.ndarray, tau: float = TAU_VO2, vo2max: float = VO2MAX) -> np.ndarray:
    """First-order VO2 response to the metabolic power demand, W/kg."""
    p = np.minimum(np.nan_to_num(p_met), vo2max)
    alpha = (1 / FPS) / tau
    vo2 = np.empty_like(p)
    level = p[0]
    for i, target in enumerate(p):   # exponential smoothing = Euler step of the ODE
        level += alpha * (target - level)
        vo2[i] = level
    return vo2


@dataclass(frozen=True)
class MPE:
    start_frame: int
    end_frame: int
    duration: float          # s
    mean_power: float        # W/kg
    peak_power: float        # W/kg
    recovery_before: float   # s since the previous MPE ended (nan for the first)


def detect_mpe(p_met: np.ndarray, vo2: np.ndarray, frames: np.ndarray,
               min_peak: float = MPE_MIN_PEAK,
               min_duration: float = MPE_MIN_DURATION) -> list[MPE]:
    above = np.nan_to_num(p_met) > vo2
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    events, last_end = [], None
    for a, b in zip(edges[::2], edges[1::2]):
        seg = p_met[a:b]
        if (b - a) / FPS < min_duration or np.nanmax(seg) < min_peak:
            continue
        rec = (a - last_end) / FPS if last_end is not None else float("nan")
        events.append(MPE(int(frames[a]), int(frames[b - 1]), (b - a) / FPS,
                          float(np.nanmean(seg)), float(np.nanmax(seg)), rec))
        last_end = b
    return events


# --------------------------------------------------------------------------
# E. Most intense periods of mechanical power (Buchheit, GPS 3.0)
# --------------------------------------------------------------------------
def most_intense_period(signal: np.ndarray, window_s: float) -> float:
    """Highest rolling mean of ``signal`` over ``window_s`` seconds."""
    w = int(window_s * FPS)
    x = np.nan_to_num(np.asarray(signal, float))
    if len(x) < w:
        return float("nan")
    c = np.cumsum(np.r_[0.0, x])
    return float(((c[w:] - c[:-w]) / w).max())


def _count_runs(mask: np.ndarray, min_duration: float) -> int:
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask).astype(int), 0]))
    return int(sum((b - a) / FPS >= min_duration for a, b in zip(edges[::2], edges[1::2])))


# --------------------------------------------------------------------------
# Per-player summary
# --------------------------------------------------------------------------
def power_summary(df: pd.DataFrame, profiles: pd.DataFrame | None = None,
                  terrain: float = 1.0, burst_fraction: float = 0.8) -> pd.DataFrame:
    """One row per player: energy, mechanical work, braking, turning, bursts.

    ``profiles``: optional table indexed by player with ``A0_ms2``, ``S0_ms``
    (from :mod:`pitchtrack.profiles`); when given, bursts use the player's
    own profile, otherwise a population profile (A0 = 6.5, S0 = 9.5).
    """
    rows = []
    dt = 1 / FPS
    frames = df["frame"].to_numpy()
    # Players without a reliable profile of their own get the median of the
    # reliable ones (Osgnach et al. 2023 used the average of the same role).
    fallback_a0, fallback_s0 = 6.43, 9.23   # Serie B team average, Osgnach et al. 2023
    if profiles is not None and "reliable" in profiles and profiles["reliable"].any():
        rel = profiles[profiles["reliable"].astype(bool)]
        fallback_a0, fallback_s0 = float(rel["A0_ms2"].median()), float(rel["S0_ms"].median())
    for p in player_ids(df):
        speed = df[f"{p}_speed"].to_numpy(float)
        acc = df[f"{p}_acc"].to_numpy(float)
        acc_c = df.get(f"{p}_acc_c", pd.Series(0.0, index=df.index)).to_numpy(float)
        ok = np.isfinite(speed) & np.isfinite(acc)
        minutes = ok.sum() * dt / 60
        if minutes < 1:
            continue
        s, a, ac = np.where(ok, speed, 0), np.where(ok, acc, 0), np.nan_to_num(acc_c)

        met = metabolic_power(s, a, terrain=terrain)
        met_speed_only = metabolic_power(s, np.zeros_like(s), terrain=terrain)
        mech = mechanical_power(s, a)
        p_turn = ac * s

        a0, s0, own = fallback_a0, fallback_s0, False
        if profiles is not None and p in profiles.index and bool(profiles.loc[p].get("reliable", True)):
            a0, s0, own = float(profiles.loc[p, "A0_ms2"]), float(profiles.loc[p, "S0_ms"]), True
        bursts, above = detect_bursts(s, mech["p_ext"], frames, a0, s0, burst_fraction)
        high_ew = (mech["p_ext"] * above * ok).sum() * dt
        zone_of = np.where(s < s0 / 3, "force", np.where(s < 2 * s0 / 3, "power", "speed"))

        vo2 = estimate_vo2(np.where(ok, met["p_met"], 0))
        mpes = detect_mpe(np.where(ok, met["p_met"], 0), vo2, frames)
        anaerobic = (np.maximum(met["p_met"] - vo2, 0) * ok).sum() * dt / 1000
        p_mech = mech["p_const"] + np.abs(mech["p_acc"]) + p_turn
        mw_stride = (mech["p_const"] * ok).sum() * dt / 1000
        mw_thigh = ((np.abs(mech["p_acc"]) + p_turn) * ok).sum() * dt / 1000
        energy = (met["p_met"] * ok).sum() * dt / 1000                # kJ/kg
        energy_speed = (met_speed_only["p_met"] * ok).sum() * dt / 1000
        rows.append({
            "player": p, "minutes": round(minutes, 1),
            "energy_kj_kg": round(energy, 1),
            "energy_speed_only_kj_kg": round(energy_speed, 1),
            "edi": round(energy / energy_speed, 3) if energy_speed else np.nan,
            "p_met_mean": round(np.nanmean(met["p_met"][ok]), 2),
            "time_above_20w_s": round((met["p_met"] > 20).sum() * dt),
            # Distance above 20 W/kg over total distance (Osgnach et al. 2023: 0.247)
            "d_mp20_ratio": round(float((s * (met["p_met"] > 20) * ok).sum() / max((s * ok).sum(), 1e-9)), 3),
            "acc_events_2_5": _count_runs(a * ok >= 2.5, 0.5),
            "time_walking_pct": round(100 * met["walking"][ok].mean(), 1),
            "ext_work_kj_kg": round((mech["p_ext"] * ok).sum() * dt / 1000, 2),
            "brake_work_kj_kg": round((mech["p_brake"] * ok).sum() * dt / 1000, 2),
            "turn_index_kj_kg": round((p_turn * ok).sum() * dt / 1000, 2),
            "turn_share_pct": round(100 * (p_turn * ok).sum() /
                                    max(((np.maximum(mech["p_acc"], 0) + p_turn) * ok).sum(), 1e-9), 1),
            "p_ext_mean": round(np.nanmean(mech["p_ext"][ok]), 2),
            "bursts": len(bursts),
            "bursts_per_90": round(90 * len(bursts) / minutes, 1),
            "high_ew_j_kg": round(high_ew),
            **{f"high_ew_{z}_j_kg": round((mech["p_ext"] * above * ok * (zone_of == z)).sum() * dt)
               for z in ("force", "power", "speed")},
            "mpe": len(mpes),
            "mpe_per_15min": round(15 * len(mpes) / minutes, 1),
            "mpe_duration_s": round(float(np.mean([e.duration for e in mpes])), 1) if mpes else np.nan,
            "mpe_power_w_kg": round(float(np.mean([e.mean_power for e in mpes])), 1) if mpes else np.nan,
            "mpe_recovery_s": round(float(np.nanmean([e.recovery_before for e in mpes])), 1) if len(mpes) > 1 else np.nan,
            "anaerobic_share_pct": round(100 * anaerobic / energy, 1) if energy else np.nan,
            "mw_stride_kj_kg": round(mw_stride, 2),
            "mw_thigh_kj_kg": round(mw_thigh, 2),
            "mw_thigh_pct": round(100 * mw_thigh / (mw_stride + mw_thigh), 1),
            "p_mech_mean": round(float(np.mean(p_mech[ok])), 2),
            "p_mech_mip_1min": round(most_intense_period(np.where(ok, p_mech, 0), 60), 2),
            "p_mech_mip_3min": round(most_intense_period(np.where(ok, p_mech, 0), 180), 2),
            "iet_s": round(above.sum() * dt, 1),
            "a0_used": round(a0, 2), "s0_used": round(s0, 2),
            "own_profile": own,
        })
    return pd.DataFrame(rows).set_index("player")


def power_time_series(df: pd.DataFrame, player: str, terrain: float = 1.0) -> pd.DataFrame:
    """Frame-by-frame power components for one player (for plots and audits)."""
    s = df[f"{player}_speed"].to_numpy(float)
    a = df[f"{player}_acc"].to_numpy(float)
    ac = np.nan_to_num(df.get(f"{player}_acc_c", pd.Series(0.0, index=df.index)).to_numpy(float))
    met = metabolic_power(np.nan_to_num(s), np.nan_to_num(a), terrain=terrain)
    mech = mechanical_power(np.nan_to_num(s), np.nan_to_num(a))
    return pd.DataFrame({"frame": df["frame"].to_numpy(), "time": df["time"].to_numpy(),
                         "speed": s, "acc": a, "acc_c": ac, "es": met["es"], "em": met["em"],
                         "ec": met["ec"], "walking": met["walking"], "p_met": met["p_met"],
                         "p_const": mech["p_const"], "p_acc": mech["p_acc"],
                         "p_ext": mech["p_ext"], "p_brake": mech["p_brake"], "p_turn": ac * s,
                         "p_mech": mech["p_const"] + np.abs(mech["p_acc"]) + ac * s,
                         "vo2": estimate_vo2(met["p_met"])})
