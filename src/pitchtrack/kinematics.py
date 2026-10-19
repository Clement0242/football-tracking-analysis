"""Velocity, acceleration and high-intensity effort detection from tracking data.

Conventions follow common sport-science practice:
- positions are smoothed implicitly by filtering velocity with a
  Savitzky-Golay filter (raw 25 Hz differences are far too noisy);
- speeds above ``MAX_SPEED`` (12 m/s) are treated as tracking errors;
- an *effort* is an acceleration (or deceleration) whose magnitude stays
  above a threshold for at least ``min_duration`` seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pitchtrack.io import FPS, player_ids

MAX_SPEED = 12.0  # m/s, faster than any recorded sprint
MAX_ACC = 10.0    # m/s^2, beyond human capacity: treated as a tracking error
SPEED_ZONES = {  # m/s, lower bounds (inclusive)
    "walk": 0.0,
    "jog": 2.0,
    "run": 4.0,
    "hsr": 5.5,      # high-speed running
    "sprint": 7.0,
}


def _smooth(series: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay on each contiguous non-NaN run (keeps NaN gaps)."""
    out = np.full_like(series, np.nan, dtype=float)
    valid = ~np.isnan(series)
    if not valid.any():
        return out
    # Boundaries of contiguous valid segments
    edges = np.flatnonzero(np.diff(np.r_[0, valid.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        seg = series[start:stop]
        # Segments shorter than the filter are tracking fragments around
        # gaps; their derivatives are unreliable, so they stay NaN.
        if len(seg) > window:
            out[start:stop] = savgol_filter(seg, window, poly)
    return out


def add_velocities(df: pd.DataFrame, window: int = 13, poly: int = 1,
                   acc_window: int = 25, max_speed: float = MAX_SPEED,
                   max_acc: float = MAX_ACC) -> pd.DataFrame:
    """Add ``<player>_vx``, ``_vy``, ``_speed``, ``_acc`` and ``_acc_c`` columns.

    - ``acc``: tangential acceleration, i.e. the change of *speed* (m/s^2,
      signed). This is what GPS reports usually call acceleration.
    - ``acc_c``: centripetal acceleration, i.e. the change of *direction*
      (m/s^2, >= 0), equal to |v x a| / |v|. Running a curve at constant
      speed has zero tangential but non-zero centripetal acceleration;
      Buchheit et al. (GPS 3.0) argue this is the part classic metrics miss.

    Velocity is filtered over ``window`` frames (0.5 s at 25 Hz) and
    acceleration, a second derivative and so much noisier, over
    ``acc_window`` frames (1 s). With a shorter acceleration window the
    in-situ A-S profile gives A0 values above 15 m/s^2, which no footballer
    produces. Computed per period so half-time creates no artificial jump.
    """
    df = df.copy()
    dt = 1.0 / FPS
    new_cols = {}
    for p in player_ids(df):
        cols = {k: np.full(len(df), np.nan) for k in ("vx", "vy", "speed", "acc", "acc_c")}
        for _, idx in df.groupby("period").indices.items():
            x = df[f"{p}_x"].to_numpy(dtype=float)[idx]
            y = df[f"{p}_y"].to_numpy(dtype=float)[idx]
            vx = np.r_[np.nan, np.diff(x)] / dt
            vy = np.r_[np.nan, np.diff(y)] / dt
            raw_speed = np.hypot(vx, vy)
            bad = raw_speed > max_speed
            vx[bad] = np.nan
            vy[bad] = np.nan
            vx = _smooth(vx, window, poly)
            vy = _smooth(vy, window, poly)
            speed = np.hypot(vx, vy)
            acc = _smooth(np.r_[np.nan, np.diff(speed)] / dt, acc_window, poly)
            acc[np.abs(acc) > max_acc] = np.nan
            # Centripetal part: |v x a| / |v|, from the smoothed velocity vector.
            ax = _smooth(np.r_[np.nan, np.diff(vx)] / dt, acc_window, poly)
            ay = _smooth(np.r_[np.nan, np.diff(vy)] / dt, acc_window, poly)
            with np.errstate(invalid="ignore", divide="ignore"):
                acc_c = np.abs(vx * ay - vy * ax) / speed
            # Direction is undefined when nearly still: no centripetal load.
            acc_c[speed < 1.0] = 0.0
            acc_c[acc_c > max_acc] = np.nan
            cols["vx"][idx], cols["vy"][idx] = vx, vy
            cols["speed"][idx], cols["acc"][idx] = speed, acc
            cols["acc_c"][idx] = acc_c
        for k, v in cols.items():
            new_cols[f"{p}_{k}"] = v
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


@dataclass(frozen=True)
class Effort:
    player: str
    kind: str          # "acceleration" or "deceleration"
    start_frame: int
    end_frame: int
    duration: float    # s
    peak: float        # m/s^2 (signed)
    speed_change: float  # m/s (signed)


def detect_efforts(df: pd.DataFrame, player: str, threshold: float = 2.5,
                   min_duration: float = 0.5) -> list[Effort]:
    """Accelerations above +threshold and decelerations below -threshold.

    ``df`` must contain the columns produced by :func:`add_velocities`.
    """
    acc = df[f"{player}_acc"].to_numpy(dtype=float)
    speed = df[f"{player}_speed"].to_numpy(dtype=float)
    frames = df["frame"].to_numpy()
    periods = df["period"].to_numpy()
    min_frames = int(round(min_duration * FPS))
    efforts: list[Effort] = []

    for kind, mask in (("acceleration", acc >= threshold), ("deceleration", acc <= -threshold)):
        mask = np.nan_to_num(mask, nan=0).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, mask, 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start < min_frames or periods[start] != periods[stop - 1]:
                continue
            seg = acc[start:stop]
            peak = seg.max() if kind == "acceleration" else seg.min()
            efforts.append(Effort(
                player=player, kind=kind,
                start_frame=int(frames[start]), end_frame=int(frames[stop - 1]),
                duration=(stop - start) / FPS, peak=float(peak),
                speed_change=float(speed[stop - 1] - speed[start]),
            ))
    return sorted(efforts, key=lambda e: e.start_frame)


def speed_zone_distances(df: pd.DataFrame, player: str) -> dict[str, float]:
    """Distance covered (m) in each speed zone."""
    speed = df[f"{player}_speed"].to_numpy(dtype=float)
    dist = np.nan_to_num(speed) / FPS
    bounds = list(SPEED_ZONES.values()) + [np.inf]
    out = {}
    for (name, lo), hi in zip(SPEED_ZONES.items(), bounds[1:]):
        out[name] = float(dist[(speed >= lo) & (speed < hi)].sum())
    return out


def physical_summary(df: pd.DataFrame, threshold: float = 2.5,
                     min_duration: float = 0.5) -> pd.DataFrame:
    """One row per player: minutes, distances by zone, peak speed, efforts."""
    rows = []
    for p in player_ids(df):
        speed = df[f"{p}_speed"]
        minutes = speed.notna().sum() / FPS / 60
        if minutes < 1:
            continue
        zones = speed_zone_distances(df, p)
        efforts = detect_efforts(df, p, threshold, min_duration)
        n_acc = sum(e.kind == "acceleration" for e in efforts)
        n_dec = sum(e.kind == "deceleration" for e in efforts)
        rows.append({
            "player": p,
            "minutes": round(minutes, 1),
            "total_distance_m": round(sum(zones.values())),
            "hsr_m": round(zones["hsr"]),
            "sprint_m": round(zones["sprint"]),
            "peak_speed_ms": round(float(speed.quantile(0.999)), 2),
            "accelerations": n_acc,
            "decelerations": n_dec,
            "acc_per_90": round(90 * n_acc / minutes, 1),
            "dec_per_90": round(90 * n_dec / minutes, 1),
            "m_per_min": round(sum(zones.values()) / minutes, 1),
        })
    return pd.DataFrame(rows).set_index("player")


def rolling_intensity(df: pd.DataFrame, player: str, window_min: float = 5.0) -> pd.Series:
    """Rolling metres per minute over ``window_min`` minutes (peak-demand analysis)."""
    dist = df[f"{player}_speed"].fillna(0) / FPS
    window = int(window_min * 60 * FPS)
    return dist.rolling(window, min_periods=window).sum() / window_min
