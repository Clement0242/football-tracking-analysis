"""In-situ acceleration-speed (A-S) profile.

Following Morin et al. (2021, *Journal of Biomechanics*), the maximal
acceleration a player produces drops roughly linearly with the speed he
starts from. Fitting that line on match data gives two numbers a staff can
track without a dedicated sprint test:

- ``A0``: theoretical maximal acceleration at zero speed (m/s^2);
- ``S0``: theoretical maximal speed, where acceleration reaches zero (m/s).

Method: from 3 m/s upward, split speed into 0.2 m/s bins, keep the two
highest accelerations of each bin, fit a line, drop points whose residual is
beyond 2 standard deviations, and fit again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ASProfile:
    a0: float          # m/s^2
    s0: float          # m/s
    slope: float       # (m/s^2) per (m/s)
    r2: float
    points: pd.DataFrame  # speed / acc points used in the final fit


def acceleration_speed_profile(speed: np.ndarray, acc: np.ndarray,
                               min_speed: float = 3.0, bin_width: float = 0.2,
                               top_n: int = 2) -> ASProfile | None:
    speed = np.asarray(speed, dtype=float)
    acc = np.asarray(acc, dtype=float)
    ok = np.isfinite(speed) & np.isfinite(acc) & (speed >= min_speed) & (acc > 0)
    if ok.sum() < 20:
        return None
    df = pd.DataFrame({"speed": speed[ok], "acc": acc[ok]})
    df["bin"] = np.floor(df["speed"] / bin_width).astype(int)
    top = (df.sort_values("acc", ascending=False)
             .groupby("bin", sort=False).head(top_n)
             .sort_values("speed"))
    if len(top) < 6:
        return None

    def fit(points: pd.DataFrame):
        slope, intercept = np.polyfit(points["speed"], points["acc"], 1)
        resid = points["acc"] - (intercept + slope * points["speed"])
        return slope, intercept, resid

    slope, intercept, resid = fit(top)
    top = top[np.abs(resid) <= 2 * resid.std(ddof=1)]
    slope, intercept, resid = fit(top)
    if slope >= 0:
        return None  # no usable relation (too few hard accelerations)

    ss_tot = ((top["acc"] - top["acc"].mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return ASProfile(a0=float(intercept), s0=float(-intercept / slope),
                     slope=float(slope), r2=float(r2),
                     points=top[["speed", "acc"]].reset_index(drop=True))
