"""Figures and tables for the mechanical / metabolic power analysis.

    python scripts/make_power_figures.py

Needs results/acceleration_speed_profiles_game2.csv (from make_figures.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchtrack import power as P
from pitchtrack import viz
from pitchtrack.io import FPS, read_tracking
from pitchtrack.kinematics import add_velocities

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
FULL = 85


def label(p: str) -> str:
    team, n = p.split("_")
    return f"{team} #{n}"


# --------------------------------------------------------------------------
def fig_anatomy(df: pd.DataFrame, player: str, a0: float, s0: float) -> dict:
    """One real burst, decomposed step by step. Returns the worked-example numbers."""
    ts = P.power_time_series(df, player)
    bursts, _ = P.detect_bursts(ts["speed"].to_numpy(), ts["p_ext"].to_numpy(),
                                ts["frame"].to_numpy(), a0, s0)
    # Pick the burst with the highest peak power that starts from low speed.
    b = max((x for x in bursts if x.start_speed < 3), key=lambda x: x.peak_power)
    i0 = int(np.searchsorted(ts["frame"], b.start_frame))
    win = ts.iloc[max(0, i0 - 2 * FPS): i0 + 6 * FPS].copy()
    t = win["time"] - win["time"].iloc[0]
    k = int(win["p_met"].to_numpy().argmax())          # instant used in the worked example

    fig, axes = plt.subplots(4, 1, figsize=(8.6, 10.2), sharex=True,
                             gridspec_kw={"hspace": 0.38})
    ax = axes[0]
    ax.plot(t, win["speed"], color=viz.SERIES[0], label="Speed v (m/s)")
    ax2 = ax.twinx()
    ax2.plot(t, win["acc"], color=viz.SERIES[1], label="Acceleration a (m/s²)")
    ax2.axhline(0, color=viz.BASELINE, lw=0.8)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_ylabel("v (m/s)")
    ax2.set_ylabel("a (m/s²)")
    ax.set_title("① Tracking gives speed and acceleration")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)

    ax = axes[1]
    ax.plot(t, win["es"], color=viz.SERIES[0], label="Equivalent slope  ES = a/g + k·v²/g")
    ax.plot(t, win["em"] - 1, color=viz.SERIES[2], label="Extra equivalent mass  EM − 1 = √(a²/g² + 1) − 1")
    ax.axhline(0, color=viz.BASELINE, lw=0.8)
    ax.set_ylabel("dimensionless")
    ax.set_title("② Accelerating = running uphill: slope and 'heavier' body")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[2]
    ax.plot(t, win["ec"], color=viz.SERIES[0], label="Energy cost EC (Minetti polynomial × EM)")
    ax.axhline(P.EC0, color=viz.MUTED, ls="--", lw=1)
    ax.text(t.iloc[-1], P.EC0 + 1.6, f"constant-speed running: {P.EC0} J/kg/m",
            ha="right", fontsize=8.5, color=viz.INK_2)
    ax.set_ylabel("J/kg/m")
    ax.set_title("③ Cost of each metre on that 'slope'")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[3]
    ax.plot(t, win["p_met"], color=viz.SERIES[0], label="Metabolic power  P_met = EC · v")
    ax.plot(t, win["vo2"], color=viz.SERIES[2], label="Estimated VO₂ (first-order, τ = 20 s)")
    ax.fill_between(t, win["vo2"], win["p_met"], where=win["p_met"] > win["vo2"],
                    color=viz.SERIES[0], alpha=0.12, lw=0, label="Anaerobic share")
    ax.plot(t, win["p_ext"], color=viz.SERIES[1], label="External mechanical power  P_ext")
    ax.plot(t, win["p_mech"] - win["p_ext"] + 0 * win["p_ext"], color=viz.SERIES[6], lw=1.4,
            label="Braking + turning load (|a·v| when a<0, a_c·v)")
    ax.axvline(t.iloc[k], color=viz.MUTED, ls=":", lw=1)
    ax.set_ylabel("W/kg")
    ax.set_xlabel("Time (s)")
    ax.set_title("④ Power = cost × speed; mechanical vs metabolic")
    ax.set_ylim(top=win["p_met"].max() * 1.65)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(f"Anatomy of one burst — {label(player)}, game 2", x=0.02, ha="left",
                 fontsize=13, fontweight="semibold", y=0.995)
    fig.savefig(FIG / "power_anatomy.png")
    plt.close(fig)

    r = win.iloc[k]
    return {"player": player, "time_in_window_s": round(float(t.iloc[k]), 2),
            **{c: float(r[c]) for c in ("speed", "acc", "acc_c", "es", "em", "ec", "p_met",
                                         "p_const", "p_acc", "p_ext", "vo2")},
            "walking": bool(r["walking"])}


# --------------------------------------------------------------------------
def fig_dome(df: pd.DataFrame, player: str, a0: float, s0: float) -> None:
    """External power vs speed with the individual maximal-power dome and the 80 % threshold."""
    ts = P.power_time_series(df, player)
    ok = ts["speed"].notna() & ts["acc"].notna() & (ts["acc"] > 0)
    v = np.linspace(0, s0, 200)
    dome = P.max_external_power(v, a0, s0)
    const = P.constant_speed_work(v) * v
    bursts, above = P.detect_bursts(ts["speed"].to_numpy(), ts["p_ext"].to_numpy(),
                                    ts["frame"].to_numpy(), a0, s0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = ts[ok].iloc[::4]
    ax.scatter(sub["speed"], sub["p_ext"], s=3, color=viz.GRID, rasterized=True,
               label="All accelerating frames (1 in 4)")
    hi = ts[ok & above]
    ax.scatter(hi["speed"], hi["p_ext"], s=10, color=viz.SERIES[1], zorder=3,
               label=f"Above threshold ({len(bursts)} bursts)")
    ax.plot(v, dome, color=viz.INK, lw=2, label="Max power from the A-S profile, EP_ASP(v)")
    ax.plot(v, 0.8 * dome, color=viz.INK, lw=1.2, ls="--", label="80 % threshold (burst)")
    ax.plot(v, const, color=viz.SERIES[0], lw=1.6, label="Constant-speed running, W_const·v")
    for frac in (1 / 3, 2 / 3):
        ax.axvline(frac * s0, color=viz.BASELINE, lw=0.8, ls=":")
    for x, name in ((s0 / 6, "force"), (s0 / 2, "power"), (5 * s0 / 6, "speed")):
        ax.text(x, dome.max() * 1.07, name, ha="center", fontsize=9, color=viz.INK_2)
    ax.set_xlim(0, s0 * 1.02)
    ax.set_ylim(0, dome.max() * 1.15)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("External mechanical power (W/kg)")
    ax.set_title(f"An individual threshold instead of 'a > 2.5 m/s²' — {label(player)}")
    ax.legend(loc="lower right", fontsize=8.3, frameon=True, facecolor=viz.SURFACE,
              edgecolor=viz.GRID, framealpha=0.95)
    fig.savefig(FIG / "power_dome.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_work_split(summary: pd.DataFrame, per_player_parts: pd.DataFrame) -> None:
    """Buchheit's MW_stride vs MW_thigh, MW_thigh split into its three sources."""
    d = per_player_parts.loc[summary.index].sort_values("total")
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(d) + 1.6))
    left = np.zeros(len(d))
    parts = [("stride", "Stride (constant-speed running)", viz.BLUE_RAMP[1]),
             ("acc", "Accelerating (a·v > 0)", viz.SERIES[1]),
             ("brake", "Braking (|a·v|, a < 0)", viz.SERIES[7]),
             ("turn", "Turning (a_c·v)", viz.SERIES[6])]
    for col, name, color in parts:
        ax.barh([label(p) for p in d.index], d[col], left=left, color=color, height=0.62,
                edgecolor=viz.SURFACE, linewidth=2, label=name)
        left += d[col].to_numpy()
    for y, (tot, pct) in enumerate(zip(left, summary.loc[d.index, "mw_thigh_pct"])):
        ax.text(tot + 0.3, y, f"{tot:.0f} kJ/kg · thigh {pct:.0f} %", va="center", fontsize=8.5,
                color=viz.INK_2)
    ax.set_xlim(0, left.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mechanical load (kJ/kg)")
    ax.set_title("Where the mechanical load comes from — full-match outfield players, game 2")
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
    fig.savefig(FIG / "power_work_split.png")
    plt.close(fig)


def work_parts(df: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    rows = {}
    for p in players:
        ts = P.power_time_series(df, p)
        ok = ts["speed"].notna() & ts["acc"].notna()
        dt = 1 / FPS / 1000
        rows[p] = {"stride": (ts["p_const"] * ok).sum() * dt,
                   "acc": (ts["p_acc"].clip(lower=0) * ok).sum() * dt,
                   "brake": (ts["p_brake"] * ok).sum() * dt,
                   "turn": (ts["p_turn"] * ok).sum() * dt}
    out = pd.DataFrame(rows).T
    out["total"] = out.sum(axis=1)
    return out


# --------------------------------------------------------------------------
def fig_speed_vs_power(summary: pd.DataFrame) -> None:
    """Energy from speed alone vs with accelerations (EDI), and AnE/EE."""
    d = summary.sort_values("edi")
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.hlines(y, d["energy_speed_only_kj_kg"], d["energy_kj_kg"], color=viz.BASELINE, lw=2)
    ax.scatter(d["energy_speed_only_kj_kg"], y, s=60, color=viz.MUTED, zorder=3,
               edgecolor=viz.SURFACE, linewidth=2, label="Speed only (as if every metre were at constant pace)")
    ax.scatter(d["energy_kj_kg"], y, s=60, color=viz.SERIES[0], zorder=3,
               edgecolor=viz.SURFACE, linewidth=2, label="With accelerations (di Prampero & Osgnach 2018)")
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r["energy_kj_kg"] + 0.6, yi, f"+{100 * (r['edi'] - 1):.0f} %", va="center",
                fontsize=8.5, color=viz.INK_2)
    ax.set_yticks(y, [label(p) for p in d.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Energy expenditure above rest (kJ/kg)")
    ax.set_title("What speed-based reports miss: the cost of accelerating")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
    fig.savefig(FIG / "power_energy_vs_speed.png")
    plt.close(fig)


def main() -> None:
    viz.apply_style()
    profiles = pd.read_csv(RES / "acceleration_speed_profiles_game2.csv", index_col=0)

    summaries, parts = [], []
    frames = {}
    for game in (1, 2):
        for side in ("home", "away"):
            df = add_velocities(read_tracking(game, side))
            frames[(game, side)] = df
            s = P.power_summary(df, profiles if game == 2 else None)
            s.insert(0, "game", game)
            summaries.append(s)
            if game == 2:
                full = s[(s["minutes"] >= FULL) & (s["energy_kj_kg"] > 20)].index.tolist()
                parts.append(work_parts(df, full))
    summary = pd.concat(summaries)
    summary.to_csv(RES / "power_summary.csv")

    g2 = summary[(summary["game"] == 2) & (summary["minutes"] >= FULL) & (summary["energy_kj_kg"] > 20)]
    fig_speed_vs_power(g2)
    fig_work_split(g2, pd.concat(parts))

    home2 = frames[(2, "home")]
    player = "Home_6"                    # reliable own A-S profile (R² 0.95)
    a0, s0 = float(profiles.loc[player, "A0_ms2"]), float(profiles.loc[player, "S0_ms"])
    fig_dome(home2, player, a0, s0)
    example = fig_anatomy(home2, player, a0, s0)
    pd.Series(example).to_json(RES / "power_worked_example.json", indent=2)

    outfield = summary[(summary["minutes"] >= FULL) & (summary["energy_kj_kg"] > 20)]
    cmp = pd.DataFrame({
        "this_repo_median": outfield[["energy_kj_kg", "edi", "anaerobic_share_pct", "d_mp20_ratio",
                                      "mpe_per_15min", "mpe_duration_s", "mpe_power_w_kg",
                                      "mpe_recovery_s", "bursts_per_90", "acc_events_2_5"]].median(),
    })
    print(cmp.round(3))
    ratio = (outfield["bursts"] / outfield["acc_events_2_5"].replace(0, np.nan)).median()
    print("burst / acc>2.5 ratio:", round(ratio, 2))
    print(example)


if __name__ == "__main__":
    main()
