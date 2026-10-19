"""Build every figure and results table from the Metrica sample games.

    python scripts/make_figures.py

Writes PNGs to ``figures/`` and CSVs to ``results/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitchtrack.io import FPS, attacking_direction, player_ids, read_events, read_tracking
from pitchtrack.kinematics import SPEED_ZONES, add_velocities, physical_summary, rolling_intensity, speed_zone_distances
from pitchtrack.pitch_control import pitch_control_surface
from pitchtrack.profiles import acceleration_speed_profile
from pitchtrack import viz

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FULL_MATCH_MIN = 85  # minutes on the pitch to count as a "full match" player

ZONE_LABELS = {"walk": "Walk <2 m/s", "jog": "Jog 2–4", "run": "Run 4–5.5",
               "hsr": "HSR 5.5–7", "sprint": "Sprint >7"}


def label(player: str) -> str:
    team, n = player.split("_")
    return f"{team} #{n}"


def load_game(game: int):
    home = add_velocities(read_tracking(game, "home"))
    away = add_velocities(read_tracking(game, "away"))
    return home, away, read_events(game)


# --------------------------------------------------------------------------
def fig_speed_zones(home: pd.DataFrame, summary: pd.DataFrame) -> None:
    players = summary[summary["minutes"] >= FULL_MATCH_MIN].sort_values("total_distance_m").index
    zones = pd.DataFrame({p: speed_zone_distances(home, p) for p in players}).T / 1000

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(players) + 1.6))
    left = np.zeros(len(players))
    for zone, color in zip(SPEED_ZONES, viz.BLUE_RAMP):
        ax.barh([label(p) for p in players], zones[zone], left=left, color=color,
                height=0.62, edgecolor=viz.SURFACE, linewidth=2, label=ZONE_LABELS[zone])
        left += zones[zone].to_numpy()
    for y, total in enumerate(left):
        ax.text(total + 0.12, y, f"{total:.1f} km", va="center", fontsize=9, color=viz.INK_2)
    ax.set_xlabel("Distance (km)")
    ax.set_xlim(0, left.max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title("Distance by speed zone — home team, full-match players")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8.5,
              handlelength=1.2, columnspacing=1.2)
    fig.savefig(FIG / "speed_zones.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def as_profiles(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in frames:
        for p in player_ids(df):
            prof = acceleration_speed_profile(df[f"{p}_speed"], df[f"{p}_acc"])
            minutes = df[f"{p}_speed"].notna().sum() / FPS / 60
            if prof is None or minutes < 30:
                continue
            rows.append({"player": p, "minutes": round(minutes, 1), "A0_ms2": round(prof.a0, 2),
                         "S0_ms": round(prof.s0, 2), "r2": round(prof.r2, 2),
                         # One match rarely holds enough maximal efforts: below
                         # R^2 0.7 the profile is reported but not trusted.
                         "reliable": prof.r2 >= 0.7})
    return pd.DataFrame(rows).set_index("player")


def fig_as_profile(df: pd.DataFrame, player: str) -> None:
    speed, acc = df[f"{player}_speed"], df[f"{player}_acc"]
    prof = acceleration_speed_profile(speed, acc)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ok = (acc > 0) & speed.notna()
    ax.scatter(speed[ok][::5], acc[ok][::5], s=3, color=viz.GRID, rasterized=True,
               label="All positive accelerations (1 in 5 frames)")
    ax.scatter(prof.points["speed"], prof.points["acc"], s=34, color=viz.SERIES[0],
               edgecolor=viz.SURFACE, linewidth=1.5, zorder=3,
               label="Top 2 per 0.2 m/s bin (fit points)")
    xs = np.linspace(0, prof.s0, 50)
    ax.plot(xs, prof.a0 + prof.slope * xs, color=viz.SERIES[1], zorder=2,
            label=f"Linear A-S profile (R² = {prof.r2:.2f})")
    ax.annotate(f"A0 = {prof.a0:.2f} m/s²", (0, prof.a0), xytext=(0.4, prof.a0 + 0.35),
                fontsize=9.5, color=viz.INK)
    ax.annotate(f"S0 = {prof.s0:.2f} m/s", (prof.s0, 0), xytext=(prof.s0 + 0.25, 0.25),
                fontsize=9.5, color=viz.INK)
    ax.set_xlim(0, max(prof.s0, float(speed.quantile(0.999))) + 2)
    ax.set_ylim(0, prof.a0 + 1)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(f"In-situ acceleration–speed profile — {label(player)}, one match")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.savefig(FIG / "acceleration_speed_profile.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_peak_demands(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    players = summary[(summary["minutes"] >= FULL_MATCH_MIN)].index
    # Exclude the goalkeeper (lowest distance) from a running-demand chart.
    players = [p for p in players if p != summary.loc[players, "total_distance_m"].idxmin()]
    rows = []
    for p in players:
        r = rolling_intensity(df, p, 5.0)
        first, second = df["period"] == 1, df["period"] == 2
        d1 = df.loc[first, f"{p}_speed"].fillna(0).sum() / FPS / (first.sum() / FPS / 60)
        d2 = df.loc[second, f"{p}_speed"].fillna(0).sum() / FPS / (second.sum() / FPS / 60)
        rows.append({"player": p, "avg_m_per_min": summary.loc[p, "m_per_min"],
                     "peak5_m_per_min": round(float(r.max()), 1),
                     "first_half": round(d1, 1), "second_half": round(d2, 1)})
    peaks = pd.DataFrame(rows).set_index("player").sort_values("peak5_m_per_min")

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(peaks) + 1.6))
    y = np.arange(len(peaks))
    ax.hlines(y, peaks["avg_m_per_min"], peaks["peak5_m_per_min"], color=viz.BASELINE, lw=2)
    ax.scatter(peaks["avg_m_per_min"], y, s=64, color=viz.MUTED, zorder=3,
               edgecolor=viz.SURFACE, linewidth=2, label="Match average")
    ax.scatter(peaks["peak5_m_per_min"], y, s=64, color=viz.SERIES[0], zorder=3,
               edgecolor=viz.SURFACE, linewidth=2, label="Most intense 5 minutes")
    for yi, (_, r) in zip(y, peaks.iterrows()):
        ratio = r["peak5_m_per_min"] / r["avg_m_per_min"]
        ax.text(r["peak5_m_per_min"] + 2, yi, f"×{ratio:.2f}", va="center", fontsize=9, color=viz.INK_2)
    ax.set_yticks(y, [label(p) for p in peaks.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Metres per minute")
    ax.set_title("Worst-case 5-minute running demand vs match average")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=9)
    fig.savefig(FIG / "peak_demands.png")
    plt.close(fig)
    return peaks


# --------------------------------------------------------------------------
def fig_pitch_control(home: pd.DataFrame, away: pd.DataFrame, events: pd.DataFrame) -> None:
    goal = events[(events["type"] == "SHOT") & events["subtype"].str.contains("GOAL")
                  & (events["team"] == "Home")].iloc[0]
    frame, period = int(goal["start_frame"]), int(goal["period"])
    # Home attacks +x in period 1 when it defends -x; flip for the 2nd half.
    home_sign = attacking_direction(home)
    home_defends = -home_sign if period == 1 else home_sign

    h, a = home.loc[frame], away.loc[frame]
    ball = (h["ball_x"], h["ball_y"])
    xg, yg, pc = pitch_control_surface(h, a, ball, home_defends, n_grid_x=105)

    fig, ax = plt.subplots(figsize=(10, 6.9))
    L, W = 52.5, 34
    im = ax.imshow(pc, extent=(-L, L, -W, W), origin="lower", cmap=viz.DIVERGING,
                   vmin=0, vmax=1, alpha=0.85, zorder=0, interpolation="bilinear")
    viz.draw_pitch(ax, line_color="#ffffff", lw=1.2)
    for team_df, row, color in ((home, h, viz.HOME), (away, a, viz.AWAY)):
        for p in player_ids(team_df):
            x, yv = row[f"{p}_x"], row[f"{p}_y"]
            if np.isnan(x):
                continue
            ax.arrow(x, yv, row[f"{p}_vx"], row[f"{p}_vy"], color=viz.INK, width=0.12,
                     head_width=0.9, length_includes_head=True, zorder=3, alpha=0.7)
            ax.scatter(x, yv, s=110, color=color, edgecolor="white", linewidth=1.8, zorder=4)
    ax.scatter(*ball, s=45, color="white", edgecolor=viz.INK, linewidth=1.2, zorder=5)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03)
    cbar.set_ticks([0, 0.5, 1], labels=["Away controls", "Contested", "Home controls"])
    cbar.outline.set_visible(False)
    minute = goal["start_time_s"] / 60
    ax.set_title(f"Pitch control at the moment of the home goal ({minute:.0f}′, game 2) — "
                 "arrows = player velocity over 1 s", fontsize=11)
    fig.savefig(FIG / "pitch_control_goal.png")
    plt.close(fig)


def main() -> None:
    viz.apply_style()
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    games = {g: load_game(g) for g in (1, 2)}
    home2, away2, events2 = games[2]

    summaries = []
    for g, (home, away, _) in games.items():
        for side, df in (("home", home), ("away", away)):
            s = physical_summary(df)
            s.insert(0, "game", g)
            summaries.append(s)
    summary = pd.concat(summaries)
    summary.to_csv(RES / "physical_summary.csv")

    home2_summary = summary[(summary["game"] == 2) & summary.index.str.startswith("Home")]
    fig_speed_zones(home2, home2_summary)
    peaks = fig_peak_demands(home2, home2_summary)
    peaks.to_csv(RES / "peak_demands_game2_home.csv")

    profiles = as_profiles([home2, away2])
    profiles.to_csv(RES / "acceleration_speed_profiles_game2.csv")
    best = profiles[profiles["minutes"] >= FULL_MATCH_MIN].sort_values("r2").index[-1]
    fig_as_profile(home2 if best.startswith("Home") else away2, best)

    fig_pitch_control(home2, away2, events2)
    print(summary.groupby("game")[["total_distance_m", "hsr_m", "sprint_m"]].median())
    print(profiles.describe().round(2))
    print(peaks)


if __name__ == "__main__":
    main()
