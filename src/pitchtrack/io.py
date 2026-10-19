"""Readers for Metrica Sports open tracking data (sample games 1 and 2).

Metrica coordinates are normalised to [0, 1] with the origin at the top-left
corner. We convert to metres with the origin at the centre spot, x pointing
towards the right-hand goal and y pointing up, on a 105 x 68 m pitch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FPS = 25

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "metrica" / "data"


def _game_dir(game: int, data_dir: Path) -> Path:
    return data_dir / f"Sample_Game_{game}"


def read_tracking(game: int, team: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return wide tracking data for one team, in metres.

    Columns: ``period``, ``frame``, ``time``, then ``<Team>_<n>_x`` / ``_y``
    for each player and ``ball_x`` / ``ball_y``. Index is the frame number.
    """
    team = team.capitalize()
    path = _game_dir(game, data_dir) / f"Sample_Game_{game}_RawTrackingData_{team}_Team.csv"
    header = pd.read_csv(path, nrows=3, header=None)
    jerseys = header.iloc[1]

    columns = ["period", "frame", "time"]
    for i in range(3, len(jerseys) - 2, 2):
        n = int(float(jerseys[i]))
        columns += [f"{team}_{n}_x", f"{team}_{n}_y"]
    columns += ["ball_x", "ball_y"]

    df = pd.read_csv(path, skiprows=3, header=None, names=columns)
    df = df.set_index("frame", drop=False)
    return to_metric(df)


def read_events(game: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = _game_dir(game, data_dir) / f"Sample_Game_{game}_RawEventsData.csv"
    ev = pd.read_csv(path)
    ev.columns = [c.lower().replace(" ", "_").replace("[s]", "s").strip("_") for c in ev.columns]
    for axis, length, sign in (("x", PITCH_LENGTH, 1), ("y", PITCH_WIDTH, -1)):
        for col in (f"start_{axis}", f"end_{axis}"):
            ev[col] = sign * (ev[col] - 0.5) * length
    return ev


def to_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Metrica unit coordinates to centred metres (in place copy)."""
    df = df.copy()
    xs = [c for c in df.columns if c.endswith("_x")]
    ys = [c for c in df.columns if c.endswith("_y")]
    df[xs] = (df[xs] - 0.5) * PITCH_LENGTH
    df[ys] = -(df[ys] - 0.5) * PITCH_WIDTH
    return df


def to_single_direction(home: pd.DataFrame, away: pd.DataFrame,
                        events: pd.DataFrame | None = None):
    """Flip second-half coordinates so each team attacks the same way all game."""
    out = []
    for df in (home, away):
        df = df.copy()
        cols = [c for c in df.columns if c.endswith(("_x", "_y"))]
        second = df["period"] == 2
        df.loc[second, cols] *= -1
        out.append(df)
    if events is not None:
        events = events.copy()
        cols = ["start_x", "start_y", "end_x", "end_y"]
        events.loc[events["period"] == 2, cols] *= -1
        out.append(events)
    return tuple(out)


def player_ids(df: pd.DataFrame) -> list[str]:
    """Players in a wide tracking frame, e.g. ``['Home_1', 'Home_2', ...]``."""
    return sorted({c[:-2] for c in df.columns if c.endswith("_x") and not c.startswith("ball")},
                  key=lambda s: int(s.split("_")[1]))


def attacking_direction(team_df: pd.DataFrame) -> int:
    """+1 if the team attacks towards positive x in period 1, else -1.

    At kick-off every player stands in his own half, so the sign of the mean
    x position tells which goal the team defends.
    """
    first = team_df.iloc[0]
    xs = np.array([first[f"{p}_x"] for p in player_ids(team_df)], dtype=float)
    return -1 if np.nanmean(xs) > 0 else 1
