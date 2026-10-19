"""Vectorised implementation of Spearman's (2018) physics-based pitch control.

Reference: W. Spearman, "Beyond Expected Goals", MIT Sloan Sports Analytics
Conference, 2018.

For every location on the pitch the model asks: if the ball were played
there now, which team would most likely control it? Each player's time to
intercept assumes he keeps his current velocity during a reaction time and
then runs straight at maximum speed. Control probability then accumulates
over time once the ball has arrived, with a logistic uncertainty on each
player's arrival time.

The whole grid is integrated at once with NumPy (no Python loop over cells).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitchtrack.io import PITCH_LENGTH, PITCH_WIDTH, player_ids


@dataclass(frozen=True)
class PitchControlParams:
    max_player_speed: float = 5.0     # m/s
    reaction_time: float = 0.7        # s
    tti_sigma: float = 0.45           # s, uncertainty on time-to-intercept
    lambda_att: float = 4.3           # 1/s, ball control rate (attackers)
    lambda_def: float = 4.3           # 1/s, ball control rate (defenders)
    lambda_gk_factor: float = 3.0     # goalkeepers catch the ball faster
    average_ball_speed: float = 15.0  # m/s
    int_dt: float = 0.04              # s, integration step
    max_int_time: float = 10.0        # s


def _team_state(frame: pd.Series, team_cols: list[str]):
    pos = np.array([[frame[f"{p}_x"], frame[f"{p}_y"]] for p in team_cols], dtype=float)
    vel = np.array([[frame.get(f"{p}_vx", 0.0), frame.get(f"{p}_vy", 0.0)] for p in team_cols],
                   dtype=float)
    vel = np.nan_to_num(vel)
    on_pitch = ~np.isnan(pos).any(axis=1)
    return pos[on_pitch], vel[on_pitch], [p for p, ok in zip(team_cols, on_pitch) if ok]


def _goalkeeper(pos: np.ndarray, defend_sign: int) -> int:
    """Index of the player closest to the goal the team defends."""
    goal = np.array([defend_sign * PITCH_LENGTH / 2, 0.0])
    return int(np.argmin(np.linalg.norm(pos - goal, axis=1)))


def time_to_intercept(pos: np.ndarray, vel: np.ndarray, targets: np.ndarray,
                      params: PitchControlParams) -> np.ndarray:
    """Shape (n_targets, n_players): seconds for each player to reach each target."""
    reaction = pos + vel * params.reaction_time                  # (P, 2)
    dist = np.linalg.norm(targets[:, None, :] - reaction[None, :, :], axis=2)
    return params.reaction_time + dist / params.max_player_speed


def pitch_control_surface(
    att_frame: pd.Series,
    def_frame: pd.Series,
    ball_xy: tuple[float, float],
    att_defends_sign: int,
    params: PitchControlParams = PitchControlParams(),
    n_grid_x: int = 50,
):
    """Return ``(xgrid, ygrid, ppcf_att)`` where ``ppcf_att[j, i]`` is the
    probability that the attacking team controls a ball played to
    ``(xgrid[i], ygrid[j])``.

    ``att_defends_sign`` is -1 if the attacking team defends the goal at
    negative x (i.e. attacks towards +x), else +1.
    """
    n_grid_y = int(round(n_grid_x * PITCH_WIDTH / PITCH_LENGTH))
    dx, dy = PITCH_LENGTH / n_grid_x, PITCH_WIDTH / n_grid_y
    xgrid = np.arange(n_grid_x) * dx - PITCH_LENGTH / 2 + dx / 2
    ygrid = np.arange(n_grid_y) * dy - PITCH_WIDTH / 2 + dy / 2
    gx, gy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([gx.ravel(), gy.ravel()])         # (T, 2)

    att_pos, att_vel, _ = _team_state(att_frame, player_ids(att_frame.to_frame().T))
    def_pos, def_vel, _ = _team_state(def_frame, player_ids(def_frame.to_frame().T))

    ppcf_att = integrate_control(targets, np.asarray(ball_xy, dtype=float),
                                 att_pos, att_vel, def_pos, def_vel,
                                 att_gk=_goalkeeper(att_pos, att_defends_sign),
                                 def_gk=_goalkeeper(def_pos, -att_defends_sign),
                                 params=params)
    return xgrid, ygrid, ppcf_att.reshape(n_grid_y, n_grid_x)


def integrate_control(targets, ball, att_pos, att_vel, def_pos, def_vel,
                      att_gk: int | None, def_gk: int | None,
                      params: PitchControlParams = PitchControlParams()) -> np.ndarray:
    """Probability (n_targets,) that the attacking team controls each target."""
    if np.isnan(ball).any():
        ball_time = np.zeros(len(targets))
    else:
        ball_time = np.linalg.norm(targets - ball, axis=1) / params.average_ball_speed

    tti_att = time_to_intercept(att_pos, att_vel, targets, params)
    tti_def = time_to_intercept(def_pos, def_vel, targets, params)

    lam_att = np.full(len(att_pos), params.lambda_att)
    lam_def = np.full(len(def_pos), params.lambda_def)
    if att_gk is not None:
        lam_att[att_gk] *= params.lambda_gk_factor
    if def_gk is not None:
        lam_def[def_gk] *= params.lambda_gk_factor

    k = np.pi / np.sqrt(3.0) / params.tti_sigma
    n_steps = int(params.max_int_time / params.int_dt)
    p_att = np.zeros_like(tti_att)          # (T, Pa) cumulative control per player
    p_def = np.zeros_like(tti_def)
    remaining = np.ones((len(targets), 1))  # probability nobody controls the ball yet
    for step in range(1, n_steps + 1):
        if remaining.max() < 1e-3:
            break
        t = ball_time[:, None] + step * params.int_dt
        # Instantaneous control rate ("hazard") of each player at time t.
        h_att = lam_att / (1.0 + np.exp(-k * (t - tti_att)))
        h_def = lam_def / (1.0 + np.exp(-k * (t - tti_def)))
        h_tot = h_att.sum(axis=1, keepdims=True) + h_def.sum(axis=1, keepdims=True)
        # Exact solution of dR/dt = -h_tot * R over one step: unlike the plain
        # Euler update, it can never hand out more than the remaining probability.
        gained = remaining * (1.0 - np.exp(-h_tot * params.int_dt))
        share = gained / np.maximum(h_tot, 1e-12)
        p_att += share * h_att
        p_def += share * h_def
        remaining = remaining - gained

    total_att, total_def = p_att.sum(axis=1), p_def.sum(axis=1)
    return total_att / np.maximum(total_att + total_def, 1e-12)
