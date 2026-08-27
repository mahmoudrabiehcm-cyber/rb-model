"""
setpiece.py
Step 3c — Set-Piece Role Signal (FPL Projection Model v4.0).

Sits alongside Step 3's Decay Schedule, not inside it: a role-confirmation
override, same category as Step 4's xM Floor Rule.

Practical implementation note (read before changing the numbers below):
the official API only exposes each player's *current* penalty/corner/
free-kick order, not a history of when that order last changed. A
Streamlit Community Cloud deploy also has no database to remember last
week's order between runs. So "newly confirmed" is approximated instead
of tracked: the multiplier is scaled by how much of the Decay Schedule's
weight is still on the historical leg (`1 - current_weight`). Early in a
role change that's ~1.0 (full multiplier); as real current-season minutes
accrue and the schedule shifts weight onto the current leg, the multiplier
fades toward 1.00 on its own — the same decay-out behaviour the v4.0 doc
specifies, just driven by the existing schedule instead of a remembered
"first seen" timestamp. If you later add persistent storage (a small
committed CSV of "role last changed GW", updated by hand each week), swap
this for exact confirmation-week tracking — see README.
"""
from __future__ import annotations
import pandas as pd

import fpl_engine as eng


def setpiece_multiplier(player_row: pd.Series, gw: int, cfg: dict) -> float:
    sp_cfg = cfg.get("setpiece_signal", {})
    if not sp_cfg.get("enabled", True):
        return 1.0

    pen_order = player_row.get("penalties_order")
    corner_order = player_row.get("corners_and_indirect_freekicks_order")
    fk_order = player_row.get("direct_freekicks_order")

    is_primary_pen = pd.notna(pen_order) and int(pen_order) == 1
    is_primary_dead_ball = (pd.notna(corner_order) and int(corner_order) == 1) or \
                            (pd.notna(fk_order) and int(fk_order) == 1)

    if not is_primary_pen and not is_primary_dead_ball:
        return 1.0

    h_w, c_w = eng.decay_weights(gw, cfg)  # current_weight rises across the season
    fade = max(0.0, 1.0 - c_w)  # ~1.0 early, ~0.15 by GW13+ — see module docstring

    if is_primary_pen:
        band = sp_cfg.get("penalty_multiplier_max", 1.20)
    else:
        band = sp_cfg.get("dead_ball_multiplier_max", 1.10)

    return 1.0 + (band - 1.0) * fade


def apply_to_npxg(npxg_blend: float, player_row: pd.Series, gw: int, cfg: dict) -> tuple[float, float]:
    """Returns (adjusted_npxg_blend, multiplier_applied) — call this after
    fpl_engine.blend_rate() for npxG and before fpl_engine.compute_player_gw_xpts()."""
    if npxg_blend is None or pd.isna(npxg_blend):
        return npxg_blend, 1.0
    mult = setpiece_multiplier(player_row, gw, cfg)
    return npxg_blend * mult, mult
