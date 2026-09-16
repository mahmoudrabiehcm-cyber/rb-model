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
import numpy as np
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

    # v6.0 / Standing Rule #38 (Patch 18): this fade decorates npxg_blend
    # specifically (shot-taking role confirmation), so it now follows the
    # npxG-specific curve rather than the old shared schedule -- it fades to
    # its floor faster than pre-Patch-18 (npxG's curve is now accelerated
    # relative to the old shared one). current_weight rises across the season.
    h_w, c_w = eng.decay_weights(gw, cfg, metric="npxg")
    fade = max(0.0, 1.0 - c_w)  # ~1.0 early, ~0.12 by GW13+ per the npxG curve

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


# ---------------------------------------------------------------------------
# Patch 50 -- vectorized twins, additive alongside the scalar functions above
# (see fpl_engine.py's Patch 50 section for the same pattern/rationale).
# `gw` may be a single int (the pinned-gw badge use case) or a per-row
# pd.Series of gw values (the per-fixture-per-gw npxG-adjustment use case) --
# either way the decay-weight lookup is only ever done once per DISTINCT gw
# value present, never once per row.
# ---------------------------------------------------------------------------
def setpiece_multiplier_vec(df: pd.DataFrame, gw, cfg: dict) -> pd.Series:
    sp_cfg = cfg.get("setpiece_signal", {})
    idx = df.index
    if not sp_cfg.get("enabled", True):
        return pd.Series(1.0, index=idx)

    gw_series = gw if isinstance(gw, pd.Series) else pd.Series(gw, index=idx)

    def col(name):
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
            else pd.Series(np.nan, index=idx)

    pen_order = col("penalties_order")
    corner_order = col("corners_and_indirect_freekicks_order")
    fk_order = col("direct_freekicks_order")

    is_primary_pen = pen_order.notna() & (pen_order == 1)
    is_primary_dead_ball = (corner_order.notna() & (corner_order == 1)) | \
                            (fk_order.notna() & (fk_order == 1))

    unique_gws = gw_series.dropna().unique()
    cw_dict = {g: eng.decay_weights(int(g), cfg, metric="npxg")[1] for g in unique_gws}
    c_w = gw_series.map(cw_dict)
    fade = (1.0 - c_w).clip(lower=0.0)

    pen_band = sp_cfg.get("penalty_multiplier_max", 1.20)
    db_band = sp_cfg.get("dead_ball_multiplier_max", 1.10)

    mult = pd.Series(1.0, index=idx)
    mult = mult.where(~is_primary_dead_ball, 1.0 + (db_band - 1.0) * fade)
    mult = mult.where(~is_primary_pen, 1.0 + (pen_band - 1.0) * fade)
    return mult


def apply_to_npxg_vec(npxg_blend: pd.Series, df: pd.DataFrame, gw, cfg: dict):
    """Returns (adjusted_npxg_blend, multiplier_applied), same contract as
    apply_to_npxg() but vectorized. NaN in npxg_blend passes straight
    through (nan * anything == nan), matching the scalar early-return."""
    mult = setpiece_multiplier_vec(df, gw, cfg)
    adjusted = npxg_blend * mult
    return adjusted, mult
