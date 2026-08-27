"""
fpl_engine.py
Core xPts calculation engine — implements "FPL Projection Model v3.3"'s
Core Formula, Decay Schedule, DEFCON calibration, CS% Poisson model,
Team Rating %, Transfer Path Optimization, and Captaincy Protocol.

    xPts = xM * [ 2 + npxG*GoalPts + xA*3 + CS%*CSPts + P_defcon*2 + xBonus ] - DiscCost

All tunable weights live in model_config.yaml — this file contains the
mechanism, not the numbers, so a model-logic change is a config edit, not a
code change.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

CFG_PATH = Path(__file__).parent / "model_config.yaml"
OVERRIDES_PATH = Path(__file__).parent / "manual_overrides.csv"


def load_config(path: Path = CFG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_overrides(path: Path = OVERRIDES_PATH) -> pd.DataFrame:
    """Human/Claude-supplied qualitative layer: xM floor confirmations, CS%
    tier-2/3 pastes, BPS profile tags, Manager Tenure Split discounts, and
    Step 2b External Data Rescue rates for tiny-sample players.
    Columns: player_code, xm_override, cs_pct_override, bps_profile,
             tenure_discount, npxg90_rescue, xa90_rescue, dc90_rescue, note
    Missing file -> empty frame (engine falls back to automatic values)."""
    cols = ["player_code", "xm_override", "cs_pct_override", "bps_profile",
            "tenure_discount", "npxg90_rescue", "xa90_rescue", "dc90_rescue", "note"]
    if path.exists():
        df = pd.read_csv(path)
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Decay Schedule (§5)
# ---------------------------------------------------------------------------
def decay_weights(gw: int, cfg: dict) -> tuple[float, float]:
    for row in cfg["decay_schedule"]:
        if row["gw_from"] <= gw <= row["gw_to"]:
            return row["historical"], row["current"]
    last = cfg["decay_schedule"][-1]
    return last["historical"], last["current"]


def blend_rate(historical: float, current: float, gw: int, cfg: dict,
               current_sample_matches: int = 0) -> float:
    """Blend a per-90 output rate (npxG/90, xA/90, DEFCON/90) per the Decay
    Schedule. If there's literally no current-season sample yet, current
    weight collapses to the historical leg regardless of schedule (nothing
    to blend)."""
    h_w, c_w = decay_weights(gw, cfg)
    if current_sample_matches == 0 or pd.isna(current):
        return historical if not pd.isna(historical) else 0.0
    historical = 0.0 if pd.isna(historical) else historical
    return h_w * historical + c_w * current


# ---------------------------------------------------------------------------
# DEFCON probability — calibrated curve (v2.4)
# ---------------------------------------------------------------------------
def defcon_probability(dc90: float, position: str, cfg: dict) -> float:
    if position == "FWD" or dc90 is None or pd.isna(dc90):
        return 0.0
    key = "DEF" if position in ("GK", "DEF") else "MID_FWD"
    table = cfg["defcon_calibration"][key]
    points = [(row["dc90"], row["p"]) for row in table]
    if dc90 <= points[0][0]:
        return points[0][1]
    if dc90 >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= dc90 <= x1:
            if x1 == x0:
                return y0
            frac = (dc90 - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return points[-1][1]


# ---------------------------------------------------------------------------
# xM estimation (Step 4 proxy) — overridable via manual_overrides.csv
# ---------------------------------------------------------------------------
def estimate_xm(row: pd.Series, cfg: dict, override: Optional[float]) -> float:
    if override is not None and not pd.isna(override):
        return float(override)

    heur = cfg["xm_heuristic"]
    status = str(row.get("status", "a"))
    if status in heur["unavailable_statuses"]:
        return 0.0

    starts = row.get("starts", 0) or 0
    minutes = row.get("minutes", 0) or 0
    starts_per_90 = row.get("starts_per_90", None)

    # Confirmed current-season start this season -> xM Floor Rule
    if starts and starts >= 1:
        base = heur["confirmed_current_season_start_floor"]
    elif starts_per_90 not in (None,) and not pd.isna(starts_per_90) and starts_per_90 > 0:
        base = min(heur["max_xm"], float(starts_per_90))
    elif minutes and minutes > 0:
        # crude participation proxy from historical minutes vs a full season
        base = min(heur["max_xm"], minutes / (38 * 90))
    else:
        base = 0.15  # unproven / fringe squad player, conservative default

    if heur.get("doubtful_status_multiplier_applies", True):
        cop = row.get("chance_of_playing_next_round", None)
        if cop is not None and not pd.isna(cop):
            base *= float(cop) / 100.0

    # Step 4a -- Manager Tenure Split Check. `tenure_discount` (0.40-1.00,
    # scaled by red-flag count per the v3.3 doc) is a judgment call -- a
    # researched read on whether this player's minutes are safe under the
    # current manager -- so it can only ever arrive by hand via
    # manual_overrides.csv, the same mechanism as xm_override/cs_pct_override.
    # NOTE: this column was previously loaded by load_overrides() and merged
    # into every player row, but never actually multiplied into xM anywhere
    # -- a dead column. Fixed here. Clamped to the documented 0.40-1.00 range
    # so a stray typo in the CSV can't zero out or inflate a player's xM.
    tenure_discount = row.get("tenure_discount", None)
    if tenure_discount is not None and not pd.isna(tenure_discount):
        td = max(0.40, min(1.00, float(tenure_discount)))
        base *= td

    return min(heur["max_xm"], base)


# ---------------------------------------------------------------------------
# CS% — MODEL_POISSON tier from official team strength ratings, with
# manual override slot for a pasted tier-2/3 number (oddschecker / soccerstats)
# ---------------------------------------------------------------------------
def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def cs_pct_poisson(team_row: pd.Series, opp_row: pd.Series, is_home: bool) -> float:
    """Very compact Poisson clean-sheet model from FPL's own attack/defence
    strength ratings (0-1300ish scale). This is the free, automatic
    MODEL_POISSON tier (~ rank 5 of §3's hierarchy); paste a tier-2/3 number
    into manual_overrides.csv to promote a specific fixture."""
    if is_home:
        att = team_row.get("strength_attack_home", 1100)
        deff_opp = opp_row.get("strength_defence_away", 1100)
    else:
        att = team_row.get("strength_attack_away", 1100)
        deff_opp = opp_row.get("strength_defence_home", 1100)

    for v in (att, deff_opp):
        if v is None or pd.isna(v) or v == 0:
            return 0.30  # neutral fallback if ratings absent (e.g. GW1, no history)

    # opponent's expected goals against this team, roughly scaled around a
    # league-average ~1.35 goals/game baseline
    opp_expected_goals = 1.35 * (deff_opp / 1100.0) * (1100.0 / att) * 0.5 \
        + 1.35 * (att / deff_opp) * 0.5
    opp_expected_goals = max(0.15, min(3.5, opp_expected_goals))
    return round(_poisson_pmf(0, opp_expected_goals), 4)


# ---------------------------------------------------------------------------
# Core Formula
# ---------------------------------------------------------------------------
def compute_player_gw_xpts(player: pd.Series, position: str, cs_pct: float,
                            xm: float, npxg_blend: float, xa_blend: float,
                            dc90_blend: float, comp_discount: float,
                            cfg: dict, override_row: Optional[pd.Series]) -> dict:
    pm = cfg["position_multipliers"][position]
    p_defcon = defcon_probability(dc90_blend, position, cfg)

    xbonus_hist = player.get("bonus_per_start_hist", 0.0) or 0.0
    bps_profile = "default"
    if override_row is not None and not pd.isna(override_row.get("bps_profile", None)):
        bps_profile = override_row["bps_profile"]
    bps_mult = cfg["bps_profile_multiplier"].get(bps_profile, 1.0)
    xbonus_adj = xbonus_hist * bps_mult

    inner = (2
             + npxg_blend * pm["goal_pts"]
             + xa_blend * pm["assist_pts"]
             + cs_pct * pm["clean_sheet_pts"]
             + (p_defcon * 2 if pm["defcon_threshold"] is not None else 0.0)
             + xbonus_adj)

    xpts = xm * comp_discount * inner - cfg["disc_cost"]
    return {
        "xpts": round(max(xpts, 0.0), 3),
        "xm": round(xm, 3),
        "npxg_blend": round(npxg_blend, 3),
        "xa_blend": round(xa_blend, 3),
        "cs_pct": round(cs_pct, 3),
        "p_defcon": round(p_defcon, 3),
        "xbonus_adj": round(xbonus_adj, 3),
        "comp_discount": comp_discount,
    }


# ---------------------------------------------------------------------------
# §1a Team Rating %
# ---------------------------------------------------------------------------
def team_rating_pct(squad_xpts_total: float, ceiling_xpts_total: float,
                     tier_label: str) -> dict:
    if ceiling_xpts_total <= 0:
        return {"rating_pct": None, "tier": tier_label}
    return {
        "rating_pct": round(100 * squad_xpts_total / ceiling_xpts_total, 1),
        "tier": tier_label,
    }


# ---------------------------------------------------------------------------
# Step 7a — Transfer Path Optimization
# ---------------------------------------------------------------------------
def transfer_net_gain(xpts_in_horizon: float, xpts_out_horizon: float,
                       num_transfers: int, cfg: dict) -> dict:
    hit_cost = max(0, num_transfers - 1) * cfg["transfer"]["hit_cost_per_transfer"]
    net = xpts_in_horizon - xpts_out_horizon - hit_cost
    justified = net >= cfg["transfer"]["hit_justification_threshold"] if hit_cost > 0 else True
    return {"net_gain": round(net, 2), "hit_cost": hit_cost,
            "hit_justified": justified}


# ---------------------------------------------------------------------------
# Step 8 — Captaincy Protocol
# ---------------------------------------------------------------------------
def captaincy_protocol(candidates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """candidates needs columns: web_name, xpts_this_gw, selected_by_percent
    (used as an ownership proxy for EO; true EO = ownership% + captaincy%,
    supply a `captaincy_pct` column if you have better data)."""
    c = candidates.copy()
    top = c["xpts_this_gw"].max()
    window = cfg["captaincy"]["shortlist_xpts_window"]
    c["shortlisted"] = c["xpts_this_gw"] >= (top - window)
    if "captaincy_pct" in c.columns:
        c["eo"] = c["selected_by_percent"].astype(float) + c["captaincy_pct"].astype(float)
    else:
        c["eo"] = c["selected_by_percent"].astype(float)
    tiers = cfg["captaincy"]["eo_tiers"]

    def tier(eo):
        if eo < tiers["contrarian_max"]:
            return "contrarian"
        if eo < tiers["mixed_max"]:
            return "mixed"
        return "rank-neutral"

    c["eo_tier"] = c["eo"].apply(tier)
    return c.sort_values("xpts_this_gw", ascending=False)
