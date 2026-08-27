"""
style_profiles.py
§7 Manager Style Profile — parameterized (FPL Projection Model v4.0).

Four selectable presets, same dials v3.3 described narratively for one
manager. "Calculated Maverick" is the default and matches v3.3's §7 exactly
— selecting a different profile changes recommendations, never the
underlying xPts numbers themselves (Standing Rule #23: every output must
disclose which profile was active).
"""
from __future__ import annotations
import pandas as pd

PROFILES = {
    "Calculated Maverick": {
        "description": "Mild pull toward differentials when the xPts gap is small. Default.",
        "hit_cost_threshold": 6.0,
        "differential_ceiling": "top15",   # a low-EO pick must clear the top-15 pool average
        "captain_tiebreak": "mixed_case_by_case",
        "eo_pull": "mild_low_eo",
    },
    "Template Hugger / Rank Protector": {
        "description": "Pulls toward high-EO picks when xPts is close. Prioritises rank safety.",
        "hit_cost_threshold": 8.0,
        "differential_ceiling": None,
        "captain_tiebreak": "highest_eo",
        "eo_pull": "strong_high_eo",
    },
    "Aggressive Differential Hunter": {
        "description": "Strong pull toward low-EO picks even when xPts is roughly level.",
        "hit_cost_threshold": 4.5,
        "differential_ceiling": "top8",    # a low-EO pick must clear the top-8 pool average
        "captain_tiebreak": "lowest_eo_in_shortlist",
        "eo_pull": "strong_low_eo",
    },
    "Balanced / Pure xPts": {
        "description": "No EO weighting at all — ranks strictly by projection.",
        "hit_cost_threshold": 6.0,
        "differential_ceiling": None,
        "captain_tiebreak": "highest_raw_xpts",
        "eo_pull": "none",
    },
}

DEFAULT_PROFILE = "Calculated Maverick"


def get_profile(name: str) -> dict:
    return PROFILES.get(name, PROFILES[DEFAULT_PROFILE])


def differential_floor(pool: pd.DataFrame, xpts_col: str, ceiling_key: str | None) -> float:
    """The pool-average floor a low-owned pick must clear on merit before a
    profile's EO pull is allowed to favour it — never a relaxation of the
    xPts standard, per the v4.0 doc's explicit caveat."""
    if ceiling_key is None or pool.empty:
        return float("-inf")
    n = 15 if ceiling_key == "top15" else 8
    top_n = pool.sort_values(xpts_col, ascending=False).head(n)
    return top_n[xpts_col].mean() if not top_n.empty else float("-inf")


def captaincy_pick(cap_result: pd.DataFrame, profile_name: str) -> pd.Series:
    """cap_result comes from fpl_engine.captaincy_protocol() — already has
    `shortlisted`, `eo`, `eo_tier` columns, sorted by xpts_this_gw desc.
    Applies the profile's tie-break (Step 8 addendum) among shortlisted
    (statistically-tied, within the ~1.0 xPts window) candidates."""
    profile = get_profile(profile_name)
    shortlist = cap_result[cap_result["shortlisted"]]
    if shortlist.empty:
        shortlist = cap_result.head(1)

    rule = profile["captain_tiebreak"]
    if rule == "highest_eo":
        return shortlist.sort_values("eo", ascending=False).iloc[0]
    if rule == "lowest_eo_in_shortlist":
        return shortlist.sort_values("eo", ascending=True).iloc[0]
    if rule == "highest_raw_xpts":
        return shortlist.sort_values("xpts_this_gw", ascending=False).iloc[0]
    # "mixed_case_by_case" (Calculated Maverick default) — no single mechanical
    # rule per the v4.0 doc; fall back to highest raw xPts as the objective
    # anchor, flagged in the UI as a case-by-case call rather than a solver verdict.
    return shortlist.sort_values("xpts_this_gw", ascending=False).iloc[0]
