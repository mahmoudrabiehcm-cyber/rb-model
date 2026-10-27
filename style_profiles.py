"""
style_profiles.py
§7 Manager Style Profile — parameterized (FPL Projection Model v4.0).

Four selectable presets, same dials v3.3 described narratively for one
manager. "Calculated Maverick" is the default and matches v3.3's §7 exactly
— selecting a different profile changes recommendations, never the
underlying xPts numbers themselves (M103; authority Standing Rule #31 and
#54(b), Step 8: every output must disclose which profile was active).
"""
from __future__ import annotations
import pandas as pd

PROFILES = {
    "Calculated Maverick": {
        "differential_slots_max": 3,   # R2 Patch 23 (M106): ceiling on differentials among outfield starters, never a quota
        "description": "Mild pull toward differentials when the xPts gap is small. Default.",
        "hit_cost_threshold": 6.0,
        "differential_ceiling": "top15",   # a low-EO pick must clear the top-15 pool average
        "captain_tiebreak": "mixed_case_by_case",
        "eo_pull": "mild_low_eo",
        "fade_template": True,   # R2 Patch 23 (M109): swaps only fade a template (>25 %) pick, on a tie, no dearer
    },
    "Template Hugger / Rank Protector": {
        "differential_slots_max": 1,   # R2 Patch 23 (M106): ceiling on differentials among outfield starters, never a quota
        "description": "Pulls toward high-EO picks when xPts is close. Prioritises rank safety.",
        "hit_cost_threshold": 8.0,
        "differential_ceiling": None,
        "captain_tiebreak": "highest_eo",
        "eo_pull": "strong_high_eo",
    },
    "Aggressive Differential Hunter": {
        "differential_slots_max": None,   # R2 Patch 23 (M106): ceiling on differentials among outfield starters, never a quota
        "description": "Strong pull toward low-EO picks even when xPts is roughly level.",
        "hit_cost_threshold": 4.5,
        "differential_ceiling": "top8",    # a low-EO pick must clear the top-8 pool average
        "captain_tiebreak": "lowest_eo_in_shortlist",
        "eo_pull": "strong_low_eo",
    },
    "Balanced / Pure xPts": {
        "differential_slots_max": None,   # R2 Patch 23 (M106): ceiling on differentials among outfield starters, never a quota
        "description": "No EO weighting at all — ranks strictly by projection.",
        "hit_cost_threshold": 6.0,
        "differential_ceiling": None,
        "captain_tiebreak": "highest_raw_xpts",
        "eo_pull": "none",
    },
}

DEFAULT_PROFILE = "Calculated Maverick"


def get_profile(name: str) -> dict:
    key = name if name in PROFILES else DEFAULT_PROFILE
    return {**PROFILES[key], "_name": key}


def differential_floor(pool: pd.DataFrame, xpts_col: str, ceiling_key: str | None) -> float:
    """The pool-average floor a low-owned pick must clear on merit before a
    profile's EO pull is allowed to favour it — never a relaxation of the
    xPts standard, per the v4.0 doc's explicit caveat."""
    if ceiling_key is None or pool.empty:
        return float("-inf")
    n = 15 if ceiling_key == "top15" else 8
    top_n = pool.sort_values(xpts_col, ascending=False).head(n)
    return top_n[xpts_col].mean() if not top_n.empty else float("-inf")


def captaincy_pick(cap_result: pd.DataFrame, profile_name: str, team_table=None,
                    cfg: dict | None = None) -> pd.Series:
    """cap_result comes from fpl_engine.captaincy_protocol() — already has
    `shortlisted`, `eo`, `eo_tier` columns, sorted by xpts_this_gw desc.
    Applies the profile's tie-break (Step 8 addendum) among shortlisted
    (statistically-tied, within the ~1.0 xPts window) candidates.

    Patch 27 (v6.3 / Standing Rule #40) — before the profile's own
    tiebreak, the shortlist is first narrowed by
    fpl_engine.team_stability_tiebreak() whenever the candidates' teams
    show a genuinely disclosed-threshold gap in results reliability
    (league position / close-margin-result share — see that function's
    docstring for why this is an EST-tagged proxy, not literal comeback
    detection). This never overrides a clear formula leader — it only
    ever narrows an ALREADY-TIED shortlist — and silently no-ops
    (falls through unchanged) when `team_table`/`cfg` aren't supplied,
    so existing callers are unaffected until they opt in."""
    profile = get_profile(profile_name)
    shortlist = cap_result[cap_result["shortlisted"]]
    if shortlist.empty:
        shortlist = cap_result.head(1)

    stability_note = None
    if team_table is not None and cfg is not None and len(shortlist) > 1:
        import fpl_engine as eng
        ts = eng.team_stability_tiebreak(shortlist, team_table, cfg)
        if ts["applied"] and not ts["narrowed"].empty:
            shortlist = ts["narrowed"]
            stability_note = ts["reason"]

    style_note = None
    rule = profile["captain_tiebreak"]
    if rule == "highest_eo":
        picked = shortlist.sort_values("eo", ascending=False).iloc[0]
    elif rule == "lowest_eo_in_shortlist":
        picked = shortlist.sort_values("eo", ascending=True).iloc[0]
    elif rule == "highest_raw_xpts":
        picked = shortlist.sort_values("xpts_this_gw", ascending=False).iloc[0]
    else:
        # "mixed_case_by_case" (Calculated Maverick default) — no single mechanical
        # rule per the v4.0 doc; fall back to highest raw xPts as the objective
        # anchor, flagged in the UI as a case-by-case call rather than a solver verdict.
        picked = shortlist.sort_values("xpts_this_gw", ascending=False).iloc[0]
        # R2 Patch 23 (M107): Maverick may captain a differential (raw ownership under the differential band) only when it is
        # within `maverick_captain_gap` of the best captain; the gap, the tier and the cost are printed.
        rules = style_rules(cfg)
        if rules["enabled"] and len(shortlist) > 1:
            d_below, _t = ownership_bands(cfg)
            raw = shortlist["selected_by_percent"] if "selected_by_percent" in shortlist.columns else shortlist["eo"]
            top_x = float(shortlist["xpts_this_gw"].max())
            is_diff = pd.to_numeric(raw, errors="coerce") < d_below
            ok = shortlist[is_diff & (shortlist["xpts_this_gw"] >= top_x - float(rules["maverick_captain_gap"]) - 1e-9)]
            if not ok.empty and not bool(pd.to_numeric(picked.get("selected_by_percent", picked.get("eo")), errors="coerce") < d_below):
                cand = ok.sort_values(["xpts_this_gw", "eo"], ascending=[False, True]).iloc[0]
                gap = top_x - float(cand["xpts_this_gw"])
                style_note = (f"Maverick differential captain: {cand['web_name']} is {gap:.2f} xPts behind {picked['web_name']} "
                              f"(limit {float(rules['maverick_captain_gap']):.1f}); raw ownership {float(cand.get('selected_by_percent', cand['eo'])):.1f}%, "
                              f"tier {cand.get('eo_tier', 'n/a')}; cost: {gap:.2f} xPts of expected points given up.")
                lead = picked
                picked = cand
    picked = picked.copy()
    picked["team_stability_note"] = stability_note
    try:
        lead_name = cap_result.sort_values("xpts_this_gw", ascending=False).iloc[0]["web_name"]
        if style_note is None and picked["web_name"] != lead_name:
            gap = float(cap_result["xpts_this_gw"].max()) - float(picked["xpts_this_gw"])
            style_note = (f"{profile_name} picked {picked['web_name']} over the formula leader {lead_name} "
                          f"(gap {gap:.2f} xPts, inside the shortlist window).")
    except Exception:
        pass
    picked["style_note"] = style_note
    return picked


def captain_alt_pick(cap_result: pd.DataFrame, primary_web_name: str, profile_name: str):
    """The captaincy panel's second slot — was hardcoded to always show the
    lowest-EO shortlisted player regardless of style profile, which made no
    sense for "Template Hugger / Rank Protector" (whose whole point is
    avoiding risk, not surfacing one). Now follows the same eo_pull dial as
    everything else in this file: low-EO profiles get a genuine differential
    alternative, the high-EO profile gets its next-safest pick, and the
    no-EO-weighting profile gets its next-best pick on raw projection —
    consistent with each profile's own stated logic (M103; Standing Rule #31 / Step 8).
    Returns (row_or_None, label) — label describes what the slot actually is
    this time, since it isn't always "Differential" any more.

    Patch 3: when the shortlist has no other member (a clear standout
    week — one player genuinely clear of the field, per the Rule #34
    margin-of-error window), falls back to the `near_miss` pool
    (fpl_engine.captaincy_protocol's near-just-outside-the-window
    candidates) using the same eo_pull direction, labelled "Near miss –"
    so the UI never implies it's a genuine statistical tie."""
    shortlist = cap_result[cap_result["shortlisted"]]
    pool = shortlist[shortlist["web_name"] != primary_web_name]
    near_miss_fallback = False
    if pool.empty and "near_miss" in cap_result.columns:
        pool = cap_result[cap_result["near_miss"]]
        near_miss_fallback = True
    if pool.empty:
        return None, "Alternative"

    eo_pull = get_profile(profile_name).get("eo_pull", "mild_low_eo")
    if eo_pull == "strong_high_eo":
        row, label = pool.sort_values("eo", ascending=False).iloc[0], "Next-safest"
    elif eo_pull == "none":
        row, label = pool.sort_values("xpts_this_gw", ascending=False).iloc[0], "Next-best"
    else:
        # "mild_low_eo" / "strong_low_eo" — a genuine differential alternative
        row, label = pool.sort_values("eo", ascending=True).iloc[0], "Differential"
    if near_miss_fallback:
        label = f"Near miss – {label}"
    return row, label


# ---------------------------------------------------------------------------
# R2 Patch 23 (M103-M109) helpers
# ---------------------------------------------------------------------------
def ownership_bands(cfg: dict | None) -> tuple[float, float]:
    """(differential_below, template_above) on RAW ownership (selected_by_percent) - M104."""
    o = (cfg or {}).get("style_ownership", {}) or {}
    return float(o.get("differential_below", 10.0)), float(o.get("template_above", 25.0))


def style_rules(cfg: dict | None) -> dict:
    r = dict((cfg or {}).get("style_rules", {}) or {})
    r.setdefault("enabled", True)
    r.setdefault("maverick_fade_edge", 0.5)
    r.setdefault("maverick_captain_gap", 0.6)
    r.setdefault("style_cost_warn", 3.0)
    return r


def captain_flag_filter(cands: pd.DataFrame, pool: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str], bool]:
    """M107: a captain flagged below 100 % (status other than 'a', or a chance figure under 100) is excluded for every style.
    `cands` needs `code`; status / chance come from `pool` (by code). Returns (kept, names_removed, fell_back):
    when nobody is left the unfiltered list is returned with fell_back=True (the caller prints a warning)."""
    if cands is None or cands.empty or pool is None or pool.empty or "code" not in cands.columns:
        return cands, [], False
    cols = [c for c in ("code", "status", "chance_of_playing_next_round") if c in pool.columns]
    if "status" not in cols and "chance_of_playing_next_round" not in cols:
        return cands, [], False
    m = pool[cols].drop_duplicates("code").set_index("code")
    st = cands["code"].map(m["status"]) if "status" in m.columns else pd.Series("a", index=cands.index)
    st = st.fillna("a").astype(str).str.lower()
    ch = (pd.to_numeric(cands["code"].map(m["chance_of_playing_next_round"]), errors="coerce")
          if "chance_of_playing_next_round" in m.columns else pd.Series(float("nan"), index=cands.index))
    flagged = (st != "a") | (ch.notna() & (ch < 100))
    if not flagged.any():
        return cands, [], False
    kept = cands[~flagged]
    names = [str(x) for x in cands.loc[flagged, "web_name"]] if "web_name" in cands.columns else []
    if kept.empty:
        return cands, names, True
    return kept, names, False
