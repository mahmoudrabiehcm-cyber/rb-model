"""
data_pipeline.py
Shared player-table build + xPts computation, used by app.py. Ported from
the CLI tool's run_weekly_report.py so the Streamlit app and the CLI stay
on the exact same pipeline — this is the one place that logic lives.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import fpl_data
import fpl_engine as eng
import optimizer as opt
import setpiece


def build_player_table(cfg: dict, snap: fpl_data.FplSnapshot, hist_df: pd.DataFrame,
                        overrides: pd.DataFrame) -> pd.DataFrame:
    df = snap.players.copy()

    if "element_type" in df.columns:
        df["position"] = df["element_type"].map(eng.POSITION_MAP)
    else:
        df["position"] = "MID"

    # Patch 14 — Standing Rule #19 (Bench GK Verification) support: whether
    # each player started (>=60 mins) in any of the last N finished GWs
    # (see fpl_data.fetch_recent_start_ids). True/False when the signal is
    # available this run, None when it isn't (mirror-fallback path, or every
    # live/{gw}/ fetch failed) — estimate_xm() must treat None as "unknown,
    # keep the old season-total behavior," never silently treat it as False.
    if snap.recent_start_checked_gws and "id" in df.columns:
        started_ids = snap.recent_start_ids or set()
        df["recent_start"] = df["id"].isin(started_ids)
    else:
        df["recent_start"] = None

    numcols = ["expected_goals", "expected_assists", "expected_goals_per_90",
               "expected_assists_per_90", "defensive_contribution",
               "defensive_contribution_per_90", "minutes", "starts",
               "starts_per_90", "bonus", "now_cost", "selected_by_percent",
               "chance_of_playing_next_round", "total_points",
               "penalties_order", "corners_and_indirect_freekicks_order",
               "direct_freekicks_order",
               # v6.4 / Standing Rule #41's Rule #24 price-drop-flow override --
               # both already present on every bootstrap-static element, just
               # never previously selected out into the per-player projection
               # row (see compute_all's `rec` dict below).
               "transfers_in_event", "transfers_out_event"]
    for c in numcols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    games90 = (df.get("minutes", 0) / 90.0).replace(0, np.nan)
    df["npxg90_cur"] = df.get("expected_goals", np.nan) / games90
    df["xa90_cur"] = df.get("expected_assists", np.nan) / games90
    df["dc90_cur"] = df.get("defensive_contribution", np.nan) / games90
    df["bonus_per_start_cur"] = df.get("bonus", 0) / df.get("starts", np.nan).replace(0, np.nan)

    if hist_df is not None and not hist_df.empty and "code" in hist_df.columns:
        h = hist_df[["code", "expected_goals_per_90", "expected_assists_per_90",
                      "defensive_contribution_per_90", "bonus", "starts", "minutes"]].copy()
        for c in ["expected_goals_per_90", "expected_assists_per_90",
                  "defensive_contribution_per_90", "bonus", "starts", "minutes"]:
            h[c] = pd.to_numeric(h[c], errors="coerce")
        h["bonus_per_start_hist"] = h["bonus"] / h["starts"].replace(0, np.nan)
        h = h.rename(columns={
            "expected_goals_per_90": "npxg90_hist",
            "expected_assists_per_90": "xa90_hist",
            "defensive_contribution_per_90": "dc90_hist",
            "minutes": "minutes_hist",
        })[["code", "npxg90_hist", "xa90_hist", "dc90_hist", "bonus_per_start_hist", "minutes_hist"]]
        df = df.merge(h, on="code", how="left")
    else:
        df["npxg90_hist"] = np.nan
        df["xa90_hist"] = np.nan
        df["dc90_hist"] = np.nan
        df["bonus_per_start_hist"] = np.nan
        df["minutes_hist"] = np.nan

    # Standing Rule #21 / Step 2b — tiny-sample gate (the "Dowman bug" fix).
    min_min = cfg["data_quality"]["min_sample_minutes"]
    df["hist_sample_ok"] = df["minutes_hist"].fillna(0) >= min_min
    df["cur_sample_ok"] = df.get("minutes", pd.Series(0, index=df.index)).fillna(0) >= min_min
    for col in ["npxg90_hist", "xa90_hist", "dc90_hist"]:
        df.loc[~df["hist_sample_ok"], col] = np.nan
    for col in ["npxg90_cur", "xa90_cur", "dc90_cur"]:
        df.loc[~df["cur_sample_ok"], col] = np.nan
    df["est_rescue_needed"] = ~df["hist_sample_ok"] & ~df["cur_sample_ok"]

    df = df.merge(overrides, left_on="code", right_on="player_code", how="left")

    for src, dst in [("npxg90_rescue", "npxg90_hist"), ("xa90_rescue", "xa90_hist"),
                      ("dc90_rescue", "dc90_hist")]:
        if src in df.columns:
            df[src] = pd.to_numeric(df[src], errors="coerce")
            mask = df[src].notna()
            if mask.any():
                df.loc[mask, dst] = df.loc[mask, src]
                df.loc[mask, "est_rescue_needed"] = False

    return df


def price(now_cost) -> float | None:
    return round(now_cost / 10.0, 1) if pd.notna(now_cost) else None


def team_short_name(teams: pd.DataFrame, team_id) -> str:
    row = teams.loc[teams["id"] == team_id]
    return row["short_name"].iloc[0] if not row.empty and "short_name" in row.columns else "?"


def comp_discount_for_team(cfg: dict, short_name: str) -> float:
    euro = cfg["european_competition_teams"]
    disc = cfg["competition_load_discount"]
    if short_name in euro.get("champions_league", []):
        return disc["champions_league"]
    if short_name in euro.get("europa_league", []):
        return disc["europa_league"]
    if short_name in euro.get("conference_league", []):
        return disc["conference_league"]
    return disc["domestic_only"]


def get_fixture_for_gw(fixtures: pd.DataFrame, team_id: int, gw: int):
    """Returns (opp_team_id, is_home, official_difficulty) tuples —
    official_difficulty is FPL's own team_h_difficulty/team_a_difficulty
    field (1-5 scale) when the fixtures source carries it, else None (the
    caller falls back to a strength-rating-based tier — see
    `_fdr_tier_from_strength`)."""
    if fixtures.empty or "event" not in fixtures.columns:
        return []
    rows = fixtures[fixtures["event"] == gw]
    out = []
    for _, r in rows.iterrows():
        if r.get("team_h") == team_id:
            out.append((int(r["team_a"]), True, r.get("team_h_difficulty")))
        elif r.get("team_a") == team_id:
            out.append((int(r["team_h"]), False, r.get("team_a_difficulty")))
    return out


def _fdr_tier_from_official(value) -> str | None:
    """Maps FPL's official 1-5 team_h_difficulty/team_a_difficulty scale to
    the three-tier easy/mid/hard the fixture ticker (Patch 4) renders as
    dots: 1-2 -> easy, 3 -> mid, 4-5 -> hard. Returns None when the value
    is missing so the caller can fall back to strength-rating tiering."""
    if value is None or pd.isna(value):
        return None
    v = int(value)
    if v <= 2:
        return "easy"
    if v == 3:
        return "mid"
    return "hard"


def _fdr_tier_from_strength(opp_row: pd.Series, is_home: bool) -> str:
    """Fallback fixture-difficulty tier for the rare case a fixtures source
    is missing the official difficulty field entirely — reuses the same
    strength_attack_home/away ratings `fpl_engine.cs_pct_poisson()` already
    reads, so it needs no extra data source. Bands the OPPONENT's attack
    rating (the threat facing our player's defence/clean-sheet odds) around
    the raw 1000-1300ish scale's rough league-average midpoint."""
    opp_att = opp_row.get("strength_attack_away") if is_home else opp_row.get("strength_attack_home")
    if opp_att is None or pd.isna(opp_att) or opp_att == 0:
        return "mid"
    if opp_att < 1050:
        return "easy"
    if opp_att > 1200:
        return "hard"
    return "mid"


def _fdr_tier_from_official_vec(values: pd.Series) -> pd.Series:
    """Vectorized twin of _fdr_tier_from_official() -- see fpl_engine.py's
    Patch 50 section for the general pattern. NaN in `values` naturally
    fails every np.select condition, so it falls through to `default=None`
    on its own -- no separate NaN branch needed."""
    v = pd.to_numeric(values, errors="coerce")
    result = np.select([v <= 2, v == 3, v > 3], ["easy", "mid", "hard"], default=None)
    return pd.Series(result, index=values.index, dtype=object)


def _fdr_tier_from_strength_vec(opp_att: pd.Series) -> pd.Series:
    """Vectorized twin of _fdr_tier_from_strength(); `opp_att` is already
    the caller's home/away-selected opponent attack rating."""
    a = pd.to_numeric(opp_att, errors="coerce")
    result = np.select([a.isna() | (a == 0), a < 1050, a > 1200],
                        ["mid", "easy", "hard"], default="mid")
    return pd.Series(result, index=opp_att.index, dtype=object)


def compute_all(cfg: dict, snap: fpl_data.FplSnapshot, players: pd.DataFrame,
                 gw_list: list[int]) -> pd.DataFrame:
    """Same Core Formula pipeline as the CLI tool, plus Step 3c's set-piece
    multiplier applied to npxG_blend right after Step 3's decay blend.

    Patch 50 -- rewritten as a vectorized pandas/numpy pipeline (was a plain
    Python for-player / for-gw / for-fixture nested loop doing scalar
    `.loc` team lookups on every iteration -- benchmarked as 49% of a full
    page load over a 616-player/6-GW pool). Proven numerically identical to
    the original scalar loop by test_patch50_vectorized_correctness.py
    (which diffs against a stashed copy of the old implementation) and by
    re-running test_patch43_merge_correctness.py against this version.

    Approach: every player-gw-fixture combination is exploded into one long
    table (so a double gameweek is just 2 rows and a blank gameweek is a
    single all-NaN-fixture row for that player-gw), every column the old
    loop computed is derived with vectorized pandas/numpy ops across that
    whole table at once, then grouped back to one row per player with the
    usual xpts_gw{n}/opp_gw{n}/fdr_gw{n} wide columns."""
    teams_raw = snap.teams.copy()
    if "id" not in teams_raw.columns:
        teams_raw["id"] = teams_raw.index

    if "position" not in players.columns:
        return pd.DataFrame()
    pl = players[players["position"].isin(["GK", "DEF", "MID", "FWD"])].copy()
    if pl.empty:
        return pd.DataFrame()
    pl = pl.reset_index(drop=True)
    pl["_pidx"] = pl.index

    # --- per-team lookups (only ~20 teams -- a plain .apply here is cheap) ---
    teams = teams_raw.copy()
    teams["_short"] = teams.get("short_name", pd.Series("?", index=teams.index))
    teams["_short"] = teams["_short"].fillna("?")
    teams["_comp_disc"] = teams["_short"].apply(lambda s: comp_discount_for_team(cfg, s))
    for c in ("strength_attack_home", "strength_attack_away",
              "strength_defence_home", "strength_defence_away"):
        if c not in teams.columns:
            teams[c] = np.nan
    domestic_disc = cfg["competition_load_discount"]["domestic_only"]
    team_lookup = teams.drop_duplicates(subset=["id"], keep="first").set_index("id")[
        ["_short", "_comp_disc", "strength_attack_home", "strength_attack_away",
         "strength_defence_home", "strength_defence_away"]
    ]

    pl = pl.merge(
        team_lookup.rename(columns={
            "_short": "_team_short", "_comp_disc": "_comp_disc",
            "strength_attack_home": "_own_att_home", "strength_attack_away": "_own_att_away",
            "strength_defence_home": "_own_def_home", "strength_defence_away": "_own_def_away",
        }), left_on="team", right_index=True, how="left")
    pl["_team_short"] = pl["_team_short"].fillna("?")
    pl["_comp_disc"] = pl["_comp_disc"].fillna(domestic_disc)

    # --- xM (Step 4), once per player, independent of gw ---
    pl["_xm"] = eng.estimate_xm_vec(pl, cfg)

    # --- bonus (Step "p_for_bonus" override: current takes priority) ---
    bonus_cur = pd.to_numeric(pl.get("bonus_per_start_cur", pd.Series(np.nan, index=pl.index)),
                               errors="coerce")
    bonus_hist = pd.to_numeric(pl.get("bonus_per_start_hist", pd.Series(np.nan, index=pl.index)),
                                errors="coerce")
    bonus_eff = bonus_cur.where(bonus_cur.notna(), bonus_hist)
    bps_profile_col = pl.get("bps_profile", pd.Series(np.nan, index=pl.index))
    bps_mult_table = cfg["bps_profile_multiplier"]
    bps_mult = bps_profile_col.apply(lambda v: eng._bps_mult_lookup(v, bps_mult_table))
    pl["_xbonus_adj"] = bonus_eff * bps_mult

    # --- per-position multipliers (only 4 positions) ---
    pm_df = pd.DataFrame(cfg["position_multipliers"]).T
    pl = pl.merge(
        pm_df.rename(columns={"goal_pts": "_goal_pts", "assist_pts": "_assist_pts",
                               "clean_sheet_pts": "_cs_pts"})[
            ["_goal_pts", "_assist_pts", "_cs_pts", "defcon_threshold"]],
        left_on="position", right_index=True, how="left")
    pl["_defcon_applies"] = pl["defcon_threshold"].notna()

    # --- Patch 43's pinned-to-gw_list[0] set-piece badge ---
    if gw_list:
        pl["_sp_mult_ref"] = setpiece.setpiece_multiplier_vec(pl, gw_list[0], cfg)
    else:
        pl["_sp_mult_ref"] = 1.0

    # --- explode fixtures into a long (team, opp, is_home, event) table ---
    fixtures = snap.fixtures
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        fixtures_long = pd.DataFrame(columns=["team", "opp", "is_home", "official_diff",
                                               "event", "_forder"])
    else:
        f = fixtures.copy()
        f["_forder"] = np.arange(len(f))
        home = f[["event", "team_h", "team_a", "team_h_difficulty", "_forder"]].rename(
            columns={"team_h": "team", "team_a": "opp", "team_h_difficulty": "official_diff"})
        home["is_home"] = True
        away = f[["event", "team_a", "team_h", "team_a_difficulty", "_forder"]].rename(
            columns={"team_a": "team", "team_h": "opp", "team_a_difficulty": "official_diff"})
        away["is_home"] = False
        fixtures_long = pd.concat([home, away], ignore_index=True)
        # stable sort so, for a double gameweek, rows come out in the same
        # order the old code's `fixtures.iterrows()` would have visited them
        # (raw fixture-row order, home-check-before-away-check per row).
        fixtures_long = fixtures_long.sort_values("_forder", kind="mergesort")
        fixtures_long = fixtures_long[fixtures_long["event"].isin(gw_list)]

    # --- cross join: one row per player per requested gw ---
    if not gw_list:
        exploded = pd.DataFrame(columns=["_pidx", "team", "event"])
    else:
        gw_arr = np.asarray(gw_list)
        n = len(pl)
        exploded = pd.DataFrame({
            "_pidx": np.repeat(pl["_pidx"].to_numpy(), len(gw_arr)),
            "team": np.repeat(pl["team"].to_numpy(), len(gw_arr)),
            "event": np.tile(gw_arr, n),
        })

    exploded = exploded.merge(
        fixtures_long[["team", "event", "opp", "is_home", "official_diff", "_forder"]],
        on=["team", "event"], how="left")
    exploded = exploded.sort_values(["_pidx", "event", "_forder"], kind="mergesort",
                                     na_position="last").reset_index(drop=True)

    player_cols = ["_pidx", "position", "_xm", "_comp_disc", "_xbonus_adj",
                   "_goal_pts", "_assist_pts", "_cs_pts", "_defcon_applies",
                   "_own_att_home", "_own_att_away", "_own_def_home", "_own_def_away",
                   "npxg90_hist", "npxg90_cur", "xa90_hist", "xa90_cur",
                   "dc90_hist", "dc90_cur", "starts", "cs_pct_override",
                   "penalties_order", "corners_and_indirect_freekicks_order",
                   "direct_freekicks_order"]
    for c in player_cols:
        if c not in pl.columns:
            pl[c] = np.nan
    exploded = exploded.merge(pl[player_cols], on="_pidx", how="left")

    opp_lookup = team_lookup.rename(columns={
        "_short": "_opp_short",
        "strength_attack_home": "_opp_att_home", "strength_attack_away": "_opp_att_away",
        "strength_defence_home": "_opp_def_home", "strength_defence_away": "_opp_def_away",
    })[["_opp_short", "_opp_att_home", "_opp_att_away", "_opp_def_home", "_opp_def_away"]]
    exploded = exploded.merge(opp_lookup, left_on="opp", right_index=True, how="left")
    exploded["_opp_short"] = exploded["_opp_short"].fillna("?")

    is_blank = exploded["opp"].isna()
    is_home_true = exploded["is_home"] == True  # noqa: E712 (NaN -> False, correct for blanks)

    # --- CS% (Step 3 / manual override) ---
    own_att = np.where(is_home_true, exploded["_own_att_home"], exploded["_own_att_away"])
    opp_def = np.where(is_home_true, exploded["_opp_def_away"], exploded["_opp_def_home"])
    cs_pct_calc = eng.cs_pct_poisson_vec(pd.Series(own_att, index=exploded.index),
                                          pd.Series(opp_def, index=exploded.index))
    cs_override = pd.to_numeric(exploded["cs_pct_override"], errors="coerce")
    cs_pct = cs_override.where(cs_override.notna(), cs_pct_calc)

    # --- FDR tier (official difficulty, falling back to strength rating) ---
    tier_official = _fdr_tier_from_official_vec(exploded["official_diff"])
    opp_att_for_fdr = pd.Series(
        np.where(is_home_true, exploded["_opp_att_away"], exploded["_opp_att_home"]),
        index=exploded.index)
    tier_strength = _fdr_tier_from_strength_vec(opp_att_for_fdr)
    fdr_tier = pd.Series(tier_official, index=exploded.index).where(
        pd.Series(tier_official, index=exploded.index).notna(), tier_strength)
    fdr_tier = fdr_tier.where(~is_blank, "")

    # --- rate blends (Step 3, per-metric decay schedule) + set-piece bump ---
    starts_int = pd.to_numeric(exploded["starts"], errors="coerce").fillna(0)
    unique_gws = sorted(exploded["event"].dropna().unique())

    def blended(hist_col, cur_col, metric):
        result = pd.Series(np.nan, index=exploded.index, dtype=float)
        for g in unique_gws:
            m = exploded["event"] == g
            result.loc[m] = eng.blend_rate_vec(exploded.loc[m, hist_col], exploded.loc[m, cur_col],
                                                int(g), cfg, metric, starts_int.loc[m])
        return result

    npxg_raw = blended("npxg90_hist", "npxg90_cur", "npxg")
    npxg_adj, _ = setpiece.apply_to_npxg_vec(npxg_raw, exploded, exploded["event"], cfg)
    xa_blend = blended("xa90_hist", "xa90_cur", "xa")
    dc_blend = blended("dc90_hist", "dc90_cur", "dc")

    p_defcon = eng.defcon_probability_vec(dc_blend, exploded["position"], cfg)
    defcon_add = np.where(exploded["_defcon_applies"], p_defcon * 2.0, 0.0)

    inner = (2.0
             + npxg_adj * exploded["_goal_pts"].astype(float)
             + xa_blend * exploded["_assist_pts"].astype(float)
             + cs_pct * exploded["_cs_pts"].astype(float)
             + defcon_add
             + exploded["_xbonus_adj"].astype(float))

    xpts_raw = exploded["_xm"].astype(float) * exploded["_comp_disc"].astype(float) * inner \
        - cfg["disc_cost"]
    xpts_fixture = eng._pyround(pd.Series(np.maximum(xpts_raw, 0.0), index=exploded.index), 3)
    xpts_fixture = xpts_fixture.where(~is_blank, 0.0)

    opp_label = exploded["_opp_short"].astype(str) + " (" + np.where(is_home_true, "H", "A") + ")"
    opp_label = pd.Series(opp_label, index=exploded.index).where(~is_blank, "")

    exploded["_xpts_fixture"] = xpts_fixture
    exploded["_opp_label"] = opp_label
    exploded["_fdr_rank"] = fdr_tier.map({"easy": 0, "mid": 1, "hard": 2})

    grp = exploded.groupby(["_pidx", "event"], sort=False)
    # skipna=False: a NaN per-fixture xpts (e.g. a player with no bonus data
    # at all, historical or current -- see xbonus_adj above) must propagate
    # through the sum exactly like the scalar loop's `gw_total += res["xpts"]`
    # (plain float addition, which never silently drops a NaN to 0).
    grouped_xpts = eng._pyround(grp["_xpts_fixture"].sum(skipna=False), 3)
    grouped_opp = grp["_opp_label"].agg(" / ".join)
    grouped_rank = grp["_fdr_rank"].max()
    rank_to_tier = {0: "easy", 1: "mid", 2: "hard"}
    grouped_fdr = grouped_rank.map(lambda r: rank_to_tier[r] if pd.notna(r) else "")

    if gw_list:
        xpts_wide = grouped_xpts.unstack("event").reindex(index=pl["_pidx"], columns=gw_list)
        opp_wide = grouped_opp.unstack("event").reindex(index=pl["_pidx"], columns=gw_list)
        fdr_wide = grouped_fdr.unstack("event").reindex(index=pl["_pidx"], columns=gw_list)
    else:
        xpts_wide = pd.DataFrame(index=pl["_pidx"])
        opp_wide = pd.DataFrame(index=pl["_pidx"])
        fdr_wide = pd.DataFrame(index=pl["_pidx"])

    now_cost = pd.to_numeric(pl["now_cost"], errors="coerce") if "now_cost" in pl.columns \
        else pd.Series(np.nan, index=pl.index)
    price_vals = eng._pyround(now_cost / 10.0, 1)
    price_vals = price_vals.where(now_cost.notna(), np.nan)

    tin = pd.to_numeric(pl.get("transfers_in_event", pd.Series(np.nan, index=pl.index)),
                         errors="coerce")
    tout = pd.to_numeric(pl.get("transfers_out_event", pd.Series(np.nan, index=pl.index)),
                          errors="coerce")
    either_notna = tin.notna() | tout.notna()
    net_transfers = pd.Series(np.where(either_notna, tin.fillna(0) - tout.fillna(0), np.nan),
                               index=pl.index)

    out = pd.DataFrame({
        "code": pl["code"].values if "code" in pl.columns else None,
        "id": pl["id"].values if "id" in pl.columns else None,
        "web_name": pl["web_name"].values if "web_name" in pl.columns else None,
        "team": pl["_team_short"].values,
        "team_id": pl["team"].values,
        "position": pl["position"].values,
        "price": price_vals.values,
        "status": pl["status"].values if "status" in pl.columns else None,
        "news": pl["news"].values if "news" in pl.columns else None,
        "selected_by_percent": pl["selected_by_percent"].values if "selected_by_percent" in pl.columns else None,
        "transfers_in_event": pl.get("transfers_in_event", pd.Series(np.nan, index=pl.index)).values,
        "transfers_out_event": pl.get("transfers_out_event", pd.Series(np.nan, index=pl.index)).values,
        "net_transfers_event": net_transfers.values,
        "xm": eng._pyround(pl["_xm"], 3).values,
        "est_rescue_needed": pl.get("est_rescue_needed", pd.Series(False, index=pl.index)).fillna(False).astype(bool).values,
        "setpiece_flag": (pl["_sp_mult_ref"] > 1.001).values,
        "setpiece_multiplier": eng._pyround(pl["_sp_mult_ref"].astype(float), 3).values,
    })

    # NOTE: xpts is NOT fillna(0.0)'d here -- a genuinely blank gameweek
    # already got its 0.0 baked in back when `xpts_fixture` was built (the
    # `.where(~is_blank, 0.0)` line above), via the single cross-joined
    # "blank" row every player-gw pair is guaranteed to have. A NaN that
    # survives to here is the OTHER case (e.g. a player with no bonus data
    # at all -- see xbonus_adj) and must stay NaN, exactly like the scalar
    # loop's rec[f"xpts_gw{gw}"] = gw_xpts[gw] does (no rescue-to-zero).
    for g in gw_list:
        out[f"xpts_gw{g}"] = xpts_wide[g].to_numpy() if g in xpts_wide.columns else 0.0
        out[f"opp_gw{g}"] = opp_wide[g].fillna("").to_numpy() if g in opp_wide.columns else ""
        out[f"fdr_gw{g}"] = fdr_wide[g].fillna("").to_numpy() if g in fdr_wide.columns else ""

    if gw_list:
        # skipna=False to match Python's plain `sum(gw_xpts.values())`,
        # which propagates a NaN gw value instead of silently treating it
        # as 0 (see the per-gw NOTE above).
        horizon_sum = xpts_wide.reindex(columns=gw_list).sum(axis=1, skipna=False)
        out["xpts_horizon_sum"] = eng._pyround(horizon_sum, 3).to_numpy()
    else:
        out["xpts_horizon_sum"] = 0.0

    return out


def solve_ceiling(cfg: dict, proj: pd.DataFrame):
    """§1a's unconstrained Ceiling_xPts — the best possible £100m/2-5-5-3/
    max-3-per-club squad from the FULL pool, ignoring what you currently own
    or how many transfers you have. Kept as the "theoretical" reference
    number (Standing Rules #16/#18 disclosure) alongside the reachable
    ceiling below, which is what the Team Rating % headline now uses."""
    return opt.solve_squad(proj, cfg, budget=cfg["squad_rules"]["budget"])


def solve_reachable_ceiling(cfg: dict, proj: pd.DataFrame, current_squad_codes: list,
                             free_transfers: int):
    """The constrained counterpart to solve_ceiling(): the best squad
    actually reachable FROM the current squad using only the free transfers
    available right now (min_retain = 15 - free transfers, so up to that
    many transfers can freely change; the rest of the 15 must be kept).
    This is what makes Team Rating % answer "how close am I to the best
    *reachable* squad this week", not "how close am I to a fantasy ideal
    that assumes I own nobody and have unlimited transfers" — the latter is
    structurally near-impossible to score well on regardless of squad
    quality, which is why it read as permanently low no matter how much
    manual_overrides.csv research went into the pool.
    Same £100m budget proxy as solve_ceiling — real budget is bank + each
    player's individual sell price, which isn't tracked per-player here, so
    current price stands in for it (disclosed simplification, same standard
    as the Bench Value Rule's autosub discount in model_config.yaml)."""
    min_retain = max(0, min(15, 15 - max(0, free_transfers)))
    return opt.solve_squad(proj, cfg, budget=cfg["squad_rules"]["budget"],
                            retain_pool_codes=current_squad_codes, min_retain=min_retain)


def solve_free_hit_rebuild(cfg: dict, proj: pd.DataFrame, total_value: float, gw: int):
    """Step 8b Free Hit Evaluation (Standing Rule #25): a full 15-man
    rebuild against the manager's TOTAL team value (bank + current squad's
    sell-value proxy — see solve_reachable_ceiling's note on that proxy),
    optimized for the single target gameweek only, never the multi-GW
    horizon sum, since a Free Hit's squad reverts after that one week
    (Horizon-Matching Rule). Never a marginal swap-budget check — always the
    full rebuild, per Rule #25.

    NOTE: this is the play/hold VERDICT solve (Chip Advisor) — it maximizes
    the raw 15-man sum, same as solve_squad() always has, because that's
    all a play-vs-hold comparison needs. For the "what would the optimal
    squad actually look like" display feature, see
    solve_free_hit_optimal_squad() below, which deliberately optimizes
    differently (highest 11 starters, light bench)."""
    col = f"xpts_gw{gw}"
    if col not in proj.columns:
        return None
    return opt.solve_squad(proj, cfg, budget=total_value, objective_col=col)


def solve_free_hit_optimal_squad(cfg: dict, proj: pd.DataFrame, total_value: float, gw: int):
    """Free Hit "optimal team for this GW" display feature (2026-09-07
    discussion, Patch 19). Unlike solve_free_hit_rebuild() above (which
    answers "is playing Free Hit this GW worth it at all," maximizing the
    raw 15-man sum for that comparison), this answers "if I play it, what's
    the actual best squad" — highest-scoring legal Starting XI plus the
    cheapest legal bench, via optimizer.solve_xi_first_squad(). Same
    Horizon-Matching Rule basis (single target GW only, never a multi-GW
    sum, since a Free Hit squad reverts after one week) and same total-value
    budget basis (Rule #25) as the play/hold solve."""
    col = f"xpts_gw{gw}"
    if col not in proj.columns:
        return None
    return opt.solve_xi_first_squad(proj, cfg, budget=total_value, gw_col=col)
