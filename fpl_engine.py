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
# Decay Schedule (§5, v6.0 / Standing Rule #38 -- METRIC-SPECIFIC)
# ---------------------------------------------------------------------------
_DECAY_CONFIG_KEYS = {
    "npxg": "decay_schedule_npxg",
    "xa": "decay_schedule_xa",
    "dc": "decay_schedule_dc",
}


def decay_weights(gw: int, cfg: dict, metric: str) -> tuple[float, float]:
    """metric must be one of "npxg" / "xa" / "dc" (Standing Rule #38 -- each
    output rate has its own historical/current blend curve as of v6.0; there
    is no longer a single shared schedule). Raises KeyError on an unknown
    metric rather than silently falling back, so a typo/new-metric call site
    fails loudly instead of quietly reusing the wrong curve."""
    cfg_key = _DECAY_CONFIG_KEYS[metric]
    schedule = cfg[cfg_key]
    for row in schedule:
        if row["gw_from"] <= gw <= row["gw_to"]:
            return row["historical"], row["current"]
    last = schedule[-1]
    return last["historical"], last["current"]


def blend_rate(historical: float, current: float, gw: int, cfg: dict, metric: str,
               current_sample_matches: int = 0) -> float:
    """Blend a per-90 output rate (npxG/90, xA/90, or DEFCON/90 -- pass the
    matching `metric`) per that metric's own Decay Schedule curve (v6.0,
    Standing Rule #38). If there's literally no current-season sample yet,
    current weight collapses to the historical leg regardless of schedule
    (nothing to blend)."""
    h_w, c_w = decay_weights(gw, cfg, metric)
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
    recent_start = row.get("recent_start", None)  # True/False/None(signal unavailable) — Patch 14

    # Confirmed current-season start -> xM Floor Rule. Patch 14 (Standing
    # Rule #19, "Bench GK Verification" — documented but never actually
    # implemented until now): a start ANYWHERE this season used to be
    # enough to grant this floor with no recency check at all, so a keeper
    # who started once months ago (injury cover, a cup match) and has been
    # firmly benched since kept an inflated ~0.88 "basically nailed" xM
    # indefinitely — exactly the "bench GK who isn't actually his club's
    # #1" failure the model doc's Rule #19 exists to catch. Now the floor
    # additionally requires a start within the recency window (see
    # fpl_data.fetch_recent_start_ids) whenever that signal was actually
    # available this run (`recent_start is not False` — i.e. either True,
    # or None meaning the signal couldn't be fetched, in which case this
    # falls back to the pre-Patch-14 season-total behavior rather than
    # silently treating "unknown" as "not nailed"). No recent start with the
    # signal available (`recent_start is False`) falls through to the
    # starts_per_90 / minutes-based estimate below instead of the floor,
    # which will correctly come out low for a genuinely benched player —
    # this is a per-player recency correction, so it self-resolves the
    # "own two keepers from the same club" case too: whichever one has
    # actually been playing keeps the floor, the one who hasn't gets
    # correctly discounted, no special-casing needed for that pairing.
    if starts and starts >= 1 and recent_start is not False:
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
# manual override slot for a pasted tier-2/3 number (oddschecker / soccerstats /
# Spreadex / Solio Analytics — v6.1 note: Solio is EST-tier/provisional until
# independently re-confirmed a second time, per Rule #1)
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
# Standing Rule #34 — Margin-of-Error Tie Rule (v4.6)
# ---------------------------------------------------------------------------
def margin_of_error_threshold(compared_total: float, cfg: dict | None = None,
                               floor_points: float | None = None,
                               pct_of_total: float | None = None) -> float:
    """A cumulative xPts difference smaller than this must be reported as a
    statistical tie, never a clear winner/verdict — demonstrated single-
    player weekly correction noise runs 3-9 points, so anything inside this
    band is indistinguishable from that noise. threshold = max(floor_points,
    pct_of_total * the compared horizon's combined total), defaulting to
    2.0 / 0.02 (model_config.yaml's `margin_of_error` section) — the generic
    Standing Rule #34 band used for Team Rating %, transfer-path comparisons,
    and the captaincy shortlist window.

    Explicit `floor_points`/`pct_of_total` args override the config default
    for this one call — added (Patch 5) so the Chip Advisor can apply a
    chip-specific band instead of the generic one for its own play/hold
    verdicts (see model_config.yaml's `chip_advisor_thresholds`). This is a
    disclosed EXTENSION beyond Rule #34's own text, not something the model
    doc specifies per-chip itself: Rule #34 states one blanket formula.
    Widening it per chip for Bench Boost/Triple Captain/Free Hit specifically
    was a manager-directed adjustment (higher stakes / higher single-player
    variance than a routine transfer comparison), never presented as if it
    were already in the source document."""
    fp, pct = 2.0, 0.02
    if cfg is not None:
        moe_cfg = cfg.get("margin_of_error", {})
        fp = moe_cfg.get("floor_points", fp)
        pct = moe_cfg.get("pct_of_total", pct)
    if floor_points is not None:
        fp = floor_points
    if pct_of_total is not None:
        pct = pct_of_total
    return max(fp, pct * abs(compared_total))


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
    supply a `captaincy_pct` column if you have better data).

    Shortlist window reverted to the documented Step 8 value (Patch 9,
    reversing a Patch 3 error): the model doc states "any starter within
    ~1.0 xPts of the squad's highest single-GW projection — treat as
    statistically tied," a fixed, ~1.0-xPts window. Patch 3 replaced this
    with `margin_of_error_threshold(top, cfg)` (Standing Rule #34's general
    formula, floor 2.0) reasoning that a flat 1.0 band didn't scale with the
    top score — but Rule #34's floor is TWICE as wide as Step 8's own
    documented figure, and that's exactly what produced a real, confirmed
    wrong call: two candidates 2.0 xPts apart (a genuine, meaningful gap)
    were reported as a "coin-flip," when the model's own text would treat
    anything past ~1.0 xPts as clearly separated. `captaincy.
    shortlist_xpts_window` (was marked legacy/superseded) is live again. A
    `near_miss` column still flags candidates just outside the shortlist but
    within `near_miss_multiplier x` the same window — used by
    `style_profiles.captain_alt_pick()` so a genuine "clear standout" week
    still surfaces a profile-aware alternative instead of no alt slot at
    all."""
    c = candidates.copy()
    top = c["xpts_this_gw"].max()
    window = cfg.get("captaincy", {}).get("shortlist_xpts_window", 1.0)
    c["shortlisted"] = c["xpts_this_gw"] >= (top - window)
    near_miss_mult = cfg.get("captaincy", {}).get("near_miss_multiplier", 2.0)
    c["near_miss"] = (~c["shortlisted"]) & (c["xpts_this_gw"] >= (top - window * near_miss_mult))
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


# ---------------------------------------------------------------------------
# Standing Rule #40 (v6.3, Patch 27) — Team-Stability Captaincy Check
# ---------------------------------------------------------------------------
def team_league_table(fixtures: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Mechanical proxy for Rule #40's "team-level results form" signal,
    built entirely from data already in this pipeline (finished fixtures'
    final scores) rather than a new manually-researched field -- the
    manager's explicit choice over a manual_overrides.csv-style column.

    League position is computed properly (points, then goal difference,
    then goals for -- the standard PL tiebreak order) from finished
    fixtures, so that part of Rule #40 is a genuine, exact signal.

    "Controlled performances versus repeated late rescues" is the part
    this pipeline has no data to detect directly -- the official fixtures
    endpoint carries final scores only, no goal-minute data, so there is
    no mechanical way to see a stoppage-time equalizer. `close_margin_share`
    is a disclosed, EST-tagged APPROXIMATION: the share of a team's played
    matches decided by a single goal or drawn (|goal difference| <= 1).
    A high share means a team is grinding out tight results -- consistent
    with, but not proof of, the "repeated late rescue" pattern Rule #40
    describes; a low share (games settled by 2+ goals) reads as more
    genuinely "controlled." This is disclosed wherever it's shown, never
    presented as if it were literal comeback/rescue detection.

    Returns one row per team_id with: position, played, points, gf, ga,
    gd, close_margin_share. Teams with 0 played matches get position =
    NaN (not yet rankable) rather than a misleading last place."""
    cols_needed = {"team_h", "team_a", "team_h_score", "team_a_score", "finished"}
    if fixtures is None or fixtures.empty or not cols_needed.issubset(fixtures.columns):
        return pd.DataFrame(columns=["team_id", "position", "played", "points", "gf", "ga",
                                      "gd", "close_margin_share"]).set_index("team_id")

    played = fixtures[fixtures["finished"] == True].dropna(subset=["team_h_score", "team_a_score"])
    team_ids = teams["id"].tolist() if teams is not None and "id" in teams.columns else \
        pd.unique(played[["team_h", "team_a"]].values.ravel())

    rows = []
    for tid in team_ids:
        home = played[played["team_h"] == tid]
        away = played[played["team_a"] == tid]
        gf = int(home["team_h_score"].sum() + away["team_a_score"].sum())
        ga = int(home["team_a_score"].sum() + away["team_h_score"].sum())
        n = len(home) + len(away)
        wins = int((home["team_h_score"] > home["team_a_score"]).sum() +
                   (away["team_a_score"] > away["team_h_score"]).sum())
        draws = int((home["team_h_score"] == home["team_a_score"]).sum() +
                    (away["team_a_score"] == away["team_h_score"]).sum())
        points = wins * 3 + draws
        close = int((abs(home["team_h_score"] - home["team_a_score"]) <= 1).sum() +
                    (abs(away["team_a_score"] - away["team_h_score"]) <= 1).sum())
        rows.append({"team_id": tid, "played": n, "points": points, "gf": gf, "ga": ga,
                     "gd": gf - ga, "close_margin_share": (close / n) if n else None})

    table = pd.DataFrame(rows).set_index("team_id")
    # Standard PL tiebreak order (points, then goal difference, then goals
    # for) -- assigned from sorted row order directly rather than rank(),
    # since rank() on points alone wouldn't apply the gd/gf tiebreak.
    ranked = table[table["played"] > 0].sort_values(["points", "gd", "gf"], ascending=False)
    table["position"] = pd.Series(range(1, len(ranked) + 1), index=ranked.index).reindex(table.index)
    return table


def team_stability_tiebreak(shortlist: pd.DataFrame, team_table: pd.DataFrame, cfg: dict) -> dict:
    """Rule #40: among an ALREADY-TIED captaincy shortlist (Step 8's
    ~1.0-xPts window), narrow to the candidate(s) whose team is showing
    a meaningfully more "controlled" results pattern -- never a standalone
    ranking input, never touching a candidate the formula already separated.

    Only acts when the signal is a genuine, disclosed-threshold gap, not
    any nonzero difference -- `captaincy.team_stability_position_gap` and
    `captaincy.team_stability_grind_gap` in model_config.yaml are a
    manager-directed EST extension (the doc names the two signals but
    gives no numeric threshold, same disclosed-extension pattern as
    chip_advisor_thresholds). Returns {"narrowed": DataFrame, "applied":
    bool, "reason": str} -- "narrowed" is the full shortlist unchanged
    when the signal doesn't clearly separate the candidates."""
    teams_in_play = shortlist["team"].unique() if "team" in shortlist.columns else []
    if team_table is None or team_table.empty or len(teams_in_play) < 2:
        return {"narrowed": shortlist, "applied": False, "reason": "single team in shortlist or no table"}

    ts_cfg = cfg.get("captaincy", {})
    pos_gap = ts_cfg.get("team_stability_position_gap", 6)
    grind_gap = ts_cfg.get("team_stability_grind_gap", 0.25)

    rows = shortlist.copy()
    rows["_position"] = rows["team"].map(team_table["position"])
    rows["_close_margin_share"] = rows["team"].map(team_table["close_margin_share"])
    if rows["_position"].isna().any():
        return {"narrowed": shortlist, "applied": False, "reason": "a candidate's team has no played matches yet"}

    best_pos = rows["_position"].min()
    worst_pos = rows["_position"].max()
    if worst_pos - best_pos >= pos_gap:
        narrowed = rows[rows["_position"] == best_pos].drop(columns=["_position", "_close_margin_share"])
        return {"narrowed": narrowed, "applied": True,
                "reason": f"league position gap of {int(worst_pos - best_pos)} places (>= {pos_gap})"}

    best_grind = rows["_close_margin_share"].min()
    worst_grind = rows["_close_margin_share"].max()
    if (worst_grind - best_grind) >= grind_gap:
        narrowed = rows[rows["_close_margin_share"] == best_grind].drop(
            columns=["_position", "_close_margin_share"])
        return {"narrowed": narrowed, "applied": True,
                "reason": f"close-margin-result share gap of {worst_grind - best_grind:.2f} "
                          f"(>= {grind_gap:.2f}) -- fewer 1-goal/drawn results reads as more controlled"}

    return {"narrowed": shortlist, "applied": False,
            "reason": "neither signal cleared its disclosed threshold -- no genuine separation"}


def disruption_check(squad_df: pd.DataFrame, gw_list: list, planned_chip_gw: int | None) -> dict:
    """Standing Rule #41 (Disruption-Horizon Rule, v6.4) + its companion
    Rule #24 price-drop-flow override.

    Automatic disruption-check (v6.4's Step 0 clarification, app-side
    equivalent). This tool re-fetches live data every run (no cached
    assumptions carry forward -- Step 2), so "disrupted" is read straight
    off this run's own `status`/`est_rescue_needed` fields, the exact same
    definition app.py already uses to build its Wildcard-flag
    `flagged_players` set, reused here as the single source of truth so
    the two never disagree. What this function can NOT do is the doc's
    literal "against what was known last session" diff -- this app has no
    persistent memory between runs (a fresh container each time) -- so a
    genuinely NEW-this-week disruption isn't distinguished from one that's
    been known for weeks; every currently-disrupted squad player is
    surfaced every run, which is a strict superset of the doc's trigger,
    never a gap that could hide a real disruption.

    Rule #41: `planned_chip_gw` is a manager-stated "next full-rebuild
    chip" GW (there is nowhere in the live API data for this app to infer
    a still-unplayed chip's intended DATE on its own -- that stays a
    rolling re-test per Standing Rule #32, even though Patch 30 made
    Wildcard's own trigger CONDITION mechanical). When a
    disruption is found AND that GW falls inside or at the start of
    `gw_list`, the ordinary transfer net-gain horizon is capped to stop
    before it -- gains projected for weeks the chip will already have
    reset the squad are not a real reason to transfer a disrupted player
    now. If the chip lands on the very next gameweek itself, the capped
    horizon is empty; that's reported as its own note (the rebuild already
    handles it) rather than silently solving over a 0-GW horizon.

    Rule #24 override: for each disrupted player, this run's
    `net_transfers_event` (transfers_in_event - transfers_out_event, both
    pulled fresh from bootstrap-static every run) being negative -- more
    managers selling him than buying -- is a concrete, checkable
    price-drop-flow signal. The doc gives no magnitude threshold, so any
    negative net flow triggers it (disclosed EST extension, same pattern
    as chip_advisor_thresholds): Rule #24's default-to-wait is overridden
    the same way a price-RISE risk already overrides it in the other
    direction, since waiting risks a further price fall before the
    manager acts.

    Returns {"players": [{code, web_name, status, news, net_transfers_event,
    price_drop_flow: bool}, ...], "capped_gw_list": list|None, "notes":
    [str, ...]}. Empty "players"/"notes" and capped_gw_list=None when no
    current squad player is disrupted -- a silent no-op for an undisrupted
    squad, same as team_stability_tiebreak's no-op when nothing is tied."""
    empty = {"players": [], "capped_gw_list": None, "notes": []}
    if squad_df is None or squad_df.empty or "status" not in squad_df.columns:
        return empty

    rescue_col = squad_df["est_rescue_needed"] if "est_rescue_needed" in squad_df.columns \
        else pd.Series(False, index=squad_df.index)
    disrupted = squad_df[(squad_df["status"] != "a") | rescue_col].copy()
    if disrupted.empty:
        return empty

    players = []
    price_drop_names = []
    for _, r in disrupted.iterrows():
        net_flow = r.get("net_transfers_event")
        price_drop = pd.notna(net_flow) and net_flow < 0
        players.append({
            "code": r.get("code"), "web_name": r.get("web_name"), "status": r.get("status"),
            "news": r.get("news"), "net_transfers_event": net_flow, "price_drop_flow": bool(price_drop),
        })
        if price_drop:
            price_drop_names.append(r.get("web_name"))

    notes = [f"Disruption check (Rule #41 auto-trigger): {', '.join(p['web_name'] for p in players)} "
             f"currently carr{'ies' if len(players) == 1 else 'y'} a live status/data-quality flag."]

    capped_gw_list = None
    if planned_chip_gw is not None and gw_list:
        if planned_chip_gw <= gw_list[0]:
            capped_gw_list = []
            notes.append(f"A full-rebuild chip is planned for GW{planned_chip_gw} -- at or before this run's "
                          f"horizon start, so no separate transfer-vs-hold decision applies to the flagged "
                          f"player(s) this run; the rebuild already replaces them (Rule #41).")
        elif planned_chip_gw <= gw_list[-1]:
            capped_gw_list = [g for g in gw_list if g < planned_chip_gw]
            notes.append(f"Transfer net-gain horizon capped at GW{capped_gw_list[-1] if capped_gw_list else '-'} "
                          f"(was GW{gw_list[-1]}) for the flagged player(s) above -- the planned GW{planned_chip_gw} "
                          f"rebuild chip means gains projected past it aren't a real reason to move them now "
                          f"(Standing Rule #41).")

    if price_drop_names:
        notes.append(f"Price-drop-flow override (Rule #24): {', '.join(price_drop_names)} current transfers-out "
                      f"exceed transfers-in this run -- overrides the default to wait, the same way a price-rise "
                      f"risk already overrides it for an incoming target (EST-tier signal, no doc-specified "
                      f"magnitude threshold).")

    return {"players": players, "capped_gw_list": capped_gw_list, "notes": notes}


def wildcard_trigger_check(squad_df: pd.DataFrame, reachable_squad_df: pd.DataFrame,
                            detect_gw_list: list, cfg: dict) -> dict:
    """Patch 30 (2026-09-14, manager-flagged correction) — v6.4's ACTUAL
    documented Wildcard trigger, replacing the app's old ad hoc rank-decline
    +flagged-player-count heuristic entirely (that heuristic pre-dated v6.4
    and had never been updated once the doc gave Wildcard real numbers).

    Corrects a real mislabeling found in this same review: the old code
    cited "Standing Rule #24" as the reason Wildcard stays non-mechanical.
    Rule #24 is the Transfer Timing Discipline Rule (ordinary transfers,
    hold-until-deadline default) — it says nothing about Wildcard. The rule
    that actually governs Wildcard timing is Standing Rule #32 (Dynamic
    Chip Timing Rule): "a planned chip date is a working hypothesis, not a
    fixed commitment... re-test at every Step 0 review." That rule blocks
    treating a DATE as locked in — it does not block the model from
    computing whether the trigger CONDITION itself currently holds.

    v6.4's literal trigger text: "average Team Rating % across [the 3-4 GW
    detection] horizon below ~78-80%, or a cumulative xPts gap of ~15+
    points versus the bounded-ceiling optimal over the same window."
    "Bounded-ceiling optimal" = the squad actually reachable using the free
    transfers on hand (data_pipeline.solve_reachable_ceiling()) — the same
    ceiling the app's own Team Rating % header stat already compares
    against — never the fully unconstrained pool ceiling (that one measures
    something else: how far even a Wildcard's own rebuild sits from a
    fantasy-ideal squad, not whether ordinary transfers can already close
    the gap without one).

    `squad_df`/`reachable_squad_df` must both already carry `xpts_gw{n}`
    columns for every GW in `detect_gw_list` (i.e. both projected onto the
    SAME window) — same Rule #22 Systematic Application discipline as the
    Team Rating % header stat: identical calculation
    (`optimizer.rating_gw_value`) on both sides, every GW.

    Config: `wildcard_trigger.team_rating_pct_ceiling` (default 79.0, the
    doc's own "~78-80%" band's midpoint) and
    `wildcard_trigger.cumulative_gap_threshold` (default 15.0, the doc's own
    "~15+" figure) — both ARE the doc's stated numbers, not a manager-
    directed extension like chip_advisor_thresholds; the "~" in the doc's
    own text is why a single midpoint/floor value stands in for a range.

    Returns {"active": bool, "avg_rating_pct": float|None, "cumulative_gap":
    float|None, "by_gw": {gw: {"squad": v, "reachable": v, "rating_pct": v}},
    "reason": str}. "active" is a genuinely mechanical yes/no on the trigger
    CONDITION — it is still never a single-GW "play" verdict (v6.4's 8-GW
    decay-weighted build horizon means Wildcard timing stays a rolling
    re-test per Rule #32, not a one-week pick the way Free Hit gets)."""
    import optimizer as opt

    empty = {"active": False, "avg_rating_pct": None, "cumulative_gap": None, "by_gw": {},
             "reason": "insufficient data to evaluate this run"}
    if squad_df is None or squad_df.empty or reachable_squad_df is None or reachable_squad_df.empty \
            or not detect_gw_list:
        return empty

    wt_cfg = cfg.get("wildcard_trigger", {})
    rating_ceiling = wt_cfg.get("team_rating_pct_ceiling", 79.0)
    gap_threshold = wt_cfg.get("cumulative_gap_threshold", 15.0)

    by_gw = {}
    for gw in detect_gw_list:
        col = f"xpts_gw{gw}"
        if col not in squad_df.columns or col not in reachable_squad_df.columns:
            continue
        squad_val = opt.rating_gw_value(squad_df, col, cfg)["total_realized"]
        reachable_val = opt.rating_gw_value(reachable_squad_df, col, cfg)["total_realized"]
        rating = team_rating_pct(squad_val, reachable_val, "")["rating_pct"]
        by_gw[gw] = {"squad": round(squad_val, 2), "reachable": round(reachable_val, 2), "rating_pct": rating}

    if not by_gw:
        return empty

    valid_ratings = [v["rating_pct"] for v in by_gw.values() if v["rating_pct"] is not None]
    avg_rating = round(sum(valid_ratings) / len(valid_ratings), 1) if valid_ratings else None
    cumulative_gap = round(sum(v["reachable"] - v["squad"] for v in by_gw.values()), 2)

    triggers = []
    if avg_rating is not None and avg_rating < rating_ceiling:
        triggers.append(f"average Team Rating % across GW{min(by_gw)}-GW{max(by_gw)} is {avg_rating}%, "
                          f"below the {rating_ceiling:.0f}% ceiling")
    if cumulative_gap >= gap_threshold:
        triggers.append(f"cumulative gap to your bounded-ceiling optimal over that window is {cumulative_gap:.1f} "
                          f"xPts, at/above the {gap_threshold:.0f}-point threshold")

    active = bool(triggers)
    reason = ("; ".join(triggers) if triggers else
              f"average Team Rating % ({avg_rating}%) and cumulative gap ({cumulative_gap:.1f} xPts) over "
              f"GW{min(by_gw)}-GW{max(by_gw)} both stay inside the doc's noise band — no trigger this run")
    return {"active": active, "avg_rating_pct": avg_rating, "cumulative_gap": cumulative_gap,
            "by_gw": by_gw, "reason": reason}
