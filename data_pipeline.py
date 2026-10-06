"""
data_pipeline.py
Shared player-table build + xPts computation, used by app.py. Ported from
the CLI tool's run_weekly_report.py so the Streamlit app and the CLI stay
on the exact same pipeline — this is the one place that logic lives.
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

import fpl_data
import fpl_engine as eng
import optimizer as opt
import setpiece


# ---------------------------------------------------------------------------
# Patch 83 (v6.9 Standing Rule #46, Fixture-Adjusted Attack Rule) -- team-
# strength-tier fixture factor. Manager-confirmed scope (2026-09-29): build
# this on real match-level team xG (fpl_data.fetch_team_match_xg(), verified
# live before writing this), not the static official strength_attack/
# strength_defence ratings the CS% calc already uses -- those are fixed
# preseason numbers that never update in-season, so they can't capture "this
# opponent's defence has actually been shipping goals all year" the way real
# current-season match xG can. Market odds (the doc's higher tier when
# available) are an explicit fast-follow, not built here -- this session
# verified oddschecker's odds pages are genuinely live, but could not verify
# a plain-requests scrape (the same method every other source in this file
# uses) actually works from a production deploy, and the doc itself flags
# automated-fetch terms of use as unverified for that source. So every GW
# here runs at the SAME team-strength tier (never "market" for GW+1) --
# a disclosed scope decision, not the doc's literal GW+1-market/GW+2+-
# team-strength split, since there is no market leg wired in yet.
# ---------------------------------------------------------------------------

class CsDiag:
    """Diagnostics table carried in DataFrame.attrs. pandas compares attrs with == on
    concat; a bare DataFrame there raises 'truth value is ambiguous', so wrap it."""
    def __init__(self, df):
        self.df = df
    def __eq__(self, other):
        return isinstance(other, CsDiag)
    __hash__ = None
    def __len__(self):
        return len(self.df)
    def __getitem__(self, k):
        return self.df[k]
    def __getattr__(self, n):
        if n == "df" or (n.startswith("__") and n.endswith("__")):
            raise AttributeError(n)  # never delegate dunders (deepcopy/pickle would return a bare DataFrame)
        return getattr(self.df, n)

def compute_team_fixture_baselines(cfg: dict, team_match_xg: pd.DataFrame,
                                    teams_df: pd.DataFrame) -> dict:
    """Builds, per team `id` (not `code` -- see fetch_team_match_xg's
    docstring on why a code->id map is needed), a shrunk-to-league-average
    "how many xG does this team really create / concede per match this
    season" baseline pair, from real finished-match data.

    Shrinkage: baseline = (n*team_mean + k*league_avg) / (n+k), k = cfg's
    fixture_adjustment.xg_shrinkage_matches (default 6 -- a disclosed,
    open-to-revision modeling choice, not a number the doc itself specifies;
    it exists so 1-2 early-season matches don't produce a wild baseline,
    matching the doc's own "early-season baselines shrink to the league
    average" default). A team with zero finished matches this season (n=0)
    collapses fully to the league average -- exactly league-average, no
    adjustment, never a crash or a missing entry.

    Returns {"by_id": {team_id: {"xg_for": v, "xg_against": v, "n": n}},
    "league_avg_xg": v, "n_teams_with_data": k}. `league_avg_xg` is the
    single pooled mean of every recorded xg_for value (== the pooled mean of
    xg_against too, by construction, since every match's xg_for on one side
    is some other team's xg_against) -- used both as the shrinkage prior and
    as Rule #46's own-baseline-relative denominator for the opponent side."""
    fa_cfg = cfg.get("fixture_adjustment", {})
    k = fa_cfg.get("xg_shrinkage_matches", 6)
    empty = {"by_id": {}, "league_avg_xg": None, "n_teams_with_data": 0}
    if team_match_xg is None or team_match_xg.empty or teams_df is None or teams_df.empty:
        return empty
    if "code" not in teams_df.columns or "id" not in teams_df.columns:
        return empty

    league_avg = float(team_match_xg["xg_for"].mean())
    if not np.isfinite(league_avg) or league_avg <= 0:
        return empty

    grouped = team_match_xg.groupby("team_code").agg(
        xg_for_mean=("xg_for", "mean"), xg_against_mean=("xg_against", "mean"),
        n=("xg_for", "count"))

    code_to_id = teams_df.drop_duplicates(subset=["code"], keep="first").set_index("code")["id"]

    by_id = {}
    for code, row in grouped.iterrows():
        if code not in code_to_id.index:
            continue  # unmapped code (e.g. a club the current teams table doesn't carry) -- skip, not crash
        team_id = code_to_id.loc[code]
        n = float(row["n"])
        shrunk_for = (n * row["xg_for_mean"] + k * league_avg) / (n + k)
        shrunk_against = (n * row["xg_against_mean"] + k * league_avg) / (n + k)
        by_id[team_id] = {"xg_for": round(float(shrunk_for), 4),
                           "xg_against": round(float(shrunk_against), 4), "n": int(n)}
    return {"by_id": by_id, "league_avg_xg": round(league_avg, 4), "n_teams_with_data": len(by_id)}


# ---------------------------------------------------------------------------
# Patch 90 (v6.9 Standing Rule #46/#47, market-odds leg) -- confirmed via
# code read that no odds-fetching or odds-fitting function existed anywhere
# in this app before this patch, and that fixture_attack_factor_vec() below
# had no gw-distance input at all (its only caller, compute_all() further
# down this file, passes none either) -- so the team-strength tier applied
# the SAME flat strength to every horizon gameweek, contradicting Rule
# #46(e)'s "s shrinking toward 0 with distance ... beyond GW+5 ... s = 0".
# Both gaps are fixed together since they're the same doc clause. See
# test_patch90_market_odds.py for the full disclosed design rationale
# (written BEFORE this implementation, per manager instruction).
# ---------------------------------------------------------------------------
def implied_probs_from_odds(home_odds, draw_odds, away_odds) -> dict | None:
    """Decimal 1X2 odds -> overround-stripped, normalized implied
    probabilities. Rule #46(e): overround must be within [1.00, 1.10]
    (inclusive both ends -- an exactly arb-free book is fine, anything below
    1.00 signals bad/stale data since no single real bookmaker prices
    negative margin, and anything above 1.10 is an abnormally fat margin for
    an EPL 1X2 market, also treated as unusable). Returns None (never
    raises) for non-positive/missing odds or an out-of-band overround."""
    try:
        h, d, a = float(home_odds), float(draw_odds), float(away_odds)
    except (TypeError, ValueError):
        return None
    if h <= 0 or d <= 0 or a <= 0:
        return None
    inv_h, inv_d, inv_a = 1.0 / h, 1.0 / d, 1.0 / a
    overround = inv_h + inv_d + inv_a
    if not (1.00 <= overround <= 1.10):
        return None
    return {"home": inv_h / overround, "draw": inv_d / overround, "away": inv_a / overround,
            "overround": round(overround, 4)}


def fit_fixture_goals_from_probs(p_home: float, p_draw: float, p_away: float,
                                  max_mu: float = 6.0) -> tuple[float, float] | None:
    """Solves for (mu_home, mu_away), the two independent-Poisson means whose
    implied Skellam(mu_home - mu_away) win/draw/loss split best matches the
    market's (p_home, p_draw, p_away) -- Rule #46(e)'s "Poisson fit". Only
    p_home and p_draw are used as the fit's 2 residuals (p_away carries no
    extra information once the other two are fixed, since all three sum to
    1); p_away is still required as an input so a caller can never pass a
    plainly-inconsistent triple without at least being asked for it.

    This fit's ABSOLUTE total (mu_home + mu_away) is known to read low --
    see rescale_goals_to_league_level() for the required correction -- so
    only the RATIO between the two returned means should be trusted directly
    out of this function; the caller rescales the total separately.

    Returns None (never raises) if the inputs aren't valid probabilities or
    the solver fails to converge to a sane, positive pair."""
    from scipy.optimize import least_squares
    from scipy.stats import skellam

    try:
        p_home, p_draw, p_away = float(p_home), float(p_draw), float(p_away)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= p <= 1.0 for p in (p_home, p_draw, p_away)):
        return None
    if not math.isclose(p_home + p_draw + p_away, 1.0, abs_tol=0.05):
        return None
    if p_draw <= 0.0 or p_draw >= 1.0:
        return None

    def residuals(x):
        mu_h, mu_a = x
        mu_h, mu_a = max(mu_h, 1e-4), max(mu_a, 1e-4)
        fit_p_home = 1 - skellam.cdf(0, mu_h, mu_a)
        fit_p_draw = skellam.pmf(0, mu_h, mu_a)
        return [fit_p_home - p_home, fit_p_draw - p_draw]

    # Initial guess: a plausible EPL-shaped total (2.6 goals) split by the
    # market's own home-vs-away lean.
    lean = 0.5 if (p_home + p_away) == 0 else p_home / (p_home + p_away)
    x0 = [max(1e-3, 2.6 * lean), max(1e-3, 2.6 * (1 - lean))]
    try:
        result = least_squares(residuals, x0, bounds=([1e-4, 1e-4], [max_mu, max_mu]))
    except Exception:
        return None
    if not result.success:
        return None
    mu_h, mu_a = float(result.x[0]), float(result.x[1])
    if mu_h <= 0 or mu_a <= 0 or mu_h > max_mu or mu_a > max_mu:
        return None
    # Confirm the fit actually reproduces the market split within a sane
    # tolerance -- a "successful" least-squares result can still land far
    # from the target if the problem is degenerate near the boundary.
    res = residuals([mu_h, mu_a])
    if max(abs(res[0]), abs(res[1])) > 0.03:
        return None
    return (mu_h, mu_a)


def rescale_goals_to_league_level(mu_home: float, mu_away: float,
                                   league_avg_total_goals: float) -> tuple[float, float]:
    """Rule #46(e): "1X2-only conversions read low and are rescaled to the
    league xG level". Preserves the fitted home/away RATIO exactly, scales
    the total up (or down) to match `league_avg_total_goals` (the season's
    real average total goals per match, i.e. 2 * a
    compute_team_fixture_baselines() league_avg_xg). Zero-safe: if mu_away
    is 0 (a degenerate fit), all of the rescaled total is assigned to home
    rather than dividing by zero."""
    total = mu_home + mu_away
    if total <= 0:
        return (league_avg_total_goals, 0.0)
    ratio = mu_home / total
    return (ratio * league_avg_total_goals, (1 - ratio) * league_avg_total_goals)


def cross_check_goal_estimates(estimates: list[tuple[float, float]],
                                tolerance: float = 0.3) -> bool:
    """Rule #46(e): "cross-checked against a second market-derived source
    ... sources must agree within 0.3 goals". Requires at least 2 raw
    (pre-rescale) (mu_home, mu_away) fits -- a single bookmaker can never be
    cross-checked against anything, so one (or zero) estimates always fails
    this check rather than being silently accepted. Passes only if EVERY
    pair of estimates agrees within `tolerance` goals on BOTH the home and
    away side."""
    if len(estimates) < 2:
        return False
    for i in range(len(estimates)):
        for j in range(i + 1, len(estimates)):
            h_i, a_i = estimates[i]
            h_j, a_j = estimates[j]
            if abs(h_i - h_j) > tolerance or abs(a_i - a_j) > tolerance:
                return False
    return True


def is_odds_stale(fetched_at: float, now: float, max_hours: float = 48) -> bool:
    """Rule #46(e): "when every source is stale (over 48 hours), s = 0".
    Strictly-greater-than: exactly 48.0 hours old is not yet stale, matching
    the doc's "over 48 hours" wording (not "48 hours or more")."""
    age_hours = (now - fetched_at) / 3600.0
    return age_hours > max_hours


def effective_fixture_strength(base_strength: float, gws_ahead: int) -> float:
    """Rule #46(e): team-strength-tier s "shrinks toward 0 with distance"
    from GW+2 onward, reaching 0 "beyond GW+5". Disclosed shape (the doc
    gives the endpoints, not the curve): full `base_strength` through
    gws_ahead<=1 (the near-term horizon this tier still uses when the market
    tier isn't available/passing its checks for GW+1 itself), linear decay
    across gws_ahead 2..5 down to 0, and exactly 0 for gws_ahead>=6."""
    if gws_ahead <= 1:
        return base_strength
    if gws_ahead >= 6:
        return 0.0
    # gws_ahead in {2,3,4,5} -> decay fraction 4/5, 3/5, 2/5, 1/5 (strictly
    # below full strength starting at GW+2, strictly above 0 through GW+5,
    # matching "shrinking toward 0 with distance ... beyond GW+5 ... s=0").
    frac = (6 - gws_ahead) / 5.0
    return base_strength * frac


def fixture_attack_factor_vec(opp_id: pd.Series, baselines: dict, cfg: dict,
                               gws_ahead: pd.Series | int | None = None) -> pd.Series:
    """Rule #46's FF, vectorized: FF = (opponent's shrunk xG-against baseline
    / league average xG) ^ strength, clamped to [min_ff, max_ff] (a disclosed
    safety clamp of this implementation, not a number the doc itself states
    -- guards against an extreme still-early-season baseline before
    shrinkage has fully kicked in; mirrors the existing clamp pattern
    fpl_engine.cs_pct_poisson_vec() already uses for the same reason).

    Note: this ratio depends ONLY on the opponent's own defensive baseline
    relative to the league average -- the player's own team's baseline
    algebraically cancels out of Rule #46(a)'s "own baseline" ratio under
    this multiplicative construction (fixture xG for team = team's own
    baseline x opponent's relative weakness), which is expected: a fixture
    adjustment is inherently about how this OPPONENT compares to an average
    opponent, not about the player's own team's absolute output level.

    Returns 1.0 (no adjustment) for any opponent with no baseline entry
    (unmapped/blank fixture) -- never NaN, never a crash.

    `gws_ahead` (Patch 90, Rule #46(e) distance decay -- optional, defaults
    to None which preserves the exact pre-Patch-90 flat-strength behavior
    every existing caller/test relies on): a per-row gameweek distance (0 =
    this/next GW being projected, per effective_fixture_strength()'s own
    convention) or a single int applied to every row. When given, each row's
    exponent is effective_fixture_strength(base_strength, that row's
    distance) instead of the flat configured strength -- rows at gws_ahead
    >= 6 collapse to FF = 1.0 (ratio ** 0 == 1) regardless of clamping."""
    fa_cfg = cfg.get("fixture_adjustment", {})
    if not fa_cfg.get("enabled", True) or not baselines.get("by_id"):
        return pd.Series(1.0, index=opp_id.index)
    base_strength = fa_cfg.get("strength", 0.6)
    min_ff = fa_cfg.get("min_ff", 0.7)
    max_ff = fa_cfg.get("max_ff", 1.4)
    league_avg = baselines["league_avg_xg"]
    by_id = baselines["by_id"]

    if gws_ahead is None:
        strength = base_strength
    elif isinstance(gws_ahead, pd.Series):
        strength = gws_ahead.map(lambda g: effective_fixture_strength(base_strength, int(g)))
        strength = strength.reindex(opp_id.index)
    else:
        strength = effective_fixture_strength(base_strength, int(gws_ahead))

    opp_against = opp_id.map(lambda i: by_id.get(i, {}).get("xg_against"))
    opp_against = pd.to_numeric(opp_against, errors="coerce")
    ratio = (opp_against / league_avg).where(opp_against.notna(), 1.0)
    ff = ratio.clip(lower=1e-6) ** strength
    return ff.clip(lower=min_ff, upper=max_ff).fillna(1.0)


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
                 gw_list: list[int], fixture_baselines: dict | None = None) -> pd.DataFrame:
    """Same Core Formula pipeline as the CLI tool, plus Step 3c's set-piece
    multiplier applied to npxG_blend right after Step 3's decay blend.

    `fixture_baselines` (Patch 83, v6.9 Standing Rule #46): the dict
    compute_team_fixture_baselines() returns, or None/empty to skip fixture
    adjustment entirely (e.g. a caller with no olbauday data this run, or an
    older test that predates this parameter -- default is None so every
    existing call site keeps working unmodified). When present, goal/assist
    terms (npxG/xA only, per Rule #46(d) -- clean sheets/bonus/DEFCON/cards
    untouched) are scaled by fixture_attack_factor_vec() for MID/DEF/FWD
    (Rule #46(d) explicitly lists these three, not GK).

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
                   "dc90_hist", "dc90_cur", "starts", "cs_pct_override", "cs_gw",
                   "penalties_order", "corners_and_indirect_freekicks_order",
                   "direct_freekicks_order"]
    # Patch 120: per-gameweek clean-sheet columns of manual_overrides.csv (cs_pct_gw6, cs_pct_gw7, ...)
    _pergw = eng.pergw_columns(pl)
    player_cols = player_cols + [c for c in _pergw.values() if c not in player_cols]
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
    # Patch 118: a clean-sheet override is valid for ONE gameweek (cs_gw). Other weeks use the model's own formula. A row with
    # no gameweek anywhere (unscoped) keeps the old behaviour. Switch overrides.cs_scope_to_gameweek (False = old behaviour).
    cs_gw_col = pd.to_numeric(exploded["cs_gw"], errors="coerce") if "cs_gw" in exploded.columns \
        else pd.Series(np.nan, index=exploded.index)
    if cfg.get("overrides", {}).get("cs_scope_to_gameweek", True):
        cs_applies = cs_override.notna() & (cs_gw_col.isna() | (pd.to_numeric(exploded["event"], errors="coerce") == cs_gw_col))
    else:
        cs_applies = cs_override.notna()
    # Patch 120: a per-gameweek value (cs_pct_gw<event>) is used for its OWN gameweek and beats the single-value column; a week
    # with no value uses the single-value column (scoped as above), then the model formula.
    _ev = pd.to_numeric(exploded["event"], errors="coerce")
    cs_pergw = pd.Series(np.nan, index=exploded.index)
    for _g, _c in _pergw.items():
        _m = (_ev == _g)
        if _m.any():
            cs_pergw = cs_pergw.where(~_m, pd.to_numeric(exploded[_c], errors="coerce"))
    _pg_applies = cs_pergw.notna()
    cs_gw_col = cs_gw_col.where(~_pg_applies, _ev)
    cs_override = cs_pergw.where(_pg_applies, cs_override)
    cs_applies = cs_applies | _pg_applies
    cs_pct = cs_override.where(cs_applies, cs_pct_calc)

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

    # Patch 83 (v6.9 Standing Rule #46, team-strength tier) -- fixture-adjust
    # goal/assist terms only (clean sheets/bonus/DEFCON/cards untouched, per
    # Rule #46(d)), for MID/DEF/FWD (not GK, same rule). `opp` here is
    # already the fixture opponent's FPL team `id` (fixtures_long above is
    # built straight from team_h/team_a), matching what
    # compute_team_fixture_baselines() keys its "by_id" dict on.
    if fixture_baselines and fixture_baselines.get("by_id"):
        ff = fixture_attack_factor_vec(exploded["opp"], fixture_baselines, cfg)
        applies_ff = exploded["position"].isin(["MID", "DEF", "FWD"]) & ~is_blank
        ff = ff.where(applies_ff, 1.0)
        npxg_adj = npxg_adj * ff
        xa_blend = xa_blend * ff

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
        # Patch 67 (manager question: "can we fetch the latest news from the
        # FPL app or website?") — both already sit on every bootstrap-static
        # element (same source `pl["news"]`/`pl["status"]` above already come
        # from), just never previously selected out into the player table.
        # chance_of_playing_next_round was already read further upstream
        # (build_player_table's own numcols coercion, see the Rule #24 note
        # near the top of this function) for xm/rescue-flag purposes, but
        # was never itself carried through to this returned table — added
        # here so the news feed can show the actual percentage, not just the
        # a/d/i/s/u status letter. news_added is new: the ISO timestamp FPL
        # itself puts on that news text, used to sort the news feed newest-
        # first and to give it the exact, verified timestamp this project's
        # Standing Rules require for any news item referenced.
        "chance_of_playing_next_round": (pl["chance_of_playing_next_round"].values
                                          if "chance_of_playing_next_round" in pl.columns else None),
        "news_added": pl["news_added"].values if "news_added" in pl.columns else None,
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

    # Patch 118: sheet-vs-formula clean-sheet diagnostics (only overridden rows), travels with the projection as .attrs
    try:
        _d = pd.DataFrame({"_pidx": exploded["_pidx"].values, "event": exploded["event"].values,
                           "cs_calc": cs_pct_calc.values, "cs_sheet": cs_override.values, "cs_gw": cs_gw_col.values,
                           "applied": cs_applies.values})
        _d = _d[_d["cs_sheet"].notna() & _d["event"].notna()].copy()
        _code = pl.set_index("_pidx")["code"] if "code" in pl.columns else pd.Series(dtype=float)
        _d["code"] = _d["_pidx"].map(_code)
        out.attrs["cs_diag"] = CsDiag(_d[["code", "event", "cs_calc", "cs_sheet", "cs_gw", "applied"]].reset_index(drop=True))
    except Exception:
        pass
    return out


def solve_ceiling(cfg: dict, proj: pd.DataFrame):
    """§1a's unconstrained Ceiling_xPts — the best possible £100m/2-5-5-3/
    max-3-per-club squad from the FULL pool, ignoring what you currently own
    or how many transfers you have. Kept as the "theoretical" reference
    number (Standing Rules #16/#18 disclosure) alongside the reachable
    ceiling below, which is what the Team Rating % headline now uses.

    label="theoretical_ceiling" (Patch 72) — lets app.py surface the real
    None-return reason (via opt.get_diagnostic()) directly in the "Team
    Rating % (GWx-y)" card's tooltip if this returns None on a live run,
    instead of just a silent "—" with no explanation."""
    return opt.solve_squad(proj, cfg, budget=cfg["squad_rules"]["budget"], label="theoretical_ceiling")


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
    # label="reachable_ceiling" (Patch 72) — surfaces the real None-return
    # reason (via opt.get_diagnostic()) in the Wildcard trigger card's
    # tooltip when this returns None on a live run.
    return opt.solve_squad(proj, cfg, budget=cfg["squad_rules"]["budget"],
                            retain_pool_codes=current_squad_codes, min_retain=min_retain,
                            label="reachable_ceiling")


def solve_reachable_ceiling_by_gw(cfg: dict, proj: pd.DataFrame, current_squad_codes: list,
                                   free_transfers: int, detect_gw_list: list) -> dict:
    """v6.9 amended Wildcard trigger (Standing Rule #45's §8c amendment,
    verified against the doc, not inferred): "the reachable ceiling is the
    best squad reachable with the free transfers available and accruing one
    per gameweek." The pre-v6.9 app code (solve_reachable_ceiling(), still
    used elsewhere for the Team Rating % headline, which the doc's §1a
    formula does NOT ask to accrue) solved ONE reachable squad with a
    single static free_transfers count and reused it for every GW in the
    detection window — confirmed by reading app.py's caller (the single
    `reachable_detect = data_pipeline.solve_reachable_ceiling(...)` call
    feeding every GW of wildcard_trigger_check's by_gw loop). That
    understates the reachable ceiling for the later weeks of a 3-4 GW
    window, since by week 3 a manager holding transfers would genuinely
    have more of them banked than at week 1.

    This solves one reachable squad PER GW in detect_gw_list, with
    min_retain tightened week-by-week to reflect free_transfers + (that
    week's 0-based offset into the sorted window), capped at the Standing
    Rule #35/#37 bank cap of 5 (a manager can never actually have more than
    5 free transfers banked at once, accrual or not). detect_gw_list is
    assumed already sorted ascending (app.py builds it via
    `range(planning_gw, planning_gw + window_size)`).

    Returns {gw: solve_squad()'s dict-or-None}, one entry per GW in
    detect_gw_list — never a single shared squad."""
    result = {}
    for offset, gw in enumerate(sorted(detect_gw_list)):
        accrued_ft = min(5, max(0, free_transfers) + offset)
        result[gw] = solve_reachable_ceiling(cfg, proj, current_squad_codes, accrued_ft)
    return result


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
    budget basis (Rule #25) as the play/hold solve.

    label="free_hit_optimal" (Patch 72) — surfaces the real None-return
    reason (via opt.get_diagnostic()) in the "GW{n} Rating" card's tooltip
    when this returns None on a live run, including the case right below
    where the column itself is missing (never even reaches optimizer.py)."""
    col = f"xpts_gw{gw}"
    if col not in proj.columns:
        opt.set_diagnostic("free_hit_optimal", f"projection column '{col}' is not present in this run's "
                                                 f"player pool at all (columns present: "
                                                 f"{[c for c in proj.columns if c.startswith('xpts_gw')]}) — "
                                                 f"GW{gw} may be outside the projected horizon this run.")
        return None
    return opt.solve_xi_first_squad(proj, cfg, budget=total_value, gw_col=col, label="free_hit_optimal")
