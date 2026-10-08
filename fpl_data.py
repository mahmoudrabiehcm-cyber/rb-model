"""
fpl_data.py
Live data-fetch layer for the FPL Projection Model (v3.3).

Free, no API key. Two source tiers, tried in order, mirroring the model's
own Step 2 / Step 0 design:

  1. Official FPL API (fantasy.premierleague.com/api/...) — freshest,
     includes entry/squad endpoints. Some sandboxed environments block
     this host by network policy; the fallback below covers that case.
  2. GitHub mirror (github.com/vaastav/Fantasy-Premier-League) — updates
     on its own cadence and can lag; per Standing Rule #17 we spot-check
     freshness before trusting it for "current gameweek" claims.

Run this file directly to smoke-test connectivity:  python3 fpl_data.py
"""
from __future__ import annotations
import io
import json
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

FPL_API = "https://fantasy.premierleague.com/api"
GITHUB_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fpl-model-tool/1.0)"}


def _get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
        return None
    except requests.RequestException:
        return None


@dataclass
class FplSnapshot:
    players: pd.DataFrame
    teams: pd.DataFrame
    fixtures: pd.DataFrame
    events: pd.DataFrame
    source: str          # "official_api" or "github_mirror"
    fetched_at: float
    current_gw: int      # last COMPLETED/locked gameweek. Use this for
                          # fetching the manager's picks snapshot -- the
                          # official API only has a picks record for a
                          # gameweek that's actually happened (querying a
                          # future gameweek 404s until its deadline passes
                          # or a transfer is made for it). NOT the
                          # gameweek to plan transfers/captaincy/chips
                          # for -- see planning_gw below.
    stale_warning: Optional[str] = None
    raw_boot: Optional[dict] = None   # full bootstrap-static payload when source=="official_api" (carries .chips)
    current_gw_finished: bool = False      # Patch 11 -- bootstrap events[current_gw].finished. False means
                                            # some fixture in that gameweek genuinely hasn't been played yet.
    current_gw_data_checked: bool = False  # Patch 11 -- bootstrap events[current_gw].data_checked. FPL sets
                                            # this only once it has manually confirmed bonus points and locked
                                            # the gameweek's final scores/prices for good. A gameweek can be
                                            # `finished=True` (every match has a final whistle) for HOURS to a
                                            # day+ while `data_checked` is still False -- during that window the
                                            # entry's points/rank from `entry/{id}/` and `entry/{id}/history/`
                                            # are real numbers pulled live from the same official API this app
                                            # already calls, but they are PROVISIONAL: bonus points can still
                                            # move, and overall rank keeps shifting as other managers' gameweeks
                                            # get processed too. This is an FPL-side lag this app cannot bypass
                                            # (the official site/app shows the exact same provisional numbers
                                            # during this window) -- the fix here is disclosure, not a different
                                            # data source: tell the manager which state a rank/points figure is
                                            # in rather than silently presenting a still-moving number as final
                                            # (Standing Rule #4).
    recent_start_ids: Optional[set] = None      # Patch 14 — Standing Rule #19 support,
                                                 # see fetch_recent_start_ids(). None/empty
                                                 # (with recent_start_checked_gws empty too)
                                                 # means the signal is UNAVAILABLE this run.
    recent_start_checked_gws: Optional[list] = None
    recent_start_window: Optional[tuple] = None  # (window_start_gw, window_end_gw)
    planning_gw: Optional[int] = None  # next gameweek whose deadline HASN'T
                                        # passed yet -- the one xPts
                                        # projections, transfer suggestions,
                                        # captaincy, and chip advisories
                                        # should target. Distinct from
                                        # current_gw: the FPL API keeps a
                                        # just-finished gameweek marked
                                        # is_current for a while (through
                                        # kickoff and results processing)
                                        # even though its deadline is long
                                        # past and there's nothing left to
                                        # decide for it. Falls back to
                                        # current_gw when there's no next
                                        # gameweek (end of season).


def fetch_bootstrap_official() -> Optional[dict]:
    r = _get(f"{FPL_API}/bootstrap-static/")
    if r is None:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def fetch_fixtures_official() -> Optional[list]:
    r = _get(f"{FPL_API}/fixtures/")
    if r is None:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def fetch_entry_official(entry_id: int) -> Optional[dict]:
    r = _get(f"{FPL_API}/entry/{entry_id}/")
    if r is None:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def fetch_entry_picks_official(entry_id: int, gw: int) -> Optional[dict]:
    r = _get(f"{FPL_API}/entry/{entry_id}/event/{gw}/picks/")
    if r is None:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def fetch_entry_transfers_official(entry_id: int) -> Optional[list]:
    """Patch 117c: the manager's full transfer list (element_in/out, costs in tenths, event, time). Used to
    work out each owned player's selling price."""
    r = _get(f"{FPL_API}/entry/{entry_id}/transfers/")
    if r is None:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def fetch_entry_history_official(entry_id: int) -> Optional[dict]:
    """Per-GW points/rank/value/event_transfers/event_transfers_cost/points_on_bench
    for the current season, plus `chips` (name + played gameweek) and `past`
    (prior seasons summary). Free, no auth — same public entry namespace as
    the picks/entry endpoints above. Used for: free-transfer derivation (Step
    7a addendum), chip status tracking (Step 9), and the Season Ledger /
    rank-trend display."""
    r = _get(f"{FPL_API}/entry/{entry_id}/history/")
    if r is None:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def fetch_event_live_official(gw: int) -> Optional[dict]:
    r = _get(f"{FPL_API}/event/{gw}/live/")
    if r is None:
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        return None


def fetch_recent_start_ids(last_completed_gw: int, window: int) -> dict:
    """Standing Rule #19 (Bench GK Verification) support — Patch 14. The xM
    Floor Rule's `confirmed_current_season_start_floor` used to trigger off
    ANY start this season (`starts >= 1`, a season-long total with no
    recency), which meant a keeper (or any player) who started once months
    ago covering an injury or a cup match, then went straight back to the
    bench, still got treated as if he had a ~0.88 expected-minutes floor —
    exactly the "backup GK who isn't actually nailed" failure the model
    doc's Standing Rule #19 exists to catch, but that rule was documented,
    not implemented anywhere in this codebase.

    This builds a per-player "did he start in the last `window` FINISHED
    gameweeks" signal so `estimate_xm()` can require RECENT evidence, not
    just season-long evidence, before granting the confirmed-start floor.
    Cost: one extra request PER GAMEWEEK in the window (not per player) —
    `event/{gw}/live/` returns every player's stats for that single
    gameweek in one call, so a 4-GW window is 4 extra requests total,
    regardless of squad/pool size.

    `minutes >= 60` is used as the "started" proxy (the live payload's
    per-player stats block doesn't carry an explicit start flag) — a
    reasonable proxy, but disclosed as one rather than presented as an
    exact match to the official "starts" definition.

    Returns {"started_ids": set[int], "checked_gws": [int,...],
    "window_start_gw": int, "window_end_gw": int} — `checked_gws` may be
    shorter than the requested window if a fetch failed for some gw in it
    (best-effort signal, degrades gracefully rather than blocking the
    pipeline); an empty `checked_gws` means the caller should treat this
    signal as UNAVAILABLE this run, not as "nobody started recently.\""""
    started = set()
    checked_gws = []
    window_start = max(1, last_completed_gw - window + 1)
    for gw in range(window_start, last_completed_gw + 1):
        live = fetch_event_live_official(gw)
        if live is None or "elements" not in live:
            continue
        checked_gws.append(gw)
        for el in live["elements"]:
            stats = el.get("stats", {}) or {}
            if (stats.get("minutes", 0) or 0) >= 60:
                started.add(el.get("id"))
    return {"started_ids": started, "checked_gws": checked_gws,
            "window_start_gw": window_start, "window_end_gw": last_completed_gw}


def fetch_bootstrap_chips(boot: dict) -> list:
    """bootstrap-static's `chips` array: the season's full chip calendar,
    each with a usable gameweek window (`chip_type`, `start_event`,
    `stop_event`). Free, already pulled with bootstrap-static — no extra
    fetch. Falls back to an empty list on older/mirrored snapshots that
    don't carry it."""
    return boot.get("chips", []) if boot else []


def fetch_csv_mirror(season: str, filename: str) -> Optional[pd.DataFrame]:
    r = _get(f"{GITHUB_RAW}/{season}/{filename}")
    if r is None:
        return None
    try:
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None


def load_snapshot(season: str = "2026-27", recency_window: int = 4) -> FplSnapshot:
    """
    Try the official API first (freshest + gives us `events` for current GW
    and `element_type` -> position mapping consistently). Fall back to the
    GitHub mirror CSVs if the API host is blocked by network policy.

    `recency_window`: how many recently-FINISHED gameweeks to check for
    Standing Rule #19 support (see `fetch_recent_start_ids`). Only fetched
    on the official-API path — the mirror path has no per-gameweek live
    endpoint, so the recency signal is left unavailable there (disclosed via
    empty `recent_start_checked_gws`, not silently assumed)."""
    boot = fetch_bootstrap_official()
    fixtures_json = fetch_fixtures_official()

    if boot is not None:
        players = pd.DataFrame(boot["elements"])
        teams = pd.DataFrame(boot["teams"])
        events = pd.DataFrame(boot["events"])
        fixtures = pd.DataFrame(fixtures_json) if fixtures_json else pd.DataFrame()
        current_gw = _infer_current_gw(events)
        planning_gw = _infer_planning_gw(events, current_gw)
        gw_finished, gw_checked = False, False
        if "id" in events.columns:
            cur_row = events[events["id"] == current_gw]
            if not cur_row.empty:
                gw_finished = bool(cur_row.iloc[0].get("finished", False))
                gw_checked = bool(cur_row.iloc[0].get("data_checked", False))
        # Standing Rule #19 support: check the last `recency_window` gameweeks
        # that have actually been PLAYED. current_gw itself may still be
        # mid-processing (see current_gw_finished above) — event/{gw}/live/
        # works fine for an unfinished-but-played gameweek too (it's how
        # in-play scores are served), so this intentionally includes
        # current_gw rather than only fully-finished ones.
        last_playable_gw = current_gw if current_gw >= 1 else 1
        recent = fetch_recent_start_ids(last_playable_gw, recency_window)
        return FplSnapshot(players, teams, fixtures, events, "official_api",
                            time.time(), current_gw, raw_boot=boot,
                            current_gw_finished=gw_finished, current_gw_data_checked=gw_checked,
                            recent_start_ids=recent["started_ids"],
                            recent_start_checked_gws=recent["checked_gws"],
                            recent_start_window=(recent["window_start_gw"], recent["window_end_gw"]),
                            planning_gw=planning_gw)

    # ---- fallback: GitHub mirror ----
    players = fetch_csv_mirror(season, "players_raw.csv")
    teams = fetch_csv_mirror(season, "teams.csv")
    fixtures = fetch_csv_mirror(season, "fixtures.csv")
    if players is None or teams is None:
        raise RuntimeError(
            "Could not reach either the official FPL API or the GitHub mirror. "
            "Check network/egress settings for this environment."
        )
    events = pd.DataFrame()
    current_gw = _infer_current_gw_from_fixtures(fixtures) if fixtures is not None else 1
    planning_gw = _infer_planning_gw_from_fixtures(fixtures, current_gw) if fixtures is not None else 1
    warning = None
    if (teams.get("played") is not None) and teams["played"].fillna(0).sum() == 0 and current_gw > 1:
        warning = ("GitHub mirror shows 0 played matches for all teams while fixtures "
                   "suggest GW1+ has kicked off -- classic Standing Rule #17 staleness. "
                   "Treat player-level totals as possibly lagged; do not trust "
                   "current-gameweek exact points from this source (use it for "
                   "historical/career baselines only).")
    return FplSnapshot(players, teams, fixtures if fixtures is not None else pd.DataFrame(),
                        events, "github_mirror", time.time(), current_gw, warning,
                        planning_gw=planning_gw)


def _infer_current_gw(events: pd.DataFrame) -> int:
    """Last COMPLETED/locked gameweek (is_current) -- NOT the one to plan
    for. See _infer_planning_gw."""
    if events.empty:
        return 1
    if "is_current" in events.columns and events["is_current"].any():
        return int(events.loc[events["is_current"], "id"].iloc[0])
    if "is_next" in events.columns and events["is_next"].any():
        return int(events.loc[events["is_next"], "id"].iloc[0])
    finished = events[events.get("finished", False) == True] if "finished" in events.columns else pd.DataFrame()
    return int(finished["id"].max()) + 1 if not finished.empty else 1


def _infer_planning_gw(events: pd.DataFrame, fallback: int) -> int:
    """Next gameweek whose deadline HASN'T passed (is_next) -- the one
    xPts/transfer/captaincy/chip logic should target. is_current lags this
    by one gameweek from the moment a deadline passes until that
    gameweek's results are fully processed, which is exactly the window
    this app is most likely to be used in (reviewing last week, planning
    next week)."""
    if events.empty:
        return fallback
    if "is_next" in events.columns and events["is_next"].any():
        return int(events.loc[events["is_next"], "id"].iloc[0])
    return fallback


def _infer_current_gw_from_fixtures(fixtures: pd.DataFrame) -> int:
    """Last COMPLETED gameweek, mirror-CSV path. NOT the one to plan for."""
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        return 1
    finished = fixtures[fixtures.get("finished", False) == True]
    if finished.empty:
        return 1
    return int(finished["event"].max())


def _infer_planning_gw_from_fixtures(fixtures: pd.DataFrame, fallback: int) -> int:
    """Next gameweek with an unplayed fixture, mirror-CSV path -- the one
    to actually plan for."""
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        return fallback
    unfinished = fixtures[fixtures.get("finished", False) == False]
    if unfinished.empty:
        return fallback
    return int(unfinished["event"].min())


def load_historical_snapshot(season: str) -> Optional[pd.DataFrame]:
    """Prior-season players_raw.csv, for the Decay Schedule's historical leg.
    Match players across seasons on the stable `code` field, not `id`."""
    return fetch_csv_mirror(season, "players_raw.csv")


# ---------------------------------------------------------------------------
# Patch 83 (v6.9 Standing Rule #46, Fixture-Adjusted Attack Rule) -- new
# automated source. Manager confirmed the scope explicitly (2026-09-29):
# build the team-strength tier now (real match-level team xG, verified live
# before writing this), leave the market-odds tier as a fast-follow rather
# than guess at scraping reliability this session couldn't fully verify.
#
# olbauday/FPL-Core-Insights ("Sources added in v6.9" in the model doc) --
# real schema confirmed live (not guessed) via a direct fetch of
# data/2026-2027/By Gameweek/GW1/matches.csv and GW6/matches.csv:
#   - one CSV per gameweek at
#     data/{season}/By Gameweek/GW{n}/matches.csv (NOT one combined
#     season-level file -- that path 404s; confirmed by trying
#     "2026-2027/matches.csv" and "2026-27/matches.csv" first, both 404).
#   - columns include: gameweek, home_team, away_team, finished, tournament,
#     home_expected_goals_xg, away_expected_goals_xg (there's also a
#     non-penalty variant, home/away_non_penalty_xg, not used here --
#     Rule #46 doesn't specify non-penalty for the team-level baseline, and
#     penalties are a small, roughly team-neutral share of team xG).
#   - home_team/away_team are numeric but NOT the season-specific 1-20 `id`
#     FPL players/fixtures use -- they matched the STABLE `code` field
#     instead (verified: Bournemouth=91, Brentford=94, Aston Villa=7 in a
#     real fetched row, cross-checked against vaastav's 2026-27 teams.csv
#     `code` column, which has exactly those same code->club mappings).
#     Joining this data into the pipeline (which keys everything off `id`)
#     requires the caller to build a code->id map from the current season's
#     teams table.
#   - tournament=="prem" filters to real Premier League matches only (the
#     per-GW file observed was already prem-only in practice, but this repo
#     explicitly also carries cups/friendlies/Euro competitions per its own
#     README, so the filter is kept defensively rather than assumed).
# ---------------------------------------------------------------------------
OLBAUDAY_RAW = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"


def fetch_team_match_xg(season_slug: str, gw_list: list[int]) -> tuple[pd.DataFrame, Optional[str]]:
    """Pulls olbauday's per-GW matches.csv for every GW in `gw_list`,
    filters to finished, tournament=="prem" rows, and returns one row PER
    TEAM PER MATCH (i.e. each match contributes two rows, home and away) so
    the caller can group by team code directly:
        team_code, opp_code, is_home, gameweek, xg_for, xg_against
    `season_slug` is olbauday's own season folder name (e.g. "2026-2027" --
    note the 4-digit-dash-4-digit form, different from vaastav's "2026-27").

    Never raises -- a GW that fails to fetch (network blocked, file not
    published yet, schema drift) is silently skipped, and the whole call
    returns whatever finished matches it DID get plus a warning string
    naming which GWs were missing (empty warning if none were). A totally
    empty result (e.g. every GW blocked) still returns a valid, empty
    DataFrame with the right columns -- callers must treat that as
    "no fixture-adjustment data available this run" and fall back to FF=1.0
    (Rule #46(a)'s own-baseline normalisation is a no-op with no baseline
    data), never crash the page over it."""
    cols = ["team_code", "opp_code", "is_home", "gameweek", "xg_for", "xg_against"]
    frames = []
    missing_gws = []
    for gw in gw_list:
        url = f"{OLBAUDAY_RAW}/{season_slug}/By%20Gameweek/GW{gw}/matches.csv"
        r = _get(url, timeout=15)
        if r is None:
            missing_gws.append(gw)
            continue
        try:
            df = pd.read_csv(io.StringIO(r.text))
        except Exception:
            missing_gws.append(gw)
            continue
        needed = {"home_team", "away_team", "finished", "tournament",
                  "home_expected_goals_xg", "away_expected_goals_xg", "gameweek"}
        if not needed.issubset(df.columns):
            missing_gws.append(gw)
            continue
        df = df[(df["tournament"] == "prem") & (df["finished"] == True)]  # noqa: E712
        if df.empty:
            continue
        home_rows = pd.DataFrame({
            "team_code": df["home_team"], "opp_code": df["away_team"], "is_home": True,
            "gameweek": df["gameweek"],
            "xg_for": pd.to_numeric(df["home_expected_goals_xg"], errors="coerce"),
            "xg_against": pd.to_numeric(df["away_expected_goals_xg"], errors="coerce"),
        })
        away_rows = pd.DataFrame({
            "team_code": df["away_team"], "opp_code": df["home_team"], "is_home": False,
            "gameweek": df["gameweek"],
            "xg_for": pd.to_numeric(df["away_expected_goals_xg"], errors="coerce"),
            "xg_against": pd.to_numeric(df["home_expected_goals_xg"], errors="coerce"),
        })
        frames.append(pd.concat([home_rows, away_rows], ignore_index=True))
    if not frames:
        warning = (f"could not fetch any of olbauday's per-GW matches.csv files for GWs {gw_list} "
                   f"(network blocked or files not published yet)" if missing_gws else None)
        return pd.DataFrame(columns=cols), warning
    result = pd.concat(frames, ignore_index=True).dropna(subset=["xg_for", "xg_against"])
    warning = (f"olbauday matches.csv unavailable for GW(s) {missing_gws} -- fixture-adjustment "
               f"baselines are built from the other finished GWs only" if missing_gws else None)
    return result[cols], warning


# ---------------------------------------------------------------------------
# Patch 90 (v6.9 Standing Rule #46(e), market-odds leg) -- confirmed via code
# read (grep for "odds" across this file) that no odds-fetching function
# existed anywhere before this patch. Uses The Odds API
# (the-odds-api.com) -- picked over the two other candidates this session
# researched live: an earlier session's oddschecker scrape attempt (see
# data_pipeline.py's Patch 83 comment) could not verify automated-fetch
# terms of use; SportsGameOdds' free tier documentation never confirms EPL
# is included (only 8 named leagues, none of them EPL, are listed for its
# free "Amateur" tier). The Odds API has a dedicated, documented EPL
# endpoint on every tier including free, and its Terms & Conditions
# (read in full this session, 2026-09-30) explicitly permit "displaying our
# data in a UI... including for commercial use" and "using our data to
# train statistical and machine learning models" -- exactly this use --
# while only prohibiting reselling the raw odds as a competing data feed,
# which this app never does.
# ---------------------------------------------------------------------------
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"


def fetch_market_odds_epl(api_key: str, region: str = "uk",
                           market: str = "h2h") -> tuple[list, Optional[str]]:
    """Pulls current EPL 1X2 ("h2h") odds from every bookmaker The Odds API
    returns for `region`. Never raises -- any network failure, non-200
    status, or malformed payload returns ([], a warning string) rather than
    crashing the page, mirroring fetch_team_match_xg()'s own contract.

    Returns (events, warning) where each event is:
        {"home_team": str, "away_team": str, "commence_time": str,
         "bookmakers": [{"key": str, "home_odds": float, "draw_odds": float,
                          "away_odds": float}, ...]}
    A bookmaker whose h2h market can't be parsed into a clean
    home/draw/away triple (missing outcome, unexpected team name) is
    silently dropped from that event's bookmaker list rather than failing
    the whole event -- an event can end up with an empty bookmakers list,
    which callers must treat as "no usable odds this fixture" (falls back
    to the team-strength tier), never a crash."""
    params = {"apiKey": api_key, "regions": region, "markets": market, "oddsFormat": "decimal"}
    try:
        r = requests.get(ODDS_API_BASE, params=params, timeout=20)
    except requests.RequestException as e:
        return [], f"market odds fetch failed (network error: {e})"
    if r.status_code != 200:
        return [], f"market odds fetch failed (HTTP {r.status_code} from The Odds API)"
    try:
        payload = r.json()
    except Exception:
        return [], "market odds fetch failed (response was not valid JSON)"
    if not isinstance(payload, list):
        return [], "market odds fetch failed (unexpected response shape)"

    events = []
    skipped = 0
    for raw_ev in payload:
        try:
            home_team = raw_ev["home_team"]
            away_team = raw_ev["away_team"]
        except (KeyError, TypeError):
            skipped += 1
            continue
        bookmakers = []
        for bk in raw_ev.get("bookmakers", []) or []:
            try:
                h2h = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
                if h2h is None:
                    continue
                outcomes = {o["name"]: float(o["price"]) for o in h2h.get("outcomes", [])}
                home_odds = outcomes.get(home_team)
                away_odds = outcomes.get(away_team)
                draw_odds = outcomes.get("Draw")
                if home_odds is None or away_odds is None or draw_odds is None:
                    continue
                bookmakers.append({"key": bk.get("key", "unknown"), "home_odds": home_odds,
                                    "draw_odds": draw_odds, "away_odds": away_odds})
            except (KeyError, TypeError, ValueError):
                continue
        events.append({"home_team": home_team, "away_team": away_team,
                        "commence_time": raw_ev.get("commence_time"), "bookmakers": bookmakers})
    warning = f"{skipped} event(s) in the odds response were missing team names and were skipped" \
        if skipped else None
    return events, warning


if __name__ == "__main__":
    snap = load_snapshot()
    print(f"Source: {snap.source} | last completed GW: {snap.current_gw} | "
          f"planning GW: {snap.planning_gw}")
    print(f"Players: {len(snap.players)} | Teams: {len(snap.teams)} | Fixtures: {len(snap.fixtures)}")
    if snap.stale_warning:
        print("WARNING:", snap.stale_warning)
    print(snap.players.head(3).to_string())
