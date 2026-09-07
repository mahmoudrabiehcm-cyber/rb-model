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


if __name__ == "__main__":
    snap = load_snapshot()
    print(f"Source: {snap.source} | last completed GW: {snap.current_gw} | "
          f"planning GW: {snap.planning_gw}")
    print(f"Players: {len(snap.players)} | Teams: {len(snap.teams)} | Fixtures: {len(snap.fixtures)}")
    if snap.stale_warning:
        print("WARNING:", snap.stale_warning)
    print(snap.players.head(3).to_string())
