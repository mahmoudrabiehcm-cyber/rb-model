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
    current_gw: int
    stale_warning: Optional[str] = None
    raw_boot: Optional[dict] = None   # full bootstrap-static payload when source=="official_api" (carries .chips)


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


def load_snapshot(season: str = "2026-27") -> FplSnapshot:
    """
    Try the official API first (freshest + gives us `events` for current GW
    and `element_type` -> position mapping consistently). Fall back to the
    GitHub mirror CSVs if the API host is blocked by network policy.
    """
    boot = fetch_bootstrap_official()
    fixtures_json = fetch_fixtures_official()

    if boot is not None:
        players = pd.DataFrame(boot["elements"])
        teams = pd.DataFrame(boot["teams"])
        events = pd.DataFrame(boot["events"])
        fixtures = pd.DataFrame(fixtures_json) if fixtures_json else pd.DataFrame()
        current_gw = _infer_current_gw(events)
        return FplSnapshot(players, teams, fixtures, events, "official_api",
                            time.time(), current_gw, raw_boot=boot)

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
    warning = None
    if (teams.get("played") is not None) and teams["played"].fillna(0).sum() == 0 and current_gw > 1:
        warning = ("GitHub mirror shows 0 played matches for all teams while fixtures "
                   "suggest GW1+ has kicked off -- classic Standing Rule #17 staleness. "
                   "Treat player-level totals as possibly lagged; do not trust "
                   "current-gameweek exact points from this source (use it for "
                   "historical/career baselines only).")
    return FplSnapshot(players, teams, fixtures if fixtures is not None else pd.DataFrame(),
                        events, "github_mirror", time.time(), current_gw, warning)


def _infer_current_gw(events: pd.DataFrame) -> int:
    if events.empty:
        return 1
    if "is_current" in events.columns and events["is_current"].any():
        return int(events.loc[events["is_current"], "id"].iloc[0])
    if "is_next" in events.columns and events["is_next"].any():
        return int(events.loc[events["is_next"], "id"].iloc[0])
    finished = events[events.get("finished", False) == True] if "finished" in events.columns else pd.DataFrame()
    return int(finished["id"].max()) + 1 if not finished.empty else 1


def _infer_current_gw_from_fixtures(fixtures: pd.DataFrame) -> int:
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        return 1
    unfinished = fixtures[fixtures.get("finished", False) == False]
    if unfinished.empty:
        return int(fixtures["event"].max())
    return int(unfinished["event"].min())


def load_historical_snapshot(season: str) -> Optional[pd.DataFrame]:
    """Prior-season players_raw.csv, for the Decay Schedule's historical leg.
    Match players across seasons on the stable `code` field, not `id`."""
    return fetch_csv_mirror(season, "players_raw.csv")


if __name__ == "__main__":
    snap = load_snapshot()
    print(f"Source: {snap.source} | current GW (inferred): {snap.current_gw}")
    print(f"Players: {len(snap.players)} | Teams: {len(snap.teams)} | Fixtures: {len(snap.fixtures)}")
    if snap.stale_warning:
        print("WARNING:", snap.stale_warning)
    print(snap.players.head(3).to_string())
