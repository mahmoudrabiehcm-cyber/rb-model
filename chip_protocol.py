"""
chip_protocol.py
Step 9 — Chip Timing Protocol (FPL Projection Model v4.0).

Two mechanical pieces (chip status tracking, DGW/BGW detection) plus one
deliberately non-mechanical piece (Wildcard timing — always a flag, never
a verdict, per Standing Rule #24).
"""
from __future__ import annotations
import pandas as pd

CHIP_LABELS = {
    "wildcard": "Wildcard",
    "3xc": "Triple Captain",
    "bboost": "Bench Boost",
    "freehit": "Free Hit",
}


def chip_status(boot_chips: list, history_chips: list) -> list[dict]:
    """boot_chips: bootstrap-static['chips'] — the season's full chip
    calendar (each entry has name/start_event/stop_event). Up to two
    windows per chip name in a season with a reset at the winter break.
    history_chips: entry history's ['chips'] — chips this manager has
    actually played (name + event). Cross-referencing the two gives, per
    chip window: used (with GW) or available (with window)."""
    played = {}  # name -> list of events played
    for c in history_chips or []:
        played.setdefault(c.get("name"), []).append(c.get("event"))

    # group calendar windows per chip name, in start_event order, so the
    # Nth played instance of a name consumes the Nth calendar window
    windows_by_name: dict[str, list[dict]] = {}
    for c in sorted(boot_chips or [], key=lambda c: (c.get("name", ""), c.get("start_event", 0))):
        windows_by_name.setdefault(c.get("name"), []).append(c)

    rows = []
    for name, windows in windows_by_name.items():
        used_events = sorted(played.get(name, []))
        label_base = CHIP_LABELS.get(name, name)
        for idx, w in enumerate(windows):
            tag = f"{label_base} {idx + 1}" if len(windows) > 1 else label_base
            if idx < len(used_events):
                rows.append({"chip": tag, "status": "used", "event": used_events[idx],
                             "window": (w.get("start_event"), w.get("stop_event"))})
            else:
                rows.append({"chip": tag, "status": "available", "event": None,
                             "window": (w.get("start_event"), w.get("stop_event"))})
    return rows


def fixture_counts_by_team(fixtures: pd.DataFrame, gw_list: list[int]) -> dict:
    """{gw: {team_id: fixture_count}} for the given gameweeks — 0 = blank,
    2+ = double. Mechanical, from official fixtures data (Standing Rule #17:
    only trust this a few gameweeks out, not a season-long projection)."""
    out = {gw: {} for gw in gw_list}
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        return out
    for gw in gw_list:
        rows = fixtures[fixtures["event"] == gw]
        counts: dict[int, int] = {}
        for _, r in rows.iterrows():
            for tid in (r.get("team_h"), r.get("team_a")):
                if pd.notna(tid):
                    counts[int(tid)] = counts.get(int(tid), 0) + 1
        out[gw] = counts
    return out


def dgw_bgw_flags(fixture_counts: dict, all_team_ids: list[int]) -> dict:
    """{gw: {"doubles": [team_id,...], "blanks": [team_id,...]}}"""
    out = {}
    for gw, counts in fixture_counts.items():
        doubles = [t for t in all_team_ids if counts.get(t, 1) >= 2]
        blanks = [t for t in all_team_ids if counts.get(t, 1) == 0]
        out[gw] = {"doubles": doubles, "blanks": blanks}
    return out


def chip_recommendations(status_rows: list[dict], dgw_bgw: dict, squad_teams: list[int],
                          squad_size: int) -> list[str]:
    """Mechanical BB/TC/FH surfacing against confirmed doubles/blanks with
    genuine squad exposure. Never fabricates a fixture — only reads what
    dgw_bgw (built from real fixture rows) already found."""
    available = {r["chip"] for r in status_rows if r["status"] == "available"}
    notes = []
    for gw, flags in dgw_bgw.items():
        doubles_in_squad = [t for t in flags["doubles"] if t in squad_teams]
        blanks_in_squad = [t for t in flags["blanks"] if t in squad_teams]
        exposure_frac = len(doubles_in_squad) / squad_size if squad_size else 0

        for chip_tag in available:
            if chip_tag.startswith("Bench Boost") and doubles_in_squad and exposure_frac >= 0.25:
                notes.append(f"GW{gw}: confirmed double for {len(doubles_in_squad)} of your "
                             f"clubs — {chip_tag} has real exposure here (verified fixture data).")
            if chip_tag.startswith("Triple Captain") and doubles_in_squad:
                notes.append(f"GW{gw}: confirmed double for {len(doubles_in_squad)} of your "
                             f"clubs — {chip_tag} is live against a real DGW here.")
            if chip_tag.startswith("Free Hit") and blanks_in_squad and \
                    len(blanks_in_squad) / squad_size >= 0.35:
                notes.append(f"GW{gw}: confirmed blank hits {len(blanks_in_squad)} of your "
                             f"clubs — {chip_tag} has genuine exposure to cover here.")
    return notes


def wildcard_flag(rank_history: list[int], flagged_player_count: int) -> str | None:
    """NEVER a verdict — Standing Rule #24. Surfaces a flag with its trigger
    stated explicitly; the decision stays with the manager. rank_history:
    overall rank for the last few gameweeks, oldest first (lower = better)."""
    triggers = []
    if len(rank_history) >= 3:
        recent = rank_history[-3:]
        if recent[0] < recent[1] < recent[2]:
            triggers.append(f"overall rank has worsened for 2 consecutive gameweeks "
                             f"({recent[0]:,} -> {recent[1]:,} -> {recent[2]:,})")
    if flagged_player_count >= 3:
        triggers.append(f"{flagged_player_count} squad players currently carry a "
                         f"data-quality or availability flag")
    if not triggers:
        return None
    return "Consider a Wildcard — " + "; ".join(triggers) + ". This is a flag, not a recommendation: the timing call is yours."
