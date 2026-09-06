"""
chip_protocol.py
Step 9 — Chip Timing Protocol (FPL Projection Model v4.0).

Two mechanical pieces (chip status tracking, DGW/BGW detection) plus one
deliberately non-mechanical piece (Wildcard timing — always a flag, never
a verdict, per Standing Rule #24).
"""
from __future__ import annotations
import pandas as pd

import optimizer as opt

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


def wildcard_flag(rank_history: list[int], flagged_player_count: int,
                   squad_xpts_total: float | None = None,
                   reachable_ceiling_total: float | None = None,
                   moe_threshold: float | None = None) -> str | None:
    """NEVER a verdict — Standing Rule #24. Surfaces a flag with its trigger
    stated explicitly; the decision stays with the manager. rank_history:
    overall rank for the last few gameweeks, oldest first (lower = better).
    Optional squad_xpts_total/reachable_ceiling_total/moe_threshold (v4.6+)
    add the quantified squad-vs-reachable-ceiling gap as an extra disclosed
    data point alongside the existing rank-trend/flagged-player triggers —
    still just another number on the flag, never a mechanical trigger of
    its own (that would contradict Rule #24)."""
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
    gap_note = ""
    if squad_xpts_total is not None and reachable_ceiling_total is not None and reachable_ceiling_total > 0:
        gap = reachable_ceiling_total - squad_xpts_total
        if moe_threshold is not None and gap < moe_threshold:
            gap_note = (f" (for reference: your squad is within margin-of-error of its own reachable "
                        f"ceiling right now — {gap:.1f} xPts gap — so a Wildcard's upside here is limited "
                        f"to whatever a full rebuild alone would add)")
        else:
            gap_note = f" (for reference: {gap:.1f} xPts gap to your own reachable ceiling)"
    return ("Consider a Wildcard — " + "; ".join(triggers) + gap_note +
            ". This is a flag, not a recommendation: the timing call is yours.")


# ---------------------------------------------------------------------------
# Chip Advisor (v5.0 addition) — quantified play/hold verdicts for Bench
# Boost, Triple Captain and Free Hit within the manager's chosen horizon,
# gated by Standing Rule #34's margin-of-error threshold so a difference
# inside demonstrated weekly noise is reported as a tie ("hold"), never
# dressed up as a clear verdict. Wildcard deliberately has no function here
# — it stays flag-only per Rule #24, handled by wildcard_flag() above.
# ---------------------------------------------------------------------------
def evaluate_bench_boost(bench_df: pd.DataFrame, gw_list: list[int], moe_fn) -> dict:
    """Scans the horizon for the single best Bench Boost gameweek: sums the
    bench's projected xPts per GW, and only returns a play-GW{n} verdict if
    that GW's bench total clears the best-vs-runner-up gap by more than
    Rule #34's margin-of-error — otherwise the horizon is a statistical tie
    and there's no genuine best week to call yet ('hold')."""
    if bench_df is None or bench_df.empty or not gw_list:
        return {"verdict": "hold", "best_gw": None, "by_gw": {}}
    by_gw = {}
    for gw in gw_list:
        col = f"xpts_gw{gw}"
        by_gw[gw] = round(float(bench_df[col].sum()), 2) if col in bench_df.columns else 0.0
    best_gw = max(by_gw, key=by_gw.get)
    best_total = by_gw[best_gw]
    others = [v for g, v in by_gw.items() if g != best_gw]
    runner_up = max(others) if others else 0.0
    margin = round(best_total - runner_up, 2)
    threshold = round(moe_fn(best_total), 2)
    verdict = f"play_gw{best_gw}" if (best_total > 0 and margin >= threshold) else "hold"
    return {"verdict": verdict, "best_gw": best_gw, "by_gw": by_gw, "margin": margin, "threshold": threshold}


def evaluate_triple_captain(starters_df: pd.DataFrame, gw_list: list[int], moe_fn) -> dict:
    """Same horizon-scan logic as evaluate_bench_boost(), applied to the
    single best individual starter's projected xPts per GW (a Triple
    Captain's value is entirely about that one player's best week, not the
    squad total)."""
    if starters_df is None or starters_df.empty or not gw_list:
        return {"verdict": "hold", "best_gw": None, "by_gw": {}}
    by_gw = {}
    for gw in gw_list:
        col = f"xpts_gw{gw}"
        by_gw[gw] = round(float(starters_df[col].max()), 2) if col in starters_df.columns and not starters_df[col].isna().all() else 0.0
    best_gw = max(by_gw, key=by_gw.get)
    best_total = by_gw[best_gw]
    others = [v for g, v in by_gw.items() if g != best_gw]
    runner_up = max(others) if others else 0.0
    margin = round(best_total - runner_up, 2)
    threshold = round(moe_fn(best_total), 2)
    verdict = f"play_gw{best_gw}" if (best_total > 0 and margin >= threshold) else "hold"
    return {"verdict": verdict, "best_gw": best_gw, "by_gw": by_gw, "margin": margin, "threshold": threshold}


def evaluate_free_hit(squad_df: pd.DataFrame, gw_list: list[int], rebuild_fn, moe_fn) -> dict:
    """Standing Rule #25: Free Hit is only ever evaluated as a full 15-man
    rebuild against total team value, compared against the manager's OWN
    best starting XI for that single gameweek (Horizon-Matching Rule — never
    the multi-GW horizon sum, since the Free Hit squad reverts after one
    week). rebuild_fn(gw) must return a data_pipeline.solve_free_hit_rebuild()
    result or None (e.g. if the solver has no feasible squad that week).
    A play-GW{n} verdict requires the rebuild to beat the manager's own best
    week by more than Rule #34's margin-of-error — otherwise 'hold'."""
    if squad_df is None or squad_df.empty or not gw_list:
        return {"verdict": "hold", "best_gw": None, "by_gw": {}}
    by_gw = {}
    for gw in gw_list:
        col = f"xpts_gw{gw}"
        if col not in squad_df.columns:
            continue
        current_best = opt.best_starting_xi(squad_df, col)
        current_total = current_best["total"] if current_best else 0.0
        rebuild = rebuild_fn(gw)
        if not rebuild or rebuild.get("squad") is None or rebuild["squad"].empty:
            continue
        rebuild_best = opt.best_starting_xi(rebuild["squad"], col)
        rebuild_total = rebuild_best["total"] if rebuild_best else rebuild["total_xpts"]
        by_gw[gw] = {"current": round(float(current_total), 2), "rebuild": round(float(rebuild_total), 2),
                     "gap": round(float(rebuild_total) - float(current_total), 2)}
    if not by_gw:
        return {"verdict": "hold", "best_gw": None, "by_gw": {}}
    best_gw = max(by_gw, key=lambda g: by_gw[g]["gap"])
    gap = by_gw[best_gw]["gap"]
    threshold = round(moe_fn(by_gw[best_gw]["rebuild"]), 2)
    verdict = f"play_gw{best_gw}" if gap >= threshold else "hold"
    return {"verdict": verdict, "best_gw": best_gw, "by_gw": by_gw, "threshold": threshold}
