"""
chip_protocol.py
Step 9 — Chip Timing Protocol (FPL Projection Model v4.0).

Mechanical pieces (chip status tracking, DGW/BGW detection, and — since
Patch 30 — the v6.4 Wildcard trigger itself) plus one piece that stays
deliberately non-mechanical: the specific GW a Wildcard actually gets
played on, which stays a rolling re-test rather than a one-week pick, per
Standing Rule #32 (Dynamic Chip Timing Rule — "a planned chip date is a
working hypothesis, not a fixed commitment"). Patch 30 corrected an
app-code mislabeling that had cited Standing Rule #24 (the Transfer Timing
Discipline Rule — about ordinary transfers, not Wildcard at all) as the
reason Wildcard stayed flag-only; #32 is the actual governing rule, and it
never forbade computing the trigger condition mechanically, only forbade
treating a chosen date as locked in.
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
    """{gw: {team_id: fixture_count} | None} for the given gameweeks — 0 =
    blank, 2+ = double. Mechanical, from official fixtures data (Standing
    Rule #17: only trust this a few gameweeks out, not a season-long
    projection).

    A gw with NO fixture rows at all gets `None`, not `{}` — the official
    fixtures feed can legitimately not have that round published yet (a GW
    well beyond the currently-scheduled fixture list), and that's a
    different situation from a GW that IS published but where a specific
    team genuinely has zero matches (a real blank). Collapsing both into the
    same empty dict is what silently made blank detection impossible in
    dgw_bgw_flags() below (fixed same patch, 2026-09-14) — every absent team
    read as "assume 1, normal week" instead of "this team blanks.\""""
    out: dict[int, dict | None] = {gw: None for gw in gw_list}
    if fixtures is None or fixtures.empty or "event" not in fixtures.columns:
        return out
    for gw in gw_list:
        rows = fixtures[fixtures["event"] == gw]
        if rows.empty:
            continue  # this round isn't in the fixtures feed yet — stays None, not a false blank
        counts: dict[int, int] = {}
        for _, r in rows.iterrows():
            for tid in (r.get("team_h"), r.get("team_a")):
                if pd.notna(tid):
                    counts[int(tid)] = counts.get(int(tid), 0) + 1
        out[gw] = counts
    return out


def dgw_bgw_flags(fixture_counts: dict, all_team_ids: list[int]) -> dict:
    """{gw: {"doubles": [team_id,...], "blanks": [team_id,...]}}. A gw whose
    fixture_counts entry is None (round not yet published — see
    fixture_counts_by_team()) reports no doubles/blanks rather than guessing
    — there's no fixture data yet to mechanically confirm either."""
    out = {}
    for gw, counts in fixture_counts.items():
        if counts is None:
            out[gw] = {"doubles": [], "blanks": []}
            continue
        doubles = [t for t in all_team_ids if counts.get(t, 0) >= 2]
        blanks = [t for t in all_team_ids if counts.get(t, 0) == 0]
        out[gw] = {"doubles": doubles, "blanks": blanks}
    return out


def chip_advisor_gw_window(planning_gw: int, fixtures: pd.DataFrame, all_team_ids: list[int],
                            cfg: dict) -> dict:
    """Patch 32 (2026-09-14 manager report). The Bench Boost/Triple Captain/
    Free Hit "which GW" verdicts (evaluate_bench_boost/evaluate_triple_captain/
    evaluate_free_hit below) need their OWN independent scan window — reusing
    the sidebar's transfer-planning Horizon slider was the actual bug: at
    horizon=1 there's only ever one candidate GW, so "PLAY GW{n}" wasn't
    finding an optimal week, it was just confirming the sole option. The
    Horizon slider stays exactly as-is for transfer decisions; this builds a
    separate window sized by chip_advisor_horizon: in model_config.yaml
    (disclosed EST extension, same pattern as chip_advisor_thresholds/
    chip_shape_test — Rule #34 specifies the play/hold PROCEDURE, not a scan
    length).

    Auto-extends past the default window to the nearest confirmed Double or
    Blank gameweek (via dgw_bgw_flags(), same mechanical fixture-count source
    already used for the advisory notes elsewhere) — a DGW is definitionally
    the best Bench Boost/Triple Captain week and a BGW the biggest Free Hit
    case, so stopping short of a known one would blind the advisor to the
    real optimal GW. The extension is capped at max_extend_gws, since Free
    Hit's verdict costs one full MILP rebuild solve per candidate GW.

    Returns {"gw_list": [...], "default_end": gw, "extended_to": gw|None,
    "extended": bool, "nearest_event_gw": gw|None}."""
    cah_cfg = cfg.get("chip_advisor_horizon", {})
    default_gws = max(1, cah_cfg.get("default_gws", 8))
    max_gws = max(default_gws, cah_cfg.get("max_extend_gws", 16))
    base_end = planning_gw + default_gws - 1
    max_end = planning_gw + max_gws - 1

    nearest_event_gw = None
    if fixtures is not None and not fixtures.empty and all_team_ids:
        lookahead_gws = list(range(planning_gw, max_end + 1))
        fixture_counts = fixture_counts_by_team(fixtures, lookahead_gws)
        flags = dgw_bgw_flags(fixture_counts, all_team_ids)
        for gw in range(base_end + 1, max_end + 1):
            f = flags.get(gw, {})
            if f.get("doubles") or f.get("blanks"):
                nearest_event_gw = gw
                break

    extended = nearest_event_gw is not None
    extend_to = nearest_event_gw if extended else base_end
    gw_list = list(range(planning_gw, extend_to + 1))
    return {"gw_list": gw_list, "default_end": base_end,
            "extended_to": extend_to if extended else None,
            "extended": extended, "nearest_event_gw": nearest_event_gw}


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


def wildcard_trigger_flag(trigger: dict, rank_history: list[int] | None = None) -> str | None:
    """Patch 30 (2026-09-14) — REPLACES the old wildcard_flag(), which ran an
    ad hoc rank-decline + flagged-player-count heuristic that pre-dated v6.4
    and was never updated once the doc gave Wildcard a real numeric trigger.
    `trigger` is fpl_engine.wildcard_trigger_check()'s result — the doc's
    actual mechanical condition (average Team Rating % below ~78-80% OR a
    cumulative gap to the bounded-ceiling optimal of ~15+ xPts over the 3-4
    GW detection window). Returns None when `trigger["active"]` is False —
    no manufactured flag just because rank happened to dip or a bench player
    picked up a knock; those aren't in the doc's own trigger condition.

    `rank_history` is now supplementary COLOR ONLY, never a trigger input —
    included in the message when a genuine 2-GW worsening trend happens to
    coincide with an active trigger, since that's still true and relevant
    context, just no longer part of deciding whether the flag fires at all.

    Still a flag, not a play verdict — v6.4's 8-GW decay-weighted build
    horizon means Wildcard timing stays a rolling re-test (Standing Rule
    #32), never a single best-week pick the way Free Hit gets one."""
    if not trigger or not trigger.get("active"):
        return None
    rank_note = ""
    if rank_history and len(rank_history) >= 3:
        recent = rank_history[-3:]
        if recent[0] < recent[1] < recent[2]:
            rank_note = (f" Overall rank has also worsened for 2 consecutive gameweeks "
                          f"({recent[0]:,} -> {recent[1]:,} -> {recent[2]:,}), for context.")
    return (f"Wildcard trigger ACTIVE (v6.4 mechanical condition): {trigger['reason']}.{rank_note} "
            f"This is a genuine trigger, not a play verdict — the specific GW to actually play it stays your "
            f"call, re-tested every run (Standing Rule #32).")


# ---------------------------------------------------------------------------
# Chip Advisor (v5.0 addition) — quantified play/hold verdicts for Bench
# Boost, Triple Captain and Free Hit within the manager's chosen horizon,
# gated by Standing Rule #34's margin-of-error threshold so a difference
# inside demonstrated weekly noise is reported as a tie ("hold"), never
# dressed up as a clear verdict. Wildcard's own trigger CONDITION is now
# mechanical too (Patch 30, wildcard_trigger_check()) — what stays
# deliberately non-mechanical here is which single GW to actually play it
# on, per Standing Rule #32, so it still isn't part of this "which GW"
# advisor block.
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


def evaluate_wildcard_whatif(current_squad_future_proj: pd.DataFrame, future_pool_proj: pd.DataFrame,
                              cfg: dict, team_value: float, future_gw_list: list[int]) -> dict:
    """Manager-directed what-if (Patch 6): "what if I played my Wildcard at
    GW{n}?" — on-demand version of Step 8c's own reactive-layer mechanic
    (recompute the gap between the current squad's trajectory and a
    freshly-solved rebuild at a candidate date), run against a manager-
    chosen date instead of only the model's currently-planned one.

    Standing Rule #32 applies in full: for a MANAGER-CHOSEN candidate date
    like this one, the caller must present the gap as a disclosed number for
    the manager's own judgment, never as "play"/"hold" wording (that framing
    is reserved for the Chip Advisor's Bench Boost/Triple Captain/Free Hit
    verdicts, which are genuinely mechanical per Rule #34). This is
    separate from wildcard_trigger_check()'s own ACTIVE/not-active
    condition (Patch 30) — that one IS mechanical, it just never names a
    date either, for the same Rule #32 reason.

    Unlike Free Hit (Standing Rule #25), a Wildcard squad does NOT revert —
    so the rebuild is optimized on the full `future_gw_list` horizon sum,
    never a single GW, and compared against the CURRENT squad's own
    realized value (Rule #12) over that identical window (i.e. "if you
    changed nothing between now and then and just held").

    `current_squad_future_proj`/`future_pool_proj`: the current squad and
    remaining pool re-projected onto `future_gw_list` (a different window
    than the manager's default horizon) — the caller must re-run the
    projection pipeline for that window first; this function only solves
    and compares, it doesn't re-project.
    `team_value`: bank + current squad's sell-value proxy — Rule #25's full-
    rebuild mechanic, same total-value basis as Free Hit, just non-
    reverting."""
    empty = {"feasible": False, "hold_total": 0.0, "rebuild_total": 0.0, "gap": 0.0}
    if current_squad_future_proj is None or current_squad_future_proj.empty or not future_gw_list:
        return empty
    hold_total = opt.realized_horizon_value(current_squad_future_proj, future_gw_list, cfg)
    full_pool = pd.concat([current_squad_future_proj, future_pool_proj], ignore_index=True, sort=False)
    if "code" in full_pool.columns:
        full_pool = full_pool.drop_duplicates(subset=["code"], keep="first")
    rebuild = opt.solve_squad(full_pool, cfg, budget=team_value, objective_col="xpts_horizon_sum")
    if rebuild is None or rebuild.get("squad") is None or rebuild["squad"].empty:
        return {**empty, "hold_total": round(hold_total, 2)}
    rebuild_total = opt.realized_horizon_value(rebuild["squad"], future_gw_list, cfg)
    gap = round(rebuild_total - hold_total, 2)
    return {"feasible": True, "hold_total": round(hold_total, 2), "rebuild_total": round(rebuild_total, 2),
            "gap": gap, "rebuild_squad": rebuild["squad"]}


# ---------------------------------------------------------------------------
# Step 8c shape-test (v6.4) — Wildcard answers a structural gap (the squad
# is behind the optimal team persistently, regardless of fixtures); Free Hit
# answers a fixture-shaped gap (fine on average, one week diverges sharply
# then reverts). Mandatory before either chip's size threshold, per the doc,
# so a sharp one-week spike is never misread as sustained Wildcard evidence.
# ---------------------------------------------------------------------------
def wildcard_freehit_shape_test(current_squad: pd.DataFrame, full_pool: pd.DataFrame, cfg: dict,
                                 detection_gw_list: list[int], team_value: float) -> dict:
    """Solves the standalone single-GW-optimal squad (unconstrained — no
    retain-pool tie to the current 15, since the question is "what does the
    ideal team look like this week," not "what's reachable") for every GW in
    the detection window, and measures its overlap with the current squad
    (shared player codes / 15) each week — a direct read on "how far is the
    current squad's shape from that week's optimal shape."

    Classification:
    - "wildcard_shaped": overlap stays at/below `structural_overlap_ceiling`
      in EVERY detection-window GW — the gap doesn't revert on its own, a
      genuine persistent structural mismatch.
    - "freehit_shaped": one or two GWs' overlap drops to/below that same
      ceiling while the OTHER GWs in the window sit at/above
      `spike_overlap_floor` (i.e. close to the current squad the rest of the
      time) — a sharp, reverting, fixture-shaped divergence, naming which
      GW(s) spiked.
    - "no_signal": neither pattern clears its threshold — not useful evidence
      either way this run.
    - "insufficient_data": couldn't solve enough of the window to classify
      (missing projections / solver infeasible).

    `structural_overlap_ceiling` / `spike_overlap_floor` (model_config.yaml
    `chip_shape_test:`) are a disclosed, manager-directed EST extension — the
    doc names the shape-test procedure but gives no numeric overlap
    threshold, same pattern as `chip_advisor_thresholds` and the Team-
    Stability gaps. Returns {"classification": str, "by_gw": {gw: overlap},
    "spike_gws": [gw,...], "notes": [str,...]} — not itself the Wildcard
    trigger (that's wildcard_trigger_check(), Patch 30); this is a
    cross-check that decides WHICH chip a fired trigger actually points to,
    Wildcard or Free Hit."""
    import optimizer as opt

    shape_cfg = cfg.get("chip_shape_test", {})
    structural_ceiling = shape_cfg.get("structural_overlap_ceiling", 0.7)
    spike_floor = shape_cfg.get("spike_overlap_floor", 0.85)
    empty = {"classification": "insufficient_data", "by_gw": {}, "spike_gws": [], "notes": []}

    if current_squad is None or current_squad.empty or not detection_gw_list or full_pool is None:
        return empty
    current_codes = set(current_squad["code"]) if "code" in current_squad.columns else set()
    if not current_codes:
        return empty

    by_gw = {}
    for gw in detection_gw_list:
        col = f"xpts_gw{gw}"
        if col not in full_pool.columns:
            continue
        result = opt.solve_squad(full_pool, cfg, budget=team_value, objective_col=col)
        if result is None or result.get("squad") is None or result["squad"].empty:
            continue
        optimal_codes = set(result["squad"]["code"])
        by_gw[gw] = round(len(current_codes & optimal_codes) / 15.0, 3)

    if len(by_gw) < 2:
        return {**empty, "by_gw": by_gw}

    overlaps = list(by_gw.values())
    if all(v <= structural_ceiling for v in overlaps):
        notes = [f"Shape-test (Step 8c, v6.4): the squad-vs-optimal gap holds at/below "
                 f"{structural_ceiling:.0%} overlap across the whole GW{min(by_gw)}–GW{max(by_gw)} "
                 f"detection window — a persistent, structural gap. Wildcard-shaped, not a fixture spike."]
        return {"classification": "wildcard_shaped", "by_gw": by_gw, "spike_gws": [], "notes": notes}

    spike_gws = []
    for gw, v in by_gw.items():
        others = [ov for g, ov in by_gw.items() if g != gw]
        if v <= structural_ceiling and others and min(others) >= spike_floor:
            spike_gws.append(gw)

    if spike_gws:
        notes = [f"Shape-test (Step 8c, v6.4): overlap with the current squad drops sharply only at "
                 f"{', '.join(f'GW{g}' for g in spike_gws)} (≤{structural_ceiling:.0%}) while the rest of "
                 f"the GW{min(by_gw)}–GW{max(by_gw)} window sits at/above {spike_floor:.0%} — a one-off, "
                 f"reverting, fixture-shaped gap. Free-Hit-shaped, not sustained Wildcard evidence."]
        return {"classification": "freehit_shaped", "by_gw": by_gw, "spike_gws": spike_gws, "notes": notes}

    return {"classification": "no_signal", "by_gw": by_gw, "spike_gws": [],
            "notes": [f"Shape-test (Step 8c, v6.4): overlap across GW{min(by_gw)}–GW{max(by_gw)} "
                      f"({', '.join(f'GW{g}:{v:.0%}' for g, v in sorted(by_gw.items()))}) clears neither the "
                      f"structural nor the spike pattern this run — no clean Wildcard/Free Hit shape signal yet."]}


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


# ---------------------------------------------------------------------------
# Chip Strategy — combined view (Patch 29, 2026-09-14 manager request). A
# PRESENTATION-LAYER SYNTHESIS ONLY: every number here was already computed
# by wildcard_trigger_check()/wildcard_trigger_flag()/wildcard_freehit_shape_
# test()/evaluate_bench_boost()/evaluate_triple_captain()/evaluate_free_hit()
# above — this function adds no new math, it reads those results together
# and orders them into one narrative, because the manager found the
# individual pieces scattered across the page hard to act on together.
#
# Standing Rule #32 still applies in full: Wildcard's trigger CONDITION is
# genuinely mechanical since Patch 30 (unlike Bench Boost/Triple Captain/
# Free Hit, it never resolves to one specific "play GW{n}" — v6.4's 8-GW
# decay-weighted build horizon keeps the actual date a rolling re-test, per
# Rule #32). Wildcard appears here as that trigger's own message (already
# phrased as ACTIVE/not, never "play") plus its shape-test cross-reference.
# ---------------------------------------------------------------------------
def chip_strategy_summary(wc_flag: str | None, shape_test: dict | None,
                           bb_advisor: dict | None, tc_advisor: dict | None, fh_advisor: dict | None,
                           chip_rows: list[dict], disruption_notes: list[str] | None = None,
                           wc_trigger: dict | None = None) -> list[str]:
    """Returns an ordered list of plain-language strategy lines for a single
    combined "Chip Strategy" panel. Section order: (1) near-term mechanical
    plays (BB/TC/FH verdicts that actually fired, soonest GW first) — these
    are the closest thing to a genuine "do this" this tool ever gives; (2)
    Wildcard's own v6.4 mechanical trigger (Patch 30 — `wc_flag` here is
    wildcard_trigger_flag()'s output, already stating the real Team
    Rating %/cumulative-gap numbers, not the old rank/flagged-count
    heuristic), cross-referenced against the shape-test so its Wildcard-vs-
    Free-Hit caveat travels with it; (3) a same-week chip-clash caution when
    a mechanical play verdict and an active Wildcard trigger land in a way
    that would waste one of them; (4) the v6.4 chip-chaining reminder (bench
    strength for Bench Boost, a nailed premium's run for Triple Captain) —
    this tool can't compute that reminder's actual numbers outside a real
    Wildcard-build scenario, so it points at the "Evaluate a scenario" panel
    rather than fabricating a check.

    `disruption_notes` is accepted for backward-compatible call signatures
    but, since Patch 52 (2026-09-16, manager decision: chip-rack pill
    tooltip is the one place these now live, not both there AND here),
    deliberately no longer echoed into `lines` — doing so used to reprint
    the exact same disruption-note strings the Chip Rack's own pills
    already show (Patch 51 made those pills full-sentence too, which is
    the regression Patch 52 fixes at the pill level; this was the matching
    duplicate at the expander level). Sections (1)-(4) below are
    independently-composed synthesis prose, not verbatim repeats of any
    other panel's text, and are unchanged."""
    lines = []

    # (1) Near-term mechanical plays, soonest first.
    plays = []
    for name, adv in (("Bench Boost", bb_advisor), ("Triple Captain", tc_advisor), ("Free Hit", fh_advisor)):
        if adv and isinstance(adv.get("verdict"), str) and adv["verdict"].startswith("play_gw"):
            gw = adv.get("best_gw")
            plays.append((gw, name))
    plays.sort(key=lambda t: (t[0] is None, t[0]))
    if plays:
        play_bits = "; ".join(f"{name} at GW{gw}" for gw, name in plays)
        lines.append(f"Near-term plays this horizon, soonest first: {play_bits}. These are genuinely mechanical "
                      f"verdicts (Rule #34) — the closest thing to a \"do this\" call this tool makes.")
    else:
        lines.append("No Bench Boost / Triple Captain / Free Hit play verdict fired this run — everything still "
                      "available is a statistical hold for now (see Chip Advisor above for each one's own margin).")

    # (2) Wildcard — its own v6.4 mechanical trigger (Patch 30), cross-
    # referenced with the shape-test. Explicit "no trigger" line so its
    # absence isn't mistaken for "not checked."
    if wc_flag:
        wc_line = f"Wildcard: {wc_flag}"
        if shape_test and shape_test.get("classification") == "wildcard_shaped":
            wc_line += (" The shape-test backs this up as a persistent, structural gap, not a fixture spike — "
                        "the case for a Wildcard (over a one-week Free Hit) builds the longer this holds.")
        elif shape_test and shape_test.get("classification") == "freehit_shaped":
            spike = ", ".join(f"GW{g}" for g in shape_test.get("spike_gws", []))
            wc_line += (f" CAUTION — the shape-test reads this as fixture-shaped, concentrated at {spike}, not a "
                        f"sustained gap: worth checking whether Free Hit covers it before committing a Wildcard "
                        f"here (Step 8c).")
        elif shape_test and shape_test.get("classification") == "no_signal":
            wc_line += " The shape-test found no clean structural-vs-spike pattern yet either way this run."
        lines.append(wc_line)
    elif wc_trigger and wc_trigger.get("avg_rating_pct") is not None:
        lines.append(f"Wildcard: no trigger this run — {wc_trigger['reason']}.")
    else:
        lines.append("Wildcard: no trigger this run (insufficient data to evaluate — see Chip Rack for detail).")

    # (3) Same-week clash caution — a mechanical play verdict landing on the
    # same GW an active Wildcard flag is live is worth a plain heads-up,
    # since a Wildcard resets the whole squad and would make that separate
    # chip's build largely redundant that week.
    if wc_flag and plays:
        clash_gws = {gw for gw, _ in plays}
        if clash_gws:
            lines.append(f"If you're weighing playing the Wildcard around GW{min(clash_gws)}–GW{max(clash_gws)}, "
                          f"note the mechanical play verdict(s) above land in that same window — playing both "
                          f"the same week is redundant (a Wildcard already rebuilds everything). Sequencing "
                          f"between them is your call, not something this tool decides.")

    # (4) Chip-chaining reminder (v6.4) — genuinely can't be computed here
    # (needs an actual built Wildcard squad), so this points at the tool that
    # can rather than fabricating a check.
    wc_available = any(r["status"] == "available" and r["chip"].startswith("Wildcard") for r in (chip_rows or []))
    if wc_flag and wc_available:
        lines.append("Chip-chaining checklist (Step 8c, v6.4) — before playing a Wildcard, check its build's "
                      "bench strength (does it make a near-term Bench Boost live?) and captain ceiling (does it "
                      "include a nailed premium worth a near-term Triple Captain?). Run \"Evaluate a scenario\" "
                      "for a candidate Wildcard GW below to see the actual build and check both.")

    # (5) Disruption notes (Rule #41) — REMOVED Patch 52 (2026-09-16, manager
    # decision: tooltip only, drop from the expander). This used to
    # `lines.append(note)` for every note in `disruption_notes`, verbatim —
    # the exact same strings the Chip Rack pills above already carry, now
    # each with its own short headline + full-text tooltip (Patch 52's Fix
    # 1). Kept as a no-op comment rather than deleted silently so the
    # section numbering above ((1)-(4)) still matches what's actually in
    # this function. `disruption_notes` is still accepted as a parameter
    # (call sites unchanged) but is intentionally unused now.

    return lines
