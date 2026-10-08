"""
optimizer.py
Constrained squad solver (Step 7): maximizes horizon xPts subject to
£100m budget, 2-5-5-3 squad shape, max 3 players per club — a real MILP,
not a greedy heuristic, because that's what the formation/budget/club
constraints actually require.

Also used to produce Ceiling_xPts for §1a's Team Rating %.

Free & open-source: PuLP with its bundled CBC solver, no license, no cost.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None

try:
    import streamlit as st
except ImportError:  # pragma: no cover — keeps this module importable from a
    # plain, non-Streamlit test script even if streamlit somehow isn't
    # installed in that environment; falls back to an uncached passthrough.
    st = None


# -----------------------------------------------------------------------------
# Patch 76 (2026-09-27) — ROOT CAUSE CONFIRMED, not inferred. The manager's
# deployed traceback ("TypeError: LpVariable.__init__() got an unexpected
# keyword argument 'cat'") is the real cause behind every "—"/"N/A" this
# whole thread has been chasing. Patch 70/71's try/except was correctly
# catching it and degrading to None all along; Patch 72-75's diagnostics
# just finally made it visible.
#
# What actually happened, confirmed by directly installing and running the
# real releases (not guessed from behavior): this sandbox's Python (3.11)
# can only ever be offered pulp up to 3.3.2, where `cat=` works fine — which
# is exactly why this was invisible from here at first. The manager's
# Streamlit Cloud traceback showed their real venv path running Python
# 3.14, and PuLP shipped an actual, non-alpha PuLP 4.0.0 requiring Python
# >=3.12. requirements.txt's unpinned `pulp>=2.7` let Streamlit Cloud's
# newer Python runtime install it. Built a real Python 3.12 venv here,
# installed genuine pulp==4.0.0, and reproduced the manager's exact error —
# then reproduced their SECOND error too ("'str' object has no attribute
# 'set_lb'") from an earlier, wrong attempt at this fix that tried to patch
# around it with post-construction attribute assignment.
#
# PuLP 4.0 is not a small API tweak: it's a full rewrite onto a Rust-backed
# core (`LpVariable.__init__` now takes a `_rustcore.Variable`, not a name/
# bounds/cat at all) — exactly what 3.3.2's own deprecation notice had
# already been telegraphing ("Constructing LpVariable(name, ...) directly
# is deprecated; in PuLP 4.0 use prob.add_variable(...)"). No amount of
# attribute patching can bridge a rewritten core; the only real fix is to
# use the actual documented, non-deprecated construction path. Confirmed
# directly: `prob.add_variable(name, lowBound=0, upBound=1, cat="Binary")`
# has the IDENTICAL signature and produces a fully usable variable on both
# a real pulp 3.3.2 install AND a real pulp 4.0.0 install — this is the one
# call that's actually correct on both, not a version branch or a guess.
#
# A second, separate 4.0 break was found the same way (running the real
# solve end-to-end under real 4.0.0, not stopping at the first fix that
# looked plausible): `pulp.PULP_CBC_CMD` — used to invoke the solver — no
# longer exists in 4.0's public API at all (`pulp.listSolvers()` doesn't
# even list it), and no MIP solver binary ships bundled by default anymore
# (`pulp.listSolvers(onlyAvailable=True)` returns `[]` on a plain `pip
# install pulp==4.0.0`). Confirmed the fix: PyPI's `pulp[cbc]` extra installs
# the same underlying CBC binary (`pulp.COIN_CMD`) 4.0 needs, is a no-op-safe
# addition to requirements.txt for the pre-4.0 range too (tested `pulp[cbc]
# ==3.3.2` directly — behaves identically to plain `pulp==3.3.2`), and keeps
# every existing "CBC status=..." diagnostic string in this file accurate,
# since it's still genuinely CBC underneath either way — not a switch to a
# different solver engine.
def _binary_var(prob, name: str):
    """Patch 76 (revised -- see the model_config.yaml Patch 76 note for the
    full story, including the two earlier attempts at this fix that didn't
    hold up under real testing). The manager's live traceback turned out to
    be caused by a genuine PuLP 4.0.0 install (confirmed directly: installed
    real pulp==4.0.0 under a real Python 3.12 venv in this sandbox and
    reproduced the manager's EXACT two errors in sequence -- first
    "LpVariable.__init__() got an unexpected keyword argument 'cat'" from
    the direct-construction call this function used to make, then, from the
    first attempted fallback, "'str' object has no attribute 'set_lb'").
    PuLP 4.0 is not a small API tweak -- it's a full rewrite with a Rust-
    backed core (`LpVariable.__init__` now takes a `_rustcore.Variable`
    object, not a name/bounds/cat at all), which is exactly what pulp
    3.3.2's own deprecation warning had been telegraphing: "Constructing
    LpVariable(name, ...) directly is deprecated; in PuLP 4.0 use
    prob.add_variable(name, lowBound, upBound, cat=...)." No amount of
    post-construction attribute patching can bridge that -- the fix is to
    actually use the documented, non-deprecated construction path.
    Confirmed directly (not assumed) that `prob.add_variable(name,
    lowBound=0, upBound=1, cat="Binary")` has the IDENTICAL signature and
    produces a fully usable variable on both a real pulp 3.3.2 install and a
    real pulp 4.0.0 install -- this is the one call that's actually correct
    on both, not a version-detection branch or a fallback guess."""
    return prob.add_variable(name, lowBound=0, upBound=1, cat="Binary")


def _pulp_forensics() -> str:
    """One-line environment forensics appended to any unexpected-exception
    diagnostic (Patch 76) -- if pulp itself is somehow still the culprit in a
    way _binary_var()'s fallback doesn't catch, this tells us (from the
    manager's own screenshot, without needing server-log access) exactly
    which pulp build/location produced it, instead of leaving that as a
    guess for next time."""
    try:
        return f"[pulp {getattr(pulp, '__version__', '?')} @ {getattr(pulp, '__file__', '?')}]"
    except Exception:
        return "[pulp version/location unavailable]"


_DEFAULT_SOLVE_TIME_LIMIT_SECONDS = 12  # kept in sync with model_config.yaml's solver.time_limit_seconds;
                                          # used only if a caller can't reach cfg for some reason.


def _cbc_solver(msg: int = 0, time_limit: float | None = _DEFAULT_SOLVE_TIME_LIMIT_SECONDS):
    """Patch 76 -- version-tolerant replacement for the bare
    `pulp.PULP_CBC_CMD(msg=0)` call this file used to make directly at every
    `prob.solve(...)` site. `PULP_CBC_CMD` doesn't exist in pulp 4.0's public
    API at all (confirmed: `hasattr(pulp, "PULP_CBC_CMD")` is False on a real
    4.0.0 install) -- `COIN_CMD` is the equivalent there, and requires the
    `pulp[cbc]` extra (now in requirements.txt) to have an actual CBC binary
    to call. On pulp <4.0, PULP_CBC_CMD remains the right, always-bundled
    choice, so this only takes the COIN_CMD path when it has to.

    gapRel=0/gapAbs=0 on the COIN_CMD path: found by actually running a
    real solve against a real pulp[cbc]==4.0.0 install (not assumed) --
    the CBC binary that ships via 4.0's `[cbc]` extra (a different bundled
    build, `cbcbox`, than pre-4.0's own internal one) defaults to stopping
    at a nonzero optimality gap, reporting status "GapLimit" instead of
    "Optimal" even on trivially small problems (confirmed directly: a
    239-variable knapsack-style problem that should solve to exact
    optimality in milliseconds reported "GapLimit" with default options,
    and "Optimal" once gapRel=0/gapAbs=0 were passed explicitly). Every
    diagnostic and caller in this file already checks for the literal
    string "Optimal" -- rather than loosen every one of those checks to
    also accept "GapLimit" (silently accepting a solution that's merely
    close to optimal, not the genuine best XI/squad the manager is relying
    on this app for), this forces the same exact-optimal behavior pre-4.0's
    bundled CBC always gave by default. Left off the PULP_CBC_CMD path
    since that one was never observed to need it.

    `time_limit` (2026-09-28 investigation, manager report: Horizon=3 with
    "Hit if worth it" hung on "Fetching live data..." indefinitely, then
    Streamlit Cloud killed the whole session -- see model_config.yaml's
    `solver.time_limit_seconds` comment for the full root-cause writeup).
    Every solve in this file previously had NO wall-clock cap at all, so a
    single genuinely hard MILP instance (this app's own retain-pool-
    constrained solves are already confirmed, via the Patch 78 GapLimit
    story, capable of being non-trivial) could run indefinitely, and
    `plan_transfer_schedule()`'s per-week `k in range(...)` loop multiplies
    that exposure by the Horizon slider's GW count. Passing a `timeLimit` to
    the underlying solver bounds any ONE solve's worst-case wall-clock time.
    Confirmed empirically (not assumed) on both pulp builds this app's
    requirements.txt allows that a solve cut short by the time limit before
    proving optimality reports a status that is neither "Optimal" nor the
    disclosed-safe "GapLimit" exact-optimum case -- pulp 3.3.2's
    PULP_CBC_CMD path returns "Not Solved", pulp 4.0.0's COIN_CMD path
    returns "TimeLimit" -- so it flows straight through the existing
    `if _status != "Optimal": ... return None` handling every other
    non-Optimal status already uses. A time-limited, unproven-optimal
    solution is therefore NEVER silently served as if it were the genuine
    best squad; worst case, that one candidate is skipped, exactly like an
    infeasible one already is. `time_limit=None` restores the old
    unbounded behavior (kept as an explicit opt-out for tests that need to
    force a specific status)."""
    if hasattr(pulp, "PULP_CBC_CMD"):
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    return pulp.COIN_CMD(msg=msg, gapRel=0, gapAbs=0, timeLimit=time_limit)


def parallel_workers(n_tasks: int, cap: int = 4) -> int:
    """Patch 115 fix: how many CBC subprocess solves may run side by side -- never more than the tasks, the cap (4) or the
    CPU count (a 2-core host running four heavy solves at once slowed each past the 12 s solver cap, so every squad was dropped)."""
    import os as _os
    return max(1, min(int(cap), int(n_tasks), int(_os.cpu_count() or 1)))


def _solve_time_limit(cfg: dict) -> float | None:
    """Reads `solver.time_limit_seconds` from model_config.yaml (see that
    file's comment for the full 2026-09-28 investigation this backs), falling
    back to `_DEFAULT_SOLVE_TIME_LIMIT_SECONDS` if the key is absent (e.g. an
    older config file) so this never hard-fails on a missing key."""
    if not isinstance(cfg, dict):
        return _DEFAULT_SOLVE_TIME_LIMIT_SECONDS
    return cfg.get("solver", {}).get("time_limit_seconds", _DEFAULT_SOLVE_TIME_LIMIT_SECONDS)


def _solve_and_get_status(prob, solver) -> str:
    """Patch 76 -- solves `prob` and returns the human-readable status string
    ("Optimal", "Infeasible", "Unbounded", ...) across both pulp APIs.
    Pre-4.0 (confirmed against a real 3.3.2 install): `prob.solve(solver)`
    returns a plain int status code (also stored on `prob.status`), decoded
    via `pulp.LpStatus[prob.status]`. 4.0+ (confirmed against a real 4.0.0
    install): `LpProblem` no longer has a `.status` attribute at all --
    `prob.solve(solver)` instead returns an `LpSolveStats` object whose
    `.status` is an enum with a `.name` equal to the same string ("Optimal",
    etc.). Every CBC-status diagnostic string elsewhere in this file already
    expects exactly this string, unchanged either way.

    Patch 78 (manager, live screenshot testing a different team: the
    Wildcard card showed "CBC solver returned status=GapLimit (not Optimal)
    ... no legal squad exists" on a genuinely solvable 343-candidate,
    14-of-15-retain problem -- Patch 77's gapRel=0/gapAbs=0 fix was already
    live (header showed Patch 77) and did NOT stop this. Root-caused by
    reading pulp 4.0.0's OWN source directly (not assumed):
    apis/coin.py's `get_status()` maps CBC's solution-file first line
    "Optimal (within gap tolerance)" to `LpSolveStatus.GapLimit`, and only a
    BARE "Optimal" (no suffix) maps to `LpSolveStatus.Optimal` -- entirely a
    label for WHICH CODE PATH CBC took to declare termination (gap check vs.
    full branch-and-bound tree exhaustion), independent of the actual
    tolerance value used. Reproduced directly: 30/30 easy 600-candidate
    unconstrained solves returned bare "Optimal", but harder/more
    constrained problems (confirmed via the manager's own live 343-candidate
    retain-constrained case) can legitimately terminate via the gap-check
    path even at gapRel=0/gapAbs=0 and get the "(within gap tolerance)"
    phrasing -- CBC's internal choice, not evidence of a worse-quality
    solution. Since `_cbc_solver()` (above) ALWAYS passes gapRel=0,
    gapAbs=0, a "GapLimit" status from our own solver call can only mean
    "terminated via the zero-tolerance gap check" -- mathematically
    indistinguishable from a proven exact optimum (a 0 gap between the
    incumbent and the LP bound already IS the definition of optimal; CBC
    just took the gap-check exit door instead of the tree-exhaustion one to
    get there). Normalized here rather than reproducing this same reasoning
    at all 3 call sites separately."""
    result = prob.solve(solver)
    if hasattr(result, "status") and hasattr(result.status, "name"):
        status = result.status.name
    else:
        status = pulp.LpStatus[prob.status]
    if status == "GapLimit":
        return "Optimal"
    return status


# -----------------------------------------------------------------------------
# Patch 72 (manager, 2026-09-27: after Patch 71 was deployed, the crash was
# gone but "GW6 Rating", "Team Rating % (GW6-9)" and the Wildcard card all
# started showing "—" / "N/A — insufficient data this run" instead of real
# numbers). Traced directly in code: all three depend on solve_squad() and/or
# solve_xi_first_squad() returning a real result; when either returns None
# (empty candidate pool, an infeasible/non-Optimal CBC solve, or an exception
# now caught by Patch 70/71's try/except), every caller downstream degrades
# to "—"/"N/A" exactly as designed — the crash-hardening is working. What it
# was NEVER able to do is explain WHY a given solve returned None on the
# manager's actual live data, because Streamlit Cloud redacts on-page errors
# and this sandbox has no network route to the live FPL API to reproduce the
# manager's exact GW6 pool. Every previous patch this session (70, 71) could
# only print the failure to stderr, which lands in Streamlit Cloud's "Manage
# app" logs — a place the manager has already reported difficulty finding.
#
# This module-level dict is a lightweight, in-process diagnostic channel:
# every None-return path in solve_squad()/solve_xi_first_squad() below now
# also records a plain-English reason here, keyed by an optional `label`
# each call site can pass (e.g. "team_rating_ceiling", "wildcard_reachable",
# "free_hit_optimal"). app.py reads it back with get_diagnostic(label)
# immediately after a None result and surfaces it directly in that card's
# own tooltip — so the NEXT real run against live data will show the actual
# cause (e.g. "CBC status: Infeasible" or "empty candidate pool: 0/612 rows
# survived filtering") right there in the UI, with no Streamlit Cloud log
# access needed at all. A successful solve clears its label's entry, so a
# stale reason is never shown once whatever caused it stops happening.
# -----------------------------------------------------------------------------
_DIAGNOSTICS: dict[str, str] = {}


def _diag(label: str | None, message: str) -> None:
    if label:
        _DIAGNOSTICS[label] = message


def _clear_diag(label: str | None) -> None:
    if label:
        _DIAGNOSTICS.pop(label, None)


def get_diagnostic(label: str) -> str | None:
    """Patch 72 — the most recent None-return reason recorded for this
    label, or None if that label's last call either succeeded or has never
    run. See the module-level _DIAGNOSTICS note above for why this exists."""
    return _DIAGNOSTICS.get(label)


def set_diagnostic(label: str, message: str) -> None:
    """Patch 72 — public wrapper around _diag(), for callers OUTSIDE this
    module (data_pipeline.py's solve_free_hit_optimal_squad/rebuild, which
    can return None before ever calling into optimizer.py at all, when the
    target GW's projection column isn't in `proj` yet) that still want their
    own None-return reason to show up wherever the caller reads
    get_diagnostic(label)."""
    _diag(label, message)


def _cache_decorator(fn):
    """Patch 42 (manager, 2026-09-15: "the performance is too too slow" —
    traced to Patch 39's own weekly-planner hit-cost extension, which made
    "Hit if worth it" run the full k=0-5 MILP search PER WEEK of the
    horizon instead of once, roughly tripling solver calls for a 3-GW run.
    Manager-confirmed fix (kept the full k-range rather than narrowing
    search power): cache solve_squad() itself, so re-running the exact same
    MILP problem — which happens constantly, since Streamlit reruns the
    whole script on nearly every widget interaction and many of those
    reruns don't actually change the pool/budget/retain-set — returns
    instantly instead of re-solving. Pure caching, zero change to what gets
    recommended: a cache hit requires byte-identical inputs (players'
    content, cfg's content, and every other argument), so a genuine data
    refresh or a genuinely different candidate search always gets a fresh
    solve. Falls back to uncached if streamlit isn't importable (e.g. a
    bare test script run without it installed) rather than breaking."""
    if st is None:
        return fn
    return st.cache_data(show_spinner=False, max_entries=1024)(fn)


@_cache_decorator
def solve_squad(players: pd.DataFrame, cfg: dict, budget: float = 100.0,
                 must_include_codes: list | None = None,
                 exclude_codes: list | None = None,
                 retain_pool_codes: list | None = None,
                 min_retain: int = 0,
                 objective_col: str = "xpts_horizon_sum",
                 label: str | None = None) -> dict | None:
    """players needs columns: code, web_name, team, position, price,
    <objective_col>, status. Returns dict with squad picks, total_xpts, cost.
    Only picks status=='a' (available) players unless explicitly must_include.

    objective_col lets the same solver serve two different Team Rating /
    Chip Advisor needs without duplicating the MILP: the default
    "xpts_horizon_sum" for a multi-GW ceiling, or a single "xpts_gw{n}"
    column for a Free Hit rebuild (Step 8b / Rule #25 — a Free Hit's squad
    reverts after one week, so it should never be optimized against a
    multi-GW horizon sum).

    retain_pool_codes + min_retain add a "keep at least N of these codes"
    constraint (>= not ==, so the solver can still improve on the retained
    core with its remaining slots) — this is what turns an unconstrained
    Ceiling into a reachable one: pass the current squad's codes and
    min_retain = 15 - available free transfers, and the solve becomes "the
    best squad actually reachable this week," not a fantasy ideal that
    ignores you already own 15 players and only have N free moves."""
    if pulp is None:
        _diag(label, "pulp (the MILP library this solver needs) is not installed/importable in this environment.")
        return None

    must_include_codes = must_include_codes or []
    exclude_codes = exclude_codes or []

    # Patch 10 (original bug) — dropping a row for a missing
    # objective_col/price BEFORE the retain-pool constraint is built means a
    # currently-owned player with an incomplete projection this run (a
    # sparse-minutes bench player is the classic case) silently vanishes
    # from the candidate set entirely — and the retain-pool constraint below
    # ('keep at least N of your squad') then recomputes its denominator from
    # whoever survived, quietly tightening to "keep ALL of the survivors."
    # That can use up the one transfer slot a k=1 request was supposed to
    # give the manager on a player they never asked to touch, then force an
    # unrelated second swap (and its hit cost) just to also fit in the swap
    # they actually wanted.
    #
    # Patch 10's first fix (filling the gap with 0.0 and leaving the player
    # freely tradeable) was ITSELF a real bug, confirmed live: a 0.0 reads to
    # the solver as "the single worst player in the entire pool," which is
    # an active INCENTIVE to swap him out — not neutral. That produced two
    # confirmed bad outputs: under "Hit if worth it," the solver happily
    # paid a hit to drop him alongside an unrelated, genuinely-wanted swap
    # (dropping a "0 xPts" player looks free); under "No hits" multi-week
    # pacing, it spent GW-now's free transfer swapping him out first,
    # pushing the manager's actual target to a later week for no reason.
    # Neither is a real judgment about him — both are an artifact of a
    # fabricated placeholder number silently driving a real decision, which
    # is exactly what Standing Rule #4 warns against (disclosing the gap via
    # `data_gap_codes` was not enough — the estimate itself still shaped the
    # recommendation).
    #
    # Fix (Patch 15): a currently-owned or must-include player with a
    # missing objective_col/price is never dropped AND never left freely
    # tradeable on a fabricated value — he is PINNED (forced to stay, same
    # mechanism as an explicit must_include) so the solver cannot select him
    # out for any reason this run, while remaining a normal, present row so
    # the retain-pool's true size (15, not "however many survived a drop")
    # is preserved. His price/objective are still filled with 0.0 purely so
    # the LP has a real number to work with — that number can no longer
    # influence whether he stays, only slightly understate the squad's
    # reported total (already covered by the `data_gap_codes` disclosure).
    # Net effect: the model simply declines to make any decision about a
    # player it can't currently project, in either direction, until his data
    # is actually available — never silently estimates one.
    players = players.copy()
    protected_codes = set(must_include_codes) | set(retain_pool_codes or [])
    data_gap_codes = []
    pinned_gap_codes = []
    if protected_codes and "code" in players.columns:
        protected_mask = players["code"].isin(protected_codes)
        gap_mask_any = pd.Series(False, index=players.index)
        for col in (objective_col, "price"):
            if col in players.columns:
                gap_mask_any = gap_mask_any | (protected_mask & players[col].isna())
        if gap_mask_any.any():
            gap_codes = players.loc[gap_mask_any, "code"].tolist()
            data_gap_codes.extend(gap_codes)
            pinned_gap_codes.extend(gap_codes)
            gap_rows_mask = players["code"].isin(gap_codes)
            for col in (objective_col, "price"):
                if col in players.columns:
                    players.loc[gap_rows_mask & players[col].isna(), col] = 0.0

    _players_in = len(players)
    df = players.dropna(subset=["price", objective_col, "position"]).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    if exclude_codes:
        df = df[~df["code"].isin(exclude_codes)]
    df = df[(df["status"] == "a") | (df["code"].isin(must_include_codes)) |
            (df["code"].isin(retain_pool_codes or []))]
    if df.empty:
        _diag(label, f"empty candidate pool after filtering: 0 of {_players_in} input rows survived the "
                      f"price/{objective_col}/position not-null check, GK/DEF/MID/FWD position check, exclude-list, "
                      f"and status=='a' (or must-include/retain-pool) filters — nothing left for the solver to "
                      f"choose from.")
        return None

    # Patch 70 (manager, live TypeError crash: "This app has encountered an
    # error", traceback pointing at this exact dict-comprehension line) —
    # two hardening fixes, applied together since the redacted Streamlit
    # Cloud error hid the actual exception message and this couldn't be
    # reproduced locally against synthetic data with the current pulp
    # version (3.3.2), so this addresses the two concrete risks code review
    # actually found here rather than guessing at one unconfirmed cause:
    #
    # 1. `df`'s index, at this point, is whatever survived dropna/position/
    #    exclude/status filtering from the CALLER's `players` frame — not
    #    guaranteed unique (a caller could hand in an already-concatenated
    #    or otherwise non-reset-index frame). A duplicate index label here
    #    is a real, confirmed-in-code latent bug: `df.loc[i, objective_col]`
    #    for a duplicated `i` returns a pandas Series (not a scalar), which
    #    a later line multiplies against an LpVariable — an operation pulp
    #    cannot know how to perform. Reset to a clean, guaranteed-unique
    #    RangeIndex here — nothing below this point depends on the index's
    #    original values, only on it being consistent with itself.
    df = df.reset_index(drop=True)

    # 2. Every other failure mode in this function (pulp missing, an empty
    #    candidate pool, an infeasible/non-Optimal solve) already returns
    #    None and every caller already handles that gracefully (`if
    #    reachable else 0.0`, etc.) — but an unexpected exception during
    #    the actual MILP construction/solve was NOT caught, so it propagated
    #    all the way up and crashed the entire page instead of degrading
    #    like every other failure path here already does. Wrapping this in
    #    try/except turns any such exception into the same graceful "this
    #    solve didn't work out this run" None-return every caller already
    #    expects, and prints the real exception (class + message + which
    #    solve this was) to stderr so it's visible in Streamlit Cloud's
    #    "Manage app" logs even though the user-facing page redacts it.
    try:
        prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
        x = {i: _binary_var(prob, f"x_{i}") for i in df.index}

        # Performance (measured, lossless): ~93% of this function's wall
        # time was Python-side model construction doing a pandas
        # `df.loc[i, col]` scalar lookup per player per constraint (tens of
        # thousands of lookups per solve), not CBC itself. The columns are
        # read ONCE into plain Python lists here; the model built is
        # mathematically identical (same coefficients, same constraints).
        _idx = list(df.index)
        _obj_vals = df[objective_col].tolist()
        _price_vals = df["price"].tolist()
        _pos_vals = df["position"].tolist()
        _team_vals = df["team"].tolist()

        prob += pulp.lpSum(x[i] * v for i, v in zip(_idx, _obj_vals))

        prob += pulp.lpSum(x[i] * v for i, v in zip(_idx, _price_vals)) <= budget
        prob += pulp.lpSum(x[i] for i in _idx) == cfg["squad_rules"]["squad_size"]

        formation = cfg["squad_rules"]["formation"]
        for pos, count in formation.items():
            prob += pulp.lpSum(x[i] for i, p in zip(_idx, _pos_vals) if p == pos) == count

        max_per_club = cfg["squad_rules"]["max_per_club"]
        for team in df["team"].unique():
            prob += pulp.lpSum(x[i] for i, t in zip(_idx, _team_vals) if t == team) <= max_per_club

        for code in must_include_codes:
            idxs = df[df["code"] == code].index
            for i in idxs:
                prob += x[i] == 1

        # Patch 15 — pin data-gap protected players too (see comment above):
        # forced to stay exactly as they are this run, the same as an explicit
        # must_include, so their fabricated 0.0 placeholder can never be read by
        # the solver as "safe/attractive to drop."
        for code in pinned_gap_codes:
            idxs = df[df["code"] == code].index
            for i in idxs:
                prob += x[i] == 1

        if retain_pool_codes and min_retain > 0:
            idxs = df[df["code"].isin(retain_pool_codes)].index
            if len(idxs) > 0:
                prob += pulp.lpSum(x[i] for i in idxs) >= min(min_retain, len(idxs))

        _status = _solve_and_get_status(prob, _cbc_solver(msg=0, time_limit=_solve_time_limit(cfg)))

        if _status != "Optimal":
            _diag(label, f"CBC solver returned status={_status} (not Optimal) — "
                          f"{len(df)} candidates, budget={budget}, squad_size="
                          f"{cfg['squad_rules']['squad_size']}, max_per_club={cfg['squad_rules']['max_per_club']}, "
                          f"formation={cfg['squad_rules']['formation']}"
                          + (f", must retain >= {min(min_retain, len(df[df['code'].isin(retain_pool_codes or [])]))} "
                             f"of {len(retain_pool_codes or [])} retain-pool codes" if retain_pool_codes and min_retain > 0 else "")
                          + " — no legal squad exists under these constraints with this candidate pool.")
            return None

        chosen = [i for i in df.index if x[i].value() == 1]
        squad = df.loc[chosen].sort_values(["position", objective_col], ascending=[True, False])
        _clear_diag(label)
        return {
            "squad": squad,
            "total_xpts": round(squad[objective_col].sum(), 2),
            "cost": round(squad["price"].sum(), 1),
            "data_gap_codes": data_gap_codes,
        }
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see Patch 70 note above
        print(f"[optimizer.solve_squad] MILP build/solve failed ({objective_col}, "
              f"{len(df)} candidates): {type(exc).__name__}: {exc}", file=sys.stderr)
        _diag(label, f"an unexpected exception hit the MILP build/solve: {type(exc).__name__}: {exc} "
                      f"({len(df)} candidates, objective_col={objective_col}). {_pulp_forensics()}")
        return None


@_cache_decorator
def solve_squad_xi_weighted(players: pd.DataFrame, cfg: dict, budget: float, window_cols: list,
                            bench_weight: float = 0.08, bb_col: str | None = None,
                            bonus: dict | None = None, label: str | None = None,
                            week_weights: list | None = None, captain: bool = False,
                            retain_pool_codes: list | None = None, min_retain: int = 0,
                            captain_k: int = 0, must_include_codes: list | None = None) -> dict | None:
    """Patch 114 (manager: the Wildcard must be the best team over the window, then shaped for the chips). Verified in the
    Patch 113 code: solve_squad() maximises the plain SUM of all 15 players, i.e. every player counts as a starter in
    every week. Here the squad (x_i) and a legal starting XI for EACH week of the window (y_ig) are chosen together:

        maximise  sum_g sum_i [ y_ig * v_ig  +  (x_i - y_ig) * w_g * v_ig ]  +  bonus
        w_g = 1.0 in the Bench Boost week (`bb_col` names its column), else `bench_weight`
        bonus = the Triple Captain's extra x1: {"code": c, "col": "xpts_gwN"} adds v_cN to y_(c,N)

    i.e. each week's best XI counts in full and the bench at the planner's own bench weight (Rule #12 / transfer.
    bench_weight_non_bb_gw), exactly how the weekly scoring values a squad. bench_weight=1.0 reproduces solve_squad's
    optimum. MEASURED before shipping (3 sandbox scenarios, GW6-13 window, realized value): a static single-XI variant
    of this idea was WORSE than the 15-man sum in 2 of 3 scenarios, whereas this per-week model beat the 15-man sum in
    all 3 (+2.7 to +6.7 xPts) and solves in ~1s.
    Patch 115 (model v6.12, Rule #54): `week_weights` (one per window column, e.g. recommend.decay_weights) multiply every
    term of their week -- nearer weeks count more; `captain=True` adds each week's captain (the XI's top scorer, one extra x1)
    via continuous variables c_ig (sum_i c_ig = 1, c_ig <= y_ig): maximising picks the best XI member, and the LP vertex is
    integral, so it is exact without extra binaries; `retain_pool_codes`/`min_retain` keep at least that many of the given
    players (the reachable-from-your-squad variant used by the Rule #48 scan). Not modelled (disclosed): the autosub
    probability curve inside the solve (the squad is scored with it afterwards). Same legality as solve_squad:
    2/5/5/3, club cap, budget, available players only."""
    if pulp is None:
        _diag(label, "pulp (the MILP library this solver needs) is not installed/importable in this environment.")
        return None
    cols = [c for c in (window_cols or []) if c in players.columns]
    if not cols:
        _diag(label, "no window column present for the XI-weighted solve.")
        return None
    df = players.dropna(subset=["price", "position"] + cols).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    if "status" in df.columns:
        _keep = set(retain_pool_codes or []) if (retain_pool_codes and int(min_retain) > 0) else set()
        _keep |= set(must_include_codes or [])                        # Patch 117d: a locked player stays eligible
        df = df[(df["status"] == "a") | df["code"].isin(_keep)]       # a retained player stays eligible (as solve_squad)
    if df.empty:
        _diag(label, "empty candidate pool for the XI-weighted solve (nothing survived the price/projection/position/"
                      "status filters).")
        return None
    df = df.reset_index(drop=True)
    bw = float(bench_weight)
    try:
        prob = pulp.LpProblem("fpl_squad_weekly_xi", pulp.LpMaximize)
        I = list(df.index)
        x = {i: _binary_var(prob, f"x_{i}") for i in I}
        y = {(i, c): _binary_var(prob, f"y_{i}_{k}") for k, c in enumerate(cols) for i in I}
        V = {c: df[c].tolist() for c in cols}
        POS = df["position"].tolist()
        TEAM = df["team"].tolist()
        PRICE = df["price"].tolist()
        CODE = df["code"].tolist()
        b_code = int(bonus["code"]) if bonus and bonus.get("code") is not None else None
        b_col = bonus.get("col") if bonus else None
        WK = list(week_weights) if week_weights else [1.0] * len(cols)
        if len(WK) != len(cols):
            WK = (WK + [WK[-1]] * len(cols))[:len(cols)] if WK else [1.0] * len(cols)
        cap_v = {}
        if captain:
            # Patch 115 fix: only each week's top-`captain_k` scorers can be the captain (0 = everyone). The captain is the
            # XI's best scorer, so a squad whose XI holds none of the top 60 of a week is not a realistic optimum; measured
            # identical squad and total on 3 pools, ~2x faster under four parallel solves.
            TOP = {c: (set(sorted(I, key=lambda i, c=c: -V[c][i])[:int(captain_k)]) if int(captain_k) > 0 else set(I)) for c in cols}
            cap_v = {(i, c): pulp.LpVariable(f"c_{i}_{k}", lowBound=0, upBound=1) for k, c in enumerate(cols) for i in I
                     if i in TOP[c]}
        terms = []
        for k, c in enumerate(cols):
            w = 1.0 if (bb_col is not None and c == bb_col) else bw
            wk = float(WK[k])
            for i in I:
                v = V[c][i]
                extra = v if (b_code is not None and CODE[i] == b_code and c == b_col) else 0.0
                terms.append(wk * (y[(i, c)] * ((1.0 - w) * v + extra) + x[i] * (w * v)))
                if captain and (i, c) in cap_v:
                    terms.append(cap_v[(i, c)] * (wk * v))
        prob += pulp.lpSum(terms)
        if captain:
            for c in cols:
                prob += pulp.lpSum(cap_v[(i, c)] for i in I if (i, c) in cap_v) == 1
                for i in I:
                    if (i, c) in cap_v:
                        prob += cap_v[(i, c)] <= y[(i, c)]
        if retain_pool_codes and int(min_retain) > 0:
            _ret = set(retain_pool_codes)
            prob += pulp.lpSum(x[i] for i in I if CODE[i] in _ret) >= int(min_retain)
        for _mi in I:                                                 # Patch 117d: Locked players are always in the squad
            if must_include_codes and CODE[_mi] in set(must_include_codes):
                prob += x[_mi] == 1
        prob += pulp.lpSum(x[i] * PRICE[i] for i in I) <= budget
        prob += pulp.lpSum(x[i] for i in I) == cfg["squad_rules"]["squad_size"]
        for pos, count in cfg["squad_rules"]["formation"].items():
            prob += pulp.lpSum(x[i] for i in I if POS[i] == pos) == count
        max_per_club = cfg["squad_rules"]["max_per_club"]
        for team in df["team"].unique():
            prob += pulp.lpSum(x[i] for i in I if TEAM[i] == team) <= max_per_club
        bounds = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
        for c in cols:
            prob += pulp.lpSum(y[(i, c)] for i in I) == 11
            for pos, (lo, hi) in bounds.items():
                s_ = pulp.lpSum(y[(i, c)] for i in I if POS[i] == pos)
                prob += s_ >= lo
                prob += s_ <= hi
            for i in I:
                prob += y[(i, c)] <= x[i]
        status = _solve_and_get_status(prob, _cbc_solver(msg=0, time_limit=_solve_time_limit(cfg)))
        if status != "Optimal":
            _diag(label, f"CBC solver returned status={status} for the XI-weighted squad solve ({len(df)} candidates, "
                          f"{len(cols)} weeks, budget={budget}).")
            return None
        chosen = [i for i in I if x[i].value() is not None and x[i].value() > 0.5]
        squad = df.loc[chosen].sort_values(["position", cols[0]], ascending=[True, False])
        _clear_diag(label)
        return {"squad": squad, "total_xpts": round(float(sum(squad[c].sum() for c in cols)), 2),
                "cost": round(squad["price"].sum(), 1), "data_gap_codes": []}
    except Exception as exc:  # noqa: BLE001 -- same graceful degradation as solve_squad
        print(f"[optimizer.solve_squad_xi_weighted] MILP build/solve failed ({len(cols)} weeks, {len(df)} candidates): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        _diag(label, f"an unexpected exception hit the XI-weighted MILP: {type(exc).__name__}: {exc}")
        return None


@_cache_decorator
def solve_xi_first_squad(players: pd.DataFrame, cfg: dict, budget: float, gw_col: str,
                          label: str | None = None) -> dict | None:
    """Patch 82 (2026-09-28, manager:
    "the performance generally is too slow"). Found in code, not assumed:
    this function does 8 shape solves (Stage 1) + up to 8 bench solves
    (Stage 2) = up to 16 real MILP solves per call, and unlike its sibling
    solve_squad() (cached since Patch 42, for the exact same "too slow"
    complaint), this had NO `@_cache_decorator` at all -- confirmed
    empirically: two back-to-back calls with byte-identical inputs both
    took ~3.8s each before this fix, since every call re-solved from
    scratch. app.py calls this (via data_pipeline.solve_free_hit_optimal_
    squad()) from multiple places every run -- at minimum once for the
    always-on "GW{n} Rating" header card, and again, with FREQUENTLY
    IDENTICAL arguments, whenever the Chip Advisor's Free Hit verdict is
    "PLAY GW{n}" for that same GW -- so this was often being solved twice
    over for literally the same answer, and re-solved again on every
    unrelated widget interaction that triggers a Streamlit rerun (moving
    the Horizon slider, changing Style, anything), since nothing was
    caching it. Adding the same, already-proven `@_cache_decorator`
    solve_squad() already uses closes this gap identically: a cache hit
    requires byte-identical inputs (players' content, cfg, budget, gw_col),
    so a genuine data refresh or a genuinely different GW/budget always
    still gets a fresh solve -- zero change to what gets recommended, pure
    caching, same guarantee Patch 42 already established for solve_squad().

    Free Hit "optimal team for this GW" feature (2026-09-07 discussion,
    Patch 19) — Option A (two-stage, manager-confirmed): unlike solve_squad()
    (which maximizes the raw sum of all 15 players' projections and has no
    concept of starter vs. bench at solve time, so it has no actual incentive
    to keep a bench cheap), this deliberately solves for "highest 11
    starters, light bench" as two separate stages:

    Stage 1 — best legal Starting XI. For each of the 8 valid outfield
    shapes (same VALID_SHAPES as best_starting_xi(), since the XI must be a
    real, playable formation), solve a MILP picking exactly 1 GK + that
    shape's DEF/MID/FWD counts maximizing this single GW's projection,
    under a reserved sub-budget (total budget minus a cheap-bench estimate)
    and the max-per-club limit. Take whichever shape scores highest.

    Stage 2 — cheapest legal bench. With the XI fixed, solve a second, small
    MILP: fill the remaining squad slots needed to reach the full 2-5-5-3
    (1 more GK + whatever DEF/MID/FWD the chosen shape didn't use) by
    MINIMIZING total price from whatever's left in the pool, respecting the
    combined max-per-club limit (XI's club counts + bench's) and whatever
    budget the XI didn't spend.

    This is deliberately two solves rather than one combined MILP: the
    "cheapest bench" objective only makes sense once the XI (and therefore
    which position-counts still need filling, and how much budget is left)
    is already fixed — a single-pass objective can't express "maximize
    these 11, minimize these 4" without a made-up relative weighting between
    the two goals, which is exactly the kind of guessed number this project
    avoids (see model_config.yaml's legacy, superseded flat
    `bench_autosub_discount`).

    Returns None if either stage can't find a feasible solution (e.g. the
    reserved bench budget estimate turns out too tight for the club mix the
    best XI happened to pick) — caller should treat that as "couldn't solve
    a Free Hit squad for this GW this run," same as solve_squad() returning
    None."""
    if pulp is None:
        _diag(label, "pulp (the MILP library this solver needs) is not installed/importable in this environment.")
        return None

    VALID_SHAPES = [(3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3)]
    max_per_club = cfg["squad_rules"]["max_per_club"]
    formation = cfg["squad_rules"]["formation"]  # {GK: 2, DEF: 5, MID: 5, FWD: 3}

    _players_in = len(players)
    df = players.dropna(subset=["price", gw_col, "position"]).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    df = df[df["status"] == "a"]
    if df.empty:
        _diag(label, f"empty candidate pool after filtering: 0 of {_players_in} input rows survived the "
                      f"price/{gw_col}/position not-null check, GK/DEF/MID/FWD position check and status=='a' "
                      f"filter — nothing left for the solver to choose from. If {gw_col} is mostly missing this "
                      f"gameweek (e.g. this GW's fixture/projection data hasn't fully loaded), that alone would "
                      f"empty the pool.")
        return None
    # Patch 71 (manager, live TypeError crash on this exact function --
    # `_solve_xi_for_shape` below -- immediately after Patch 70 hardened the
    # sibling solve_squad() but missed this second, separate MILP builder in
    # this same file). Same fix, same reasoning as solve_squad()'s Patch 70
    # note: `df`'s index here is whatever survived dropna/position/status
    # filtering from the caller's frame, never explicitly reset, so it's not
    # guaranteed unique -- and `bench_pool` below is filtered straight from
    # this same `df`, so resetting it here also covers Stage 2's bench solve.
    df = df.reset_index(drop=True)

    # Cheap-bench budget reserve estimate for Stage 1: the 4 lowest prices
    # available across GK/DEF/MID/FWD that a bench (1 GK + 3 outfield, in
    # some position mix) could possibly need — a lower bound, not a real
    # allocation (Stage 2 computes the real one once the XI/shape is fixed).
    cheapest_by_pos = {pos: sorted(df[df["position"] == pos]["price"].tolist())
                        for pos in ["GK", "DEF", "MID", "FWD"]}
    if any(len(v) == 0 for v in cheapest_by_pos.values()):
        _missing_pos = [pos for pos, v in cheapest_by_pos.items() if len(v) == 0]
        _diag(label, f"no available (status=='a') candidates at all in position(s) {_missing_pos} after "
                      f"filtering — a bench needs at least one of every position, so the reserve estimate "
                      f"can't even be computed.")
        return None
    bench_reserve_estimate = (cheapest_by_pos["GK"][0] +
                               sum(sorted(cheapest_by_pos["DEF"] + cheapest_by_pos["MID"] +
                                          cheapest_by_pos["FWD"])[:3]))
    xi_budget_cap = max(0.0, budget - bench_reserve_estimate)

    # Patch 72 — per-shape/per-attempt failure reasons, collected here (not
    # returned from the closures themselves, which must keep their existing
    # None-on-failure contract intact for the caller loop below) so a
    # diagnostic can still be recorded if EVERY shape/bench attempt fails.
    _shape_failures: list[str] = []
    _bench_failures: list[str] = []

    # Performance (measured, lossless -- same fix as solve_squad() above):
    # read the columns once into plain lists instead of a pandas
    # `df.loc[i, col]` scalar lookup per player per constraint per shape
    # (this builds up to 8 shape models per call). Identical model.
    _xi_idx = list(df.index)
    _xi_gw_vals = df[gw_col].tolist()
    _xi_price_vals = df["price"].tolist()
    _xi_pos_vals = df["position"].tolist()
    _xi_team_vals = df["team"].tolist()

    def _solve_xi_for_shape(d: int, m: int, f: int) -> dict | None:
        # Patch 71 — same try/except hardening as solve_squad() (Patch 70):
        # every OTHER failure path in this module already returns None on a
        # genuine infeasible/non-Optimal solve, and the caller already
        # tolerates individual shapes failing (`if result: xi_candidates.
        # append(result)`) -- but an unexpected exception during THIS solve
        # previously wasn't caught at all, so it crashed the whole page
        # instead of just skipping this one shape like an infeasible shape
        # already does.
        try:
            prob = pulp.LpProblem("fh_xi", pulp.LpMaximize)
            x = {i: _binary_var(prob, f"xi_{i}") for i in df.index}
            prob += pulp.lpSum(x[i] * v for i, v in zip(_xi_idx, _xi_gw_vals))
            prob += pulp.lpSum(x[i] * v for i, v in zip(_xi_idx, _xi_price_vals)) <= xi_budget_cap
            prob += pulp.lpSum(x[i] for i, p in zip(_xi_idx, _xi_pos_vals) if p == "GK") == 1
            prob += pulp.lpSum(x[i] for i, p in zip(_xi_idx, _xi_pos_vals) if p == "DEF") == d
            prob += pulp.lpSum(x[i] for i, p in zip(_xi_idx, _xi_pos_vals) if p == "MID") == m
            prob += pulp.lpSum(x[i] for i, p in zip(_xi_idx, _xi_pos_vals) if p == "FWD") == f
            for team in df["team"].unique():
                prob += pulp.lpSum(x[i] for i, t in zip(_xi_idx, _xi_team_vals) if t == team) <= max_per_club
            _status = _solve_and_get_status(prob, _cbc_solver(msg=0, time_limit=_solve_time_limit(cfg)))
            if _status != "Optimal":
                _shape_failures.append(f"shape {(d, m, f)}: CBC status={_status} "
                                        f"(xi_budget_cap={xi_budget_cap:.1f}, {len(df)} candidates)")
                return None
            chosen = [i for i in df.index if x[i].value() == 1]
            xi = df.loc[chosen]
            return {"xi": xi, "total": round(xi[gw_col].sum(), 2), "shape": (d, m, f)}
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see Patch 71 note above
            print(f"[optimizer.solve_xi_first_squad._solve_xi_for_shape] MILP build/solve failed "
                  f"(shape {(d, m, f)}, {len(df)} candidates): {type(exc).__name__}: {exc}", file=sys.stderr)
            _shape_failures.append(f"shape {(d, m, f)}: {type(exc).__name__}: {exc} {_pulp_forensics()}")
            return None

    def _solve_bench_for_xi(xi_result: dict) -> dict | None:
        xi_df = xi_result["xi"]
        xi_codes = set(xi_df["code"])
        xi_cost = float(xi_df["price"].sum())
        xi_club_counts = xi_df["team"].value_counts().to_dict()
        need = {
            "GK": formation["GK"] - 1,
            "DEF": formation["DEF"] - xi_result["shape"][0],
            "MID": formation["MID"] - xi_result["shape"][1],
            "FWD": formation["FWD"] - xi_result["shape"][2],
        }
        remaining_budget = max(0.0, budget - xi_cost)
        bench_pool = df[~df["code"].isin(xi_codes)]

        # Patch 71 — same try/except hardening as _solve_xi_for_shape above
        # and solve_squad() (Patch 70): an unexpected exception during this
        # solve previously crashed the whole page instead of degrading to
        # None like every genuine infeasible-solve path already does (the
        # caller already falls back to the next-best XI candidate when a
        # bench solve returns None, so this is a safe, already-expected
        # failure mode to route an exception into).
        try:
            prob2 = pulp.LpProblem("fh_bench", pulp.LpMinimize)
            y = {i: _binary_var(prob2, f"bn_{i}") for i in bench_pool.index}
            _b_idx = list(bench_pool.index)
            _b_price = bench_pool["price"].tolist()
            _b_pos = bench_pool["position"].tolist()
            _b_team = bench_pool["team"].tolist()
            prob2 += pulp.lpSum(y[i] * v for i, v in zip(_b_idx, _b_price))
            prob2 += pulp.lpSum(y[i] * v for i, v in zip(_b_idx, _b_price)) <= remaining_budget
            for pos, n in need.items():
                prob2 += pulp.lpSum(y[i] for i, p in zip(_b_idx, _b_pos) if p == pos) == n
            for team in bench_pool["team"].unique():
                already = xi_club_counts.get(team, 0)
                prob2 += pulp.lpSum(y[i] for i, t in zip(_b_idx, _b_team) if t == team) \
                    <= max(0, max_per_club - already)
            _status = _solve_and_get_status(prob2, _cbc_solver(msg=0, time_limit=_solve_time_limit(cfg)))
            if _status != "Optimal":
                _bench_failures.append(f"shape {xi_result.get('shape')}: CBC status={_status} "
                                        f"(remaining_budget={remaining_budget:.1f}, need={need}, "
                                        f"{len(bench_pool)} candidates)")
                return None

            bench_chosen = [i for i in bench_pool.index if y[i].value() == 1]
            bench_df = bench_pool.loc[bench_chosen]
            squad = pd.concat([xi_df, bench_df], ignore_index=False, sort=False) \
                .sort_values(["position", gw_col], ascending=[True, False])
            return {
                "squad": squad,
                "xi_codes": xi_codes,
                "shape": xi_result["shape"],
                "xi_total": xi_result["total"],
                "bench_cost": round(float(bench_df["price"].sum()), 1),
                "total_cost": round(xi_cost + float(bench_df["price"].sum()), 1),
            }
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see Patch 71 note above
            print(f"[optimizer.solve_xi_first_squad._solve_bench_for_xi] MILP build/solve failed "
                  f"(shape {xi_result.get('shape')}, {len(bench_pool)} candidates): "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            _bench_failures.append(f"shape {xi_result.get('shape')}: {type(exc).__name__}: {exc} {_pulp_forensics()}")
            return None

    # Solve every shape's XI, then try Stage 2 against them in descending
    # XI-score order — the single best-scoring XI can still leave Stage 2
    # infeasible (its particular club mix can exhaust the max-3-per-club
    # limit at a club whose players happen to be the cheapest available for
    # a still-needed bench position, with no budget room left to go
    # elsewhere). Falling back to the next-best XI whenever that happens is
    # what makes this genuinely "the optimal team," not just "the optimal
    # XI, if we got lucky on the bench" — confirmed necessary by testing:
    # a synthetic pool reproduced exactly this failure on the single-best-
    # XI-only version of this function.
    xi_candidates = []
    for d, m, f in VALID_SHAPES:
        result = _solve_xi_for_shape(d, m, f)
        if result:
            xi_candidates.append(result)
    xi_candidates.sort(key=lambda r: r["total"], reverse=True)

    # Patch 72 — if every shape failed, that's the diagnosable reason (no
    # legal Starting XI at all under xi_budget_cap/max_per_club this run).
    if not xi_candidates:
        _diag(label, f"no feasible Starting XI found across any of the 8 valid shapes this run "
                      f"(xi_budget_cap={xi_budget_cap:.1f}, {len(df)} candidates). Per-shape detail: "
                      + "; ".join(_shape_failures))
        return None

    for xi_result in xi_candidates:
        solved = _solve_bench_for_xi(xi_result)
        if solved:
            _clear_diag(label)
            return solved
    # Patch 72 — every shape found a legal XI, but none of them left a
    # buildable bench (budget/club-limit conflict at Stage 2 for all of
    # them) — a genuinely different failure mode than "no XI at all," worth
    # distinguishing in the diagnostic.
    _diag(label, f"found {len(xi_candidates)} feasible Starting XI shape(s) this run, but no legal bench "
                  f"(budget + max-per-club) could be built for any of them. Per-attempt detail: "
                  + "; ".join(_bench_failures))
    return None


def bench_autosub_prob(position: str, bench_rank: int, starters_xi: pd.DataFrame,
                        xm_col: str, cfg: dict) -> float:
    """Standing Rule #12 (Bench Value Rule) heuristic. Disclosed EST, not a
    full per-fixture autosub model — the doc's own text flags this as
    revisable once real per-gameweek autosub data exists (Patch 5, reversing
    the Patch 3 error of retiring this rule entirely on the mistaken
    reasoning that a player's own `xm` already prices it in; the manager
    corrected this directly — `xm` prices a player's OWN minutes if picked,
    it says nothing about the separate question of whether an autosub even
    fires for him).

    Two independent factors, multiplied:
    - Exposure: how likely the starting XI's own players at this position
      fail to feature at all, approximated as 1 - the XI's average `xm` in
      that position, clamped to [0.03, 0.6]. GK is a special case — there is
      only ever one starting GK, and a Premier League #1 missing a match
      entirely is rare, so the backup keeper gets a small fixed floor
      instead (`bench_gk_autosub_prob`), never the outfield formula.
    - Bench-order decay: an autosub chain rarely reaches past the first
      reserve or two, so the Nth-highest-projected outfield bench player
      (0 = first reserve) is scaled down by `bench_order_decay[N]`.
    """
    tcfg = cfg.get("transfer", {}) if cfg else {}
    if position == "GK":
        return float(tcfg.get("bench_gk_autosub_prob", 0.05))
    curve = tcfg.get("bench_order_decay", [1.0, 0.55, 0.30, 0.15])
    decay = curve[min(bench_rank, len(curve) - 1)]
    xm_vals = starters_xi[xm_col].dropna() if (xm_col in starters_xi.columns) else pd.Series(dtype=float)
    avg_xm = float(xm_vals.mean()) if not xm_vals.empty else 0.8
    exposure = max(0.03, min(0.6, 1.0 - avg_xm))
    return round(exposure * decay, 3)


def _bench_autosub_total(squad: pd.DataFrame, xi: pd.DataFrame, gw_col: str, xm_col: str, cfg: dict) -> float:
    """Patch 109 (performance, lossless): the bench-autosub loop shared by
    realized_gw_value() and rating_gw_value(). Same per-player maths as before
    (`bench_autosub_prob(pos, rank, xi, xm_col, cfg)` x the player's GW points,
    summed in the same position/rank order); the XI's average `xm` -- which
    does not depend on which bench player is being priced -- is computed once
    here instead of once per bench player, and rows are read as arrays instead
    of via iterrows(). bench_autosub_prob() itself is unchanged."""
    bench = squad[~squad.index.isin(xi.index)]
    if bench.empty:
        return 0.0
    tcfg = cfg.get("transfer", {}) if cfg else {}
    gk_prob = float(tcfg.get("bench_gk_autosub_prob", 0.05))
    curve = tcfg.get("bench_order_decay", [1.0, 0.55, 0.30, 0.15])
    xm_vals = xi[xm_col].dropna() if (xm_col in xi.columns) else pd.Series(dtype=float)
    avg_xm = float(xm_vals.mean()) if not xm_vals.empty else 0.8
    exposure = max(0.03, min(0.6, 1.0 - avg_xm))
    pos_arr = bench["position"].to_numpy()
    total = 0.0
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_bench = bench[pos_arr == pos].sort_values(gw_col, ascending=False)
        for rank, pts in enumerate(pos_bench[gw_col].to_numpy()):
            pts = 0.0 if pd.isna(pts) else float(pts)
            if pos == "GK":
                prob = gk_prob
            else:
                prob = round(exposure * curve[min(rank, len(curve) - 1)], 3)
            total += prob * pts
    return total


_RGV_MEMO: dict = {}
_RGV_MEMO_MAX = 200_000
_NAN_KEY = -1.0e18


def _rgv_key(squad: pd.DataFrame, gw_col: str, cfg: dict, xm_col: str, bench_weight_scale: float):
    """Patch 111 (performance, lossless): everything realized_gw_value() reads, as a hashable key --
    row order, index labels (the bench step filters by label), position, the GW points, `xm`, the
    scale and the two config values bench_autosub_prob() uses. NaN is mapped to a sentinel because
    NaN != NaN would defeat dict lookups."""
    tcfg = cfg.get("transfer", {}) if cfg else {}
    vals = np.nan_to_num(squad[gw_col].to_numpy(dtype=float, na_value=np.nan), nan=_NAN_KEY)
    xm = (np.nan_to_num(squad[xm_col].to_numpy(dtype=float, na_value=np.nan), nan=_NAN_KEY)
          if xm_col in squad.columns else None)
    return (gw_col, xm_col, float(bench_weight_scale), bool(tcfg.get("planner_captain", False)),
            float(tcfg.get("bench_gk_autosub_prob", 0.05)),
            tuple(tcfg.get("bench_order_decay", [1.0, 0.55, 0.30, 0.15])), tuple(squad.index.tolist()),
            tuple(squad["position"].tolist()), tuple(vals.tolist()), None if xm is None else tuple(xm.tolist()))


def realized_gw_value(squad: pd.DataFrame, gw_col: str, cfg: dict, xm_col: str = "xm",
                       bench_weight_scale: float = 1.0) -> dict:
    """Memoised wrapper (Patch 111): the transfer planner and the Wildcard chain comparison re-score the
    exact same squad/GW thousands of times (measured: 89% of calls were exact repeats). The result is a
    pure function of the key built by _rgv_key(), so caching is lossless (parity-tested against the
    Patch 108 implementation, including tied values, NaNs and duplicate index labels)."""
    if squad is None or squad.empty or gw_col not in squad.columns:
        return {"xi_total": 0.0, "bench_total": 0.0, "total_realized": 0.0}
    try:
        key = _rgv_key(squad, gw_col, cfg, xm_col, bench_weight_scale)
    except Exception:
        return _realized_gw_value_uncached(squad, gw_col, cfg, xm_col, bench_weight_scale)
    hit = _RGV_MEMO.get(key)
    if hit is not None:
        return dict(hit)
    res = _realized_gw_value_uncached(squad, gw_col, cfg, xm_col, bench_weight_scale)
    if len(_RGV_MEMO) >= _RGV_MEMO_MAX:
        _RGV_MEMO.clear()
    _RGV_MEMO[key] = dict(res)
    return res


def _realized_gw_value_uncached(squad: pd.DataFrame, gw_col: str, cfg: dict, xm_col: str = "xm",
                                 bench_weight_scale: float = 1.0) -> dict:
    """Standing Rule #12 (Bench Value Rule): "a bench player's value in any
    comparison is P(autosub triggers) x their points in that scenario, never
    their full 'if they started every week' xPts — and this must actually be
    implemented in any solver's scoring function, not just documented."

    Picks the best valid starting XI for this single GW (Horizon-Matching
    Rule, same mechanic as `best_starting_xi`), then values the 4 remaining
    bench slots at their autosub-discounted rate (`bench_autosub_prob`)
    instead of their raw projection, so the returned total is a squad's
    REALIZED value for this week — not a fantasy "everyone started" total.

    `bench_weight_scale` (Patch 37, manager report 2026-09-14: a transfer
    that only won on bench-quality credit — Gomez displacing a dead Foden on
    the bench, not a real starting-XI upgrade — was inflating a 1.4-1.6 xPts
    direct swap into a reported +2.9 net gain) further scales DOWN just the
    bench_total component, leaving xi_total (a real starting-XI swap) always
    counted in full. Defaults to 1.0 (unchanged) for every existing caller —
    only the transfer-recommendation net-gain math in recommend.py opts into
    a reduced scale via `realized_horizon_value`'s own `bench_weight_scale`/
    `bb_play_gw` params."""
    empty = {"xi_total": 0.0, "bench_total": 0.0, "total_realized": 0.0}
    if squad is None or squad.empty or gw_col not in squad.columns:
        return empty
    xi_result = best_starting_xi(squad, gw_col)
    if xi_result is None:
        return empty
    xi = xi_result["xi"]
    xi_total = float(xi_result["total"])
    bench_total = _bench_autosub_total(squad, xi, gw_col, xm_col, cfg)
    bench_total *= bench_weight_scale
    if bool((cfg.get("transfer", {}) if cfg else {}).get("planner_captain", False)):
        # Patch 115 fix 2: OPT-IN sensitivity switch (default OFF = Standing Rule #31, values unchanged). ON adds the
        # XI's top scorer once, like Rule #54 asks of the Wildcard build.
        cap = float(xi[gw_col].max()) if not xi.empty else 0.0
        return {"xi_total": round(xi_total, 2), "bench_total": round(bench_total, 2), "captain_bonus": round(cap, 2),
                "total_realized": round(xi_total + bench_total + cap, 2)}
    return {"xi_total": round(xi_total, 2), "bench_total": round(bench_total, 2),
            "total_realized": round(xi_total + bench_total, 2)}


def realized_horizon_value(squad: pd.DataFrame, gw_list: list[int], cfg: dict, xm_col: str = "xm",
                            bench_weight_scale: float = 1.0, bb_play_gw: int | None = None) -> float:
    """Sums `realized_gw_value()`'s total across every GW in the horizon —
    the Rule #12-compliant replacement for a raw `xpts_horizon_sum` sum
    whenever squads/transfer-candidates are being SCORED against each other.
    Never used to display a single player's own projection — only to decide
    which candidate squad actually wins a comparison.

    Patch 37: `bench_weight_scale` (default 1.0, unchanged) lets a caller
    de-weight bench-autosub credit uniformly across the horizon; `bb_play_gw`,
    when given, forces the scale back to a full 1.0 for that ONE gameweek
    only — Bench Boost genuinely counts full bench value that specific week,
    every other GW in `gw_list` uses `bench_weight_scale`."""
    total = 0.0
    for gw in gw_list:
        scale = 1.0 if (bb_play_gw is not None and gw == bb_play_gw) else bench_weight_scale
        total += realized_gw_value(squad, f"xpts_gw{gw}", cfg, xm_col, bench_weight_scale=scale)["total_realized"]
    return round(total, 2)


def realized_horizon_breakdown(squad: pd.DataFrame, gw_list: list[int], cfg: dict, xm_col: str = "xm",
                                bench_weight_scale: float = 1.0, bb_play_gw: int | None = None) -> dict:
    """Patch 38 (manager report, 2026-09-14: "we spent the whole day
    explaining the logic and still the same issue" — every dispute in that
    thread took multiple screenshot round-trips to resolve because the
    starting-XI swap value and the bench-autosub credit were only ever
    visible bundled together as one net-gain number). Same per-GW loop as
    `realized_horizon_value()`, but keeps the xi_total and bench_total
    components separate across the whole horizon instead of summing them
    into one number — lets a disclosure line show the direct swap and the
    (now near-zero, per Patch 37) bench credit as two auditable figures
    instead of one black-box total."""
    xi_sum, bench_sum = 0.0, 0.0
    for gw in gw_list:
        scale = 1.0 if (bb_play_gw is not None and gw == bb_play_gw) else bench_weight_scale
        gv = realized_gw_value(squad, f"xpts_gw{gw}", cfg, xm_col, bench_weight_scale=scale)
        xi_sum += gv["xi_total"]
        bench_sum += gv["bench_total"]
    return {"xi_total": round(xi_sum, 2), "bench_total": round(bench_sum, 2),
            "total": round(xi_sum + bench_sum, 2)}


def rating_gw_value(squad: pd.DataFrame, gw_col: str, cfg: dict, xm_col: str = "xm") -> dict:
    """§1a Team Rating % ONLY (Patch 20, 2026-09-07 discussion) — a single
    GW's contribution to Squad_xPts/Ceiling_xPts, "with captaincy applied
    per Step 7's joint per-week XI+captain evaluation" as §1a's own text
    requires: best legal XI for this GW, PLUS the captain bonus (the XI's
    own top scorer counted a second time — the actual doubling effect,
    never a separate/different player), PLUS the 4 bench slots valued at
    their Rule #12 autosub-discounted rate (never full raw value — a
    benched player only scores if an autosub actually fires).

    DELIBERATELY SEPARATE from realized_gw_value()/realized_horizon_value()
    above, which transfer-path comparisons and the Wildcard what-if reuse —
    those are barred from including captaincy in their primary ranking by
    Standing Rule #31 ("this information is disclosure, never a scoring
    input... the primary path ranking... remains anchored purely to squad
    xPts and Team Rating %"). §1a's captaincy-inclusion is a named exception
    specific to Team Rating % itself; folding it into the shared function
    would either break §1a's formula or violate Rule #31 for every other
    caller of realized_gw_value(). Keeping two functions is what lets both
    rules hold at once.

    Call this identically for the candidate/current squad AND for whichever
    squad a Ceiling/Reachable-Ceiling solve produced (Rule #22 Systematic
    Application) — never one side via this function and the other via a
    raw sum."""
    empty = {"xi_total": 0.0, "captain_bonus": 0.0, "bench_total": 0.0, "total_realized": 0.0}
    if squad is None or squad.empty or gw_col not in squad.columns:
        return empty
    xi_result = best_starting_xi(squad, gw_col)
    if xi_result is None:
        return empty
    xi = xi_result["xi"]
    xi_total = float(xi_result["total"])
    captain_bonus = float(xi[gw_col].max()) if not xi.empty else 0.0
    bench_total = _bench_autosub_total(squad, xi, gw_col, xm_col, cfg)
    total = xi_total + captain_bonus + bench_total
    return {"xi_total": round(xi_total, 2), "captain_bonus": round(captain_bonus, 2),
            "bench_total": round(bench_total, 2), "total_realized": round(total, 2)}


def rating_horizon_value(squad: pd.DataFrame, gw_list: list[int], cfg: dict, xm_col: str = "xm") -> float:
    """Sums rating_gw_value()'s total across every GW in the horizon — the
    §1a-compliant number for Team Rating %'s Squad_xPts/Ceiling_xPts, never
    used for transfer-path/Wildcard scoring (see rating_gw_value's own
    docstring for why those stay on realized_horizon_value() instead)."""
    total = 0.0
    for gw in gw_list:
        total += rating_gw_value(squad, f"xpts_gw{gw}", cfg, xm_col)["total_realized"]
    return round(total, 2)


def best_starting_xi(squad: pd.DataFrame, gw_col: str) -> dict:
    """Pick the highest-scoring valid formation (1 GK + valid outfield shape)
    for a single gameweek from a fixed 15-man squad — Horizon-Matching Rule:
    always uses that week's single-GW column, never a multi-week average.

    Patch 109 (performance, lossless): each position is sorted ONCE and every
    formation takes a prefix of that sort, instead of re-sorting identical data
    24 times. Patch 112 (performance, lossless): the sorts run on the single GW
    column (one Series per position, same pandas sort => same tie order) and the
    15-row XI frame is built once with a positional take, instead of filtering,
    sorting and concatenating whole multi-column frames. Parity-tested against
    the Patch 108 implementation (tied values, NaN, duplicate index labels,
    pyarrow-backed floats)."""
    VALID_SHAPES = [  # (DEF, MID, FWD)
        (3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3),
    ]
    pos_arr = squad["position"].to_numpy()
    s_all = squad[gw_col].reset_index(drop=True)

    def _sorted(pos):
        return s_all[pos_arr == pos].sort_values(ascending=False)

    gk_s = _sorted("GK").head(1)
    def_s, mid_s, fwd_s = _sorted("DEF"), _sorted("MID"), _sorted("FWD")
    # numpy float64 columns: sum plain arrays (identical to Series.sum for numpy data, ~10x cheaper than
    # building 8 Series). Other dtypes (e.g. pyarrow-backed floats sum differently in the last digit) keep
    # the original Series-sum semantics.
    fast = s_all.dtype == np.float64
    if fast:
        gk_v, def_v, mid_v, fwd_v = gk_s.to_numpy(), def_s.to_numpy(), mid_s.to_numpy(), fwd_s.to_numpy()
    best_shape, best_total = None, None
    for d, m, f in VALID_SHAPES:
        if len(def_s) < d or len(mid_s) < m or len(fwd_s) < f:
            continue
        if fast:
            total = pd.Series(np.concatenate([gk_v, def_v[:d], mid_v[:m], fwd_v[:f]])).sum()
        else:
            total = pd.concat([gk_s, def_s.head(d), mid_s.head(m), fwd_s.head(f)]).sum()
        if best_total is None or total > best_total:
            best_shape, best_total = (d, m, f), total
    if best_shape is None:
        return None
    d, m, f = best_shape
    pos_idx = np.concatenate([gk_s.index.to_numpy(), def_s.index.to_numpy()[:d],
                              mid_s.index.to_numpy()[:m], fwd_s.index.to_numpy()[:f]])
    xi = squad.iloc[pos_idx]
    return {"xi": xi, "total": best_total, "shape": best_shape}
