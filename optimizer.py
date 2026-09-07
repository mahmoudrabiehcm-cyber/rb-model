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
import pandas as pd

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None


def solve_squad(players: pd.DataFrame, cfg: dict, budget: float = 100.0,
                 must_include_codes: list | None = None,
                 exclude_codes: list | None = None,
                 retain_pool_codes: list | None = None,
                 min_retain: int = 0,
                 objective_col: str = "xpts_horizon_sum") -> dict | None:
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

    df = players.dropna(subset=["price", objective_col, "position"]).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    if exclude_codes:
        df = df[~df["code"].isin(exclude_codes)]
    df = df[(df["status"] == "a") | (df["code"].isin(must_include_codes)) |
            (df["code"].isin(retain_pool_codes or []))]
    if df.empty:
        return None

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}

    prob += pulp.lpSum(x[i] * df.loc[i, objective_col] for i in df.index)

    prob += pulp.lpSum(x[i] * df.loc[i, "price"] for i in df.index) <= budget
    prob += pulp.lpSum(x[i] for i in df.index) == cfg["squad_rules"]["squad_size"]

    formation = cfg["squad_rules"]["formation"]
    for pos, count in formation.items():
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == pos) == count

    max_per_club = cfg["squad_rules"]["max_per_club"]
    for team in df["team"].unique():
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "team"] == team) <= max_per_club

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

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    chosen = [i for i in df.index if x[i].value() == 1]
    squad = df.loc[chosen].sort_values(["position", objective_col], ascending=[True, False])
    return {
        "squad": squad,
        "total_xpts": round(squad[objective_col].sum(), 2),
        "cost": round(squad["price"].sum(), 1),
        "data_gap_codes": data_gap_codes,
    }


def solve_xi_first_squad(players: pd.DataFrame, cfg: dict, budget: float, gw_col: str) -> dict | None:
    """Free Hit "optimal team for this GW" feature (2026-09-07 discussion,
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
        return None

    VALID_SHAPES = [(3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3)]
    max_per_club = cfg["squad_rules"]["max_per_club"]
    formation = cfg["squad_rules"]["formation"]  # {GK: 2, DEF: 5, MID: 5, FWD: 3}

    df = players.dropna(subset=["price", gw_col, "position"]).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    df = df[df["status"] == "a"]
    if df.empty:
        return None

    # Cheap-bench budget reserve estimate for Stage 1: the 4 lowest prices
    # available across GK/DEF/MID/FWD that a bench (1 GK + 3 outfield, in
    # some position mix) could possibly need — a lower bound, not a real
    # allocation (Stage 2 computes the real one once the XI/shape is fixed).
    cheapest_by_pos = {pos: sorted(df[df["position"] == pos]["price"].tolist())
                        for pos in ["GK", "DEF", "MID", "FWD"]}
    if any(len(v) == 0 for v in cheapest_by_pos.values()):
        return None
    bench_reserve_estimate = (cheapest_by_pos["GK"][0] +
                               sum(sorted(cheapest_by_pos["DEF"] + cheapest_by_pos["MID"] +
                                          cheapest_by_pos["FWD"])[:3]))
    xi_budget_cap = max(0.0, budget - bench_reserve_estimate)

    def _solve_xi_for_shape(d: int, m: int, f: int) -> dict | None:
        prob = pulp.LpProblem("fh_xi", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"xi_{i}", cat="Binary") for i in df.index}
        prob += pulp.lpSum(x[i] * df.loc[i, gw_col] for i in df.index)
        prob += pulp.lpSum(x[i] * df.loc[i, "price"] for i in df.index) <= xi_budget_cap
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "GK") == 1
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "DEF") == d
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "MID") == m
        prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "position"] == "FWD") == f
        for team in df["team"].unique():
            prob += pulp.lpSum(x[i] for i in df.index if df.loc[i, "team"] == team) <= max_per_club
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != "Optimal":
            return None
        chosen = [i for i in df.index if x[i].value() == 1]
        xi = df.loc[chosen]
        return {"xi": xi, "total": round(xi[gw_col].sum(), 2), "shape": (d, m, f)}

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

        prob2 = pulp.LpProblem("fh_bench", pulp.LpMinimize)
        y = {i: pulp.LpVariable(f"bn_{i}", cat="Binary") for i in bench_pool.index}
        prob2 += pulp.lpSum(y[i] * bench_pool.loc[i, "price"] for i in bench_pool.index)
        prob2 += pulp.lpSum(y[i] * bench_pool.loc[i, "price"] for i in bench_pool.index) <= remaining_budget
        for pos, n in need.items():
            prob2 += pulp.lpSum(y[i] for i in bench_pool.index if bench_pool.loc[i, "position"] == pos) == n
        for team in bench_pool["team"].unique():
            already = xi_club_counts.get(team, 0)
            prob2 += pulp.lpSum(y[i] for i in bench_pool.index if bench_pool.loc[i, "team"] == team) \
                <= max(0, max_per_club - already)
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob2.status] != "Optimal":
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

    for xi_result in xi_candidates:
        solved = _solve_bench_for_xi(xi_result)
        if solved:
            return solved
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


def realized_gw_value(squad: pd.DataFrame, gw_col: str, cfg: dict, xm_col: str = "xm") -> dict:
    """Standing Rule #12 (Bench Value Rule): "a bench player's value in any
    comparison is P(autosub triggers) x their points in that scenario, never
    their full 'if they started every week' xPts — and this must actually be
    implemented in any solver's scoring function, not just documented."

    Picks the best valid starting XI for this single GW (Horizon-Matching
    Rule, same mechanic as `best_starting_xi`), then values the 4 remaining
    bench slots at their autosub-discounted rate (`bench_autosub_prob`)
    instead of their raw projection, so the returned total is a squad's
    REALIZED value for this week — not a fantasy "everyone started" total."""
    empty = {"xi_total": 0.0, "bench_total": 0.0, "total_realized": 0.0}
    if squad is None or squad.empty or gw_col not in squad.columns:
        return empty
    xi_result = best_starting_xi(squad, gw_col)
    if xi_result is None:
        return empty
    xi = xi_result["xi"]
    xi_total = float(xi_result["total"])
    bench = squad[~squad.index.isin(xi.index)]
    bench_total = 0.0
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_bench = bench[bench["position"] == pos].sort_values(gw_col, ascending=False)
        for rank, (_, row) in enumerate(pos_bench.iterrows()):
            pts = row.get(gw_col, 0.0)
            pts = 0.0 if pd.isna(pts) else float(pts)
            prob = bench_autosub_prob(pos, rank, xi, xm_col, cfg)
            bench_total += prob * pts
    return {"xi_total": round(xi_total, 2), "bench_total": round(bench_total, 2),
            "total_realized": round(xi_total + bench_total, 2)}


def realized_horizon_value(squad: pd.DataFrame, gw_list: list[int], cfg: dict, xm_col: str = "xm") -> float:
    """Sums `realized_gw_value()`'s total across every GW in the horizon —
    the Rule #12-compliant replacement for a raw `xpts_horizon_sum` sum
    whenever squads/transfer-candidates are being SCORED against each other.
    Never used to display a single player's own projection — only to decide
    which candidate squad actually wins a comparison."""
    total = 0.0
    for gw in gw_list:
        total += realized_gw_value(squad, f"xpts_gw{gw}", cfg, xm_col)["total_realized"]
    return round(total, 2)


def best_starting_xi(squad: pd.DataFrame, gw_col: str) -> dict:
    """Pick the highest-scoring valid formation (1 GK + valid outfield shape)
    for a single gameweek from a fixed 15-man squad — Horizon-Matching Rule:
    always uses that week's single-GW column, never a multi-week average."""
    VALID_SHAPES = [  # (DEF, MID, FWD)
        (3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3), (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3),
    ]
    gk = squad[squad["position"] == "GK"].sort_values(gw_col, ascending=False).head(1)
    best = None
    for d, m, f in VALID_SHAPES:
        defs = squad[squad["position"] == "DEF"].sort_values(gw_col, ascending=False).head(d)
        mids = squad[squad["position"] == "MID"].sort_values(gw_col, ascending=False).head(m)
        fwds = squad[squad["position"] == "FWD"].sort_values(gw_col, ascending=False).head(f)
        if len(defs) < d or len(mids) < m or len(fwds) < f:
            continue
        xi = pd.concat([gk, defs, mids, fwds])
        total = xi[gw_col].sum()
        if best is None or total > best["total"]:
            best = {"xi": xi, "total": total, "shape": (d, m, f)}
    return best
