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

    df = players.dropna(subset=["price", objective_col, "position"]).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    must_include_codes = must_include_codes or []
    exclude_codes = exclude_codes or []
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
    }


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
