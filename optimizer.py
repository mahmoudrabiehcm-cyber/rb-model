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
