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
                 exclude_codes: list | None = None) -> dict | None:
    """players needs columns: code, web_name, team, position, price,
    xpts_horizon_sum, status. Returns dict with squad picks, total_xpts, cost.
    Only picks status=='a' (available) players unless explicitly must_include."""
    if pulp is None:
        return None

    df = players.dropna(subset=["price", "xpts_horizon_sum", "position"]).copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    must_include_codes = must_include_codes or []
    exclude_codes = exclude_codes or []
    if exclude_codes:
        df = df[~df["code"].isin(exclude_codes)]
    df = df[(df["status"] == "a") | (df["code"].isin(must_include_codes))]
    if df.empty:
        return None

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}

    prob += pulp.lpSum(x[i] * df.loc[i, "xpts_horizon_sum"] for i in df.index)

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

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    chosen = [i for i in df.index if x[i].value() == 1]
    squad = df.loc[chosen].sort_values(["position", "xpts_horizon_sum"], ascending=[True, False])
    return {
        "squad": squad,
        "total_xpts": round(squad["xpts_horizon_sum"].sum(), 2),
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
