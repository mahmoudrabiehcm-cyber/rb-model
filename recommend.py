"""
recommend.py
Transfer suggestions (Step 7a) and the chess-themed one/two-phrase season
verdict. Both are plain deterministic logic — no external AI call, so both
stay inside the zero-cost design (an LLM-generated verdict would need a
paid API key; this doesn't).

KNOWN LIMITATION (documented, not hidden): budget-fit uses each player's
current `now_cost`, not your actual banked sale price (FPL sells a player
you've held a while for less than its current price once it's risen —
"sell-on fee" mechanics). The official API doesn't expose your exact sale
price outside the authenticated /my-team/ endpoint, which needs a login
cookie this zero-cost, no-account tool deliberately doesn't ask for. In
practice this makes the suggested budget slightly conservative (real
proceeds are usually <= now_cost), not optimistic — worth a manual gut
check on tight-budget swaps.
"""
from __future__ import annotations
import pandas as pd

import fpl_engine as eng
import style_profiles


def suggest_transfers(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                       profile_name: str, hit_stance: str, free_transfers: int,
                       bank: float, forced_count: int | None = None) -> dict:
    profile = style_profiles.get_profile(profile_name)
    hit_cost_per = cfg["transfer"]["hit_cost_per_transfer"]
    threshold = profile["hit_cost_threshold"]

    raw = []
    for _, out_p in squad_df.iterrows():
        same_pos = pool_df[(pool_df["position"] == out_p["position"]) &
                            (pool_df["status"] == "a")]
        budget_cap = out_p["price"] + bank
        afford = same_pos[same_pos["price"] <= budget_cap]
        for _, in_p in afford.iterrows():
            gain = round(in_p["xpts_horizon_sum"] - out_p["xpts_horizon_sum"], 2)
            if gain <= 0:
                continue
            raw.append({
                "out": out_p["web_name"], "out_team": out_p["team"], "out_price": out_p["price"],
                "in": in_p["web_name"], "in_team": in_p["team"], "in_price": in_p["price"],
                "position": out_p["position"], "xpts_gain": gain,
                "in_eo": in_p.get("selected_by_percent"),
                "setpiece_flag": bool(in_p.get("setpiece_flag", False)),
            })

    cols = ["out", "out_team", "out_price", "in", "in_team", "in_price",
            "position", "xpts_gain", "in_eo", "setpiece_flag"]
    ranked = pd.DataFrame(raw, columns=cols)
    if not ranked.empty:
        # one suggestion per OUT player at most (best replacement for that slot)
        ranked = ranked.sort_values("xpts_gain", ascending=False).drop_duplicates(subset=["out"], keep="first")

    if hit_stance == "No hits":
        chosen = ranked.head(free_transfers)
        moves = [{**r, "hit_cost": 0, "justified": True, "paid": False}
                 for r in chosen.to_dict("records")]

    elif hit_stance == "Force":
        n = forced_count if forced_count is not None else free_transfers
        chosen = ranked.head(n)
        moves = []
        for i, r in enumerate(chosen.to_dict("records")):
            hit_cost = hit_cost_per * max(0, i + 1 - free_transfers)
            net = r["xpts_gain"] - hit_cost
            moves.append({**r, "hit_cost": hit_cost, "net_gain": round(net, 2),
                          "justified": net >= threshold if hit_cost > 0 else True, "paid": hit_cost > 0})

    else:  # "Hit if worth it" (default)
        free_moves = ranked.head(free_transfers).to_dict("records")
        moves = [{**r, "hit_cost": 0, "justified": True, "paid": False} for r in free_moves]
        extra_pool = ranked.iloc[free_transfers:free_transfers + 2]
        for r in extra_pool.to_dict("records"):
            hit_cost = hit_cost_per  # one extra paid transfer beyond the free allowance
            net_gain = r["xpts_gain"] - hit_cost
            if net_gain >= threshold:
                moves.append({**r, "hit_cost": hit_cost, "net_gain": round(net_gain, 2),
                              "justified": True, "paid": True})

    return {
        "moves": moves,
        "profile_used": profile_name,
        "hit_cost_threshold": threshold,
        "hit_stance": hit_stance,
        "free_transfers": free_transfers,
    }


# ---------------------------------------------------------------------------
# Chess-themed season verdict — deterministic phrase bank, no external AI call.
# ---------------------------------------------------------------------------
_OPENINGS = [
    ("rank_climbing_no_hits", "The Long Endgame",
     "A patient climb: no hits taken, tight bench management, and a rank trajectory that's quietly improved."),
    ("rank_climbing_with_hits", "The Calculated Gambit",
     "Points spent to win the position — hits taken, and the rank trend says they've paid off so far."),
    ("rank_falling_stable_squad", "A Slow Retreat",
     "The squad hasn't collapsed, but the clock is running — rank has drifted back over recent gameweeks."),
    ("rank_falling_high_hits", "Overextended",
     "Too many pieces moved too fast — repeated hits without the rank gains to justify them."),
    ("flat_early_season", "The Opening Book",
     "Early days — the position is still being built. Too soon for a verdict, not too soon for a plan."),
    ("rank_stable_strong", "Consolidation",
     "No fireworks, no damage — a settled position banking points quietly while others thrash around it."),
]


def chess_verdict(rank_history: list[int], hits_last_n: int, current_gw: int) -> dict:
    """rank_history: overall rank per finished GW, oldest first (lower=better).
    Deterministic rule-based classification -> a fixed phrase pair. No LLM
    call, so this stays inside the tool's zero-cost design."""
    if current_gw <= 3 or len(rank_history) < 3:
        key = "flat_early_season"
    else:
        recent = rank_history[-3:]
        improving = recent[0] > recent[-1]
        worsening = recent[0] < recent[-1]
        if improving and hits_last_n == 0:
            key = "rank_climbing_no_hits"
        elif improving and hits_last_n > 0:
            key = "rank_climbing_with_hits"
        elif worsening and hits_last_n >= 2:
            key = "rank_falling_high_hits"
        elif worsening:
            key = "rank_falling_stable_squad"
        else:
            key = "rank_stable_strong"

    for k, headline, body in _OPENINGS:
        if k == key:
            return {"headline": headline, "body": body, "key": key}
    return {"headline": "The Opening Book", "body": "Too soon for a verdict, not too soon for a plan.", "key": "flat_early_season"}
