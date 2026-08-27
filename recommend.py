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
                       bank: float, current_gw: int, gw_list: list[int],
                       forced_count: int | None = None,
                       meaningful_bar: float | None = None,
                       bench_codes: set | None = None,
                       chip_advisory: str | None = None) -> dict:
    """Free transfers are never auto-spent just because they're banked
    (Free-Transfer Materiality Rule, model_config.yaml `transfer.
    minimum_meaningful_gain_free`): a move has to clear that horizon-xPts
    bar in "No hits" / "Hit if worth it" mode or the model recommends
    Roll instead. "Force" bypasses the bar on purpose — that mode exists
    for the manager to override the model, not the other way round.

    Bench Value Rule (Standing Rule #12): pass `bench_codes` (the set of
    player codes currently sitting on the bench) and a swap whose OUT
    player is one of them gets its xPts gain discounted by
    `transfer.bench_autosub_discount` before the materiality check — a
    big raw gain on a player who mostly doesn't play barely moves your
    actual score, so it shouldn't clear the same bar a starter swap does.

    `chip_advisory`: an optional pre-built sentence (the caller already
    has chip status/DGW data) appended to the plan as-is — this function
    stays agnostic of chip internals, it just surfaces what it's given.

    Pass `meaningful_bar` to override the config default from the UI;
    leave it None to use model_config.yaml's value."""
    profile = style_profiles.get_profile(profile_name)
    hit_cost_per = cfg["transfer"]["hit_cost_per_transfer"]
    threshold = profile["hit_cost_threshold"]
    if meaningful_bar is None:
        meaningful_bar = cfg["transfer"].get("minimum_meaningful_gain_free", 1.5)
    bench_discount = cfg["transfer"].get("bench_autosub_discount", 0.2)
    bench_codes = bench_codes or set()
    this_gw_col = f"xpts_gw{current_gw}"

    # Defensive numeric coercion — a None (rather than NaN) price/xPts value
    # anywhere in these columns turns a pandas comparison into a TypeError
    # ("'<=' not supported between instances of 'NoneType' and 'float'").
    # Coercing explicitly makes any such row a clean, comparison-safe NaN
    # instead of crashing the whole page over one bad row.
    squad_df = squad_df.copy()
    pool_df = pool_df.copy()
    for df in (squad_df, pool_df):
        for col in ("price", "xpts_horizon_sum", this_gw_col):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    bank = 0.0 if bank is None or pd.isna(bank) else float(bank)

    raw = []
    for _, out_p in squad_df.iterrows():
        if pd.isna(out_p.get("price")) or pd.isna(out_p.get("xpts_horizon_sum")):
            continue  # can't evaluate a swap against an incomplete row — skip, don't crash
        is_bench_out = out_p.get("code") in bench_codes
        same_pos = pool_df[(pool_df["position"] == out_p["position"]) &
                            (pool_df["status"] == "a") & pool_df["price"].notna()]
        budget_cap = out_p["price"] + bank
        afford = same_pos[same_pos["price"] <= budget_cap]
        for _, in_p in afford.iterrows():
            if pd.isna(in_p.get("xpts_horizon_sum")):
                continue
            gain = round(in_p["xpts_horizon_sum"] - out_p["xpts_horizon_sum"], 2)
            if gain <= 0:
                continue
            effective_gain = round(gain * bench_discount, 2) if is_bench_out else gain
            in_gw = in_p.get(this_gw_col, 0)
            out_gw = out_p.get(this_gw_col, 0)
            in_gw = 0 if pd.isna(in_gw) else in_gw
            out_gw = 0 if pd.isna(out_gw) else out_gw
            gain_this_gw = round(in_gw - out_gw, 2)
            raw.append({
                "out": out_p["web_name"], "out_team": out_p["team"], "out_price": out_p["price"],
                "in": in_p["web_name"], "in_team": in_p["team"], "in_price": in_p["price"],
                "position": out_p["position"], "xpts_gain": gain, "effective_gain": effective_gain,
                "xpts_gain_this_gw": gain_this_gw, "is_bench_out": is_bench_out,
                "in_eo": in_p.get("selected_by_percent"),
                "setpiece_flag": bool(in_p.get("setpiece_flag", False)),
            })

    cols = ["out", "out_team", "out_price", "in", "in_team", "in_price",
            "position", "xpts_gain", "effective_gain", "xpts_gain_this_gw", "is_bench_out",
            "in_eo", "setpiece_flag"]
    ranked = pd.DataFrame(raw, columns=cols)
    if not ranked.empty:
        # one suggestion per OUT player at most (best replacement for that slot),
        # ranked by effective (bench-discounted) gain — a bench-only upgrade
        # shouldn't outrank a smaller but real starter upgrade.
        ranked = ranked.sort_values("effective_gain", ascending=False).drop_duplicates(subset=["out"], keep="first")

    plan = []
    horizon_n = len(gw_list)

    if hit_stance == "Force":
        n = forced_count if forced_count is not None else free_transfers
        chosen = ranked.head(n)
        moves = []
        for i, r in enumerate(chosen.to_dict("records")):
            hit_cost = hit_cost_per * max(0, i + 1 - free_transfers)
            net = r["xpts_gain"] - hit_cost
            moves.append({**r, "hit_cost": hit_cost, "net_gain": round(net, 2),
                          "justified": net >= threshold if hit_cost > 0 else True, "paid": hit_cost > 0})
        plan.append(f"GW{current_gw}: Forced {len(moves)} transfer(s) — bypasses the materiality bar by design; "
                     f"check `justified` per row before locking these in.")

    else:
        # Gate on effective (bench-discounted) gain, not raw — a bench-only
        # upgrade has to clear the bar on its real, autosub-weighted value.
        meets_bar = ranked[ranked["effective_gain"] >= meaningful_bar] if not ranked.empty else ranked
        free_moves = meets_bar.head(free_transfers).to_dict("records")
        moves = [{**r, "hit_cost": 0, "justified": True, "paid": False} for r in free_moves]
        used_free = len(free_moves)
        rolled = free_transfers - used_free

        for r in free_moves:
            bench_note = (f" (bench swap — {r['xpts_gain']} raw discounted to {r['effective_gain']} "
                          f"at {int(bench_discount*100)}% autosub weighting, still clears the bar)"
                          if r["is_bench_out"] else "")
            plan.append(f"GW{current_gw}: {r['out']} → {r['in']} — "
                        f"+{r['xpts_gain_this_gw']} xPts this GW, +{r['xpts_gain']} over {horizon_n} GW(s)"
                        f"{bench_note}. Clears the {meaningful_bar} xPts bar — worth the free transfer.")

        if rolled > 0:
            taken_out_names = {r["out"] for r in free_moves}
            rejected = ranked[~ranked["out"].isin(taken_out_names)] if not ranked.empty else ranked
            best_rejected = rejected.iloc[0].to_dict() if not rejected.empty else None
            if best_rejected:
                bench_note = (f" — it's a bench-only upgrade ({best_rejected['xpts_gain']} raw, only "
                              f"{best_rejected['effective_gain']} once autosub-weighted since that player "
                              f"isn't in your starting XI)" if best_rejected["is_bench_out"] else "")
                plan.append(f"GW{current_gw}: Roll {rolled} free transfer(s) — best remaining option "
                            f"({best_rejected['out']} → {best_rejected['in']}) only gains "
                            f"+{best_rejected['effective_gain']} (effective) over {horizon_n} GW(s), below the "
                            f"{meaningful_bar} xPts bar{bench_note}. Bank it (up to 5) for a move that "
                            f"actually clears it.")
            else:
                plan.append(f"GW{current_gw}: Roll {rolled} free transfer(s) — no upgrade found in the pool "
                            f"this run. Reassess next gameweek once prices/fixtures move.")

        if hit_stance == "Hit if worth it":
            excluded = {r["out"] for r in free_moves}
            extra_pool = ranked[~ranked["out"].isin(excluded)].iloc[:2] if not ranked.empty else ranked
            for r in extra_pool.to_dict("records"):
                hit_cost = hit_cost_per
                net_gain = r["effective_gain"] - hit_cost
                if net_gain >= threshold:
                    moves.append({**r, "hit_cost": hit_cost, "net_gain": round(net_gain, 2),
                                  "justified": True, "paid": True})
                    plan.append(f"GW{current_gw}: paid move {r['out']} → {r['in']} — net "
                                f"+{round(net_gain,2)} xPts after the {hit_cost}-pt hit, clears the "
                                f"{profile_name} profile's {threshold} xPts hit-cost threshold.")

    if chip_advisory:
        plan.append(chip_advisory)

    return {
        "moves": moves,
        "plan": plan,
        "profile_used": profile_name,
        "hit_cost_threshold": threshold,
        "minimum_meaningful_gain_free": meaningful_bar,
        "bench_autosub_discount": bench_discount,
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
