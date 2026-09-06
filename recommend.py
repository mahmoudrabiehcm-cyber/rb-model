"""
recommend.py
Transfer suggestions (Step 7a) and the one/two-phrase season verdict —
football-commentary language (Patch 2; was chess-themed before), keeping
the same considered, situational-headline structure. Both are plain
deterministic logic — no external AI call, so both stay inside the
zero-cost design (an LLM-generated verdict would need a paid API key; this
doesn't).

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
import optimizer as opt
import style_profiles


def _pair_moves(old_squad: pd.DataFrame, new_squad: pd.DataFrame, this_gw_col: str) -> list[dict]:
    """Turns an OUT-code-set/IN-code-set (the diff between two 15-man squads
    from the joint solver) into a readable move-by-move list, paired by
    position. Rule #36 (Squad-Legality Validation) is already satisfied by
    construction — both squads came out of the same formation-constrained
    MILP — so this is purely presentational: sort each side by price
    (descending) within a position and zip, so a big-money OUT reads
    alongside the big-money IN that replaced it rather than a random pairing."""
    old_codes = set(old_squad["code"])
    new_codes = set(new_squad["code"])
    out_df = old_squad[old_squad["code"].isin(old_codes - new_codes)]
    in_df = new_squad[new_squad["code"].isin(new_codes - old_codes)]

    # Mechanical check (Rule #36): the position multisets on each side must
    # match exactly, or something upstream broke the formation constraint.
    assert sorted(out_df["position"]) == sorted(in_df["position"]), (
        "Squad-legality check failed: OUT/IN position multisets don't match "
        f"({sorted(out_df['position'])} vs {sorted(in_df['position'])}) — the joint "
        "solve should never produce this; treat as a bug, not a valid transfer plan.")

    pairs = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        outs = out_df[out_df["position"] == pos].sort_values("price", ascending=False).to_dict("records")
        ins = in_df[in_df["position"] == pos].sort_values("price", ascending=False).to_dict("records")
        for o, i in zip(outs, ins):
            o_gw = o.get(this_gw_col, 0) or 0
            i_gw = i.get(this_gw_col, 0) or 0
            pairs.append({
                "position": pos,
                "out_code": o["code"], "out": o["web_name"], "out_team": o["team"], "out_price": o["price"],
                "out_xpts": o.get("xpts_horizon_sum", 0.0) or 0.0, "out_gw": 0 if pd.isna(o_gw) else o_gw,
                "in_code": i["code"], "in": i["web_name"], "in_team": i["team"], "in_price": i["price"],
                "in_xpts": i.get("xpts_horizon_sum", 0.0) or 0.0, "in_gw": 0 if pd.isna(i_gw) else i_gw,
                "in_eo": i.get("selected_by_percent"),
                "setpiece_flag": bool(i.get("setpiece_flag", False)),
            })
    return pairs


def _apply_eo_pull(pairs: list[dict], full_pool: pd.DataFrame, profile: dict, cfg: dict,
                    this_gw_col: str, out_codes: set, in_codes: set) -> list[dict]:
    """Style-Fit tie-break (Rule #23 style consistency), applied AFTER the
    joint xPts solve has already picked the best squad, never instead of
    it: for each chosen IN player, look for a same-position, same-or-
    cheaper alternative in the full pool that is genuinely within
    margin-of-error (Rule #34) of that player's own projected xPts. Among
    those statistically-tied alternatives, substitute per the profile's
    `eo_pull` direction — "none" (Balanced / Pure xPts) never substitutes,
    "strong_high_eo" favours the safest/highest-owned tied option, and the
    low-EO profiles favour the most differential tied option that ALSO
    clears `style_profiles.differential_floor()` — a merit bar, not a
    relaxation, per the model doc's explicit caveat that EO pull should
    never downgrade genuine projection for a cheaper narrative."""
    eo_pull = profile.get("eo_pull", "none")
    if eo_pull == "none" or not pairs:
        return pairs
    ceiling_key = profile.get("differential_ceiling")
    taken_in_codes = set(in_codes)  # never pick a code the solve already used elsewhere in this batch

    result = []
    for p in pairs:
        pos = p["position"]
        pos_pool = full_pool[(full_pool["position"] == pos) &
                              (full_pool.get("status", pd.Series(dtype=object)) == "a")].copy()
        pos_pool = pos_pool[pos_pool["code"] != p["in_code"]]
        pos_pool = pos_pool[~pos_pool["code"].isin(taken_in_codes | out_codes)]
        moe = eng.margin_of_error_threshold(p["in_xpts"], cfg)
        tied = pos_pool[(pos_pool["xpts_horizon_sum"] >= (p["in_xpts"] - moe)) &
                         (pos_pool["price"] <= p["in_price"])]

        if eo_pull != "strong_high_eo":
            floor = style_profiles.differential_floor(pos_pool, "xpts_horizon_sum", ceiling_key)
            tied = tied[tied["xpts_horizon_sum"] >= floor]

        if tied.empty:
            result.append(p)
            taken_in_codes.add(p["in_code"])
            continue

        best = (tied.sort_values("selected_by_percent", ascending=False).iloc[0] if eo_pull == "strong_high_eo"
                else tied.sort_values("selected_by_percent", ascending=True).iloc[0])
        if best["code"] == p["in_code"]:
            result.append(p)
            taken_in_codes.add(p["in_code"])
            continue

        gw_val = best.get(this_gw_col, 0)
        gw_val = 0 if pd.isna(gw_val) else gw_val
        result.append({**p,
                        "in_code": best["code"], "in": best["web_name"], "in_team": best["team"],
                        "in_price": best["price"], "in_xpts": best.get("xpts_horizon_sum", 0.0) or 0.0,
                        "in_gw": gw_val, "in_eo": best.get("selected_by_percent"),
                        "setpiece_flag": bool(best.get("setpiece_flag", False)),
                        "eo_pull_applied": True})
        taken_in_codes.add(best["code"])
    return result


def suggest_transfers(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                       profile_name: str, hit_stance: str, free_transfers: int,
                       bank: float, current_gw: int, gw_list: list[int],
                       forced_count: int | None = None,
                       meaningful_bar: float | None = None,
                       bench_codes: set | None = None,
                       chip_advisory: str | None = None) -> dict:
    """Patch 3 — joint multi-transfer optimization (Standing Rules #28/#30/
    #34/#35/#36), replacing the old pairwise best-single-swap-per-slot
    heuristic entirely:

    - Rule #30 (Full-Pool Default): solves against squad+pool COMBINED via
      `optimizer.solve_squad()`, never a pre-filtered "best replacement per
      OUT slot" subset — the old approach could miss a genuinely better
      3-for-3 reshuffle because no single swap in isolation looked best.
    - Rule #28 (Joint Transfer Optimization): a k-transfer path is solved
      as one MILP (`retain_pool_codes=current squad, min_retain=15-k`),
      not k independent pairwise choices.
    - Rule #35 (Transfer-Count Optimization): every k in the stance's
      allowed range is solved and compared as a genuine candidate —
      fewest-transfers can win outright, including 0 (roll).
    - Rule #34 (Margin-of-Error Tie Rule): among k's whose net gain is
      within `fpl_engine.margin_of_error_threshold()` of the best net gain
      found, the SMALLEST k wins — a bigger k only gets picked if it clears
      noise by a real margin, not by a decimal.
    - Rule #36 (Squad-Legality Validation): `_pair_moves()` mechanically
      asserts the OUT/IN position multisets match before returning a plan.

    The old Bench Value Rule (Standing Rule #12) autosub-discount heuristic
    is retired here (flagged to the manager, not silently dropped): each
    player's own `xm` already prices in their expected minutes within
    `xpts_horizon_sum`, so a bench player's low game-time is already
    reflected in a low projection — a separate flat discount on top of that
    double-counts the same signal. `bench_codes` is accepted for backward
    compatibility but no longer changes the math.

    Free transfers are never auto-spent just because they're banked
    (Free-Transfer Materiality Rule, `transfer.minimum_meaningful_gain_free`):
    a free move still has to clear that horizon-xPts bar in "No hits" /
    "Hit if worth it" mode or the model recommends Roll instead. "Force"
    bypasses the bar on purpose — that mode exists for the manager to
    override the model, not the other way round.

    `chip_advisory`: an optional pre-built sentence (the caller already has
    chip status/DGW data) appended to the plan as-is — this function stays
    agnostic of chip internals, it just surfaces what it's given.

    Pass `meaningful_bar` to override the config default from the UI; leave
    it None to use model_config.yaml's value."""
    profile = style_profiles.get_profile(profile_name)
    hit_cost_per = cfg["transfer"]["hit_cost_per_transfer"]
    threshold = profile["hit_cost_threshold"]
    if meaningful_bar is None:
        meaningful_bar = cfg["transfer"].get("minimum_meaningful_gain_free", 1.5)
    bench_discount = cfg["transfer"].get("bench_autosub_discount", 0.2)  # kept for return-dict compatibility only
    this_gw_col = f"xpts_gw{current_gw}"
    horizon_n = len(gw_list)

    empty_result = {
        "moves": [], "plan": [], "profile_used": profile_name, "hit_cost_threshold": threshold,
        "minimum_meaningful_gain_free": meaningful_bar, "bench_autosub_discount": bench_discount,
        "hit_stance": hit_stance, "free_transfers": free_transfers,
    }

    if squad_df is None or squad_df.empty or "code" not in squad_df.columns:
        return empty_result

    # Defensive numeric coercion — a None (rather than NaN) price/xPts value
    # anywhere in these columns turns a pandas comparison into a TypeError
    # ("'<=' not supported between instances of 'NoneType' and 'float'").
    # Coercing explicitly makes any such row a clean, comparison-safe NaN
    # instead of crashing the whole page over one bad row.
    squad_df = squad_df.copy()
    pool_df = pool_df.copy() if pool_df is not None else pd.DataFrame(columns=squad_df.columns)
    for df in (squad_df, pool_df):
        for col in ("price", "xpts_horizon_sum", this_gw_col):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    bank = 0.0 if bank is None or pd.isna(bank) else float(bank)

    current_codes = list(squad_df["code"])
    out_codes_all = set(current_codes)
    old_total = squad_df["xpts_horizon_sum"].sum(skipna=True)
    old_total = 0.0 if pd.isna(old_total) else float(old_total)
    team_value = round(bank + (squad_df["price"].sum(skipna=True) or 0.0), 1)

    full_pool = pd.concat([squad_df, pool_df], ignore_index=True, sort=False)
    if "code" in full_pool.columns:
        full_pool = full_pool.drop_duplicates(subset=["code"], keep="first")

    moe = eng.margin_of_error_threshold(old_total, cfg)

    if hit_stance == "Force":
        k_wanted = forced_count if forced_count is not None else free_transfers
        k_range = [k_wanted]
    elif hit_stance == "No hits":
        k_range = list(range(0, free_transfers + 1))
    else:  # "Hit if worth it"
        k_range = list(range(0, 6))

    candidates = {0: {"squad": squad_df, "total": old_total, "hit_cost": 0.0, "net_gain": 0.0, "actual_k": 0}}
    for k in k_range:
        if k <= 0:
            continue
        min_retain = max(0, 15 - k)
        result = opt.solve_squad(full_pool, cfg, budget=team_value, retain_pool_codes=current_codes,
                                  min_retain=min_retain, objective_col="xpts_horizon_sum")
        if result is None:
            continue
        new_squad = result["squad"]
        actual_k = len(out_codes_all - set(new_squad["code"]))
        if actual_k == 0:
            continue  # solver found nothing worth swapping at this k — already covered by k=0
        hit_cost = hit_cost_per * max(0, actual_k - free_transfers)
        net_gain = round(result["total_xpts"] - old_total - hit_cost, 2)
        # keyed by actual_k so two requested k's that land on the same real
        # swap count don't create a spurious "tie" against themselves
        if actual_k not in candidates or net_gain > candidates[actual_k]["net_gain"]:
            candidates[actual_k] = {"squad": new_squad, "total": result["total_xpts"],
                                     "hit_cost": hit_cost, "net_gain": net_gain, "actual_k": actual_k}

    plan = []
    moves = []

    if hit_stance == "Force":
        chosen_k = k_wanted if k_wanted in candidates else max(candidates, key=lambda k: 0 if k != k_wanted else 1)
        # if the exact forced count wasn't solvable, fall back to the closest
        # smaller count that was, rather than silently doing nothing
        if chosen_k != k_wanted:
            smaller = [k for k in candidates if k <= k_wanted]
            chosen_k = max(smaller) if smaller else 0
        chosen = candidates[chosen_k]
        if chosen["actual_k"] == 0:
            plan.append(f"GW{current_gw}: Forced transfer requested, but no legal improving swap was found in "
                        f"the full pool at that count — squad unchanged this run.")
        else:
            pairs = _pair_moves(squad_df, chosen["squad"], this_gw_col)
            pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes_all,
                                    {p["in_code"] for p in pairs})
            moves = [_move_row(p, chosen["hit_cost"], chosen["net_gain"], True) for p in pairs]
            plan.append(f"GW{current_gw}: Forced {chosen['actual_k']} transfer(s) — bypasses the materiality bar "
                        f"by design; net {chosen['net_gain']:+.2f} xPts after the {chosen['hit_cost']:.0f}-pt "
                        f"hit ({old_total:.1f} → {chosen['total']:.1f} xPts over {horizon_n} GW(s)).")

    else:
        best_net = max(c["net_gain"] for c in candidates.values())

        def clears_bar(c):
            if c["actual_k"] == 0:
                return True
            bar = threshold if c["hit_cost"] > 0 else meaningful_bar
            return c["net_gain"] >= bar

        tied_ks = sorted(k for k, c in candidates.items() if (best_net - c["net_gain"]) < moe)
        viable = [k for k in tied_ks if clears_bar(candidates[k])]
        chosen_k = min(viable) if viable else 0
        chosen = candidates[chosen_k]

        if chosen["actual_k"] == 0:
            best_alt_k = max((k for k in candidates if k != 0), key=lambda k: candidates[k]["net_gain"], default=None)
            if best_alt_k is not None and candidates[best_alt_k]["actual_k"] > 0:
                alt = candidates[best_alt_k]
                bar = threshold if alt["hit_cost"] > 0 else meaningful_bar
                plan.append(f"GW{current_gw}: Roll — best alternative found ({alt['actual_k']} move(s), jointly "
                            f"optimized across the full pool) nets {alt['net_gain']:+.2f} xPts after cost, below "
                            f"the {bar} xPts bar. Bank free transfer(s) (up to 5) for a move that actually "
                            f"clears it.")
            else:
                plan.append(f"GW{current_gw}: Roll — no improving swap found in the full pool this run. "
                            f"Reassess next gameweek once prices/fixtures move.")
        else:
            pairs = _pair_moves(squad_df, chosen["squad"], this_gw_col)
            pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes_all,
                                    {p["in_code"] for p in pairs})
            moves = [_move_row(p, chosen["hit_cost"], chosen["net_gain"], True) for p in pairs]
            hit_note = (f" — {chosen['hit_cost']:.0f}-pt hit taken, clears the {profile_name} profile's "
                        f"{threshold} xPts hit-cost threshold" if chosen["hit_cost"] > 0 else "")
            eo_note = " (one leg adjusted for style fit — see `eo_pull_applied` rows)" if any(
                p.get("eo_pull_applied") for p in pairs) else ""
            plan.append(f"GW{current_gw}: {chosen['actual_k']} transfer(s) — jointly optimized across the full "
                        f"player pool (Rules #28/#30), net {chosen['net_gain']:+.2f} xPts over {horizon_n} "
                        f"GW(s){hit_note}{eo_note}. Fewest-transfers tie-break applied within the {moe:.1f} "
                        f"xPts margin-of-error band (Rules #34/#35).")
            used_free = min(chosen["actual_k"], free_transfers)
            rolled = free_transfers - used_free
            if rolled > 0:
                plan.append(f"GW{current_gw}: {rolled} free transfer(s) banked (up to 5) after this move.")

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


def _move_row(p: dict, hit_cost: float, net_gain: float, justified: bool) -> dict:
    """One pair (from `_pair_moves`/`_apply_eo_pull`) -> a move-table row.
    `hit_cost`/`net_gain` are the BATCH total for the whole chosen transfer
    path, repeated on every row in the batch (not summed per-row) — a joint
    k-transfer solve incurs one combined hit fee, not k independent ones,
    so splitting it across rows would misstate what any single row cost on
    its own. The accompanying plan-text line states the batch total once."""
    return {
        "out": p["out"], "out_team": p["out_team"], "out_price": p["out_price"],
        "in": p["in"], "in_team": p["in_team"], "in_price": p["in_price"],
        "position": p["position"],
        "xpts_gain": round(p["in_xpts"] - p["out_xpts"], 2),
        "xpts_gain_this_gw": round(p["in_gw"] - p["out_gw"], 2),
        "in_eo": p["in_eo"], "setpiece_flag": p["setpiece_flag"],
        "hit_cost": hit_cost, "net_gain": net_gain, "justified": justified,
        "eo_pull_applied": bool(p.get("eo_pull_applied", False)),
    }


# ---------------------------------------------------------------------------
# Season verdict — deterministic phrase bank, no external AI call. Football-
# commentary vocabulary (Patch 2): each headline is a genuine footballing
# phrase for the situation it describes, not an invented metaphor — "losing
# the run of play" and "chasing the game" mean exactly what they say on any
# football broadcast, they just map naturally onto a rank/hits classification
# the same way chess's opening/middlegame/endgame language used to.
# ---------------------------------------------------------------------------
_VERDICTS = [
    ("rank_climbing_no_hits", "The Patient Build",
     "A patient climb: no hits taken, tight bench management, and a rank trajectory that's quietly improved."),
    ("rank_climbing_with_hits", "The Calculated Punt",
     "Points spent to push the position forward — hits taken, and the rank trend says they've paid off so far."),
    ("rank_falling_stable_squad", "Losing the Run of Play",
     "The squad hasn't collapsed, but the clock is running — rank has drifted back over recent gameweeks."),
    ("rank_falling_high_hits", "Chasing the Game",
     "Too many changes made too fast — repeated hits without the rank gains to justify them."),
    ("flat_early_season", "Finding Your XI",
     "Early days — the XI is still taking shape. Too soon for a verdict, not too soon for a plan."),
    ("rank_stable_strong", "Game Management",
     "No fireworks, no damage — a settled team banking points quietly while others thrash around it."),
]


def season_verdict(rank_history: list[int], hits_last_n: int, current_gw: int) -> dict:
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

    for k, headline, body in _VERDICTS:
        if k == key:
            return {"headline": headline, "body": body, "key": key}
    return {"headline": "Finding Your XI", "body": "Too soon for a verdict, not too soon for a plan.", "key": "flat_early_season"}
