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
import transfers


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


def plan_transfer_schedule(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                            profile_name: str, free_transfers: int, bank: float,
                            current_gw: int, gw_list: list[int],
                            meaningful_bar: float | None = None,
                            chip_advisory: str | None = None) -> dict:
    """No-hits, multi-GW pacing plan (project discussion, 2026-09-07) — see
    the call site in `suggest_transfers()` for why this exists. Simulates
    forward through every GW in `gw_list`:

    - Free-transfer accrual is modeled honestly: +1 FT for a week that isn't
      fully used, capped at `transfers.MAX_BANK` (5) — the actual 2026/27
      rule (`transfers.derive_free_transfers` already encodes this for
      deriving the manager's CURRENT count; this reuses the same cap for
      projecting it FORWARD).
    - CHAINS: each week's solve uses the squad that resulted from every
      prior week's chosen move in this same schedule (`sim_squad`), so the
      plan stays internally consistent — never suggests selling a player
      twice, budget/club-limits reflect the squad as it would actually
      stand that week if the plan were followed in order.
    - Never proposes a hit. If no free move clears the materiality bar in a
      given week (Rules #13/#34, same bars as `suggest_transfers()`), that
      week is Roll and the free transfer banks forward (up to the cap).
    - Every week in the schedule is stated with equal weight — this is not
      hedged as "GW-now is real, later GWs are placeholders" (manager's
      explicit choice). It is still, structurally, built from this run's
      own projections and gets recomputed fresh the next time the model
      runs, same as every other output in this tool.
    - `bank` is held constant across the simulated horizon (no attempt to
      project future price rises/sale proceeds) — the same disclosed
      simplification `suggest_transfers()` already carries (see this
      module's top-of-file docstring).

    Returns the same top-level keys `suggest_transfers()` returns (so
    existing callers/UI code work unchanged), plus `weekly_plan`: a list of
    one dict per GW — {gw, moves, net_gain, ft_available, ft_used,
    ft_banked_after, summary, data_gap_note} — and `is_weekly_schedule`:
    True, so a caller can distinguish this shape from the single-decision
    return if it wants to render it differently."""
    profile = style_profiles.get_profile(profile_name)
    if meaningful_bar is None:
        meaningful_bar = cfg["transfer"].get("minimum_meaningful_gain_free", 2.0)
    bench_discount = cfg["transfer"].get("bench_autosub_discount", 0.2)

    empty_result = {
        "moves": [], "plan": [], "summary": [], "net_gain": 0.0, "profile_used": profile_name,
        "hit_cost_threshold": profile["hit_cost_threshold"], "minimum_meaningful_gain_free": meaningful_bar,
        "bench_autosub_discount": bench_discount, "hit_stance": "No hits", "free_transfers": free_transfers,
        "margin_of_error": eng.margin_of_error_threshold(0.0, cfg),
        "weekly_plan": [], "is_weekly_schedule": True,
    }
    if squad_df is None or squad_df.empty or "code" not in squad_df.columns:
        return empty_result

    squad_df = squad_df.copy()
    pool_df = pool_df.copy() if pool_df is not None else pd.DataFrame(columns=squad_df.columns)
    numeric_cols = {"price", "xpts_horizon_sum"} | {f"xpts_gw{g}" for g in gw_list}
    for df in (squad_df, pool_df):
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    bank = 0.0 if bank is None or pd.isna(bank) else float(bank)

    full_pool = pd.concat([squad_df, pool_df], ignore_index=True, sort=False)
    if "code" in full_pool.columns:
        full_pool = full_pool.drop_duplicates(subset=["code"], keep="first")

    sim_squad = squad_df.copy()
    ft_bank = free_transfers
    weekly_plan = []
    plan = []
    summary = []
    total_net_gain = 0.0

    for wi, gw in enumerate(gw_list):
        remaining_gws = gw_list[wi:]
        this_gw_col = f"xpts_gw{gw}"
        current_codes = list(sim_squad["code"])
        out_codes_all = set(current_codes)
        team_value = round(bank + (sim_squad["price"].sum(skipna=True) or 0.0), 1)
        old_total = opt.realized_horizon_value(sim_squad, remaining_gws, cfg)
        moe = eng.margin_of_error_threshold(old_total, cfg)

        candidates = {0: {"squad": sim_squad, "total": old_total, "net_gain": 0.0,
                          "actual_k": 0, "data_gap_codes": []}}
        for k in range(1, ft_bank + 1):
            min_retain = max(0, 15 - k)
            result = opt.solve_squad(full_pool, cfg, budget=team_value, retain_pool_codes=current_codes,
                                      min_retain=min_retain, objective_col="xpts_horizon_sum")
            if result is None:
                continue
            new_squad = result["squad"]
            actual_k = len(out_codes_all - set(new_squad["code"]))
            if actual_k == 0:
                continue  # nothing worth swapping at this k — already covered by k=0
            new_total = opt.realized_horizon_value(new_squad, remaining_gws, cfg)
            net_gain = round(new_total - old_total, 2)  # never a hit cost in this no-hits path
            if actual_k not in candidates or net_gain > candidates[actual_k]["net_gain"]:
                candidates[actual_k] = {"squad": new_squad, "total": new_total, "net_gain": net_gain,
                                         "actual_k": actual_k, "data_gap_codes": result.get("data_gap_codes", [])}

        best_net = max(c["net_gain"] for c in candidates.values())
        tied_ks = sorted(k for k, c in candidates.items() if (best_net - c["net_gain"]) < moe)
        viable = [k for k in tied_ks if k == 0 or candidates[k]["net_gain"] >= meaningful_bar]
        chosen_k = min(viable) if viable else 0
        chosen = candidates[chosen_k]

        ft_used = min(chosen["actual_k"], ft_bank)
        ft_after = min(transfers.MAX_BANK, (ft_bank - ft_used) + 1)

        gap_codes = chosen.get("data_gap_codes", [])
        data_gap_note = None
        if gap_codes:
            gap_names = full_pool[full_pool["code"].isin(gap_codes)]["web_name"].tolist()
            names_txt = ", ".join(gap_names) if gap_names else f"{len(gap_codes)} player(s)"
            data_gap_note = (f"Data gap flagged: {names_txt} had a missing projection this week and was "
                              f"treated as 0 xPts so it wouldn't be silently forced out (Standing Rule #4).")

        if chosen["actual_k"] == 0:
            week_moves = []
            week_summary = (f"GW{gw}: Roll — no free move clears the bar this week. "
                             f"{ft_bank} FT banked → {ft_after} for GW{gw + 1}.")
        else:
            pairs = _pair_moves(sim_squad, chosen["squad"], this_gw_col)
            week_moves = [{**_move_row(p, 0.0, chosen["net_gain"], True), "gw": gw} for p in pairs]
            move_bits = ", ".join(f"{p['out']} → {p['in']}" for p in pairs)
            week_summary = (f"GW{gw}: {move_bits} (free) — net {chosen['net_gain']:+.1f} xPts over the "
                             f"remaining horizon. {ft_bank - ft_used} FT left banked → {ft_after} for GW{gw + 1}.")
            sim_squad = chosen["squad"]

        if data_gap_note:
            week_summary += f" {data_gap_note}"

        weekly_plan.append({
            "gw": gw, "moves": week_moves, "net_gain": chosen["net_gain"],
            "ft_available": ft_bank, "ft_used": ft_used, "ft_banked_after": ft_after,
            "summary": week_summary, "data_gap_note": data_gap_note,
        })
        summary.append(week_summary)
        plan.append(f"GW{gw}: {chosen['actual_k']} free transfer(s) this week (chained pacing plan) — "
                    f"net {chosen['net_gain']:+.2f} xPts vs. the squad as it stood after GW{gw - 1}'s "
                    f"suggested move. Fewest-transfers tie-break within the {moe:.1f} xPts margin-of-error "
                    f"band (Rules #34/#35), scored on realized (bench-discounted) value (Rule #12).")
        total_net_gain += chosen["net_gain"]
        ft_bank = ft_after

    if chip_advisory:
        summary.append(chip_advisory)
        plan.append(chip_advisory)

    return {
        "moves": [m for wk in weekly_plan for m in wk["moves"]],
        "plan": plan,
        "summary": summary,
        "net_gain": round(total_net_gain, 2),
        "profile_used": profile_name,
        "hit_cost_threshold": profile["hit_cost_threshold"],
        "minimum_meaningful_gain_free": meaningful_bar,
        "bench_autosub_discount": bench_discount,
        "hit_stance": "No hits",
        "free_transfers": free_transfers,
        "margin_of_error": eng.margin_of_error_threshold(0.0, cfg),
        "weekly_plan": weekly_plan,
        "is_weekly_schedule": True,
    }


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

    Bench Value Rule reinstated (Patch 5, reversing a Patch 3 error). Every
    squad comparison in this function — the current squad's own baseline
    and every k-transfer candidate — is scored via
    `optimizer.realized_horizon_value()`, which values each GW's best
    starting XI at full projection and the 4 bench slots at their
    autosub-discounted value (Standing Rule #12), never a bench player's raw
    "if he started every week" number. This is what actually fixes the
    reported failure mode: a bench-only swap (e.g. a backup-GK upgrade that
    never affects the starting XI) now nets close to zero real gain instead
    of being scored as if the swap were a starting-XI upgrade, so it
    correctly fails the materiality/hit-cost bar and "Roll" wins instead.
    `bench_codes` is kept for backward compatibility (unused by the math
    directly — the realized-value calculation re-derives who's actually
    bench per GW from each candidate squad's own best-XI solve, since bench
    membership can shift week to week even for a fixed 15).

    Note on scope: the MILP inside `optimizer.solve_squad()` still searches
    for candidate 15-man squads using the raw `xpts_horizon_sum` objective —
    that's a tractable way to explore the combinatorial squad space, and
    isn't itself where Rule #12 bites. The rule is enforced at the actual
    DECISION point: which candidate is scored best, and whether any
    transfer clears the bar, both computed from realized (bench-discounted)
    value below. Flagged here explicitly per Rule #4 ("show the inputs") —
    this is a disclosed engineering simplification, not a claim that every
    internal MILP coefficient itself carries the discount.

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
        meaningful_bar = cfg["transfer"].get("minimum_meaningful_gain_free", 2.0)
    bench_discount = cfg["transfer"].get("bench_autosub_discount", 0.2)  # kept for return-dict compatibility only
    this_gw_col = f"xpts_gw{current_gw}"
    horizon_n = len(gw_list)

    empty_result = {
        "moves": [], "plan": [], "summary": [], "net_gain": 0.0, "profile_used": profile_name,
        "hit_cost_threshold": threshold, "minimum_meaningful_gain_free": meaningful_bar,
        "bench_autosub_discount": bench_discount, "hit_stance": hit_stance, "free_transfers": free_transfers,
        "margin_of_error": eng.margin_of_error_threshold(0.0, cfg),
        "weekly_plan": [], "is_weekly_schedule": False,
    }

    if squad_df is None or squad_df.empty or "code" not in squad_df.columns:
        return empty_result

    # Project-discussion fix (2026-09-07): under "No hits" with a horizon
    # wider than 1 GW, the old single-lump output was genuinely ambiguous
    # about WHEN to make a move the model could see was worth it over the
    # full horizon but couldn't afford this week without a hit — it only
    # ever considered CURRENTLY banked free transfers
    # (`k_range = range(0, free_transfers + 1)`), never modeling that a 2nd
    # or 3rd good move becomes free once next week's free transfer accrues.
    # `plan_transfer_schedule()` replaces the single decision with a chained,
    # week-by-week pacing plan for exactly this combination (manager-
    # confirmed design: replace, not show alongside; chain each week off the
    # prior week's chosen squad; every week carries equal weight rather than
    # hedging later weeks as placeholders). "Hit if worth it" and "Force"
    # keep the original single-GW logic below unchanged — they can already
    # resolve multiple transfers in one go by paying for them, so this
    # specific ambiguity doesn't apply to them.
    if hit_stance == "No hits" and horizon_n > 1:
        return plan_transfer_schedule(squad_df, pool_df, cfg, profile_name, free_transfers, bank,
                                       current_gw, gw_list, meaningful_bar, chip_advisory)

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
    # Rule #12 (Bench Value Rule): the baseline is the squad's REALIZED
    # value (best XI + autosub-discounted bench, per GW, summed over the
    # horizon) — never a raw sum of all 15 players' full projections, which
    # is exactly the number that let a bench-only swap look like a genuine
    # upgrade in Patch 3.
    old_total = opt.realized_horizon_value(squad_df, gw_list, cfg)
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
        # Rule #12: score the candidate on its REALIZED value (best XI +
        # autosub-discounted bench), not the MILP's raw xpts_horizon_sum —
        # the MILP objective is only a search heuristic for finding
        # candidate squads, the realized value is what actually decides.
        new_total = opt.realized_horizon_value(new_squad, gw_list, cfg)
        net_gain = round(new_total - old_total - hit_cost, 2)
        # keyed by actual_k so two requested k's that land on the same real
        # swap count don't create a spurious "tie" against themselves
        if actual_k not in candidates or net_gain > candidates[actual_k]["net_gain"]:
            candidates[actual_k] = {"squad": new_squad, "total": new_total,
                                     "hit_cost": hit_cost, "net_gain": net_gain, "actual_k": actual_k,
                                     "data_gap_codes": result.get("data_gap_codes", [])}

    # `plan`: full technical trace (rule citations, candidate math) — kept
    # for the "How this was worked out" detail expander. `summary`: the
    # plain-language recommendation itself, one or two short lines, no rule
    # numbers — this is what's shown by default (manager feedback: the old
    # UI surfaced the trace as the primary content, which read as noise).
    plan = []
    summary = []
    moves = []
    chosen_net_gain = 0.0  # exposed on the return dict (Patch 6) so a manager
    # what-if evaluation can state how it compares to the model's own pick

    if hit_stance == "Force":
        chosen_k = k_wanted if k_wanted in candidates else max(candidates, key=lambda k: 0 if k != k_wanted else 1)
        # if the exact forced count wasn't solvable, fall back to the closest
        # smaller count that was, rather than silently doing nothing
        if chosen_k != k_wanted:
            smaller = [k for k in candidates if k <= k_wanted]
            chosen_k = max(smaller) if smaller else 0
        chosen = candidates[chosen_k]
        chosen_net_gain = chosen["net_gain"]
        if chosen["actual_k"] == 0:
            summary.append("No legal improving swap found at the forced transfer count — squad unchanged.")
            plan.append(f"GW{current_gw}: Forced transfer requested, but no legal improving swap was found in "
                        f"the full pool at that count — squad unchanged this run.")
        else:
            pairs = _pair_moves(squad_df, chosen["squad"], this_gw_col)
            pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes_all,
                                    {p["in_code"] for p in pairs})
            moves = [_move_row(p, chosen["hit_cost"], chosen["net_gain"], True) for p in pairs]
            summary.append(f"Forced: {chosen['actual_k']} transfer(s), net {chosen['net_gain']:+.1f} xPts "
                            f"after a {chosen['hit_cost']:.0f}-pt hit.")
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
        chosen_net_gain = chosen["net_gain"]

        if chosen["actual_k"] == 0:
            best_alt_k = max((k for k in candidates if k != 0), key=lambda k: candidates[k]["net_gain"], default=None)
            if best_alt_k is not None and candidates[best_alt_k]["actual_k"] > 0:
                alt = candidates[best_alt_k]
                bar = threshold if alt["hit_cost"] > 0 else meaningful_bar
                # Two genuinely different reasons can land on Roll here, and
                # the message must name the one that actually applied — a
                # real bug this fixes: the old text always cited `bar`
                # regardless of cause, which could print "below the 1.5 xPts
                # bar" for a move that nets +1.85 (i.e. NOT below it) when
                # the real reason was Rule #34's margin-of-error tie against
                # doing nothing at all, not the materiality/hit-cost bar.
                zero_tied = (best_net - candidates[0]["net_gain"]) < moe
                if zero_tied:
                    reason = (f"within this model's own margin-of-error ({moe:.1f} xPts) of making no change at "
                              f"all — not a confident enough edge over doing nothing to call it a genuine "
                              f"improvement (Standing Rule #34)")
                    summary_reason = f"within margin-of-error of no change ({moe:.1f} xPts)"
                else:
                    reason = f"below the {bar} xPts bar this move needs to clear"
                    summary_reason = "not enough to be worth it yet"
                summary.append(f"Roll your transfer(s) — the best available move nets "
                                f"{alt['net_gain']:+.1f} xPts, {summary_reason}.")
                plan.append(f"GW{current_gw}: Roll — best alternative found ({alt['actual_k']} move(s), jointly "
                            f"optimized across the full pool) nets {alt['net_gain']:+.2f} xPts after cost, {reason}. "
                            f"Bank free transfer(s) (up to 5) for a move that actually clears it.")
            else:
                summary.append("Roll your transfer(s) — no improving swap found this run.")
                plan.append(f"GW{current_gw}: Roll — no improving swap found in the full pool this run. "
                            f"Reassess next gameweek once prices/fixtures move.")
        else:
            pairs = _pair_moves(squad_df, chosen["squad"], this_gw_col)
            pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes_all,
                                    {p["in_code"] for p in pairs})
            moves = [_move_row(p, chosen["hit_cost"], chosen["net_gain"], True) for p in pairs]
            move_bits = ", ".join(f"{p['out']} → {p['in']}" for p in pairs)
            hit_note = f" (−{chosen['hit_cost']:.0f} pt hit)" if chosen["hit_cost"] > 0 else " (free)"
            summary.append(f"{move_bits}{hit_note} — net {chosen['net_gain']:+.1f} xPts over {horizon_n} GW(s).")
            hit_note2 = (f" — {chosen['hit_cost']:.0f}-pt hit taken, clears the {profile_name} profile's "
                        f"{threshold} xPts hit-cost threshold" if chosen["hit_cost"] > 0 else "")
            eo_note = " (one leg adjusted for style fit — see `eo_pull_applied` rows)" if any(
                p.get("eo_pull_applied") for p in pairs) else ""
            plan.append(f"GW{current_gw}: {chosen['actual_k']} transfer(s) — jointly optimized across the full "
                        f"player pool (Rules #28/#30), net {chosen['net_gain']:+.2f} xPts over {horizon_n} "
                        f"GW(s){hit_note2}{eo_note}. Fewest-transfers tie-break applied within the {moe:.1f} "
                        f"xPts margin-of-error band (Rules #34/#35). Scored on realized (bench-discounted) "
                        f"value per Standing Rule #12.")
            used_free = min(chosen["actual_k"], free_transfers)
            rolled = free_transfers - used_free
            if rolled > 0:
                summary.append(f"{rolled} free transfer(s) banked after this move.")
                plan.append(f"GW{current_gw}: {rolled} free transfer(s) banked (up to 5) after this move.")
            # Patch 10 disclosure (Standing Rule #4): a currently-owned player
            # with a missing projection this run is now kept as a real
            # candidate (patched to 0.0) instead of silently vanishing and
            # corrupting the retain-pool count — but if that same player got
            # swapped out here, it may be reacting to the data gap rather
            # than a genuine upgrade, so it's flagged rather than presented
            # as ordinary model output.
            gap_codes = chosen.get("data_gap_codes", [])
            if gap_codes:
                gap_names = full_pool[full_pool["code"].isin(gap_codes)]["web_name"].tolist()
                names_txt = ", ".join(gap_names) if gap_names else f"{len(gap_codes)} player(s)"
                summary.append(f"Data gap flagged: {names_txt} had a missing projection this run (treated as 0 "
                               f"xPts so they weren't silently forced out) — check their raw projection before "
                               f"trusting this move if they're one of the players above.")
                plan.append(f"GW{current_gw}: Data gap — {names_txt} had a missing `xpts_horizon_sum`/`price` "
                            f"this run; patched to 0.0 rather than dropped, per Standing Rule #4.")

    if chip_advisory:
        summary.append(chip_advisory)
        plan.append(chip_advisory)

    return {
        "moves": moves,
        "plan": plan,
        "summary": summary,
        "net_gain": chosen_net_gain,
        "profile_used": profile_name,
        "hit_cost_threshold": threshold,
        "minimum_meaningful_gain_free": meaningful_bar,
        "bench_autosub_discount": bench_discount,
        "hit_stance": hit_stance,
        "free_transfers": free_transfers,
        "margin_of_error": moe,
        "weekly_plan": [],
        "is_weekly_schedule": False,
    }


def evaluate_target_transfer(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                              profile_name: str, hit_stance: str, free_transfers: int,
                              bank: float, current_gw: int, gw_list: list[int],
                              target_code, default_net_gain: float | None = None) -> dict:
    """Manager-directed what-if (Patch 6): "if I bring THIS specific player
    in, is it worth it?" — auto-solving the cheapest legal way to fund him
    (`optimizer.solve_squad`'s `must_include_codes`), scored the exact same
    Rule #12-compliant realized-value way as `suggest_transfers()`, and
    checked against the exact same materiality/hit-cost/margin-of-error bars
    (Rules #13/#34). This is a SCOPE-RESTRICTED comparison — Standing Rule
    #30 requires that be stated, not silently treated as the model's own
    pick — so the caller must show this ALONGSIDE `suggest_transfers()`'s
    own full-pool recommendation, never in place of it. Equal-Scrutiny Rule
    #8 still applies to the target the same as any candidate: this function
    doesn't run the role-evidence check itself (that's Step 4/4a, upstream
    in the data pipeline) — it assumes the projection fed in already cleared
    it, same as every other player in `pool_df`.

    `default_net_gain`: optionally pass `suggest_transfers()`'s own chosen
    net_gain for the same run, purely so the result can state how this
    scenario compares to the model's own pick — informational only, never
    used to gate this function's own verdict.

    Returns a dict: `feasible`, `already_owned`, `summary` (plain-language
    lines), `moves`, `net_gain`, `hit_cost`, `clears_bar`, `chosen_k`."""
    profile = style_profiles.get_profile(profile_name)
    hit_cost_per = cfg["transfer"]["hit_cost_per_transfer"]
    threshold = profile["hit_cost_threshold"]
    meaningful_bar = cfg["transfer"].get("minimum_meaningful_gain_free", 2.0)
    this_gw_col = f"xpts_gw{current_gw}"
    horizon_n = len(gw_list)

    empty = {"feasible": False, "already_owned": False, "summary": [], "moves": [],
             "net_gain": None, "hit_cost": None, "clears_bar": False, "chosen_k": None}

    if squad_df is None or squad_df.empty or "code" not in squad_df.columns:
        return empty
    if target_code in set(squad_df["code"]):
        return {**empty, "already_owned": True,
                "summary": ["That player is already in your squad — nothing to evaluate."]}

    squad_df = squad_df.copy()
    pool_df = pool_df.copy() if pool_df is not None else pd.DataFrame(columns=squad_df.columns)
    for df in (squad_df, pool_df):
        for col in ("price", "xpts_horizon_sum", this_gw_col):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    bank = 0.0 if bank is None or pd.isna(bank) else float(bank)

    full_pool = pd.concat([squad_df, pool_df], ignore_index=True, sort=False)
    if "code" in full_pool.columns:
        full_pool = full_pool.drop_duplicates(subset=["code"], keep="first")
    if target_code not in set(full_pool["code"]):
        return {**empty, "summary": ["That player isn't in the current projection pool — can't evaluate this GW."]}

    current_codes = list(squad_df["code"])
    out_codes_all = set(current_codes)
    old_total = opt.realized_horizon_value(squad_df, gw_list, cfg)
    team_value = round(bank + (squad_df["price"].sum(skipna=True) or 0.0), 1)
    moe = eng.margin_of_error_threshold(old_total, cfg)

    # Diagnostic (Patch 8): reasons a straight 1-for-1 in the target's own
    # position might not be legal — computed directly rather than inferred
    # from a solver failure, so a "why wasn't this a clean swap" question
    # has a concrete, checkable answer instead of a black-box "no legal
    # way." Reported regardless of outcome when the target's own row looks
    # incomplete (Standing Rule #4 — show the inputs, never silently
    # estimate), and appended to the summary whenever the eventual result
    # needs more than 1 transfer, so the manager can see exactly which
    # check the clean swap actually failed.
    diagnostic = []
    target_row = full_pool[full_pool["code"] == target_code].iloc[0]
    target_price = target_row.get("price")
    target_pos = target_row.get("position")
    target_team = target_row.get("team")
    for col_name, val in [("price", target_price), ("position", target_pos),
                           ("xpts_horizon_sum", target_row.get("xpts_horizon_sum")),
                           (this_gw_col, target_row.get(this_gw_col) if this_gw_col in full_pool.columns else None)]:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            diagnostic.append(f"Data gap: this player's `{col_name}` is missing for this run — the solver silently "
                               f"drops any row missing price/position/projection, which can make a completely "
                               f"legal-looking swap fail to appear at any transfer count. Worth checking the raw "
                               f"projection for this player before trusting a 'no legal way' result.")
    if not diagnostic and pd.notna(target_price) and pd.notna(target_pos):
        # Check EVERY same-position squad player, not just the cheapest one
        # (a real bug this fixes: checking only the cheapest picked whichever
        # player leaves the LARGEST funding gap to cover — the opposite of
        # useful — and could name a player the manager never intended to
        # sell instead of the one they actually asked about, e.g. reporting
        # a budget shortfall against a cheap bench midfielder when the
        # manager meant a specific, similarly-priced starter).
        same_pos = squad_df[squad_df["position"] == target_pos].copy()
        legal_options = []
        checked = []
        for _, out_row in same_pos.iterrows():
            gap = round(float(target_price) - float(out_row["price"]), 1)
            budget_ok = gap <= bank + 1e-9
            new_team_counts = squad_df[squad_df["code"] != out_row["code"]]["team"].value_counts()
            new_target_team_count = int(new_team_counts.get(target_team, 0)) + 1
            club_ok = new_target_team_count <= cfg["squad_rules"]["max_per_club"]
            checked.append(out_row["web_name"])
            if budget_ok and club_ok:
                legal_options.append((out_row["web_name"], out_row["price"], gap))
        if legal_options:
            names = ", ".join(f"{n} (£{p}m, gap £{g}m)" for n, p, g in legal_options)
            diagnostic.append(f"A clean 1-for-1 IS budget/club-legal for at least one same-position player you own: "
                               f"{names}. If the model still used more than 1 transfer, that points to the solver "
                               f"preferring a different combination for a higher total, or a data gap on one "
                               f"player's projection — not a genuine legality problem.")
        elif checked:
            diagnostic.append(f"Checked every {target_pos} you currently own ({', '.join(checked)}) — none can be "
                               f"swapped for this player as a clean 1-for-1 within your £{bank}m bank and the "
                               f"{cfg['squad_rules']['max_per_club']}-per-club cap. That's a genuine budget/club "
                               f"constraint, not a bug.")

    # Always search the full 1-5 transfer range here, regardless of hit
    # stance (fixing a real bug: capping this at `free_transfers` under "No
    # hits" meant a target requiring 2 legal swaps was reported as "no legal
    # way" when a real, findable fit needed one more transfer than was ever
    # attempted). Possible reasons a clean 1-for-1 isn't found are checked
    # directly above (`diagnostic`) rather than guessed at here — budget,
    # club-limit, or a data gap on one of the two players, not assumed to be
    # any one of those by default. This is a what-if tool: the manager named
    # a specific target on purpose, so it should always show what it would
    # actually take, hit or no hit, and let the manager judge — same
    # principle as "Force" mode existing to let the manager override the
    # model's own default caution.
    candidates = {}
    for k in range(1, 6):
        min_retain = max(0, 15 - k)
        result = opt.solve_squad(full_pool, cfg, budget=team_value, retain_pool_codes=current_codes,
                                  min_retain=min_retain, must_include_codes=[target_code],
                                  objective_col="xpts_horizon_sum")
        if result is None:
            continue
        new_squad = result["squad"]
        actual_k = len(out_codes_all - set(new_squad["code"]))
        if actual_k == 0 or target_code not in set(new_squad["code"]):
            continue  # solver couldn't actually fit the target in at this k
        hit_cost = hit_cost_per * max(0, actual_k - free_transfers)
        new_total = opt.realized_horizon_value(new_squad, gw_list, cfg)
        net_gain = round(new_total - old_total - hit_cost, 2)
        if actual_k not in candidates or net_gain > candidates[actual_k]["net_gain"]:
            candidates[actual_k] = {"squad": new_squad, "total": new_total,
                                     "hit_cost": hit_cost, "net_gain": net_gain, "actual_k": actual_k,
                                     "data_gap_codes": result.get("data_gap_codes", [])}

    if not candidates:
        fallback = [f"Tried 1 through 5 transfers and found no legal, budget-fitting way to add this player — "
                    f"your total team value this run is £{team_value}m."]
        return {**empty, "summary": diagnostic + fallback}

    # Patch 16 — a real inconsistency this closes: this scenario tool always
    # searched k=1..5 regardless of hit_stance (Patch 6b, so a target
    # genuinely needing 2 legal swaps was still found instead of a false
    # "no legal way"), but it then picked its headline answer from ALL of
    # those candidates even under "No hits" — so it could hand back a
    # hit-costing plan as "Your scenario" while the sidebar said "No hits,"
    # with nothing telling the manager the two had diverged. Confirmed live:
    # a Palmer evaluation returned a −4 hit while "No hits" was selected.
    # Fix: still SEARCH every k (so a legitimate multi-swap free route is
    # never missed), but under "No hits" only let a hit-free candidate
    # (hit_cost == 0) win the headline "Your scenario" slot — same rule
    # `suggest_transfers()` already applies to its own default pick. If no
    # hit-free route exists at all, say so plainly and show the cheapest
    # hit-requiring route ONLY as clearly-labeled informational context, not
    # as a recommendation.
    free_candidates = {k: c for k, c in candidates.items() if c["hit_cost"] == 0}
    if hit_stance == "No hits" and not free_candidates:
        cheapest_k = min(candidates, key=lambda k: (candidates[k]["hit_cost"], -candidates[k]["net_gain"]))
        cheapest = candidates[cheapest_k]
        pairs = _pair_moves(squad_df, cheapest["squad"], this_gw_col)
        moves = [_move_row(p, cheapest["hit_cost"], cheapest["net_gain"], False) for p in pairs]
        move_bits = ", ".join(f"{p['out']} → {p['in']}" for p in pairs)
        summary = [f"No hit-free way to add this player this run — your sidebar stance is 'No hits', and every "
                    f"legal route found needs at least a {cheapest['hit_cost']:.0f}-pt hit.",
                   f"Informational only, not recommended under 'No hits': the cheapest hit-requiring route — "
                    f"{move_bits} ({cheapest['actual_k']} transfer(s), −{cheapest['hit_cost']:.0f} pts) — "
                    f"nets {cheapest['net_gain']:+.1f} xPts over {horizon_n} GW(s)."]
        if default_net_gain is not None:
            summary.append("Switch to 'Hit if worth it' or 'Force' in the sidebar to let this scenario actually "
                            "recommend a hit-costing route.")
        return {"feasible": True, "already_owned": False, "summary": summary, "moves": moves,
                "net_gain": None, "hit_cost": cheapest["hit_cost"], "clears_bar": False, "chosen_k": None}

    working = free_candidates if hit_stance == "No hits" else candidates

    # Cheapest legal way in that also maximizes net gain: same fewest-
    # transfers-within-margin-of-error tie-break as suggest_transfers().
    best_net = max(c["net_gain"] for c in working.values())
    tied_ks = sorted(k for k, c in working.items() if (best_net - c["net_gain"]) < moe)
    chosen_k = min(tied_ks) if tied_ks else min(working, key=lambda k: working[k]["net_gain"])
    chosen = working[chosen_k]
    bar = threshold if chosen["hit_cost"] > 0 else meaningful_bar
    clears = chosen["net_gain"] >= bar

    pairs = _pair_moves(squad_df, chosen["squad"], this_gw_col)
    moves = [_move_row(p, chosen["hit_cost"], chosen["net_gain"], clears) for p in pairs]
    move_bits = ", ".join(f"{p['out']} → {p['in']}" for p in pairs)
    hit_note = f" (−{chosen['hit_cost']:.0f} pt hit)" if chosen["hit_cost"] > 0 else " (free)"
    verdict_note = ("clears its bar — a genuine improvement" if clears else
                     f"doesn't clear the {bar} xPts bar this move needs — not worth it as evaluated")
    summary = [f"Your scenario — {move_bits}{hit_note}: net {chosen['net_gain']:+.1f} xPts over "
               f"{horizon_n} GW(s), {verdict_note}."]
    if chosen["actual_k"] > 1:
        # Say WHY more than one swap was needed using the diagnostic actually
        # computed above (budget/club-limit/data-gap), never a generic guess
        # — a prior version of this message asserted a formation-shape cause
        # by default, which turned out to be wrong for a same-position
        # MID-for-MID swap in a real case; that guess is retired in favor of
        # the concrete checks above.
        summary.append(f"Needed {chosen['actual_k']} linked swaps, not a single 1-for-1:")
        summary.extend(diagnostic)
        # Patch 10 — the actual root cause of the confirmed Semenyo→Palmer
        # case: a currently-owned player with a missing projection this run
        # (e.g. a sparse-minutes backup GK) used to be silently dropped from
        # candidacy entirely by `optimizer.solve_squad()`, which quietly
        # consumed the one transfer slot a k=1 request should have spent on
        # the manager's actual target instead. The solver no longer drops
        # them (their value is patched to 0.0 so they stay a real candidate
        # it can choose to keep or drop on the merits), but Standing Rule #4
        # ("show the inputs, never silently estimate") still requires
        # disclosing that this happened, since it's evidence the extra
        # swap(s) may be about filling a data gap, not a genuine upgrade.
        gap_codes = chosen.get("data_gap_codes", [])
        if gap_codes:
            gap_names = full_pool[full_pool["code"].isin(gap_codes)]["web_name"].tolist()
            names_txt = ", ".join(gap_names) if gap_names else f"{len(gap_codes)} player(s)"
            summary.append(f"Data gap flagged: {names_txt} had a missing projection this run and was treated as "
                            f"0 xPts so it wouldn't be silently forced out of your squad — if the solver also "
                            f"swapped this player, it may be reacting to that gap rather than a genuine "
                            f"upgrade. Worth checking their raw projection before trusting that leg of the move.")
    if default_net_gain is not None:
        diff = round(chosen["net_gain"] - default_net_gain, 2)
        if abs(diff) < moe:
            summary.append(f"Statistically tied with the model's own pick this run (within the {moe:.1f} xPts "
                            f"margin-of-error).")
        elif diff > 0:
            summary.append(f"This nets {diff:+.1f} xPts more than the model's own recommendation this run.")
        else:
            summary.append(f"This nets {diff:.1f} xPts less than the model's own recommendation this run.")

    return {"feasible": True, "already_owned": False, "summary": summary, "moves": moves,
            "net_gain": chosen["net_gain"], "hit_cost": chosen["hit_cost"], "clears_bar": clears,
            "chosen_k": chosen["actual_k"]}


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
