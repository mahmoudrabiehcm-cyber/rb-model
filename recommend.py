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
                "out_xm": o.get("xm"),
                "in_code": i["code"], "in": i["web_name"], "in_team": i["team"], "in_price": i["price"],
                "in_xpts": i.get("xpts_horizon_sum", 0.0) or 0.0, "in_gw": 0 if pd.isna(i_gw) else i_gw,
                "in_xm": i.get("xm"),
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


def starting_xi_impact_check(old_squad: pd.DataFrame, new_squad: pd.DataFrame, in_codes: set,
                              gw_list: list[int], cfg: dict, bb_play_gw: int | None = None,
                              capped_gw_list: list[int] | None = None,
                              out_codes: set | None = None,
                              disrupted_codes: set | None = None) -> dict:
    """Patch 34 (manager report, 2026-09-14): `realized_horizon_value()`
    already discounts a bench player down to P(autosub) x points (Standing
    Rule #12) rather than their full "if they started" number — but a swap
    whose ENTIRE net gain comes from that small autosub-chance discount,
    with the incoming player never actually entering the starting XI at any
    point in the horizon, still cleared the materiality/margin-of-error bars
    as if it were a real week-to-week scoring change. It isn't: it only ever
    pays off if an autosub happens to fire. The manager's own framing: if a
    transfer's incoming player "won't make it to the starting XI" across the
    whole horizon, saving the transfer is the right call, not spending it on
    a swap that may never actually move your score.

    Solves the best starting XI for OLD and NEW squads at every GW in the
    checked window (`capped_gw_list` if given — e.g. truncated before a
    manager-planned full-rebuild chip, same idea as
    `fpl_engine.disruption_check()`'s horizon cap, generalized here to any
    transfer, not just a disrupted player — else the full `gw_list`), and
    checks whether any TRANSFERRED-IN player actually appears in that GW's
    starting XI. If none do across the whole window, `has_impact` is False
    UNLESS one of two overrides applies:
    - `bb_play_gw` (a Bench Boost "play" verdict gw already inside this
      window): the swap raises that specific GW's total bench value (raw,
      not discounted — Bench Boost bypasses the autosub uncertainty
      entirely, so bench value is real value that week).
    - `disrupted_codes` (2026-09-14 manager report — a Foden→Damsgaard swap
      wasn't vetoed, and it turned out the real reason was that Foden was
      disruption-flagged, but nothing said so): if `out_codes` intersects
      `disrupted_codes`, moving on a flagged player has real value on its
      own — a live status/data-quality flag means his own currently-
      discounted-but-nonzero projection is optimistic, so removing him
      isn't gated behind the incoming player alone reaching the XI.

    Returns {"has_impact": bool, "shifts": [{"gw", "in": [names], "out":
    [names]}, ...], "bench_boost_gw": gw|None, "incoming_entered_gws":
    [gw, ...], "disrupted_out": bool}. `shifts` lists every GW where XI
    membership changes for an EXISTING squad player as a side effect of the
    swap (the transferred-in player(s) themselves are excluded from this
    list — manager report: their own arrival is the headline move, not a
    side effect, and silently including/excluding them read as
    inconsistent); `incoming_entered_gws` lists the GWs where a transferred-
    in player actually started, for a "reaches your XI at GW{n}" disclosure;
    `bench_boost_gw`/`disrupted_out` name which override, if any, is what
    actually saved the swap from a "no impact" verdict."""
    empty = {"has_impact": True, "shifts": [], "bench_boost_gw": None,
             "incoming_entered_gws": [], "disrupted_out": False}  # fail-open: never block on missing data
    if old_squad is None or old_squad.empty or new_squad is None or new_squad.empty or not gw_list:
        return empty
    check_gws = capped_gw_list if capped_gw_list is not None else gw_list
    if not check_gws:
        # entire horizon capped away (e.g. a rebuild chip lands immediately) —
        # nothing left to check, so there's genuinely nothing this transfer
        # can impact within the checked window.
        return {"has_impact": False, "shifts": [], "bench_boost_gw": None,
                "incoming_entered_gws": [], "disrupted_out": False}

    in_codes_set = set(in_codes)
    disrupted_out = bool(out_codes and disrupted_codes and (set(out_codes) & set(disrupted_codes)))

    shifts = []
    any_impact = disrupted_out
    bb_gw_hit = None
    incoming_entered_gws = []
    for gw in check_gws:
        col = f"xpts_gw{gw}"
        if col not in old_squad.columns or col not in new_squad.columns:
            continue
        old_xi = best_starting_xi_safe(old_squad, col)
        new_xi = best_starting_xi_safe(new_squad, col)
        old_codes = set(old_xi["code"]) if old_xi is not None else set()
        new_codes = set(new_xi["code"]) if new_xi is not None else set()
        entering = new_codes - old_codes
        leaving = old_codes - new_codes
        if entering & in_codes_set:
            any_impact = True
            incoming_entered_gws.append(gw)
        # Side-effect disclosure only — excludes the transferred-in player(s)
        # themselves, so this only ever names an EXISTING squad player whose
        # bench/XI status changed as a knock-on of the swap.
        side_entering = entering - in_codes_set
        if side_entering or leaving:
            in_names = new_squad[new_squad["code"].isin(side_entering)]["web_name"].tolist()
            out_names = old_squad[old_squad["code"].isin(leaving)]["web_name"].tolist()
            if in_names or out_names:
                shifts.append({"gw": gw, "in": in_names, "out": out_names})
        if bb_play_gw is not None and gw == bb_play_gw:
            old_bench_sum = old_squad[~old_squad["code"].isin(old_codes)][col].fillna(0).sum()
            new_bench_sum = new_squad[~new_squad["code"].isin(new_codes)][col].fillna(0).sum()
            if new_bench_sum > old_bench_sum + 0.5:  # small materiality guard against float noise
                any_impact = True
                bb_gw_hit = gw
    return {"has_impact": any_impact, "shifts": shifts, "bench_boost_gw": bb_gw_hit,
            "incoming_entered_gws": incoming_entered_gws, "disrupted_out": disrupted_out}


def best_starting_xi_safe(squad: pd.DataFrame, gw_col: str):
    """Thin wrapper so `starting_xi_impact_check()` doesn't need to import
    `optimizer` directly (avoids a circular-import risk — `optimizer.py`
    doesn't import `recommend.py`, but keeping the boundary one-directional
    is cheap insurance) and doesn't crash on a missing/empty XI result."""
    if gw_col not in squad.columns:
        return None
    result = opt.best_starting_xi(squad, gw_col)
    return result["xi"] if result else None


def apply_style_to_wildcard_squad(current_squad: pd.DataFrame, rebuild_squad: pd.DataFrame,
                                   full_pool: pd.DataFrame, profile_name: str, cfg: dict,
                                   this_gw_col: str) -> pd.DataFrame:
    """Wildcard-list feature (2026-09-07 discussion): a Wildcard rebuild is
    the single biggest squad decision in the whole tool, so it should be
    consistent with whichever Style Profile is active in the sidebar, same
    as every ordinary transfer recommendation already is — not silently
    switch to a neutral pure-xPts optimization just because it's a
    from-scratch rebuild instead of a k-transfer plan.

    Reuses the exact same mechanism ordinary transfers already get: treat
    the (current squad -> rebuild squad) difference as an, up to 15-leg,
    transfer plan via `_pair_moves` (safe even at this size — both squads
    are valid 2-5-5-3s, so the dropped/added position multisets always
    match by construction, satisfying `_pair_moves`'s own legality assert),
    then let `_apply_eo_pull` substitute any IN leg for a genuinely
    statistically-tied, cheaper/more-differential alternative per the
    active profile's `eo_pull` setting — a merit-gated substitution, never a
    downgrade, exactly like a normal transfer. 'Balanced / Pure xPts'
    (`eo_pull: none`) never substitutes, so this is a no-op for that
    profile and the raw rebuild is returned unchanged.

    Returns the adjusted 15-man squad DataFrame (retained players +
    each pair's, possibly substituted, IN player)."""
    profile = style_profiles.get_profile(profile_name)
    if profile.get("eo_pull", "none") == "none" or current_squad is None or current_squad.empty \
            or rebuild_squad is None or rebuild_squad.empty:
        return rebuild_squad

    out_codes = set(current_squad["code"]) - set(rebuild_squad["code"])
    pairs = _pair_moves(current_squad, rebuild_squad, this_gw_col)
    if not pairs:
        return rebuild_squad
    in_codes = {p["in_code"] for p in pairs}
    pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes, in_codes)

    retained_codes = set(current_squad["code"]) & set(rebuild_squad["code"])
    retained = rebuild_squad[rebuild_squad["code"].isin(retained_codes)]
    in_rows = full_pool[full_pool["code"].isin({p["in_code"] for p in pairs})]
    adjusted = pd.concat([retained, in_rows], ignore_index=True, sort=False)
    if "code" in adjusted.columns:
        adjusted = adjusted.drop_duplicates(subset=["code"], keep="first")
    return adjusted


def _position_tie_break(chosen: dict, squad_df: pd.DataFrame, full_pool: pd.DataFrame,
                         gw_list: list[int], cfg: dict, bench_w: float, bb_play_gw: int | None,
                         moe: float, team_value: float, current_gw: int) -> tuple[dict, list[str] | None]:
    """Patch 41 (manager, 2026-09-14, screenshot: Damsgaard — the model's own
    automatic pick, +2.5 xPts — and Tavernier — manually evaluated via
    "Evaluate your own scenario", +2.54 xPts — landed statistically tied at
    the requested horizon: "if 2 candidates are so close the model needs to
    look at +1 GW horizon to identify the best candidate"). Root cause: the
    automatic recommendation comes from a SINGLE MILP solve per transfer
    count k (`optimizer.solve_squad`, searching the whole pool by raw
    xpts_horizon_sum) — it returns exactly one candidate squad, never
    explicitly comparing specific alternative in-players against each other
    on realized net-gain. A genuinely-tied or better alternative can go
    completely unseen by the automatic pick even though it's realistically
    as good or better — Tavernier only surfaced because the manager tested
    him by hand.

    Scope, confirmed with the manager (2026-09-14): only the straight
    1-for-1 swap case (`actual_k == 1`, the reported scenario and the common
    case in practice) — for a clean 1-for-1, every other same-position,
    budget/club-legal pool player can be substituted directly with no MILP
    re-solve needed (the rest of the squad doesn't change), scored on the
    exact same realized-value math as the model's own chosen candidate, so
    a genuine full-pool scan is cheap here. Multi-transfer (k>=2) candidates
    are left alone — recombining a k-way swap combinatorially would need a
    fresh MILP solve per alternative, which is the expensive case the
    manager did NOT ask this to cover.

    Manager-confirmed behavior: a full scan every run (not capped to a
    top-N), and when a genuine tie is found and a further-out GW's data
    resolves it clearly, the winner REPLACES the headline recommendation
    (not just flagged) — the displayed net_gain still reflects the
    ORIGINAL requested horizon for whichever player wins; only the decision
    of WHICH player to show is informed by the extra GW.

    Returns `(possibly-updated chosen dict, plan lines to append or None)`.
    Never mutates `chosen`; returns it unchanged (second element None) when
    there's nothing to do (not a 1-for-1, no real tie found, or no data to
    extend into)."""
    if chosen.get("actual_k") != 1 or chosen.get("squad") is None:
        return chosen, None
    out_codes = set(squad_df["code"]) - set(chosen["squad"]["code"])
    in_codes = set(chosen["squad"]["code"]) - set(squad_df["code"])
    if len(out_codes) != 1 or len(in_codes) != 1:
        return chosen, None  # not a clean 1-for-1 — defensive, shouldn't happen at actual_k==1
    out_code, in_code = next(iter(out_codes)), next(iter(in_codes))
    out_rows = squad_df[squad_df["code"] == out_code]
    if out_rows.empty:
        return chosen, None
    out_row = out_rows.iloc[0]
    out_pos, out_price, out_team = out_row.get("position"), out_row.get("price"), out_row.get("team")
    if pd.isna(out_price) or out_pos is None:
        return chosen, None

    max_per_club = cfg["squad_rules"]["max_per_club"]
    squad_total_price = squad_df["price"].sum(skipna=True) or 0.0
    other_team_counts = squad_df[squad_df["code"] != out_code]["team"].value_counts()
    baseline_total = chosen.get("baseline_total", 0.0)
    hit_cost = chosen.get("hit_cost", 0.0)

    # Manager-confirmed scope (2026-09-14, reaffirmed after checking the
    # cost): a full scan of every same-position, budget/club-legal pool
    # player, not capped to a top-N — this never calls the MILP solver (it's
    # a direct 1-for-1 substitution scored with the same cheap realized-
    # value math used everywhere else), so pool size doesn't meaningfully
    # affect runtime the way the actual MILP solves elsewhere in this
    # function do.
    same_pos_pool = full_pool[(full_pool["position"] == out_pos)
                               & (~full_pool["code"].isin(set(squad_df["code"]) - {out_code}))].copy()
    if same_pos_pool.empty:
        return chosen, None

    scored: dict = {in_code: {"net_gain": chosen["net_gain"], "squad": chosen["squad"], "total": chosen["total"],
                               "web_name": chosen["squad"][chosen["squad"]["code"] == in_code]["web_name"]
                               .iloc[0] if in_code in set(chosen["squad"]["code"]) else in_code,
                               "xi_total": chosen.get("new_xi"), "bench_total": chosen.get("new_bench")}}
    for _, alt_row in same_pos_pool.iterrows():
        alt_code = alt_row.get("code")
        if alt_code == in_code or alt_code == out_code:
            continue
        alt_price = alt_row.get("price")
        if pd.isna(alt_price):
            continue
        if (squad_total_price - out_price + alt_price) > team_value + 1e-9:
            continue  # not budget-legal as a straight swap
        alt_team = alt_row.get("team")
        if int(other_team_counts.get(alt_team, 0)) + 1 > max_per_club:
            continue  # would breach the per-club cap
        alt_squad = pd.concat([squad_df[squad_df["code"] != out_code], pd.DataFrame([alt_row])],
                               ignore_index=True, sort=False)
        if "code" in alt_squad.columns:
            alt_squad = alt_squad.drop_duplicates(subset=["code"], keep="first")
        alt_bd = opt.realized_horizon_breakdown(alt_squad, gw_list, cfg, bench_weight_scale=bench_w,
                                                 bb_play_gw=bb_play_gw)
        alt_net = round(alt_bd["total"] - baseline_total - hit_cost, 2)
        scored[alt_code] = {"net_gain": alt_net, "squad": alt_squad, "total": alt_bd["total"],
                             "web_name": alt_row.get("web_name", alt_code),
                             "xi_total": alt_bd["xi_total"], "bench_total": alt_bd["bench_total"]}

    best_net_here = max(v["net_gain"] for v in scored.values())
    tied_codes = [code for code, v in scored.items() if (best_net_here - v["net_gain"]) < moe]
    if len(tied_codes) <= 1:
        return chosen, None  # the model's own pick wasn't actually tied with anything real

    next_gw = max(gw_list) + 1
    next_col = f"xpts_gw{next_gw}"
    if next_col not in full_pool.columns or next_col not in squad_df.columns:
        tied_names = ", ".join(scored[c]["web_name"] for c in tied_codes)
        return chosen, [f"GW{current_gw}: Tie-break: {tied_names} are statistically tied at this horizon "
                        f"(within {moe:.1f} xPts) but there's no GW{next_gw} projection data this run to break "
                        f"the tie — keeping the model's original pick."]

    extended_gw_list = gw_list + [next_gw]
    ext_baseline = realistic_baseline_value(squad_df, {out_code}, extended_gw_list, cfg,
                                             bb_play_gw=bb_play_gw)["baseline_total"]
    ext_scores = {}
    for code in tied_codes:
        ext_bd = opt.realized_horizon_breakdown(scored[code]["squad"], extended_gw_list, cfg,
                                                 bench_weight_scale=bench_w, bb_play_gw=bb_play_gw)
        ext_scores[code] = round(ext_bd["total"] - ext_baseline - hit_cost, 2)
    winner_code = max(ext_scores, key=lambda c: ext_scores[c])

    if winner_code == in_code:
        return chosen, [f"GW{current_gw}: Tie-break: {len(tied_codes)} candidates were statistically tied at this "
                        f"horizon (within {moe:.1f} xPts) — extended to GW{next_gw} to check, and the model's "
                        f"original pick ({scored[in_code]['web_name']}) still comes out ahead "
                        f"({ext_scores[in_code]:+.2f} vs {max(v for c, v in ext_scores.items() if c != in_code):+.2f} "
                        f"xPts over the extended window)."]

    winner = scored[winner_code]
    new_chosen = {**chosen, "squad": winner["squad"], "total": winner["total"], "net_gain": winner["net_gain"],
                  "new_xi": winner["xi_total"], "new_bench": winner["bench_total"], "data_gap_codes": []}
    loser_ext = ext_scores[in_code]
    plan_lines = [f"GW{current_gw}: Tie-break: {scored[in_code]['web_name']} (the model's original pick) and "
                  f"{winner['web_name']} were statistically tied at this horizon "
                  f"({chosen['net_gain']:+.2f} vs {scored[winner_code]['net_gain']:+.2f} xPts, within "
                  f"{moe:.1f} xPts) — extended to GW{next_gw} to break it: {winner['web_name']} nets "
                  f"{ext_scores[winner_code]:+.2f} xPts vs {scored[in_code]['web_name']}'s {loser_ext:+.2f} xPts "
                  f"over the extended window, so the recommendation switched to {winner['web_name']}."]
    return new_chosen, plan_lines


def plan_transfer_schedule(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                            profile_name: str, free_transfers: int, bank: float,
                            current_gw: int, gw_list: list[int],
                            meaningful_bar: float | None = None,
                            chip_advisory: str | None = None,
                            bb_play_gw: int | None = None,
                            disrupted_codes: set | None = None,
                            hit_stance: str = "No hits") -> dict:
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
    - Under `hit_stance="No hits"` (the default), never proposes a hit — if
      no free move clears the materiality bar in a given week (Rules
      #13/#34, same bars as `suggest_transfers()`), that week is Roll and
      the free transfer banks forward (up to the cap). Under
      `hit_stance="Hit if worth it"` (manager-confirmed extension, project
      discussion 2026-09-14: "extend the weekly planner to allow hits"),
      each week may also take a hit if a paid-for move still clears the
      stricter `hit_cost_threshold` bar (mirrors `suggest_transfers()`'s own
      k_range/threshold logic) — hit cost is deducted from that week's
      net_gain before comparing against the bar and against Roll.
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
    hit_cost_per = cfg["transfer"]["hit_cost_per_transfer"]
    threshold = profile["hit_cost_threshold"]
    if meaningful_bar is None:
        meaningful_bar = cfg["transfer"].get("minimum_meaningful_gain_free", 2.0)
    bench_discount = cfg["transfer"].get("bench_autosub_discount", 0.2)
    allow_hits = hit_stance == "Hit if worth it"

    empty_result = {
        "moves": [], "plan": [], "summary": [], "net_gain": 0.0, "profile_used": profile_name,
        "hit_cost_threshold": threshold, "minimum_meaningful_gain_free": meaningful_bar,
        "bench_autosub_discount": bench_discount, "hit_stance": hit_stance, "free_transfers": free_transfers,
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
        # Patch 37: same near-zero bench weight (except bb_play_gw) as
        # suggest_transfers() — applied here too so the weekly pacing plan
        # doesn't let bench-quality alone carry a marginal move.
        bench_w = cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08)
        old_total = opt.realized_horizon_value(sim_squad, remaining_gws, cfg, bench_weight_scale=bench_w,
                                                bb_play_gw=bb_play_gw)
        moe = eng.margin_of_error_threshold(old_total, cfg)

        candidates = {0: {"squad": sim_squad, "total": old_total, "net_gain": 0.0, "hit_cost": 0.0,
                          "actual_k": 0, "data_gap_codes": []}}
        # Patch 39 (manager, 2026-09-14: "extend the weekly planner to allow
        # hits"): under "Hit if worth it" this week's search range extends
        # past the banked free transfers (up to 5, same cap suggest_transfers()
        # uses for its "Hit if worth it" k_range) so a paid-for move can be
        # considered too, not just free ones.
        k_upper = max(ft_bank, 5) if allow_hits else ft_bank
        for k in range(1, k_upper + 1):
            min_retain = max(0, 15 - k)
            result = opt.solve_squad(full_pool, cfg, budget=team_value, retain_pool_codes=current_codes,
                                      min_retain=min_retain, objective_col="xpts_horizon_sum")
            if result is None:
                continue
            new_squad = result["squad"]
            actual_k = len(out_codes_all - set(new_squad["code"]))
            if actual_k == 0:
                continue  # nothing worth swapping at this k — already covered by k=0
            hit_cost = hit_cost_per * max(0, actual_k - ft_bank) if allow_hits else 0.0
            new_total = opt.realized_horizon_value(new_squad, remaining_gws, cfg, bench_weight_scale=bench_w,
                                                    bb_play_gw=bb_play_gw)
            # Patch 36 — same nailed-gate baseline as suggest_transfers(),
            # applied per week: a non-nailed out-player's projection is
            # zeroed across the remaining weeks before the baseline is
            # recomputed, so the incoming player is judged against the free
            # bench replacement, not the outgoing player's own (possibly
            # near-zero) live number.
            out_codes_this = out_codes_all - set(new_squad["code"])
            baseline_info = realistic_baseline_value(sim_squad, out_codes_this, remaining_gws, cfg,
                                                      bb_play_gw=bb_play_gw)
            baseline_total = baseline_info["baseline_total"]
            net_gain = round(new_total - baseline_total - hit_cost, 2)
            if actual_k not in candidates or net_gain > candidates[actual_k]["net_gain"]:
                candidates[actual_k] = {"squad": new_squad, "total": new_total, "net_gain": net_gain,
                                         "hit_cost": hit_cost,
                                         "actual_k": actual_k, "data_gap_codes": result.get("data_gap_codes", []),
                                         "baseline_total": baseline_total,
                                         "baseline_adjusted": baseline_info["adjusted"],
                                         "baseline_zeroed_names": baseline_info["zeroed_names"]}

        best_net = max(c["net_gain"] for c in candidates.values())
        tied_ks = sorted(k for k, c in candidates.items() if (best_net - c["net_gain"]) < moe)
        # A candidate that costs a hit this week has to clear the stricter
        # hit_cost_threshold bar (same as suggest_transfers()'s own
        # `bar = threshold if hit_cost > 0 else meaningful_bar` rule) — a
        # free move only needs the lower meaningful_bar.
        viable = [k for k in tied_ks if k == 0
                  or candidates[k]["net_gain"] >= (threshold if candidates[k]["hit_cost"] > 0 else meaningful_bar)]

        # Patch 34 — same Starting-XI Impact Check as suggest_transfers(),
        # applied per week ("everywhere", per manager request): a candidate
        # that only clears the bar via bench-autosub value with no
        # transferred-in player reaching the XI in the remaining weeks (and
        # no Bench Boost override) isn't a real weekly scoring change.
        week_impact: dict[int, dict] = {}
        week_bench_only: list[int] = []
        for k in list(viable):
            if k == 0:
                continue
            c = candidates[k]
            in_codes = set(c["squad"]["code"]) - out_codes_all
            out_codes_this = out_codes_all - set(c["squad"]["code"])
            check = starting_xi_impact_check(sim_squad, c["squad"], in_codes, remaining_gws, cfg,
                                              bb_play_gw=bb_play_gw, out_codes=out_codes_this,
                                              disrupted_codes=disrupted_codes)
            week_impact[k] = check
            if not check["has_impact"]:
                week_bench_only.append(k)
        viable = [k for k in viable if k == 0 or k not in week_bench_only]
        chosen_k = min(viable) if viable else 0
        chosen = candidates[chosen_k]

        # Patch 41 — same position tie-break as the single-decision path
        # (see _position_tie_break()'s docstring): the same "one MILP pick
        # per k, never compared against real alternatives" blind spot
        # exists per week in this chained pacing plan too.
        if chosen.get("actual_k") == 1:
            chosen, _tie_plan = _position_tie_break(chosen, sim_squad, full_pool, remaining_gws, cfg,
                                                      bench_w, bb_play_gw, moe, team_value, gw)
            if _tie_plan:
                plan.extend(_tie_plan)

        ft_used = min(chosen["actual_k"], ft_bank)
        ft_after = min(transfers.MAX_BANK, (ft_bank - ft_used) + 1)

        gap_codes = chosen.get("data_gap_codes", [])
        data_gap_note = None
        if gap_codes:
            gap_names = full_pool[full_pool["code"].isin(gap_codes)]["web_name"].tolist()
            names_txt = ", ".join(gap_names) if gap_names else f"{len(gap_codes)} player(s)"
            data_gap_note = (f"Data gap flagged: {names_txt} had a missing projection this week and was "
                              f"treated as 0 xPts so it wouldn't be silently forced out (Standing Rule #4).")

        # Patch 39 (manager, 2026-09-14: "keep only the recommendation and
        # avoid the explanation"): `week_summary` (shown always, one line per
        # week) now carries ONLY the headline move/Roll line. Every
        # supporting detail — Why/Side effect/Baseline note/data-gap text —
        # goes to `plan` only (rendered in the collapsed "Why — full trace"
        # expander), matching the same declutter already applied to
        # suggest_transfers()'s single-decision path.
        if chosen["actual_k"] == 0 and week_bench_only:
            bo_k = min(week_bench_only)
            bo = candidates[bo_k]
            pairs_bo = _pair_moves(sim_squad, bo["squad"], this_gw_col)
            move_bits_bo = ", ".join(f"{p['out']} → {p['in']}" for p in pairs_bo)
            week_moves = []
            week_summary = f"GW{gw}: Roll — {ft_bank} FT banked → {ft_after} for GW{gw + 1}."
            plan.append(f"GW{gw}: Roll — {move_bits_bo} nets {bo['net_gain']:+.1f} xPts but is bench-autosub "
                        f"value only (no XI impact) — saving the transfer is stronger.")
        elif chosen["actual_k"] == 0:
            week_moves = []
            week_summary = f"GW{gw}: Roll — {ft_bank} FT banked → {ft_after} for GW{gw + 1}."
            plan.append(f"GW{gw}: Roll — no move clears the bar this week.")
        else:
            pairs = _pair_moves(sim_squad, chosen["squad"], this_gw_col)
            week_moves = [{**_move_row(p, 0.0, chosen["net_gain"], True), "gw": gw} for p in pairs]
            move_bits = ", ".join(
                f"{p['out']}{_xm_badge(p.get('out_xm'), cfg)} → {p['in']}{_xm_badge(p.get('in_xm'), cfg)}"
                for p in pairs)
            hit_note = f" (−{chosen['hit_cost']:.0f} hit)" if chosen.get("hit_cost", 0.0) > 0 else " (free)"
            week_summary = (f"GW{gw}: {move_bits}{hit_note} — net {chosen['net_gain']:+.1f} xPts over the "
                             f"remaining horizon.")
            week_check = week_impact.get(chosen_k, {})
            reason_bits = []
            if week_check.get("incoming_entered_gws"):
                gws_txt = ", ".join(f"GW{g}" for g in week_check["incoming_entered_gws"])
                reason_bits.append(f"reaches your starting XI at {gws_txt}")
            if week_check.get("disrupted_out"):
                reason_bits.append("outgoing player is disruption-flagged (Rule #41)")
            if week_check.get("bench_boost_gw"):
                reason_bits.append(f"raises your GW{week_check['bench_boost_gw']} Bench Boost bench value")
            if reason_bits:
                plan.append(f"GW{gw}: Why this counts as a real change: {'; '.join(reason_bits)}.")
            if chosen.get("hit_cost", 0.0) > 0:
                plan.append(f"GW{gw}: Hit taken — {chosen['actual_k']} transfer(s) made, {ft_bank} free, "
                            f"{chosen['actual_k'] - ft_bank} paid at {hit_cost_per:.0f} pts each "
                            f"({chosen['hit_cost']:.0f} pts total), still clears the stricter "
                            f"{threshold:.1f}-xPts hit-cost bar.")
            shift_bits = [f"{', '.join(s['in'])} in GW{s['gw']}" for s in week_check.get("shifts", []) if s["in"]]
            if shift_bits:
                plan.append(f"GW{gw}: Side effect — already on your bench, no transfer needed: "
                            f"{'; '.join(shift_bits)}.")
            if chosen.get("baseline_adjusted"):
                zeroed_txt = ", ".join(chosen.get("baseline_zeroed_names", []))
                plan.append(f"GW{gw}: Baseline note: {zeroed_txt} isn't nailed over the remaining horizon, so "
                            f"net-gain compares the incoming player against {zeroed_txt} benched for free.")
            plan.append(f"GW{gw}: {ft_bank - ft_used} FT left banked → {ft_after} for GW{gw + 1}.")
            sim_squad = chosen["squad"]

        if data_gap_note:
            plan.append(f"GW{gw}: {data_gap_note}")

        weekly_plan.append({
            "gw": gw, "moves": week_moves, "net_gain": chosen["net_gain"],
            "ft_available": ft_bank, "ft_used": ft_used, "ft_banked_after": ft_after,
            "summary": week_summary, "data_gap_note": data_gap_note,
        })
        summary.append(week_summary)
        plan.append(f"GW{gw}: {chosen['actual_k']} transfer(s) this week (chained pacing plan) — "
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
        "hit_stance": hit_stance,
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
                       chip_advisory: str | None = None,
                       bb_play_gw: int | None = None,
                       chip_capped_gw_list: list[int] | None = None,
                       disrupted_codes: set | None = None) -> dict:
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
    # Patch 39 (manager, 2026-09-14: "3 GWs horizon doesn't give me the
    # specific plan for each week 'transfer, roll'" + confirmed choice
    # "Extend the weekly planner to allow hits"): "Hit if worth it" now also
    # routes through the weekly pacing plan at horizon>1, with hits allowed
    # per week. "Force" keeps the original single joint-decision path below —
    # it's an explicit manager override of a specific k, not a pacing
    # question the planner should re-decide week by week.
    if hit_stance in ("No hits", "Hit if worth it") and horizon_n > 1:
        return plan_transfer_schedule(squad_df, pool_df, cfg, profile_name, free_transfers, bank,
                                       current_gw, gw_list, meaningful_bar, chip_advisory,
                                       bb_play_gw=bb_play_gw, disrupted_codes=disrupted_codes,
                                       hit_stance=hit_stance)

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
    # Patch 37: near-zero bench weight (except the specific bb_play_gw week)
    # for every net-gain comparison in this function — a stronger bench
    # alone should not be able to carry a marginal transfer over the bar.
    bench_w = cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08)
    old_total = opt.realized_horizon_value(squad_df, gw_list, cfg, bench_weight_scale=bench_w, bb_play_gw=bb_play_gw)
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
        # Patch 38: breakdown (not just the summed total) so the disclosure
        # below can show the direct starting-XI swap and the (near-zero,
        # per Patch 37) bench-autosub credit as two separate, auditable
        # numbers instead of one combined figure a manager has to take on
        # faith.
        new_breakdown = opt.realized_horizon_breakdown(new_squad, gw_list, cfg, bench_weight_scale=bench_w,
                                                        bb_play_gw=bb_play_gw)
        new_total = new_breakdown["total"]
        # Patch 36 (manager report, 2026-09-14: "Foden dead at 0 xPts, this
        # doesn't make sense") — the comparison baseline for THIS candidate
        # is no longer always the squad's plain current total. A nailed
        # out-player's own live-discounted projection is trusted as-is (a
        # single bad-fixture week is fairly reflected already). A non-nailed
        # out-player (rotation/bench-risk xM tier) has his projection zeroed
        # for the checked GWs and the baseline is recomputed — i.e. the
        # incoming player is judged against what the free bench replacement
        # would have delivered, not against the outgoing player's own
        # (possibly near-zero) number. Per the manager's explicit choice,
        # this IS the real net-gain math now, not a side display.
        out_codes_this = out_codes_all - set(new_squad["code"])
        baseline_info = realistic_baseline_value(squad_df, out_codes_this, gw_list, cfg, bb_play_gw=bb_play_gw)
        baseline_total = baseline_info["baseline_total"]
        net_gain = round(new_total - baseline_total - hit_cost, 2)
        # keyed by actual_k so two requested k's that land on the same real
        # swap count don't create a spurious "tie" against themselves
        if actual_k not in candidates or net_gain > candidates[actual_k]["net_gain"]:
            candidates[actual_k] = {"squad": new_squad, "total": new_total,
                                     "hit_cost": hit_cost, "net_gain": net_gain, "actual_k": actual_k,
                                     "data_gap_codes": result.get("data_gap_codes", []),
                                     "baseline_total": baseline_total,
                                     "baseline_adjusted": baseline_info["adjusted"],
                                     "baseline_zeroed_names": baseline_info["zeroed_names"],
                                     "new_xi": new_breakdown["xi_total"], "new_bench": new_breakdown["bench_total"],
                                     "baseline_xi": baseline_info.get("baseline_xi", baseline_total),
                                     "baseline_bench": baseline_info.get("baseline_bench", 0.0)}

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
        # Patch 41 — same position tie-break as the default recommendation
        # path below applies here too: Force still picks whichever single
        # player the MILP happened to land on for the forced count, with
        # the same blind spot to a statistically-tied alternative.
        if chosen.get("actual_k") == 1:
            chosen, _tie_plan = _position_tie_break(chosen, squad_df, full_pool, gw_list, cfg,
                                                      bench_w, bb_play_gw, moe, team_value, current_gw)
            chosen_net_gain = chosen["net_gain"]
            if _tie_plan:
                plan.extend(_tie_plan)
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
            if chosen.get("baseline_adjusted"):
                zeroed_txt = ", ".join(chosen.get("baseline_zeroed_names", []))
                plan.append(f"GW{current_gw}: Baseline note: {zeroed_txt} isn't a nailed starter over this "
                            f"horizon, so net-gain compares the incoming player against {zeroed_txt} benched for "
                            f"free, not his own live-discounted number.")

    else:
        best_net = max(c["net_gain"] for c in candidates.values())

        def clears_bar(c):
            if c["actual_k"] == 0:
                return True
            bar = threshold if c["hit_cost"] > 0 else meaningful_bar
            return c["net_gain"] >= bar

        tied_ks = sorted(k for k, c in candidates.items() if (best_net - c["net_gain"]) < moe)
        viable = [k for k in tied_ks if clears_bar(candidates[k])]

        # Patch 34 — Starting-XI Impact Check (manager report): a candidate
        # that only clears the bar via bench-autosub-discounted value, with
        # no transferred-in player ever reaching the starting XI across the
        # horizon (and no Bench-Boost-specific override), isn't a real
        # week-to-week scoring change — filtered out here before selection,
        # same tier as Rule #34's margin-of-error filter just above.
        impact_notes: dict[int, dict] = {}
        bench_only_ks: list[int] = []

        def _passes_impact(k):
            if k == 0:
                return True
            c = candidates[k]
            in_codes = set(c["squad"]["code"]) - out_codes_all
            out_codes_this = out_codes_all - set(c["squad"]["code"])
            # Uses the FULL gw_list here, not chip_capped_gw_list — that cap
            # is reserved for the separate "Chip-aware alt" comparison below
            # (a genuinely different question: "is this still worth it if a
            # rebuild chip is coming", not "does it ever have XI impact").
            check = starting_xi_impact_check(squad_df, c["squad"], in_codes, gw_list, cfg,
                                              bb_play_gw=bb_play_gw, out_codes=out_codes_this,
                                              disrupted_codes=disrupted_codes)
            impact_notes[k] = check
            if not check["has_impact"]:
                bench_only_ks.append(k)
            return check["has_impact"]

        viable_real = [k for k in viable if _passes_impact(k)]
        bench_only_viable = viable and not viable_real  # something cleared the bar, but only via bench value
        viable = viable_real
        chosen_k = min(viable) if viable else 0
        chosen = candidates[chosen_k]
        chosen_net_gain = chosen["net_gain"]

        # Patch 41 (manager, 2026-09-14: Damsgaard vs Tavernier both landing
        # at ~2.5 xPts, statistically tied, with the model showing only its
        # own MILP pick and never comparing the two directly) — see
        # _position_tie_break()'s docstring for the full root-cause and
        # scope. Only meaningful for a clean 1-for-1 (actual_k == 1); a
        # no-op otherwise.
        if chosen.get("actual_k") == 1:
            chosen, _tie_plan = _position_tie_break(chosen, squad_df, full_pool, gw_list, cfg,
                                                      bench_w, bb_play_gw, moe, team_value, current_gw)
            chosen_net_gain = chosen["net_gain"]
            if _tie_plan:
                plan.extend(_tie_plan)

        if chosen["actual_k"] == 0 and bench_only_viable:
            bo_k = min(bench_only_ks)
            bo = candidates[bo_k]
            pairs_bo = _pair_moves(squad_df, bo["squad"], this_gw_col)
            move_bits_bo = ", ".join(f"{p['out']} → {p['in']}" for p in pairs_bo)
            summary.append(f"Roll — {move_bits_bo} nets {bo['net_gain']:+.1f} xPts, but that's entirely bench-"
                            f"autosub value: no incoming player reaches your starting XI over this horizon. "
                            f"Saving the transfer is the stronger play.")
            plan.append(f"GW{current_gw}: Roll — {move_bits_bo} clears the materiality/margin-of-error bars "
                        f"({bo['net_gain']:+.2f} xPts) purely via Rule #12's autosub-discounted bench value; the "
                        f"Starting-XI Impact Check (Patch 34) found no transferred-in player entering the best "
                        f"XI in any checked GW, and no Bench Boost override applied. Not a real scoring change — "
                        f"banking the transfer is preferred.")
            # Chip-aware alt (manager request): if a Bench Boost play verdict
            # sits inside this horizon and IS what the bench-only candidate
            # would have helped, surface that as an alternate angle rather
            # than silently vetoing it — the manager may still want it for
            # that specific week.
            bo_check = impact_notes.get(bo_k, {})
            if bo_check.get("bench_boost_gw"):
                plan.append(f"GW{current_gw}: Chip-aware alt: Play {move_bits_bo} — helps your "
                            f"GW{bo_check['bench_boost_gw']} Bench Boost specifically (full bench value that "
                            f"week, not autosub-discounted).")

        elif chosen["actual_k"] == 0:
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
            # Patch 33 — xM rotation-risk badge inline on both legs, so a
            # good net-xPts number doesn't quietly hide an incoming player
            # who isn't actually a confirmed starter.
            move_bits = ", ".join(
                f"{p['out']}{_xm_badge(p.get('out_xm'), cfg)} → {p['in']}{_xm_badge(p.get('in_xm'), cfg)}"
                for p in pairs)
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

            # Patch 34 (2026-09-14 manager report: "why the app still
            # recommending the transfer" when the incoming player didn't
            # visibly reach the XI) — state WHY this move cleared the
            # Starting-XI Impact Check, right on the recommendation itself,
            # instead of leaving it to a disconnected chip-context paragraph
            # or an unexplained side-effect name.
            chosen_check = impact_notes.get(chosen_k, {})
            reason_bits = []
            if chosen_check.get("incoming_entered_gws"):
                gws_txt = ", ".join(f"GW{g}" for g in chosen_check["incoming_entered_gws"])
                reason_bits.append(f"reaches your starting XI at {gws_txt}")
            if chosen_check.get("disrupted_out"):
                reason_bits.append("the outgoing player is disruption-flagged (Rule #41) — moving him on has "
                                    "value on its own, independent of whether the incoming player himself starts")
            if chosen_check.get("bench_boost_gw"):
                reason_bits.append(f"raises your GW{chosen_check['bench_boost_gw']} Bench Boost bench value "
                                    f"(full value that week, not autosub-discounted)")
            # Patch 39 (manager report, 2026-09-14: "keep only the
            # recommendation and avoid the explanation" — everything below
            # used to print directly under the headline move as separate
            # visible lines; it now goes to `plan` only, which the caller
            # shows in the collapsed "Why — full trace" expander, available
            # on demand instead of always-on clutter). `summary` above this
            # point holds exactly one line: the headline move itself.
            if reason_bits:
                plan.append(f"GW{current_gw}: Why this counts as a real change: {'; '.join(reason_bits)}.")

            # Patch 36 disclosure: if the outgoing player wasn't nailed, the
            # net-gain figure above already compares the incoming player
            # against the free bench replacement he'd have gotten anyway,
            # not against his own (possibly near-zero) live projection —
            # say so explicitly rather than leaving the number unexplained.
            if chosen.get("baseline_adjusted"):
                zeroed_txt = ", ".join(chosen.get("baseline_zeroed_names", []))
                plan.append(f"GW{current_gw}: Baseline note: {zeroed_txt} isn't a nailed starter over this "
                            f"horizon (xM tier: rotation/bench risk), so the net-gain above compares the "
                            f"incoming player against what your best XI would score with {zeroed_txt} benched "
                            f"for free — not against {zeroed_txt}'s own live-discounted number.")

            # Patch 38 (manager report, 2026-09-14: "we spent the whole day
            # explaining the logic and still the same issue" — every dispute
            # took a full round-trip of screenshots and manual reconstruction
            # to settle): show the actual arithmetic behind the headline
            # net-gain number directly, so a disagreement can be checked from
            # this one line instead of another round of "is this really the
            # patch you sent." Splits the direct starting-XI swap from the
            # (near-zero, per Patch 37) bench-autosub credit.
            xi_delta = round(chosen.get("new_xi", 0.0) - chosen.get("baseline_xi", 0.0), 2)
            bench_delta = round(chosen.get("new_bench", 0.0) - chosen.get("baseline_bench", 0.0), 2)
            arith_hit_txt = f" after the {chosen['hit_cost']:.0f}-pt hit" if chosen["hit_cost"] > 0 else ""
            plan.append(f"GW{current_gw}: Arithmetic: starting-XI value {chosen.get('baseline_xi', 0.0):.1f} → "
                        f"{chosen.get('new_xi', 0.0):.1f} (direct swap {xi_delta:+.1f} xPts); bench-autosub "
                        f"credit {bench_delta:+.2f} xPts (capped near-zero unless Bench Boost is this week's "
                        f"play); combined net {chosen['net_gain']:+.2f} xPts{arith_hit_txt}.")

            # Minimal "XI shifts" disclosure: an EXISTING squad player (never
            # the incoming transfer target — that's the headline move above,
            # not a side effect) whose bench/XI status changes as a knock-on
            # of this swap, rather than re-showing the whole squad (manager
            # request: "just mentioning the changes not the full squad").
            shift_bits = [f"{', '.join(s['in'])} in GW{s['gw']}" for s in chosen_check.get("shifts", []) if s["in"]]
            if shift_bits:
                plan.append(f"GW{current_gw}: Side effect — already on your bench, no transfer needed: "
                            f"{'; '.join(shift_bits)}.")

            # Patch 34 — chip-aware alt: if a planned full-rebuild chip GW
            # truncates this horizon (chip_capped_gw_list), re-check whether
            # THIS candidate still clears its bar over just the pre-rebuild
            # window — cheap (reuses the already-solved squads, no new MILP
            # solve) and surfaces a genuinely different angle rather than
            # silently using one horizon or the other.
            if chip_capped_gw_list is not None and chip_capped_gw_list != gw_list:
                old_trunc = opt.realized_horizon_value(squad_df, chip_capped_gw_list, cfg) if chip_capped_gw_list \
                    else 0.0
                new_trunc = opt.realized_horizon_value(chosen["squad"], chip_capped_gw_list, cfg) \
                    if chip_capped_gw_list else 0.0
                net_trunc = round(new_trunc - old_trunc - chosen["hit_cost"], 2)
                bar_trunc = threshold if chosen["hit_cost"] > 0 else meaningful_bar
                if not chip_capped_gw_list or net_trunc < bar_trunc:
                    plan.append(f"GW{current_gw}: Chip-aware alt: Roll — a full-rebuild chip is planned before "
                                f"this horizon ends, and this move doesn't clear its bar over just the "
                                f"pre-rebuild window ({net_trunc:+.1f} xPts) — the rebuild would replace this "
                                f"player anyway.")

            used_free = min(chosen["actual_k"], free_transfers)
            rolled = free_transfers - used_free
            if rolled > 0:
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
                plan.append(f"GW{current_gw}: Data gap — {names_txt} had a missing `xpts_horizon_sum`/`price` "
                            f"this run; patched to 0.0 rather than dropped, per Standing Rule #4. Check their raw "
                            f"projection before trusting this move if they're one of the players above.")

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
                              target_code, default_net_gain: float | None = None,
                              disrupted_codes: set | None = None,
                              bb_play_gw: int | None = None) -> dict:
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

    empty = {"feasible": False, "already_owned": False, "summary": [], "plan": [], "moves": [],
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
    # Patch 37: same near-zero bench weight (except bb_play_gw) as
    # suggest_transfers() — a manager-named target shouldn't clear the bar on
    # bench-quality credit alone any more than a model-picked one should.
    bench_w = cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08)
    old_total = opt.realized_horizon_value(squad_df, gw_list, cfg, bench_weight_scale=bench_w, bb_play_gw=bb_play_gw)
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
        new_total = opt.realized_horizon_value(new_squad, gw_list, cfg, bench_weight_scale=bench_w,
                                                bb_play_gw=bb_play_gw)
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
        move_bits = ", ".join(
            f"{p['out']}{_xm_badge(p.get('out_xm'), cfg)} → {p['in']}{_xm_badge(p.get('in_xm'), cfg)}"
            for p in pairs)
        summary = [f"No hit-free way to add this player this run — your sidebar stance is 'No hits', and every "
                    f"legal route found needs at least a {cheapest['hit_cost']:.0f}-pt hit.",
                   f"Informational only, not recommended under 'No hits': the cheapest hit-requiring route — "
                    f"{move_bits} ({cheapest['actual_k']} transfer(s), −{cheapest['hit_cost']:.0f} pts) — "
                    f"nets {cheapest['net_gain']:+.1f} xPts over {horizon_n} GW(s)."]
        if default_net_gain is not None:
            summary.append("Switch to 'Hit if worth it' or 'Force' in the sidebar to let this scenario actually "
                            "recommend a hit-costing route.")
        return {"feasible": True, "already_owned": False, "summary": summary, "plan": [], "moves": moves,
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
    move_bits = ", ".join(
        f"{p['out']}{_xm_badge(p.get('out_xm'), cfg)} → {p['in']}{_xm_badge(p.get('in_xm'), cfg)}"
        for p in pairs)
    hit_note = f" (−{chosen['hit_cost']:.0f} pt hit)" if chosen["hit_cost"] > 0 else " (free)"
    verdict_note = ("clears its bar — a genuine improvement" if clears else
                     f"doesn't clear the {bar} xPts bar this move needs — not worth it as evaluated")
    # Patch 39 (manager, 2026-09-14: "keep only the recommendation and avoid
    # the explanation ... for the evaluate scenario"): `summary` carries only
    # this one headline line; everything else below goes to `plan` only,
    # shown in a collapsed "Why — full trace" expander by the caller, same
    # declutter already applied to suggest_transfers()/plan_transfer_schedule().
    summary = [f"Your scenario — {move_bits}{hit_note}: net {chosen['net_gain']:+.1f} xPts over "
               f"{horizon_n} GW(s), {verdict_note}."]
    plan = []

    # Patch 34 — Starting-XI Impact Check, disclosure only here (not a gate
    # like suggest_transfers()/plan_transfer_schedule(): this is a manager-
    # chosen target, so the model states what it found rather than
    # overriding an explicit choice with Roll).
    in_codes_scenario = set(chosen["squad"]["code"]) - out_codes_all
    out_codes_scenario = out_codes_all - set(chosen["squad"]["code"])
    scenario_check = starting_xi_impact_check(squad_df, chosen["squad"], in_codes_scenario, gw_list, cfg,
                                               out_codes=out_codes_scenario, disrupted_codes=disrupted_codes)
    if not scenario_check["has_impact"]:
        plan.append(f"Note: {target_row['web_name']} isn't projected to reach your starting XI at any point "
                    f"in this horizon — this net gain is entirely bench-autosub value (Standing Rule #12), "
                    f"not a real week-to-week scoring change.")
    else:
        reason_bits = []
        if scenario_check.get("incoming_entered_gws"):
            gws_txt = ", ".join(f"GW{g}" for g in scenario_check["incoming_entered_gws"])
            reason_bits.append(f"{target_row['web_name']} reaches your starting XI at {gws_txt}")
        if scenario_check.get("disrupted_out"):
            reason_bits.append("the outgoing player is disruption-flagged (Rule #41)")
        if reason_bits:
            plan.append(f"Why this counts as a real change: {'; '.join(reason_bits)}.")
        shift_bits = [f"{', '.join(s['in'])} in GW{s['gw']}" for s in scenario_check.get("shifts", []) if s["in"]]
        if shift_bits:
            plan.append(f"Side effect — already on your bench, no transfer needed: {'; '.join(shift_bits)}.")

    if chosen["actual_k"] > 1:
        # Say WHY more than one swap was needed using the diagnostic actually
        # computed above (budget/club-limit/data-gap), never a generic guess
        # — a prior version of this message asserted a formation-shape cause
        # by default, which turned out to be wrong for a same-position
        # MID-for-MID swap in a real case; that guess is retired in favor of
        # the concrete checks above.
        plan.append(f"Needed {chosen['actual_k']} linked swaps, not a single 1-for-1:")
        plan.extend(diagnostic)
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
            plan.append(f"Data gap flagged: {names_txt} had a missing projection this run and was treated as "
                        f"0 xPts so it wouldn't be silently forced out of your squad — if the solver also "
                        f"swapped this player, it may be reacting to that gap rather than a genuine "
                        f"upgrade. Worth checking their raw projection before trusting that leg of the move.")
    if default_net_gain is not None:
        diff = round(chosen["net_gain"] - default_net_gain, 2)
        if abs(diff) < moe:
            plan.append(f"Statistically tied with the model's own pick this run (within the {moe:.1f} xPts "
                        f"margin-of-error).")
        elif diff > 0:
            plan.append(f"This nets {diff:+.1f} xPts more than the model's own recommendation this run.")
        else:
            plan.append(f"This nets {diff:.1f} xPts less than the model's own recommendation this run.")

    return {"feasible": True, "already_owned": False, "summary": summary, "plan": plan, "moves": moves,
            "net_gain": chosen["net_gain"], "hit_cost": chosen["hit_cost"], "clears_bar": clears,
            "chosen_k": chosen["actual_k"]}


def _xm_tier(xm: float | None, cfg: dict | None = None) -> str | None:
    """Shared nailed/rotation/bench-risk classification (Patch 33's display
    tiers, split out in Patch 36 so the SAME threshold that decides the
    xM badge also decides whether an OUT candidate's own projection can be
    trusted as this week's baseline — see `realistic_baseline_value()`).
    Anchored on the xM Floor Rule's own 0.88 "confirmed nailed" figure
    (`xm_heuristic.confirmed_current_season_start_floor` in
    model_config.yaml) as the top of the "nailed" band. Returns None for a
    missing xm (caller decides the fail-open behavior)."""
    if xm is None or pd.isna(xm):
        return None
    nailed_floor = (cfg or {}).get("xm_heuristic", {}).get("confirmed_current_season_start_floor", 0.88)
    if xm >= nailed_floor - 0.08:  # a little below the strict "confirmed nailed" floor still reads as safe
        return "nailed"
    elif xm >= 0.5:
        return "rotation"
    return "risk"


def _xm_badge(xm: float | None, cfg: dict | None = None) -> str:
    """Patch 33 (manager report: a transfer's xPts already bakes in expected
    minutes via `xm`, but that was never disclosed next to the recommendation
    itself, so a rotation risk masquerading as a good net-xPts number was
    invisible without opening the raw projection). Tiers are a disclosed,
    manager-directed display extension — not a doc-specified threshold, same
    pattern as chip_advisor_thresholds/chip_shape_test elsewhere."""
    tier = _xm_tier(xm, cfg)
    if tier is None:
        return ""
    label = {"nailed": "nailed", "rotation": "rotation risk", "risk": "bench risk"}[tier]
    cls = {"nailed": "nailed", "rotation": "rotation", "risk": "risk"}[tier]
    return (f' <span class="xm-badge {cls}" title="Expected-minutes multiplier (xM) {xm:.2f} — already priced '
            f'into this player\'s xPts above, shown here so rotation risk isn\'t hidden behind a good net number.">'
            f'xM {xm:.2f} {label}</span>')


def realistic_baseline_value(squad_df: pd.DataFrame, out_codes: set, gw_list: list[int], cfg: dict,
                              bb_play_gw: int | None = None) -> dict:
    """Patch 36 (manager report, 2026-09-14, following the Foden->Damsgaard
    case): "if the model chosen someone to be replaced ... the model needs
    to check maybe the recommended player [i.e. the OUT candidate] is a good
    pick on the horizon so we should keep him on the bench for just 1GW, if
    he isn't nailed for 3 GWs then the model should choose the starting xi
    without this player ... then compare the recommended player replacement
    with the starting xi to confirm if it's worth it."

    Two-step check, confirmed with the manager:
    1. A NAILED out-player (xM tier, same threshold as the xM badges) keeps
       his own real per-GW projection as the baseline — his number already
       fairly reflects a genuine fixture-driven dip, so second-guessing it
       would be wrong. A single bad GW for an otherwise-secure starter is
       not, on its own, a reason to distrust the model's number for him.
    2. A NON-NAILED out-player (rotation/bench risk) may still carry a
       live-data-driven projection that's more optimistic than reality
       (the same gap `fpl_engine.free_lineup_fix_check()` surfaces for a
       disruption-flagged player, generalized here to ANY shaky starter,
       and now feeding the transfer's own net-gain math directly per the
       manager's explicit choice, not just sitting beside it as a
       footnote): this recomputes the no-transfer baseline with that
       player's projection zeroed for the checked GWs, so the incoming
       transfer target is judged against what you'd already get for free
       from your own bench, not against a possibly-generous number.

    Applied per GW across the whole checked window (manager: "check the
    horizon if the side bar is for more than on GW") via
    `opt.realized_horizon_value()`'s own per-GW best-XI selection — zeroing
    a non-nailed player for the GWs he's actually being evaluated over,
    never just GW1.

    Returns {"baseline_total": float, "adjusted": bool, "zeroed_names":
    [name, ...]}. "adjusted": False means every out-player was nailed, so
    `baseline_total` is just the plain, unmodified realized value.

    Patch 37: uses the same near-zero-except-bb_play_gw bench weighting as
    the rest of the transfer-recommendation net-gain math (manager report:
    a stronger bench alone — e.g. a non-nailed OUT player who was already
    benched at 0, so THIS function's own zeroing has no effect on the XI —
    shouldn't be able to inflate the comparison via bench-autosub credit)."""
    bench_w = cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08)
    if squad_df is None or squad_df.empty or not out_codes or not gw_list:
        if squad_df is not None and not squad_df.empty:
            bd = opt.realized_horizon_breakdown(squad_df, gw_list, cfg, bench_weight_scale=bench_w,
                                                 bb_play_gw=bb_play_gw)
        else:
            bd = {"xi_total": 0.0, "bench_total": 0.0, "total": 0.0}
        return {"baseline_total": bd["total"], "baseline_xi": bd["xi_total"], "baseline_bench": bd["bench_total"],
                "adjusted": False, "zeroed_names": []}

    non_nailed = []
    for code in out_codes:
        row = squad_df[squad_df["code"] == code]
        if row.empty:
            continue
        tier = _xm_tier(row.iloc[0].get("xm"), cfg)
        if tier in ("rotation", "risk"):
            non_nailed.append((code, row.iloc[0].get("web_name")))

    if not non_nailed:
        bd = opt.realized_horizon_breakdown(squad_df, gw_list, cfg, bench_weight_scale=bench_w,
                                             bb_play_gw=bb_play_gw)
        return {"baseline_total": bd["total"], "baseline_xi": bd["xi_total"], "baseline_bench": bd["bench_total"],
                "adjusted": False, "zeroed_names": []}

    zeroed_squad = squad_df.copy()
    for code, _ in non_nailed:
        for gw in gw_list:
            col = f"xpts_gw{gw}"
            if col in zeroed_squad.columns:
                zeroed_squad.loc[zeroed_squad["code"] == code, col] = 0.0
    bd = opt.realized_horizon_breakdown(zeroed_squad, gw_list, cfg, bench_weight_scale=bench_w,
                                         bb_play_gw=bb_play_gw)
    return {"baseline_total": bd["total"], "baseline_xi": bd["xi_total"], "baseline_bench": bd["bench_total"],
            "adjusted": True, "zeroed_names": [n for _, n in non_nailed]}


def _move_row(p: dict, hit_cost: float, net_gain: float, justified: bool) -> dict:
    """One pair (from `_pair_moves`/`_apply_eo_pull`) -> a move-table row.
    `hit_cost`/`net_gain` are the BATCH total for the whole chosen transfer
    path, repeated on every row in the batch (not summed per-row) — a joint
    k-transfer solve incurs one combined hit fee, not k independent ones,
    so splitting it across rows would misstate what any single row cost on
    its own. The accompanying plan-text line states the batch total once."""
    return {
        "out": p["out"], "out_code": p["out_code"], "out_team": p["out_team"], "out_price": p["out_price"],
        "out_xm": p.get("out_xm"),
        "in": p["in"], "in_code": p["in_code"], "in_team": p["in_team"], "in_price": p["in_price"],
        "in_xm": p.get("in_xm"),
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
