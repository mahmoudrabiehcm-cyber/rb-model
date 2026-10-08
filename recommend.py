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
                    this_gw_col: str, out_codes: set, in_codes: set,
                    retained_club_counts: dict | None = None,
                    retained_codes: set | None = None) -> list[dict]:
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
    never downgrade genuine projection for a cheaper narrative.

    Patch 56 (manager report, 2026-09-20, screenshot: a GW6 Wildcard "what-
    if" squad came back with 4 Leeds players — Bogle, Bijol, Justin, and
    Trafford). Root cause (verified in code): the joint solve that picks
    `pairs` in the first place (`optimizer.solve_squad()`) always enforces
    max-3-per-club, but this function runs AFTER that solve and substitutes
    individual IN legs for cheaper/more-differential same-position
    alternatives per the active Style Profile's `eo_pull` setting — and
    never checked the substitute's club against the rest of the squad being
    assembled. Each leg was evaluated independently, so two different legs
    could each legally pick a same-position, same-price-band Leeds player
    with nothing to notice the combined squad now had 4. This is the one
    squad-construction path in the whole codebase that lacked a club-count
    check — `optimizer.solve_squad()` (max_per_club constraint on every
    team) and `optimizer.solve_xi_first_squad()` (both its XI-stage MILP and
    its bench-stage cap on `max_per_club - already-in-XI`) already enforce
    it correctly.

    Fix: `retained_club_counts` (the club counts of whichever squad members
    are NOT being transferred this batch — passed in by the caller, who
    already knows the full old/new squad) seeds a running `club_counts`
    tally that's updated after every leg, substituted or not, in processing
    order. A substitute is only considered if its club's running count is
    still below `max_per_club` — so leg 2 correctly sees leg 1's already-
    assigned club before it gets to pick, the same way
    `solve_xi_first_squad`'s bench stage already accounts for the XI's club
    counts. `retained_club_counts=None` (the old call signature) preserves
    the exact old, buggy-if-not-fixed-by-caller behavior for any caller
    that hasn't been updated — every current caller has been.

    Patch 59 (manager report, 2026-09-21: "only 2 benched players!!" on a
    Wildcard scenario). Second, separate bug found investigating that report
    (reproduced directly against this function with synthetic data, not
    guessed): the substitute-candidate pool was only ever excluded by
    `taken_in_codes | out_codes` — it never excluded `retained_codes`, i.e.
    the squad members who are NEITHER being transferred out NOR one of this
    batch's own IN legs, but are simply staying in the squad untouched. A
    same-position eo_pull substitution for one leg could legally "pick" an
    already-retained player as its cheaper/more-differential alternative —
    that code then appears BOTH in the untouched `retained` slice AND in
    this batch's substituted `in_rows` slice when the caller reconstructs
    the final squad, and `drop_duplicates(subset="code")` silently collapses
    the two into one row, shrinking the squad below 15 with no error and no
    warning (reproduced: 2 collisions -> a 13-man squad, 11 XI + 2 bench —
    matching the report exactly). `retained_codes` (new parameter) is now
    added to the same exclusion set as `taken_in_codes`/`out_codes`, so a
    substitute can never re-pick a player the squad is already keeping.
    `retained_codes=None` (the old call signature) preserves prior behavior
    for any caller not yet updated — every current caller has been."""
    eo_pull = profile.get("eo_pull", "none")
    if eo_pull == "none" or not pairs:
        return pairs
    ceiling_key = profile.get("differential_ceiling")
    taken_in_codes = set(in_codes)  # never pick a code the solve already used elsewhere in this batch
    excluded_codes = taken_in_codes | set(out_codes) | (set(retained_codes) if retained_codes else set())
    max_per_club = cfg["squad_rules"]["max_per_club"]
    club_counts = dict(retained_club_counts or {})

    result = []
    for p in pairs:
        pos = p["position"]
        pos_pool = full_pool[(full_pool["position"] == pos) &
                              (full_pool.get("status", pd.Series(dtype=object)) == "a")].copy()
        pos_pool = pos_pool[pos_pool["code"] != p["in_code"]]
        # Patch 59 — `excluded_codes` now also carries `retained_codes` (see
        # docstring): a substitute must never collide with a squad member
        # who's already staying untouched, not just with this batch's own
        # out-players or already-chosen in-players.
        pos_pool = pos_pool[~pos_pool["code"].isin(excluded_codes)]
        # Patch 56 — a substitute may only be considered if its club still
        # has room under max_per_club, given the squad assembled so far
        # (retained players + every earlier leg's, possibly substituted,
        # pick this batch).
        # Patch 59 (manager report: "only 2 benched players!!") — Patch 56's
        # own fix had a real crash bug: when `pos_pool` is ALREADY EMPTY at
        # this point (a completely normal case — e.g. every same-position,
        # not-yet-taken candidate got filtered out by the MOE/price checks
        # above, nothing to do with club caps at all), `pos_pool["team"].
        # map(...)` on a 0-row Series returns an empty Series with dtype
        # `object` (pandas can't infer `bool` from zero elements) — and
        # boolean-indexing an already-empty DataFrame with an object-dtype
        # (not bool-dtype) empty mask silently collapses it to 0 COLUMNS,
        # not just 0 rows (reproduced directly: `pd.DataFrame([],
        # columns=[...])[<0-row object-dtype mask>]` -> shape (0, 0)). The
        # very next line then raised `KeyError: 'xpts_horizon_sum'` reading
        # a column that no longer existed on the corrupted frame — an
        # unhandled exception inside a Wildcard/transfer-recommendation
        # solve, which is consistent with a squad coming back short (e.g.
        # the reported 2-bench/13-total squad) if anything upstream caught
        # and partially recovered from it, or with the section failing
        # outright. Guarding the `.map()` call to only run when `pos_pool`
        # already has rows sidesteps the empty-Series dtype gotcha entirely
        # -- an empty `pos_pool` simply stays empty (0 rows, all columns
        # intact), and the existing `if tied.empty: result.append(p)...`
        # fallback below already handles that correctly (keep the original
        # pick, no crash) exactly as it was designed to.
        if not pos_pool.empty:
            pos_pool = pos_pool[pos_pool["team"].map(lambda t: club_counts.get(t, 0) < max_per_club)]
        moe = eng.margin_of_error_threshold(p["in_xpts"], cfg)
        tied = pos_pool[(pos_pool["xpts_horizon_sum"] >= (p["in_xpts"] - moe)) &
                         (pos_pool["price"] <= p["in_price"])]

        if eo_pull != "strong_high_eo":
            floor = style_profiles.differential_floor(pos_pool, "xpts_horizon_sum", ceiling_key)
            tied = tied[tied["xpts_horizon_sum"] >= floor]

        if tied.empty:
            result.append(p)
            taken_in_codes.add(p["in_code"])
            excluded_codes.add(p["in_code"])  # Patch 59 — keep excluded_codes in sync with taken_in_codes
            club_counts[p["in_team"]] = club_counts.get(p["in_team"], 0) + 1
            continue

        best = (tied.sort_values("selected_by_percent", ascending=False).iloc[0] if eo_pull == "strong_high_eo"
                else tied.sort_values("selected_by_percent", ascending=True).iloc[0])
        if best["code"] == p["in_code"]:
            result.append(p)
            taken_in_codes.add(p["in_code"])
            excluded_codes.add(p["in_code"])  # Patch 59 — keep excluded_codes in sync with taken_in_codes
            club_counts[p["in_team"]] = club_counts.get(p["in_team"], 0) + 1
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
        excluded_codes.add(best["code"])  # Patch 59 — keep excluded_codes in sync with taken_in_codes
        club_counts[best["team"]] = club_counts.get(best["team"], 0) + 1
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
    # Patch 56 — the retained (non-transferred) players' club counts seed
    # _apply_eo_pull's running max-per-club check, so a same-position
    # substitute can never push a club past the cap across independently
    # -processed legs (see that function's docstring for the reported bug).
    retained_club_counts = current_squad[~current_squad["code"].isin(out_codes)]["team"].value_counts().to_dict()
    # Patch 59 — see _apply_eo_pull's docstring: a substitute must never
    # re-pick a code that's already staying in the squad untouched (this is
    # the actual root cause of the reported "only 2 benched players").
    _retained_codes_for_eo_pull = set(current_squad["code"]) - out_codes
    pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes, in_codes, retained_club_counts,
                            _retained_codes_for_eo_pull)

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
    # Patch 111: the model's own pick may sit OUTSIDE the tied set (others beat it by more than the margin
    # while tied with each other) -- its extended score is still needed for the "switched" message below.
    # The winner is still chosen among the tied candidates only (unchanged behaviour).
    if in_code not in ext_scores:
        _ib = opt.realized_horizon_breakdown(scored[in_code]["squad"], extended_gw_list, cfg,
                                              bench_weight_scale=bench_w, bb_play_gw=bb_play_gw)
        ext_scores[in_code] = round(_ib["total"] - ext_baseline - hit_cost, 2)

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


def resolve_cross_check_horizon(planning_gw: int, current_max_gw: int, target_gw: int | None,
                                 max_extension: int = 8) -> dict | None:
    """Patch 95 (v6.9 Rule #49 cross-check extension — manager discussion,
    2026-09-30): decides whether the Wildcard cross-check's horizon needs
    extending to reach a scheduled chip's GW, and how far, BEFORE either
    extra family of MILP solves (a longer `plan_transfer_schedule()` run, a
    wider `data_pipeline.solve_reachable_ceiling_by_gw()` window) is
    invoked. `current_max_gw` is whichever of the transfer-plan horizon and
    the reachable-ceiling detection window currently reaches LESS far (the
    real limiting factor today — confirmed via code read that these are two
    separate, differently-sized windows, neither tied to when a chip is
    actually scheduled).

    Returns None when no extension applies — `target_gw` is unset, or
    already within `current_max_gw` (nothing to extend for). Otherwise
    returns {"gw_list": [...], "reached_target": bool}: `gw_list` runs from
    `planning_gw` through `min(target_gw, current_max_gw + max_extension)` —
    capping how far this cross-check-only computation extends beyond the
    app's normal windows, so a chip scheduled very far out doesn't trigger
    an unbounded extra solve. `reached_target` is False when the cap
    prevented `gw_list` from actually reaching `target_gw`, so the caller
    can disclose the cap explicitly rather than silently show a comparison
    that still doesn't cover the real chip week."""
    if target_gw is None or target_gw <= current_max_gw:
        return None
    capped_target = min(target_gw, current_max_gw + max_extension)
    return {"gw_list": list(range(planning_gw, capped_target + 1)),
            "reached_target": capped_target >= target_gw}


def wc_extend_requested(session_state_flag: bool, auto_wildcard_gw: int | None) -> bool:
    """Patch 101 (2026-10-01, manager: "it's 4 minutes 46 seconds now ,, i
    need it below 2 minutes"). The Patch 95 extended Wildcard cross-check
    (resolve_cross_check_horizon() above, run via app.py's
    _extended_wc_cross_check_calc()) used to fire automatically whenever a
    Wildcard was scheduled beyond the normal window -- a real, measured cost
    (a full extra plan_transfer_schedule() solve plus a full extra
    solve_reachable_ceiling_by_gw() scan over +8 GWs) the manager had
    explicitly chosen to keep running every time ("i want to be with the
    run" -- prior session). With the new 2-minute target making every solve
    matter, the manager revisited that call and asked to make it opt-in,
    the same button-gated pattern Cross-Tool Reconciliation already uses.

    Patch 104 (2026-10-01, manager screenshot: the button this gates had
    gone correctly-but-unhelpfully disabled once Patch 103 widened the
    normal check's own reach far enough to already cover the one scheduled
    Wildcard on screen -- "why it's grayed , it should give an option for
    another 5 GWs beyond the one maximum used for the current run" /
    "i didn't get it!!"). Confirmed with the manager directly (AskUserQuestion)
    that the button was built on the wrong philosophy: a conditional
    "only matters if a scheduled chip needs it" gate, when what's wanted is
    an always-available manual "push this check further out" control,
    useful on its own merits even with nothing currently scheduled beyond
    the normal reach. `auto_wildcard_gw` is therefore no longer part of the
    gate at all -- kept as a parameter (ignored) only so existing call
    sites/tests that pass it don't need to change shape. The only real gate
    now is the manager's own explicit click this run."""
    return bool(session_state_flag)


def resolve_wildcard_extend_target(current_max_gw: int, auto_wildcard_gw: int | None,
                                    fixed_increment: int = 5) -> int:
    """Patch 104 (2026-10-01, manager screenshot + AskUserQuestion confirmation):
    the extend button's target GW, now computed unconditionally every run
    (the button itself is always clickable -- see wc_extend_requested()
    above). Always reaches at least `current_max_gw + fixed_increment`
    ("another 5 GWs beyond the one maximum used for the current run", the
    manager's own words) regardless of whether any chip is scheduled in
    that range. If a chip IS scheduled further out than the fixed
    increment would reach, the target extends to cover it too -- the
    button should never show a target that's short of a real, known
    decision point just because the fixed increment alone would stop
    earlier.

    `current_max_gw`: the normal (unextended) cross-check's own reach
    (recommend.resolve_wildcard_check_gw_window(...)[-1] in app.py).
    `auto_wildcard_gw`: the chip portfolio's scheduled Wildcard GW, or
    None if none is scheduled.
    """
    target = current_max_gw + fixed_increment
    if auto_wildcard_gw is not None and auto_wildcard_gw > target:
        target = auto_wildcard_gw
    return target


def wildcard_verdict(trigger_active: bool, seq_scheduled, seq_gw, seq_value, closes, gap_pct) -> dict | None:
    """Patch 108 (2026-10-04, manager: "if my team for this week is 90% do you
    think that triggering the chip is the right call ... is this the best way
    to present it"). The 6% trigger (Standing Rule #45) is UNCHANGED and still
    decides `trigger_active`; this only decides how the card presents the
    decision, using values the model already computes -- no new thresholds:

      closes is True                      -> MONITOR (your recommended transfers
                                             already close the gap)
      scheduled by the joint sequence     -> PLAY GWn, headline = net xPts vs
                                             the best no-chip transfer path
      sequence found no positive slot     -> HOLD (chip worth more later)
      no joint sequence (<2 chips left)   -> TRIGGER ACTIVE + gap %, as before

    Returns None when the trigger isn't active (card keeps its existing path)."""
    if not trigger_active:
        return None
    gap_txt = f"{gap_pct:.1f}%" if gap_pct is not None else "n/a"
    if closes is True:
        return {"badge": "MONITOR", "badge_cls": "hold", "card_cls": "",
                "stat": f"{gap_txt} gap", "sub": f"transfers close the {gap_txt} gap — no chip needed yet. "
                f"Measured with the plan's full transfer allowance over the window, not only the free transfers banked now",
                "note": "Your recommended transfers already close this", "note_cls": "good"}
    if seq_scheduled is True and seq_gw is not None:
        stat = f"{seq_value:+.1f} xPts" if seq_value is not None else f"{gap_txt} gap"
        return {"badge": f"PLAY GW{seq_gw}", "badge_cls": "play", "card_cls": "is-play",
                "stat": stat, "sub": f"vs best transfers · need gap {gap_txt}",
                "note": "Transfers alone don't close it", "note_cls": "warn"}
    if seq_scheduled is False:
        return {"badge": "HOLD", "badge_cls": "hold", "card_cls": "",
                "stat": f"{gap_txt} gap", "sub": f"no positive slot — chip worth more later · gap {gap_txt}",
                "note": None, "note_cls": ""}
    return {"badge": "TRIGGER ACTIVE", "badge_cls": "active", "card_cls": "is-active",
            "stat": f"{gap_txt} gap", "sub": "structural gap detected — date is your call",
            "note": None, "note_cls": ""}


def bridge_verdict(ratings: dict | None, planning_gw: int, threshold: float = 94.0,
                    consecutive: int = 2) -> dict | None:
    """Patch 110 (2026-10-04, manager: "can the free transfers + benching carry
    the team to the best chip GW?"). `ratings` = {gw: Rating %} of the squad you
    would field using FREE TRANSFERS ONLY (no hits, flagged players benched)
    against a full-rebuild ("dream") squad for that GW. The bridge BREAKS at the
    first GW that starts `consecutive` weeks in a row below `threshold` (a single
    bad week is a blip, not a dead team). The 94% line is the document's 6% trigger
    gap (Standing Rule #45), reused -- NOT separately validated.
    Returns None when there are no ratings."""
    clean = {int(g): float(r) for g, r in (ratings or {}).items() if r is not None}
    if not clean:
        return None
    gws = sorted(clean)
    need = min(max(1, consecutive), len(gws))
    break_gw = None
    for i in range(len(gws) - need + 1):
        if all(clean[g] < threshold for g in gws[i:i + need]):
            break_gw = gws[i]
            break
    min_gw = min(gws, key=lambda g: clean[g])
    return {"holds": break_gw is None, "break_gw": break_gw, "min_rating": clean[min_gw], "min_gw": min_gw,
            "threshold": threshold, "consecutive": need, "ratings": clean, "planning_gw": planning_gw}


def wildcard_final_decision(trigger_active: bool, bridge: dict | None, seq_scheduled, seq_gw, seq_value,
                             gap_pct, planning_gw: int, today_pct: float | None = None) -> dict | None:
    """Patch 110: the ONE Wildcard decision that the card, the top pill and the
    best-GW table all read. The 6% trigger (Rule #45) still decides whether this
    runs; the bridge decides WHEN:

      bridge breaks at GW b      -> PLAY NOW (b == current GW) / PLAY GWb
      bridge holds + sequence GW -> HOLD -> GWn (play the sequence's best week)
      bridge holds, no slot      -> HOLD (no week)
      no bridge available        -> unverified: shows the sequence's week, if any

    `gw` is the single Wildcard week the table shows (None = none)."""
    if not trigger_active:
        return None
    today_txt = f"today {today_pct:.0f}%" if today_pct is not None else None
    gap_txt = f"{gap_pct:.1f}%" if gap_pct is not None else "n/a"
    thr = bridge["threshold"] if bridge else 94.0

    def _pack(kind, gw, badge, cls, card_cls, stat, sub, pill, note=None, note_cls=""):
        return {"kind": kind, "gw": gw, "badge": badge, "badge_cls": cls, "card_cls": card_cls, "stat": stat,
                "sub": sub, "pill": pill, "note": note, "note_cls": note_cls}

    if bridge is None:
        if seq_scheduled and seq_gw is not None:
            return _pack("unverified", seq_gw, f"SEQUENCE GW{seq_gw}", "active", "is-active",
                         f"{seq_value:+.1f} xPts" if seq_value is not None else f"{gap_txt} gap",
                         " · ".join(x for x in [today_txt, "bridge check unavailable"] if x),
                         f"Wildcard: sequence says GW{seq_gw} (bridge unavailable)")
        return _pack("unverified", None, "TRIGGER ACTIVE", "active", "is-active", f"{gap_txt} gap",
                     " · ".join(x for x in [today_txt, "bridge check unavailable"] if x),
                     "Wildcard: trigger active (bridge unavailable)")
    if not bridge["holds"]:
        b = bridge["break_gw"]
        low = bridge["ratings"].get(b)
        sub = " · ".join(x for x in [today_txt, f"free transfers can't hold {thr:.0f}%"] if x)
        if b == planning_gw:
            return _pack("play_now", b, "PLAY NOW", "play", "is-play", f"{low:.0f}% vs dream", sub,
                         f"Wildcard: PLAY NOW — free transfers can't hold {thr:.0f}%",
                         "Below the line this week and next", "warn")
        return _pack("play", b, f"PLAY GW{b}", "play", "is-play", f"{low:.0f}% at GW{b}", sub,
                     f"Wildcard: PLAY GW{b} — free transfers stop holding {thr:.0f}%",
                     "Bridge breaks here", "warn")
    if seq_scheduled and seq_gw is not None:
        sub = " · ".join(x for x in [today_txt, f"free transfers keep you ≥{thr:.0f}% until then"] if x)
        return _pack("hold_until", seq_gw, f"HOLD → GW{seq_gw}", "hold", "",
                     f"{seq_value:+.1f} xPts" if seq_value is not None else f"{gap_txt} gap", sub,
                     f"Wildcard: HOLD → GW{seq_gw} — free transfers cover until then",
                     "Free transfers carry the team", "good")
    sub = " · ".join(x for x in [today_txt, "no positive Wildcard slot found"] if x)
    return _pack("hold", None, "HOLD", "hold", "", f"{gap_txt} gap", sub,
                 "Wildcard: HOLD — no positive slot", None, "")


def final_chips_by_gw(seq_gws: dict | None, labels: dict, wc_decision: dict | None,
                       wc_in_play: bool) -> tuple[dict, list[str]]:
    """Patch 110: {gw: [chip labels]} for the table/headline from ONE source. Bench
    Boost / Triple Captain / Free Hit come from the joint sequence; the Wildcard
    appears ONLY at the decision's GW (none when HOLD without a slot). When the
    trigger is not active, a sequence Wildcard is labelled "(value only)".
    Returns (by_gw, collision_notes)."""
    seq_gws = seq_gws or {}
    by_gw: dict[int, list[str]] = {}
    for ck, info in seq_gws.items():
        if ck == "wildcard":
            continue
        by_gw.setdefault(info["gw"], []).append(labels.get(ck, ck))
    notes: list[str] = []
    if wc_in_play:
        wc_gw, wc_label = None, "Wildcard"
        if wc_decision is not None:
            wc_gw = wc_decision.get("gw")
        elif seq_gws.get("wildcard"):
            wc_gw, wc_label = seq_gws["wildcard"]["gw"], "Wildcard (value only)"
        if wc_gw is not None:
            if wc_gw in by_gw:
                notes.append(f"Wildcard and {', '.join(by_gw[wc_gw])} both land on GW{wc_gw} — only one chip per GW.")
            by_gw.setdefault(wc_gw, []).append(wc_label)
    return by_gw, notes


def chip_plan_headline(chips_by_gw: dict, wc_decision: dict | None, tc_player: dict | None = None) -> str:
    """Patch 110: one line, GW order, same data as the table. Patch 113: the Triple Captain names its player."""
    def _lab(g, v):
        return [(f"Triple Captain ({tc_player['name']})" if (x == "Triple Captain" and tc_player and tc_player.get("gw") == g
                                                              and tc_player.get("name")) else x) for x in v]
    parts = [f"{', '.join(_lab(g, v))} GW{g}" for g, v in sorted(chips_by_gw.items())]
    head = ""
    if wc_decision is not None and wc_decision.get("gw") is None:
        head = "Wildcard HOLD (no week) · "
    elif wc_decision is not None and wc_decision.get("kind") == "hold_until":
        head = "Wildcard on HOLD until its week · "
    return "Chip plan: " + head + " · ".join(parts) if parts else "Chip plan: " + (head.rstrip(" ·") or "no chip scheduled")


def extended_chip_target(planning_gw: int, chip_span: int = 12) -> int:
    """Patch 110: the furthest GW the Extended Check reaches for the BB/TC/FH
    windows (Current + chip_span - 1 = Current + 11)."""
    return int(planning_gw) + int(chip_span) - 1


def chain_candidate_gws(wc_by_gw: dict | None, planning_gw: int, eval_gws: list, seq_gw: int | None = None,
                         max_n: int = 4) -> list[int]:
    """Patch 111: the Wildcard weeks worth a full chained plan (each costs one plan + one rebuild
    solve). Always the current GW and the joint sequence's GW (when inside `eval_gws`), plus the
    top Rule #48 window-value weeks up to `max_n` extra. Sorted ascending."""
    ev = set(eval_gws or [])
    cands: set[int] = set()
    if planning_gw in ev:
        cands.add(planning_gw)
    if seq_gw is not None and seq_gw in ev:
        cands.add(seq_gw)
    scored = [(v.get("gap"), g) for g, v in (wc_by_gw or {}).items()
              if g in ev and isinstance(v, dict) and v.get("gap") is not None]
    for _, g in sorted(scored, key=lambda x: (-x[0], x[1]))[:max_n]:
        cands.add(g)
    return sorted(cands)


def decay_weights(n: int, decay: float) -> list:
    """Patch 115 (model v6.12 rulings 3b and 4): weekly weights decay^k, k=0 the nearest week. With decay < 1 the front half of
    a window always carries more total weight than the back half (0.9 over eight weeks: 62% on the front four)."""
    return [round(float(decay) ** k, 6) for k in range(max(0, int(n)))]


def post_chip_gain(score: dict, base: dict, t: int, gws: list, n: int, decay: float, skip_gw: int | None = None):
    """Patch 115 (v6.12 ruling 3a/3b): the Wildcard-at-`t` gain over the next `n` gameweeks (decay-weighted), the SAME `n`
    for every candidate. A candidate with fewer than `n` weeks left in `gws` is NOT scored (None) -- never scored on fewer
    weeks (Rule #48(a)). The Free Hit week (`skip_gw`) is left out of the sum."""
    weeks = [g for g in gws if g >= t][:int(n)]
    if len(weeks) < int(n):
        return None
    w = decay_weights(len(weeks), decay)
    gain = sum(w[i] * (score[g] - base[g]) for i, g in enumerate(weeks) if g != skip_gw)
    return {"gain": round(float(gain), 2), "weeks": weeks}


def decay_sensitivity(cands: dict, base_score: dict, gws: list, n: int, skip_gw, decays, decide_fn):
    """Patch 115 fix 2: how much does the Wildcard pick depend on the estimate-tier decay (0.9)? Re-decides the week at each
    alternative decay from the SAME solved squads (their weekly scores; no re-solve). Gain = decay-weighted post-chip chain
    gain + the candidate's chip gain. `decide_fn(gains)` is the real decision (band, later week, guardrail). A flip means the
    verdict is low confidence; the pick itself is never changed. None when there is nothing to vary."""
    if not cands or not int(n) or not decays:
        return None
    picks, all_gains = {}, {}
    for d in decays:
        g = {}
        for t, c in cands.items():
            pg = post_chip_gain(c["score"], base_score, t, gws, int(n), float(d), skip_gw=skip_gw)
            if pg is None:
                continue
            g[t] = round(pg["gain"] + float(c.get("gain_chips") or 0.0), 2)
        if not g:
            continue
        dec = decide_fn(g)
        picks[d] = (dec or {}).get("gw")
        all_gains[d] = g
    if not picks:
        return None
    flips = len(set(picks.values())) > 1
    shown = ", ".join(f"decay {d}: " + (f"GW{p}" if p is not None else "HOLD") for d, p in picks.items())
    text = (f"Decay sensitivity (0.9 is estimate-tier): {shown}"
            + (" - the week moves, so this verdict is low confidence." if flips else " - same week at every decay tested."))
    return {"picks": picks, "gains": all_gains, "flips": flips, "text": text}


def common_post_weeks(cands: list, gws: list, cap: int = 6, minimum: int = 4):
    """Patch 115 (v6.12 ruling 3a): ONE post-chip window length for every candidate = min(cap, fewest weeks left among the
    scorable candidates), never below `minimum` (Rule #48's four). Candidates with fewer than `minimum` weeks left are
    'not scored (window truncated)'. Returns (n, scored, not_scored)."""
    avail = {t: len([g for g in gws if g >= t]) for t in cands}
    scored = [t for t in cands if avail[t] >= minimum]
    not_scored = [t for t in cands if avail[t] < minimum]
    if not scored:
        return int(minimum), [], list(not_scored)
    return int(min(cap, min(avail[t] for t in scored))), scored, not_scored


def fallback_weeks(weeks: list, planning_gw: int, coverage: int = 5) -> list:
    """Patch 115 (v6.12 ruling 3c): weeks beyond market coverage (Current + `coverage`) are fallback-tier (Rule #46(e))."""
    return [g for g in weeks if g > int(planning_gw) + int(coverage)]


def four_gw_crosscheck(by_gw_gaps: dict | None, chain_gw: int | None, moe_fn=None) -> dict | None:
    """Patch 115 (v6.12 ruling 3): the Rule #48 four-GW window value is always computed and shown beside the chain
    comparison; a disagreement is reported. `by_gw_gaps` = {t: four-GW Wildcard value}. The four-GW pick follows the same
    rule as the chain (band = Rule #34, LATER week inside it)."""
    gaps = {int(k): float(v) for k, v in (by_gw_gaps or {}).items() if v is not None}
    if not gaps:
        return None
    best = max(gaps.values())
    band = float(moe_fn(best)) if moe_fn is not None else 2.0
    tie = sorted(g for g, v in gaps.items() if best - v <= band + 1e-9)
    pick = max(tie)
    agrees = (chain_gw is None) or (chain_gw in tie)
    if chain_gw is None:
        text = f"Four-GW value (Rule #48) puts the Wildcard at GW{pick} (+{gaps[pick]:.1f}; tie set " + ", ".join(f"GW{g}" for g in tie) + ")."
    elif agrees:
        text = (f"Four-GW cross-check (Rule #48) agrees: GW{chain_gw} is inside its tie set (" +
                ", ".join(f"GW{g}" for g in tie) + f"), best GW{pick} +{gaps[pick]:.1f}.")
    else:
        text = (f"The chain comparison says GW{chain_gw} but the four-GW value (Rule #48) puts GW{pick} "
                f"(+{gaps[pick]:.1f}; tie set " + ", ".join(f"GW{g}" for g in tie) + ") -- they disagree.")
    return {"best_gw": pick, "tie_set": tie, "agrees": bool(agrees), "text": text, "band": round(band, 2)}


def wc_shift_label(prev_gw: int | None, new_gw: int | None) -> str:
    """Patch 115 (v6.12 ruling 2): how the recommended Wildcard week moved versus the previous logged run."""
    if new_gw is None:
        return "hold"
    if prev_gw is None:
        return "new"
    return "later" if new_gw > prev_gw else ("earlier" if new_gw < prev_gw else "same")


def apply_chain_guardrail(gains: dict, health: dict | None, base_health: float | None, band: float,
                          health_band: float) -> dict:
    """Patch 115 (v6.12 ruling 6, Rule #52): the squad-health guardrail on the chain comparison. `gains` = {Wildcard GW:
    gain}; only weeks inside Rule #34's `band` of the best gain are alternatives. `health` = {GW: the squad's xPts at the
    checkpoint week once the chip has played}. A week is REJECTED when its checkpoint health is more than `health_band`
    below (a) the healthiest in-band alternative or (b) the no-chip path (`base_health`: a chip that leaves the squad weaker
    than doing nothing). Rejected weeks come back with their gain and reason; if every alternative is rejected the verdict
    is 'hold' (no week). Weeks with no health figure cannot be checked and pass. Returns {'verdict','gw','gain','passed',
    'rejected','changed'}."""
    if not gains:
        return {"verdict": "pass", "gw": None, "gain": None, "passed": [], "rejected": [], "changed": False}
    best = max(gains.values())
    alts = {g: v for g, v in gains.items() if v >= best - float(band) - 1e-9}
    h = {g: float(health[g]) for g in alts if health and health.get(g) is not None}
    healthiest = max(h.values()) if h else None
    healthiest_gw = max((g for g, v in h.items() if v == healthiest)) if h else None
    rejected, passed = [], []
    for g in sorted(alts):
        reason = None
        if g in h:
            if healthiest is not None and h[g] < healthiest - float(health_band) - 1e-9:
                reason = (f"squad health: checkpoint {h[g]:.1f} xPts is {healthiest - h[g]:.1f} below the healthiest in-band "
                          f"alternative (GW{healthiest_gw}, {healthiest:.1f}); band {health_band:.1f}")
            elif base_health is not None and h[g] < float(base_health) - float(health_band) - 1e-9:
                reason = (f"squad health: checkpoint {h[g]:.1f} xPts leaves the squad {float(base_health) - h[g]:.1f} below the "
                          f"no-chip path ({float(base_health):.1f}); band {health_band:.1f}")
        if reason:
            rejected.append({"gw": g, "gain": round(float(alts[g]), 2), "checkpoint": round(h[g], 1), "reason": reason})
        else:
            passed.append(g)
    if not passed:
        return {"verdict": "hold", "gw": None, "gain": None, "passed": [], "rejected": rejected, "changed": True}
    pick = max(passed)
    return {"verdict": "pass", "gw": pick, "gain": round(float(alts[pick]), 2), "passed": passed,
            "rejected": rejected, "changed": bool(rejected) and pick != max(alts)}


def tc_target_from_schedule(pool: pd.DataFrame, tc_sched_gw: int | None, own_codes, owned_top_xpts: float | None = None):
    """Patch 115 (v6.12 ruling 5): the Triple Captain premium reaches the planner ONLY when TC is scheduled in the Rule #49
    harmonized assignment (`tc_sched_gw` None -> no premium). The premium is the named captain's projected points in the
    TC week, counted once; the target is the best pool scorer that week when he beats the best player already owned."""
    if tc_sched_gw is None or pool is None:
        return None
    col = f"xpts_gw{int(tc_sched_gw)}"
    if col not in pool.columns:
        return None
    own = set(own_codes or [])
    if owned_top_xpts is None:
        o = pool[pool["code"].isin(own)]
        owned_top_xpts = float(pd.to_numeric(o[col], errors="coerce").max()) if not o.empty else 0.0
        if owned_top_xpts != owned_top_xpts:        # NaN
            owned_top_xpts = 0.0
    cand = tc_target_candidates(pool, int(tc_sched_gw), own, k=1)
    if cand and cand[0]["xpts"] > float(owned_top_xpts) + 1e-9:
        return {"code": cand[0]["code"], "gw": int(tc_sched_gw), "name": cand[0]["name"], "xpts": cand[0]["xpts"]}
    return None


def wildcard_chain_decision(trigger_active: bool, gains: dict | None, planning_gw: int, margin: float = 2.0,
                             floor: float = 2.0, today_pct: float | None = None, gap_pct: float | None = None,
                             seq_gw: int | None = None, seq_value: float | None = None,
                             window_totals: dict | None = None, moe_fn=None, health: dict | None = None,
                             base_health: float | None = None, health_band: float | None = None,
                             not_scored: list | None = None, fallback: list | None = None,
                             prev_gw: int | None = None, crosscheck: dict | None = None,
                             band_totals: dict | None = None, unavailable_reason: str | None = None,
                             extra_lines: list | None = None, gains4: dict | None = None, waiting_used: float = 0.0,
                             wait_cap_frac: float = 0.5, provisional: str | None = None,
                             checkpoint_gw: int | None = None) -> dict | None:
    """Patch 117 (model ruling amending v6.12 Rules #48(a) / #34 after the Patch 116 screen).

    DECIDING MEASURE: `gains4` = {candidate GW: FOUR-gameweek plain chain gain + chip value}; the tie band (Rule #34, greater
    of 2 xPts or 2% of the compared four-week total, `band_totals`) is built on the same measure it is compared on. `gains`
    (the six-week decay-weighted gain) is shown beside it and does not decide. Without `gains4` the old measure decides.

    TIE RULE: weeks inside the band of the best form the tie set (flagged low confidence). Inside it the LATER week wins only
    if waiting costs no more than HALF the band (cost = the best passing week's value minus the later week's) AND the
    waiting budget holds: `waiting_used` (cost of consecutive earlier deferrals, from the trigger log) + this cost must stay
    within the band. Otherwise the best-value week is the plan. Pending news is never a reason to defer (`provisional` only
    labels the card). A later week that scores equal or higher is simply the best week.

    GUARDRAIL (Rule #52): `health` = checkpoint xPts per candidate at the named `checkpoint_gw`; a week is rejected when it is
    more than the single-week `health_band` below the healthiest in-band week or below the no-chip path (`base_health`). Every
    in-band week rejected -> HOLD (`hold_source` 'guardrail'); best gain below `floor` -> HOLD (`hold_source` 'value'). A passing
    week outside the tie set is context only, never the plan.

    Earlier history (Patch 115 fix): `unavailable_reason` says why no chain comparison exists; a later-week pick carries
    `wait_cost`. A later shift against `prev_gw` is a Rule #11 reversal (`shift`, `shift_text`). No gains -> 'unverified'."""
    if not trigger_active:
        return None
    today_txt = f"today {today_pct:.0f}%" if today_pct is not None else None
    gap_txt = f"{gap_pct:.1f}%" if gap_pct is not None else "n/a"
    prov = f", provisional ({provisional})" if provisional else ""

    def _pack(kind, gw, badge, cls, card_cls, stat, sub, pill, note=None, note_cls="", gains_=None, best=None, **extra):
        d = {"kind": kind, "gw": gw, "badge": badge, "badge_cls": cls, "card_cls": card_cls, "stat": stat,
             "sub": sub, "pill": pill, "note": note, "note_cls": note_cls,
             "gains": dict(gains_ or {}), "best_gain": best, "band": None, "tie_set": [], "shift": wc_shift_label(prev_gw, gw),
             "shift_text": "", "guardrail": None, "not_scored": list(not_scored or []), "fallback_weeks": list(fallback or []),
             "crosscheck": crosscheck, "detail_lines": [], "wait_cost": None,
             "gains4": dict(gains4 or {}), "hold_source": None, "low_confidence": False, "deferred": False,
             "budget_after": None, "waiting_used": round(float(waiting_used or 0.0), 2)}
        d.update(extra)
        return d

    def _lines(band, tie, guard, gw, G=None, extra_=None):
        L = []
        if band is not None:
            L.append(f"Tie band (Rule #34) {band:.1f} xPts"
                     + (f" - tie set " + ", ".join(f"GW{g}" for g in tie) + "; low confidence" if len(tie) > 1 else ""))
        if gains4 and gains:
            L.append("Deciding measure (four-week plain gain + chip value): "
                     + ", ".join(f"GW{g} {v:+.1f}" for g, v in sorted(gains4.items() if gains4 else []))
                     + " | six-week decay-weighted, shown beside it: "
                     + ", ".join(f"GW{g} {v:+.1f}" for g, v in sorted(gains.items())))
        if health is not None and checkpoint_gw is not None:
            L.append(f"Squad-health checkpoint: GW{checkpoint_gw} (single-week band {float(health_band if health_band is not None else margin):.1f})")
        if guard and guard.get("rejected"):
            for r_ in guard["rejected"]:
                L.append(f"Rejected by the squad-health guardrail (Rule #52): GW{r_['gw']} ({r_['gain']:+.1f} xPts) - {r_['reason']}")
        if float(waiting_used or 0.0) > 0:
            L.append(f"Waiting budget used by earlier deferrals: {float(waiting_used):.1f} xPts"
                     + (f" of the {band:.1f} band" if band is not None else ""))
        for x_ in (extra_ or []):
            L.append(x_)
        if not_scored:
            L.append("Not scored (window truncated): " + ", ".join(f"GW{g}" for g in not_scored))
        if fallback:
            L.append("Rests on fallback-tier weeks (beyond market coverage): " + ", ".join(f"GW{g}" for g in fallback))
        if crosscheck and crosscheck.get("text"):
            L.append(crosscheck["text"])
        L.append("cap_use_bar 0.25 and value tail 3 are estimate-tier (unvalidated) and can move this verdict")
        for x_ in (extra_lines or []):
            L.append(x_)
        return L

    if not (gains or gains4):
        why = f" — {unavailable_reason}" if unavailable_reason else ""
        sub = " · ".join(x for x in [today_txt, "chain comparison unavailable"] if x)
        why_lines = ([f"Chain comparison unavailable: {unavailable_reason}"] if unavailable_reason else []) + list(extra_lines or [])
        if seq_gw is not None:
            return _pack("unverified", seq_gw, f"SEQUENCE GW{seq_gw}", "active", "is-active",
                         f"{seq_value:+.1f} xPts" if seq_value is not None else f"{gap_txt} gap", sub,
                         f"Wildcard: sequence says GW{seq_gw} (chain comparison unavailable{why})",
                         detail_lines=why_lines)
        return _pack("unverified", None, "TRIGGER ACTIVE", "active", "is-active", f"{gap_txt} gap", sub,
                     f"Wildcard: trigger active (chain comparison unavailable{why})", detail_lines=why_lines)
    G = {int(k): float(v) for k, v in (gains4 or {}).items() if v is not None} or {int(k): float(v) for k, v in gains.items()}
    show = dict(gains or G)
    best = max(G.values())
    best_gw0 = max(g for g, v in G.items() if v == best)
    if best < floor:
        sub = " · ".join(x for x in [today_txt, f"adds only {best:+.1f} xPts over the window (below the value floor {floor:.1f})"] if x)
        return _pack("hold", None, "HOLD", "hold", "", f"{best:+.1f} xPts", sub,
                     f"Wildcard: HOLD — adds only {best:+.1f} xPts, below the value floor", None, "", show, best,
                     detail_lines=_lines(None, [], None, None), hold_source="value")
    ref_total = (band_totals or {}).get(best_gw0)
    if ref_total is None:
        ref_total = (window_totals or {}).get(best_gw0)
    band = float(moe_fn(ref_total)) if (moe_fn is not None and ref_total is not None) else float(margin)
    tie = sorted(g for g, v in G.items() if v >= best - band - 1e-9)
    cap = band * float(wait_cap_frac)
    guard = None
    if health is not None:
        guard = apply_chain_guardrail({g: G[g] for g in tie}, health, base_health, band,
                                      float(health_band if health_band is not None else margin))
        if guard["verdict"] == "hold":
            sub = " · ".join(x for x in [today_txt, "every in-band week failed the squad-health guardrail"] if x)
            return _pack("hold", None, "HOLD", "hold", "", f"{best:+.1f} xPts", sub,
                         "Wildcard: HOLD — the squad-health guardrail (Rule #52) rejected every in-band week"
                         + (f" at the GW{checkpoint_gw} checkpoint" if checkpoint_gw is not None else ""), None, "",
                         show, best, band=band, tie_set=tie, guardrail=guard, hold_source="guardrail",
                         low_confidence=len(tie) > 1, detail_lines=_lines(band, tie, guard, None))
        passed = list(guard["passed"])
    else:
        passed = list(tie)
    used = max(0.0, float(waiting_used or 0.0))
    ref_val = max(G[g] for g in passed)
    ref_gw = max(g for g in passed if G[g] == ref_val)

    def cost(w):
        return round(ref_val - G[w], 2)

    allowed = [w for w in passed if cost(w) <= 1e-9 or (cost(w) <= cap + 1e-9 and used + cost(w) <= band + 1e-9)]
    best_gw = max(allowed)
    gain = G[best_gw]
    refused = sorted(w for w in passed if w > best_gw)
    shift = wc_shift_label(prev_gw, best_gw)
    shift_text = ""
    if shift == "later" and prev_gw is not None:
        shift_text = (f"Rule #11 reversal: the recommended Wildcard week moved LATER (GW{prev_gw} → GW{best_gw}) since the "
                      f"previous run; logged in the trigger log.")
    c_pick = cost(best_gw)
    deferred = best_gw > planning_gw
    wait = {"best_gw": ref_gw, "pick_gw": best_gw, "cost": c_pick} if (best_gw != ref_gw and c_pick > 0) else None
    ex_lines = []
    why_txt = None
    if refused:
        L = min(refused)
        c_l = cost(L)
        if c_l > cap + 1e-9:
            why_txt = (f"GW{best_gw} and GW{L} are inside the tie band ({band:.1f}). Waiting costs {c_l:.1f} xPts, more than "
                       f"half the band, so the earlier week is kept. Rule #48(a).")
        else:
            why_txt = (f"GW{best_gw} and GW{L} are inside the tie band ({band:.1f}). Waiting costs {c_l:.1f} xPts, but the "
                       f"waiting budget ({used:.1f} already used + {c_l:.1f} > band {band:.1f}) is spent, so the earlier week is kept. Rule #48(a).")
    elif wait:
        why_txt = f"Inside the tie band; waiting costs {c_pick:.1f} xPts, within half the band."
    if wait:
        ex_lines.append(f"Waiting from GW{ref_gw} to GW{best_gw} costs about {c_pick:.1f} xPts against the best week "
                        f"(cap {cap:.1f} = half the band); the date is your call (Rule #53(d))")
    extra = dict(band=band, tie_set=tie, guardrail=guard, shift=shift, shift_text=shift_text, wait_cost=wait,
                 low_confidence=len(tie) > 1, deferred=bool(deferred), gains4=dict(gains4 or {}),
                 budget_after=round(used + (c_pick if deferred else 0.0), 2),
                 detail_lines=_lines(band, tie, guard, best_gw, extra_=ex_lines))
    six = show.get(best_gw)
    six_txt = f"; six-week {six:+.1f}" if (gains4 and six is not None) else ""
    if best_gw == planning_gw:
        sub = " · ".join(x for x in [today_txt, f"Plan GW{best_gw}{prov}." + (f" {why_txt}" if why_txt else " Best week in the window.")] if x)
        return _pack("play_now", best_gw, "PLAY NOW", "play", "is-play", f"{gain:+.1f} xPts", sub,
                     f"Wildcard: PLAY NOW — {gain:+.1f} xPts (four-week{six_txt})" + (" · provisional" if provisional else ""),
                     "Best week is this one", "warn", show, gain, **extra)
    now = G.get(planning_gw)
    if wait:
        verdict_txt = why_txt
    elif now is not None and now < gain - 1e-9:
        verdict_txt = f"waiting beats playing now ({gain:+.1f} vs {now:+.1f} now)"
    else:
        verdict_txt = "the later week scores at least as high as now"
    sub = " · ".join(x for x in [today_txt, f"Plan GW{best_gw}{prov}. {verdict_txt}"] if x)
    return _pack("hold_until", best_gw, f"HOLD → GW{best_gw}", "hold", "", f"{gain:+.1f} xPts", sub,
                 f"Wildcard: HOLD → GW{best_gw} — {gain:+.1f} xPts (four-week{six_txt})",
                 "Free transfers carry the team", "good", show, gain, **extra)


def waiting_budget_used(path: str, team_id, planning_gw: int) -> float:
    """Patch 117 (model ruling, waiting budget): the total cost of the CONSECUTIVE Wildcard deferrals logged before
    `planning_gw` for this team (trigger-log rows with fired and wc_deferred = 1; the last row of each gameweek counts; the
    chain stops at the first gameweek that was not a deferral or has no row). 0.0 when there is no usable log. NOTE: on
    Streamlit Cloud the log lives on an ephemeral disk, so a reboot/redeploy resets this to 0 (the budget then restarts)."""
    import csv as _csv
    import os as _os
    try:
        if not _os.path.exists(path):
            return 0.0
        last = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                if str(r.get("team_id")) != str(team_id):
                    continue
                if str(r.get("fired", "")).strip().lower() not in ("true", "1", "1.0"):
                    continue
                try:
                    g = int(float(r.get("planning_gw")))
                except Exception:
                    continue
                last[g] = r
        total, g = 0.0, int(planning_gw) - 1
        while g in last:
            r = last[g]
            try:
                d = int(float(r.get("wc_deferred") or 0))
                c = float(r.get("wc_wait_cost") or 0.0)
            except Exception:
                break
            if d != 1:
                break
            total += c
            g -= 1
        return round(total, 2)
    except Exception:
        return 0.0


def plan_timeline_lines(weekly_plan: list | None, first_week: tuple | None = None) -> list[str]:
    """Patch 111/112: GW-by-GW lines from a chained weekly_plan -- "GW7: Roll (4 FT banked)",
    "GW8: Palmer → Saka, Isak → Haaland (2 FT left)", "GW9: WILDCARD (11 changes)". A move forced by the
    5-transfer cap says so. `first_week` = (gw, [move texts][, ft_after]) is the manager's own recommended
    move for the current GW, shown first (the chain starts after it)."""
    lines: list[str] = []
    if first_week is not None:
        gw0, moves0 = first_week[0], first_week[1]
        ft0 = first_week[2] if len(first_week) > 2 else None
        txt = (", ".join(moves0) if moves0 else "Roll")
        lines.append(f"GW{gw0}: {txt}" + (f" ({ft0} FT left)" if ft0 is not None and moves0 else
                                          (f" ({ft0} FT banked)" if ft0 is not None else "")))
    for wk in weekly_plan or []:
        gw = wk.get("gw")
        moves = wk.get("moves") or []
        ft = wk.get("ft_banked_after")
        if wk.get("chip_played") == "wildcard":
            lines.append(f"GW{gw}: WILDCARD ({len(moves)} changes)")
        elif moves:
            tail = f" ({ft} FT left)" if ft is not None else ""
            if wk.get("cap_forced"):
                tail += " · uses banked FT at the 5-FT cap"
            lines.append(f"GW{gw}: " + ", ".join(f"{m.get('out', '?')} → {m.get('in', '?')}" for m in moves) + tail)
        else:
            lines.append(f"GW{gw}: Roll" + (f" ({ft} FT banked)" if ft is not None else ""))
    return lines


_CAL_COLUMNS = ["logged_at", "team_id", "team_name", "planning_gw", "style", "gap_pct", "rating_pct",
                "cumulative_gap", "fired", "verdict", "wc_value_xpts", "wc_gw", "transfers_close_gap", "patch",
                "wc_shift", "wc_band", "wc_wait_cost", "wc_deferred"]


def calibration_log_row(team_id, team_name, planning_gw, patch, style, gap_pct, rating_pct, cumulative_gap,
                         active, verdict, wc_value, wc_gw, closes, shift=None, band=None, wait_cost=None, deferred=None) -> dict:
    """Patch 108: one row of the Wildcard calibration log -- the data needed
    to validate the (document-flagged "unvalidated") 6% threshold from the
    manager's own weeks."""
    import datetime as _dt
    return {"logged_at": _dt.datetime.now().replace(microsecond=0).isoformat(), "team_id": team_id,
            "team_name": team_name, "planning_gw": planning_gw, "style": style, "gap_pct": gap_pct,
            "rating_pct": rating_pct, "cumulative_gap": cumulative_gap, "fired": bool(active),
            "verdict": verdict, "wc_value_xpts": wc_value, "wc_gw": wc_gw,
            "transfers_close_gap": closes, "patch": patch, "wc_shift": shift, "wc_band": band,
            "wc_wait_cost": wait_cost, "wc_deferred": (None if deferred is None else int(bool(deferred)))}


def last_logged_wc_gw(path: str, team_id) -> int | None:
    """Patch 115 (v6.12 ruling 2): the Wildcard week the previous logged run recommended for this team (None if none)."""
    import csv as _csv
    import os as _os
    try:
        if not _os.path.exists(path):
            return None
        last = None
        with open(path, newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                if str(r.get("team_id")) == str(team_id) and str(r.get("wc_gw", "")).strip() not in ("", "None", "nan"):
                    last = int(float(r["wc_gw"]))
        return last
    except Exception:
        return None


def append_calibration_row(path: str, row: dict) -> bool:
    """Append `row` to the CSV at `path` (header written on first use).
    Skips a repeat of the SAME run (same team, GW, style, gap, verdict and
    value as the most recent row for that team/GW) so reruns don't spam the
    log. Never raises -- logging must never take the page down. Returns True
    only if a row was written."""
    import csv as _csv
    import os as _os
    try:
        key = ("team_id", "planning_gw", "style", "gap_pct", "verdict", "wc_value_xpts")
        if _os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                for existing in _csv.DictReader(fh):
                    if all(str(existing.get(k)) == str(row.get(k)) for k in key):
                        return False
        new_file = not _os.path.exists(path)
        if not new_file:
            # Patch 115: an older log (fewer columns) is migrated in place so the new columns line up
            with open(path, newline="", encoding="utf-8") as fh:
                rd = _csv.DictReader(fh)
                header = list(rd.fieldnames or [])
                old_rows = list(rd) if header != _CAL_COLUMNS else []
            if header != _CAL_COLUMNS:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w0 = _csv.DictWriter(fh, fieldnames=_CAL_COLUMNS)
                    w0.writeheader()
                    for r0 in old_rows:
                        w0.writerow({c: r0.get(c) for c in _CAL_COLUMNS})
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=_CAL_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow({c: row.get(c) for c in _CAL_COLUMNS})
        return True
    except Exception:
        return False


def extended_window_target(planning_gw: int, span: int = 10) -> int:
    """Patch 107 (2026-10-04): the extended check always covers
    Current..Current+(span-1) -- "Current GW + 9" at the default span of 10."""
    return planning_gw + span - 1


def extended_proj_gws(planning_gw: int, span: int = 10, window_len: int = 4) -> list[int]:
    """GWs that must be projected for extended mode: the span itself plus
    (window_len - 1) trailing GWs, so a Wildcard candidate at the LAST GW of
    the span is still scored over a full window_len-GW window rather than a
    truncated one (a truncated window would systematically understate late
    candidates' window value and bias the "best GW" verdict toward early GWs)."""
    return list(range(planning_gw, planning_gw + span + max(0, window_len - 1)))


def extended_scan_full_gw_list(candidate_gws: list[int], window_len: int = 4) -> list[int]:
    """The full_gw_list wildcard_window_value_scan should be handed in
    extended mode: the candidate GWs plus trailing GWs for the last
    candidates' windows. Empty in, empty out."""
    if not candidate_gws:
        return []
    last = max(candidate_gws)
    return list(range(min(candidate_gws), last + max(0, window_len - 1) + 1))


def extended_button_label(planning_gw: int, target_gw: int, span: int = 10) -> str:
    """Patch 107: a button name that is dynamic, never a hard-coded GW --
    "Current GW + 9 (GW6 -> GW15)"."""
    return f"Extended Check GW{target_gw}"


_CHIP_KEY_LABELS = {"wildcard": "Wildcard", "bboost": "Bench Boost", "3xc": "Triple Captain",
                    "freehit": "Free Hit"}


def chip_change_tags(baseline: dict | None, now: dict) -> dict:
    """Patch 107: compares each chip's scheduled GW from the user's previous
    NORMAL run (`baseline`, {chip_key: gw|None}) against this extended run
    (`now`) and returns {chip_key: tag_text} for ONLY the chips that moved.
    No baseline (extended clicked before any normal run) -> {} -- never
    invents a "was" value. Chips absent from the baseline are ignored."""
    if not baseline:
        return {}
    tags = {}
    for key, new_gw in (now or {}).items():
        if key not in baseline:
            continue
        old_gw = baseline[key]
        if old_gw == new_gw:
            continue
        if old_gw is not None and new_gw is not None:
            tags[key] = f"updated by extended check (was GW{old_gw})"
        elif old_gw is not None and new_gw is None:
            tags[key] = f"updated by extended check (was GW{old_gw}; now no slot in the 10-GW sequence)"
        else:
            tags[key] = "updated by extended check (was not scheduled)"
    return tags


def build_best_gw_table(planning_gw: int, gws: list[int], wc_by_gw: dict | None,
                         rating_by_gw: dict, chips_by_gw: dict) -> dict:
    """Patch 107: the "best GW between Current and Current+9" answer.
    rows: one per GW with the Wildcard window-value gap (xPts, from the
    Rule #48 scan), the cross-check rating % (if computed for that GW) and
    whichever chips the joint sequence placed there. verdict: names the best
    Wildcard GW and compares it with playing it at the current GW."""
    wc_by_gw = wc_by_gw or {}
    rows = []
    for g in gws:
        gap = wc_by_gw.get(g, {}).get("gap") if isinstance(wc_by_gw.get(g), dict) else None
        rows.append({"gw": g, "wc_gap": gap, "rating_pct": rating_by_gw.get(g),
                     "chips": ", ".join(chips_by_gw.get(g, []))})
    scored = {g: v["gap"] for g, v in wc_by_gw.items() if isinstance(v, dict) and v.get("gap") is not None}
    if not scored:
        return {"rows": rows, "best_wc_gw": None,
                "verdict": "No Wildcard window-value scan available this run, so no best Wildcard GW can be named."}
    best = max(scored, key=lambda g: scored[g])
    if best == planning_gw:
        verdict = (f"Best Wildcard GW in the span: GW{best} (the current GW, {scored[best]:+.1f} xPts window "
                   f"value) — no later GW beats playing it now.")
    else:
        now_val = scored.get(planning_gw)
        vs = f" vs {now_val:+.1f} at GW{planning_gw}" if now_val is not None else ""
        verdict = f"Best Wildcard GW in the span: GW{best} ({scored[best]:+.1f} xPts window value{vs})."
    return {"rows": rows, "best_wc_gw": best, "verdict": verdict}


def extend_reach_status(target_gw: int, reached_gw: int) -> tuple[bool, str]:
    """Patch 106 (2026-10-04): compares the GW the extend button promised
    against the GW the cross-check actually reached, so a shortfall is never
    reported quietly as success. Returns (ok, warning_message)."""
    if reached_gw >= target_gw:
        return True, ""
    return False, (f"Asked to reach GW{target_gw} but only reached GW{reached_gw} — GW{reached_gw + 1}-"
                   f"GW{target_gw} had no projection data, so treat this as a partial check.")


def build_wc_extend_note(chip_driven: bool, reached_target: bool, check_gws_last: int,
                          auto_wildcard_gw: int | None) -> str:
    """Patch 105 (2026-10-01, manager screenshot: "already clicked but i
    don't have a confirmation!!!!" on the Patch 104 always-on extend
    button). While fixing the missing-confirmation gap, a second, more
    serious bug was found in code: this note's text was still hard-coded to
    claim the extension ran "to reach your scheduled Wildcard at GW{x}" —
    true for a chip-driven extension, but actively WRONG for Patch 104's new
    exploratory case (no scheduled chip beyond normal reach at all, or one
    already within it) — the manager could click "Exploratory" and get back
    a result that falsely describes itself as chip-driven.

    `chip_driven` is the same recommend.wc_extend... decision app.py's
    `_wc_extend_chip_driven` flag already computes (auto_wildcard_gw is not
    None and beyond the normal reach) -- computed ONCE in app.py and passed
    in here, never recomputed a second time (the same two-sources-of-truth
    class of bug already fixed once, in Patch 103, for `_wc_current_max_gw`
    — not repeating it here).
    """
    if chip_driven:
        note = (f" (auto-extended to GW{check_gws_last} to reach your scheduled Wildcard at "
                 f"GW{auto_wildcard_gw}, beyond your current Horizon/detection window — this extension is "
                 f"only for this cross-check, your Transfer Recommendations and Wildcard trigger % above are "
                 f"unaffected)")
        if not reached_target:
            note += (f" — capped at +8 GWs and did NOT reach GW{auto_wildcard_gw} yet; treat this as a "
                      f"partial check, not the full picture.")
        return note
    return (f" (manually extended to GW{check_gws_last} at your request — exploratory, no scheduled chip "
            f"currently sits beyond your normal Horizon/detection window; this extension is only for this "
            f"cross-check, your Transfer Recommendations and Wildcard trigger % above are unaffected)")


def resolve_wildcard_check_gw_window(transfer_gw_list: list[int], detect_gw_list: list[int] | None) -> list[int]:
    """Patch 102 (2026-10-01, manager report: Chip Plan tab showed "Wildcard
    trigger ACTIVE (93.6%)" right next to "Wildcard may not be needed --
    plan reaches 96.6%" on a week Transfer Recommendations said "Roll" (no
    transfer made) -- looked contradictory. Traced in code: the 93.6%
    (fpl_engine.wildcard_trigger_check()) is averaged over `detect_gw_list`
    (chip_shape_test.detection_window_gws, 4 GWs). The "may not be needed"
    line's own average was computed over a DIFFERENT, shorter window --
    `transfer_gw_list`, which at Horizon=1 is just ONE GW -- whenever the
    Patch 101 extended cross-check isn't active (the default since Patch 101
    made that extension opt-in). With zero transfers made, the 93.6-vs-96.6
    gap was fixture-variance noise from comparing a 4-GW average against a
    1-GW snapshot, not evidence a transfer had closed anything.

    This resolver picks whichever of the two windows is LONGER -- never
    shorter than the trigger's own `detect_gw_list` window, so the two
    numbers are always computed over at least the same span and stop being
    spuriously comparable-looking while actually measuring different
    things. A tie keeps `transfer_gw_list` (the plan's own real window,
    nothing to gain by switching). Costs nothing extra to compute: the
    caller's `reachable_by_gw` is already solved over every GW in
    `detect_gw_list` (that's what feeds the trigger itself), so widening to
    it needs no new solve, cached or otherwise."""
    if not detect_gw_list:
        return transfer_gw_list
    if not transfer_gw_list:
        return list(detect_gw_list)
    return list(detect_gw_list) if len(detect_gw_list) > len(transfer_gw_list) else transfer_gw_list


def build_squad_after_by_gw(squad_df: pd.DataFrame, weekly_plan: list[dict],
                             pool_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Patch 94 (v6.9 Rule #49 cross-check correctness fix — discussion,
    2026-09-30: before building the requested "cross-check UI" extension,
    checked the codebase and found app.py's existing Wildcard cross-check
    note (Patch 46-49) reconstructed "the squad after your plan" by
    flattening EVERY week's moves from `weekly_plan` into one list and
    applying them all in a single pd.concat, regardless of which week each
    move actually belonged to — already misleading for any multi-week plan,
    and actively wrong once a scheduled Wildcard week injects a ~13-player
    wholesale rebuild into that same flat list (Patch 92).

    This is the single correct reconstruction, extracted from what was
    previously a private closure nested inside app.py's
    `_render_pitch_navigator()` (Patch 84's `_apply_moves` + its per-week
    loop) so both that navigator AND the Wildcard cross-check now share one
    source of truth instead of two, the same "stop having two disconnected
    reconstruction methods" fix pattern as Patch 91/92/93.

    Chains each week's moves onto the LITERAL resulting squad of the week
    before it, in `weekly_plan` order — exactly how
    `plan_transfer_schedule()` built `sim_squad` internally, so replaying
    the same moves here in the same order reproduces that exact chain. A
    week with no moves (a "Roll" week, or a week whose moves are malformed —
    missing `out_code`/`in_code`) simply carries the prior week's squad
    forward unchanged. A Wildcard week's wholesale-rebuild moves (however
    many) are applied as one atomic swap against the PRIOR week's squad, not
    blended with any other week's moves.

    Returns `{gw: squad_df_after_that_week}` for every week in
    `weekly_plan`, in order. Empty `weekly_plan` returns `{}`. Incoming
    players are pulled from `pool_df` — pass whichever full projection pool
    is appropriate for the caller's context (e.g. the pitch navigator's
    window-scoped pool, or the main run's full `proj`)."""
    result: dict[int, pd.DataFrame] = {}
    running = squad_df
    for wk in weekly_plan or []:
        moves = wk.get("moves") or []
        if moves:
            moves_df = pd.DataFrame(moves)
            if "out_code" in moves_df.columns and "in_code" in moves_df.columns:
                out_codes = set(moves_df["out_code"])
                in_codes = set(moves_df["in_code"])
                running = pd.concat([
                    running[~running["code"].isin(out_codes)],
                    pool_df[pool_df["code"].isin(in_codes)],
                ], ignore_index=True, sort=False)
                if "code" in running.columns:
                    running = running.drop_duplicates(subset=["code"], keep="first")
        result[wk.get("gw")] = running
    return result


def resolve_chip_capped_gw_list(gw_list: list[int], manual_planned_chip_gw: int | None = None,
                                 auto_wildcard_gw: int | None = None) -> list[int] | None:
    """Patch 93 (v6.9 Rule #49 follow-up — closing the same "two disconnected
    sources of truth" gap Patch 91/92 fixed elsewhere, this time on
    `suggest_transfers()`'s single-decision path, used at horizon=1 or
    "Force"): resolves the GW that should truncate the "Chip-aware alt"
    horizon check (see the `chip_capped_gw_list` param on `suggest_transfers()`
    below), preferring an explicit manual planned-chip GW (the sidebar's
    "Next planned full-rebuild chip GW" dropdown — a manager override that
    may reflect information the auto-scheduler doesn't have) over the
    auto-detected Wildcard GW from the Rule #49 joint scheduler
    (`chip_portfolio["assignment"]["wildcard"]`), and falling back to the
    auto value only when the manual one is unset (None — never a falsy-but-
    real GW like 0, checked with `is not None` throughout, not truthiness).

    Returns None when neither is set, or when the resolved chip GW falls
    after the full horizon (`gw_list[-1]`) — same edge-case contract the old
    manual-only inline logic in app.py already had. Can return an EMPTY list
    (not None) when the chip GW lands at or before `gw_list[0]` — callers
    must keep testing `is not None` to distinguish "no cap at all" from "cap
    to nothing", exactly as `suggest_transfers()` already does internally."""
    chip_gw = manual_planned_chip_gw if manual_planned_chip_gw is not None else auto_wildcard_gw
    if chip_gw is None or not gw_list or chip_gw > gw_list[-1]:
        return None
    return [g for g in gw_list if g < chip_gw]


def resolve_bb_play_gw(isolated_bb_play_gw: int | None, harmonized_bb_gw: int | None) -> int | None:
    """Patch 98 (v6.9 Rule #52 follow-up, "a more wide rule" — manager,
    2026-09-30): closes the same "two disconnected sources of truth" gap
    Patch 91/92/93 already fixed elsewhere, this time on the transfer
    planner's Bench Boost timing. Confirmed via code read of app.py's
    pre-Patch-98 `_bb_play_gw` (`int(bb_advisor["verdict"].split("gw")[1])`):
    it was parsed ONLY from the ISOLATED Bench Boost advisor's own
    individually-best week, never from `chip_portfolio`'s harmonized joint
    schedule (`chip_portfolio["assignment"]["bboost"]`) — the number Rule
    #49/#50's joint scheduler can, by design since Patch 96, place on a
    DIFFERENT week than the isolated advisor's own optimum, whenever doing
    so raises the combined 4-chip total. Without this, the Chip Plan tab
    could show one Bench Boost week while the transfer planner silently
    valued bench strength for a different one.

    Prefers `harmonized_bb_gw` (the joint scheduler's pick — more informed,
    since it knows about all 4 chips together) whenever it's set, falling
    back to `isolated_bb_play_gw` (the standalone advisor's own verdict)
    only when no joint schedule exists. Checked with `is not None`
    throughout, never truthiness — a real GW of 0 must not be treated as
    unset, same discipline `resolve_chip_capped_gw_list()` already
    documents for this exact class of bug."""
    return harmonized_bb_gw if harmonized_bb_gw is not None else isolated_bb_play_gw


_PLAN_SOLVE_MEMO: dict = {}
_PLAN_SOLVE_MEMO_MAX = 600


def _frame_fingerprint(df: pd.DataFrame):
    """Patch 112 (performance, lossless): a content hash of the WHOLE frame (every column, so it also covers
    the columns that only flow into the returned squad rows). None when a column holds unhashable cells --
    callers then skip the memo and behave exactly as before."""
    try:
        import hashlib
        h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy().tobytes()
        return (len(df), tuple(map(str, df.columns)), hashlib.blake2b(h, digest_size=16).hexdigest())
    except Exception:
        return None


def _solve_retain_memo(full_pool, pool_fp, cfg, cfg_fp, team_value, current_codes, min_retain, locks=()):
    """Patch 112: opt.solve_squad(...) for the planner's k-loop, memoised on (pool fingerprint, owned codes,
    min_retain, budget). The solve is deterministic in those inputs (objective column fixed), and
    Streamlit's own cache re-hashed the entire player table on every call. Falls back to a plain call."""
    if pool_fp is None:
        return opt.solve_squad(full_pool, cfg, budget=team_value, retain_pool_codes=current_codes,
                               min_retain=min_retain, must_include_codes=list(locks), objective_col="xpts_horizon_sum")
    key = (pool_fp, cfg_fp, tuple(sorted(current_codes)), int(min_retain), round(float(team_value), 3),
           tuple(sorted(locks)))
    if key in _PLAN_SOLVE_MEMO:
        res = _PLAN_SOLVE_MEMO[key]
    else:
        res = opt.solve_squad(full_pool, cfg, budget=team_value, retain_pool_codes=current_codes,
                              min_retain=min_retain, must_include_codes=list(locks), objective_col="xpts_horizon_sum")
        if len(_PLAN_SOLVE_MEMO) >= _PLAN_SOLVE_MEMO_MAX:
            _PLAN_SOLVE_MEMO.clear()
        _PLAN_SOLVE_MEMO[key] = res
    if res is None:
        return None
    out = dict(res)
    if isinstance(out.get("squad"), pd.DataFrame):
        out["squad"] = out["squad"].copy()
    return out


def plan_transfer_schedule(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                            profile_name: str, free_transfers: int, bank: float,
                            current_gw: int, gw_list: list[int],
                            meaningful_bar: float | None = None,
                            chip_advisory: str | None = None,
                            bb_play_gw: int | None = None,
                            disrupted_codes: set | None = None,
                            hit_stance: str = "No hits",
                            chip_schedule: dict | None = None,
                            use_tie_break: bool = True,
                            cap_use_bar: float | None = None,
                            value_tail: list | None = None,
                            tc_target: dict | None = None) -> dict:
    """Patch 114: `tc_target` = {"code": int, "gw": int} -- the Triple Captain player and week the Chip Plan picked. When
    given (and the week is inside the planned/valued weeks), the squad's top XI scorer in that week counts one extra time
    (the tripled captain's extra x1), and the target's horizon sum gets the same bonus so the planner can BUY him with a
    free transfer when that beats rolling. None (default) = unchanged behaviour. DEVIATION from Standing Rule #31 (same
    switchable chip-value deviation as Patch 112/113).

    No-hits, multi-GW pacing plan (project discussion, 2026-09-07) — see
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

    Patch 92 (v6.9 Rules #44/#48/#49 read together, chip-aware weekly
    transfer plan — manager, 2026-09-30: "let's go" on the deeper
    Wildcard-aware fix over the simpler cross-check-note UI, after catching
    that the planner had zero Wildcard awareness): optional
    `chip_schedule: {"wildcard_gw": int|None, "wildcard_rebuild_squad":
    pd.DataFrame|None}`, deliberately Wildcard-ONLY in scope — Free Hit
    reverts after one week (Rule #44/#49c) so it never changes the
    persisted squad path this chained plan tracks, and Bench Boost/Triple
    Captain never change squad composition at all. When given:
      (a) every week strictly BEFORE `wildcard_gw` values its transfer
          decision against a horizon TRUNCATED at `wildcard_gw` (exclusive)
          — a candidate whose value only exists at/after the Wildcard week
          is never counted, mirroring `suggest_transfers()`'s own
          `chip_capped_gw_list` concept but automatic and inside the chain;
      (b) AT `wildcard_gw`, the ordinary k=1..k_upper search is skipped
          entirely and `sim_squad` is replaced wholesale by
          `wildcard_rebuild_squad` — this draws NO free transfer (Rule #44:
          a chip's transfers are free and never touch the real FT balance)
          while still accruing its own +1 FT (capped at
          `transfers.MAX_BANK`) exactly as an unused week would;
      (c) every week AFTER `wildcard_gw` continues the normal chained
          k-search, but starting from the rebuilt squad.
    Omitting `chip_schedule` (None, the default) reproduces exact
    pre-Patch-92 behavior — verified by the full existing test suite
    passing unmodified.

    Patch 98 (v6.9 Rule #52 follow-up, "a more wide rule" — manager,
    2026-09-30, after confirming Patch 96's harmonized chip sequence was
    live and correct on team 26073, then asking what the Transfers side
    does "if we will use the chip" and explicitly generalizing the ask):
    `chip_schedule` gains an optional `"freehit_gw": int|None` key. Free
    Hit has the identical "illusory value" problem Wildcard already had a
    fix for (a) above, and had NO fix at all before this patch — a
    transfer's valuation for any week could be credited with points that
    only exist at the scheduled Free Hit week, but that week is actually
    played by an entirely different, temporary rebuild squad, so that value
    is never actually realized by the persisted squad this plan tracks.
    Unlike Wildcard, Free Hit reverts after one week, so this is an
    EXCLUSION of that one GW from `remaining_gws` wherever it's used, never
    a truncation of everything from it onward — a candidate whose value
    lands at a week AFTER the Free Hit week is completely unaffected.
    Omitting the key (or omitting `chip_schedule` entirely) reproduces
    exact pre-Patch-98 behavior.

    Returns the same top-level keys `suggest_transfers()` returns (so
    existing callers/UI code work unchanged), plus `weekly_plan`: a list of
    one dict per GW — {gw, moves, net_gain, ft_available, ft_used,
    ft_banked_after, summary, data_gap_note, chip_played (Patch 92; absent
    or None for a normal week, "wildcard" for the substitution week)} —
    and `is_weekly_schedule`: True, so a caller can distinguish this shape
    from the single-decision return if it wants to render it differently."""
    wildcard_gw = chip_schedule.get("wildcard_gw") if chip_schedule else None
    wildcard_rebuild_squad = chip_schedule.get("wildcard_rebuild_squad") if chip_schedule else None
    # Patch 98 -- see the docstring's "Patch 98" paragraph below; None (the
    # default, and the value when `chip_schedule` omits this key entirely)
    # reproduces exact pre-Patch-98 behavior.
    freehit_gw = chip_schedule.get("freehit_gw") if chip_schedule else None
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
    value_tail = [g for g in (value_tail or []) if g not in gw_list]
    numeric_cols = {"price", "xpts_horizon_sum"} | {f"xpts_gw{g}" for g in list(gw_list) + value_tail}
    for df in (squad_df, pool_df):
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    bank = 0.0 if bank is None or pd.isna(bank) else float(bank)

    full_pool = pd.concat([squad_df, pool_df], ignore_index=True, sort=False)
    if "code" in full_pool.columns:
        full_pool = full_pool.drop_duplicates(subset=["code"], keep="first")

    _tc_gw = int(tc_target["gw"]) if tc_target and tc_target.get("gw") is not None else None
    _tc_col = f"xpts_gw{_tc_gw}" if _tc_gw is not None else None
    if _tc_col and _tc_col in full_pool.columns and "xpts_horizon_sum" in full_pool.columns \
            and (_tc_gw in gw_list or _tc_gw in value_tail):
        _m = full_pool["code"] == tc_target["code"]
        if _m.any():
            full_pool.loc[_m, "xpts_horizon_sum"] = (pd.to_numeric(full_pool.loc[_m, "xpts_horizon_sum"], errors="coerce")
                                                     + pd.to_numeric(full_pool.loc[_m, _tc_col], errors="coerce").fillna(0.0))
    else:
        _tc_gw = None

    def _tc_extra(sq, gws_valued):
        """Triple Captain's extra x1: the best XI's top scorer in the TC week (0 when no target / week not valued)."""
        if _tc_gw is None or _tc_gw not in gws_valued or sq is None or _tc_col not in sq.columns:
            return 0.0
        b = opt.best_starting_xi(sq, _tc_col)
        if not b or b.get("xi") is None or b["xi"].empty:
            return 0.0
        v = pd.to_numeric(b["xi"][_tc_col], errors="coerce").max()
        return 0.0 if pd.isna(v) else float(v)

    sim_squad = squad_df.copy()
    ft_bank = free_transfers
    weekly_plan = []
    plan = []
    summary = []
    total_net_gain = 0.0

    full_pool_fp = _frame_fingerprint(full_pool)  # Patch 112: one cheap content hash per plan, reused by every solve
    cfg_fp = repr(sorted((cfg.get("squad_rules") or {}).items())) if isinstance(cfg.get("squad_rules"), dict) else ""
    for wi, gw in enumerate(gw_list):
        # Patch 112: `value_tail` = extra GWs AFTER the planned span that are used only to VALUE a move (so late
        # weeks are not judged on 1-2 remaining GWs); they are never planned or displayed.
        remaining_gws = list(gw_list[wi:]) + value_tail
        this_gw_col = f"xpts_gw{gw}"

        # Patch 92 (a): weeks strictly before a scheduled Wildcard must not
        # be able to "see" value that only exists at/after the Wildcard
        # rebuild — that value is illusory since the Wildcard wipes the
        # squad clean regardless of what this week's transfer decision was.
        if wildcard_gw is not None and gw < wildcard_gw:
            remaining_gws = [g for g in remaining_gws if g < wildcard_gw]

        # Patch 98 (Rule #52 follow-up, "a more wide rule" — manager,
        # 2026-09-30): the SAME illusory-value problem Wildcard already had
        # a fix for also applies to a scheduled Free Hit, and had NO fix at
        # all before this patch — confirmed via code read of the whole
        # function body. Unlike Wildcard, Free Hit reverts after one week,
        # so this is an EXCLUSION of that one GW only, never a truncation of
        # everything from it onward (weeks after it are unaffected — the
        # persisted squad resumes normally). Applies to every week's
        # valuation, before, at, or after the Free Hit week, since
        # `remaining_gws` here is reused both for THIS week's own old/new
        # comparison and (via the wholesale-rebuild branch just below, and
        # every later iteration's own `gw_list[wi:]` slice) for every other
        # week's too.
        if freehit_gw is not None:
            remaining_gws = [g for g in remaining_gws if g != freehit_gw]

        # Patch 92 (b): at the scheduled Wildcard week itself, bypass the
        # ordinary k=1..k_upper search entirely — the whole squad is
        # replaced wholesale by the pre-computed rebuild squad, for free
        # (Rule #44), and the week still accrues its own +1 FT exactly as
        # an unused week would (no ft_used).
        if wildcard_gw is not None and gw == wildcard_gw and wildcard_rebuild_squad is not None:
            new_squad = wildcard_rebuild_squad.copy()
            bench_w_wc = cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08)
            # Patch 98 -- reuses `remaining_gws` (already Free-Hit-excluded
            # above) instead of recomputing an independent `gw_list[wi:]`
            # slice, so the Wildcard substitution week's own valuation gets
            # the same Free Hit exclusion every other week gets.
            wc_remaining = remaining_gws
            old_total_wc = opt.realized_horizon_value(sim_squad, wc_remaining, cfg,
                                                        bench_weight_scale=bench_w_wc, bb_play_gw=bb_play_gw) \
                + _tc_extra(sim_squad, wc_remaining)
            new_total_wc = opt.realized_horizon_value(new_squad, wc_remaining, cfg,
                                                        bench_weight_scale=bench_w_wc, bb_play_gw=bb_play_gw) \
                + _tc_extra(new_squad, wc_remaining)
            net_gain_wc = round(new_total_wc - old_total_wc, 2)
            pairs_wc = _pair_moves(sim_squad, new_squad, this_gw_col)
            week_moves = [{**_move_row(p, 0.0, net_gain_wc, True), "gw": gw} for p in pairs_wc]
            ft_used = 0
            ft_after = min(transfers.MAX_BANK, ft_bank + 1)
            week_summary = (f"GW{gw}: WILDCARD — full squad rebuild ({len(pairs_wc)} changes) — "
                             f"net {net_gain_wc:+.1f} xPts over the remaining horizon.")
            plan.append(f"GW{gw}: Wildcard played — squad rebuilt wholesale, no free transfer drawn "
                        f"(Rule #44), {ft_bank} FT banked → {ft_after} for GW{gw + 1} (unused-week accrual).")
            weekly_plan.append({
                "gw": gw, "moves": week_moves, "net_gain": net_gain_wc,
                "ft_available": ft_bank, "ft_used": ft_used, "ft_banked_after": ft_after,
                "summary": week_summary, "data_gap_note": None, "chip_played": "wildcard",
            })
            summary.append(week_summary)
            total_net_gain += net_gain_wc
            sim_squad = new_squad
            ft_bank = ft_after
            continue

        current_codes = list(sim_squad["code"])
        out_codes_all = set(current_codes)
        team_value = round(bank + (sim_squad["price"].sum(skipna=True) or 0.0), 1)
        # Patch 37: same near-zero bench weight (except bb_play_gw) as
        # suggest_transfers() — applied here too so the weekly pacing plan
        # doesn't let bench-quality alone carry a marginal move.
        bench_w = cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08)
        old_total = opt.realized_horizon_value(sim_squad, remaining_gws, cfg, bench_weight_scale=bench_w,
                                                bb_play_gw=bb_play_gw) + _tc_extra(sim_squad, remaining_gws)
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
            result = _solve_retain_memo(full_pool, full_pool_fp, cfg, cfg_fp, team_value, current_codes, min_retain,
                                        locks=lock_codes_of(sim_squad))
            if result is None:
                continue
            new_squad = result["squad"]
            actual_k = len(out_codes_all - set(new_squad["code"]))
            if actual_k == 0:
                continue  # nothing worth swapping at this k — already covered by k=0
            hit_cost = hit_cost_per * max(0, actual_k - ft_bank) if allow_hits else 0.0
            new_total = opt.realized_horizon_value(new_squad, remaining_gws, cfg, bench_weight_scale=bench_w,
                                                    bb_play_gw=bb_play_gw) + _tc_extra(new_squad, remaining_gws)
            # Patch 36 — same nailed-gate baseline as suggest_transfers(),
            # applied per week: a non-nailed out-player's projection is
            # zeroed across the remaining weeks before the baseline is
            # recomputed, so the incoming player is judged against the free
            # bench replacement, not the outgoing player's own (possibly
            # near-zero) live number.
            out_codes_this = out_codes_all - set(new_squad["code"])
            baseline_info = realistic_baseline_value(sim_squad, out_codes_this, remaining_gws, cfg,
                                                      bb_play_gw=bb_play_gw)
            baseline_total = baseline_info["baseline_total"] + _tc_extra(sim_squad, remaining_gws)
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
        # Patch 112 (chain only; `cap_use_bar=None` keeps the old behaviour): with the FT bank AT THE CAP a Roll
        # throws a transfer away (ft_after is capped at transfers.MAX_BANK), so any free move with a real gain
        # >= cap_use_bar is preferred over Rolling. It still has to pass the Starting-XI impact check below, and
        # a hit is never taken to do this.
        cap_forced = False
        if cap_use_bar is not None and ft_bank >= transfers.MAX_BANK:
            _free_ks = [k for k, c in candidates.items()
                        if k != 0 and c["hit_cost"] == 0 and c["net_gain"] >= cap_use_bar]
            if _free_ks:
                _best_free = max(candidates[k]["net_gain"] for k in _free_ks)
                viable = sorted(k for k in _free_ks if (_best_free - candidates[k]["net_gain"]) < moe)
                cap_forced = True

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
        # Patch 111: `use_tie_break=False` (default True = unchanged) skips this scan; it is ~93% of the
        # planner's runtime and only decides between near-tied in-players. Used by the Wildcard chain
        # comparison, whose first week comes from the manager's actual recommendation instead.
        if use_tie_break and chosen.get("actual_k") == 1:
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
            week_moves = [{**r, "gw": gw} for r in _assign_move_hits(
                [_move_row(p, 0.0, chosen["net_gain"], True) for p in pairs],
                chosen.get("hit_cost", 0.0), hit_cost_per)]
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
            "cap_forced": bool(cap_forced and chosen["actual_k"] > 0),
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
                       disrupted_codes: set | None = None,
                       chip_schedule: dict | None = None) -> dict:
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

    Patch 92: `chip_schedule` (see `plan_transfer_schedule()`'s own docstring
    for the full design) is passed straight through when this function
    dispatches to the chained weekly planner (horizon>1, "No hits"/"Hit if
    worth it") — it has no effect on the single-GW/"Force" path below, which
    doesn't chain across weeks so a scheduled Wildcard's squad-rebuild has
    nothing to be chained through in the first place.

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
                                       hit_stance=hit_stance, chip_schedule=chip_schedule)

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
                                  min_retain=min_retain, must_include_codes=lock_codes_of(squad_df),
                                  objective_col="xpts_horizon_sum")
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
            # Patch 56 — see _apply_eo_pull's docstring: retained_club_counts
            # seeds the running max-per-club check so EO-pull substitutions
            # across different legs can't combine into an over-3-per-club
            # squad the way a GW6 Wildcard "what-if" (4 Leeds players) did.
            _actual_out_codes = set(squad_df["code"]) - set(chosen["squad"]["code"])
            _retained_club_counts = squad_df[~squad_df["code"].isin(_actual_out_codes)]["team"] \
                .value_counts().to_dict()
            # Patch 59 — see _apply_eo_pull's docstring: a substitute must
            # never re-pick a code that's already staying in the squad
            # untouched (root cause of the reported "only 2 benched
            # players," found evaluating a Wildcard scenario but structurally
            # identical here — an ordinary transfer's eo_pull substitution
            # could just as easily collide with a retained squad member).
            _retained_codes_for_eo_pull = set(squad_df["code"]) - _actual_out_codes
            pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes_all,
                                    {p["in_code"] for p in pairs}, _retained_club_counts,
                                    _retained_codes_for_eo_pull)
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
            # Patch 56 — see _apply_eo_pull's docstring / the Force-branch
            # comment above: same running max-per-club fix for the default
            # (non-Forced) recommendation path.
            _actual_out_codes = set(squad_df["code"]) - set(chosen["squad"]["code"])
            _retained_club_counts = squad_df[~squad_df["code"].isin(_actual_out_codes)]["team"] \
                .value_counts().to_dict()
            # Patch 59 — see the Force-branch comment above / _apply_eo_pull's
            # docstring: same retained-code collision fix for the default
            # (non-Forced) recommendation path.
            _retained_codes_for_eo_pull = set(squad_df["code"]) - _actual_out_codes
            pairs = _apply_eo_pull(pairs, full_pool, profile, cfg, this_gw_col, out_codes_all,
                                    {p["in_code"] for p in pairs}, _retained_club_counts,
                                    _retained_codes_for_eo_pull)
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
                                  min_retain=min_retain, must_include_codes=[target_code] + lock_codes_of(squad_df),
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



def _assign_move_hits(rows: list, total_hit: float, hit_cost_per: float) -> list:
    """Patch 117a: per-move hit badge for a weekly-plan batch. The batch pays
    `total_hit` = hit_cost_per x (moves beyond the free transfers). Show the
    FREE badge on the highest-gain moves and the hit on the lowest-gain ones
    (the free transfers are best spent on the best moves). Row order is kept;
    every row's hit_cost is 0 or hit_cost_per. Display only - the batch
    net_gain and the decision are unchanged."""
    if not rows or not hit_cost_per or total_hit <= 0:
        return [{**r, "hit_cost": 0.0} for r in rows]
    n_hit = min(len(rows), int(round(total_hit / hit_cost_per)))
    order = sorted(range(len(rows)), key=lambda i: (rows[i].get("xpts_gain", 0.0), -i))
    hit_idx = set(order[:n_hit])
    return [{**r, "hit_cost": float(hit_cost_per) if i in hit_idx else 0.0} for i, r in enumerate(rows)]


def trigger_detect_window(detect_gw_list, base_size: int) -> list:
    """Patch 117a: the Wildcard trigger always reads the first `base_size`
    GWs of the detection list (4 = Rule #48's window). An Extended run widens
    detect_gw_list to 10 GWs for the reachable table and the cross-check; the
    trigger must not average over that wider list."""
    return list(detect_gw_list or [])[:int(base_size)]


def scan_alternative_line(rows: list, decision_gw) -> str | None:
    """Patch 117a: when the Rule #48 one-shot scan's best week differs from the
    week the chain decision plans, show the scan reading as a LABELLED,
    low-confidence alternative (it does not decide). Disclosed defect: the scan
    credits accrued free transfers as free + len(window) - 1 independent of the
    candidate week (chip_protocol.wildcard_window_value_scan), so later weeks are
    not penalised for the transfers they would have banked. `rows` = the best-GW
    table rows (gw, wc_gap)."""
    scored = [(r["gw"], r["wc_gap"]) for r in (rows or []) if r.get("wc_gap") is not None]
    if not scored or decision_gw is None:
        return None
    best_gw, best_val = max(scored, key=lambda x: (x[1], -x[0]))
    if best_gw == decision_gw:
        return None
    dec_val = dict(scored).get(decision_gw)
    dec_txt = f" (GW{decision_gw} scores {dec_val:+.1f} on the same scan)" if dec_val is not None else ""
    return (f"Scan-only reading (Rule #48, Wildcard played alone): best week GW{best_gw} {best_val:+.1f}{dec_txt}. "
            f"LOW CONFIDENCE, shown as an alternative only - it does not decide. Known defect: the scan counts "
            f"accrued free transfers independent of the candidate week, which flatters later weeks.")


def cs_carry_label(n_cs_override_players: int) -> str:
    """Patch 117a: disclosure for manual cs_pct_override entries. Patch 117
    applies one override value per player to EVERY gameweek (no per-GW scoping)."""
    if not n_cs_override_players:
        return ""
    return (f" Clean sheet carried at the GW6 sheet value for every week for {int(n_cs_override_players)} "
            f"player(s) with a cs_pct_override (one value per player, not per fixture).")


def apply_budget_override(bank: float, squad_market_price: float, override) -> tuple:
    """Patch 117b: the app's budget is bank + the squad's MARKET prices (no selling prices). If the manager types
    his real budget (squad value + bank from the FPL Transfers page, 0 = off) it replaces that. Returns
    (bank_effective, team_value, note). bank_effective is chosen so bank_effective + squad market price = the entered
    budget everywhere the app adds them (Wildcard / Free Hit / chain builds are exact; a single sale still uses the
    market price of the player sold, which is approximate). Values outside 50-150 are ignored."""
    market = round(float(bank) + float(squad_market_price), 1)
    try:
        ov = float(override) if override is not None else 0.0
    except (TypeError, ValueError):
        ov = 0.0
    if not (50.0 <= ov <= 150.0):
        return float(bank), market, None
    gap = round(market - ov, 1)
    note = (f"Budget set by you: {ov:.1f}m (squad value + bank). The app's market-price figure was {market:.1f}m "
            f"({gap:+.1f}m difference).")
    return round(ov - float(squad_market_price), 1), round(ov, 1), note


def sell_price_tenths(buy: int, now: int) -> int:
    """Patch 117c: FPL selling price in tenths of a million. If the price rose, you keep half the rise rounded DOWN
    to 0.1; if it fell or is unchanged you get the current price."""
    buy, now = int(buy), int(now)
    return buy + (now - buy) // 2 if now > buy else now


def selling_prices(squad_df, transfers, skip_events=None) -> dict:
    """Patch 117c: per-player selling price for the current squad from the manager's transfer history.
    Buy price = the player's LATEST purchase in `transfers` (element_in_cost, tenths); events in `skip_events`
    (Free Hit weeks, whose squad is reverted) are ignored. No purchase -> start-of-season price
    (now_cost - cost_change_start). No usable data -> market price, counted in n_estimated.
    Returns {"rows": [...], "total": float or None, "n_estimated": int}. total is None when the squad is empty."""
    skip = set(skip_events or [])
    latest = {}
    for t in sorted(transfers or [], key=lambda x: (str(x.get("time", "")), x.get("event", 0))):
        if t.get("event") in skip or t.get("element_in") is None or t.get("element_in_cost") is None:
            continue
        latest[int(t["element_in"])] = int(t["element_in_cost"])
    rows, n_est = [], 0
    for _, r in squad_df.iterrows():
        pid = r.get("id")
        now = int(round(float(r["price"]) * 10))
        buy, src = None, "market"
        if pid is not None and pd.notna(pid) and int(pid) in latest:
            buy, src = latest[int(pid)], "transfer"
        else:
            ccs = r.get("cost_change_start") if "cost_change_start" in squad_df.columns else None
            if ccs is not None and pd.notna(ccs):
                buy, src = now - int(ccs), "start_price"
        if buy is None:
            sell = now
            n_est += 1
        else:
            sell = sell_price_tenths(buy, now)
        rows.append({"id": pid, "buy": None if buy is None else round(buy / 10.0, 1), "now": round(now / 10.0, 1),
                     "sell": round(sell / 10.0, 1), "source": src})
    total = round(sum(x["sell"] for x in rows), 1) if rows else None
    return {"rows": rows, "total": total, "n_estimated": n_est}


def resolve_budget(bank: float, squad_market_price: float, selling_total, override, n_estimated: int = 0,
                   market_sum=None) -> tuple:
    """Patch 117c: budget = bank + selling prices (auto). A typed override (Advanced box) wins. If the selling total
    is unavailable, fall back to market prices and say so. Returns (bank_effective, team_value, note, source) with
    source in {"manual", "auto", "market"}; bank_effective + squad market price = team_value everywhere the app adds them."""
    try:
        ov = float(override) if override is not None else 0.0
    except (TypeError, ValueError):
        ov = 0.0
    market = round(float(bank) + float(squad_market_price), 1)
    if 50.0 <= ov <= 150.0:
        b, tv, note = apply_budget_override(bank, squad_market_price, ov)
        return b, tv, note, "manual"
    if selling_total is None:
        return float(bank), market, ("Budget from market prices: %.1fm (selling prices unavailable - the FPL transfers "
                                     "feed could not be read)." % market), "market"
    tv = round(float(bank) + float(selling_total), 1)
    if market_sum is not None:        # Patch 117d: the squad is already priced at selling prices, the bank stays the real bank
        market = round(float(bank) + float(market_sum), 1)
    note = ("Budget %.1fm (auto: bank %.1fm + selling prices %.1fm; market-price figure was %.1fm, %+.1fm difference)."
            % (tv, float(bank), float(selling_total), market, round(market - tv, 1)))
    if n_estimated:
        note += " %d player(s) had no buy price in the feed and were estimated at market price." % int(n_estimated)
    return round(tv - float(squad_market_price), 1), tv, note, "auto"



# --------------------------------------------------------------------------------------------------------------------
# Patch 117d helpers: owned players at selling price, Locked players, chained plan, snapshot, short UI chips.
# --------------------------------------------------------------------------------------------------------------------
def lock_codes_of(df) -> list:
    """Codes of the manager's Locked players, read from the boolean `_locked` column carried on the squad / pool frame
    (absent or NaN = not locked). Only the Wildcard rebuild and the transfer planner read it; ceilings, the Free Hit
    optimum and the Rating % yardstick never do."""
    if df is None or not hasattr(df, "columns") or "_locked" not in df.columns or "code" not in df.columns:
        return []
    m = df["_locked"].fillna(False).astype(bool)
    return sorted(int(c) for c in df.loc[m, "code"].tolist())


def lock_flags(squad_df, lock_codes) -> list:
    """[(name, status)] for locked players who are not fully available (status != 'a') - the only lock problem possible,
    since locks come from a legal squad and cost their selling price (always affordable)."""
    out = []
    want = set(lock_codes or [])
    for _, r in squad_df.iterrows():
        if r["code"] in want and str(r.get("status", "a")) != "a":
            out.append((r.get("web_name"), str(r.get("status"))))
    return out


def lock_cost(free_sq, locked_sq, window_gws, cfg) -> float:
    """xPts the locks cost: the unlocked Wildcard squad's window value minus the locked one's, same measure for both
    (realized value per week incl. bench and captain, summed over the window)."""
    def _v(sq):
        return sum(opt.rating_gw_value(sq, f"xpts_gw{g}", cfg)["total_realized"] for g in window_gws
                   if f"xpts_gw{g}" in sq.columns)
    return float(_v(free_sq) - _v(locked_sq))


def apply_sell_prices(pool, sp):
    """Owned players priced at what you would RECEIVE (selling price), everyone else at the live price. Keeping an owned
    player then costs his selling price, selling returns it, buying costs the live price. The market price is kept in
    `market_price`. Returns a copy; unknown ids are ignored."""
    out = pool.copy()
    if "id" not in out.columns:
        return out
    sell = {int(r["id"]): float(r["sell"]) for r in (sp or {}).get("rows", []) if r.get("id") is not None and pd.notna(r.get("id"))}
    if not sell:
        return out
    ids = out["id"]
    mask = ids.notna() & ids.fillna(-1).astype(int).isin(sell.keys())
    out["market_price"] = out["price"]
    out.loc[mask, "price"] = [sell[int(i)] for i in ids[mask]]
    return out


def reconcile_chip_schedule(chip_schedule, decision_gw, planning_gw, rebuild_by_gw):
    """Patch 117d: the weekly transfer plan must follow the Chip Plan decision. Wildcard decided NOW -> the plain list is the
    no-Wildcard alternative (no wildcard line). Wildcard decided for a later week -> the plan rolls to that week and
    rebuilds there (`rebuild_by_gw[week]`); with no squad for that week, no wildcard line. No decision -> unchanged."""
    if chip_schedule is None or decision_gw is None:
        return chip_schedule
    cur = chip_schedule.get("wildcard_gw")
    if cur is not None and int(cur) == int(decision_gw) and int(decision_gw) != int(planning_gw):
        return chip_schedule
    out = {k: v for k, v in chip_schedule.items() if k not in ("wildcard_gw", "wildcard_rebuild_squad")}
    if int(decision_gw) != int(planning_gw):
        rb = (rebuild_by_gw or {}).get(int(decision_gw))
        if rb is not None and not getattr(rb, "empty", False):
            out["wildcard_gw"] = int(decision_gw)
            out["wildcard_rebuild_squad"] = rb
    return out


def build_snapshot_zip(pool, owned_codes, locked_codes, bank, budget, meta, cfg, overrides_bytes=None) -> bytes:
    """Patch 117d: one downloadable file with everything needed to reproduce this run offline - the full pool (prices as the
    solver saw them, market price, xPts columns, owned / locked flags), the budget facts, the config, and the manual sheet."""
    import io, json, zipfile
    df = pool.copy()
    df["owned"] = df["code"].isin(set(owned_codes))
    df["locked"] = df["code"].isin(set(locked_codes or []))
    if "_locked" in df.columns:
        df = df.drop(columns=["_locked"])
    m = dict(meta or {})
    m.update({"bank": float(bank), "budget": float(budget), "locked_codes": [int(c) for c in (locked_codes or [])],
              "owned_codes": [int(c) for c in owned_codes]})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pool.csv", df.to_csv(index=False))
        z.writestr("meta.json", json.dumps(m, indent=1, default=str))
        z.writestr("config.json", json.dumps(cfg, indent=1, default=str))
        if overrides_bytes is not None:
            z.writestr("manual_overrides.csv", overrides_bytes)
    return buf.getvalue()


_CHIP_COL = {"ok": ("#1b6b3a", "#e3f3e8"), "warn": ("#8a5a00", "#fff1d6"), "bad": ("#9b1c1c", "#fde4e4"), "info": ("#33506b", "#e6eef5")}


def ui_chip(text, kind="info", sub=None, tip=None) -> str:
    """One short coloured chip (HTML). Text and tooltip are escaped; the long explanation goes in `tip` (hover)."""
    import html
    fg, bg = _CHIP_COL.get(kind, _CHIP_COL["info"])
    t = f' title="{html.escape(str(tip), quote=True)}"' if tip else ""
    s = f'<span{t} style="background:{bg};color:{fg};border-radius:10px;padding:2px 9px;font-size:12px;font-weight:600;">{html.escape(str(text))}</span>'
    if sub:
        s += f' <span style="color:{fg};font-size:11px;opacity:.8;">{html.escape(str(sub))}</span>'
    return s


def capped_note(raw_by_gw) -> str | None:
    """'Capped at 100% in N weeks - raw up to X% (GWn)'; None when nothing was capped."""
    over = {g: float(v) for g, v in (raw_by_gw or {}).items() if v is not None and float(v) > 100.0}
    if not over:
        return None
    g, v = max(over.items(), key=lambda kv: kv[1])
    return f"Capped at 100% in {len(over)} week{'s' if len(over) != 1 else ''} - raw up to {v:.1f}% (GW{g})"


def strip_md(text) -> str:
    """Plain text from light markdown (for tooltips)."""
    import re
    t = str(text)
    t = re.sub(r"[*`]", "", t)
    t = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", t)
    return t


def note_chip(label, kind="info", body=None) -> str:
    """Short chip for a message; the full text rides along as a hover tooltip."""
    return ui_chip(label, kind, tip=strip_md(body) if body else None)


def note_html(label, kind="info", body=None) -> str:
    """Chip + tap-to-open details (native <details>, no layout nesting). No body -> chip only."""
    import html
    if not body:
        return note_chip(label, kind)
    return (f'<details style="margin:2px 0;"><summary style="cursor:pointer;list-style:none;">'
            f'{note_chip(label, kind, body)} <span style="color:#778;font-size:12px;">ⓘ</span></summary>'
            f'<div style="font-size:12px;color:#556;padding:4px 2px 2px 6px;">{html.escape(strip_md(body))}</div></details>')


def details_html(summary_html, body_html) -> str:
    """Summary line (already html) that opens the full html block on tap."""
    return (f'<details style="margin:2px 0;"><summary style="cursor:pointer;">{summary_html} '
            f'<span style="color:#778;font-size:12px;">ⓘ</span></summary>{body_html}</details>')


def table_formats(columns) -> dict:
    """Column name -> display kind ('money' | 'num1' | 'bar'); text columns are left out."""
    out = {}
    for c in columns:
        s = str(c)
        low = s.lower()
        if "£" in s or low in ("price", "cost £m"):
            out[c] = "money"
        elif "–" in s or "total" in low or "horizon" in low or "window" in low:
            out[c] = "bar"
        elif low.startswith("xpts") or "gain" in low or low in ("net", "hit", "cost (xpts)") or "net" in low.split():
            out[c] = "num1"
        elif "xpts" in low or low.startswith("cost"):
            out[c] = "num1"
    return out


def chips_row_html(items) -> str:
    """Several chip/detail html fragments laid out in one wrapping row; '' when empty."""
    items = [i for i in (items or []) if i]
    if not items:
        return ""
    return '<div style="display:flex;flex-wrap:wrap;gap:6px 8px;align-items:flex-start;margin:2px 0 4px 0;">' + "".join(items) + "</div>"


def chance_bar_html(pct) -> str:
    """Small 'chance of playing' bar with the % beside it; '' when unknown."""
    if pct is None:
        return ""
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return ""
    if v != v:
        return ""
    p = max(0.0, min(100.0, v))
    col = "#2e9d57" if p >= 90 else ("#d9a400" if p >= 50 else "#d64545")
    return (f'<span style="display:inline-flex;align-items:center;gap:5px;" title="Chance of playing next round">'
            f'<span style="display:inline-block;width:46px;height:6px;background:#e8ecf0;border-radius:4px;">'
            f'<span style="display:block;height:6px;width:{p:.0f}%;background:{col};border-radius:4px;"></span></span>'
            f'<span style="font-size:11px;color:{col};font-weight:600;">{p:.0f}%</span></span>')


def strip_chance_suffix(text) -> str:
    """Remove a trailing ' - NN% chance of playing' (the bar shows it)."""
    import re
    return re.sub(r"\s*[-–—]\s*\d+%\s*chance of playing\s*$", "", str(text or "").strip()).strip()


def rating_bar_html(pct, label="") -> str:
    """Thin horizontal bar 0-100 % (clamped); '' when pct is None."""
    import html
    if pct is None:
        return ""
    p = max(0.0, min(100.0, float(pct)))
    col = "#2e9d57" if p >= 90 else ("#d9a400" if p >= 75 else "#d64545")
    return (f'<div style="margin:2px 0 6px 0;max-width:320px;"><div style="font-size:11px;color:#556;">{html.escape(str(label))} '
            f'<b>{float(pct):.1f}%</b></div>'
            f'<div style="background:#e8ecf0;border-radius:6px;height:8px;width:100%;">'
            f'<div style="background:{col};border-radius:6px;height:8px;width:{p:.0f}%;"></div></div></div>')


def budget_chip_text(budget, source, gap) -> str:
    """e.g. 'Budget 99.8m (auto, -0.8)'. gap = budget minus the live-price figure."""
    tag = {"auto": "auto", "manual": "manual", "market": "live prices"}.get(source, source)
    g = "" if source == "market" or abs(float(gap)) < 0.05 else f", {float(gap):+.1f}"
    return f"Budget {float(budget):.1f}m ({tag}{g})"


def plan_conflict_chip(plan_wc_gw, decision_gw):
    """Short flag when the weekly plan and the Chip Plan name different Wildcard weeks; None when they agree."""
    if plan_wc_gw is None or decision_gw is None or int(plan_wc_gw) == int(decision_gw):
        return None
    return f"Plan WC GW{plan_wc_gw} vs Chip Plan GW{decision_gw}"


def wildcard_week_conflict(plan_wc_gw, decision_gw):
    """Patch 117b: the weekly transfer plan is built before the Wildcard decision and uses the chip portfolio's
    (scan-based) Wildcard week; the Chip Plan decision uses the chain (model ruling M8). When they differ, say so."""
    if plan_wc_gw is None or decision_gw is None or int(plan_wc_gw) == int(decision_gw):
        return None
    return (f"The weekly plan below was built with the Wildcard at GW{plan_wc_gw} (chip portfolio / scan), but the "
            f"Chip Plan decision is GW{decision_gw} (chain comparison). Follow the Chip Plan: the plan's Wildcard line "
            f"is out of date, and its earlier weeks were valued against a horizon cut at GW{plan_wc_gw}.")


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


def top_scorer(sq: pd.DataFrame, col: str) -> dict | None:
    """Patch 113: the best-XI top scorer on `sq` for `col` -- the player a Triple Captain (or the armband) would sit on."""
    if sq is None or col not in sq.columns:
        return None
    best = opt.best_starting_xi(sq, col)
    if not best or best.get("xi") is None or best["xi"].empty:
        return None
    xi = best["xi"]
    vals = pd.to_numeric(xi[col], errors="coerce")
    if vals.isna().all():
        return None
    row = xi.loc[vals.idxmax()]
    return {"name": str(row.get("web_name", "")), "code": int(row["code"]) if "code" in xi.columns else None,
            "team": row.get("team", ""), "xpts": round(float(vals.max()), 2)}


def wc_rebuild_squad(pool: pd.DataFrame, cfg: dict, budget: float, window_gws: list, mode: str = "weekly_xi",
                     bench_w: float | None = None, bb_gw: int | None = None, tc_target: dict | None = None,
                     label: str = "chain_wc_rebuild", captain: bool | None = None) -> pd.DataFrame | None:
    """Patch 114: the Wildcard's rebuilt squad over `window_gws`. mode 'weekly_xi' (default): the squad and a legal starting
    XI for EACH week are chosen together -- the XI counts in full, the bench at the planner's low bench weight; the
    chip-aware variant also counts the Bench Boost week's bench in full (`bb_gw`) and the Triple Captain target's extra x1
    (`tc_target` {code, gw}). 'sum15': the Patch 113 plain 15-man sum (switch chip_extended_check.wc_objective).
    Patch 115 (Rule #54): weekly weights decay^k (chip_extended_check.wc_build_decay, 0.9) and the captain term
    (wc_build_captain) are part of the 'weekly_xi' objective.
    Returns the 15-man squad frame, or None when the window/solve is unavailable."""
    cols = [f"xpts_gw{g}" for g in window_gws if f"xpts_gw{g}" in pool.columns]
    if not cols:
        return None
    wp = pool.copy()
    if mode == "sum15":
        wp["_wc_obj"] = wp[cols].sum(axis=1)
        res = opt.solve_squad(wp, cfg, budget=float(budget), objective_col="_wc_obj", label=label,
                              must_include_codes=lock_codes_of(wp))
    else:
        bw = float(bench_w if bench_w is not None else cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08))
        bb_col = f"xpts_gw{bb_gw}" if (bb_gw is not None and f"xpts_gw{bb_gw}" in cols) else None
        bonus = None
        if tc_target and tc_target.get("gw") is not None and f"xpts_gw{int(tc_target['gw'])}" in cols:
            bonus = {"code": int(tc_target["code"]), "col": f"xpts_gw{int(tc_target['gw'])}"}
        ece = cfg.get("chip_extended_check", {}) or {}
        # Patch 115 (Rule #54): nearer weeks count more (decay) and each week's captain is in the objective
        wts = decay_weights(len(cols), float(ece.get("wc_build_decay", 0.9)))
        cap_on = bool(ece.get("wc_build_captain", True)) if captain is None else bool(captain)
        res = opt.solve_squad_xi_weighted(wp, cfg, float(budget), cols, bench_weight=bw, bb_col=bb_col, bonus=bonus,
                                          label=label, week_weights=wts, captain=cap_on,
                                          captain_k=int(ece.get("wc_build_captain_k", 60)),
                                          must_include_codes=lock_codes_of(wp))
    if res is None or res.get("squad") is None:
        return None
    sq = res["squad"]
    if not isinstance(sq, pd.DataFrame):
        sq = wp[wp["code"].isin([p["code"] for p in sq])]
    sq = sq.copy()
    sq["_wc_obj"] = sq[cols].sum(axis=1)
    return sq


parallel_workers = opt.parallel_workers


def four_week_total(score: dict, t: int, gws: list, weeks: int = 4):
    """Patch 115 fix (Rule #34 / #48(a)): the compared total for the Wildcard tie band = the squad's scored xPts over the
    FOUR gameweeks from `t` (about 250, so a band of about 5), whatever the primary window length. None if fewer remain."""
    ws = [g for g in gws if g >= t][:int(weeks)]
    if len(ws) < int(weeks):
        return None
    return round(float(sum(score[g] for g in ws)), 2)


def wc_rebuild_with_fallback(pool: pd.DataFrame, cfg: dict, budget: float, window_gws: list, mode: str = "weekly_xi",
                             bench_w: float | None = None, bb_gw: int | None = None, tc_target: dict | None = None,
                             label: str = "chain_wc_rebuild") -> dict:
    """Patch 115 fix: the Wildcard rebuild with a visible fallback ladder instead of a silent drop.
      tier 'weekly_xi_captain' -> the full Rule #54 objective (weekly XI + captain + bench + decay)
      tier 'weekly_xi'         -> the same without the captain term; `note` says Rule #54(a) is not met this run
      tier None                -> unavailable; `note` says why. NEVER the plain 15-man sum (Rule #54) -- that exists only when
                                  mode='sum15' is asked for explicitly (switch chip_extended_check.wc_objective).
    Returns {'squad', 'tier', 'note'}."""
    if mode == "sum15":
        sq = wc_rebuild_squad(pool, cfg, budget, window_gws, mode="sum15", label=label)
        return {"squad": sq, "tier": "sum15" if sq is not None else None,
                "note": None if sq is not None else "Wildcard rebuild unavailable: the 15-man solve returned no squad."}
    want_cap = bool((cfg.get("chip_extended_check", {}) or {}).get("wc_build_captain", True))
    sq = wc_rebuild_squad(pool, cfg, budget, window_gws, mode="weekly_xi", bench_w=bench_w, bb_gw=bb_gw,
                          tc_target=tc_target, label=label, captain=want_cap)
    if sq is not None:
        return {"squad": sq, "tier": "weekly_xi_captain" if want_cap else "weekly_xi", "note": None}
    if want_cap:
        sq = wc_rebuild_squad(pool, cfg, budget, window_gws, mode="weekly_xi", bench_w=bench_w, bb_gw=bb_gw,
                              tc_target=tc_target, label=label, captain=False)
        if sq is not None:
            return {"squad": sq, "tier": "weekly_xi",
                    "note": "Rule #54(a) not met this run: the captain term could not be solved in time, so the Wildcard squad "
                            "was built on the weekly XI without the captain (still the weekly objective, never the 15-man sum)."}
    diag = None
    try:
        diag = opt.get_diagnostic(label)
    except Exception:
        diag = None
    return {"squad": None, "tier": None,
            "note": "Wildcard rebuild unavailable: the weekly-XI solve returned no squad (solver time cap "
                    f"{(cfg.get('solver', {}) or {}).get('time_limit_seconds', '?')} s reached, or infeasible)"
                    + (f" - {diag}" if diag else "") + "."}


def pick_wc_variant(plain_total: float | None, aware_total: float | None, margin: float) -> dict:
    """Patch 114: plain (XI-first) Wildcard unless the chip-aware one (Bench Boost bench + Triple Captain target) is clearly
    better -- strictly more than the margin of error. Returns {'variant': 'plain'|'chip-aware', 'edge': aware - plain}."""
    if aware_total is None:
        return {"variant": "plain", "edge": None}
    if plain_total is None:
        return {"variant": "chip-aware", "edge": None}
    edge = round(float(aware_total) - float(plain_total), 6)
    return {"variant": "chip-aware" if edge > float(margin) + 1e-9 else "plain", "edge": edge}


def tc_target_candidates(pool: pd.DataFrame, tc_gw: int, own_codes, k: int = 3) -> list[dict]:
    """Patch 114: the top-`k` available players by xPts in the Triple Captain week who are NOT already in the squad."""
    col = f"xpts_gw{int(tc_gw)}"
    if pool is None or pool.empty or col not in pool.columns:
        return []
    d = pool[~pool["code"].isin(set(own_codes))]
    if "status" in d.columns:
        d = d[d["status"] == "a"]
    d = d.assign(_v=pd.to_numeric(d[col], errors="coerce")).dropna(subset=["_v"]).sort_values("_v", ascending=False).head(int(k))
    return [{"code": int(r["code"]), "name": str(r.get("web_name", "")), "xpts": round(float(r["_v"]), 2),
             "position": r.get("position"), "price": r.get("price")} for _, r in d.iterrows()]


def chain_chip_value(squads_by_gw: dict, gws: list, cfg: dict, wc_gw: int | None, fh_gw: int | None,
                     fh_ref_score: float | None, types: tuple, moe_fn=None, pick_rule: str = "best_horizon",
                     allowed: dict | None = None) -> dict:
    """Patch 112/113 (manager: Triple Captain / Bench Boost / Free Hit must count in the Wildcard decision).
    Chip value for ONE squad path (`squads_by_gw` = the squad fielded each GW). DEVIATION from Standing Rule #31
    (captaincy is disclosure, never a scoring input) -- explicit manager instruction, flagged for the model chat.
      Bench Boost   = that week's bench xPts (best XI recomputed per week); never the Wildcard/Free Hit week.
      Triple Captain= the best-XI top scorer's xPts (the extra x1 on top of the captain doubling already counted),
                      never the Wildcard/Free Hit/Bench Boost week; the PLAYER is named (`tc_player`).
      Free Hit      = (fh_ref_score - squad's best-XI value) at `fh_gw`, floored at 0 (it reverts, so only the gap counts).
    Patch 113 pick rule `pick_rule="best_horizon"` (default): the best week across the WHOLE horizon; when both TC and BB
    are wanted they are chosen as a PAIR (best combined value, different weeks). Near-ties are reported (`tc_conf`/
    `bb_conf` = 'near-tie', and `tc_tie_set`/`bb_tie_set` = every week within Rule #34's band of the best; compared horizon
    = one week, so the 2-point floor) instead of moving the chip to another week.
    Patch 115 (model v6.12 ruling 1): EXACT ties go to the LATER week. `pick_rule="best_horizon_earlier"` keeps the Patch 113
    behaviour (exact ties -> earlier); `pick_rule="latest_tied"` restores the Patch 112 rule (ties within the margin of
    error -> latest week, BB first).
    `allowed` = {chip key: GWs} limits a chip to its earliest still-available window (a first-half chip lapses at the
    deadline); None = no restriction. `types` uses the chip keys '3xc', 'bboost', 'freehit'.
    Returns {'tc','bb','fh': (gw, value)|None, 'total', 'tc_player', '*_by_gw', '*_gap', '*_best', '*_conf'}."""
    blocked = {g for g in (wc_gw, fh_gw) if g is not None}
    per = {}
    for g in gws:
        sq = squads_by_gw.get(g)
        col = f"xpts_gw{g}"
        if sq is None or col not in sq.columns:
            continue
        best = opt.best_starting_xi(sq, col)
        if not best or best.get("xi") is None or best["xi"].empty:
            continue
        xi = best["xi"]
        xi_codes = set(xi["code"])
        bench = sq[~sq["code"].isin(xi_codes)]
        xv = pd.to_numeric(xi[col], errors="coerce")
        top = xi.loc[xv.idxmax()] if not xv.isna().all() else None
        per[g] = {"top": round(float(xv.max()), 2),
                  "bench": round(float(pd.to_numeric(bench[col], errors="coerce").sum(skipna=True)), 2) if not bench.empty else 0.0,
                  "player": ({"gw": g, "name": str(top.get("web_name", "")), "code": int(top["code"]),
                              "team": top.get("team", ""), "xpts": round(float(xv.max()), 2)} if top is not None else None)}
    out = {"tc": None, "bb": None, "fh": None, "tc_by_gw": {}, "bb_by_gw": {}, "tc_gap": None, "bb_gap": None,
           "tc_best": None, "bb_best": None, "tc_player": None, "tc_conf": None, "bb_conf": None,
           "tc_runner": None, "bb_runner": None, "tc_tie_set": [], "bb_tie_set": []}

    def _ok(key, g):
        return g not in blocked and (allowed is None or key not in allowed or g in set(allowed[key]))

    def _moe(v):
        return float(moe_fn(v)) if moe_fn is not None else 0.0

    def _gap(cand, g):
        rest = [v for k, v in cand.items() if k != g]
        return round(cand[g] - max(rest), 2) if rest else None

    def _runner(cand, g):
        rest = {k: v for k, v in cand.items() if k != g}
        if not rest:
            return None
        k = max(rest, key=lambda x: (rest[x], -x))
        return (k, rest[k])

    def _latest_tied(cand):
        best = max(cand.values())
        tied = [k for k, v in cand.items() if best - v <= _moe(best) + 1e-9]
        return max(tied)

    later = pick_rule != "best_horizon_earlier"

    def _argmax(cand):
        # highest value; Patch 115: exact ties -> the LATER week (model v6.12), unless the earlier-week switch is set
        return max(cand, key=(lambda k: (cand[k], k)) if later else (lambda k: (cand[k], -k)))

    def _tie_set(cand):
        best = max(cand.values())
        return sorted((k, round(v, 2)) for k, v in cand.items() if best - v <= _moe(best) + 1e-9)

    def _fill(slot, cand, g):
        out[slot] = (g, cand[g])
        out[slot + "_by_gw"] = dict(cand)
        out[slot + "_gap"] = _gap(cand, g)
        out[slot + "_runner"] = _runner(cand, g)
        b = _argmax(cand)
        out[slot + "_best"] = (b, cand[b])
        out[slot + "_tie_set"] = _tie_set(cand)
        gp = out[slot + "_gap"]
        out[slot + "_conf"] = "clear" if (gp is not None and gp > _moe(cand[g]) + 1e-9) else "near-tie"

    want_bb, want_tc = "bboost" in types, "3xc" in types
    cand_bb = {g: v["bench"] for g, v in per.items() if want_bb and _ok("bboost", g)}
    cand_tc = {g: v["top"] for g, v in per.items() if want_tc and _ok("3xc", g)}
    gb = gt = None
    if pick_rule == "latest_tied":
        if cand_bb:
            gb = _latest_tied(cand_bb)
        if cand_tc:
            cand_tc = {g: v for g, v in cand_tc.items() if g != gb}
            if cand_tc:
                gt = _latest_tied(cand_tc)
    else:
        if cand_bb and cand_tc:
            pairs = [(cand_bb[b] + cand_tc[t], b, t) for b in cand_bb for t in cand_tc if b != t]
            if pairs:
                _, gb, gt = max(pairs, key=(lambda x: (x[0], x[1], x[2])) if later else (lambda x: (x[0], -x[1], -x[2])))
            else:
                gb, gt = _argmax(cand_bb), None
        elif cand_bb:
            gb = _argmax(cand_bb)
        elif cand_tc:
            gt = _argmax(cand_tc)
    if gb is not None:
        _fill("bb", cand_bb, gb)
    if gt is not None:
        tc_pool = {g: v for g, v in cand_tc.items() if gb is None or g != gb} if pick_rule == "latest_tied" else cand_tc
        _fill("tc", tc_pool if gt in tc_pool else cand_tc, gt)
        out["tc_player"] = per[gt]["player"]
    # caption honesty: when a chip's best week was given to the other chip (pair total), say which
    out["tc_yield"] = ("the Bench Boost" if out["tc"] and out["tc_best"] and out["tc_best"][0] != out["tc"][0]
                       and gb is not None and out["tc_best"][0] == gb else None)
    out["bb_yield"] = ("the Triple Captain" if out["bb"] and out["bb_best"] and out["bb_best"][0] != out["bb"][0]
                       and gt is not None and out["bb_best"][0] == gt else None)
    if "freehit" in types and fh_gw is not None and fh_ref_score is not None and squads_by_gw.get(fh_gw) is not None:
        cur = opt.rating_gw_value(squads_by_gw[fh_gw], f"xpts_gw{fh_gw}", cfg)["total_realized"]
        out["fh"] = (fh_gw, round(max(0.0, float(fh_ref_score) - float(cur)), 2))
    out["total"] = round(sum(v[1] for v in (out["tc"], out["bb"], out["fh"]) if v), 2)
    return out


def chip_edge_text(chosen, best, gap, conf, what, yielded_to=None, tie_set=None) -> str:
    """Patch 113: one honest sentence about how clear a chip week is. Always starts with a capital and ends with a
    full stop, so it can follow any caption without stray punctuation."""
    if chosen is None:
        return ""
    gw, val = chosen
    if best is not None and best[0] != gw:
        why = f" because GW{best[0]} hosts {yielded_to}" if yielded_to else ""
        return (f"GW{best[0]} scores higher on its own ({best[1]:.1f} xPts) but GW{gw} is used{why}; "
                f"the pair together has the higher total.")
    if gap is None:
        return f"GW{gw} is the only {what} available."
    if conf == "clear":
        return f"GW{gw} is {gap:+.1f} xPts clear of the next-best {what}."
    tie_txt = ""
    others = [w for w, _ in (tie_set or []) if w != gw]
    if others:
        tie_txt = " Tie set (inside the band): " + ", ".join(f"GW{w}" for w in sorted(others + [gw])) + "."
    return (f"GW{gw} is the best {what} but only {max(gap, 0.0):+.1f} xPts ahead of the next one, inside the margin of "
            f"error, so treat it as low confidence." + tie_txt)


def chip_captain_rows(now: dict | None, planning_gw: int, wc_gw: int | None, wc_top: dict | None,
                      tc_gw: int | None, tc_top: dict | None) -> list[dict]:
    """Patch 113: the armband story in one list -- now (current squad), after the recommended Wildcard, and the Triple
    Captain week. Rows for chips that are not planned are left out."""
    rows = []
    if now:
        rows.append({"kind": "now", "gw": int(planning_gw), "name": now.get("name", ""), "xpts": float(now.get("xpts", 0.0)),
                     "label": "Now, with your current squad"})
    if wc_gw is not None and wc_top:
        rows.append({"kind": "wildcard", "gw": int(wc_gw), "name": wc_top.get("name", ""), "xpts": float(wc_top.get("xpts", 0.0)),
                     "label": f"After the Wildcard (GW{wc_gw}), on the Wildcard squad"})
    if tc_gw is not None and tc_top:
        x = float(tc_top.get("xpts", 0.0))
        rows.append({"kind": "triple_captain", "gw": int(tc_gw), "name": tc_top.get("name", ""), "xpts": x,
                     "total": round(x * 3, 2), "label": f"Triple Captain (GW{tc_gw}), tripled"})
    return rows


def override_chip_detail(detail: dict | None, path_chips: dict | None, floor: float) -> tuple[dict, dict]:
    """Patch 112: swap the joint-sequence weeks of Triple Captain / Bench Boost / Free Hit for the weeks picked on the
    chain's own weekly squads (plain Wildcard rebuild + later transfers). Chips the sequence scheduled are rewritten;
    a chip it did not schedule is added only if its chain value clears `floor`. The Wildcard entry is untouched.
    Returns (new_detail, {chip_key: (old_gw, new_gw)}) -- the input is not mutated."""
    detail = detail or {}
    new = {k: dict(v) for k, v in detail.items()}
    changes: dict = {}
    if not path_chips:
        return new, changes
    for key, slot in (("3xc", "tc"), ("bboost", "bb"), ("freehit", "fh")):
        pick = path_chips.get(slot)
        if not pick:
            continue
        gw, val = pick
        if key in new and float(val) < float(floor):
            changes[key] = (new[key].get("gw"), None)       # no real edge: hold the chip (Rule #34 -- play only above the margin)
            del new[key]
            continue
        if key in new or float(val) >= float(floor):
            old = new.get(key, {}).get("gw")
            e = dict(new.get(key, {}))
            e["gw"], e["value"] = int(gw), round(float(val), 2)
            if key == "3xc":
                _tp = path_chips.get("tc_player")
                if _tp and _tp.get("gw") == int(gw):
                    e["player"] = _tp.get("name")
                else:
                    e.pop("player", None)
            new[key] = e
            if old != gw:
                changes[key] = (old, int(gw))
    return new, changes


def style_swap_overlay(plain: pd.DataFrame, styled: pd.DataFrame, window_cols: list) -> tuple[list, float]:
    """Patch 112: the Wildcard is decided and scored PLAIN (best xPts squad); a style profile only swaps tied players.
    Lists each swap (out -> in, matched by position) with its xPts cost over `window_cols` (plain minus styled,
    positive = the style costs points) and the total. Empty when the style changes nothing."""
    if plain is None or styled is None or plain.empty or styled.empty:
        return [], 0.0
    p_codes, s_codes = set(plain["code"]), set(styled["code"])
    outs = plain[plain["code"].isin(p_codes - s_codes)]
    ins = styled[styled["code"].isin(s_codes - p_codes)]
    cols = [c for c in window_cols if c in plain.columns and c in styled.columns]

    def _v(r):
        return float(pd.to_numeric(r[cols], errors="coerce").fillna(0.0).sum()) if cols else 0.0
    swaps, used = [], set()
    for _, o in outs.iterrows():
        cand = ins[(ins["position"] == o["position"]) & (~ins["code"].isin(used))]
        if cand.empty:
            continue
        i = cand.iloc[0]
        used.add(i["code"])
        swaps.append({"out_code": o["code"], "in_code": i["code"], "out": o.get("web_name", o["code"]),
                      "in": i.get("web_name", i["code"]), "cost": round(_v(o) - _v(i), 2)})
    return swaps, round(sum(x["cost"] for x in swaps), 2)
