"""
Patch 82 (2026-09-28, manager uploaded the v6.9 model doc and asked to
confirm the model version + what should change in the app before a new
patch, then said to start the first patch combining that with the
already-tested-but-held-back performance fixes).

Verifies, against the REAL current code (not a hand re-derivation), the two
changes v6.9's amended §8c "Wildcard trigger definition" requires and that
the pre-Patch-82 code was confirmed NOT to do:

1. The dual-leg trigger (avg Team Rating % < 79% OR cumulative gap >= 15
   xPts, fpl_engine.wildcard_trigger_check() pre-Patch-82) is RETIRED and
   replaced by a single scale-free gap check: avg_gap_pct =
   100 - avg_rating_pct >= wildcard_trigger.gap_pct_threshold (default 6.0).
2. The reachable ceiling used by the trigger now accrues one free transfer
   per GW across the detection window (data_pipeline.
   solve_reachable_ceiling_by_gw()), instead of solving ONE reachable squad
   with a single static free_transfers count and reusing it for every GW
   (confirmed present in the pre-Patch-82 code by reading the old
   app.py call site).
"""
import numpy as np
import pandas as pd
import yaml

import data_pipeline
import fpl_engine as eng
import optimizer as opt

with open("model_config.yaml") as f:
    CFG = yaml.safe_load(f)


def _pool(seed=11, n=500, detect_gw_list=(6, 7, 8, 9)):
    np.random.seed(seed)
    positions = np.random.choice(["GK", "DEF", "MID", "FWD"], size=n, p=[0.13, 0.35, 0.35, 0.17])
    codes = np.arange(1, n + 1)
    proj = pd.DataFrame({
        "code": codes, "web_name": [f"Player{i}" for i in range(n)],
        "team": np.random.choice([f"T{i}" for i in range(20)], size=n),
        "position": positions, "price": np.round(np.random.uniform(4.0, 13.0, n), 1),
        "status": "a",
    })
    for gw in detect_gw_list:
        proj[f"xpts_gw{gw}"] = np.round(np.random.uniform(1, 10, n), 2)
    proj["xpts_horizon_sum"] = proj[[f"xpts_gw{gw}" for gw in detect_gw_list]].sum(axis=1)
    return proj


def _legal_squad_codes(pool: pd.DataFrame, n_per_pos=(("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3))) -> list:
    """Picks a squad with the EXACT legal 2-5-5-3 position composition
    (spread across as many different teams as the pool allows, to also
    avoid an accidental max-3-per-club infeasibility). A random `.iloc[:15]`
    slice of a synthetic pool can easily land on a position mix (e.g. only
    1 usable GK) that makes even a loose retain-pool constraint
    structurally infeasible regardless of budget/price -- a synthetic-data
    artifact, not something a real manager's actual (always-legal) squad
    would ever produce."""
    codes = []
    for pos, k in n_per_pos:
        pos_rows = pool[pool["position"] == pos].drop_duplicates(subset="team").head(k)
        if len(pos_rows) < k:
            pos_rows = pool[pool["position"] == pos].head(k)
        codes.extend(pos_rows["code"].tolist())
    return codes


def test_config_key_is_retired_and_replaced():
    """Structural check: the old dual-leg keys are gone from the live
    config, and the new single scale-free threshold is present."""
    wt = CFG.get("wildcard_trigger", {})
    assert "team_rating_pct_ceiling" not in wt, "the retired 79% ceiling key should no longer be read from config"
    assert "cumulative_gap_threshold" not in wt, "the retired 15 xPts absolute-gap key should no longer be read"
    assert "gap_pct_threshold" in wt, "the new v6.9 scale-free gap_pct_threshold key is missing from model_config.yaml"
    assert wt["gap_pct_threshold"] == 6.0, f"expected the doc's own ~6% figure as the default, got {wt['gap_pct_threshold']}"
    print("Test 1 PASSED: model_config.yaml's wildcard_trigger block carries the new v6.9 gap_pct_threshold key, "
          "the retired dual-leg keys are gone")


def test_trigger_fires_at_scale_free_gap_not_old_dual_leg():
    """Build a squad exactly at the old trigger's dead zone (Team Rating %
    comfortably above 79%, e.g. 92%) -- under the OLD dual-leg logic this
    would never fire (92% > 79%, and if the absolute gap is also small it
    wouldn't cross 15 xPts either). Under v6.9's 6% gap threshold, 92%
    rating = 8% gap, which DOES fire (8% >= 6%). This is the exact
    contradiction v6.9's changelog describes the old trigger having."""
    detect_gw_list = [6, 7, 8, 9]
    pool = _pool(detect_gw_list=detect_gw_list)
    squad_codes = _legal_squad_codes(pool)
    squad_df = pool[pool["code"].isin(squad_codes)].copy()

    # Build a "reachable" squad per GW whose realized total is always
    # exactly squad_total / 0.92, i.e. squad sits at 92% Team Rating (8% gap)
    # every week -- deterministic, no MILP noise, isolates the threshold
    # logic itself.
    reachable_by_gw = {}
    for gw in detect_gw_list:
        col = f"xpts_gw{gw}"
        squad_val = opt.rating_gw_value(squad_df, col, CFG)["total_realized"]
        reachable_val = squad_val / 0.92
        r_squad = squad_df.copy()
        # Scale this fake "reachable" squad's own projections up so
        # rating_gw_value on it reproduces reachable_val exactly enough
        # (best-XI/captain logic is monotonic in the underlying values for
        # a fixed squad shape, so a uniform per-row scale-up achieves this).
        r_squad[col] = r_squad[col] * (reachable_val / max(squad_val, 1e-9))
        reachable_by_gw[gw] = {"squad": r_squad}

    result = eng.wildcard_trigger_check(squad_df, reachable_by_gw, detect_gw_list, CFG)
    assert result["avg_rating_pct"] is not None
    assert 88.0 <= result["avg_rating_pct"] <= 96.0, f"expected ~92% rating, got {result['avg_rating_pct']}"
    assert result["active"] is True, (
        f"a squad at ~92% Team Rating (8% gap) must trigger under v6.9's 6% scale-free threshold -- got "
        f"active={result['active']}, reason={result['reason']}"
    )
    assert "avg_gap_pct" in result and result["avg_gap_pct"] >= 6.0
    print(f"Test 2 PASSED: a squad at {result['avg_rating_pct']}% Team Rating ({result['avg_gap_pct']}% gap) -- "
          f"comfortably above the RETIRED 79% ceiling and thus a dead zone under the old dual-leg trigger -- "
          f"correctly fires under v6.9's scale-free 6% threshold. Reason: {result['reason']}")


def test_trigger_does_not_fire_inside_the_new_threshold():
    """Symmetric check: a squad at ~97% Team Rating (3% gap, below the 6%
    threshold) must NOT trigger."""
    detect_gw_list = [6, 7, 8, 9]
    pool = _pool(detect_gw_list=detect_gw_list)
    squad_codes = _legal_squad_codes(pool)
    squad_df = pool[pool["code"].isin(squad_codes)].copy()

    reachable_by_gw = {}
    for gw in detect_gw_list:
        col = f"xpts_gw{gw}"
        squad_val = opt.rating_gw_value(squad_df, col, CFG)["total_realized"]
        reachable_val = squad_val / 0.97
        r_squad = squad_df.copy()
        r_squad[col] = r_squad[col] * (reachable_val / max(squad_val, 1e-9))
        reachable_by_gw[gw] = {"squad": r_squad}

    result = eng.wildcard_trigger_check(squad_df, reachable_by_gw, detect_gw_list, CFG)
    assert result["active"] is False, (
        f"a squad at ~97% Team Rating (3% gap) is inside the 6% threshold and must not trigger -- got "
        f"active={result['active']}, reason={result['reason']}"
    )
    print(f"Test 3 PASSED: a squad at {result['avg_rating_pct']}% Team Rating ({result['avg_gap_pct']}% gap) "
          f"correctly stays HOLD under the 6% threshold")


def test_reachable_ceiling_accrues_free_transfers_per_gw():
    """The real end-to-end check on data_pipeline.solve_reachable_ceiling_
    by_gw(): later GWs in the detection window must use a looser retain
    constraint (more accrued free transfers) than earlier ones, capped at
    the real bank cap of 5 -- verified by reading each returned squad's
    min_retain indirectly via how many of the original squad codes survive
    in the solved reachable squad (fewer retained codes = a looser
    constraint was available, since the solver is always free to drop
    retained codes it doesn't need)."""
    detect_gw_list = [6, 7, 8, 9]
    pool = _pool(detect_gw_list=detect_gw_list, n=600)
    squad_codes = _legal_squad_codes(pool)
    # min_retain reaches up to 14 here (free_transfers=1) -- keep the
    # retained squad's own prices modest so retaining that many of them
    # plus filling the rest is affordable within the £100m budget (see the
    # identical reasoning in test_patch73/74's golden-slice fixes).
    pool.loc[pool["code"].isin(squad_codes), "price"] = np.round(np.random.uniform(4.0, 6.0, 15), 1)

    result = data_pipeline.solve_reachable_ceiling_by_gw(CFG, pool, squad_codes, free_transfers=1,
                                                           detect_gw_list=detect_gw_list)
    assert set(result.keys()) == set(detect_gw_list), "expected one entry per GW in detect_gw_list"
    for gw, entry in result.items():
        assert entry is not None and entry.get("squad") is not None and not entry["squad"].empty, \
            f"GW{gw}'s reachable-ceiling solve should succeed on this generously-sized, fully-available pool"

    # free_transfers=1 -> GW6 (offset 0) accrued_ft=1, min_retain=14;
    # GW9 (offset 3) accrued_ft=4, min_retain=11 -- GW9's constraint is
    # strictly looser, so it should retain no MORE original squad codes
    # than GW6 does (and, on a reasonably rich pool, strictly fewer, since
    # more freedom to improve should get used).
    retained_counts = {}
    for gw, entry in result.items():
        retained_counts[gw] = len(set(entry["squad"]["code"]) & set(squad_codes))
    assert retained_counts[6] >= retained_counts[9], (
        f"GW6 (least accrued FT, tightest retain constraint) should retain at least as many original squad codes "
        f"as GW9 (most accrued FT, loosest constraint) -- got {retained_counts}"
    )
    assert retained_counts[6] >= 14, f"GW6 with free_transfers=1 should retain >=14 of the 15 original codes, got {retained_counts[6]}"
    assert retained_counts[9] <= 12, f"GW9 with 4 accrued FT should retain <=11-12 of the 15 original codes, got {retained_counts[9]}"
    print(f"Test 4 PASSED: retained original-squad-code counts by GW (free_transfers=1, accruing): "
          f"{retained_counts} -- later GWs in the window get a genuinely looser reachable-ceiling constraint, "
          f"not the same static snapshot reused for every week")


def test_accrual_is_capped_at_the_real_bank_cap_of_five():
    """A very large starting free_transfers must never push accrued_ft past
    5 (Standing Rule #35/#37's real bank cap) -- verified by checking that
    GW9 (offset 3) with a starting free_transfers of 10 behaves identically
    (same retained-code count) to starting free_transfers of 5, since both
    should cap at accrued_ft=5 by GW9 (5+3 and 10+3 both clamp to 5)."""
    detect_gw_list = [6, 7, 8, 9]
    pool = _pool(detect_gw_list=detect_gw_list, seed=23, n=600)
    squad_codes = _legal_squad_codes(pool)
    pool.loc[pool["code"].isin(squad_codes), "price"] = np.round(np.random.uniform(4.0, 6.0, 15), 1)

    r_ft5 = data_pipeline.solve_reachable_ceiling_by_gw(CFG, pool, squad_codes, free_transfers=5,
                                                          detect_gw_list=detect_gw_list)
    r_ft10 = data_pipeline.solve_reachable_ceiling_by_gw(CFG, pool, squad_codes, free_transfers=10,
                                                           detect_gw_list=detect_gw_list)
    for gw in detect_gw_list:
        c5 = len(set(r_ft5[gw]["squad"]["code"]) & set(squad_codes))
        c10 = len(set(r_ft10[gw]["squad"]["code"]) & set(squad_codes))
        assert c5 == c10, (
            f"GW{gw}: free_transfers=5 and free_transfers=10 should both clamp accrued FT at the real bank cap "
            f"of 5 and produce the same retain constraint -- got retained counts {c5} vs {c10}"
        )
    print("Test 5 PASSED: accrued free transfers never exceed the real 5-transfer bank cap regardless of how "
          "large a starting free_transfers value is passed in")


if __name__ == "__main__":
    test_config_key_is_retired_and_replaced()
    test_trigger_fires_at_scale_free_gap_not_old_dual_leg()
    test_trigger_does_not_fire_inside_the_new_threshold()
    test_reachable_ceiling_accrues_free_transfers_per_gw()
    test_accrual_is_capped_at_the_real_bank_cap_of_five()
    print("\nALL PATCH 82 WILDCARD-TRIGGER-V6.9 TESTS PASSED")
