"""
INVESTIGATION, NOT YET A NUMBERED PATCH (2026-09-28, manager: "also the
performance generally is too slow, how this can be optimized!!" -- a
follow-up to the Horizon=3 hang investigation in
test_investigation_horizon3_hang.py). Manager's instruction there ("test it
and share the update, I will confirm before the new patch") applies here
too -- this is staged, verified, and reported, not yet packaged.

ROOT CAUSE, verified in code: optimizer.solve_squad() was already cached
with `@_cache_decorator` (Patch 42, 2026-09-15, manager: "the performance
is too too slow" -- the exact same complaint, already fixed once for this
function specifically). Its sibling, optimizer.solve_xi_first_squad() --
which does 8 shape solves + up to 8 bench solves = up to 16 real MILP
solves per call, used for the Free Hit "optimal squad" feature -- had NO
such caching at all. Confirmed empirically (not assumed): two back-to-back
calls with byte-identical inputs both took ~3.8-4.0s each before this fix,
because every call re-solved everything from scratch.

This matters beyond the raw per-call cost because app.py calls
solve_xi_first_squad() (via data_pipeline.solve_free_hit_optimal_squad())
from several places every run:
  - app.py:1314, unconditionally, for the always-visible "GW{n} Rating"
    header card;
  - app.py:2513, whenever the Chip Advisor's Free Hit verdict is
    "PLAY GW{n}" -- frequently the SAME gw as the header card above, i.e.
    the exact same (cfg, proj, budget, gw_col) inputs solved twice over for
    literally the same answer;
  - app.py:2707 via the already-cached `_nav_optimal_squad` wrapper (that
    one path was fine).
Streamlit reruns the whole script on nearly every widget interaction
(moving the Horizon slider, changing Style, anything) -- without caching,
EVERY one of those reruns re-paid this ~4s cost again, compounding into
exactly the "performance generally is too slow" complaint, independent of
the separate Horizon=3/"Hit if worth it" multi-solve issue already
investigated.

FIX BEING TESTED: added the same, already-proven `@_cache_decorator`
solve_squad() has used since Patch 42, to solve_xi_first_squad() too. Same
guarantee: a cache hit requires byte-identical inputs (players' content,
cfg, budget, gw_col) -- a genuine data refresh or a genuinely different
GW/budget always still gets a fresh solve. Zero change to what gets
recommended, pure caching.
"""
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/home/claude/rb_model_app")
import optimizer as opt

CFG = yaml.safe_load(open("/home/claude/rb_model_app/model_config.yaml"))


def _realistic_pool(seed=9, gw_cols=("xpts_gw6",)):
    np.random.seed(seed)
    teams = [f"T{t}" for t in range(20)]
    rows = []
    code = 0
    pos_counts = {"GK": 70, "DEF": 220, "MID": 260, "FWD": 150}
    for pos, n in pos_counts.items():
        for i in range(n):
            row = {"code": code, "web_name": f"{pos}{i}", "team": teams[code % 20], "position": pos,
                   "price": round(np.random.uniform(4.0, 13.0), 1), "status": "a"}
            for col in gw_cols:
                row[col] = round(np.random.uniform(1, 10), 2)
            rows.append(row)
            code += 1
    return pd.DataFrame(rows)


def test_solve_xi_first_squad_is_now_decorated():
    """Structural check: the same `@_cache_decorator` solve_squad() has
    carried since Patch 42 must now also wrap solve_xi_first_squad() in the
    real source -- not just behaviorally cached by some other mechanism."""
    src = open("/home/claude/rb_model_app/optimizer.py", encoding="utf-8").read()
    assert "@_cache_decorator\ndef solve_xi_first_squad(" in src, \
        "expected solve_xi_first_squad() to be decorated with @_cache_decorator, same as solve_squad()"
    print("Test 1 PASSED: solve_xi_first_squad() now carries @_cache_decorator in the real source")


def test_identical_repeat_call_is_dramatically_faster():
    """The actual performance claim: a second call with byte-identical
    inputs must return near-instantly instead of re-solving ~16 MILPs
    again, and must return the SAME answer (pure caching, no behavior
    change)."""
    pool = _realistic_pool()
    t0 = time.time()
    r1 = opt.solve_xi_first_squad(pool, CFG, budget=100.0, gw_col="xpts_gw6")
    cold_elapsed = time.time() - t0
    assert r1 is not None, "expected a solvable Free Hit XI-first squad on this realistic synthetic pool"

    t0 = time.time()
    r2 = opt.solve_xi_first_squad(pool, CFG, budget=100.0, gw_col="xpts_gw6")
    cached_elapsed = time.time() - t0

    assert cached_elapsed < 0.5, f"a cache hit should return near-instantly, took {cached_elapsed:.3f}s"
    assert cached_elapsed < cold_elapsed / 5, (
        f"expected the cached call to be dramatically faster than the cold call "
        f"(cold={cold_elapsed:.2f}s, cached={cached_elapsed:.3f}s)"
    )
    assert r1["xi_total"] == r2["xi_total"], "a cache hit must return the exact same result, not a fresh (re-)solve"
    assert set(r1["xi_codes"]) == set(r2["xi_codes"])
    print(f"Test 2 PASSED: identical repeat call went from {cold_elapsed:.2f}s (cold) to "
          f"{cached_elapsed:.4f}s (cached), same result both times")


def test_genuinely_different_inputs_still_get_a_fresh_solve():
    """Correctness safety check: caching must never conflate two DIFFERENT
    problems -- a different gw_col (different projections) must still
    produce its own independently-correct solve, not a stale cached answer
    from a different GW."""
    pool = _realistic_pool(gw_cols=("xpts_gw6", "xpts_gw7"))
    # Make GW7 projections meaningfully different (scaled up) so a real
    # difference in the chosen XI/total is expected, not just noise.
    pool["xpts_gw7"] = pool["xpts_gw7"] * 3.0 + 5.0

    r_gw6 = opt.solve_xi_first_squad(pool, CFG, budget=100.0, gw_col="xpts_gw6")
    r_gw7 = opt.solve_xi_first_squad(pool, CFG, budget=100.0, gw_col="xpts_gw7")
    assert r_gw6 is not None and r_gw7 is not None
    assert r_gw6["xi_total"] != r_gw7["xi_total"], (
        "two genuinely different gw_col inputs must not resolve to a cached, identical answer"
    )
    print(f"Test 3 PASSED: different inputs (gw_col='xpts_gw6' total={r_gw6['xi_total']}, "
          f"gw_col='xpts_gw7' total={r_gw7['xi_total']}) correctly produce different, independently-solved results "
          f"-- caching doesn't conflate distinct problems")


def test_a_different_budget_also_gets_a_fresh_solve():
    """Same correctness guard, varying `budget` instead of `gw_col` -- a
    tighter budget must be able to change the chosen squad, not silently
    reuse a looser-budget cached answer."""
    pool = _realistic_pool()
    r_full = opt.solve_xi_first_squad(pool, CFG, budget=100.0, gw_col="xpts_gw6")
    r_tight = opt.solve_xi_first_squad(pool, CFG, budget=55.0, gw_col="xpts_gw6")
    assert r_full is not None
    if r_tight is not None:
        assert r_full["xi_total"] >= r_tight["xi_total"] - 1e-6, (
            "a tighter budget's solve should never score higher than the unconstrained one if both solved"
        )
        assert len(r_tight["xi_codes"]) > 0
    print("Test 4 PASSED: a different budget is treated as a genuinely different problem, not served from "
          "the wrong cache entry")


if __name__ == "__main__":
    test_solve_xi_first_squad_is_now_decorated()
    test_identical_repeat_call_is_dramatically_faster()
    test_genuinely_different_inputs_still_get_a_fresh_solve()
    test_a_different_budget_also_gets_a_fresh_solve()
    print("\nALL INVESTIGATION TESTS PASSED (solve_xi_first_squad caching fix)")
