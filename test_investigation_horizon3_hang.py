"""
INVESTIGATION, NOT YET A NUMBERED PATCH (2026-09-28). Manager screenshot:
choosing Horizon=3 GWs with "Hit if worth it" left the app stuck on
"Fetching live data and computing xPts..." indefinitely, then the whole
Streamlit Cloud session crashed with a generic infra error ("Received no
response from server", not an app-level exception). Manager's explicit
instruction: identify the issue, recommend a fix, TEST it, and share the
result for confirmation before any new patch is packaged.

ROOT CAUSE, verified in code (see model_config.yaml's `solver:` section
comment and optimizer._cbc_solver()'s docstring for the full writeup):
  1. optimizer._cbc_solver() had NO wall-clock time limit at all -- every
     CBC solve runs with gapRel=0/gapAbs=0 (exact-optimality-or-bust) and,
     before this investigation, no `timeLimit`, so a single hard MILP
     instance could run indefinitely.
  2. recommend.suggest_transfers() with hit_stance in ("No hits", "Hit if
     worth it") and horizon_n > 1 routes to plan_transfer_schedule(), which
     loops once per GW in the Horizon slider's gw_list and, INSIDE each
     week, calls optimizer.solve_squad() once per k in
     range(1, k_upper + 1) -- k_upper = max(free_transfers_banked, 5) under
     "Hit if worth it" (the stance in the manager's screenshot). Horizon=1
     -> up to 5 solves. Horizon=3 (the reported case) -> up to 5 x 3 = 15
     sequential, independent, exact-optimality MILP solves in one run.
  3. Patch 78's own story already showed a realistic 343-candidate,
     14-of-15-retain solve terminating via CBC's harder-to-reach gap-check
     exit path -- i.e. solves of exactly this retain-constrained shape,
     repeated by this same loop, are already confirmed capable of being
     non-trivial, not the easy/instant case.

FIX BEING TESTED (not yet shipped as a patch): a disclosed, config-driven
wall-clock cap (`solver.time_limit_seconds` in model_config.yaml, default
12s) passed into every CBC solve via `_cbc_solver(time_limit=...)`. This
file verifies the fix mechanically and end-to-end, reusing this same
codebase's OWN precedented testing technique (test_patch78_gaplimit_fix.py
already monkeypatches the solve-status call to force a specific CBC status
deterministically, since a genuinely hard MILP instance can't be reliably
constructed on demand) rather than trying to organically reproduce a
multi-minute hang, which is inherently non-deterministic to engineer.

What was verified EMPIRICALLY against real installs (not assumed) before
writing these tests, documented in the investigation notes shared with the
manager:
  - pulp 3.3.2 (this sandbox's installed version, PULP_CBC_CMD path): a
    solve cut off by `timeLimit` before proving optimality returns
    prob.status == 0 -> pulp.LpStatus[0] == "Not Solved".
  - pulp 4.0.0 (installed separately under a python3.12 venv for this
    check, COIN_CMD path -- confirmed requirements.txt's unbounded
    `pulp[cbc]>=2.7` pin means Streamlit Cloud could install either):
    a solve cut off by `timeLimit` returns an LpSolveStats object with
    `status.name == "TimeLimit"`.
  Neither string is "Optimal" and neither is the disclosed-safe "GapLimit"
  exact-optimum case, so both must flow through the same
  `if _status != "Optimal": ... return None` graceful-degradation path
  every other non-Optimal status already uses -- confirmed below.
"""
import sys
import time

import numpy as np
import pandas as pd
import pulp
import yaml

sys.path.insert(0, "/home/claude/rb_model_app")
import optimizer as opt

CFG = yaml.safe_load(open("/home/claude/rb_model_app/model_config.yaml"))


def _tiny_squad_pool():
    """A trivially easy, fully synthetic candidate pool -- enough players in
    each position at varied prices to let solve_squad() legally fill
    2 GK/5 DEF/5 MID/3 FWD under a 100.0 budget with room to spare."""
    rows = []
    code = 0
    for pos, n, price_lo in [("GK", 6, 4.0), ("DEF", 12, 4.0), ("MID", 12, 4.5), ("FWD", 8, 4.5)]:
        for i in range(n):
            rows.append({"code": code, "web_name": f"{pos}{i}", "team": f"T{code % 10}",
                         "position": pos, "price": round(price_lo + i * 0.3, 1),
                         "xpts_horizon_sum": round(5.0 + i * 0.4, 2), "status": "a"})
            code += 1
    return pd.DataFrame(rows)


def test_config_time_limit_is_read_with_sane_default():
    limit = opt._solve_time_limit(CFG)
    assert limit == 12, f"expected model_config.yaml's solver.time_limit_seconds (12), got {limit}"
    # missing key falls back to the hardcoded default, never crashes
    assert opt._solve_time_limit({}) == opt._DEFAULT_SOLVE_TIME_LIMIT_SECONDS
    assert opt._solve_time_limit(None) == opt._DEFAULT_SOLVE_TIME_LIMIT_SECONDS
    print(f"Test 1 PASSED: _solve_time_limit() reads model_config.yaml's solver.time_limit_seconds "
          f"({limit}s) and degrades safely (never crashes) when the key or cfg itself is missing")


def test_cbc_solver_actually_sets_the_time_limit():
    """Confirms the constructed solver object genuinely carries the limit,
    not just that the parameter is accepted and silently dropped."""
    solver_default = opt._cbc_solver(msg=0)  # uses the module default (12s)
    solver_custom = opt._cbc_solver(msg=0, time_limit=3.5)
    solver_unbounded = opt._cbc_solver(msg=0, time_limit=None)
    assert getattr(solver_default, "timeLimit", "MISSING") == opt._DEFAULT_SOLVE_TIME_LIMIT_SECONDS
    assert getattr(solver_custom, "timeLimit", "MISSING") == 3.5
    assert getattr(solver_unbounded, "timeLimit", "MISSING") is None
    print("Test 2 PASSED: _cbc_solver() genuinely sets the solver object's timeLimit attribute "
          "(default, a custom value, and explicit None-for-unbounded all confirmed)")


def test_timelimit_status_is_not_normalized_to_optimal():
    """Same precedented technique test_patch78_gaplimit_fix.py already uses:
    monkeypatch prob.solve() to return the exact shape pulp 4.0's COIN_CMD
    returns on a real timeout (an object with status.name == 'TimeLimit'),
    and confirm _solve_and_get_status() reports it AS-IS -- unlike the
    disclosed-safe 'GapLimit' -> 'Optimal' normalization, a genuine
    'TimeLimit' must never be silently treated as a proven exact optimum."""
    class FakeStatus:
        name = "TimeLimit"

    class FakeResult:
        status = FakeStatus()

    class FakeProb:
        def solve(self, solver):
            return FakeResult()

    status = opt._solve_and_get_status(FakeProb(), None)
    assert status == "TimeLimit", f"expected 'TimeLimit' to pass through unnormalized, got {status!r}"
    print("Test 3 PASSED: a genuine 'TimeLimit' status is NOT silently normalized to 'Optimal' "
          "(unlike the disclosed-safe 'GapLimit' exact-optimum case) -- callers will correctly treat it "
          "as a real non-optimal outcome")


def test_pre_4_0_not_solved_status_also_not_normalized():
    """Mirrors the above for pulp <4.0's return shape (a plain int status
    code, decoded via pulp.LpStatus) -- confirmed empirically in this exact
    sandbox (pulp 3.3.2) that a real timeLimit timeout produces
    prob.status == 0 == 'Not Solved'."""
    class FakeProb:
        status = 0  # pulp.LpStatus[0] == "Not Solved"

        def solve(self, solver):
            return 0  # pre-4.0 solve() returns the same plain int

    status = opt._solve_and_get_status(FakeProb(), None)
    assert status == "Not Solved", f"expected 'Not Solved' to pass through unnormalized, got {status!r}"
    print("Test 4 PASSED: pulp <4.0's 'Not Solved' timeout status is also not silently normalized")


def test_solve_squad_degrades_gracefully_on_a_timed_out_solve():
    """End-to-end through the REAL solve_squad(), with only the final
    status-check monkeypatched to force the exact 'Not Solved' outcome a
    real timeLimit timeout produces (same technique as
    test_patch78_gaplimit_fix.py's Test 3) -- confirms the whole call chain
    degrades to None with a real diagnostic message, never crashes, and
    (this is the actual fix's point) never hangs: this call returns
    immediately regardless of how "hard" the underlying problem would have
    been, because the monkeypatch simulates the time limit having already
    fired."""
    import unittest.mock as mock
    pool = _tiny_squad_pool()
    squad_codes = pool["code"].tolist()[:15]

    with mock.patch.object(opt, "_solve_and_get_status", return_value="Not Solved"):
        t0 = time.time()
        result = opt.solve_squad(pool, CFG, budget=100.0, retain_pool_codes=squad_codes, min_retain=14,
                                  objective_col="xpts_horizon_sum", label="investigation_test")
        elapsed = time.time() - t0
    assert result is None, "a timed-out (non-Optimal) solve must degrade to None, not a partial/fabricated squad"
    assert elapsed < 2.0, f"this must return immediately (no hang) -- took {elapsed:.2f}s"
    diag = opt.get_diagnostic("investigation_test")
    assert diag and "Not Solved" in diag, f"expected the diagnostic to name the real status, got {diag!r}"
    print(f"Test 5 PASSED: solve_squad() degrades gracefully (returns None, real diagnostic, no hang -- "
          f"{elapsed:.3f}s) when the underlying solve reports a timed-out status")


def test_ordinary_solve_is_unaffected_by_the_new_default_time_limit():
    """Regression guard: a normal, easily-solvable problem must still solve
    correctly (not None, a real legal 15-man squad) well within the new
    12-second default cap -- the fix must not degrade ordinary operation."""
    pool = _tiny_squad_pool()
    t0 = time.time()
    result = opt.solve_squad(pool, CFG, objective_col="xpts_horizon_sum")
    elapsed = time.time() - t0
    assert result is not None, "an easy, unconstrained-by-retain problem must still solve successfully"
    assert len(result["squad"]) == 15
    assert elapsed < 5.0, f"an easy problem should resolve almost instantly, took {elapsed:.2f}s"
    print(f"Test 6 PASSED: an ordinary solve still succeeds correctly and quickly ({elapsed:.3f}s) "
          f"with the new 12s default cap in place -- no regression for the normal case")


def test_no_call_site_still_constructs_an_unbounded_solver_by_accident():
    """Static guard: every _cbc_solver(...) call in optimizer.py must pass a
    time_limit derived from cfg (or rely on the module default), never a
    bare _cbc_solver(msg=0) with no limit at all -- catches a future call
    site silently regressing back to the pre-fix unbounded behavior."""
    src = open("/home/claude/rb_model_app/optimizer.py", encoding="utf-8").read()
    import re
    bare_calls = re.findall(r"_cbc_solver\(msg=0\)(?!,)", src)
    # every remaining call must include time_limit= explicitly (the module
    # default on the function signature itself still applies even without
    # it, but this project's own standing practice is to make the config
    # wiring explicit at each call site rather than rely silently on a
    # parameter default -- confirmed no bare calls remain).
    assert not bare_calls, f"found {len(bare_calls)} _cbc_solver(msg=0) call(s) with no time_limit passed explicitly"
    with_limit = re.findall(r"_cbc_solver\(msg=0, time_limit=_solve_time_limit\(cfg\)\)", src)
    assert len(with_limit) == 3, f"expected all 3 known call sites to pass time_limit=_solve_time_limit(cfg), found {len(with_limit)}"
    print(f"Test 7 PASSED: all {len(with_limit)} _cbc_solver call sites in optimizer.py explicitly wire "
          f"the configured time limit -- none silently unbounded")


if __name__ == "__main__":
    test_config_time_limit_is_read_with_sane_default()
    test_cbc_solver_actually_sets_the_time_limit()
    test_timelimit_status_is_not_normalized_to_optimal()
    test_pre_4_0_not_solved_status_also_not_normalized()
    test_solve_squad_degrades_gracefully_on_a_timed_out_solve()
    test_ordinary_solve_is_unaffected_by_the_new_default_time_limit()
    test_no_call_site_still_constructs_an_unbounded_solver_by_accident()
    print("\nALL INVESTIGATION TESTS PASSED (Horizon=3 hang / unbounded-solve fix)")
