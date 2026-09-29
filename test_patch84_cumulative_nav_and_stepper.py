"""
Patch 84 -- two fixes to app.py's `_render_pitch_navigator()`, both found
from a manager screenshot (paged the pitch to GW7 with "After recommended
transfer (this week's move)" selected; the pitch still showed Gomez despite
the Transfer Recommendations panel saying GW7's chained move was
Gomez -> Groß) plus one bug found while verifying the fix live via
Playwright (not manager-reported):

  1. THE REPORTED BUG: the "After recommended transfer" toggle only ever
     reconstructed `weekly_plan[0]`'s move (the CURRENT planning GW's move)
     into a single static squad, and every later GW the stepper paged to
     reused that same static squad -- only the xPts PROJECTION column
     changed, never the squad's actual player composition. Fixed by
     reconstructing a per-GW CUMULATIVE squad (`_nav_squad_after_by_gw`),
     replaying weekly_plan's moves in order, so paging to GW7 now stacks
     GW7's move on top of GW6's.

  2. A SEPARATE BUG found while Playwright-driving the fix above (not in the
     manager's report): the ▶ (next) button's session_state mutation ran
     AFTER the GW counter label had already rendered, so clicking ▶ showed
     the CORRECT new GW's xPts/Rating/squad below, but the counter text
     between the two buttons lagged one click behind (e.g. displayed
     "GW6 (1/3)" while the metrics already said "GW7 xPts"/"GW7 Rating").
     The ◀ button never had this problem (its handler already ran before
     the counter rendered) -- only ▶ was affected. Fixed by resolving both
     buttons' clicks before rendering the counter.

These two fixes live inside a Streamlit `@st.fragment`-decorated closure
with dozens of enclosing variables (squad_df_adv, chip_adv_proj, rec,
planning_gw, ...), so they can't be unit-tested by importing a bare
function the way most other test_patch*.py files in this project do.
Instead:
  - `test_apply_moves_and_cumulative_reconstruction_logic()` below
    reproduces the EXACT `_apply_moves()` + per-GW-cumulative-dict
    algorithm now in app.py (same operations, same order) against
    synthetic squad/pool DataFrames, and checks the properties that matter:
    a later week's move stacks on an earlier week's, a "Roll" (no-move)
    week carries the prior week's squad forward unchanged, and an
    unaffected earlier GW in "Current squad" mode isn't touched by any of
    this.
  - The REAL, live-in-app behavior (both fixes, wired together, rendered by
    the actual app.py through actual Streamlit + Playwright, GW6 -> GW7 ->
    GW8) is verified separately by test_patch84_e2e_harness.py /
    test_patch84_streamlit_driver.py / test_patch84_playwright_drive.py
    (not shipped in the patch zip, matching this project's driver/harness
    convention) -- see the Patch 84 changelog entry in model_config.yaml
    and PATCH_VERSION in app.py for that run's actual captured output.
"""
import pandas as pd


def _apply_moves(base_squad: pd.DataFrame, moves: list, pool: pd.DataFrame) -> pd.DataFrame:
    """Verbatim copy of the algorithm now in app.py's `_apply_moves()`
    closure (same operations in the same order) -- see this file's
    docstring for why this can't just be imported and called directly."""
    if not moves:
        return base_squad
    moves_df = pd.DataFrame(moves)
    if "out_code" not in moves_df.columns or "in_code" not in moves_df.columns:
        return base_squad
    out_codes = set(moves_df["out_code"])
    in_codes = set(moves_df["in_code"])
    result = pd.concat([
        base_squad[~base_squad["code"].isin(out_codes)],
        pool[pool["code"].isin(in_codes)],
    ], ignore_index=True, sort=False)
    if "code" in result.columns:
        result = result.drop_duplicates(subset=["code"], keep="first")
    return result


def _build_cumulative_by_gw(squad_df_adv: pd.DataFrame, chip_adv_proj: pd.DataFrame, weekly_plan: list) -> dict:
    """Verbatim copy of the per-GW cumulative reconstruction loop now in
    app.py's `_render_pitch_navigator()`."""
    by_gw = {}
    running = squad_df_adv
    for wk in weekly_plan:
        wk_moves = wk.get("moves") or []
        if wk_moves:
            running = _apply_moves(running, wk_moves, chip_adv_proj)
        by_gw[wk.get("gw")] = running
    return by_gw


def test_apply_moves_and_cumulative_reconstruction_logic():
    squad = pd.DataFrame({"code": [1, 2, 3, 4, 5], "web_name": ["A", "B", "C", "D", "E"]})
    pool = pd.DataFrame({"code": [1, 2, 3, 4, 5, 10, 11, 12],
                         "web_name": ["A", "B", "C", "D", "E", "In10", "In11", "In12"]})
    weekly_plan = [
        {"gw": 6, "moves": [{"out_code": 5, "in_code": 10}]},   # E -> In10
        {"gw": 7, "moves": [{"out_code": 4, "in_code": 11}]},   # D -> In11 (stacks on GW6's move)
        {"gw": 8, "moves": []},                                  # Roll -- no change
        {"gw": 9, "moves": [{"out_code": 10, "in_code": 12}]},  # In10 (bought GW6) sold, In12 bought
    ]
    by_gw = _build_cumulative_by_gw(squad, pool, weekly_plan)

    gw6_codes = set(by_gw[6]["code"])
    assert gw6_codes == {1, 2, 3, 4, 10}, f"GW6: E should be OUT, In10 IN -- got {gw6_codes}"

    gw7_codes = set(by_gw[7]["code"])
    assert gw7_codes == {1, 2, 3, 10, 11}, (
        f"GW7: BOTH GW6's move (E->In10) and GW7's move (D->In11) must be stacked -- got {gw7_codes}"
    )
    assert 5 not in gw7_codes and 4 not in gw7_codes, "both original OUT players must be gone by GW7"

    gw8_codes = set(by_gw[8]["code"])
    assert gw8_codes == gw7_codes, (
        f"GW8 is a Roll week (no moves) -- it must carry GW7's squad forward UNCHANGED, "
        f"got GW7={gw7_codes} vs GW8={gw8_codes}"
    )

    gw9_codes = set(by_gw[9]["code"])
    assert gw9_codes == {1, 2, 3, 11, 12}, (
        f"GW9: In10 (bought GW6) must be sold and In12 bought, net of everything before it -- got {gw9_codes}"
    )
    assert 10 not in gw9_codes, "In10 must be gone by GW9 (sold that week) despite being IN at GW6/7/8"

    # "Current squad" mode (no moves applied at all) must remain the
    # original, completely untouched by any of the above reconstruction.
    assert set(squad["code"]) == {1, 2, 3, 4, 5}, "original squad_df_adv must never be mutated in place"

    print("Patch 84 PASSED: per-GW cumulative reconstruction stacks chained moves correctly, "
          "carries a Roll week forward unchanged, correctly nets a later buy-then-sell of the same "
          "player, and never mutates the original squad DataFrame")


if __name__ == "__main__":
    test_apply_moves_and_cumulative_reconstruction_logic()
    print("\nALL PATCH 84 TESTS PASSED")
