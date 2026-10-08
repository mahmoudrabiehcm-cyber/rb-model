"""
app.py — RB Model
Streamlit front end for the FPL Projection Model v5.0. Zero-cost: official
FPL API (free, no key), Streamlit Community Cloud (free, public apps),
Google Fonts (free). See DEPLOY.md for the full deploy walkthrough and
README.md for how the pieces fit together.

Run locally:  streamlit run app.py
"""
from __future__ import annotations
import datetime as dt
import html
import os
import re

# Patch 69 — altair, used only for the Season Rank chart's reversed y-axis
# (st.line_chart itself has no axis-reverse option). Not a new install cost:
# Streamlit has depended on altair for its own native charting (st.line_
# chart/st.area_chart/etc.) since well before this app existed, so it is
# already present in every environment that can run streamlit at all.
import altair as alt
import pandas as pd
import streamlit as st

import fpl_data
import fpl_engine as eng
import data_pipeline
import optimizer as opt
import style_profiles
import chip_protocol
import transfers
import recommend

# Patch 38 (2026-09-14, manager report: "we spent the whole day explaining
# the logic and still the same issue" — the real cause across that whole day
# was never being able to tell, from a screenshot alone, whether a fix had
# actually been redeployed or whether a number was legitimately different
# live data): a permanent, visible version stamp so that question is
# answerable at a glance, without another round of screenshots. Bump this
# with every patch that ships to the manager.
PATCH_VERSION = ("Patch 117j (2026-10-08, manager request): new sidebar box 'Excluded players' (like Locked players, for players you do NOT own): an excluded player is never bought by the weekly transfer planner (free or hit moves) and never picked in a Wildcard rebuild or its fallback ladder. Ceilings, the Free Hit optimum and the Rating % ignore it (same scope as Locked players). Included in the run snapshot. Display note: it also changes the Wildcard decision gain because it changes the rebuilt squad. NOT validated in a browser. " "Patch 117i (2026-10-08, model M9/M29, manager go): chip value (Triple Captain / Bench Boost / Free Hit) is no longer counted INSIDE the Wildcard decision (include_chip_value false): PLAY/HOLD, the week, the tie band and the 4-week gain use the PLAIN chain gain; the chip effect is still computed and shown beside it, labelled not counted. The chip-aware Wildcard variant is off (wc_chip_aware_variant false) until the model chat's variant test is built. Chips are still placed afterwards on the squads the plan fields (unchanged). Verified on the manager's GW6 Extended snapshot: Wildcard GW6 unchanged, 4-week plain gain +36.1 (was +34.7 with chips -1.4 inside), chips -1.4 shown beside. NOT changed: M8a stronger baseline, accrual fix (after the deadline). NOT validated in a browser. " "Patch 117h (2026-10-08, manager: Wildcard decision review before the deadline; display + ceiling only, no change to any decision rule): (1) the best-possible-squad ceiling also tries the XI+captain+bench solver at bench weights 0.08/0.15/0.25 (the rating counts a bench slot at about 0.1-0.3, not a flat 0.08) and keeps the best by the rating's own scoring. (2) the chip table keeps the raw % beside the 100 cap and says how many weeks were capped and the highest raw value. (3) a label under the chip table: the 'Wildcard value if played alone' scan column under-accrues transfers from GW7 (known, fix after the deadline). (4) the '4-week gain decides' chip lists the chip value inside each week's figure. Chip value is still INSIDE the decision (include_chip_value true) - the manager has not yet decided D2 (model chat M9 says outside). Verified on the manager's GW6 snapshot with Extended mode on: Wildcard GW6 +34.7, tie band 5.4, unchanged. NOT validated in a browser. " "Patch 117g (2026-10-08, manager: same visuals everywhere; Pitch rating too high): (1) ONE best-possible-squad per GW (data_pipeline.solve_best_gw_squad = best of XI-first, 15-man sum and an XI+captain+bench solver, all scored by the same realized-value rule) now feeds the Pitch rating, the header rating and the chip-table yardstick; live GW6 snapshot: header rating 84.1% -> 79.6%, Wildcard path squad 99.8% (was 100.6% capped); locks still ignored by the ceiling. (2) every table goes through one formatter (money as GBP 5.0m, xPts to 1 decimal, totals as bars). NOT validated in a browser. " "Patch 117f (2026-10-08, manager review of the 117e screens: polish, display only): chips sit in one row in the Wildcard rebuild expander; the remaining text walls (why-this-week notes, stale-data, rating no-result, chip clash, reconcile notes) are chips with a tap for detail; Transfers move notes fold into one tap line; news shows a chance-of-playing bar instead of repeating the text, timestamp inline; Wildcard tables show money as £5.0m and xPts to 1 decimal; card sub-text clamped to 2 lines with full text on hover; rating bar narrower. No numbers or decisions changed. NOT validated on live data. " "Patch 117e (2026-10-08, manager: minimal words, more visuals on all screens): display only - 40 long captions/notes/warnings on every tab are now a short coloured chip with a tap-for-details arrow (full text kept inside); a rating bar above the GW rating. No numbers or decisions changed. NOT validated on live data. " "Patch 117d (2026-10-08, manager: owned players must cost what you receive, a Locked players list, one chained plan, shorter messages): (1) every solve now prices your owned players at their SELLING price (live price kept as market_price) with your real bank, so keeping a risen player costs what you would receive, selling returns it, buying costs the live price; Wildcard budget unchanged (99.8 this week). (2) sidebar 'Locked players': the Wildcard rebuild and the weekly transfer planner must keep them and never sell them; ceilings, the Rating % and the Free Hit optimum ignore locks; a chip shows what the locks cost in xPts on the Wildcard card. (3) after the Wildcard decision the weekly plan is rebuilt on the decided week (Wildcard now = the plain list is the no-Wildcard alternative), the Pitch opens on the Chip Plan path when the Wildcard is now. (4) sidebar 'Download run snapshot' (pool, prices, flags, config, sheet) so a live run can be reproduced offline. (5) budget, plan-conflict and lock notes are short coloured chips. NOT validated on live data (FPL API blocked here); the first-week move of non-Wildcard chain paths still comes from the first plan. " "Patch 117c (2026-10-08, manager: type the budget by hand is not wanted; badge showed 117 for 117b): (1) the budget is now read automatically: bank + the selling price of each owned player, worked out from your FPL transfer history (latest purchase price; start-of-season price if never transferred in; Free Hit weeks ignored; half of any price rise kept, rounded down to 0.1). Buying still uses live prices. If the feed cannot be read the app falls back to market prices and says so on the Transfers tab. The Budget box moved into Advanced and is only a manual override (0 = auto). (2) version badge keeps the letter (Patch 117c). (3) Wildcard rebuild table label says 15-man squad total, not Starting XI. NOT validated on live data here (FPL API blocked from the sandbox); rule checked on the manager's own figures. " "Patch 117b (2026-10-07, manager: Wildcard squad 0.7m over budget; transfer plan said Wildcard GW8 while the Chip Plan said GW6): (1) sidebar box 'Budget £m' - type squad value + bank from the FPL Transfers page; it replaces the app's market-price budget for every build (Wildcard, Free Hit, chain), 0 = old behaviour; a single sale still uses the sold player's market price (approximate); selling prices are NOT read from the FPL site. (2) the Transfers tab now warns when the weekly plan's Wildcard week (chip portfolio / scan) differs from the Chip Plan decision (chain) - display only, the plan itself is not rebuilt. NOT changed: Bogle-type small-sample goal rates (model-chat question), clean sheet values. 492 tests pass; live browser run not possible here. " "Patch 117a (2026-10-07, model chat GW6 run vs app screens; display and window fixes, no change to the Wildcard decision logic): (1) provisional label now cites Rule #48(c) (was Rule #53(d)). (2) weekly-plan move cards: the -4 hit badge goes on the lowest-gain move(s), FREE on the highest (was FREE on every card even when the plan paid a hit); set-level gate unchanged. (3) Pitch and GW Rating capped at 100% with a flag showing the raw value when the ceiling solve returns a squad below yours. (4) Wildcard trigger window 5 -> 4 GWs (Rule #48); Extended runs still project 10 GWs but the trigger reads only the first 4. (5) MONITOR says it uses the plan's full transfer allowance. (6) cards say clean sheets with a manual override are carried at the sheet value for every week. (7) when the scan-only best week differs from the planned week it is shown as a labelled low-confidence alternative with the accrual defect named. NOT changed (switch left as is): chip value inside the Wildcard decision (include_chip_value). NOT validated: synthetic data; 486 tests pass, browser driver tests not runnable in this sandbox. " "Patch 117 (2026-10-05, model ruling amending v6.12 Rules #48(a) / #34 after the Patch 116 screen: GW6 +31.6, GW7 +26.6 but the card said 'waiting beats playing now'): (1) cap: inside the tie band the later week wins only if waiting costs no more than half the band (cost = best week's value minus the later week's, on the deciding measure; wc_wait_cap_frac 0.5, a judgement with no backtest); otherwise the best-value week is the plan, tie set shown, low confidence. (2) waiting budget: the cost of consecutive deferrals since the trigger fired (trigger log columns wc_wait_cost / wc_deferred) must stay within the band; on Streamlit Cloud the log is on an ephemeral disk, so a reboot resets the budget to 0. (3) deciding measure: the four-gameweek plain gain + chip value (wc_decide_weeks 4); the band is built on the same four-week total; the six-week decay-weighted gain is shown beside it and no longer decides, so the 0.9 decay no longer moves the week (decay check is skipped). (4) The card says 'waiting beats playing now' only when the later week really scores higher; the cards read 'Plan GWn, provisional' and pending news is never a reason to defer (Rule #53(d)). (5) GUARDRAIL: the checkpoint gameweek is named, HOLD names its source (value floor or guardrail), the rejected best week and reason are shown, and a passing week outside the tie set is context only. Rule #48(a) stays the cited rule. NOT validated: sandbox/synthetic data; half-band cap and band-sized budget are the model's judgement, not backtested. " "Patch 116 (2026-10-05): re-issue of Patch 115 + fix + fix 2 under a new number so the repo is unambiguous (the host stalled while starting the Patch 115 upload; the same files start fine in a clean folder, peak memory 305 MB with the Extended Check, so the cause was NOT found in the code). Contents are exactly the three blocks below; upload ALL 17 core files together. " "Patch 115 FIX 2 (2026-10-05): (1) decay sensitivity - the Wildcard week is re-decided at decay 0.85 and 0.95 from the same solved squads (config chip_extended_check.decay_check); a flip marks the verdict low confidence, the pick is unchanged. (2) planner_captain switch (transfer.planner_captain, default OFF = Rule #31 unchanged) so the planner-captain question (Rule #31 vs #54) can be measured instead of argued. " "Patch 115 FIX (2026-10-05, live failure: Wildcard chain comparison unavailable / HOLD with no week): cause = the captain term made each Wildcard rebuild ~3x slower and four ran in parallel past the 12 s solver cap, so no squad came back and the column was dropped silently. Fixed: (a) captain is exact but only for the top 60 scorers per week (wc_build_captain_k; same squad and total on 3 test pools); (b) parallel solves capped to the cores (parallel_workers); (c) ladder weekly XI + captain -> weekly XI without captain (labelled 'Rule #54(a) not met') -> unavailable; NEVER the plain 15-man sum; (d) the tie band is Rule #34 on the four-gameweek total (wc_band_weeks 4, about 5), as v6.12 says, not on the 6-week total; (e) when no comparison exists the card now says WHY (unavailable reason, Rule #54 ladder) instead of a bare HOLD; (f) the decision shows the cost of waiting one more week. OPEN for the model chat: the planner objective (Rule #31) still excludes the captain, while Rule #54 asks the no-chip comparison path to include it - disclosed, not changed; band window 4 vs 6 weeks, guardrail 'none passes' reading, decay 0.9, Rule #53(a) ceiling. captain_k 60 is estimate-tier. " "Patch 115 (2026-10-05, model v6.12 rulings on Patches 111-114): (1) TRIPLE CAPTAIN / BENCH BOOST: planned week = best week in the horizon; EXACT ties now go to the LATER week (was earlier); the tie set (weeks inside Rule #34's band of the best, one-week compared horizon = the 2-point floor) is shown and the pick is flagged low confidence; switch pick_rule (best_horizon_earlier = Patch 113, latest_tied = Patch 112). (2) WILDCARD WEEK: ties use Rule #34's band over the compared window total (greater of 2 points or 2%, about 5; the old fixed 2.0 is only its floor) and the LATER week inside it wins; a later shift vs the previous logged run is a Rule #11 reversal (warning + trigger-log columns wc_shift / wc_band). (3) WINDOW: every Wildcard candidate is valued over the SAME number of post-chip weeks (min(post_chip_weeks 6, fewest weeks left), never below 4), decay-weighted (post_chip_decay 0.9, estimate-tier); a candidate without a full window is NOT scored; weeks beyond Current+5 flagged fallback-tier (no market-odds feed is wired, so every week is fallback-tier CS%); the Rule #48 four-GW value is always shown as a cross-check and a disagreement is reported. (4) BUILD OBJECTIVE (new Rule #54): weekly XI + each week's CAPTAIN (exact, continuous c_ig variables) + bench 0.08 (1.0 in the BB week) + decay weights 0.9 (wc_build_decay, estimate-tier) actually passed to the solve; switches wc_build_captain / wc_build_decay / wc_objective. Cost: sandbox 700-player 8-week rebuild 3.2s -> 5.8s (candidates solve in parallel). (5) TC TARGET IN THE PLANNER: the premium (the named captain's points in the TC week, counted once) is used only when TC is scheduled in the Rule #49 harmonized assignment, in THAT week (tc_week_source: schedule; chain = Patch 114); the ranking without it (plain) and with it is shown; a flip inside the band stays chip-agnostic; hits: the chain planner runs hit_stance 'No hits' so Rule #13's 6.0 bar cannot be crossed by the premium. (6) SQUAD-HEALTH GUARDRAIL (Rule #52) on the chain comparison (chain_guardrail): in-band weeks whose checkpoint xPts fall more than the band below the healthiest in-band week, or below the no-chip path, are rejected and listed with their gain and reason; none passes -> HOLD. (7) Rule #48 SCAN ('Wildcard value if played alone') now uses the weekly-XI objective (wc_scan_objective; sum15 = Patch 114), candidates solved in parallel. HOUSEKEEPING: Chip Timing Harmony relabelled Rule #52 in code comments and history text; cap_use_bar and value_tail_gws are defined in model_config.yaml and tagged estimate-tier, and every Wildcard verdict says so. NOT validated: synthetic/sandbox data only; model v6.12 document not available in the project files (v6.11 PDF only) - rulings applied as written in the manager's message. " "Patch 114 (2026-10-05, manager review of the Patch 113 screens: Bench Boost squad was the Wildcard squad): (1) WILDCARD REBUILD = the best team over its 8-week window: the squad and a legal starting XI for EACH week are chosen together (XI in full, bench at the planner's own bench weight 0.08), instead of the Patch 113 plain 15-man sum that counted every player as a starter every week. MEASURED before shipping on 3 sandbox scenarios (GW6-13): the new build beats the 15-man sum on realized value in all 3 (+2.7 to +6.7 xPts); a first idea (one fixed XI for the whole window) was WORSE in 2 of 3 and was dropped, so the earlier claim that the old squad wasted budget on the bench was not supported. Not inside the solve (applied when the squad is scored): captain doubling and the autosub curve. Switch chip_extended_check.wc_objective (sum15 = Patch 113). (2) CHIPS SHAPE THE TRANSFERS (Extended Check): for the best Wildcard week the model builds the plain squad and, if a better Triple Captain target exists in the TC week, two variants - chip-aware (Bench Boost week bench + the target's tripled week inside the Wildcard build) and FT-route (the weekly planner may BUY the target with a free transfer) - and keeps one only if it beats the plain squad by more than the margin of error; with no Wildcard the planner can still buy the target (base path). The Wildcard rebuild and Triple Captain cards say which variant won and when the target is bought. Switches wc_chip_aware_variant, tc_ft_route. DEVIATIONS for the model chat: Rule #31 (TC extra counted by the planner; same family as Patch 112/113 chip value). Speed: candidate rebuilds solve in parallel; sandbox chain (700 players, 4 Wildcard candidates, cold) 8.5s -> 11.9s, cached afterwards; variants add ~4s only in the Extended Check and only when a better TC target exists; NOT live-measured. " "Patch 113 (2026-10-04, manager review of the Patch 112 screens; chips + captaincy display only, planner unchanged): (1) TRIPLE CAPTAIN / BENCH BOOST WEEK = the best week across the whole horizon (Patch 112 moved ties to the LATEST week, which is why the TC card said GW16 while its caption said GW8 was best); exact ties go to the earlier week; a near-tie is shown as low confidence instead of moving the chip; switch chip_extended_check.pick_rule (latest_tied = Patch 112). (2) TC and BB are chosen as a PAIR (best combined value, different weeks) instead of BB first. (3) The Triple Captain names ONE player in ONE week, read by the Chip Plan headline, the Transfer page card and the Captaincy tab. (4) Captaincy tab: new table - armband NOW (current squad), AFTER the Wildcard, and the Triple Captain week (tripled). (5) A first-half chip is picked inside its earliest still-available window (it lapses at the deadline). (6) Caption punctuation fixed (no more \"GW16.;\"); one sentence builder recommend.chip_edge_text. DEVIATIONS for the model chat: Rule #31 (chip value counted; unchanged from Patch 112), Rule #34/#49 tie deferral replaced by best-week-in-horizon for TC/BB (switchable). NOT validated: sandbox/synthetic data only. " "Patch 112 (2026-10-04, manager decisions after Patch 111; chain comparison only, planner defaults unchanged): (1) at the 5-FT cap the chain now USES a transfer when its gain clears cap_use_bar 0.25 xPts instead of letting it be lost; (2) late-week moves are valued 3 GWs past the shown span (value_tail_gws); (3) the Free Hit week is excluded from the chain gain sum; (4) timeline shows banked FTs per week; (5) GW reconciliation line under the table (current squad vs after-move vs reference xPts) because the Pitch Rating and the table % use different bases. CHIPS IN THE WILDCARD DECISION (manager instruction; DEVIATES from Standing Rule #31 -> model chat must approve; switch: chip_extended_check.include_chip_value): each Wildcard candidate's gain = chain xPts gain + (Triple Captain + Bench Boost + Free Hit value on that chain's squads minus the same on the no-chip chain); TC = best-XI top scorer xPts (extra x1), BB = bench xPts, FH = gap to the GW-only optimum at the sequence's FH week; a chip cannot share the Wildcard/FH week. SCREEN REVIEW FIXES: a scheduled chip worth less than the 2.0 floor (Free Hit +0.0) is now HELD instead of 'PLAY'; a chip tie resolved to the latest week is shown as 'no real edge' (no more '-0.2 clear'); the chip-table % uses the better of two per-GW optima and is capped at 100 (it read 101-103%); chip/Wildcard tables show all rows. FOLLOW-UPS: the Captaincy tab picks the armband from the Wildcard squad when the plan says Wildcard this GW; chip ties (within the margin of error) go to the LATEST week (Rule #34/#49 deferral; the first cut took the earliest); every % in the chip table is now measured against the best possible squad for THAT GW (the Pitch tab's yardstick) instead of one fixed squad for the span (which flattered a static squad to ~100%); the decided Wildcard path is re-planned once when the plan's Bench Boost week differs from the one the transfers were prepared for (replan_for_bb, +~2s). ONE LINKED PLAN: Triple Captain / Bench Boost / Free Hit weeks are now picked on the squads the plan fields each week (PLAIN Wildcard rebuild + later free transfers; chain squads carried to the chip window's end) and written into the same chip schedule the cards, Chip Plan, Transfer page and Pitch read (DEVIATION from Rule #49b's 4-GW rebuild without later transfers -> model chat); Bench Boost / Triple Captain tables now show that squad (they printed the CURRENT squad before); the Wildcard is decided and scored PLAIN, style shown as a list of swaps with xPts cost (not applied); Pitch navigator gains a 'Chip Plan path' squad mode + chip badge; Transfer list carries the Wildcard decision. Free Hit week still from the joint sequence (per-week optimum too slow). LINKS: Transfer page Free Hit/Bench Boost/Triple Captain now read the Chip Plan's joint sequence (they read standalone advisors before: Free Hit GW9 vs GW13); the Wildcard rebuild expander shows the chain's own squad for the decision week (HOLD = best candidate as reference); chips are re-picked on the chosen Wildcard week's squad. cap_use_bar and value_tail_gws are NOT validated. Speed: lossless caches (best_starting_xi rewrite 24ms to 1.2ms per call, plan-solve memo), golden-parity tested vs Patch 108; sandbox extended chain ~7s, normal ~3.9s, synthetic data, NOT live-measured. " "Patch 111 (2026-10-04, manager decisions after Patch 110): (1) WILDCARD CHAIN COMPARISON replaces the Patch 110 94% bridge rule — the table's % was the NO-CHIP path (verified: _wc_bridge_calc ran with chip_schedule=None); it now shows No chip % beside With Wildcard GWn %, and the Wildcard week is the candidate (current GW, sequence GW, top Rule #48 weeks) with the biggest summed xPts gain over the whole checked span (free transfers only, no hits, chains start from your recommended GW move, Wildcard team rebuilt for 8 GWs); earliest week within the 2.0 margin of error of the best wins, best gain below the 2.0 materiality bar = HOLD. Thresholds borrowed, NOT validated; Rule #48's own 4-GW window value stays visible as \"if played alone\". Free Hit/Bench Boost/Triple Captain are not modelled inside the chain. SYNTHETIC-squad tested only. (2) GW-by-GW plan timeline + gain-per-candidate bars. (3) PERFORMANCE — chain plans skip the tie-break scan (93% of planner time); optimizer.realized_gw_value is memoised (89% of calls were exact repeats; lossless, golden-parity tested vs Patch 108); the hit-inclusive 10-GW cross-check no longer runs in the Extended Check. Sandbox: extended chain (10 GWs, 4 Wildcard candidates) 7s, normal 3.4s; NOT live-measured. (4) Fixes: \"Done\" message named GW15 while the button said GW17; latent KeyError in recommend._position_tie_break when the model's own pick sat outside the tied set. "
                  "Patch 110 (2026-10-04, manager decisions after Patch 109): (1) ONE LINKED WILDCARD DECISION — the card, the top pill and the best-GW table now read the same object (recommend.wildcard_final_decision); verified cause of the old mismatch: the card badge came from the hit-inclusive \"transfers close the gap\" check, the table's Chips column from the joint sequence, and the \"Best Wildcard GW\" headline from the isolated Rule #48 window scan. (2) BRIDGE TEST — free transfers only (no hits), flagged players benched, squad vs a full-rebuild squad each GW: HOLD -> GWn while it stays at/above 94% (the document's 6% gap reused, UNVALIDATED), PLAY at the first GW that starts 2 consecutive weeks below it, PLAY NOW if that is the current GW. Prototype on SYNTHETIC squads only: compared against the reachable ceiling it could not tell a dead team from a healthy one (plan and ceiling share the same transfer budget); compared against a full rebuild it separated them. The 88%-style trigger number stays as a label (today). (3) Extended Check now reaches Wildcard Current+9 and BB/TC/FH Current+11 (button \"Extended Check GW{Current+11}\"). (4) Table: Wildcard value column renamed \"if played alone\"; Rating % column replaced by the bridge rating; Wildcard week shown only at the decision GW. "
                  "Patch 109 (2026-10-04): Extended check ran >6 minutes live. Measured: ~93% of the weekly-plan cost was pandas re-sorting inside optimizer.best_starting_xi and the bench-autosub loop (via recommend._position_tie_break), not the solver; both rewritten losslessly (parity-tested vs the Patch 108 code). UI: the extend button is now just \"Extended Check GW{n}\"; the intro/caption text and the calibration-log expander are removed (the log still records silently). "
                  "Patch 108 (2026-10-04): (1) LIVE CRASH FIX — team 4984023 \"Spurs\" (Wildcard already used) hit \"NameError: _wc_extend_chip_driven is not defined\": the always-visible extend-button section read variables that were only assigned inside `if wc_flag and reachable_by_gw ...`, which is skipped for any team whose Wildcard trigger isn't evaluated. Present in deployed Patch 106 (and 107). All such names are now bound before the block; the confirmation also handles teams with no Wildcard cross-check. (2) Wildcard card now leads with the DECISION — PLAY GWn / MONITOR / HOLD plus net xPts vs best transfers — instead of a bare alarm %; the 6% trigger (Standing Rule #45) is unchanged. (3) Wildcard calibration log (CSV, one row per normal run + download) so the document-flagged \"unvalidated\" 6% can be judged from real weeks; the server disk is temporary, so download regularly. (4) Less text, more visual: shorter pill and intro, per-GW table with a bar for Wildcard value. Includes everything in Patch 107. Previously, "
                  "Patch 107 (2026-10-04, manager decisions after discussion: button must be dynamic \"Current GW + 9\"; the extended run must confirm the best GW between Current and Current+9 and re-evaluate ALL chips, not only the Wildcard text; performance must be optimized): (1) PERFORMANCE — profiled (cProfile) and found ~93% of every optimizer.solve_squad() call was Python-side MILP construction (a pandas .loc lookup per player per constraint), not CBC; solve_squad() and solve_xi_first_squad() now read each column once into lists. Lossless: golden results captured from the OLD code (26 solves) match exactly (golden_solver_parity.json + test). Sandbox timing, chip stages only: 5-GW window 12.8s -> 3.1s, 10-GW window 23.4s -> 6.2s; Free Hit solver 31s -> 2.9s for 8 calls; full test suite ~55s -> ~28s. Threading was tested and gave no gain, so not used. NOT live-measured. (2) EXTENDED MODE — clicking the button now re-runs the whole Chip Plan over Current..Current+9 (chip advisor window and Wildcard detection window widened to chip_extended_check.span_gws=10; 3 trailing GWs projected so late Wildcard candidates get a full 4-GW window instead of a truncated 1-3 GW one), so all four cards update; any card that moves versus your previous normal run gets an \"updated by extended check (was GWx)\" tag (no tag, and a note, if there is no earlier normal run in the session); a per-GW table and verdict name the best Wildcard GW in the span. Button label is dynamic (recommend.extended_button_label). Superseded tests from Patches 102/104/105 updated in place. 36 new tests; full suite 215 passed. Previously, "
                  "Patch 106 (2026-10-04, manager screenshot: button said \"reach GW15\" but the result said \"reached GW10\" — \"We need it to reach Current GW + 9\"): root cause confirmed in code — the shared projection only carried xpts columns through GW10, so the extended solves for GW11-15 were silently dropped by the `_check_gws` filter. Fixed by projecting the extension GWs when the button is clicked, and by comparing requested vs reached GW (recommend.extend_reach_status) so any shortfall shows as a warning instead of a quiet success. 6 new tests in test_patch106_extend_actually_reaches_target.py. Previously, Patch 105 (2026-10-01, manager screenshot: Patch 104's always-on extend button clicked live, "
                  "red annotation \"i need a confirmation here after the run finishes that it's already done and "
                  "what is on the chips is the final\" + \"already clicked but i don't have a confirmation!!!!\"): "
                  "confirmed in code — the extended check's result (`_wc_check_note`) WAS already computed "
                  "correctly, but only ever rendered in the Wildcard card's tooltip near the TOP of the Chip Plan "
                  "tab, nowhere near the button at the bottom where the manager actually clicked and was looking. "
                  "Fixed by adding an explicit st.success()/st.warning() confirmation directly under the button, "
                  "showing the reached GW and the real cross-check result the moment it finishes. A second, more "
                  "serious bug was found investigating this: the note text itself was still hard-coded to claim "
                  "every extension ran \"to reach your scheduled Wildcard at GW{x}\" — true for a chip-driven "
                  "extension, but actively WRONG for Patch 104's new exploratory case (no scheduled chip in range "
                  "at all, or one already within normal reach) — the manager could run an exploratory +5 GW check "
                  "and get back a result that falsely describes itself as chip-driven. Fixed by extracting the "
                  "note-building into a new pure function recommend.build_wc_extend_note(chip_driven, "
                  "reached_target, check_gws_last, auto_wildcard_gw), branching correctly on the SAME "
                  "`_wc_extend_chip_driven` flag Patch 104 already computed at the button — moved to compute ONCE, "
                  "early, and threaded through to both the note-building site and the button site, rather than "
                  "letting a second independent copy drift out of sync the way `_wc_current_max_gw` did across "
                  "Patch 102/103. NEW test file test_patch105_extend_confirmation_and_note_fix.py (9 tests): the "
                  "note function's 5 cases (chip-driven wording, capped disclosure shown/omitted, exploratory "
                  "wording, exploratory with no scheduled chip at all), and 4 app.py wiring checks (single shared "
                  "chip_driven computation, note built via the new function, confirmation shown, failure case "
                  "handled). DISCLOSED: the confirmation only shows for the one run right after the click — same "
                  "one-run lifetime as the extension itself and as Cross-Tool Reconciliation's own result display; "
                  "a later, unrelated click clears it, same as before. Full regression suite: 184 passed (175 "
                  "prior + 9 new), zero regressions. Previously, Patch 104 (2026-10-01, manager screenshot: the \"Run extended Wildcard cross-check\" button "
                  "went correctly-but-confusingly disabled once Patch 103's wider normal reach (GW10) already "
                  "covered the one scheduled Wildcard on screen (GW8) — \"why it's grayed , it should give an "
                  "option for another 5 GWs beyond the one maximum used for the current run\", then \"i didn't get "
                  "it!!\" mid-explanation): investigated in code first — confirmed the disabled state was correct "
                  "per the Patch 101/102 design (a conditional gate: only relevant when a scheduled chip falls "
                  "beyond the normal reach), but that design itself was wrong for how the manager wants to use the "
                  "button. Asked directly (AskUserQuestion) rather than guessing: confirmed the manager wants an "
                  "ALWAYS-AVAILABLE manual control — clicking it should push the cross-check a further fixed +5 "
                  "GWs beyond the normal reach every time, whether or not a chip is currently scheduled in that "
                  "range. Redesigned: new recommend.resolve_wildcard_extend_target(current_max_gw, "
                  "auto_wildcard_gw, fixed_increment=5) always returns current_max_gw+5, extended further only if "
                  "a real scheduled chip sits even beyond that — so the button never falls short of a known "
                  "decision point. recommend.wc_extend_requested() no longer requires a scheduled Wildcard to "
                  "return True (supersedes Patch 101's test of the opposite behavior — updated, not silently left "
                  "to rot, in test_patch101_scan_window_and_optin_crosscheck.py). The button itself is never "
                  "disabled now; its message text still adapts — chip-driven framing when a scheduled chip is the "
                  "reason this run's extension matters, plain \"Exploratory\" framing otherwise, so the manager "
                  "always knows which situation they're in without the button hiding itself. NEW test file "
                  "test_patch104_always_on_extend_button.py (12 tests): the target resolver's 5 cases (fixed "
                  "increment, chip within it, chip beyond it, custom increment, exact-boundary), "
                  "wc_extend_requested()'s updated 3 cases, and 4 app.py wiring checks (target resolver called, "
                  "disabled branch gone, button key still defined, exploratory messaging present). DISCLOSED "
                  "COST: unchanged from Patch 101 — the extension still only runs its extra solves when clicked; "
                  "the no-click path (now the only difference is this button never renders disabled) is free. "
                  "Full regression suite: 175 passed (163 prior + 12 new), zero regressions. Previously, Patch 103 (2026-10-01, manager screenshot annotation: the extend-button's own text read "
                  "\"based on just GW6-GW6\" — \"why only till GW8, i need it to 6GWs from the current one\" / "
                  "\"this not we agreed about!!\"): confirmed a real bug, not a misreading — `_wc_current_max_gw` "
                  "(feeds both the opt-in extension's reach decision AND the extend-button's own \"normal reach\" "
                  "display text) was still computed independently as min(transfer_gw_list[-1], detect_gw_list[-1]), "
                  "degenerating to a single GW at the manager's Horizon=1 setting — while Patch 102 had already "
                  "fixed the ACTUAL cross-check calculation (`_wc_check_gw_source`) to use whichever window is "
                  "LONGER via recommend.resolve_wildcard_check_gw_window(). The two had drifted apart: the real "
                  "number was computed over the wider window, but the button still described the old, narrower "
                  "one. Fixed by deriving `_wc_current_max_gw` from that SAME resolver's own result, so the two "
                  "can never disagree again. Also, per manager confirmation this round: chip_shape_test."
                  "detection_window_gws (the Wildcard trigger's own detection window — separate from "
                  "chip_advisor_horizon, which Patch 101 already set to 5) goes 4 → 5, matching that same figure. "
                  "DISCLOSED COST: one more GW in the detection window means one more reachable-ceiling solve and "
                  "one more shape-test solve per run (~2 extra solves, well under 1s combined per the Patch 101 "
                  "benchmark) — not re-measured live, but small relative to the ~1m20s baseline. NEW test file "
                  "test_patch103_crosscheck_reach_consistency.py (3 tests): the config value, the app.py wiring "
                  "(confirms the old independent min() computation is gone), and a behavioral proof the resolver "
                  "no longer degenerates to a 1-GW window at Horizon=1 with a 5-GW detection window. Full "
                  "regression suite: 163 passed (160 prior + 3 new), zero regressions. Previously, Patch 102 (2026-10-01, manager: re-timed at 1m20s after Patch 101 — well under the 2-minute "
                  "target — then reported two live issues from a fresh screenshot): (1) the extended Wildcard "
                  "cross-check button (Patch 101) was a plain, unexplained st.button() — \"needs to be more visual "
                  "and self explained.\" Fixed: now a prominent, dynamic call-to-action naming the actual target "
                  "GW (\"Your scheduled Wildcard is GW{n}, beyond this check's normal GW{x}-GW{y} reach... reach "
                  "GW{n}\") when `_auto_wildcard_gw > _wc_current_max_gw` makes it relevant, or a disabled no-op "
                  "with a one-line reason otherwise — both states reuse variables already computed earlier in the "
                  "same script run, no new cost. (2) the Chip Plan tab showed \"Wildcard trigger ACTIVE (93.6%)\" "
                  "beside \"Wildcard may not be needed — plan reaches 96.6%\" on a week Transfer Recommendations "
                  "said \"Roll\" (no transfer made) — looked contradictory. Traced in code, not guessed: the 93.6% "
                  "(fpl_engine.wildcard_trigger_check()) averages over detect_gw_list (4 GWs, chip_shape_test."
                  "detection_window_gws); the \"may not be needed\" line averaged over a DIFFERENT, shorter window "
                  "— transfer_gw_list, just ONE GW at the manager's Horizon=1 setting — whenever the Patch 101 "
                  "extension isn't active (now the default). With zero transfers made, the 93.6-vs-96.6 gap was "
                  "fixture-variance noise from a 4-GW average vs. a 1-GW snapshot, not a transfer actually closing "
                  "anything — a mismatch that predates this session but was rarely visible before Patch 101, since "
                  "the extension used to auto-fire almost every run and incidentally widened this same window. "
                  "Fixed via new pure function recommend.resolve_wildcard_check_gw_window(transfer_gw_list, "
                  "detect_gw_list): picks whichever window is LONGER (ties keep transfer_gw_list), wired into all "
                  "three non-extended fallback branches for `_wc_check_gw_source`. Zero added solve cost — "
                  "`reachable_by_gw` already covers every GW in detect_gw_list (that's what feeds the trigger "
                  "itself), so widening to it needs no new solve. NEW test file "
                  "test_patch102_crosscheck_window_match_and_button_ux.py (8 tests): the resolver's window-length "
                  "logic (5 tests) and app.py wiring for both fixes (3 tests, including confirming the old direct "
                  "`_wc_check_gw_source = transfer_gw_list` assignment is gone from all three branches). Full "
                  "regression suite: 160 passed (152 prior + 8 new), zero regressions. Previously, Patch 101 (2026-10-01, manager: \"it's 4 minutes 46 seconds now ,, i need it below 2 minutes\"): "
                  "instrumented opt.solve_squad() directly (call counter + timer) and drove the real app functions "
                  "through a cold run matching the manager's own config/state — measured, not guessed: 55 total "
                  "solve_squad() calls, 17.4s in this sandbox. wildcard_window_value_scan() was the single biggest "
                  "piece (16 calls, 38% of the total), evaluate_free_hit()'s Chip Advisor scan next (8 calls) — both "
                  "driven by the SAME chip_advisor_horizon.default_gws window (8 GWs). Confirmed via this "
                  "measurement that caching (Patches 97/99/100) was structurally capped in how much it could ever "
                  "help: every one of these solves still runs once on a genuinely cold run regardless of caching, "
                  "so the only remaining lever was cutting solve COUNT. Two changes, both manager-confirmed before "
                  "building: (1) model_config.yaml chip_advisor_horizon.default_gws: 8 → 5 — cuts the two biggest "
                  "line items roughly in half at once; max_extend_gws (16) and the auto-extend-to-nearest-DGW/BGW "
                  "logic are UNCHANGED, so a known Double/Blank gameweek still pulls the window out to meet it, "
                  "only the routine every-run scan length shrinks. (2) the Patch 95 extended Wildcard cross-check "
                  "(a full second plan_transfer_schedule() + a full second solve_reachable_ceiling_by_gw() over +8 "
                  "GWs — the manager's own explicit choice last session to keep auto-running) is now opt-in: a new "
                  "st.button(key=\"wc_extend_run\") on the Chip Plan tab, gated through a new pure function "
                  "recommend.wc_extend_requested(session_state_flag, auto_wildcard_gw), same button-gated pattern "
                  "Cross-Tool Reconciliation already uses — a normal run no longer pays for it at all unless "
                  "clicked. Re-measured post-patch with the SAME instrumentation: 32 calls, 10.8s in this sandbox "
                  "on a normal run (extended cross-check not clicked) — a confirmed 42% fewer solves, 38% less "
                  "sandbox time, not an estimate. NEW test file test_patch101_scan_window_and_optin_crosscheck.py "
                  "(8 tests) covers the config change (default_gws=5, max_extend_gws/auto-extend unchanged) and the "
                  "new wc_extend_requested() gate (both conditions required; app.py wiring confirmed, old "
                  "unconditional gate confirmed gone). Full regression suite: 152 passed (144 prior + 8 new), zero "
                  "regressions. DISCLOSED TRADEOFF: outside an auto-extension to a confirmed DGW/BGW, the Wildcard "
                  "window-value scan and the Bench Boost/Triple Captain/Free Hit advisors now look 5 GWs ahead by "
                  "default instead of 8 — manager-confirmed acceptable given the 2-minute target. DISCLOSED, NOT "
                  "OVERCLAIMED: live re-verification of the deployed app's wall-clock time was not possible from "
                  "this sandbox — the sandbox-measured 42%/38% reduction is real and reproducible, but whether it "
                  "reaches the 2-minute target on Streamlit Community Cloud's own hardware can only be confirmed by "
                  "the manager's own re-timed run after deploying this patch. Previously, Patch 100 (2026-09-30, manager: \"dig deep on the performance\" — after choosing to keep the "
                  "extended cross-check running with every run rather than gate it behind a button): a further "
                  "call-site audit beyond Patch 99's five found SIX MORE uncached MILP-class calls, verified in "
                  "code, not inferred from shape. Three fire completely unconditionally on every single script "
                  "rerun: data_pipeline.solve_reachable_ceiling() (line ~1860, the Team Rating % headline's "
                  "single-total reachable solve — distinct from the by-GW version Patch 99 already cached), "
                  "data_pipeline.solve_ceiling() (line ~1861, the \"theoretical ceiling\" full-pool solve), and "
                  "data_pipeline.solve_free_hit_optimal_squad() (line ~1949 — this file's OWN Patch 22 comment "
                  "already said \"computed automatically every run ... not gated\"). A fourth, chip_protocol."
                  "wildcard_freehit_shape_test() (line ~2157), runs 4 separate opt.solve_squad() calls internally "
                  "(one per GW in the 4-GW chip_shape_test.detection_window_gws window) whenever the Wildcard or "
                  "Free Hit signal is active — true in both of the manager's own screenshots this session. The "
                  "last two live in the \"Team Recommendation — active signals, auto-built\" block (lines "
                  "~3608-3693), explicitly commented as NOT button-gated: chip_protocol.evaluate_wildcard_whatif() "
                  "and a second solve_free_hit_optimal_squad() call, both firing whenever their respective chip "
                  "signal is active. All six now route through new @st.cache_data(ttl=900, show_spinner=False) "
                  "wrappers (_reachable_ceiling_single_calc, _theoretical_ceiling_calc, _fh_optimal_calc — shared "
                  "by both its call sites, _shape_test_calc, _wc_whatif_calc — also reused for the pre-existing "
                  "manual \"Evaluate your own scenario\" picker's own call, adding caching there too with zero "
                  "behavior change), same proven pattern as every prior _xxx_calc wrapper since Patch 86. NEW "
                  "test file test_patch100_remaining_uncached_solves.py (11 tests) verifies (a) each old bare call "
                  "no longer appears as live code, only as historical comment text, and (b) each new wrapper is a "
                  "byte-for-byte pure pass-through to the real underlying function (no behavior change). TWO more "
                  "pre-existing golden-slice tests (test_patch73_compliant_rating_diagnostic_golden_slice.py, "
                  "test_patch73_fh_rating_diagnostic_golden_slice.py) broke on this refactor the same way "
                  "test_patch73/74 did in Patch 99 — fixed the same way, with uncached pass-through stand-ins for "
                  "the new wrapper names in their exec namespaces. Full regression suite: 144 passed (133 prior + "
                  "11 new), zero regressions, same 6 pre-existing unrelated AppTest-driver failures excluded as "
                  "every prior patch. DISCLOSED, NOT OVERCLAIMED: live re-verification of wall-clock time on the "
                  "deployed app was not possible from this sandbox — this closes a confirmed, real, previously-"
                  "missed gap (six more uncached expensive computations, now cached), on top of Patch 99's five, "
                  "but whether it further moves the ~5m10s figure can only be confirmed by the manager's own "
                  "re-timed run after deploying this patch. The extended cross-check (Patch 95) remains "
                  "intentionally un-gated per the manager's explicit \"i want to be with the run\" — its own two "
                  "solves are NOT part of this patch's fix, by design. Previously, Patch 99 (2026-09-30, manager re-timed the run AFTER deploying Patch 97 and got ~6m10s — "
                  "essentially unchanged from the original ~6m45s, confirming Patch 97's fix targeted the wrong "
                  "bottleneck; manager: \"i need this to be rechecked\"): found the REAL cause by systematically "
                  "auditing every call site in app.py against which computations were/weren't wrapped in "
                  "@st.cache_data, instead of re-guessing at the same small piece Patch 97 already (correctly, "
                  "but insufficiently) fixed. Confirmed FIVE genuinely expensive computations were called bare "
                  "at the app.py top level with NO caching at all — meaning each one re-ran IN FULL on every "
                  "single Streamlit script rerun (any widget touch: Style toggle, materiality-bar slider, "
                  "hit-stance radio, etc.), not just once per \"Run Model\" click: (1) recommend."
                  "suggest_transfers() — the actual Transfer Recommendations engine, a chained k=1..5 "
                  "(under \"Hit if worth it\") multi-GW MILP search with its own position tie-break and "
                  "Starting-XI Impact Check solves on top — almost certainly the single largest piece of this "
                  "regression; (2) data_pipeline.solve_reachable_ceiling_by_gw() for the Wildcard trigger — one "
                  "fresh MILP solve per GW in detect_gw_list; (3)-(5) the three isolated Chip Advisor cards "
                  "(evaluate_bench_boost/evaluate_triple_captain/evaluate_free_hit) — Free Hit's in particular "
                  "confirmed as a fresh MILP rebuild solve per horizon GW by this file's OWN pre-existing "
                  "comment, up to 8-16 GWs. All five now route through new @st.cache_data(ttl=900, "
                  "show_spinner=False)-wrapped functions (_suggest_transfers_calc, _reachable_ceiling_calc, "
                  "_bb_advisor_calc, _tc_advisor_calc, _fh_advisor_calc), following the exact pattern already "
                  "proven for _chip_portfolio_calc (Patch 86) and _extended_wc_cross_check_calc (Patch 95) — a "
                  "rerun with unchanged inputs is now a cache hit (near-instant) instead of a full re-solve on "
                  "ALL FIVE, not just chip_portfolio. moe_fn/rebuild_fn closures (not reliably hashable for a "
                  "cache key) are reconstructed INSIDE each cached wrapper from plain hashable pieces "
                  "(cfg/chip_key/team_value), same discipline _chip_portfolio_calc already established. TWO "
                  "existing golden-slice tests (test_patch73/74, which exec() literal source slices extracted "
                  "from app.py) broke on this refactor since they don't define the new wrapper names in their "
                  "exec namespace — fixed by adding an uncached pass-through stand-in for _reachable_ceiling_calc "
                  "in both (these tests cover diagnostic-text logic, not caching, so an uncached stand-in "
                  "preserves exactly what they're testing). Full 133-test regression suite passes, zero "
                  "regressions beyond that expected/fixed golden-slice adjustment. DISCLOSED, NOT OVERCLAIMED: "
                  "live re-verification of the actual before/after wall-clock time on the deployed app was not "
                  "possible from this sandbox (no network access to the live app or the official FPL API) — "
                  "this is a confirmed, real fix for a confirmed, real gap (five uncached expensive computations, "
                  "now cached), not a guess, but whether it fully closes the ~6-minute gap can only be confirmed "
                  "by the manager's own re-timed run after deploying this patch. Previously, Patch 98 (v6.9 Rule #52 follow-up, \"a more wide rule\" — manager, 2026-09-30, after "
                  "confirming Patch 96's harmonized chip sequence live on team 26073, asking what the Transfers "
                  "side does \"if we will use the chip\" and explicitly widening the ask beyond that one "
                  "screenshot): two real, general gaps found by reading recommend.py/app.py end to end. (1) "
                  "`_bb_play_gw` (app.py) was sourced ONLY from the isolated Bench Boost advisor's own "
                  "individually-best week, never from chip_portfolio's harmonized joint assignment — the same "
                  "\"two disconnected sources of truth\" bug class Patch 91/92/93 already fixed elsewhere, and "
                  "a live risk specifically BECAUSE Patch 96 lets the joint scheduler place Bench Boost on a "
                  "different week than its own standalone optimum. Fixed via new "
                  "recommend.resolve_bb_play_gw(isolated, harmonized) — prefers the harmonized pick, falls back "
                  "to the isolated verdict only when no joint schedule has one. (2) plan_transfer_schedule() had "
                  "no Free Hit equivalent of the Wildcard \"illusory value\" truncation it already has (Patch "
                  "92) — confirmed via reading the whole function body, no freehit_gw parameter existed at all. "
                  "A transfer's valuation for any week could be credited with points that only exist at a "
                  "scheduled Free Hit week, but that week is actually played by a different, temporary rebuild "
                  "squad, never the persisted one. Fixed via chip_schedule's new optional \"freehit_gw\" key: "
                  "excludes (not truncates — Free Hit reverts after one week, unlike Wildcard) that one GW from "
                  "every week's remaining_gws. Triple Captain deliberately left alone — it doesn't change squad "
                  "composition, so the transfer planner has nothing to get wrong there. Both fixes additive-only "
                  "(new optional params/keys, default None = exact pre-Patch-98 behavior). Test-first: 7 new "
                  "tests written and confirmed failing before implementation (4 for resolve_bb_play_gw via "
                  "AttributeError; 3 for the Free Hit exclusion, confirmed failing for the RIGHT reason — the "
                  "un-implemented freehit_gw key being silently ignored, verified by first tracing why an "
                  "initial synthetic fixture returned zero transfers even in the baseline case, a GK-position/"
                  "budget artifact in the test setup, not a code bug — before trusting the red), all passing "
                  "after; full 133-test regression suite passes (126 pre-existing + 7 new), zero regressions. "
                  "Previously, Patch 97 (2026-09-30, manager-reported ~6m45s runtime regression after deploying Patch 96, "
                  "live screenshot of team 26073's Chip Plan tab confirming Patch 96's harmony logic IS working "
                  "live — Triple Captain@GW10/Free Hit@GW11 both pre-Wildcard, Wildcard@GW12, Bench Boost@GW13 "
                  "post-Wildcard, a genuinely joint 4-chip sequence): found and fixed ONE real, confirmed "
                  "deviation from this codebase's own caching discipline — Patch 96's reachable_ceiling_by_gw "
                  "param was threaded into a @st.cache_data-wrapped function carrying data_pipeline."
                  "solve_reachable_ceiling_by_gw()'s FULL per-GW result (an embedded 15-row squad DataFrame plus "
                  "cost/total_xpts, confirmed via code read of data_pipeline.py lines 833-863), when chip_"
                  "protocol._apply_squad_health_guardrail() only ever reads one float (total_xpts) out of it — "
                  "every other value already crossing that same cache boundary (fh_gap_table) was already "
                  "reduced to plain floats first, for exactly this reason; this one wasn't. Reduced at the "
                  "app.py call site to {gw: {\"total_xpts\": float}} before it reaches _chip_portfolio_calc — "
                  "zero change to chip_protocol.py's guardrail (same shape it already expected), so every "
                  "existing Patch 96 test keeps passing unmodified (126/126, zero regressions). DISCLOSED, NOT "
                  "OVERCLAIMED: benchmarked the DataFrame-hashing overhead directly (~27ms/call in this "
                  "sandbox) and it is NOT obviously enough on its own to explain a jump to 6m45s — this is a "
                  "real, confirmed fix worth shipping regardless, but not asserted as the full explanation. "
                  "Also flagged plainly: Patch 96's own ~40-50s performance estimate was scoped ONLY to the "
                  "isolated chip_portfolio_schedule() computation benchmarked directly at the time, NOT the "
                  "full \"Run Model\" click across every tab (Transfer Recommendations' chained plan, Patch 95's "
                  "own extended Wildcard cross-check solve, captaincy, disruption checks, etc. all run on the "
                  "same click) — that scope gap was not caught before the estimate was given, and awaiting the "
                  "manager's confirmation of what the 6m45s stopwatch actually covered (this tab alone vs. the "
                  "full page) before further profiling, rather than guessing at the remaining gap. Previously, "
                  "Patch 96 (v6.9 Standing Rule #52, Chip Timing Harmony — manager discussion, 2026-09-30, "
                  "\"chips can work on harmony if it's applicable and not stand alone chips\"): confirmed via "
                  "code read BEFORE scoping anything that chip_portfolio_schedule() (Rule #49) already runs all "
                  "4 chips together maximising their COMBINED total via brute-force search — that part needed "
                  "no fix. Two real gaps did: (a) Free Hit had no pre/post-Wildcard value split (unlike Bench "
                  "Boost/Triple Captain, which already value differently before vs. after a scheduled "
                  "Wildcard) — _value_for(\"freehit\", ...) returned a flat fh_gap_table lookup regardless of "
                  "wc_choice. Fixed via chip_protocol._fh_post_table_for_squad(): reuses each week's already-"
                  "computed rebuild total (a Free Hit rebuild is squad-independent — confirmed via code read of "
                  "data_pipeline.solve_free_hit_rebuild() — so no second MILP solve is needed) and only "
                  "recomputes the cheap \"current\" side (opt.best_starting_xi(), no MILP) against the "
                  "Wildcard's rebuild squad. (b) nothing checked squad health AFTER the scheduled chips are "
                  "done, so a combination could pull Wildcard earlier purely to inflate BB/TC/FH's numbers with "
                  "zero guardrail. Fixed via chip_protocol._apply_squad_health_guardrail(): runs ONLY on the "
                  "already-small near-tie set (never the full combinatorial search — confirmed no new MILP "
                  "solve added there), rejecting any near-tie whose checkpoint-GW squad total falls more than "
                  "Rule #34's own moe_fn band below the healthiest near-tie's checkpoint total. Both wired as "
                  "new OPTIONAL chip_portfolio_schedule() params (fh_by_gw, reachable_ceiling_by_gw), default "
                  "None = byte-for-byte pre-Patch-96 behavior — every existing call site/test keeps passing "
                  "unmodified. PERFORMANCE (benchmarked 2026-09-30 before building, not guessed): a fresh "
                  "solve_squad() ~1.15s, a retain-pool reachable solve ~0.48s, opt.best_starting_xi() (what both "
                  "new pieces use) ~24ms — ~50x cheaper, confirming neither gap adds a new solve to the hot "
                  "path. A separate naive thread-based parallelization of the PRE-EXISTING ~45s worst-case "
                  "baseline was investigated and explicitly REJECTED after benchmarking: every concurrent "
                  "solve_squad() call silently returned None in this sandbox — a correctness break, not shipped. "
                  "DISCLOSED LIMITATION: the guardrail's checkpoint ceiling reuses the Wildcard trigger's "
                  "existing ~4-GW detect_gw_list reachable table (zero added solve) — when the scan's checkpoint "
                  "GW falls beyond that range, the guardrail has no ceiling to compare against and safely "
                  "no-ops (falls back to the unfiltered near-tie set) rather than firing; full coverage would "
                  "need one additional bounded solve, not yet built. NOT changed, deliberately, per manager's "
                  "own scoping correction (\"let's agree on WC1, WC2 will solve itself automatically\"): no "
                  "two-window (WC1+WC2) joint modeling was added. Test-first: 10 new tests written and "
                  "confirmed failing (AttributeError/TypeError, functions/params didn't exist) before "
                  "implementation, all passing after; full 126-test regression suite passes (116 pre-existing + "
                  "10 new), zero regressions. Live browser verification against the running app was not "
                  "possible this session (sandbox network still cannot reach the official FPL API) — "
                  "verification rests on the test suite plus direct code tracing, same as every prior patch "
                  "this session. Previously, Patch 95 (v6.9 Rule #49 cross-check extension — manager discussion continued, 2026-09-30, "
                  "resolving the \"auto-extend horizon to reach the chip week\" scope chosen for the original "
                  "'cross-check UI' idea): confirmed via code read that the Wildcard cross-check note (just "
                  "fixed for its squad-reconstruction bug in Patch 94) is bounded by TWO separate windows "
                  "neither tied to when the Rule #49 scheduler actually plans the Wildcard — the transfer-plan "
                  "horizon (transfer_gw_list, capped by the Horizon slider, max 1-6) and the reachable-ceiling "
                  "detection window (detect_gw_list, a fixed ~4-GW window from chip_shape_test."
                  "detection_window_gws, app.py lines ~1402-1412). Manager confirmed (two AskUserQuestion "
                  "rounds) extending BOTH sides to reach the scheduled Wildcard GW, capped at +8 GWs beyond "
                  "whichever window currently reaches less far, and caching the extra computation. Adds "
                  "recommend.resolve_cross_check_horizon(planning_gw, current_max_gw, target_gw, max_extension) "
                  "— a small pure function deciding whether/how far to extend, before either extra family of "
                  "MILP solves runs — and app.py's _extended_wc_cross_check_calc() (@st.cache_data(ttl=900), "
                  "same pattern as _chip_portfolio_calc), which runs an extended recommend."
                  "plan_transfer_schedule() (chip-schedule-aware, reusing Patch 92's machinery) AND an extended "
                  "data_pipeline.solve_reachable_ceiling_by_gw() over the same capped GW range, wrapped in a "
                  "try/except that falls back to the normal unextended check on any failure rather than risk "
                  "the whole Chip Plan tab over what is explicitly a supplementary disclosure, never core "
                  "output. Never touches `rec`, `reachable_by_gw`, or the Wildcard trigger's own headline % — "
                  "this is a side computation purely for the cross-check note; the Transfer Recommendations tab "
                  "is completely unaffected. The note's text now discloses whenever it's running over an "
                  "auto-extended horizon, and separately flags it if the +8 GW cap still fell short of the "
                  "actual scheduled GW, per Standing Rule #4 (show the inputs, never silently narrow the "
                  "claim). Test-first: 7 new tests written and confirmed failing (AttributeError, function "
                  "didn't exist) before implementation for resolve_cross_check_horizon() (the app.py wiring "
                  "itself is Streamlit script code, verified by syntax check + full regression suite + a "
                  "hand-traced walkthrough of every branch, same as every other app.py-only change this "
                  "session), all passing after; full 116-test regression suite passes (109 pre-existing + 7 "
                  "new), zero regressions. DISCLOSED LIMITATION (unchanged from Patch 92/93/94): live browser "
                  "verification against the running app was not possible this session — this sandbox's "
                  "network still cannot reach the official FPL API; verification rests on the test suite plus "
                  "a direct code trace of the app.py wiring. Previously, Patch 94 (v6.9 Rule #49 cross-check "
                  "correctness fix — manager discussion, 2026-09-30, "
                  "\"let's start the discussion for second point Original 'cross-check UI'\": before designing "
                  "anything new, checked the codebase for what already exists rather than assuming a blank "
                  "slate. Found the Wildcard card's existing \"Cross-check against your own recommended "
                  "transfer plan\" note (Patch 46-49) reconstructed \"the squad after your plan\" by flattening "
                  "EVERY week's moves from the chained weekly plan (`rec['moves']` = every week's moves "
                  "concatenated, confirmed via code read of recommend.plan_transfer_schedule()'s return dict) "
                  "into ONE pd.concat applied all at once, regardless of which week each move belonged to — "
                  "already misleading for any multi-week plan with more than one week of real moves, and "
                  "actively wrong now that Patch 92 makes a scheduled Wildcard week inject a ~13-player "
                  "wholesale rebuild into that same flat list, getting mashed together with ordinary "
                  "pre/post-Wildcard transfers into a squad that was never actually reachable at any single "
                  "point in time. Manager confirmed (AskUserQuestion) fixing this FIRST, standalone, before any "
                  "new cross-check design. Fixed by extracting the CORRECT reconstruction that already existed "
                  "elsewhere in this app — the pitch navigator's per-GW chained rebuild (Patch 84) — out of its "
                  "private closure inside _render_pitch_navigator() into a new top-level, testable "
                  "recommend.build_squad_after_by_gw(squad_df, weekly_plan, pool_df) function, now used by BOTH "
                  "the navigator (replacing its own local _apply_moves loop, zero behavior change there — same "
                  "chaining, same pool, same result) and the Wildcard cross-check (replacing the flat bug). "
                  "Same \"stop having two disconnected reconstruction methods\" fix pattern as Patch 91/92/93. "
                  "Test-first: 5 new tests written and confirmed failing (AttributeError, function didn't "
                  "exist) before implementation — including one that specifically reproduces the Wildcard-week "
                  "blending bug and proves the chained version threads through it correctly — all passing "
                  "after; full 109-test regression suite passes (104 pre-existing + 5 new), zero regressions. "
                  "DISCLOSED LIMITATION (unchanged from Patch 92/93): live browser verification against the "
                  "running app was not possible this session — this sandbox's network still cannot reach the "
                  "official FPL API; verification rests on the test suite plus a direct code trace of the "
                  "app.py wiring. Previously, Patch 93 (v6.9 Rule #49 follow-up, single-decision path auto "
                  "chip-capping — manager, "
                  "2026-09-30: \"let's patch the first one\" (of the queued-decisions list), closing the "
                  "DISCLOSED GAP flagged at the end of Patch 92): confirmed via code read that "
                  "suggest_transfers()'s single-decision path (horizon=1, or always under \"Force\") built its "
                  "`chip_capped_gw_list` (drives the \"Chip-aware alt: Roll\" advisory note) EXCLUSIVELY from "
                  "the sidebar's manual \"Next planned full-rebuild chip GW\" dropdown — never from the app's "
                  "own auto-computed Rule #49 joint chip schedule (chip_portfolio) that Patch 92 already wired "
                  "into the CHAINED planner. Same \"two disconnected sources of truth\" bug class as Patch "
                  "91/92, on the other code path: at horizon=1 (the app's own default), a Wildcard the model "
                  "itself had already scheduled produced NO chip-aware note unless the manager also happened "
                  "to manually set the matching dropdown value. Fixed with a new small, pure "
                  "recommend.resolve_chip_capped_gw_list(gw_list, manual_planned_chip_gw, auto_wildcard_gw) "
                  "helper: prefers an explicit manual value (an override that may reflect something the "
                  "auto-scheduler doesn't know) when the manager has set one, falls back to the auto-detected "
                  "Wildcard GW (chip_portfolio['assignment']['wildcard']) otherwise, returns None when neither "
                  "applies or the chip falls after the horizon — same edge-case contract as the inline logic "
                  "it replaced, including the empty-list-vs-None distinction the downstream check relies on. "
                  "Test-first: 6 new tests written and confirmed failing (AttributeError, function didn't "
                  "exist) before implementation, all passing after; full 104-test regression suite passes "
                  "(98 pre-existing + 6 new), zero regressions. DISCLOSED LIMITATION (unchanged from Patch "
                  "92): live browser verification against the running app was not possible this session — "
                  "this sandbox's network still cannot reach the official FPL API; verification rests on the "
                  "test suite plus a direct code trace of the app.py wiring. Previously, Patch 92 (v6.9 Rules "
                  "#44/#48/#49 read together, chip-aware weekly transfer plan — manager "
                  "discussion, 2026-09-30: \"is the wildcard considered... maybe the wildcard week will give us "
                  "another transfer plan\" caught, before building, that the originally-scoped simpler fix "
                  "(auto-extending the chained transfer plan's horizon for a cross-check note) would have been "
                  "unsafe, since recommend.plan_transfer_schedule() — confirmed via code read — had ZERO "
                  "Wildcard awareness: it would keep chaining transfers on the CURRENT squad straight through a "
                  "scheduled Wildcard week and beyond, recommending moves whose value only existed on a squad "
                  "path the Wildcard was about to wipe out. Manager confirmed \"let's go\" on the deeper fix. "
                  "plan_transfer_schedule() gains an optional chip_schedule={'wildcard_gw','wildcard_rebuild_"
                  "squad'} parameter (Wildcard-only in scope — Free Hit reverts after one week per Rule #44/#49c "
                  "so it never changes the persisted squad path this planner tracks, and Bench Boost/Triple "
                  "Captain never change squad composition at all): (a) every week strictly before wildcard_gw "
                  "values its transfer decision against a horizon truncated at wildcard_gw, so a transfer whose "
                  "payoff only exists at/after the rebuild is correctly rejected; (b) at wildcard_gw itself the "
                  "ordinary k=1..k_upper search is skipped and the squad is replaced wholesale by the pre-"
                  "computed rebuild squad, drawing NO free transfer (Rule #44) while still accruing its own +1 "
                  "FT exactly as an unused week would; (c) every week after continues the normal chained search "
                  "from the rebuilt squad. Omitting the parameter (default None) reproduces exact pre-Patch-92 "
                  "behavior. app.py's suggest_transfers() call site now builds this dict from the SAME "
                  "chip_portfolio/wc_window_scan the Rule #49 joint scheduler above it already computes this "
                  "run (chip_portfolio['assignment']['wildcard'] for the GW, wc_window_scan['by_gw'][gw]"
                  "['rebuild_squad'] for the rebuild squad) — no second, independent Wildcard-detection path. "
                  "Test-first: 5 new tests written and confirmed failing (TypeError, parameter didn't exist) "
                  "before implementation, all passing after; full 98-test regression suite passes unmodified "
                  "(93 pre-existing + 5 new), confirming backward compatibility when the parameter is omitted. "
                  "DISCLOSED GAP, not yet fixed: suggest_transfers()'s separate single-decision path (used at "
                  "horizon=1 or \"Force\") still has its own older, MANUAL chip_capped_gw_list mechanism (driven "
                  "by the sidebar's \"Next planned full-rebuild chip GW\" dropdown), which is still NOT connected "
                  "to the app's own auto-computed Rule #49 sequence — that is a separate, smaller gap from this "
                  "one and hasn't been scoped yet. DISCLOSED LIMITATION: live browser verification against the "
                  "running app was not possible this session — this sandbox's network cannot reach the live "
                  "official FPL API (fantasy.premierleague.com), confirmed via a direct curl connection-failure "
                  "test, not a code issue; verification here rests on the full test suite plus a direct code "
                  "trace of the app.py wiring instead. Previously, Patch 91 (v6.9 Rule #49, chip-expiry "
                  "correctness fix — manager report, 2026-09-30: "
                  "\"the chip expiry needs to be considered.\" Confirmed via code read: chip_protocol."
                  "chip_status() already correctly tracks each chip's own [start_event, stop_event] window "
                  "from the official chip calendar, and app.py's _clip_to_available_windows() already used "
                  "it to correctly bound the Wildcard/Free Hit detect window and the Bench Boost/Triple "
                  "Captain/Free Hit ISOLATED advisor cards' candidate weeks — but chip_portfolio_schedule() "
                  "(the Rule #49 JOINT scheduler that, since Patch 87, actually drives the four chip cards' "
                  "headline verdicts whenever >=2 chips are available) received the unclipped scan window "
                  "with no per-chip-type bound at all, so its brute-force search could recommend playing a "
                  "chip on a gameweek AFTER that chip's own currently-available window closes — a genuinely "
                  "illegal recommendation (the chip would already be expired/lost), silently overriding the "
                  "correctly-clipped isolated verdict Patch 87 made secondary. Fixed with a new optional "
                  "valid_gws_by_type parameter that restricts every chip type's candidate weeks (Wildcard "
                  "included) to its own real window before the search runs; the app.py call site now builds "
                  "this from the exact same _clip_to_available_windows() the isolated cards already use, so "
                  "there are no longer two different sources of truth for \"is this GW actually legal for "
                  "this chip.\" Test-first: 6 new tests written and confirmed failing (TypeError, parameter "
                  "didn't exist) before implementation, all passing after; full 93-test regression suite "
                  "(including all pre-existing Patch 85/87/89 portfolio tests) passes unmodified, confirming "
                  "backward compatibility when the parameter is omitted. Live-verified against the running "
                  "app: Chip Plan tab renders all four cards with sequenced GWs, zero exceptions. Previously, "
                  "Patch 90 (v6.9 Rule #46/#47, market-odds leg — BACKEND ONLY, not yet wired into the live "
                  "pipeline, paused pending the manager obtaining a free API key from The Odds API): added "
                  "chip_protocol.implied_probs_from_odds()/fit_fixture_goals_from_probs()/"
                  "rescale_goals_to_league_level()/cross_check_goal_estimates()/is_odds_stale()/"
                  "effective_fixture_strength() and fpl_data.fetch_market_odds_epl(), all built test-first "
                  "(24 tests, confirmed red before implementation) against fabricated data — no live odds "
                  "call has been made yet. Also caught and fixed in the same patch: fixture_attack_factor_vec() "
                  "applied the SAME flat fixture-adjustment strength to every horizon gameweek with no "
                  "distance decay at all, contradicting Rule #46(e)'s \"s shrinking toward 0 with distance "
                  "... beyond GW+5 ... s=0\" — extended (backward-compatibly) with an optional gws_ahead "
                  "parameter and a new effective_fixture_strength() decay curve. Previously, Patch 89 (model: "
                  "v6.9 Standing Rule #50 Chip-Gain Reporting + Rule #51 Cross-Tool "
                  "Reconciliation, plus partial Rule #49(d)/(f) — confirmed via code read that no quoted chip "
                  "gain anywhere in this app stated its horizon, formula variant, or baseline, and nothing "
                  "decomposed a disagreement with another tool's figure input-by-input. Adds chip_protocol."
                  "gain_disclosure() — a shared formatter appended to every chip-gain caption/tooltip stating "
                  "the horizon, formula variant (base/adjusted, read from cfg), tier, and baseline, with "
                  "hold-squad/no-transfer baselines explicitly flagged as NOT the Rule #50 decision basis — "
                  "applied to both Wildcard rebuild captions (auto and manual scenario) and the joint chip "
                  "sequence's card tooltips. Adds a new \"Cross-Tool Reconciliation (Rule #51)\" expander on the "
                  "Chip Plan tab: enter another tool's quoted Wildcard gain for a candidate week and "
                  "chip_protocol.reconcile_wildcard_gain() decomposes the difference one input at a time — "
                  "captain doubling (Standing Rule #31: re-scores the same squads with rating_horizon_value() "
                  "vs. this app's captain-excluded realized_horizon_value()), free-transfer accrual (one extra "
                  "solve at a static FT count), horizon length and team value (both a full re-scan at the "
                  "alternate setting) — reporting each input's share of the gap and any residual as explicitly "
                  "UNEXPLAINED rather than guessed at; formula variant (base/adjusted) and availability "
                  "assumptions are the 2 of 6 doc-listed inputs NOT yet automated, named via the panel's own "
                  "\"Not yet automated\" expander. Rule #49(d): chip_portfolio_schedule()'s Bench Boost/Triple "
                  "Captain valuation now solves each candidate week's actually-reachable squad under accruing "
                  "free transfers (new _reachable_bb_tc_tables(), reusing the same one-shot solve_squad() "
                  "pattern wildcard_window_value_scan() already uses) instead of scoring the CURRENT squad "
                  "unchanged at every week — still a one-shot approximation, not a fully chained transfer "
                  "simulation, and disclosed as such. Rule #49(f): when the joint schedule places Bench Boost on "
                  "or after Wildcard, the Wildcard card's tooltip now also reports (never re-scores) that "
                  "build's own bench sum at the Bench Boost week, reusing the already-computed rebuild-squad "
                  "path. Live-verified: all 4 automated Rule #51 components render correctly with a real "
                  "decomposition table and residual figure; the Reconcile button's own computation (several "
                  "MILP re-solves) takes several seconds, so it's now wrapped in an explicit spinner so a click "
                  "doesn't look like a no-op while it runs. NOT in this patch, staying explicitly out of scope: "
                  "the Rule #46/#47 market-odds fixture leg (team-strength tier only, per Patch 83's own "
                  "disclosure — unchanged), the v6.9 Step 0 calibration log (separate infrastructure, not "
                  "started), and Rule #51's 2 not-yet-automated inputs named above. No scheduling/tie-break "
                  "logic changed — confirmed via the full pre-existing test suite passing unmodified. "
                  "Previously, Patch 88 (manager follow-up, same day: the Patch 87 fix still left a one-line caption + a "
                  "second \"Scope note\" expander sitting below the card grid — manager's call: \"only one should "
                  "exist.\" That leftover always-visible caption and its own expander are removed entirely; the "
                  "combined total/tie-band figure moves into each scheduled card's own tooltip "
                  "(_seq_combined_note()), and the Rule #49d/f scope disclosure folds into the one expander this "
                  "tab already had (\"Full chip analysis\") instead of getting a second one. Net result: the Chip "
                  "Plan tab now has exactly one visible surface (the card grid) and one opt-in detail panel (Full "
                  "chip analysis) for everything chip-related — no second section, no duplicated numbers. "
                  "Previously, Patch 87 (manager report against a live Chip Plan screenshot, 2026-09-29): two "
                  "problems, both confirmed in code before fixing. (1) The Rules #48/49 \"Chip Sequence\" section "
                  "(Patch 85) was paragraph-and-arrow-chain prose sitting as permanent visible text below the card "
                  "grid — breaks the Patch 31 \"more visuals, more than words\" rule every other part of this tab "
                  "follows (rule citations/reasoning in hover tooltips, visible surface kept to a label + one "
                  "caption line). (2) The four Chip Signals cards (Wildcard/Bench Boost/Triple Captain/Free Hit) "
                  "and the Chip Sequence section showed genuinely DIFFERENT verdicts for the same chip (e.g. Free "
                  "Hit card: \"PLAY GW7\"; sequence: \"GW9\") because the cards come from each chip's own ISOLATED "
                  "scan (evaluate_bench_boost/evaluate_triple_captain/evaluate_free_hit/wildcard_trigger_check) "
                  "while the sequence comes from the JOINT Rule #49 assignment that accounts for the chips "
                  "competing for the same weeks — two engines, no reconciliation. Manager chose \"joint sequence "
                  "drives the cards\": each card's headline GW/verdict now IS the sequenced one whenever >=2 chip "
                  "types are available and the joint scan ran, with its own isolated-scan number folded into the "
                  "tooltip as context (see _seq_for()/_advisor_card() in app.py) instead of standing as a second, "
                  "contradicting number. The standalone prose section is replaced by a one-line caption pointing "
                  "back to the cards plus the existing "
                  "Scope Note expander — same information, no longer duplicated or in conflict. Previously, "
                  "Patch 86 (performance: manager asked whether Patch 85's new Rule #48/#49 chip computation would "
                  "hurt runtime. Confirmed via code read (grep for @st.cache_data/@st.fragment against the Patch "
                  "85 insertion point) that it was bare top-level script code with NO caching, unlike every other "
                  "comparably expensive computation in this file (_fixture_baselines, _project, _picks) — meaning "
                  "its ~16 MILP solves (Rule #48's 2-per-candidate-week scan) plus Rule #49's brute-force search "
                  "re-ran in full on EVERY Streamlit rerun, including ones triggered by unrelated widgets (Style "
                  "toggle, materiality slider) that never change the squad/transfers/bank/chip-availability it "
                  "depends on. Fixed by extracting the computation into a new _chip_portfolio_calc(), wrapped in "
                  "the same @st.cache_data(ttl=900, show_spinner=False) pattern already used elsewhere in this "
                  "file — a rerun with unchanged inputs is now a cache hit instead of a re-solve; zero change to "
                  "computed values, confirmed by the existing Patch 85 unit tests passing unmodified. Previously, "
                  "Patch 85 (model: v6.9 Standing Rules #48-49, Chip Window Value + Chip Portfolio Scheduling — "
                  "confirmed via code read that neither existed: the only prior Wildcard what-if compared against "
                  "holding the squad, not the required best-no-chip-transfer-path baseline, and nothing sequenced "
                  "BB/TC/FH/Wildcard onto distinct weeks to maximise their combined total. Adds chip_protocol."
                  "wildcard_window_value_scan() (Rule #48: per-candidate-week Wildcard rebuild vs. a free-transfer-"
                  "accruing reachable baseline, two MILP solves per week, not the ~5x-more-expensive fully-chained "
                  "simulation first tried and found too slow live) and chip_portfolio_schedule() (Rule #49: a small "
                  "brute-force assignment placing available chips on distinct weeks, switching Bench Boost/Triple "
                  "Captain valuation onto the Wildcard's own rebuild squad for any week at or after it, tie-band "
                  "ties resolved to the later commitment); both are new, disclosed-scope functions — Rule #49(d) "
                  "preparation-transfer FT accounting and #49(f) a Wildcard build's explicit bench term are NOT yet "
                  "modeled, tracked as open follow-ups. Shown as a new \"Chip Sequence\" section on the Chip Plan "
                  "tab whenever >=2 chip types remain available this half. Previously, Patch 84: manager "
                  "screenshot: paged the pitch navigator to GW7 with \"After recommended "
                  "transfer (this week's move)\" selected — the pitch still showed a player (Gomez) the Transfer "
                  "Recommendations panel's own chained pacing plan said should already be gone by GW7 (Gomez -> "
                  "Groß). Root cause, confirmed by reading _render_pitch_navigator(): the toggle only ever "
                  "reconstructed weekly_plan[0]'s move (this week's move at planning_gw) into one static squad, "
                  "reused unchanged for every later GW the stepper paged to — only the xPts projection column "
                  "changed, never the squad. Fixed: the toggle (relabeled \"After recommended transfers (chained "
                  "plan)\") now reconstructs a per-GW CUMULATIVE squad, replaying weekly_plan's moves in order, so "
                  "paging to GW7 stacks GW7's move on top of GW6's, and a no-move \"Roll\" week correctly carries "
                  "the prior week's squad forward unchanged. Also fixed, found while Playwright-verifying the "
                  "above (not manager-reported): the ▶ stepper button's index update ran AFTER the GW counter "
                  "label rendered, so a ▶ click showed the right GW's xPts/Rating/squad below but the counter "
                  "text lagged one click behind (◀ was unaffected — its handler already ran before the counter). "
                  "Previously, Patch 83: model: v6.9 Standing Rule #46 Fixture-Adjusted Attack, team-strength tier — "
                  "npxG/xA are now scaled by a fixed-strength (s=0.6) opponent-defence factor built from real "
                  "per-GW team-match xG data (olbauday/FPL-Core-Insights), shrunk toward the league average for "
                  "small samples and clamped to [0.7, 1.4]; clean sheets/bonus/DEFCON/cards untouched, GK "
                  "excluded, per Rule #46(d); Rule #47 clean-sheet tier reconfirmed compliant via its existing "
                  "team-strength fallback and now explicitly disclosed. SCOPE: team-strength tier only this "
                  "patch — the doc's market-odds leg and GW-distance decay are an explicit, tracked fast-follow, "
                  "not yet built; every run now shows a fixture-adjustment disclosure line on the Pitch tab "
                  "naming the tier, strength, and real-data coverage. Previously, Patch 82: performance: capped "
                  "every MILP solve at a configurable 12s time limit — fixes the Horizon=3 \"takes forever\"/"
                  "Streamlit-Cloud-crash hang — and cached solve_xi_first_squad() (the Free Hit optimal-squad "
                  "solver), same fix Patch 42 already gave solve_squad(), ~580x faster on repeat calls; model: "
                  "adopted v6.9's amended Wildcard trigger — retires the old 79%-ceiling/15xPts dual-leg check "
                  "for a single scale-free 6%-gap threshold, and the reachable ceiling now accrues one free "
                  "transfer per GW across the detection window instead of reusing one static snapshot for "
                  "every week)")

# Patch 78 (manager feedback: "why under the logo we are seeing this" —
# screenshot showed the full PATCH_VERSION technical changelog sentence
# rendered raw under the header logo). The deploy-confirmation STAMP itself
# (line 38's original reasoning: "a permanent, visible version stamp so
# [whether a fix redeployed] is answerable at a glance") is still genuinely
# useful and is kept — just as a short "Patch NN" tag instead of the whole
# internal changelog sentence, which now only shows on hover (title=) and in
# the full PATCH_VERSION string used everywhere else (diagnostics, forensics,
# model_config.yaml changelog). Regex, not a second hand-maintained constant,
# so it can never drift out of sync with PATCH_VERSION.
_PATCH_TAG_MATCH = re.search(r"^Patch\s+\d+[a-z]?", PATCH_VERSION)
PATCH_TAG = _PATCH_TAG_MATCH.group(0) if _PATCH_TAG_MATCH else PATCH_VERSION

st.set_page_config(page_title="RB Model", page_icon="⚽", layout="wide")

# Crest mark — "Monogram + Dot" (Release 2, replacing the Patch 2 chess-
# pawn-style crest as part of the manager-directed theme/logo overhaul): a
# rounded dark badge, a bold "R" in the primary everyday signal color, and a
# small coral dot standing in for the differential/captaincy signal — the
# two-color language the rest of the redesigned app now uses throughout.
# Raw SVG fill/stroke attributes can't read a CSS custom property, so this
# is one of the handful of genuinely hardcoded spots a :root-only retint
# always misses (Release 2.1's own changelog flagged the same class of gap)
# -- retinted by hand for Release 3 "Turf Green": badge bg/stroke now a
# dark green-black (matching --ink's hue family instead of the old
# near-neutral black), "R" now a bright grass green readable on that dark
# bg (the old cyan #2FD1D9 would clash with the new green accent), and the
# dot now uses the exact --coral hex (#E14E54) instead of a slightly
# different hardcoded coral (#FF5A5F) that was never actually the same
# color as the rest of the app's coral -- a genuine, if minor, "everything
# aligned" fix the manager asked for.
_CREST_SVG = ('<svg viewBox="0 0 100 100" width="28" height="28" style="flex:none;">'
              '<rect x="4" y="4" width="92" height="92" rx="24" fill="#152016" stroke="#233524" stroke-width="3"/>'
              '<text x="42" y="70" font-family="\'Space Grotesk\',sans-serif" font-weight="700" '
              'font-size="58" fill="#3FCB7C" text-anchor="middle">R</text>'
              '<circle cx="76" cy="76" r="10" fill="#E14E54"/></svg>')

# ---------------------------------------------------------------------------
# Style — a considered, precise identity (terse micro-copy, restraint, one
# accent per screen) expressed through football's own vocabulary (Patch 2)
# rather than chess — same underlying "soul," different, more natural words.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ---------------------------------------------------------------------
   Release 2 (2026-09-21) then Release 2.1 (same day, manager: after the
   dark "Signal/Differential" build ("Patch 61) plus its config.toml fix
   (Patch 62) still read as "still the same issue after the new patch and
   rebooting the app, also it's too dark ... modify it to be more friendly
   ... it's needed to be more clear not dark like this" — a direct design
   reversal, not another contrast bug. This is that: a light, friendly
   theme ("Daylight/Differential") replacing the dark canvas outright,
   while KEEPING every structural Release 2 upgrade that was never about
   dark-vs-light in the first place — the "Monogram + Dot" logo, the real
   CSS-drawn pitch markings, the captaincy spotlight card, and the
   transfer OUT->IN card. Cyan stays the primary everyday signal color and
   coral stays reserved for captaincy + alerts, just recalibrated for
   contrast against a light background instead of a dark one (a raw
   neon-cyan #2FD1D9 reads fine on near-black but is too washed-out for
   body text on white, so --accent-strong -- the shade actually used for
   text/headings -- is a deeper teal here, while --accent itself stays the
   brighter shade for bars/rings/borders where a lighter tone is fine).
   Because almost every component below reads its colors from these CSS
   custom properties instead of hardcoded hex, retinting :root here
   re-skins the overwhelming majority of the app automatically — see the
   Release 2.1 changelog entry in model_config.yaml for the handful of
   component-level touch-ups (hardcoded dark-tint text colors, avatar-
   fallback gradient, pitch markings color) this still required.
   --------------------------------------------------------------------- */
/* Release 3 "Turf Green" (2026-09-27, manager: sent 3 full proof-of-concept
   themes -- Kit Navy / Turf Green / Under Lights, plus a "less text, more
   visual" pass -- and picked Turf Green: "let's go"). Only the brand-accent
   family (--bg, --surface, --surface-2, --ink family, --rule, --accent
   family) actually changes here; --gold, --blue, --coral, --warn, the
   --fdr- family, and --violet stay
   identical to Release 2 (kept the same in every POC theme sent -- they're
   functional/semantic colors, not brand identity, and re-deriving them per
   theme risks breaking the position-tag/FDR-dot color language the manager
   never asked to change). Because almost every component reads these as
   CSS custom properties rather than hardcoded hex, this block is most of
   the retint — the handful of genuinely hardcoded spots (crest SVG,
   avatar-fallback gradient end, the Season Rank chart's Python-side color
   constants) are called out and fixed at their own call sites below, the
   same short list Release 2.1's own changelog entry already flagged as the
   thing a :root-only retint always misses. */
:root{
  --bg:#F6F9F1; --surface:#FFFFFF; --surface-2:#EDF3E6;
  --ink:#17241A; --ink-muted:#4B5C43; --ink-faint:#86937D;
  --rule:#DEE8D2; --accent:#2E9E5B; --accent-strong:#157A3E; --accent-tint:#E1F5E7;
  --gold:#C98A26; --gold-tint:#FBEFDC;
  --blue:#2E6E93; --blue-tint:#E1EDF3;
  --coral:#E14E54; --coral-tint:#FBE6E6;
  --warn:#E14E54; --bench:#F1F3ED;
  --fdr-easy:#2E8B57; --fdr-mid:#C98A26; --fdr-hard:#E14E54;
  --shadow:0 1px 2px rgba(23,36,26,.06), 0 8px 22px -12px rgba(23,36,26,.16);
  /* Release 2 — position-tag-only neutral accent (FWD label), kept
     separate from --coral now that coral is reserved for captaincy +
     alerts and shouldn't double as a fourth position color. */
  --violet:#7A6BC2;
}
html, body, [class*="css"]{ font-family:"IBM Plex Sans",sans-serif; color:var(--ink); }
.mono{ font-family:"IBM Plex Mono",monospace; }
.stApp{ background:var(--bg); }

/* Release 2.1 follow-up (2026-09-21, manager screenshot: a visible border
   box around the GW pagination label and around the GW xPts/Rating
   st.metric pair — neither of which this app's own CSS ever gave a
   border; confirmed in code, this file has no `border=True` on any
   st.columns/st.container call and no rule targeting those testids).
   requirements.txt pins only `streamlit>=1.38` with no upper bound, so
   Streamlit Community Cloud installs whatever the newest matching release
   is at deploy time — not necessarily the exact version this was tested
   against locally — and Streamlit has, in some releases, changed the
   default chrome around st.metric/st.columns to include a subtle border.
   Rather than chase the exact version, this resets that native chrome
   explicitly so the app's own borders (the ones this CSS actually draws,
   on .card/.tx-card/.cap-spotlight/etc.) are the only ones ever visible,
   regardless of which Streamlit release is actually deployed. */
[data-testid="stMetric"], [data-testid="stHorizontalBlock"], [data-testid="column"],
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]{
  border:none !important; box-shadow:none !important; background:transparent !important;
}

.brand-row{ display:flex; align-items:center; gap:8px; }
.brand-mark{ font-family:"Space Grotesk"; font-weight:700; font-size:2rem; line-height:1; margin-bottom:2px; }
.brand-mark .b2{ color:var(--accent); }
.brand-tag{ font-family:"IBM Plex Mono"; font-size:10.5px; color:var(--ink-faint); letter-spacing:.06em; text-transform:uppercase; }

.verdict-card{
  background:linear-gradient(160deg, var(--surface) 55%, var(--accent-tint));
  border:1px solid var(--rule); border-left:4px solid var(--accent);
  box-shadow:var(--shadow); padding:18px 20px; margin-bottom:6px;
}
.verdict-card .phase-tag{ font-family:"IBM Plex Mono"; font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:4px; display:block; }
.verdict-card .h{ font-family:"Space Grotesk"; font-weight:700; font-size:1.35rem; margin:0 0 6px; color:var(--accent-strong); }
.verdict-card .b{ margin:0; color:var(--ink-muted); font-size:.94rem; font-style:italic; }

/* Patch 80 (manager, live screenshot, annotated "what is this" over the two
   rating gauges): nowrap+overflow-x:auto meant a narrower viewport could
   cram or scroll-clip these before a reader ever saw their labels --
   switched to wrap so they always keep their full label visible, plus a
   slightly larger gap so the two "rating" gauges (see .gauge-ring below)
   read as two distinct things at a glance, not one ambiguous cluster. */
.stat-row{ display:flex; gap:26px; font-family:"IBM Plex Mono"; margin:14px 0 26px; flex-wrap:wrap; row-gap:16px; align-items:flex-start; }
.stat .n{ font-size:1.5rem; font-weight:600; color:var(--ink); }
.stat .l{ font-size:10.5px; color:var(--ink-faint); text-transform:uppercase; letter-spacing:.06em; }
.stat.rating .n{ display:flex; align-items:center; gap:6px; }
.stat.new .n{ color:var(--accent-strong); }
.trend-up{ color:var(--accent); } .trend-down{ color:var(--warn); }

/* Patch 2 — hover-hidden reasoning: rule citations / methodology notes move
   behind this "i" affordance instead of sitting as a permanent caption. */
.info-dot{ width:15px; height:15px; border-radius:50%; background:var(--surface-2); border:1px solid var(--rule);
  color:var(--ink-muted); font-family:"IBM Plex Sans"; font-size:10px; font-weight:700; display:inline-flex;
  align-items:center; justify-content:center; cursor:help; flex:none; }
.rating-basis{ font-family:"IBM Plex Sans"; font-size:10px; color:var(--ink-faint); margin-top:3px; max-width:150px; line-height:1.3; }

.section-h{ font-family:"Space Grotesk"; font-weight:700; font-size:1.15rem; margin:30px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--rule); }

.chip-rack{ display:flex; gap:10px; flex-wrap:wrap; margin:0 0 8px; }
.chip{ display:flex; align-items:center; gap:7px; background:var(--surface); border:1px solid var(--rule);
  padding:6px 11px; font-family:"IBM Plex Mono"; font-size:11.5px; box-shadow:var(--shadow); }
.chip .dot{ width:7px; height:7px; border-radius:50%; flex:none; }
.chip.available .dot{ background:var(--accent); }
.chip.used .dot{ background:var(--ink-faint); }
.chip.flagged .dot{ background:var(--gold); }
.chip.used{ color:var(--ink-faint); }
.chip.used .name{ text-decoration:line-through; }

/* Release 2 — real pitch markings (halfway line, center circle, two penalty
   boxes) drawn entirely in CSS via layered gradients/borders on ::before,
   plus a subtle grass-stripe backdrop, replacing the old flat single-tint
   rectangle. No image asset, still zero-cost. */
/* Patch 80 (manager, live screenshot, twice: "pitch is too big and doesn't
   look good at all!!") -- confirmed in code: .pitch had no max-width, so on
   Streamlit's wide layout it stretched to the full main-column width (well
   over 1000px) while a typical 11-15 card squad only fills a fraction of
   that horizontally, and .prow's 18px margin-bottom compounded across 4
   formation rows + a bench strip read as a lot of near-empty green. Capped
   width + tighter row spacing below, centered in the column instead of
   stretched across it -- the CSS pitch-marking overlay (.pitch::before)
   uses percentages/insets relative to .pitch's own box, so it stays
   correctly proportioned at the smaller width without any change there. */
.pitch{ position:relative; z-index:0; overflow:hidden; max-width:760px; margin:0 auto;
  background:repeating-linear-gradient(180deg, #E4F0DE 0px, #E4F0DE 44px, #D8E8D1 44px, #D8E8D1 88px);
  border:1px solid var(--rule); border-radius:10px; padding:20px 14px 6px; }
.pitch::before{ content:""; position:absolute; inset:14px; pointer-events:none; z-index:-1;
  border:1.5px solid rgba(255,255,255,.85); border-radius:4px;
  background:
    linear-gradient(rgba(255,255,255,.85), rgba(255,255,255,.85)) center / 100% 1.5px no-repeat,
    radial-gradient(circle at center, transparent 34px, transparent 35.5px, rgba(255,255,255,.85) 35.5px, rgba(255,255,255,.85) 37px, transparent 37px) center / 96px 96px no-repeat,
    linear-gradient(rgba(255,255,255,.85), rgba(255,255,255,.85)) top / 46% 1.5px no-repeat,
    linear-gradient(rgba(255,255,255,.85), rgba(255,255,255,.85)) top 0 left 27% / 1.5px 15% no-repeat,
    linear-gradient(rgba(255,255,255,.85), rgba(255,255,255,.85)) top 0 right 27% / 1.5px 15% no-repeat,
    linear-gradient(rgba(255,255,255,.85), rgba(255,255,255,.85)) bottom / 46% 1.5px no-repeat,
    linear-gradient(rgba(255,255,255,.85), rgba(255,255,255,.85)) bottom 0 left 27% / 1.5px 15% no-repeat,
    linear-gradient(rgba(255,255,255,.85), rgba(255,255,255,.85)) bottom 0 right 27% / 1.5px 15% no-repeat;
}
.prow{ display:flex; justify-content:center; gap:12px; margin-bottom:10px; flex-wrap:wrap; }

/* Patch 4 — card redesign: tighter top-cropped photo in a team-color ring,
   name-first info hierarchy (name -> compact pos+opponent meta line -> xPts
   as the dominant stat with price as a quiet footnote), replacing the old
   five-equal-weight stacked-line layout. */
.card{ background:var(--surface); border:1px solid var(--rule); box-shadow:var(--shadow); border-radius:10px;
  width:clamp(64px, 15vw, 112px); padding:10px 8px 9px; text-align:center; position:relative; }
.card .cap{ position:absolute; top:-9px; right:-9px; width:20px; height:20px; border-radius:50%;
  background:var(--coral); color:#2A0C0D; font-family:"IBM Plex Mono"; font-size:10.5px; font-weight:700;
  display:flex; align-items:center; justify-content:center; box-shadow:var(--shadow); z-index:2; }
.card .cap-actual{ position:absolute; bottom:-8px; right:-8px; width:16px; height:16px; border-radius:50%;
  background:var(--surface); border:2px solid var(--ink-faint); color:var(--ink-muted); font-family:"IBM Plex Mono";
  font-size:8px; font-weight:700; display:flex; align-items:center; justify-content:center; z-index:2; cursor:help; }
.card .sp{ position:absolute; top:5px; left:5px; font-family:"IBM Plex Mono"; font-size:7.5px; font-weight:700;
  color:var(--gold); border:1px solid var(--gold); border-radius:2px; padding:0 3px; }
.photo-ring{ width:48px; height:48px; border-radius:50%; margin:0 auto 7px; padding:2px;
  background:var(--team,var(--accent)); position:relative; }
.photo-ring img{ width:100%; height:100%; border-radius:50%; object-fit:cover; object-position:center 12%;
  display:block; border:2px solid var(--surface); }
.photo-ring .avatar-fallback{ position:absolute; inset:2px; border-radius:50%;
  /* #0F2417 (was #0B2C2E, a teal-black) -- Release 3 "Turf Green": the
     gradient's dark end is hardcoded (a CSS gradient stop can't read
     var(--ink) here without a second custom property per team color), so
     retinted by hand to a dark green-black matching the new --ink hue
     family instead of the old theme's teal-black. */
  background:linear-gradient(160deg, var(--team,var(--accent)), #0F2417); color:#fff; font-family:"Space Grotesk";
  font-weight:700; font-size:13px; align-items:center; justify-content:center; border:2px solid var(--surface); }
.card .name{ font-weight:700; font-size:12.5px; margin-bottom:2px; }
.card .meta{ font-family:"IBM Plex Mono"; font-size:8.5px; color:var(--ink-muted); margin-bottom:6px; }
.card .meta .pos{ font-weight:700; }
.card .meta .pos.gk{ color:var(--gold); } .card .meta .pos.def{ color:var(--blue); }
.card .meta .pos.mid{ color:var(--accent-strong); } .card .meta .pos.fwd{ color:var(--violet); }
.card .ticker{ display:flex; justify-content:center; gap:3px; margin-bottom:6px; }
.card .fdr-dot{ width:7px; height:7px; border-radius:50%; cursor:help; flex:none; }
.card .fdr-dot.easy{ background:var(--fdr-easy); } .card .fdr-dot.mid{ background:var(--fdr-mid); }
.card .fdr-dot.hard{ background:var(--fdr-hard); } .card .fdr-dot.blank{ background:var(--ink-faint); opacity:.4; }
.card .stats-row{ display:flex; align-items:baseline; justify-content:center; gap:6px; }
.card .xp{ font-family:"IBM Plex Mono"; font-size:14px; font-weight:700; color:var(--accent-strong); }
.card .xp-l{ font-family:"IBM Plex Mono"; font-size:7px; color:var(--ink-faint); text-transform:uppercase; display:block; margin-top:-2px; }
.card .price{ font-family:"IBM Plex Mono"; font-size:8.5px; color:var(--ink-faint); }
.bench-strip{ position:relative; z-index:1; background:var(--bench); margin:0 -14px; padding:12px 14px 4px;
  border-top:1px dashed var(--rule); border-radius:0 0 10px 10px; }
.bench-strip .card{ opacity:.68; width:clamp(56px, 13vw, 96px); }
.side-note{ font-size:11.5px; color:var(--ink-faint); font-family:"IBM Plex Mono"; line-height:1.5; }
/* Patch 80 -- small bordered/tinted box for sidebar caption text (see the
   Style-profile description call site) so it reads as a distinct piece of
   info like every other card/pill/chip in the app, not orphaned floating
   text. */
.side-info-box{ font-size:11.5px; color:var(--ink-muted); line-height:1.5; background:var(--surface-2);
  border:1px solid var(--rule); border-left:3px solid var(--accent); border-radius:6px;
  padding:8px 10px; margin:6px 0 16px; }

/* Patch 31 — Chip Signals grid: replaces the old paragraph-per-rule Chip
   Advisor/Chip Strategy text with compact scannable cards. Rule/step
   citations move into the card's `title` tooltip (hover-hidden, same
   pattern as .info-dot above) instead of sitting in visible text. */
.signal-grid{ display:flex; gap:12px; flex-wrap:wrap; margin:4px 0 18px; }
.signal-card{ background:var(--surface); border:1px solid var(--rule); border-left:4px solid var(--rule);
  box-shadow:var(--shadow); padding:12px 14px; min-width:150px; flex:1 1 150px; cursor:help; }
.signal-card.is-play{ border-left-color:var(--accent); }
.signal-card.is-active{ border-left-color:var(--gold); }
.signal-card.is-caution{ border-left-color:var(--coral); }
.signal-card.is-used{ opacity:.55; }
.signal-card .top{ display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:7px; }
.signal-card .name{ font-family:"IBM Plex Mono"; font-size:10.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:.04em; color:var(--ink-muted); }
.signal-card .stat{ font-family:"Space Grotesk"; font-weight:700; font-size:1.4rem; color:var(--accent-strong); line-height:1; }
.signal-card .sub{ font-size:10.5px; color:var(--ink-faint); margin-top:4px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.signal-card .note{ font-size:11px; font-weight:700; margin-top:8px; padding-top:8px; border-top:1px dashed var(--rule); }
.signal-card .note.good{ color:var(--accent-strong); }
.signal-card .note.warn{ color:var(--coral); }
.badge{ font-family:"IBM Plex Mono"; font-size:9px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  padding:2px 7px; border-radius:20px; display:inline-block; white-space:nowrap; }
.badge.play{ background:var(--accent-tint); color:var(--accent-strong); }
.badge.active{ background:var(--gold-tint); color:#7A5A16; }
.badge.hold{ background:var(--surface-2); color:var(--ink-faint); }
.badge.caution{ background:var(--coral-tint); color:var(--coral); }
.badge.used{ background:var(--surface-2); color:var(--ink-faint); }

.flag-row{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 16px; }
/* Patch 54 (2026-09-17, manager decision: "keep red boxes but with
   something better looking") — the old version painted EVERY pill in solid
   coral regardless of content, so a purely informational note ("Wildcard
   trigger: not active", "Shape-test: wildcard shaped") read with the same
   visual urgency as a genuine actionable flag ("Disruption check: N
   player(s) flagged", "Price-drop-flow override: ..."). Split into two
   tiers instead: a quiet neutral style by default, and a softer gold/amber
   tier reserved for pills whose icon is already the warning triangle
   (_flag_pill's own existing "flagged"/"CAUTION"/"Disruption" keyword
   check) — the same gold already used elsewhere in this app for "flagged"
   status (.chip.flagged's dot color), so this isn't a new color language,
   just applying the one that already exists consistently here too. */
.flag-pill{ display:flex; align-items:flex-start; gap:6px; background:var(--surface-2); border:1px solid var(--rule);
  color:var(--ink-muted); font-family:"IBM Plex Mono"; font-size:11px; padding:6px 10px; border-radius:14px;
  cursor:help; white-space:normal; max-width:420px; line-height:1.5; text-align:left; }
.flag-pill.warn{ background:var(--gold-tint); border:1px solid var(--gold); color:#7A5A16; }

/* Latest News feed (Patch 67) — real injury/status news pulled straight from
   the official FPL API's own per-player `status`/`news`/`news_added`/
   `chance_of_playing_next_round` fields (already fetched for other purposes;
   this is the first place the raw news text itself is actually shown). One
   `.news-item` per flagged player, left-border colored by severity so the
   list is scannable without reading every line. */
.news-item{ display:flex; gap:12px; align-items:flex-start; background:var(--surface); border:1px solid var(--rule);
  border-left:4px solid var(--ink-faint); border-radius:10px; padding:12px 14px; margin-bottom:9px; box-shadow:var(--shadow); }
.news-item.sev-injured{ border-left-color:var(--coral); }
.news-item.sev-doubtful{ border-left-color:var(--gold); }
.news-item.sev-suspended{ border-left-color:var(--violet); }
.news-item.sev-unavailable{ border-left-color:var(--ink-faint); }
.news-item .ring{ width:38px; height:38px; border-radius:50%; flex:none; overflow:hidden; background:var(--surface-2); position:relative; }
.news-item .ring img{ width:100%; height:100%; object-fit:cover; object-position:center 12%; }
.news-item .ring .avatar-fallback{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-family:"IBM Plex Mono"; font-weight:700; font-size:11px; color:var(--ink-faint); }
.news-item .body{ flex:1; min-width:0; }
.news-item .top-row{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.news-item .name{ font-family:"Space Grotesk"; font-weight:700; font-size:.92rem; color:var(--ink); }
.news-item .team{ font-family:"IBM Plex Mono"; font-size:10px; color:var(--ink-faint); }
.news-item .status-badge{ font-family:"IBM Plex Mono"; font-size:9.5px; text-transform:uppercase; letter-spacing:.04em;
  padding:2px 7px; border-radius:8px; font-weight:700; }
.news-item .status-badge.sev-injured{ background:var(--coral-tint); color:#7A1518; }
.news-item .status-badge.sev-doubtful{ background:var(--gold-tint); color:#7A5A16; }
.news-item .status-badge.sev-suspended{ background:#EEE9FA; color:#3E2E7A; }
.news-item .status-badge.sev-unavailable{ background:var(--surface-2); color:var(--ink-faint); }
.news-item .scope-tag{ font-family:"IBM Plex Mono"; font-size:9.5px; text-transform:uppercase; letter-spacing:.04em;
  padding:2px 7px; border-radius:8px; background:var(--accent-tint); color:var(--accent-strong); font-weight:700; }
.news-item .cop{ font-family:"IBM Plex Mono"; font-size:10px; color:var(--ink-muted); }
.news-item .text{ margin-top:5px; font-size:.85rem; color:var(--ink); line-height:1.4; }
.news-item .ts{ margin-top:5px; font-family:"IBM Plex Mono"; font-size:9.5px; color:var(--ink-faint); }

/* Team Rating % radial gauge (Patch 31) — conic-gradient ring, no SVG/JS
   library needed. Percentage is still the same number the tooltip/expander
   already computed; this only changes how it's presented. */
.gauge-wrap{ display:flex; align-items:center; gap:10px; }
/* Patch 80 -- enlarged 44px->54px and a thicker band (inset 5px->7px): at
   the old size, two same-color rings sitting close together (both a shade
   of green for a healthy percentage) read as near-identical and prompted
   the manager's "what is this" -- bigger + a visibly thicker colored arc
   makes each one legible as an actual gauge, not just a number in a
   circle. */
.gauge-ring{ width:54px; height:54px; border-radius:50%; flex:none;
  display:flex; align-items:center; justify-content:center; position:relative; }
.gauge-ring::before{ content:""; position:absolute; inset:7px; border-radius:50%; background:var(--bg); }
.gauge-ring .gauge-val{ position:relative; z-index:1; font-family:"IBM Plex Mono"; font-size:12px; font-weight:700; }

/* Patch 33 — Chip Signal cards become <details>/<summary> so the "why this
   GW" per-GW breakdown (manager report: "why GW9 ... this isn't clear")
   expands in place, no JS needed. The card's existing top/stat/sub markup
   moves inside <summary>; a small CSS bar-chart of the scanned window sits
   in the revealed body, winning GW highlighted. */
details.signal-card{ padding:0; }
details.signal-card summary{ padding:12px 14px; list-style:none; cursor:pointer; }
details.signal-card summary::-webkit-details-marker{ display:none; }
details.signal-card summary::after{ content:"▾ breakdown"; display:block; font-family:"IBM Plex Mono";
  font-size:9px; color:var(--ink-faint); margin-top:6px; text-transform:uppercase; letter-spacing:.04em; }
details.signal-card[open] summary::after{ content:"▴ hide"; }
.gw-bars{ display:flex; gap:5px; align-items:flex-end; height:56px; padding:18px 14px 20px; }
.gw-bars .bar{ flex:1; background:var(--surface-2); border-radius:2px 2px 0 0; position:relative; min-height:2px; }
.gw-bars .bar.win{ background:var(--accent-strong); }
.gw-bars .bar .val{ position:absolute; top:-15px; left:0; right:0; text-align:center;
  font-family:"IBM Plex Mono"; font-size:8.5px; color:var(--ink-muted); white-space:nowrap; }
.gw-bars .bar .lbl{ position:absolute; bottom:-16px; left:0; right:0; text-align:center;
  font-family:"IBM Plex Mono"; font-size:8.5px; color:var(--ink-faint); }

/* Patch 33 — xM rotation-risk badge, inline on a transfer recommendation
   (manager report: a transfer's xPts already factors in expected minutes,
   but that wasn't disclosed next to the recommendation itself). Reuses the
   existing .badge tiers rather than inventing new colors. */
.xm-badge{ font-family:"IBM Plex Mono"; font-size:8.5px; font-weight:700; letter-spacing:.02em;
  padding:1px 5px; border-radius:10px; margin-left:4px; white-space:nowrap; display:inline-block;
  vertical-align:middle; cursor:help; }
.xm-badge.nailed{ background:var(--accent-tint); color:var(--accent-strong); }
.xm-badge.rotation{ background:var(--gold-tint); color:#7A5A16; }
.xm-badge.risk{ background:var(--coral-tint); color:var(--coral); }

/* Patch 33 — post-transfer preview, promoted out of the "Why" expander to
   sit directly under each recommendation (manager: this already existed
   but was buried and read as missing). */
.tx-preview{ font-size:12.5px; color:var(--ink-muted); background:var(--surface-2);
  border-left:3px solid var(--accent); padding:7px 12px; margin:2px 0 10px 0; }
.tx-preview b{ color:var(--accent-strong); }

/* Patch 4 — captaincy-on-pitch caption. Kept as a fallback text style (still
   used for the disclosed team-stability tiebreak footnote alongside the new
   spotlight card below) — Release 2 recolors its accent bar to coral, since
   captaincy is now coral's one reserved everyday use. */
.cap-caption{ font-size:13.5px; color:var(--ink); background:var(--surface); border-left:3px solid var(--coral);
  box-shadow:var(--shadow); padding:9px 14px; margin-top:10px; border-radius:0 6px 6px 0; }
.cap-caption b{ color:var(--accent-strong); }

/* Patch 5 — Transfer Recommendations simplification. Kept as the style used
   by the "Evaluate your own scenario" section's own manager-directed what-
   ifs (target/Wildcard/Free Hit) — the main recommendation now uses the new
   .tx-card OUT->IN component below instead. */
.tx-reco{ font-size:13.5px; color:var(--ink); background:var(--surface); border-left:3px solid var(--accent-strong);
  box-shadow:var(--shadow); padding:9px 14px; margin-top:6px; margin-bottom:6px; border-radius:0 6px 6px 0; }

/* ---------------------------------------------------------------------
   Release 2 — Captaincy spotlight card. Replaces the plain-text armband
   caption as the PRIMARY captaincy display (the caption above still
   carries the team-stability-tiebreak footnote when one applies, appended
   directly under the card, so nothing the old text version disclosed is
   dropped — only reorganized). Coral throughout, since this is coral's one
   reserved everyday role. Photo reuses the exact same crop the pitch cards
   already use (`_photo_url`), just re-ringed in coral instead of team
   color, to make the armband pick visually unmistakable at a glance.
   --------------------------------------------------------------------- */
.cap-spotlight{ background:linear-gradient(160deg, var(--coral-tint), var(--surface) 65%);
  border:1px solid var(--rule); border-left:4px solid var(--coral); border-radius:10px;
  box-shadow:var(--shadow); padding:14px 16px; margin-top:10px; }
.cap-spotlight .row{ display:flex; align-items:center; gap:11px; }
.cap-spotlight .ring{ width:44px; height:44px; border-radius:50%; padding:2px; flex:none; background:var(--coral); position:relative; }
.cap-spotlight .ring img{ width:100%; height:100%; border-radius:50%; object-fit:cover; object-position:center 12%;
  display:block; border:2px solid var(--surface); }
.cap-spotlight .ring .avatar-fallback{ position:absolute; inset:2px; border-radius:50%; display:none;
  align-items:center; justify-content:center; background:linear-gradient(160deg, var(--coral), #5A1A1C);
  color:#fff; font-family:"Space Grotesk"; font-weight:700; font-size:14px; border:2px solid var(--surface); }
.cap-spotlight .name{ font-family:"Space Grotesk"; font-weight:700; font-size:.95rem; color:var(--ink); }
.cap-spotlight .meta{ font-family:"IBM Plex Mono"; font-size:9.5px; color:var(--ink-faint); margin-top:1px; }
.cap-spotlight .tag{ font-family:"IBM Plex Mono"; font-size:9px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--coral); font-weight:700; margin-bottom:2px; display:block; }
.cap-spotlight .xp-row{ display:flex; align-items:baseline; gap:6px; margin-top:10px; }
.cap-spotlight .xp{ font-family:"IBM Plex Mono"; font-weight:700; font-size:1.5rem; color:var(--coral); line-height:1; }
.cap-spotlight .xp-u{ font-family:"IBM Plex Mono"; font-size:10px; color:var(--ink-faint); }
.cap-spotlight .bar-track{ height:6px; background:var(--surface-2); border-radius:3px; margin-top:8px; overflow:hidden; }
.cap-spotlight .bar-fill{ height:100%; border-radius:3px; background:var(--coral); }
.cap-spotlight .eo-row{ display:flex; justify-content:space-between; font-family:"IBM Plex Mono"; font-size:10px;
  color:var(--ink-faint); margin-top:5px; }
.cap-spotlight .alt-sub{ margin-top:11px; padding-top:10px; border-top:1px dashed var(--rule);
  font-size:12px; color:var(--ink-muted); }
.cap-spotlight .alt-sub b{ color:var(--ink); }

/* ---------------------------------------------------------------------
   Release 2 — Transfer OUT->IN card. Replaces the plain-text main
   recommendation sentence with an OUT->IN photo card, an arrow, a net-xPts
   number, and a FREE/hit-cost pill. The rule-citation trace/raw move table
   stay exactly where they were (the "Why" expander) — this only replaces
   the always-visible summary line, and every field it shows (out/in
   player, position, net xPts, hit cost) was already disclosed in that
   text, just reorganized visually, not reduced. Cyan throughout — this is
   an everyday signal, not a captaincy/alert use.
   --------------------------------------------------------------------- */
.tx-card{ background:var(--surface); border:1px solid var(--rule); border-radius:10px;
  box-shadow:var(--shadow); padding:12px 15px; margin-top:6px; margin-bottom:10px; }
.tx-card .top{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.tx-card .tag{ font-family:"IBM Plex Mono"; font-size:9.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--ink-faint); }
.tx-card .hit-pill{ font-family:"IBM Plex Mono"; font-size:9.5px; font-weight:700; padding:2px 8px; border-radius:20px; }
.tx-card .hit-pill.free{ background:var(--accent-tint); color:var(--accent-strong); }
.tx-card .hit-pill.hit{ background:var(--coral-tint); color:var(--coral); }
.tx-swap{ display:flex; align-items:center; gap:10px; }
.tx-player{ flex:1; display:flex; flex-direction:column; align-items:center; text-align:center; min-width:0; }
.tx-player .ring{ width:38px; height:38px; border-radius:50%; padding:2px; margin-bottom:5px; flex:none; position:relative; }
.tx-player.out .ring{ background:var(--rule); }
.tx-player.in .ring{ background:var(--accent); }
.tx-player .ring img{ width:100%; height:100%; border-radius:50%; object-fit:cover; object-position:center 12%;
  display:block; border:2px solid var(--surface); }
.tx-player .ring .avatar-fallback{ position:absolute; inset:2px; border-radius:50%; display:none;
  align-items:center; justify-content:center; color:#fff; font-family:"Space Grotesk"; font-weight:700;
  font-size:12px; border:2px solid var(--surface); }
.tx-player.out .ring .avatar-fallback{ background:linear-gradient(160deg, var(--ink-faint), #3A3A40); }
.tx-player.in .ring .avatar-fallback{ background:linear-gradient(160deg, var(--accent), #0F2417); }
.tx-player .pname{ font-family:"Space Grotesk"; font-weight:600; font-size:.78rem; color:var(--ink); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; max-width:100%; }
.tx-player .pmeta{ font-family:"IBM Plex Mono"; font-size:8px; color:var(--ink-faint); }
.tx-arrow{ flex:none; display:flex; flex-direction:column; align-items:center; gap:3px; }
.tx-arrow svg{ display:block; }
.tx-net{ font-family:"IBM Plex Mono"; font-weight:700; font-size:.9rem; white-space:nowrap; color:var(--accent); }
.tx-net.neg{ color:var(--coral); }
.tx-net-l{ font-family:"IBM Plex Mono"; font-size:7px; color:var(--ink-faint); text-transform:uppercase; }

/* Patch 2 — mobile simplification: at narrow widths the card drops the price
   line entirely (least-needed info at this size — still visible in the GW
   Breakdown table) and shrinks text so name + position + opponent + xPts
   stay legible. .prow's flex-wrap (already set above) means a 5-defender
   row degrades to two lines here instead of a horizontal scrollbar. */
@media (max-width:480px){
  .card{ padding:7px 4px 6px; }
  .card .price{ display:none; }
  .card .name{ font-size:10.5px; }
  .card .xp{ font-size:12px; }
  .card .meta{ font-size:7.5px; }
  .photo-ring, .photo-ring img, .photo-ring .avatar-fallback{ width:32px; height:32px; }
  .photo-ring{ height:32px; }
  .photo-ring .avatar-fallback{ font-size:11px; }
  .card .cap{ width:16px; height:16px; font-size:9px; top:-7px; right:-7px; }
  .card .cap-actual{ width:13px; height:13px; font-size:7px; }
  .card .fdr-dot{ width:6px; height:6px; }
  .prow{ gap:6px; }
  .cap-spotlight .xp{ font-size:1.25rem; }
  .tx-player .pname{ font-size:.7rem; }
  .tx-net{ font-size:.8rem; }
}
</style>
""", unsafe_allow_html=True)


_ROW = {"ph": None, "items": None}


def _row_start():
    """Next chips go into ONE wrapping row (placeholder keeps its place; closed by _row_end)."""
    _ROW["ph"], _ROW["items"] = st.empty(), []


def _row_add(fragment_html):
    if _ROW["items"] is None:
        st.markdown(fragment_html, unsafe_allow_html=True)
        return
    _ROW["items"].append(fragment_html)
    _ROW["ph"].markdown(recommend.chips_row_html(_ROW["items"]), unsafe_allow_html=True)


def _row_end():
    _ROW["ph"], _ROW["items"] = None, None


def _note(label, body, kind="info"):
    """Short chip + tap-for-details (replaces long caption/info/warning text)."""
    body = str(body) if body is not None else None
    _row_add(recommend.note_html(label, kind, body))


def _xp_colcfg(rn):
    """Readable table formats: money as £5.0m, xPts to 1 dp (window total as a bar)."""
    cc = {}
    for k, v in rn.items():
        if k == "price":
            cc[v] = st.column_config.NumberColumn(v, format="£%.1fm")
        elif k == "_win_sum":
            cc[v] = st.column_config.ProgressColumn(v, format="%.1f", min_value=0, max_value=60)
        elif str(k).startswith("xpts_"):
            cc[v] = st.column_config.NumberColumn(v, format="%.1f")
    return cc


def _cfg_for(df):
    """column_config from the column names/types: money £5.0m, xPts 1 dp, totals as bars."""
    cc = {}
    try:
        kinds = recommend.table_formats(list(df.columns))
        for c, k in kinds.items():
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if k == "money":
                cc[c] = st.column_config.NumberColumn(str(c), format="£%.1fm")
            elif k == "bar":
                mx = float(pd.to_numeric(df[c], errors="coerce").max() or 0)
                cc[c] = st.column_config.ProgressColumn(str(c), format="%.1f", min_value=0, max_value=max(1.0, round(mx * 1.05, 1)))
            else:
                cc[c] = st.column_config.NumberColumn(str(c), format="%.1f")
    except Exception:
        return {}
    return cc


def _df(df, column_config=None, **kw):
    cc = _cfg_for(df) if isinstance(df, pd.DataFrame) else {}
    cc.update(column_config or {})
    st.dataframe(df, column_config=cc, **kw)


def _dm(summary, body_html, unsafe_allow_html=True):
    """Short summary line; the original longer html block opens on tap."""
    st.markdown(recommend.details_html(html.escape(str(summary)), body_html), unsafe_allow_html=True)


def _nc(label, body, help=None):
    _note(label, body, "info")


def _ni(label, body, help=None):
    _note(label, body, "info")


def _nw(label, body, help=None):
    _note(label, body, "warn")


def _ne(label, body, help=None):
    _note(label, body, "bad")


def _ns(label, body, help=None):
    _note(label, body, "ok")


def _team_color(short_name: str) -> str:
    palette = {
        "ARS": "#EF0107", "AVL": "#670E36", "BOU": "#DA291C", "BRE": "#e30613",
        "BHA": "#0057B8", "CHE": "#034694", "CRY": "#1B458F", "EVE": "#003399",
        "FUL": "#000000", "IPS": "#1B458C", "LEI": "#003090", "LIV": "#C8102E",
        "MCI": "#6CABDD", "MUN": "#DA291C", "NEW": "#241F20", "NFO": "#DD0000",
        "SOU": "#D71920", "TOT": "#132257", "WHU": "#7A263A", "WOL": "#FDB913",
        "BUR": "#6C1D45", "LEE": "#FFCD00", "SUN": "#eb172b",
    }
    return palette.get(short_name, "#1F6D45")


def _photo_url(code) -> str:
    return f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{int(code)}.png"


def _signal_card(name: str, badge_text: str, badge_cls: str, stat: str, sub: str,
                  tooltip: str, card_cls: str = "", by_gw: dict | None = None,
                  best_gw: int | None = None, note: str | None = None, note_cls: str = "") -> str:
    """Patch 31 — one compact scannable "Chip Signals" card, replacing a
    paragraph of Chip Advisor/Chip Strategy prose. Rule/step citations and
    the full quantified reasoning move into the card's `title` tooltip
    (hover-hidden, same pattern as the header's .info-dot) instead of
    sitting as permanent visible text — the visible surface is just a
    name, a status badge, one headline stat, and a one-line sub-caption.

    Patch 33 (manager report: "why GW9 ... this isn't clear" — a static
    tooltip only ever named the winning GW's margin over the runner-up, never
    the actual per-GW numbers). When `by_gw` ({gw: value}, 2+ entries) is
    given, the card becomes a <details>/<summary> — click to reveal a small
    CSS bar-chart of every scanned GW, the winning one highlighted, so "why
    this GW" is answered by the numbers themselves, not just a sentence.

    Patch 49 (manager report: a "your recommended transfers may already
    close this" finding existed but only lived in a small secondary pill —
    same size/weight as everything else in that row — not on the Wildcard
    card itself, where the trigger status it directly qualifies is shown at
    full prominence). `note`, when given, adds one more short line directly
    on the card, below the existing badge/stat/sub — same card, same
    tooltip, same badge text, nothing else changed, just one additional
    always-visible line at equal visual weight to the rest of the card."""
    body = (f'<div class="top"><span class="name">{name}</span>'
            f'<span class="badge {badge_cls}">{badge_text}</span></div>'
            f'<div class="stat">{stat}</div>'
            f'<div class="sub" title="{html.escape(re.sub(r"<[^>]+>", "", str(sub)), quote=True)}">{sub}</div>'
            + (f'<div class="note {note_cls}">{note}</div>' if note else ""))
    if by_gw and len(by_gw) >= 2:
        max_v = max(max(by_gw.values()), 0.01)
        bars = "".join(
            f'<div class="bar{" win" if gw == best_gw else ""}" '
            f'style="height:{max(4, round((v / max_v) * 100))}%;">'
            f'<span class="val">{v:.1f}</span><span class="lbl">GW{gw}</span></div>'
            for gw, v in sorted(by_gw.items())
        )
        return (f'<details class="signal-card {card_cls}"><summary title="{tooltip}">{body}</summary>'
                f'<div class="gw-bars">{bars}</div></details>')
    return f'<div class="signal-card {card_cls}" title="{tooltip}">{body}</div>'


def build_season_rank_chart(rank_df: pd.DataFrame) -> alt.Chart:
    """Patch 78 — Season Rank chart, rebuilt per manager request + the
    dataviz skill's method (log scale for order-of-magnitude data, mark
    specs, app-consistent color tokens). See the call site's comment block
    (render_dashboard(), "Season Rank" section) for the full root-cause/
    design story. Pulled out as its own module-level, independently
    testable function (rather than built inline) specifically so
    test_patch78_season_rank_chart.py can call it directly against
    synthetic data without needing to exec a slice of render_dashboard().

    `rank_df` must have exactly two columns, "GW" (int) and "Overall rank"
    (int, >=1 — a log scale cannot include 0, and FPL overall rank is never
    0 anyway). Returns a layered alt.Chart (area wash + line + points) with
    a reversed log y-axis pinned to the manager's own named tiers."""
    # Patch 80 (manager screenshot: "looking very bad, adjust and modify with
    # a good standards" on this exact chart) -- root cause confirmed in code:
    # domain_max used to be hardcoded to max(10_000_000, worst*1.05), i.e.
    # ALWAYS pinned out to the 10M tier no matter how good the manager's rank
    # actually is. For a manager sitting around 2.6M-3.2M (this account's
    # real GW1-5 history), that meant ~90% of the chart's vertical space (the
    # 1 -> 1,000,000 span) was dead space nobody's line ever visits, while
    # the real trend line was squeezed into a thin sliver at the very bottom
    # between the 1,000,000 and 10,000,000 gridlines -- which reads as "a
    # nearly-empty chart with a flat line stuck to the floor," exactly what
    # the screenshot showed. Fix: domain_max is now the SMALLEST named tier
    # that's still above the worst rank actually seen (falling back to
    # worst*1.05 only if the manager's rank is worse than every named tier),
    # so the reference lines always bracket the real data tightly instead of
    # a one-size-fits-all 10M ceiling. Added a 5,000,000 tier so a manager in
    # the 1M-5M range (not yet top-1M, but nowhere near the full 10M+ player
    # pool either) gets a sensibly-scaled chart too, not a 10x jump straight
    # to 10M.
    band_ticks = [1, 1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    worst = max(int(v) for v in rank_df["Overall rank"] if v is not None) if not rank_df.empty else 1
    domain_max = next((t for t in band_ticks if t > worst), int(worst * 1.05))
    tick_values = [t for t in band_ticks if t <= domain_max]
    if tick_values[-1] != domain_max:
        tick_values.append(domain_max)

    # Exact app CSS custom-property values (see the `:root{...}` block near
    # the top of this file) -- not new colors, so the chart stays visually
    # consistent with the rest of the app.
    # Release 3 "Turf Green" retint (was the Release 2 teal values) -- kept
    # as literal hex rather than reading st.markdown's injected CSS (Altair
    # can't see page CSS custom properties; this function's own docstring
    # already flagged these as "not new colors" pulled from :root, so they
    # must be updated by hand here whenever :root's theme changes).
    ink, ink_muted, rule = "#17241A", "#4B5C43", "#DEE8D2"
    accent_strong, accent_tint, surface = "#157A3E", "#E1F5E7", "#FFFFFF"

    log_scale = alt.Scale(type="log", domain=[1, domain_max], reverse=True, nice=False)

    area = (
        alt.Chart(rank_df)
        .mark_area(
            line=False, interpolate="monotone",
            color=alt.Gradient(gradient="linear",
                                stops=[alt.GradientStop(color=accent_tint, offset=0),
                                       alt.GradientStop(color=surface, offset=1)],
                                x1=1, x2=1, y1=1, y2=0),
            opacity=0.55,
        )
        .encode(x=alt.X("GW:O"), y=alt.Y("Overall rank:Q", scale=log_scale))
    )
    line = (
        alt.Chart(rank_df)
        .mark_line(interpolate="monotone", strokeWidth=2, color=accent_strong)
        .encode(
            x=alt.X("GW:O", title="Gameweek",
                     axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=13, labelFontWeight=600,
                                    labelColor=ink, titleColor=ink, domainColor=rule, tickColor=rule,
                                    grid=False)),
            y=alt.Y("Overall rank:Q", title="Overall rank", scale=log_scale,
                     axis=alt.Axis(values=tick_values, format=",.0f", labelFontSize=12, titleFontSize=13,
                                    labelColor=ink_muted, titleColor=ink, domainColor=rule, tickColor=rule,
                                    grid=True, gridColor=rule, gridOpacity=0.9, gridDash=[1, 0])),
        )
    )
    points = (
        alt.Chart(rank_df)
        .mark_point(filled=True, size=90, color=accent_strong, stroke=surface, strokeWidth=2)
        .encode(
            x=alt.X("GW:O"), y=alt.Y("Overall rank:Q", scale=log_scale),
            tooltip=[alt.Tooltip("GW:O", title="GW"),
                     alt.Tooltip("Overall rank:Q", title="Overall rank", format=",.0f")],
        )
    )
    return ((area + line + points).properties(height=320, background=surface)
             .configure_view(strokeWidth=0))


def _flag_pill(text: str, tooltip: str = "") -> str:
    """Patch 31 — a single compact pill for Chip Rack notes (Wildcard flag,
    shape-test, disruption, price-drop-flow) that used to render as a full
    <p class="side-note"> paragraph each. Full detail stays available on
    hover rather than being deleted.

    Patch 51 (2026-09-16, manager report + screenshot) — this used to
    hard-truncate anything past 70 characters with a bare "…", which cut
    FOUR different pill types off mid-sentence, mid-clause, right before
    their actual conclusion (the exact same class of bug Patch 48 already
    fixed for the Wildcard cross-check pill specifically, just never
    generalized to the other three: shape-test, disruption check, price-
    drop-flow). Rather than hand-craft a short verdict-first headline for
    each of those three (Patch 48's approach, only practical when a caller
    already knows its own conclusion is a short final clause), this now
    shows the FULL text on every pill, always — no truncation, nothing
    hidden — and lets the pill wrap onto multiple lines via the `.flag-pill`
    CSS's `white-space: normal` + `max-width` (below) instead of clipping.
    The hover tooltip still repeats the same full text for parity with
    Patch 48/49's pills, which pass a separate, even-more-detailed tooltip
    string."""
    is_warn = any(k in text for k in ("CAUTION", "flagged", "Disruption"))
    icon = "⚠️" if is_warn else "●"
    cls = "flag-pill warn" if is_warn else "flag-pill"
    return f'<span class="{cls}" title="{tooltip or text}">{icon} {text}</span>'


def _player_card(row: pd.Series, is_captain: bool = False, is_live_captain: bool = False,
                  xp_col: str | None = None, opp_col: str | None = None,
                  gw_list: list[int] | None = None) -> str:
    """Patch 4 card redesign: name-first info hierarchy — name, then a
    single compact "pos · opponent" meta line, then xPts as the dominant
    stat with price as a quiet footnote — plus a tight top-cropped photo in
    a team-color ring instead of a plain centered avatar.

    is_captain: model's recommended captain this run -> solid gold armband.
    is_live_captain: your actual live FPL captain, only ever passed True when
    it's a DIFFERENT player from the recommendation (Patch 2) -> a smaller
    hollow-ring secondary marker, so both are visible without implying the
    recommendation and your real team agree when they don't.

    gw_list: when given with more than one gameweek, the card shows a
    multi-GW fixture-difficulty ticker (one dot per GW, sourced from
    `fdr_gw{gw}`, hover for the exact opponent) instead of the single
    opponent chip — the "fixtures still show the current GW only" gap this
    patch closes. Falls back to the single-GW opponent chip (via `opp_col`)
    when `gw_list` is None or length 1, i.e. horizon = 1 behaves exactly as
    before."""
    team_color = _team_color(row.get("team", ""))
    initials = "".join([w[0] for w in str(row.get("web_name", "??")).split()][:2]).upper() or "??"
    xp = row.get(xp_col) if xp_col else row.get("xpts_horizon_sum")
    xp = 0.0 if pd.isna(xp) else xp
    cap_html = '<div class="cap">C</div>' if is_captain else ""
    cap_actual_html = ('<div class="cap-actual" title="Your live captain — the model recommends someone else this run">C</div>'
                        if is_live_captain else "")
    # Patch 80 (manager: "remove this", pointing at the permanent "SP tag = ..."
    # caption line that used to run under every pitch render regardless of
    # whether any player on it even had the SP tag) -- the explanation moves
    # onto the badge itself as a hover tooltip instead of a standing caption,
    # per the same "less text, more visual" direction as the earlier theme
    # proof-of-concept the manager approved.
    sp_html = ('<div class="sp" title="Newly confirmed set-piece role, decaying out as current-season minutes '
               'accrue.">SP</div>') if row.get("setpiece_flag") else ""
    pos = str(row.get("position", "")).lower()
    pos_label = row.get("position", "")
    price = row.get("price")
    price_html = f'<div class="price mono">£{price}m</div>' if price is not None else ""

    ticker_html = ""
    meta_right = row.get(opp_col) if opp_col else None
    # Patch 5 — the dominant "xp" stat on every pitch card is the CURRENT
    # planning-GW value (via xp_col, passed by the caller), matching exactly
    # what actually decided the starting XI and the captain armband (Step 7
    # / Step 8 are both single-GW), never the multi-GW horizon sum — that
    # mismatch (card showing a horizon total next to an armband picked on a
    # single-GW basis) was flagged as a real source of confusion. The
    # fixture ticker below is a SEPARATE multi-GW difficulty view (dots, no
    # points total) and stays independent of this.
    xp_label = f"GW{xp_col.split('gw')[1]} xp" if (xp_col and xp_col.startswith("xpts_gw")) else "xpts"
    if gw_list and len(gw_list) > 1:
        dots = []
        for gw in gw_list:
            opp_label = row.get(f"opp_gw{gw}", "") or "Blank"
            tier = row.get(f"fdr_gw{gw}", "") or "blank"
            dots.append(f'<span class="fdr-dot {tier}" title="GW{gw}: {opp_label}"></span>')
        ticker_html = f'<div class="ticker">{"".join(dots)}</div>'
        meta_right = f"{len(gw_list)}-GW horizon"
    meta_html = f'<div class="meta"><span class="pos {pos}">{pos_label}</span> · {meta_right}</div>' if meta_right else \
        f'<div class="meta"><span class="pos {pos}">{pos_label}</span></div>'

    # Built as ONE physical line, deliberately — a multi-line f-string here
    # (the pre-Patch-4 shape) put optional interpolations like `ticker_html`
    # alone on their own line, and whenever that value is "" (which it
    # always is at horizon=1, since the fixture ticker only renders for a
    # 2+ GW horizon) that line is blank/whitespace-only. Streamlit's
    # Markdown renderer treats a blank line inside a raw HTML block as the
    # end of that block (a CommonMark HTML-block rule) — so at horizon=1
    # every card past that point in the string silently vanished, which is
    # exactly the "only the GK shows, and only at horizon=1" bug this fixes.
    # Keeping the whole card on one line makes that class of bug structurally
    # impossible, regardless of which piece happens to be empty.
    return ('<div class="card" style="--team:' + team_color + '">' + cap_html + cap_actual_html + sp_html +
            '<div class="photo-ring">'
            '<img src="' + _photo_url(row.get('code', 0)) + '" '
            'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">'
            '<div class="avatar-fallback" style="display:none;">' + initials + '</div>'
            '</div>'
            '<div class="name">' + str(row.get('web_name', '')) + '</div>' +
            meta_html + ticker_html +
            '<div class="stats-row"><div><div class="xp">' + f"{xp:.1f}" + '</div>'
            '<span class="xp-l">' + xp_label + '</span></div>' + price_html + '</div>'
            '</div>')


# ---------------------------------------------------------------------------
# Cached data / compute layers — keyed so a sidebar widget change (style,
# hit-stance) never re-hits the network; only a new team ID or a fresh
# "Run Model" click does.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _load_data(entry_id: int, season: str, prev_season: str, recency_window: int = 4):
    snap = fpl_data.load_snapshot(season, recency_window=recency_window)
    hist = fpl_data.load_historical_snapshot(prev_season)
    entry = fpl_data.fetch_entry_official(entry_id)
    history = fpl_data.fetch_entry_history_official(entry_id)
    return snap, hist, entry, history


@st.cache_data(ttl=900, show_spinner=False)
def _load_transfers(entry_id: int):
    """Patch 117c: the manager's transfer list (None if the feed cannot be read)."""
    return fpl_data.fetch_entry_transfers_official(entry_id)


@st.cache_data(ttl=900, show_spinner=False)
def _lock_cost_calc(pool_nolock, locked_sq, cfg, team_value: float, gws: tuple):
    """Patch 117d: xPts the Locked players cost = unlocked Wildcard rebuild value minus the locked squad's, over the window."""
    free = recommend.wc_rebuild_with_fallback(pool_nolock, cfg, float(team_value), list(gws))
    if free.get("squad") is None:
        return None
    return recommend.lock_cost(free["squad"], locked_sq, list(gws), cfg)


def _olbauday_season_slug(season: str) -> str:
    """Converts vaastav-style "2026-27" (cfg["meta"]["season"]) to
    olbauday's own "2026-2027" folder-naming convention -- confirmed live
    (data/2026-2027/By Gameweek/GW1/matches.csv fetched successfully;
    data/2026-27/... 404s) before writing fpl_data.fetch_team_match_xg()."""
    a, b = season.split("-")
    return f"{a}-{a[:2]}{b}"


@st.cache_data(ttl=1800, show_spinner=False)
def _fixture_baselines(season: str, current_gw: int, teams_df, cfg):
    """Patch 83 (v6.9 Rule #46) -- one fetch of olbauday's real match-level
    team xG per finished GW this season, cached separately from `_project`
    (own network cost, own cache lifetime -- match results only change once
    a gameweek finishes, so a longer 30-minute TTL than the 15-minute live-
    data cache is fine and cuts needless re-fetching on every rerun).
    Returns (baselines_dict, warning_str_or_None) -- see
    fpl_data.fetch_team_match_xg() and data_pipeline.
    compute_team_fixture_baselines() for what each half means. Never raises:
    an empty/failed fetch returns an empty baselines dict, and
    fixture_attack_factor_vec() treats that as "no adjustment this run"
    (FF=1.0 everywhere), not a crash."""
    finished_gws = list(range(1, max(1, current_gw) + 1))
    slug = _olbauday_season_slug(season)
    match_xg, warn = fpl_data.fetch_team_match_xg(slug, finished_gws)
    baselines = data_pipeline.compute_team_fixture_baselines(cfg, match_xg, teams_df)
    return baselines, warn


@st.cache_data(ttl=900, show_spinner=False)
def _project(_snap, hist_df, overrides_df, cfg, gw_list, _fixture_baselines_dict=None):
    players = data_pipeline.build_player_table(cfg, _snap, hist_df, overrides_df)
    proj = data_pipeline.compute_all(cfg, _snap, players, gw_list, fixture_baselines=_fixture_baselines_dict)
    return proj


@st.cache_data(ttl=900, show_spinner=False)
def _picks(entry_id: int, gw: int):
    return fpl_data.fetch_entry_picks_official(entry_id, gw)


@st.cache_data(ttl=900, show_spinner=False)
def _chip_portfolio_calc(squad_df_adv, pool_df_adv, cfg, free_transfers: int, bank: float,
                          portfolio_gw_list: tuple, window_len: int, available_chip_types: tuple,
                          fh_gap_table: dict, valid_gws_by_type_items: tuple = (),
                          fh_by_gw: dict | None = None, reachable_ceiling_by_gw: dict | None = None,
                          scan_full_gw_list: tuple | None = None):
    """Patch 86 (performance fix) -- Patch 85's Rule #48/#49 computation
    (wildcard_window_value_scan: 2 opt.solve_squad() MILP solves per
    candidate GW, times up to ~8 candidate GWs by default; plus
    chip_portfolio_schedule's brute-force itertools.product search) was
    confirmed via code read (2026-09-29, grep for `@st.cache_data` /
    `@st.fragment` against the Patch 85 insertion point) to be bare
    top-level script code with NO caching, unlike every other comparably
    expensive computation in this file (_fixture_baselines, _project,
    _picks above). That meant it re-ran in full on every single Streamlit
    script rerun -- including reruns triggered by completely unrelated
    widgets (Style toggle, materiality-bar slider, etc.) that never change
    the squad, transfers, bank, or chip availability this computation
    actually depends on. Wrapping it in the same @st.cache_data(ttl=900,
    show_spinner=False) pattern already used above makes a rerun with
    unchanged inputs a cache hit (near-instant) instead of a full re-solve,
    with zero change to the computed values themselves -- confirmed by the
    existing Patch 85 unit tests (test_patch85_chip_window_and_portfolio.py)
    still passing unmodified, since this function is a pure pass-through to
    the same chip_protocol calls with the same arguments."""
    available_chip_types = set(available_chip_types)
    wc_window_scan = (chip_protocol.wildcard_window_value_scan(
        squad_df_adv, pool_df_adv, cfg, free_transfers, bank,
        list(portfolio_gw_list), list(scan_full_gw_list) if scan_full_gw_list else list(portfolio_gw_list),
        window_len=window_len,
        objective=str(cfg.get("chip_extended_check", {}).get("wc_scan_objective", "weekly_xi")))   # Patch 115 ruling 7
        if "wildcard" in available_chip_types else {"by_gw": {}, "best_gw": None, "best_gap": None})
    # Patch 89 (Rule #49d) -- pool_df_adv/free_transfers passed through so
    # chip_portfolio_schedule() can build the reachable pre-Wildcard Bench
    # Boost/Triple Captain table (_reachable_bb_tc_tables()) instead of the
    # flat, unchanged-squad one it used before this patch.
    # Patch 91 (Rule #49, chip-expiry correctness) -- valid_gws_by_type_items
    # is a hashable (cache_data requires it) tuple-of-pairs form of the real
    # {"wildcard": [...], "bboost": [...], ...} window-clipped GW lists the
    # call site builds via the SAME _clip_to_available_windows() the
    # isolated advisor cards already use; reconstructed to a dict-of-sets
    # here since chip_portfolio_schedule() itself doesn't need to be
    # cache-friendly.
    valid_gws_by_type = {k: set(v) for k, v in valid_gws_by_type_items} if valid_gws_by_type_items else None
    # Patch 96 (v6.9 Rule #52, Chip Timing Harmony) -- fh_by_gw (Gap A) lets
    # Free Hit value itself differently before/after a scheduled Wildcard,
    # same as Bench Boost/Triple Captain already do; reachable_ceiling_by_gw
    # (Gap B) lets the near-tie selection reject a combination that leaves a
    # meaningfully weaker squad once the chips are done. Both default to
    # None (no-op, byte-for-byte pre-Patch-96 behavior) when not supplied.
    chip_portfolio = chip_protocol.chip_portfolio_schedule(
        available_chip_types, wc_window_scan, fh_gap_table, squad_df_adv,
        list(portfolio_gw_list), cfg, lambda total: eng.margin_of_error_threshold(total, cfg),
        pool_df=pool_df_adv, free_transfers=free_transfers, valid_gws_by_type=valid_gws_by_type,
        fh_by_gw=fh_by_gw, reachable_ceiling_by_gw=reachable_ceiling_by_gw)
    return wc_window_scan, chip_portfolio


@st.cache_data(ttl=900, show_spinner=False)
def _chip_portfolio_fixed_wc_calc(wc_scan_one: dict, fh_gap_table: dict, squad_df_adv, portfolio_gw_list: tuple,
                                   cfg: dict, available_chip_types: tuple, valid_gws_by_type_items: tuple,
                                   fh_by_gw: dict | None, reachable_ceiling_by_gw: dict | None, wc_gw: int):
    """Patch 112: Rule #49 joint sequence again, with the Wildcard pinned to the chain decision's week so
    Triple Captain / Bench Boost / Free Hit are valued on that Wildcard's rebuild squad. Reuses the scan
    already computed (no new MILP solves except the pre-Wildcard reachable tables)."""
    valid = {k: set(v) for k, v in valid_gws_by_type_items} if valid_gws_by_type_items else None
    return chip_protocol.chip_portfolio_schedule(
        set(available_chip_types), wc_scan_one, fh_gap_table, squad_df_adv, list(portfolio_gw_list), cfg,
        lambda total: eng.margin_of_error_threshold(total, cfg), valid_gws_by_type=valid,
        fh_by_gw=fh_by_gw, reachable_ceiling_by_gw=reachable_ceiling_by_gw, force_wildcard_gw=wc_gw)


@st.cache_data(ttl=900, show_spinner=False)
def _reachable_ceiling_calc(cfg: dict, shape_proj: pd.DataFrame, squad_codes_items: tuple,
                             free_transfers: int, detect_gw_list: tuple):
    """Patch 99 -- same uncached-hot-path finding as _suggest_transfers_calc()
    above, applied to the Wildcard trigger's own reachable-ceiling solve
    (data_pipeline.solve_reachable_ceiling_by_gw(): one MILP solve PER GW in
    detect_gw_list, confirmed via code read of its own docstring). Was
    called bare at the app.py top level (pre-Patch-99), re-solving on every
    rerun regardless of whether the squad/transfers/detect window actually
    changed."""
    return data_pipeline.solve_reachable_ceiling_by_gw(
        cfg, shape_proj, list(squad_codes_items), free_transfers, list(detect_gw_list))


@st.cache_data(ttl=900, show_spinner=False)
def _bb_advisor_calc(bench_df_adv: pd.DataFrame, bb_gws: tuple, cfg: dict, chip_key: str):
    """Patch 99 -- same fix as _reachable_ceiling_calc() above, for the
    isolated Bench Boost advisor card. Reconstructs moe_fn INSIDE the cached
    function (same discipline _chip_portfolio_calc already uses for its own
    lambda) rather than accepting a closure as a param, since a lambda isn't
    reliably hashable for @st.cache_data's cache key."""
    cat_cfg = cfg.get("chip_advisor_thresholds", {})
    t = cat_cfg.get(chip_key, {})
    moe_fn = lambda total: eng.margin_of_error_threshold(
        total, cfg, floor_points=t.get("floor_points"), pct_of_total=t.get("pct_of_total"))
    return chip_protocol.evaluate_bench_boost(bench_df_adv, list(bb_gws), moe_fn)


@st.cache_data(ttl=900, show_spinner=False)
def _tc_advisor_calc(starters_df_adv: pd.DataFrame, tc_gws: tuple, cfg: dict, chip_key: str):
    """Patch 99 -- same fix, for the isolated Triple Captain advisor card."""
    cat_cfg = cfg.get("chip_advisor_thresholds", {})
    t = cat_cfg.get(chip_key, {})
    moe_fn = lambda total: eng.margin_of_error_threshold(
        total, cfg, floor_points=t.get("floor_points"), pct_of_total=t.get("pct_of_total"))
    return chip_protocol.evaluate_triple_captain(starters_df_adv, list(tc_gws), moe_fn)


@st.cache_data(ttl=900, show_spinner=False)
def _fh_advisor_calc(squad_df_adv: pd.DataFrame, fh_gws: tuple, cfg: dict, chip_adv_proj: pd.DataFrame,
                      team_value: float, chip_key: str):
    """Patch 99 -- same fix, for the isolated Free Hit advisor card. This is
    the single most expensive of the three isolated advisors -- confirmed by
    this file's own pre-existing comment ("each Free Hit check is a fresh
    MILP solve per horizon GW") -- and was, like the others, completely
    uncached before this patch. rebuild_fn is reconstructed inside this
    cached function from cfg/chip_adv_proj/team_value (all hashable/already-
    proven cache_data-safe), same reason moe_fn is reconstructed rather than
    passed in."""
    cat_cfg = cfg.get("chip_advisor_thresholds", {})
    t = cat_cfg.get(chip_key, {})
    moe_fn = lambda total: eng.margin_of_error_threshold(
        total, cfg, floor_points=t.get("floor_points"), pct_of_total=t.get("pct_of_total"))
    rebuild_fn = lambda gw: data_pipeline.solve_free_hit_rebuild(cfg, chip_adv_proj, team_value, gw)
    return chip_protocol.evaluate_free_hit(squad_df_adv, list(fh_gws), rebuild_fn, moe_fn)


@st.cache_data(ttl=900, show_spinner=False)
def _suggest_transfers_calc(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                             profile_name: str, hit_stance: str, free_transfers: int,
                             bank: float, current_gw: int, gw_list: tuple,
                             forced_count: int | None, meaningful_bar: float | None,
                             bench_codes_items: tuple, chip_advisory: str | None,
                             bb_play_gw: int | None, chip_capped_gw_list: tuple | None,
                             disrupted_codes_items: tuple | None, chip_schedule: dict | None):
    """Patch 99 (2026-09-30, manager-reported ~6m10s runtime PERSISTING after
    Patch 97's caching fix -- confirmed the fix targeted the wrong bottleneck)
    -- confirmed via code read that recommend.suggest_transfers() (and, via
    its chained-planner dispatch, plan_transfer_schedule() -- the k=1..5
    (under "Hit if worth it") chained multi-GW MILP search, with the
    position tie-break and Starting-XI Impact Check's own extra solves on
    top) was called bare at the app.py top level (pre-Patch-99 line ~2335),
    with NO @st.cache_data wrapper at all -- unlike every other comparably
    expensive computation in this file (_chip_portfolio_calc since Patch 86,
    _extended_wc_cross_check_calc since Patch 95, _project, _picks). That
    means THE app's single most expensive user-facing computation re-ran in
    full on EVERY Streamlit script rerun -- including reruns triggered by a
    completely unrelated widget (Style toggle, materiality-bar slider,
    hit-stance radio, etc.) that never changes the squad, transfers, bank,
    or chip schedule this computation actually depends on. Patch 97's
    DataFrame-in-cache-key fix was real but targeted a much smaller, already-
    cached computation (chip_portfolio) -- it was never going to touch this,
    which is exactly why the manager's re-timed run showed no real
    improvement (6m10s vs. the original 6m45s). Wrapping this call in the
    same @st.cache_data(ttl=900, show_spinner=False) pattern makes a rerun
    with unchanged inputs a cache hit (near-instant) instead of a full
    re-solve, with zero change to the computed values themselves -- the full
    133-test regression suite (unaffected by this app.py-only wiring change)
    still passes unmodified, since this function is a pure pass-through to
    the same recommend.suggest_transfers() call with the same arguments.

    Tuple-ified args (`gw_list`, `bench_codes_items`, `chip_capped_gw_list`,
    `disrupted_codes_items`) are hashable forms of the real list/set
    `suggest_transfers()` expects, reconstructed just inside this wrapper --
    same discipline `_extended_wc_cross_check_calc()` already uses for its
    own set/list params."""
    bench_codes = set(bench_codes_items) if bench_codes_items else set()
    chip_capped = list(chip_capped_gw_list) if chip_capped_gw_list is not None else None
    disrupted_codes = set(disrupted_codes_items) if disrupted_codes_items else None
    return recommend.suggest_transfers(
        squad_df, pool_df, cfg, profile_name, hit_stance, free_transfers, bank, current_gw,
        list(gw_list), forced_count, meaningful_bar, bench_codes, chip_advisory,
        bb_play_gw=bb_play_gw, chip_capped_gw_list=chip_capped,
        disrupted_codes=disrupted_codes, chip_schedule=chip_schedule)


@st.cache_data(ttl=900, show_spinner=False)
def _reachable_ceiling_single_calc(cfg: dict, proj: pd.DataFrame, squad_codes_items: tuple, free_transfers: int):
    """Patch 100 (2026-09-30, manager: "dig deep on the performance") --
    further audit beyond Patch 99's five found data_pipeline.
    solve_reachable_ceiling() (the SINGLE-total counterpart to the already-
    cached by-GW version -- this one feeds the Team Rating % headline, not
    the Wildcard trigger) called bare at the app.py top level, unconditional
    on every single script rerun, with no caching at all. Pure pass-through
    wrapper, same pattern as _reachable_ceiling_calc (Patch 99)."""
    return data_pipeline.solve_reachable_ceiling(cfg, proj, list(squad_codes_items), free_transfers)


@st.cache_data(ttl=900, show_spinner=False)
def _theoretical_ceiling_calc(cfg: dict, proj: pd.DataFrame):
    """Patch 100 -- data_pipeline.solve_ceiling() (the unconstrained full-
    pool "theoretical ceiling" solve) was called bare at the app.py top
    level, unconditional on every single script rerun. Pure pass-through."""
    return data_pipeline.solve_ceiling(cfg, proj)


@st.cache_data(ttl=900, show_spinner=False)
def _best_gw_squad_calc(cfg: dict, proj: pd.DataFrame, team_value: float, gw: int):
    """Patch 117g: shared 'best possible squad for this GW' (better of XI-first and 15-man-sum) for every rating."""
    return data_pipeline.solve_best_gw_squad(cfg, proj, team_value, gw)


@st.cache_data(ttl=900, show_spinner=False)
def _fh_optimal_calc(cfg: dict, proj: pd.DataFrame, team_value: float, gw: int):
    """Patch 100 -- data_pipeline.solve_free_hit_optimal_squad() (the "what's
    the actual best Free Hit squad this GW" display feature) had TWO bare,
    uncached call sites: (1) the header's auto Team Rating card, confirmed
    via this file's own pre-existing Patch 22 comment as "computed
    automatically every run ... not gated behind the manual picker any
    more"; (2) the "Team Recommendation -- active signals, auto-built"
    block's Free Hit expander, which fires whenever the Free Hit advisor's
    verdict is play_gwN (not button-gated, per that section's own explicit
    comment). Both now route through this single shared cached wrapper --
    same function signature, same pass-through, one fix covers both sites."""
    return data_pipeline.solve_free_hit_optimal_squad(cfg, proj, team_value, gw)


@st.cache_data(ttl=900, show_spinner=False)
def _shape_test_calc(squad_df: pd.DataFrame, shape_proj: pd.DataFrame, cfg: dict,
                      detect_gw_list_items: tuple, team_value: float):
    """Patch 100 -- chip_protocol.wildcard_freehit_shape_test() runs FOUR
    separate opt.solve_squad() calls internally (one per GW in the 4-GW
    chip_shape_test.detection_window_gws detection window), gated only by
    `wc_flag or _fh_available_now` -- true on essentially every run once
    either chip's signal is active, which is the manager's own team's
    observed state in both screenshots this session. Was called bare, no
    caching. Pure pass-through."""
    return chip_protocol.wildcard_freehit_shape_test(squad_df, shape_proj, cfg, list(detect_gw_list_items), team_value)


@st.cache_data(ttl=900, show_spinner=False)
def _wc_whatif_calc(squad_df: pd.DataFrame, pool_df: pd.DataFrame, cfg: dict,
                     team_value: float, future_gw_list_items: tuple):
    """Patch 100 -- chip_protocol.evaluate_wildcard_whatif() (one full
    horizon-sum MILP rebuild solve) had a bare, uncached call site inside
    the "Team Recommendation -- active signals, auto-built" block, which
    that section's own comment confirms is NOT button-gated -- it fires
    whenever the Wildcard trigger is active. Reused here (same signature)
    for BOTH that auto call site and the manual "Evaluate your own
    scenario" picker's call site -- the manual one was already only
    user-triggered so this adds no new gating requirement there, just
    caching if the manager re-checks the same candidate date twice."""
    return chip_protocol.evaluate_wildcard_whatif(squad_df, pool_df, cfg, team_value, list(future_gw_list_items))


@st.cache_data(ttl=900, show_spinner=False)
def _extended_wc_cross_check_calc(squad_df: pd.DataFrame, pool_df: pd.DataFrame, shape_proj: pd.DataFrame,
                                   cfg: dict, style_name: str, free_transfers: int, bank: float,
                                   planning_gw: int, ext_gw_list: tuple, meaningful_bar: float,
                                   bb_play_gw: int | None, disrupted_codes_items: tuple, hit_stance: str,
                                   squad_codes_items: tuple, wildcard_gw: int | None,
                                   wildcard_rebuild_squad: pd.DataFrame | None):
    """Patch 95 (v6.9 Rule #49 cross-check extension — manager discussion,
    2026-09-30): when `recommend.resolve_cross_check_horizon()` (see its own
    docstring) determines the Wildcard cross-check needs to reach further
    than the app's normal windows, this runs the two extra, cross-check-only
    computations that make that possible — extending BOTH the transfer plan
    AND the reachable-ceiling ladder to the same GW range, confirmed via
    code read (during this same discussion) to be two SEPARATE, differently-
    sized windows today, so extending only one would still leave the other
    as the limiting factor.

    Wrapped in the same @st.cache_data(ttl=900) pattern as
    `_chip_portfolio_calc` above, for the same reason: this is genuinely
    expensive (one MILP solve per extra GW on each side, up to +8 GWs per
    Standing manager-confirmed cap), so a rerun triggered by an unrelated
    widget (Style toggle, materiality slider) must be a cache hit, not a
    re-solve. Never touches `rec` or `reachable_by_gw` themselves — this is
    a side computation purely for the cross-check note, so the Transfer
    Recommendations tab and the Wildcard trigger's own headline % are both
    completely unaffected by whatever this returns.

    Returns {"weekly_plan": [...], "reachable_by_gw": {gw: solve_squad()
    dict}} — the pieces `recommend.build_squad_after_by_gw()` and the
    cross-check's own rating-% loop need, nothing else."""
    disrupted_codes = set(disrupted_codes_items) if disrupted_codes_items else None
    squad_codes = list(squad_codes_items)
    ext_gw_list = list(ext_gw_list)
    chip_schedule = None
    if wildcard_gw is not None and wildcard_rebuild_squad is not None and not wildcard_rebuild_squad.empty:
        chip_schedule = {"wildcard_gw": wildcard_gw, "wildcard_rebuild_squad": wildcard_rebuild_squad}
    # "Force" has no meaningful multi-week pacing semantics to extend (it's
    # a single, manager-forced transfer count for THIS week only) — falls
    # back to "Hit if worth it" for this side computation alone, same
    # fallback already applied to the extended plan's hit-stance elsewhere
    # in this app's chip-aware wiring.
    effective_stance = "Hit if worth it" if hit_stance == "Force" else hit_stance
    ext_plan = recommend.plan_transfer_schedule(
        squad_df, pool_df, cfg, style_name, free_transfers, bank, planning_gw, ext_gw_list,
        meaningful_bar, None, bb_play_gw=bb_play_gw, disrupted_codes=disrupted_codes,
        hit_stance=effective_stance, chip_schedule=chip_schedule)
    ext_reachable_by_gw = data_pipeline.solve_reachable_ceiling_by_gw(
        cfg, shape_proj, squad_codes, free_transfers, ext_gw_list)
    return {"weekly_plan": ext_plan.get("weekly_plan") or [], "reachable_by_gw": ext_reachable_by_gw}


@st.cache_data(ttl=900, show_spinner=False)
def _wc_chain_compare_calc(squad_df: pd.DataFrame, pool_df: pd.DataFrame, shape_proj: pd.DataFrame, cfg: dict,
                           style_name: str, free_transfers: int, bank: float, planning_gw: int, eval_gws: tuple,
                           cand_gws: tuple, rebuild_len: int, meaningful_bar: float, bb_play_gw: int | None,
                           disrupted_codes_items: tuple, team_value: float, first_moves: tuple,
                           freehit_gw: int | None = None, cap_use_bar: float | None = None,
                           tail_gws: tuple = (), chip_types: tuple = (), include_chip_value: bool = False,
                           chip_gws: tuple = (), pick_rule: str = "best_horizon", allowed_items: tuple = (),
                           wc_objective: str = "weekly_xi", variants: bool = False, tc_ft_route: bool = True,
                           post_weeks_cap: int = 0, post_decay: float = 0.9, tc_week_source: str = "chain",
                           tc_sched_gw: int | None = None):
    """Patch 115 (model v6.12 rulings 3, 5, 6): every Wildcard candidate is valued over the SAME number of post-chip weeks
    (`post_weeks_cap` > 0: min(cap, fewest weeks left among the scorable candidates), never below 4), decay-weighted
    (`post_decay`); a candidate without a full window is NOT scored (listed in out['not_scored'], never planned or ranked).
    out['cands'][t] carries window_total (band input), checkpoint (the squad's xPts in the span's last week, for the Rule #52
    guardrail) and out['health_band']/out['checkpoint_gw']/out['base']['checkpoint']. `tc_week_source="schedule"` takes the
    Triple Captain premium's week from `tc_sched_gw` (the Rule #49 harmonized assignment; None = no premium), "chain" = the
    chain's own best TC week (Patch 114).

    Patch 111 (2026-10-04): the Wildcard CHAIN COMPARISON. Patch 112 adds: the scheduled Free Hit week is
    left out of both the plan's valuations and the gain sum (it reverts, so the persisted squad does not
    matter that week); `cap_use_bar` makes the chain spend a transfer instead of wasting it at the 5-FT cap;
    `tail_gws` are extra GWs after the span used only to VALUE late-week moves (never planned or shown). For the NO-CHIP path and for a Wildcard
    played at each candidate GW, runs a chained free-transfers-only plan (no hits) over `eval_gws` and
    scores the squad fielded each week. The Wildcard's value at GW t is the SUM over the whole span of
    (score with the chip at t) - (score with no chip) -- not a 4-GW window. The Wildcard team is rebuilt
    for the next `rebuild_len` GWs (8) from t. The chains start from the manager's own recommended
    current-GW free move (`first_moves`, (out_code, in_code) pairs), so week 1 matches the Transfer page.
    Chain plans skip the tie-break scan (use_tie_break=False) -- ~93% of planner time, and it only picks
    between near-tied in-players. Free Hit / Bench Boost / Triple Captain are not modelled in the chain
    (they do not change the persisted squad, or are scheduled separately). Returns plain dicts or None."""
    gws = list(eval_gws)
    cands = list(cand_gws)
    # Patch 112: the chip window can reach past the planned span (Extended Check: chips to Current+11, plan to +9);
    # those extra GWs keep the last planned squad (no further transfers) and are used for chip picks only.
    allg = gws + [g for g in chip_gws if g > (gws[-1] if gws else 0)
                  and f"xpts_gw{g}" in shape_proj.columns and f"xpts_gw{g}" in squad_df.columns]
    cols = [f"xpts_gw{g}" for g in gws]
    if not gws or any(c not in shape_proj.columns for c in cols) or any(c not in squad_df.columns for c in cols):
        return {"cands": {}, "unavailable_reason": "the projection columns for the checked weeks are missing from the squad or pool"}
    disrupted_codes = set(disrupted_codes_items) if disrupted_codes_items else None

    def _apply(sq, pairs):
        outs = {o for o, _ in pairs}
        ins = {i for _, i in pairs}
        out = pd.concat([sq[~sq["code"].isin(outs)], shape_proj[shape_proj["code"].isin(ins)]],
                        ignore_index=True, sort=False)
        return out.drop_duplicates(subset=["code"], keep="first")

    squad1 = _apply(squad_df, first_moves) if first_moves else squad_df
    bank1 = float(bank) + float(squad_df["price"].sum(skipna=True)) - float(squad1["price"].sum(skipna=True))
    ft_next = min(5, max(0, free_transfers - len(first_moves)) + 1)
    rest = [g for g in gws if g != planning_gw]
    pool_all = shape_proj

    # Patch 112: the % yardstick is the SAME one the Pitch tab uses -- the best possible squad for THAT GW alone
    # (Free-Hit-optimal, total team value) -- so the two views can be compared and a squad that drifts away from the
    # best team shows up. (Before: one fixed squad for the whole span, which a static squad trivially matched ~100%.)
    from concurrent.futures import ThreadPoolExecutor

    def _ref_squad(g):
        # best of two optima under the SAME scoring (rating_gw_value weights the bench): the XI-first squad the
        # Pitch uses, and the 15-man-sum squad -- one can beat the other, and a % above 100 would be nonsense.
        res = data_pipeline.solve_best_gw_squad(cfg, shape_proj, float(team_value), g)
        return opt.rating_gw_value(res["squad"], f"xpts_gw{g}", cfg)["total_realized"] if res else None
    with ThreadPoolExecutor(max_workers=recommend.parallel_workers(len(gws))) as _ex:
        _refs = list(_ex.map(_ref_squad, gws))
    if any(r is None for r in _refs):
        return {"cands": {}, "unavailable_reason": "the best-possible-squad reference solve returned no squad for at least one "
                                                   "week (solver time cap reached, or infeasible)"}
    dream_score = dict(zip(gws, _refs))

    def _squads(start_sq, plan_wp, first_sq=None):
        after = recommend.build_squad_after_by_gw(start_sq, plan_wp or [], pool_all)
        out = {}
        for g in allg:
            if g == planning_gw and first_sq is not None:
                out[g] = first_sq
            else:
                app_ = [k for k in after if k is not None and k <= g]
                out[g] = after[max(app_)] if app_ else start_sq
        return out

    def _codes_by_gw(sqs):
        return {g: [int(c) for c in sqs[g]["code"]] for g in allg}

    def _scores(start_sq, plan_wp, first_sq=None, sqs=None):
        sqs = sqs if sqs is not None else _squads(start_sq, plan_wp, first_sq)
        return {g: opt.rating_gw_value(sqs[g], f"xpts_gw{g}", cfg)["total_realized"] for g in gws}

    def _pct(score):
        return {g: min(100.0, eng.team_rating_pct(score[g], dream_score[g], "")["rating_pct"]) for g in gws}

    def _pct_raw(score):
        return {g: eng.team_rating_pct(score[g], dream_score[g], "")["rating_pct"] for g in gws}

    fh_gw = freehit_gw if (freehit_gw is not None and freehit_gw in gws) else None
    tail = [g for g in tail_gws if f"xpts_gw{g}" in shape_proj.columns and f"xpts_gw{g}" in squad_df.columns]

    def _plan(start_sq, start_bank, ft, plan_gws, chip=None, tc_target=None):
        if not plan_gws:
            return {"weekly_plan": []}
        chip = dict(chip or {})
        if fh_gw is not None and fh_gw in plan_gws:
            chip["freehit_gw"] = fh_gw
        return recommend.plan_transfer_schedule(
            start_sq, pool_df, cfg, style_name, ft, start_bank, plan_gws[0], plan_gws, meaningful_bar, None,
            bb_play_gw=bb_play_gw, disrupted_codes=disrupted_codes, hit_stance="No hits", chip_schedule=chip or None,
            use_tie_break=False, cap_use_bar=cap_use_bar, value_tail=tail, tc_target=tc_target)

    # Patch 112: chip value (Triple Captain / Bench Boost / Free Hit) on each chain's own squads -- manager
    # instruction, a DEVIATION from Standing Rule #31 (flag include_chip_value; flagged for the model chat).
    use_chips = bool(chip_types)   # Patch 117i: chips are always valued (shown beside the gain); include_chip_value only decides whether they count INSIDE it
    fh_ref = None
    if use_chips and "freehit" in chip_types and freehit_gw is not None and freehit_gw in gws:
        _fp = shape_proj.copy()
        _fo = opt.solve_squad(_fp, cfg, budget=float(team_value), objective_col=f"xpts_gw{freehit_gw}", label="chain_fh_ref")
        if _fo is not None and _fo.get("squad") is not None:
            _fs = _fo["squad"]
            if not isinstance(_fs, pd.DataFrame):
                _fs = _fp[_fp["code"].isin([p["code"] for p in _fs])]
            fh_ref = opt.rating_gw_value(_fs, f"xpts_gw{freehit_gw}", cfg)["total_realized"]

    def _chips(sqs, wc_gw):
        if not use_chips:
            return None
        return recommend.chain_chip_value(sqs, allg, cfg, wc_gw, fh_gw, fh_ref, tuple(chip_types),
                                          moe_fn=lambda total: eng.margin_of_error_threshold(total, cfg),
                                          pick_rule=pick_rule,
                                          allowed=({k: tuple(v) for k, v in allowed_items} if allowed_items else None))

    base_plan = _plan(squad1, bank1, ft_next, rest)
    base_sqs = _squads(squad1, base_plan.get("weekly_plan"), first_sq=squad1)
    base_score = _scores(None, None, sqs=base_sqs)
    base_chips = _chips(base_sqs, None)
    g0 = gws[0]
    cur_sc = opt.rating_gw_value(squad_df, f"xpts_gw{g0}", cfg)["total_realized"]
    out = {"gws": gws, "dream": dream_score, "first_moves_n": len(first_moves), "first_ft_after": ft_next,
           "diag": {"gw": g0, "current": round(float(cur_sc), 1), "after_move": round(float(base_score[g0]), 1),
                    "reference": round(float(dream_score[g0]), 1), "n_after": int(len(squad1)),
                    "n_current": int(len(squad_df))},
           "base": {"score": base_score, "pct": _pct(base_score), "pct_raw": _pct_raw(base_score), "plan": base_plan.get("weekly_plan") or [],
                    "chips": base_chips, "squads": _codes_by_gw(base_sqs)},
           "cands": {}}
    # Patch 114: the Wildcard squad is rebuilt with a legal starting XI chosen for EACH week (XI in full, bench at the
    # planner's low bench weight) -- the Patch 113 rebuild summed all 15 players as if all were starters (switch wc_objective).
    _mode = wc_objective if wc_objective in ("weekly_xi", "sum15") else "weekly_xi"
    _margin = float(eng.margin_of_error_threshold(0.0, cfg))

    # Patch 115 (ruling 3a): ONE post-chip window length for every candidate; candidates without a full window are not scored
    if int(post_weeks_cap) > 0:
        _post_n, _scored_c, _not_scored = recommend.common_post_weeks(cands, gws, cap=int(post_weeks_cap), minimum=4)
    else:
        _post_n, _scored_c, _not_scored = 0, list(cands), []
    out["post_weeks"] = _post_n
    out["fh_gw"] = fh_gw
    out["not_scored"] = list(_not_scored)
    out["checkpoint_gw"] = gws[-1]
    out["base"]["checkpoint"] = round(float(base_score[gws[-1]]), 2)
    out["health_band"] = round(float(eng.margin_of_error_threshold(float(dream_score[gws[-1]]), cfg)), 2)

    def _entry(t, rsq, tc_tgt=None, tier=None, note=None):
        chip = {"wildcard_gw": t, "wildcard_rebuild_squad": rsq}
        if t == planning_gw:
            plan = _plan(squad_df, bank, free_transfers, gws, chip, tc_tgt)
            sqs = _squads(squad_df, plan.get("weekly_plan"))
        else:
            plan = _plan(squad1, bank1, ft_next, rest, chip, tc_tgt)
            sqs = _squads(squad1, plan.get("weekly_plan"), first_sq=squad1)
        sc = _scores(None, None, sqs=sqs)
        if _post_n:
            _pg = recommend.post_chip_gain(sc, base_score, t, gws, _post_n, post_decay, skip_gw=fh_gw)
            g_chain = _pg["gain"] if _pg else 0.0
            _wtot = round(sum(sc[g] for g in (_pg["weeks"] if _pg else [])), 2)
        else:
            g_chain = round(sum(sc[g] - base_score[g] for g in gws if g != fh_gw), 2)
            _wtot = round(sum(sc[g] for g in gws), 2)
        c_chips = _chips(sqs, t)
        g_chips = round(c_chips["total"] - base_chips["total"], 2) if (c_chips and base_chips) else 0.0
        # Patch 117i (model M9/M29): chip value is shown beside the figure, never inside it, unless include_chip_value is on
        g_chips_dec = g_chips if include_chip_value else 0.0
        # Patch 117 (model ruling): the DECIDING measure = plain FOUR-gameweek chain gain + chip value (wc_decide_weeks 4);
        # 0 = the Patch 116 measure (the six-week decay-weighted gain).
        _dw_n = int(cfg.get("chip_extended_check", {}).get("wc_decide_weeks", 4) or 0)
        _g4 = recommend.post_chip_gain(sc, base_score, t, gws, _dw_n, 1.0, skip_gw=fh_gw) if _dw_n else None
        gain_decide = round(_g4["gain"] + g_chips_dec, 2) if _g4 else round(g_chain + g_chips_dec, 2)
        return {"gain": round(g_chain + g_chips_dec, 2), "gain_decide": gain_decide, "gain_chain": g_chain, "gain_chips": g_chips,
                "gain_chips_dec": g_chips_dec, "chips_inside": bool(include_chip_value), "gain_with_chips": round(g_chain + g_chips, 2),
                "chips": c_chips, "squads": _codes_by_gw(sqs), "score": sc, "pct": _pct(sc), "pct_raw": _pct_raw(sc),
                "plan": plan.get("weekly_plan") or [], "rebuild_codes": [int(c) for c in rsq["code"]],
                "variant": "plain", "tc_target": tc_tgt, "objective": _mode,
                "window_total": _wtot, "checkpoint": round(float(sc[gws[-1]]), 2), "post_weeks": _post_n,
                # Patch 115 fix (Rule #34 / #48(a)): the tie band is read off the FOUR-gameweek total
                "band_total": (recommend.four_week_total(sc, t, gws, int(cfg.get("chip_extended_check", {}).get("wc_band_weeks", 4)))
                               or _wtot),
                "build_tier": tier, "build_note": note}

    _rsq_by_t = {}

    def _rebuild(t):
        # Patch 115 fix: a failed weekly-XI solve is reported (tier + note), never silently dropped and never turned into
        # the plain 15-man sum (Rule #54)
        return recommend.wc_rebuild_with_fallback(shape_proj, cfg, float(team_value),
                                                  list(range(t, t + int(rebuild_len))), mode=_mode)
    # the candidate rebuilds are independent and CBC runs as a subprocess, so they solve side by side (same pattern as
    # the per-GW reference squads above); results are then evaluated in order
    _cands_run = [t for t in cands if t in _scored_c]
    with ThreadPoolExecutor(max_workers=recommend.parallel_workers(len(_cands_run) or 1)) as _ex2:
        _rebuilt = list(_ex2.map(_rebuild, _cands_run))
    _fail_notes = []
    for t, rb in zip(_cands_run, _rebuilt):
        rsq = rb.get("squad") if isinstance(rb, dict) else rb
        if rsq is None:
            _fail_notes.append(f"GW{t}: " + str((rb or {}).get("note") if isinstance(rb, dict) else "no squad"))
            continue
        _rsq_by_t[t] = rsq
        out["cands"][t] = _entry(t, rsq, None, rb.get("tier") if isinstance(rb, dict) else None,
                                 rb.get("note") if isinstance(rb, dict) else None)

    def _tc_target_for(chips, sqs_codes):
        """The best Triple Captain week's top pool scorer when he is NOT already fielded that week (else None)."""
        if tc_week_source == "schedule":
            # Patch 115 (ruling 5): premium only when TC is scheduled in the Rule #49 assignment, in THAT week
            if tc_sched_gw is None:
                return None
            w = int(tc_sched_gw)
            return recommend.tc_target_from_schedule(shape_proj, w, sqs_codes.get(w) or [], None)
        tcp = (chips or {}).get("tc")
        if not tcp:
            return None
        w = int(tcp[0])
        own = sqs_codes.get(w) or []
        cand = recommend.tc_target_candidates(shape_proj, w, own, k=1)
        owned_top = ((chips or {}).get("tc_player") or {}).get("xpts", 0.0) or 0.0
        if cand and cand[0]["xpts"] > float(owned_top) + 1e-9:
            return {"code": cand[0]["code"], "gw": w, "name": cand[0]["name"], "xpts": cand[0]["xpts"]}
        return None

    if variants and use_chips and include_chip_value and out["cands"] and "3xc" in chip_types:
        # chip-aware variants for the best Wildcard candidate only (cost: ~2 extra chain plans + 1 rebuild solve)
        # Patch 115 (ruling 2): the candidate carried into the variants is picked like the decision -- the LATER week inside
        # Rule #34's band of the best gain
        _g_top = max(c_["gain"] for c_ in out["cands"].values())
        _t_top = max(out["cands"], key=lambda k: (out["cands"][k]["gain"], k))
        _band_v = float(eng.margin_of_error_threshold(out["cands"][_t_top].get("window_total", 0.0), cfg))
        t_best = max(k for k, c_ in out["cands"].items() if c_["gain"] >= _g_top - _band_v - 1e-9)
        A = out["cands"][t_best]
        tgt = _tc_target_for(A["chips"], A["squads"])
        if tgt is not None:
            totals = {"plain": A["gain"], "chip-aware": None, "ft-route": None}
            B = C = None
            if _mode == "weekly_xi":
                bbg = ((A["chips"] or {}).get("bb") or (None,))[0]
                _rb_b = recommend.wc_rebuild_with_fallback(shape_proj, cfg, float(team_value),
                                                           list(range(t_best, t_best + int(rebuild_len))), mode=_mode,
                                                           bb_gw=bbg, tc_target=tgt)
                rsq_b = _rb_b.get("squad")
                if rsq_b is not None:
                    B = _entry(t_best, rsq_b, tgt if tc_ft_route else None, _rb_b.get("tier"), _rb_b.get("note"))
                    B["variant"] = "chip-aware"
                    totals["chip-aware"] = B["gain"]
            if tc_ft_route:
                C = _entry(t_best, _rsq_by_t[t_best], tgt)
                C["variant"] = "ft-route"
                totals["ft-route"] = C["gain"]
            best_v, best_gain = A, A["gain"]
            for V in (B, C):
                if V is not None and recommend.pick_wc_variant(A["gain"], V["gain"], _band_v)["variant"] == "chip-aware" \
                        and V["gain"] > best_gain:
                    best_v, best_gain = V, V["gain"]
            best_v["variant_totals"] = totals
            best_v["variant_band"] = round(_band_v, 2)          # Patch 115 (ruling 5): a flip inside this band stays chip-agnostic
            best_v["tc_target"] = tgt if best_v is not A else None
            best_v["tc_target_considered"] = tgt
            out["cands"][t_best] = best_v

    # base path (no Wildcard): the Triple Captain target can still be bought with a free transfer
    if variants and use_chips and tc_ft_route and base_chips and "3xc" in chip_types:
        tgt0 = _tc_target_for(base_chips, out["base"]["squads"])
        if tgt0 is not None:
            p_tc = _plan(squad1, bank1, ft_next, rest, None, tgt0)
            sq_tc = _squads(squad1, p_tc.get("weekly_plan"), first_sq=squad1)
            sc_tc = _scores(None, None, sqs=sq_tc)
            ch_tc = _chips(sq_tc, None)
            g_tc = round(sum(sc_tc[g] - base_score[g] for g in gws if g != fh_gw)
                         + ((ch_tc["total"] - base_chips["total"]) if ch_tc else 0.0), 2)
            if g_tc > float(eng.margin_of_error_threshold(sum(base_score.values()), cfg)):
                out["base_tc"] = {"score": sc_tc, "pct": _pct(sc_tc), "plan": p_tc.get("weekly_plan") or [],
                                  "chips": ch_tc, "squads": _codes_by_gw(sq_tc), "tc_target": tgt0, "gain": g_tc}
    if not out["cands"]:
        # Patch 115 fix: say WHY there is no Wildcard comparison instead of returning a bare None
        why = ("; ".join(_fail_notes)[:400] if _fail_notes else
               "no candidate week has a full post-chip window (Rule #48(a)), so none could be scored")
        return {"cands": {}, "unavailable_reason": why, "not_scored": list(_not_scored)}
    return out


# ---------------------------------------------------------------------------
# Gate screen — team ID first, everything else unlocks after.
# ---------------------------------------------------------------------------
if "unlocked" not in st.session_state:
    st.session_state.unlocked = False

if not st.session_state.unlocked:
    st.markdown('<div style="max-width:420px; margin:14vh auto 0; text-align:center;">', unsafe_allow_html=True)
    st.markdown(f'<div class="brand-row" style="justify-content:center;">{_CREST_SVG}'
                f'<div class="brand-mark">RB <span class="b2">Model</span></div></div>'
                f'<div class="brand-tag">v5.0 engine · live · zero-cost</div>'
                f'<p style="margin:22px 0 14px; color:var(--ink-muted);">Enter your FPL team ID to begin.</p>',
                unsafe_allow_html=True)
    entry_input = st.text_input("Team ID", placeholder="e.g. 26073", label_visibility="collapsed")
    if st.button("Unlock →", use_container_width=True):
        if not entry_input.strip().isdigit():
            _ne("Team ID: numbers only", "Team ID should be numbers only — find it in the URL when you open 'Points' on the official FPL site.")
        else:
            with st.spinner("Checking team ID against the official API..."):
                test_entry = fpl_data.fetch_entry_official(int(entry_input.strip()))
            if test_entry is None or "id" not in test_entry:
                _ne("Team ID not found", "Couldn't find that team ID on the official FPL API. Double-check it and try again.")
            else:
                st.session_state.unlocked = True
                st.session_state.team_id = int(entry_input.strip())
                st.session_state.team_name = test_entry.get("name", "")
                # Patch 58 (manager: "still needs to click on Run model") —
                # requirements-list item 4. Previously Unlock only validated
                # the team ID and set `unlocked`; the model itself never ran
                # until the separate sidebar "Run Model ->" click (line ~578,
                # gated on `has_run`). Setting `has_run` here too means the
                # very next rerun (right after Unlock) already renders full
                # results using the sidebar's default style/hit-stance/
                # horizon, with no extra click. "Run Model ->" stays in the
                # sidebar for re-running after changing a control.
                st.session_state.has_run = True
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — unlocked state
# ---------------------------------------------------------------------------
cfg = eng.load_config()
entry_id = st.session_state.team_id

with st.sidebar:
    st.markdown(f'<div class="brand-row">{_CREST_SVG}'
                f'<div class="brand-mark">RB <span class="b2">Model</span></div></div>'
                f'<div class="brand-tag" title="{html.escape(PATCH_VERSION)}">v5.0 engine · {PATCH_TAG.lower()} · '
                f'live · zero-cost</div><br>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="side-note">TEAM ID</div>'
                f'<div style="font-family:\'IBM Plex Mono\'; color:var(--accent-strong); '
                f'font-weight:600; margin-bottom:14px;">{entry_id} · {st.session_state.get("team_name","")}</div>',
                unsafe_allow_html=True)
    if st.button("Change team", use_container_width=True):
        st.session_state.unlocked = False
        st.rerun()

    style_name = st.selectbox("Style", list(style_profiles.PROFILES.keys()), index=0)
    # Patch 80 (manager, live screenshot, annotated "no borders" pointing at
    # this exact caption): a plain st.caption() renders as bare floating
    # grey text with no visual container, unlike almost every other piece of
    # information in this sidebar/app (chips, stats, cards all sit in a
    # bordered/tinted box) -- interpreted as "this reads as an orphaned
    # scrap of text, give it the same card treatment as everything else."
    # If this reading is wrong, easy to revert -- flagged in the delivery
    # message alongside this patch.
    st.markdown(f'<div class="side-info-box">{style_profiles.get_profile(style_name)["description"]}</div>',
                unsafe_allow_html=True)

    hit_stance = st.radio("Hit stance", ["No hits", "Hit if worth it", "Force"], index=1)
    _lock_slot = st.container()      # Patch 117d: filled once the squad is known (Locked players)
    _snap_slot = st.container()      # Patch 117d: run snapshot download
    # Patch 117c: the budget is read automatically (bank + selling prices from the transfer history). The box is
    # only a manual override, kept in Advanced; 0 = auto.
    with st.expander("Advanced", expanded=False):
        budget_override = st.number_input("Budget override £m (squad value + bank; 0 = auto)",
                                          min_value=0.0, max_value=150.0, value=0.0, step=0.1, format="%.1f")
    forced_count = None
    if hit_stance == "Force":
        forced_count = st.number_input("Transfers to force", min_value=1, max_value=5, value=2, step=1)

    # Patch 45 (2026-09-15, manager report: "what is the maximum GWs we can
    # get to solve this issue" after benchmarking showed 4-6 GW horizons
    # taking 1-3 minutes) — root cause is NOT the pitch navigator (which just
    # displays whatever squad the recommendation already produced); it's
    # `recommend.plan_transfer_schedule()`, the chained weekly planner that
    # "No hits"/"Hit if worth it" route through at horizon>1 (Patch 39's own
    # dispatch rule): one full k=0-5 MILP solve PER WEEK in the chain, plus
    # the Patch 41 tie-break's full-pool scan per week — both deliberately
    # kept at full strength per the manager's own confirmed choices in
    # Patch 42/41. Measured end-to-end on the real ~616-player pool, cold
    # cache: 1 GW ~2s (single solve, no chaining), 2 GW ~20s, 3 GW ~36s,
    # 4 GW ~72s, 5 GW ~121s, 6 GW ~161s — clearly super-linear, since each
    # week's own search cost scales with ITS remaining horizon length, and
    # week 1 of a longer plan always faces the longest remaining horizon.
    # Manager confirmed (2026-09-15) capping at 3 GWs — keeps every chained
    # run under ~40s — rather than narrowing the per-week search width
    # (a real accuracy trade-off) or leaving it uncapped with just a warning.
    # The cap only applies to the two hit-stances that actually route through
    # the chained planner; "Force" always does a single one-shot solve over
    # the whole horizon regardless of length; so it keeps the full 1-6 range.
    _horizon_max = 3 if hit_stance in ("No hits", "Hit if worth it") else 6
    horizon = st.slider("Horizon (gameweeks)", min_value=1, max_value=_horizon_max, value=1,
                         help="xPts are always shown per-GW too — widen this when you want a multi-week transfer plan view, not just this week's picture."
                         + ("" if _horizon_max == 6 else
                            " Capped at 3 GWs for this hit stance — beyond that, the chained weekly planner's "
                            "own full-strength search (k=0-5 every week, plus the near-tie full-pool scan) "
                            "takes 1-3+ minutes per run (measured, Patch 45); switch to \"Force\" for a longer "
                            "one-shot horizon instead."))

    # Patch 28 (v6.4 / Standing Rule #41) — this app has no persistent memory
    # between runs (fresh container each time, Step 2), so it cannot discover
    # a still-unplayed chip's PLANNED date on its own — the actual GW a
    # Wildcard/Free Hit gets played on stays a rolling re-test, never a fixed
    # commitment (Standing Rule #32), even though Patch 30 made Wildcard's
    # own trigger CONDITION fully mechanical. State a date here if you have
    # one in mind; leave "Not set" and Rule #41's horizon cap simply doesn't
    # apply this run (no different from before Patch 28).
    planned_chip_gw_choice = st.selectbox(
        "Next planned full-rebuild chip GW (optional)", options=["Not set"] + list(range(1, 39)),
        index=0, help="Only used for Standing Rule #41 (Disruption-Horizon Rule): if a current squad "
                       "player is flagged injured/suspended/data-flagged, the transfer-vs-hold horizon "
                       "for that decision is capped to stop before this GW, since the chip will already "
                       "reset the squad by then.")
    planned_chip_gw = None if planned_chip_gw_choice == "Not set" else int(planned_chip_gw_choice)

    meaningful_bar_override = st.slider(
        "Free-transfer materiality bar (xPts)", min_value=0.0, max_value=5.0,
        value=float(cfg["transfer"].get("minimum_meaningful_gain_free", 2.0)), step=0.25,
        help="A free transfer only gets recommended if the best swap gains at least this many xPts "
             "over your horizon. Raise it if the model is suggesting moves that don't feel worth it; "
             "lower it if it's rolling too conservatively.")

    run_clicked = st.button("Run Model →", use_container_width=True, type="primary")
    # Patch 11 — the data/projection caches below are keyed with ttl=900 (15
    # min) purely to stop a sidebar-only change (style, hit stance) from
    # re-hitting the network. That's a reasonable default the rest of the
    # time, but during a live/still-processing gameweek 15 minutes is long
    # enough for rank/points/bonus to have genuinely moved again. This button
    # clears those caches so "Run Model" is guaranteed to re-fetch right now,
    # rather than the manager wondering whether a number is wrong or just
    # cached.
    if st.button("↻ Refresh live data now", use_container_width=True,
                  help="Clears the 15-minute data cache and re-fetches from the official FPL API on the next run."):
        _load_data.clear()
        _picks.clear()
        st.session_state.has_run = True
        st.rerun()

# ---------------------------------------------------------------------------
# Run / render
# ---------------------------------------------------------------------------
if run_clicked:
    st.session_state.has_run = True

if not st.session_state.get("has_run"):
    st.markdown(f'<div class="brand-row">{_CREST_SVG}<div class="brand-mark">RB <span class="b2">Model</span></div></div>',
                unsafe_allow_html=True)
    _ni("Set style + stance → Run Model", "Set your style and hit stance in the sidebar, then click **Run Model** to fetch live data and build your recommendations.")
    st.stop()

with st.spinner("Fetching live data and computing xPts..."):
    recency_window = cfg.get("xm_heuristic", {}).get("recency_window_gws", 4)
    snap, hist_df, entry, history = _load_data(entry_id, cfg["meta"]["season"], cfg["meta"]["previous_season"],
                                                recency_window)

    if snap is None or entry is None or history is None:
        _ne("FPL API unreachable — retry", "Couldn't reach the official FPL API right now. It's normally free and open with no key required — this is "
                  "most likely a transient outage or a network policy on wherever this app is currently running. Try again shortly.")
        st.stop()

    # Patch 83 (v6.9 Standing Rule #46) -- one fixture-baselines fetch/compute
    # per run, reused by every _project() call below (Rule #22 Systematic
    # Application: the same correction must apply everywhere xPts gets
    # computed this run, not just the first call). Never blocks the page: an
    # empty/failed fetch degrades to fixture_baselines={"by_id": {}, ...}
    # and every _project() call below just runs with FF=1.0 everywhere.
    fixture_baselines, _fb_warn = _fixture_baselines(cfg["meta"]["season"], snap.current_gw, snap.teams, cfg)
    if _fb_warn:
        opt.set_diagnostic("fixture_adjustment", _fb_warn)
    elif not fixture_baselines.get("by_id"):
        opt.set_diagnostic("fixture_adjustment",
                            "no fixture-adjustment baselines this run (no finished-match data yet, e.g. GW1 "
                            "before any match has kicked off) -- goal/assist terms are unadjusted (FF=1.0) "
                            "until real match data exists to build a baseline from.")
    else:
        opt.set_diagnostic("fixture_adjustment", None)

    overrides = eng.load_overrides()
    # squad_gw = last COMPLETED/locked gameweek -- the only one the official
    # API has an actual picks snapshot for (querying a not-yet-deadlined GW
    # 404s). planning_gw = the next gameweek whose deadline hasn't passed --
    # the one every recommendation (xPts, transfers, captaincy, chips)
    # should target. These used to be the same variable, which meant the
    # whole app kept planning for a gameweek that had already been played
    # for the ~week between its deadline passing and the next one arriving.
    # current_gw_override (model_config.yaml) applies to planning_gw, since
    # that's the value Standing Rule #17's "auto-inference can be wrong"
    # caveat is actually about.
    squad_gw = snap.current_gw
    planning_gw = cfg["meta"].get("current_gw_override") or snap.planning_gw
    gw_list = list(range(planning_gw, planning_gw + horizon))

    # Patch 43 (2026-09-15, performance) — chip-availability status and the
    # GW *windows* the chip shape-test / Chip Advisor need are computed here,
    # BEFORE the (expensive, non-vectorized) first compute_all() run, so all
    # three GW ranges can be unioned into ONE shared projection instead of up
    # to 3 separate, heavily-overlapping compute_all() calls on every script
    # execution. None of this — chip_status, the window sizes below — reads
    # `proj`/`squad_df`; the only things that genuinely need the SQUAD (which
    # isn't built until after `proj` exists) are the "not squad_df.empty"
    # gates on actually USING these windows further down (wc_trigger /
    # shape_test / Chip Advisor tables) — those guards are preserved exactly
    # where they were, just decoupled from the (harmless, squad-independent)
    # list computation itself. compute_all()'s per-player set-piece-decay
    # state (sp_mult_last) is carried forward SEQUENTIALLY within one call —
    # verified safe here because the union list is one ascending run starting
    # at planning_gw, so xpts_gw{n} for any n comes out identical to what the
    # old separate calls produced (each of those also started fresh at
    # planning_gw with sp_mult_last=1.0 and walked forward in the same order).
    boot_chips = fpl_data.fetch_bootstrap_chips(snap.raw_boot) if snap.raw_boot else []
    chips_played = history.get("chips", []) if history else []
    chip_rows = chip_protocol.chip_status(boot_chips, chips_played)
    all_team_ids = snap.teams["id"].tolist() if "id" in snap.teams.columns else []

    # Patch 55 (2026-09-17, manager report: team 1301651 -- Wildcard played
    # GW4, closing window 1 (GW1-19); window 2 (GW20-38) is a genuinely
    # separate, legitimately "available" calendar window per chip_status(),
    # just not open for another ~15 GWs. The OLD _wc_available_now/
    # _fh_available_now here were a raw "does ANY window have status==
    # available" check with no notion of whether that window overlaps the GW
    # actually being planned -- the exact same scan-range-blindness Patch 54
    # fixed for the Bench Boost/Triple Captain/Free Hit ADVISOR CARDS via
    # _clip_to_available_windows(), but that fix was never extended to (a)
    # the Wildcard trigger card, which is built directly (wc_card, not
    # through _advisor_card) and (b) the shape-test gate these two flags also
    # feed, independent of the advisor cards. Confirmed via a live probe
    # against chip_protocol.chip_status() with this exact scenario:
    # _wc_available_now stayed True with zero GWs of the open window
    # anywhere near the GW5-8 planning window, firing a fabricated "TRIGGER
    # ACTIVE 93.4%" Wildcard card 15 GWs before that window could ever
    # actually be played. Bench Boost/Triple Captain/Free Hit's own advisor
    # CARDS were independently confirmed correct (see manager's own live
    # test) since they already route through _clip_to_available_windows
    # further below -- this fix only touches the Wildcard/detect-window gate.
    #
    # Fix: _clip_to_available_windows() (previously defined further below,
    # next to its first use for the Chip Advisor cards) now lives here
    # instead, right after chip_rows exists, so this earlier gate uses the
    # exact same clipping logic rather than a second, divergent one. Both
    # _wc_available_now and _fh_available_now now mean "is an available
    # window's [start,stop] range actually reachable from the same detect
    # window (planning_gw..planning_gw+detection_window_gws-1) that
    # wc_trigger/shape_test will actually scan" -- not "does an available
    # window exist anywhere on the calendar, however far off."
    def _clip_to_available_windows(gw_list: list[int], name_prefix: str):
        """Returns (clipped_gw_list, last_used_event, next_available_start).
        clipped_gw_list is gw_list filtered to GWs inside any window still
        "available" for this chip name; last_used_event is the most recent
        GW this chip was actually played (None if never); next_available_
        start is the earliest start_event among its available windows."""
        avail_windows = [r["window"] for r in chip_rows
                          if r["chip"].startswith(name_prefix) and r["status"] == "available"]
        used_events = [r["event"] for r in chip_rows
                        if r["chip"].startswith(name_prefix) and r["status"] == "used" and r["event"] is not None]
        last_used = max(used_events) if used_events else None
        next_start = min((w[0] for w in avail_windows if w[0] is not None), default=None)
        if not avail_windows or not gw_list:
            return [], last_used, next_start
        clipped = [g for g in gw_list
                   if any(s is not None and e is not None and s <= g <= e for s, e in avail_windows)]
        return clipped, last_used, next_start

    available_chip_names = {r["chip"] for r in chip_rows if r["status"] == "available"}

    # Patch 107 (2026-10-04, manager: the extended check must be "Current GW +
    # 9" and re-evaluate ALL chips). `_extended_mode` is the opt-in button's
    # session-state flag, read here (before anything is projected or solved)
    # so the chip window, the Wildcard detection window and the projection
    # itself can all be widened together for this one run. Normal runs keep
    # exactly the configured windows.
    _extended_mode = bool(st.session_state.get("wc_extend_run", False))
    _ext_span = int(cfg.get("chip_extended_check", {}).get("span_gws", 10))
    # Patch 110: BB/TC/FH windows in the extended run reach Current+11 (>=4 GWs past the normal window).
    _chip_span = int(cfg.get("chip_extended_check", {}).get("chip_span_gws", 12))
    _detect_window_size_base = cfg.get("chip_shape_test", {}).get("detection_window_gws", 4)
    _detect_window_size = max(_detect_window_size_base, _ext_span) if _extended_mode else _detect_window_size_base
    _detect_candidate_gws = list(range(planning_gw, planning_gw + _detect_window_size))
    _wc_clipped_gws, _wc_last_used, _wc_next_open = _clip_to_available_windows(_detect_candidate_gws, "Wildcard")
    _fh_clipped_detect_gws, _fh_last_used_detect, _fh_next_open_detect = _clip_to_available_windows(
        _detect_candidate_gws, "Free Hit")
    _wc_available_now = bool(_wc_clipped_gws)
    _fh_available_now = bool(_fh_clipped_detect_gws)

    detect_gw_list = None
    if _wc_available_now or _fh_available_now:
        detect_gw_list = _detect_candidate_gws

    chip_adv_window = None
    chip_adv_gw_list = None
    if any(c.startswith(("Bench Boost", "Triple Captain", "Free Hit")) for c in available_chip_names):
        chip_adv_window = chip_protocol.chip_advisor_gw_window(
            planning_gw, snap.fixtures, all_team_ids, cfg, min_gws=_chip_span if _extended_mode else None)
        chip_adv_gw_list = chip_adv_window["gw_list"]

    # Patch 44 (2026-09-15, manager report: Damsgaard-vs-Tavernier tie-break
    # still not firing at a 1-GW horizon, even though it correctly fires once
    # the horizon is widened to 2 GWs) — root cause: before this patch, `proj`
    # only ever carried columns for whatever GW range was actually requested.
    # At horizon=1, that's a single GW; recommend._position_tie_break()'s
    # extended-horizon comparison needs `xpts_gw{max(gw_list)+1}` to break a
    # near-tie, and when a chip window happened to widen `proj` anyway (this
    # week's Wildcard/Free Hit/Bench Boost/Triple Captain all still available)
    # that extra GW was present as a side effect of Patch 43's merge — but a
    # week with every chip already used would have silently gone back to
    # missing that data and the tie-break bailing out ("keeping the model's
    # original pick"). Manager confirmed (2026-09-15) this should be
    # guaranteed, not a lucky side effect: `gw_list[-1] + 1` is now always
    # folded into the shared projection union, independent of chip
    # availability — one extra GW's worth of xPts columns, not a second
    # compute_all() call or a new MILP solve.
    _tie_break_lookahead_gw = gw_list[-1] + 1

    # Patch 51 (2026-09-16, §1a compliance fix) — the doc's Team Rating %
    # clause requires a FIXED 3-4 GW horizon (Standing Rule cite: §1a
    # requirement (a)), never the sidebar horizon slider (`gw_list`, which
    # can be as short as 1 GW depending on hit_stance) and never dependent
    # on whether a chip happens to be available (`detect_gw_list` is None
    # whenever neither Wildcard nor Free Hit is currently available, which
    # would otherwise silently starve `proj` of the columns this metric
    # needs). `compliant_gw_list` is that fixed window, built the same way
    # detect_gw_list already is (same config key, so the two windows agree
    # whenever detect_gw_list also happens to be populated) — and is always
    # folded into the shared projection union below so `proj` unconditionally
    # carries these GW columns regardless of chip availability or the
    # horizon slider's position.
    _compliant_window = cfg.get("chip_shape_test", {}).get("detection_window_gws", 4)
    compliant_gw_list = list(range(planning_gw, planning_gw + _compliant_window))

    # Patch 106/107: the extension GWs have to exist in `proj` or the
    # cross-check's `_check_gws` filter silently drops them (Patch 106 root
    # cause). In extended mode project Current..Current+9 PLUS the trailing
    # GWs a 4-GW Wildcard window needs for the last candidates.
    _extend_proj_gws = set()
    if _extended_mode:
        _extend_proj_gws = set(recommend.extended_proj_gws(
            planning_gw, _ext_span, cfg.get("chip_portfolio", {}).get("window_value_len", 4)))
    _gw_union = sorted(set(gw_list) | set(detect_gw_list or []) | set(chip_adv_gw_list or [])
                        | set(compliant_gw_list) | {_tie_break_lookahead_gw} | _extend_proj_gws)
    proj = _project(snap, hist_df, overrides, cfg, _gw_union, fixture_baselines)
    picks = _picks(entry_id, squad_gw)

    id_to_code = proj.set_index("id")["code"].to_dict() if "id" in proj.columns else {}
    squad_codes, captain_id, bench_codes = [], None, []
    if picks and "picks" in picks:
        for pk in picks["picks"]:
            code = id_to_code.get(pk["element"])
            if code is None:
                continue
            squad_codes.append(code)
            if pk.get("is_captain"):
                captain_id = code
            if pk.get("position", 1) > 11:
                bench_codes.append(code)

    squad_df = proj[proj["code"].isin(squad_codes)].copy()

    # Patch 117d: owned players are priced at what you would RECEIVE (selling price) in every solve; the live price stays in
    # `market_price`. Keeping a risen player costs his selling price, selling returns it, buying costs the live price.
    _sp, _sell_total, _sell_nest, _market_sum = None, None, 0, 0.0
    if not squad_df.empty:
        _market_sum = float(squad_df["price"].sum())
        _tr = _load_transfers(entry_id)
        if _tr is not None:
            _fh_events = {c.get("event") for c in (history.get("chips", []) if history else [])
                          if str(c.get("name", "")).lower() in ("freehit", "free_hit")}
            _sp = recommend.selling_prices(squad_df, _tr, skip_events=_fh_events)
            _sell_total, _sell_nest = _sp["total"], _sp["n_estimated"]
            proj = recommend.apply_sell_prices(proj, _sp)
            squad_df = proj[proj["code"].isin(squad_codes)].copy()
    # Locked players (sidebar): kept by the Wildcard rebuild, never sold by the planner. Reset when the team changes.
    _lock_names = sorted(squad_df["web_name"].tolist()) if not squad_df.empty else []
    _lock_key = f"locked_{entry_id}"
    st.session_state[_lock_key] = [n for n in st.session_state.get(_lock_key, []) if n in _lock_names]
    with _lock_slot:
        locked_names = st.multiselect("🔒 Locked players", _lock_names, key=_lock_key,
                                      help="Kept in the Wildcard, never sold in transfers.")
    _name_to_code = dict(zip(squad_df["web_name"], squad_df["code"])) if not squad_df.empty else {}
    locked_codes = [int(_name_to_code[n]) for n in locked_names if n in _name_to_code]
    proj["_locked"] = proj["code"].isin(locked_codes)
    # Patch 117j: Excluded players (sidebar): never bought by the weekly planner, never in a Wildcard rebuild. Same scope as
    # Locked players (ceilings, the Free Hit optimum and the Rating % ignore it). Only players you do not own can be excluded.
    _ex_pool = proj[~proj["code"].isin(squad_codes)] if "code" in proj.columns else proj.iloc[0:0]
    _ex_label = {int(r_["code"]): f"{r_['web_name']} ({r_['team']})" for _, r_ in _ex_pool.iterrows()}
    _ex_names = sorted(_ex_label.values())
    _ex_key = f"excluded_{entry_id}"
    st.session_state[_ex_key] = [n for n in st.session_state.get(_ex_key, []) if n in _ex_names]
    with _lock_slot:
        excluded_names = st.multiselect("🚫 Excluded players", _ex_names, key=_ex_key,
                                        help="Never bought in transfers, never picked in the Wildcard rebuild.")
    _ex_by_label = {v: k for k, v in _ex_label.items()}
    excluded_codes = [int(_ex_by_label[n]) for n in excluded_names if n in _ex_by_label]
    proj["_excluded"] = proj["code"].isin(excluded_codes)
    squad_df = proj[proj["code"].isin(squad_codes)].copy()
    _lock_flagged = recommend.lock_flags(squad_df, locked_codes)

    # Patch 2 — auto-optimized XI: the pitch, captaincy, chip advisor, and the
    # new GWn xPts stat all render the BEST valid formation from your actual
    # 15 for this gameweek (highest projected xpts_gw{planning_gw}), not a
    # copy of whatever arrangement your live FPL team happens to have set.
    # Falls back to the live split only if the optimizer can't produce a
    # valid XI (e.g. incomplete GW data right after a deadline).
    opt_col = f"xpts_gw{planning_gw}"
    optimized_xi = opt.best_starting_xi(squad_df, opt_col) if (not squad_df.empty and opt_col in squad_df.columns) else None
    if optimized_xi is not None:
        starters_df = optimized_xi["xi"]
        bench_df = squad_df[~squad_df["code"].isin(starters_df["code"])]
        gw_xpts_total = round(optimized_xi["total"], 1)
    else:
        bench_df = squad_df[squad_df["code"].isin(bench_codes)]
        starters_df = squad_df[~squad_df["code"].isin(bench_codes)]
        gw_xpts_total = round(starters_df[opt_col].sum(), 1) if (opt_col in starters_df.columns and not starters_df.empty) else 0.0

    pool_df = proj[~proj["code"].isin(squad_codes)].copy()

    # rank history + points from entry history
    cur_hist = history.get("current", []) if history else []
    rank_history = [r.get("overall_rank") for r in cur_hist if r.get("overall_rank") is not None]
    points_total = entry.get("summary_overall_points") if entry else None
    hits_last_3 = sum(1 for r in cur_hist[-3:] if (r.get("event_transfers_cost") or 0) > 0)

    # Patch 12 — the "points are right but rank is off" bug: `entry/{id}/`
    # and `entry/{id}/history/` are two DIFFERENT official-API fields for the
    # same thing, and they don't update in lockstep. `summary_overall_points`
    # (used above for points_total) happens to match the front-end's own
    # points display, but the rank header was built from
    # `history["current"][-1]["overall_rank"]` — a snapshot written into that
    # GW's history ROW, which lags behind `entry["summary_overall_rank"]`
    # (the field the official FPL app/site actually displays as your current
    # Overall Rank). Confirmed on the manager's own live data: history showed
    # 1,436,772 for GW3 while entry.summary_overall_rank showed 1,438,164 at
    # the same moment — a real, verifiable field-source mismatch, not a
    # caching artifact. Fix: the live rank always comes from
    # `entry["summary_overall_rank"]` — replacing (not just appending to)
    # the last slot of the display series, so the header and the trend arrow
    # both use the same live-correct source. `rank_history` itself is left
    # untouched for the Season Ledger table further down, since each PAST
    # (already-finalized) row there is its own historical record, not a
    # "current standing" claim.
    live_overall_rank = entry.get("summary_overall_rank") if entry else None
    rank_history_display = list(rank_history)
    if live_overall_rank is not None:
        if rank_history_display:
            rank_history_display[-1] = live_overall_rank
        else:
            rank_history_display = [live_overall_rank]

    verdict = recommend.season_verdict(rank_history_display, hits_last_3, squad_gw)

    # free transfers + bank — moved ahead of Team Rating % (below) because the
    # reachable-ceiling solve needs free_transfers to set its min-retain constraint.
    ft = transfers.derive_free_transfers(cur_hist, history.get("chips", []) if history else [])
    bank = (entry.get("last_deadline_bank", 0) or 0) / 10.0 if entry else 0.0

    # Team Rating % (§1a) — reworked (Patch 1): the headline number is now the
    # RESEARCHED-TIER ratio against a REACHABLE ceiling (best squad actually
    # gettable this week using only the free transfers on hand), not an
    # unconstrained fantasy-ideal squad nobody could reach in one week
    # regardless of research quality. The old unconstrained ceiling is kept
    # as a secondary "theoretical" reference. Standing Rule #34's
    # margin-of-error band is applied to the headline so a gap inside
    # demonstrated weekly noise reads as "at ceiling," not a misleadingly
    # precise decimal.
    # Patch 117d: budget = real bank + selling prices (squad_df is already priced at selling prices); manual box overrides.
    bank, team_value, _budget_note, _budget_src = recommend.resolve_budget(
        bank, (squad_df["price"].sum() if not squad_df.empty else 0.0), _sell_total, budget_override, _sell_nest,
        market_sum=_market_sum if _sell_total is not None else None)
    with _snap_slot:
        try:
            _ov_bytes = open("manual_overrides.csv", "rb").read() if os.path.exists("manual_overrides.csv") else None
            st.download_button("⬇ Download run snapshot", data=recommend.build_snapshot_zip(
                proj, list(squad_codes), locked_codes, bank, team_value,
                {"patch": PATCH_TAG, "entry_id": entry_id, "planning_gw": planning_gw, "budget_source": _budget_src,
                 "free_transfers": ft["free_transfers"], "sell_total": _sell_total, "market_sum": _market_sum},
                cfg, _ov_bytes, excluded_codes=excluded_codes), file_name=f"rb_snapshot_gw{planning_gw}.zip", mime="application/zip")
        except Exception:
            pass
    reachable = _reachable_ceiling_single_calc(cfg, proj, tuple(squad_codes), ft["free_transfers"])
    theoretical_ceiling = _theoretical_ceiling_calc(cfg, proj)
    # Patch 20 (2026-09-07 discussion) — §1a's own formula requires
    # Squad_xPts/Ceiling_xPts to include "captaincy applied per Step 7's
    # joint per-week XI+captain evaluation," not a flat 15-man raw sum
    # (which is what this used to be, despite the UI's own tooltip already
    # claiming "your optimized XI's projected xPts" — that text was
    # aspirational until now). opt.rating_horizon_value() picks the best
    # legal XI per GW, doubles the XI's own top scorer (the captain bonus),
    # and values the 4 bench slots at their Rule #12 autosub-discounted
    # rate instead of full value — applied identically to all three totals
    # below (Rule #22 Systematic Application), never a raw sum on one side
    # and this calculation on the other. Deliberately NOT the same function
    # transfer-path/Wildcard comparisons use (opt.realized_horizon_value,
    # no captaincy) — see rating_gw_value()'s docstring for why those two
    # must stay separate (Standing Rule #31).
    squad_total = opt.rating_horizon_value(squad_df, gw_list, cfg) if not squad_df.empty else 0.0
    reachable_total = opt.rating_horizon_value(reachable["squad"], gw_list, cfg) if reachable else 0.0
    theoretical_total = opt.rating_horizon_value(theoretical_ceiling["squad"], gw_list, cfg) \
        if theoretical_ceiling else 0.0
    moe = eng.margin_of_error_threshold(reachable_total, cfg)
    rating_gap = round(reachable_total - squad_total, 2)
    at_ceiling = reachable_total > 0 and rating_gap < moe

    # Patch 51 (2026-09-16, §1a compliance fix) — the genuinely §1a-compliant
    # "Team Rating %": Squad_xPts / Ceiling_xPts over a FIXED 3-4 GW horizon
    # (`compliant_gw_list`, computed above and independent of the sidebar
    # horizon slider), with the Ceiling side a full-player-pool, unconstrained,
    # complete 15-man £100m squad solve (§1a requirement (b)) — REUSING
    # `theoretical_ceiling` (already solved once, above, for the "theoretical"
    # reference) rather than triggering a second MILP call. Both sides go
    # through the same captaincy-aware opt.rating_horizon_value() the other
    # Team Rating variants use (Rule #22 Systematic Application), just summed
    # over `compliant_gw_list` instead of whatever `gw_list` happens to be.
    # This is DELIBERATELY separate from the header's "Quick Team Rating"
    # badge below (the old single-GW `fh_auto_rating`, kept unchanged on the
    # manager's explicit call re: ceiling choice — Patch 24) — see Standing
    # Rules #16/#18 disclosure and the Patch 49 precedent on never showing two
    # differently-scoped percentages without a cross-reference between them.
    compliant_squad_total = opt.rating_horizon_value(squad_df, compliant_gw_list, cfg) if not squad_df.empty else 0.0
    compliant_ceiling_total = opt.rating_horizon_value(theoretical_ceiling["squad"], compliant_gw_list, cfg) \
        if theoretical_ceiling else 0.0
    compliant_moe = eng.margin_of_error_threshold(compliant_ceiling_total, cfg)
    compliant_gap = round(compliant_ceiling_total - compliant_squad_total, 2)
    compliant_at_ceiling = compliant_ceiling_total > 0 and compliant_gap < compliant_moe
    compliant_rating = eng.team_rating_pct(compliant_squad_total, compliant_ceiling_total, "")
    compliant_gw_start, compliant_gw_end = compliant_gw_list[0], compliant_gw_list[-1]

    # Researched-tier coverage — live count of how much of manual_overrides.csv's
    # qualitative layer (xm_override / cs_pct_override / bps_profile /
    # tenure_discount — anything that promotes a player past the free
    # MECHANICAL-TIER default) actually applies to THIS squad and to the pool
    # at large, replacing the old static disclosure string with a real number
    # that moves as manual_overrides.csv is researched further.
    override_cols = ["xm_override", "cs_pct_override", "bps_profile", "tenure_discount"]
    if not overrides.empty:
        researched_codes = set(overrides.dropna(subset=override_cols, how="all")["player_code"])
    else:
        researched_codes = set()
    squad_researched = len(set(squad_codes) & researched_codes)
    pool_all_codes = set(proj["code"]) if "code" in proj.columns else set()
    pool_researched = len(pool_all_codes & researched_codes)

    tier_label = (f"Squad researched-tier coverage: **{squad_researched}/{len(squad_codes) or 15}** players have at "
                  f"least one manual_overrides.csv entry (xm_override / cs_pct_override / bps_profile / "
                  f"tenure_discount) promoting them past the free MECHANICAL-TIER default (MODEL_POISSON CS%, "
                  f"xM Floor Rule only). Pool-wide coverage: **{pool_researched}/{len(pool_all_codes)}**. "
                  f"Steps 4 (full Role Multiplier table), 4a (Manager Tenure Split), 5 (Pre-Season Evidence) and "
                  f"6 (Manager System Fit) need web research/judgment to promote a player past MECHANICAL-TIER — "
                  f"this coverage count is repo-wide, so every visitor to this app's URL sees the same upgraded "
                  f"numbers for any player that's been researched, not just the manager who requested it. "
                  f"Standing Rules #16/#18 disclosure.")
    rating = eng.team_rating_pct(squad_total, reachable_total, tier_label)

    # Free Hit rating — computed automatically every run (Patch 22,
    # 2026-09-07 discussion), not gated behind the manual "Evaluate your
    # own scenario" picker any more. Targets the CURRENT planning_gw only
    # (never a manager-chosen candidate date — that manual picker in
    # "Evaluate your own scenario" stays exactly as it was, untouched, for
    # exploring a DIFFERENT gameweek than this one). Reuses `proj` as-is —
    # planning_gw is always inside gw_list, so no extra re-projection call
    # is needed the way the manual scenario picker needs one for an
    # out-of-horizon date. Same opt.rating_gw_value() mechanic as the main
    # Team Rating % (Patch 20) and the manual FH comparison (Patch 21),
    # just against a genuinely unconstrained single-GW ceiling instead of
    # the free-transfer-limited "reachable ceiling" — see the discussion
    # earlier this session on why that makes this number move more
    # meaningfully than the main headline can.
    fh_auto_col = f"xpts_gw{planning_gw}"
    fh_auto_result = _best_gw_squad_calc(cfg, proj, team_value, planning_gw)
    fh_auto_current_val = opt.rating_gw_value(squad_df, fh_auto_col, cfg)["total_realized"] \
        if not squad_df.empty else 0.0
    fh_auto_optimal_val = opt.rating_gw_value(fh_auto_result["squad"], fh_auto_col, cfg)["total_realized"] \
        if fh_auto_result else 0.0
    fh_auto_rating = eng.team_rating_pct_capped(fh_auto_current_val, fh_auto_optimal_val, "")
    fh_auto_gap = round(fh_auto_optimal_val - fh_auto_current_val, 2)
    fh_auto_moe = eng.margin_of_error_threshold(fh_auto_optimal_val, cfg) if fh_auto_optimal_val else 0.0
    # Patch 24 (2026-09-08 discussion) — this FH-optimal-based ratio now IS
    # the header's "Team Rating" (manager's explicit call: it's a truer
    # current-vs-optimal read than the old reachable-ceiling version, which
    # is tautologically high whenever few free transfers are banked). The
    # old reachable-ceiling calc (`rating`/squad_total/reachable_total/moe)
    # is kept as-is for the Wildcard-flag trigger only (chip_protocol.wildcard_flag
    # below) — untouched, not displayed anywhere any more.
    # Patch 51 (2026-09-16, §1a compliance fix) — relabeled from "Team Rating
    # %" to "Quick Team Rating — single-GW, EST" (manager-approved wording).
    # The MATH here is unchanged from Patch 24 (same numbers, same ceiling
    # choice) — this is a label/disclosure-only change. It's relabeled
    # because it violates §1a requirement (a) (a fixed 3-4 GW horizon):
    # this metric is scored against GW{planning_gw} ALONE, never a multi-GW
    # window, so per the model doc it "is not a Team Rating % under this
    # clause — it's a bounded estimate and must be labeled as such." The
    # doc-compliant multi-GW metric now lives in its own separate badge,
    # "Team Rating % (GW{compliant_gw_start}-{compliant_gw_end}, full pool)"
    # — see compliant_rating below — and this tooltip cross-references it so
    # the two differently-scoped percentages are never confused for one
    # another (Patch 49 precedent: "where the points we discussed!!!!").
    # Patch 52 (2026-09-16, manager screenshot) — relabeled from the generic
    # "Quick Team Rating" to include the actual GW it scores (`planning_gw`,
    # confirmed above via fh_auto_result = data_pipeline.
    # solve_free_hit_optimal_squad(cfg, proj, team_value, planning_gw)) so
    # it's self-explanatory at a glance which week it's scoring, without
    # needing the tooltip. Math unchanged — label/disclosure only.
    fh_auto_label = f"GW{planning_gw} Rating"
    fh_auto_tooltip = (f"GW{planning_gw} Rating (single-GW, EST) = your current squad's best XI this GW (captain "
                       f"doubled, bench autosub-discounted), divided by a genuinely unconstrained optimal squad "
                       f"for GW{planning_gw} only (full player pool, no free-transfer limit — a true from-scratch "
                       f"rebuild). Gap: {fh_auto_gap:.1f} xPts (margin-of-error threshold: {fh_auto_moe:.1f} xPts). "
                       f"NOTE: this is a single-GW read, NOT the model doc's §1a Team Rating % (which requires a "
                       f"fixed 3-4 GW horizon) — see 'Team Rating % (GW{compliant_gw_start}-{compliant_gw_end})' "
                       f"below (full-pool, {len(compliant_gw_list)}-GW) for that doc-compliant metric. "
                       f"Explore a different candidate gameweek in 'Evaluate your own scenario' below.")
    # Patch 72 (manager, 2026-09-27: this card started showing "—" after
    # Patch 71 — the crash was fixed but nothing explained WHY the solve
    # behind it kept returning None on live data). If fh_auto_result is
    # None, opt.get_diagnostic("free_hit_optimal") now carries the actual
    # reason (recorded inside data_pipeline.solve_free_hit_optimal_squad /
    # optimizer.solve_xi_first_squad themselves) — surfaced directly here so
    # the NEXT live run shows the real cause with no Streamlit Cloud log
    # access needed.
    if fh_auto_result is None:
        _fh_diag = opt.get_diagnostic("free_hit_optimal")
        fh_auto_tooltip += (f" ⚠ Diagnostic (Patch 72): this run's solve returned no result — {_fh_diag}"
                            if _fh_diag else
                            " ⚠ Diagnostic (Patch 72): this run's solve returned no result, but no specific "
                            "reason was recorded — please report this exact combination so it can be added.")

    # captaincy — starting XI only, never the bench. "code" is carried through
    # (Patch 2) so the pitch view can match the recommendation back to its
    # card and place the armband there directly, rather than just displaying
    # the pick in its own section.
    cap_pick_row, cap_alt_row, cap_alt_label, cap_caption = None, None, "Alternative", None
    if not starters_df.empty:
        cap_col = f"xpts_gw{gw_list[0]}"
        cap_candidates = starters_df.rename(columns={cap_col: "xpts_this_gw"})[
            ["code", "web_name", "team", "xpts_this_gw", "selected_by_percent"]]
        cap_result = eng.captaincy_protocol(cap_candidates, cfg)
        # Patch 27 (v6.3 / Standing Rule #40) — team-level results-form
        # tiebreak, built from finished-fixture scores already in `snap`
        # (no new manual research field). Only ever narrows an
        # already-tied shortlist; see fpl_engine.team_stability_tiebreak's
        # docstring for the disclosed EST proxy this uses.
        team_table = eng.team_league_table(snap.fixtures, snap.teams)
        cap_pick = style_profiles.captaincy_pick(cap_result, style_name, team_table=team_table, cfg=cfg)
        cap_alt_row, cap_alt_label = style_profiles.captain_alt_pick(cap_result, cap_pick["web_name"], style_name)
        cap_pick_row = cap_pick

        # Patch 4 — captaincy-on-pitch caption: a single themed line replacing
        # the old standalone "Captaincy Pick" section. Genuinely distinguishes
        # a clear standout week (shortlist of one) from a real statistical tie
        # (Standing Rule #34's margin-of-error window), rather than always
        # phrasing it as a coin-flip.
        shortlist_ct = int(cap_result["shortlisted"].sum()) if "shortlisted" in cap_result.columns else 1
        cap_xp = cap_pick_row["xpts_this_gw"]
        if shortlist_ct <= 1:
            cap_caption = (f"🎯 Armband: <b>{cap_pick_row['web_name']}</b> — the standout pick this week "
                           f"({cap_xp:.1f} xPts, clear of the field).")
        elif cap_alt_row is not None and not cap_alt_label.startswith("Near miss"):
            cap_caption = (f"🎯 Armband: <b>{cap_pick_row['web_name']}</b> — a coin-flip with "
                           f"{cap_alt_row['web_name']} this week ({cap_xp:.1f} xPts); {cap_alt_label.lower()} "
                           f"given per your style profile (<b>{style_name}</b>).")
        else:
            cap_caption = (f"🎯 Armband: <b>{cap_pick_row['web_name']}</b> — a coin-flip within the shortlist "
                           f"this week ({cap_xp:.1f} xPts).")
        if cap_alt_row is not None and cap_alt_label.startswith("Near miss"):
            cap_caption += (f" Nearest alternative if this pick disappoints: <b>{cap_alt_row['web_name']}</b> "
                            f"({cap_alt_label.replace('Near miss – ', '')}, outside this week's shortlist).")
        team_stability_note = cap_pick_row.get("team_stability_note")
        if team_stability_note:
            cap_caption += (f" <i>Team-Stability tiebreak (Rule #40): {cap_pick_row['web_name']}'s team wins the "
                            f"tie on {team_stability_note} — an EST-tagged results-form proxy, not literal "
                            f"comeback detection (no goal-minute data available); see the model doc for the "
                            f"full disclosure.</i>")

    # chip status + timing — computed before transfer suggestions so the
    # transfer plan can factor in "a chip is coming, banking may beat spending"
    # (chip_rows / all_team_ids themselves now computed earlier, Patch 43 —
    # see the comment above the first _project() call)
    fixture_counts = chip_protocol.fixture_counts_by_team(snap.fixtures, gw_list)
    dgw_bgw = chip_protocol.dgw_bgw_flags(fixture_counts, all_team_ids)
    squad_team_ids = squad_df["team_id"].tolist() if "team_id" in squad_df.columns else []
    chip_notes = chip_protocol.chip_recommendations(chip_rows, dgw_bgw, squad_team_ids, max(len(squad_df), 1))
    flagged_players = squad_df[(squad_df["status"] != "a") | (squad_df["est_rescue_needed"])]

    # Patch 30 (2026-09-14) — REPLACES the old ad hoc rank-decline/flagged-
    # player-count heuristic with v6.4's actual documented Wildcard trigger
    # (average Team Rating % below ~78-80%, or a cumulative gap to the
    # bounded-ceiling optimal of ~15+ xPts, over a 3-4 GW detection window).
    # Also corrects a mislabeling: the old code cited "Standing Rule #24"
    # (Transfer Timing Discipline Rule — about ordinary transfers, not
    # Wildcard) as the reason Wildcard stayed non-mechanical; the actual
    # governing rule is #32 (Dynamic Chip Timing Rule), which blocks locking
    # in a DATE, not computing the trigger condition. Needs a squad+
    # reachable-ceiling pair projected onto the detection window specifically
    # (may be wider than the sidebar horizon), computed once here and reused
    # by both the trigger and the Step 8c shape-test below.
    # _wc_available_now / _fh_available_now / detect_gw_list computed earlier
    # (Patch 43); shape_proj is now just a view into the shared `proj` (which
    # already contains every GW column detect_gw_list needs, since it was
    # folded into the union before the first — and only — compute_all() run).
    wc_trigger = None
    shape_test = None
    shape_proj = proj if (detect_gw_list is not None and not squad_df.empty) else None

    # Patch 73 (manager, 2026-09-27: confirmed in code that Patch 72's
    # Wildcard diagnostic could NEVER fire — eng.wildcard_trigger_check()
    # does not return None on failure, it returns a placeholder dict
    # (`empty = {"active": False, "avg_rating_pct": None, ..., "reason":
    # "insufficient data to evaluate this run"}`, see fpl_engine.py). Patch
    # 72's app.py check was `wc_trigger is None`, which that placeholder
    # dict never satisfies — so the tooltip kept showing the exact same
    # generic text as before Patch 72, with the diagnostic code silently
    # never executing. This is a genuinely different bug from anything
    # Patch 70/71/72 touched, found only by reading wildcard_trigger_check's
    # actual return statements, not by re-guessing.
    #
    # wc_diag_reason is now built HERE, at the point each precondition is
    # actually known, rather than trying to reverse-engineer which of
    # wildcard_trigger_check's four possible empty-triggering conditions
    # fired from its output alone (its own "reason" field doesn't
    # distinguish them either — all four collapse to the same generic
    # string).
    wc_diag_reason = None
    if not _wc_available_now:
        wc_diag_reason = ("Wildcard isn't available to evaluate this run (_wc_available_now is False — e.g. "
                           "already played with no next window open yet this season, or otherwise out of season).")
    elif shape_proj is None:
        wc_diag_reason = ("detect_gw_list is empty/None, or the current squad is empty this run — the Wildcard "
                           "trigger has no gameweek window or squad to evaluate against.")
    else:
        squad_detect = shape_proj[shape_proj["code"].isin(squad_codes)]
        # Patch 82 (v6.9 amended §8c trigger) — one reachable squad PER GW in
        # detect_gw_list, each with that GW's own accrued free-transfer count
        # (data_pipeline.solve_reachable_ceiling_by_gw()), replacing the old
        # single shared solve reused across the whole window.
        # Patch 99 -- routed through the cached _reachable_ceiling_calc()
        # instead of calling data_pipeline.solve_reachable_ceiling_by_gw()
        # bare (was uncached: one fresh MILP solve per GW in detect_gw_list,
        # re-run on every unrelated-widget rerun).
        reachable_by_gw = _reachable_ceiling_calc(
            cfg, shape_proj, tuple(sorted(squad_codes)), ft["free_transfers"], tuple(detect_gw_list))
        # Patch 117a: the trigger reads only the first detection_window_gws GWs (same rule as
        # recommend.trigger_detect_window); an Extended run's 10-GW list is context, not the trigger.
        _trigger_gw_list = list(detect_gw_list or [])[:int(cfg.get("chip_shape_test", {}).get("detection_window_gws", 4))]
        wc_trigger = eng.wildcard_trigger_check(squad_detect, reachable_by_gw, _trigger_gw_list, cfg)
        if wc_trigger.get("avg_rating_pct") is None:
            _any_reachable = any(v is not None and v.get("squad") is not None and not v["squad"].empty
                                  for v in reachable_by_gw.values())
            if squad_detect.empty:
                wc_diag_reason = (f"squad_detect is empty — none of the {len(squad_codes)} current squad codes "
                                   f"matched this run's projected pool.")
            elif not _any_reachable:
                _rc_diag = opt.get_diagnostic("reachable_ceiling")
                wc_diag_reason = (f"every per-GW reachable-squad solve (data_pipeline."
                                   f"solve_reachable_ceiling_by_gw) returned no result — {_rc_diag}" if _rc_diag else
                                   "every per-GW reachable-squad solve returned no result, but no specific reason "
                                   "was recorded — please report this exact combination so it can be added.")
            elif not detect_gw_list:
                wc_diag_reason = "detect_gw_list is empty."
            else:
                wc_diag_reason = ("no GW in detect_gw_list had a valid xpts_gw{n} column present in both the "
                                   "current squad's and that GW's own reachable-squad projections this run.")

    wc_flag = chip_protocol.wildcard_trigger_flag(wc_trigger, rank_history_display) if wc_trigger else None

    # Patch 28 (v6.4 / Standing Rule #41 + its Rule #24 override) — reuses
    # the same `flagged_players` definition as the Wildcard trigger above so
    # the two never disagree. Only produces a capped horizon / note when a
    # squad player is actually currently disrupted; a silent no-op otherwise.
    disruption = eng.disruption_check(squad_df, gw_list, planned_chip_gw)
    transfer_gw_list = gw_list
    if disruption["capped_gw_list"] is not None:
        transfer_gw_list = disruption["capped_gw_list"] or gw_list[:1]

    # Patch 28 (v6.4 / Step 8c shape-test) — cross-checks whatever Wildcard/
    # Free Hit signal already fired above against a genuine multi-GW optimal-
    # squad-shape solve, so a fixture-shaped spike is never read as sustained
    # Wildcard evidence (or vice versa). Reuses the same detect_gw_list/
    # shape_proj the trigger above already computed.
    if (wc_flag or _fh_available_now) and shape_proj is not None:
        shape_test = _shape_test_calc(
            squad_df, shape_proj, cfg, tuple(detect_gw_list) if detect_gw_list else (), team_value)

    # Chip Advisor (v5.0 / Patch 1) — quantified play/hold verdicts within the
    # chosen horizon for the three chips that actually have a "which GW"
    # question (Bench Boost, Triple Captain, Free Hit). Only solved for chips
    # that are actually still available this season — each Free Hit check is
    # a fresh MILP solve per horizon GW, so it's skipped entirely once that
    # chip is used, rather than burning compute on a verdict nobody can act on.
    # available_chip_names computed earlier (Patch 43)
    # Chip-specific margin-of-error thresholds (Patch 5). Standing Rule #34
    # itself only defines ONE blanket band (max(2.0, 2%)) — the model doc
    # does not specify separate numeric thresholds per chip. Using a single
    # band for a chip verdict was flagged as wrong by the manager: a Bench
    # Boost verdict rests on 4 players' summed variance, a Triple Captain
    # verdict rests on a single player's single week (much higher variance),
    # and a Free Hit verdict is a full-squad rebuild whose chip cost is
    # burned regardless of outcome (highest stakes). `chip_advisor_thresholds`
    # in model_config.yaml is a disclosed, manager-directed EXTENSION beyond
    # Rule #34's text — never presented as if the source document specified
    # it — falling back to the generic band for any chip left unconfigured.
    cat_cfg = cfg.get("chip_advisor_thresholds", {})

    def _chip_moe_fn(chip_key: str):
        t = cat_cfg.get(chip_key, {})
        return lambda total: eng.margin_of_error_threshold(
            total, cfg, floor_points=t.get("floor_points"), pct_of_total=t.get("pct_of_total"))

    # Patch 32 (2026-09-14 manager report) — the Chip Advisor scans its OWN
    # window now, independent of the sidebar's transfer-planning Horizon
    # slider (that slider can legitimately be 1 GW; reusing it here meant a
    # "PLAY GW{n}" verdict was often just confirming the sole candidate, not
    # finding a genuine optimum). See chip_protocol.chip_advisor_gw_window()
    # for the sizing/DGW-BGW-extension logic (model_config.yaml
    # chip_advisor_horizon:). Only computed when at least one of BB/TC/FH is
    # still available, and re-projects bench/starters/squad onto the wider
    # window while keeping the SAME player-identity split (who's bench vs.
    # XI, who's in the squad) that the sidebar-horizon view already settled
    # on for this planning_gw.
    # chip_adv_window / chip_adv_gw_list computed earlier (Patch 43); the
    # projection itself is just the shared `proj` now (already contains every
    # GW column this window needs), so this only re-derives the bench/XI/
    # squad slices — no second compute_all() call.
    bench_df_adv, starters_df_adv, squad_df_adv, chip_adv_proj = bench_df, starters_df, squad_df, proj
    if chip_adv_window is not None and not squad_df.empty and chip_adv_gw_list != gw_list:
        bench_df_adv = chip_adv_proj[chip_adv_proj["code"].isin(bench_df["code"])]
        starters_df_adv = chip_adv_proj[chip_adv_proj["code"].isin(starters_df["code"])]
        squad_df_adv = chip_adv_proj[chip_adv_proj["code"].isin(squad_codes)]

    # Patch 54 (2026-09-17, manager report + screenshots) — chip_advisor_gw_
    # window()'s near-term scan range (planning_gw..planning_gw+~8) has no
    # awareness of which SPECIFIC calendar window (from chip_rows) is
    # actually open for a given chip. A manager who's already played, say,
    # Bench Boost 1 (window GW1-19) still has "Bench Boost" show up in
    # available_chip_names (Bench Boost 2, window GW20-38, genuinely is
    # still available) -- so the gate above was never wrong -- but the scan
    # window it fed to evaluate_bench_boost() fell entirely inside the
    # ALREADY-USED first window, so the advisor was scoring and recommending
    # a "PLAY GW9"/"HOLD" verdict over a date range where that chip literally
    # cannot be played any more. Fixed by clipping the scan window to only
    # the GWs that actually fall inside a still-"available" window for that
    # chip name; if nothing in the near-term scan overlaps any available
    # window (this manager's exact case), no verdict is computed at all --
    # the card instead shows a genuine "USED GW{n}" state (see
    # _advisor_card) with the next window's opening GW, rather than a
    # fabricated HOLD.
    #
    # Patch 55 — _clip_to_available_windows() itself now lives earlier in
    # this script (right after chip_rows is built), so the Wildcard/Free Hit
    # detect-window gate above can reuse the exact same logic instead of a
    # second, divergent copy. Still the same function, still called the same
    # way here — only its definition site moved.

    bb_advisor = None
    bb_used_state = None
    if any(c.startswith("Bench Boost") for c in available_chip_names):
        _bb_gws, _bb_last_used, _bb_next = _clip_to_available_windows(chip_adv_window["gw_list"], "Bench Boost")
        if _bb_gws:
            # Patch 99 -- routed through the cached _bb_advisor_calc().
            bb_advisor = _bb_advisor_calc(bench_df_adv, tuple(_bb_gws), cfg, "bench_boost")
        elif _bb_last_used is not None:
            bb_used_state = {"last_used_gw": _bb_last_used, "next_open_gw": _bb_next}
    tc_advisor = None
    tc_used_state = None
    if any(c.startswith("Triple Captain") for c in available_chip_names):
        _tc_gws, _tc_last_used, _tc_next = _clip_to_available_windows(chip_adv_window["gw_list"], "Triple Captain")
        if _tc_gws:
            # Patch 99 -- routed through the cached _tc_advisor_calc().
            tc_advisor = _tc_advisor_calc(starters_df_adv, tuple(_tc_gws), cfg, "triple_captain")
        elif _tc_last_used is not None:
            tc_used_state = {"last_used_gw": _tc_last_used, "next_open_gw": _tc_next}
    fh_advisor = None
    fh_used_state = None
    if any(c.startswith("Free Hit") for c in available_chip_names) and not squad_df.empty:
        _fh_gws, _fh_last_used, _fh_next = _clip_to_available_windows(chip_adv_window["gw_list"], "Free Hit")
        if _fh_gws:
            # Patch 99 -- routed through the cached _fh_advisor_calc() (the
            # single most expensive isolated advisor -- a fresh MILP rebuild
            # solve per horizon GW, per this block's own pre-existing
            # comment above).
            fh_advisor = _fh_advisor_calc(squad_df_adv, tuple(_fh_gws), cfg, chip_adv_proj, team_value, "free_hit")
        elif _fh_last_used is not None:
            fh_used_state = {"last_used_gw": _fh_last_used, "next_open_gw": _fh_next}

    # Patch 85 (v6.9 Rules #48-49: Chip Window Value + Chip Portfolio
    # Scheduling). Confirmed via code read before this patch (2026-09-29):
    # neither rule existed anywhere — the only Wildcard "what-if"
    # (evaluate_wildcard_whatif, used below for the manager-chosen-date
    # scenario tool) compares against HOLDING the squad, not the doc's
    # required "best transfer path" baseline, and nothing assigned BB/TC/FH/
    # Wildcard to distinct weeks to maximise their combined total. Runs only
    # when at least 2 chip types are still available this half — Rule #49 is
    # specifically about SEQUENCING multiple chips against each other, so a
    # single remaining chip has nothing to sequence.
    pool_df_adv = chip_adv_proj[~chip_adv_proj["code"].isin(squad_codes)] if not squad_df.empty else pool_df
    _available_chip_types = {k for k, label in chip_protocol.CHIP_LABELS.items()
                              if any(name.startswith(label) for name in available_chip_names)}
    wc_window_scan = None
    chip_portfolio = None
    if len(_available_chip_types) >= 2 and not squad_df.empty and chip_adv_window is not None:
        _portfolio_gw_list = chip_adv_window["gw_list"]
        _wc_window_len = cfg.get("chip_portfolio", {}).get("window_value_len", 4)
        _fh_gap_table = {gw: v["gap"] for gw, v in (fh_advisor or {}).get("by_gw", {}).items()}
        # Patch 96 (Rule #52a) -- the fuller {gw: {"current","rebuild","gap"}}
        # structure, needed so chip_portfolio_schedule() can recompute Free
        # Hit's "current" side against a scheduled Wildcard's rebuild squad
        # (reusing "rebuild", never a second MILP solve -- see chip_protocol.
        # _fh_post_table_for_squad()'s docstring).
        _fh_by_gw = (fh_advisor or {}).get("by_gw", {})
        # Patch 96 (Rule #52b) -- reuses the Wildcard trigger's ALREADY-
        # COMPUTED reachable-ceiling table (data_pipeline.solve_reachable_
        # ceiling_by_gw() over detect_gw_list, ~line 1880) rather than
        # solving a new one -- zero added MILP cost, per the 2026-09-30
        # performance discussion. DISCLOSED LIMITATION: detect_gw_list is a
        # ~4-GW window, while this scan (`_portfolio_gw_list`) can reach
        # 8-16 GWs -- when the guardrail's checkpoint GW (the scan's last
        # GW) falls beyond detect_gw_list's range, `reachable_by_gw` simply
        # won't have an entry for it, and _apply_squad_health_guardrail()
        # safely no-ops for that run (falls back to the unfiltered near-tie
        # set) rather than firing. Extending real coverage to the full
        # scan's checkpoint GW would need one additional bounded solve (not
        # per-candidate) -- not yet built, flagged as a fast-follow.
        # Patch 96b (2026-09-30, manager-reported ~6m45s runtime regression
        # after Patch 96's deploy) -- confirmed via code read that
        # solve_reachable_ceiling_by_gw()'s return shape is {gw: {"squad":
        # <15-row DataFrame>, "total_xpts": float, "cost": float}} (data_
        # pipeline.py lines 833-863/808-828) -- the FULL solve_squad() result,
        # not just the number _apply_squad_health_guardrail() actually reads
        # (ceiling_info.get("total_xpts"), confirmed in chip_protocol.py).
        # Passing that whole nested structure (embedded DataFrames included)
        # into a @st.cache_data-wrapped function breaks the same discipline
        # every OTHER value threaded into _chip_portfolio_calc already
        # follows -- _fh_gap_table, for exactly this reason, was already
        # reduced to a flat {gw: float} before crossing that boundary; this
        # one wasn't. Reduced here to the one number the guardrail actually
        # uses, same shape the guardrail expects ({gw: {"total_xpts": ...}})
        # so chip_protocol.py needed no change and every existing Patch 96
        # test keeps passing unmodified. Benchmarked directly (2026-09-30):
        # hashing a dict of embedded DataFrames costs ~27ms/call in this
        # sandbox, not obviously enough on its own to explain a 6m45s
        # regression -- flagged plainly to the manager as a real but likely
        # PARTIAL fix, not a confirmed full explanation, pending clarification
        # on whether the timed run covered this tab alone or the full "Run
        # Model" click across every tab (a materially different scope than
        # what Patch 96's own ~40-50s estimate covered).
        _reachable_by_gw_raw = locals().get("reachable_by_gw")
        _reachable_ceiling_for_guardrail = (
            {gw: {"total_xpts": v["total_xpts"]} for gw, v in _reachable_by_gw_raw.items()
             if v is not None and v.get("total_xpts") is not None}
            if _reachable_by_gw_raw else None)
        # Patch 91 (v6.9 Rule #49, chip-expiry correctness) -- confirmed via
        # code read (2026-09-30) that chip_portfolio_schedule() considered
        # every GW in `_portfolio_gw_list` (which can reach 8-16 GWs ahead,
        # chip_advisor_gw_window()'s own extend-to-nearest-DGW/BGW logic)
        # as a legal candidate for EVERY chip type, with no idea that a
        # chip's OWN currently-available window (chip_status()'s real
        # [start_event, stop_event] from the official chip calendar) can
        # close, or not yet be open, partway through that scan -- while the
        # ISOLATED bb_advisor/tc_advisor/fh_advisor cards just above were
        # already correctly clipped via the same _clip_to_available_windows()
        # this reuses. Since Patch 87 this joint schedule's own assignment
        # overrides those correctly-clipped isolated verdicts as the visible
        # card headline, so an unclipped joint schedule could recommend
        # playing a chip on a week it would already be expired/lost. Fixed
        # by building the same per-type clipped window every isolated card
        # already has (Wildcard's own clip is new here -- it wasn't in the
        # isolated-card block above since Wildcard has no isolated advisor
        # card of that shape) and threading it through as a hashable
        # tuple-of-pairs (st.cache_data requires hashable args).
        _valid_gws_by_type = {}
        for _ctype, _label in chip_protocol.CHIP_LABELS.items():
            if _ctype in _available_chip_types:
                _clipped_gws, _, _ = _clip_to_available_windows(_portfolio_gw_list, _label)
                _valid_gws_by_type[_ctype] = tuple(_clipped_gws)
        _valid_gws_by_type_items = tuple(sorted(_valid_gws_by_type.items()))
        # Patch 86: cached (see _chip_portfolio_calc above) -- was bare
        # top-level code, re-solving on every rerun regardless of whether
        # any of these inputs actually changed.
        wc_window_scan, chip_portfolio = _chip_portfolio_calc(
            squad_df_adv, pool_df_adv, cfg, ft["free_transfers"], bank,
            tuple(_portfolio_gw_list), _wc_window_len, tuple(sorted(_available_chip_types)),
            _fh_gap_table, _valid_gws_by_type_items,
            fh_by_gw=_fh_by_gw, reachable_ceiling_by_gw=_reachable_ceiling_for_guardrail,
            # Patch 107: extended mode scores late Wildcard candidates over a
            # FULL window (trailing GWs projected), not a truncated one.
            scan_full_gw_list=(tuple(recommend.extended_scan_full_gw_list(_portfolio_gw_list, _wc_window_len))
                               if _extended_mode else None))

    # Chip-aware transfer advisory: only from signals already computed
    # mechanically above — never a guess at the manager's intent. Wildcard:
    # only fires when its own v6.4 trigger (Patch 30) is already active AND
    # a Wildcard is currently available — both real, not invented.
    # Free Hit: only fires when chip_notes already surfaced genuine blank
    # exposure — reuses that evidence rather than re-deriving it.
    advisory_bits = []
    if wc_flag and any(r["status"] == "available" and r["chip"].startswith("Wildcard") for r in chip_rows):
        advisory_bits.append("a Wildcard review condition was triggered (see Chip Rack — rank decline / "
                             "flagged players) and the chip is unused. This does NOT mean a Wildcard is being "
                             "played now or is recommended — it's informational only, per the model's own rule "
                             "that Wildcard timing is always your call. If you're separately already planning "
                             "to play it soon, banking this transfer costs nothing since a Wildcard resets "
                             "your squad anyway")
    if any("Free Hit" in n for n in chip_notes):
        advisory_bits.append("a Free Hit has genuine exposure against a confirmed blank in your horizon "
                             "(see Chip Rack) — weigh banking against spending here too")
    # Patch 28 (v6.4 / Standing Rule #41 + Rule #24 override) — surfaced in the
    # same chip-context advisory line as the existing Wildcard/Free Hit bits.
    for note in disruption["notes"]:
        advisory_bits.append(note)
    # Patch 28 (v6.4 / Step 8c shape-test) — attached only when a shape was
    # actually classified this run (never on "insufficient_data"), so it reads
    # as a genuine cross-check on whichever chip signal above already fired,
    # not a standalone claim.
    if shape_test and shape_test["classification"] != "insufficient_data":
        advisory_bits.extend(shape_test["notes"])
    chip_advisory = f"GW{planning_gw}: Chip context — " + "; ".join(advisory_bits) + "." if advisory_bits else None

    # transfer suggestions — isolated so a bad row here can't take down the
    # rest of the page (pitch view, chip rack, captaincy, ledger all still
    # render even if this section fails). Uses `transfer_gw_list`, not the
    # sidebar's raw `gw_list` — Patch 28/Rule #41 may have capped it short of
    # a planned full-rebuild chip for a currently-disrupted squad player.
    # Patch 34 — feeds the Starting-XI Impact Check's chip-context overrides:
    # `bb_play_gw` lets a bench-only swap still count as real impact if it
    # helps a Bench Boost week specifically; `chip_capped_gw_list` (only set
    # when a manager-planned full-rebuild chip actually falls inside this
    # horizon) drives the "Chip-aware alt: Roll" comparison against just the
    # pre-rebuild window. Neither changes anything when absent/inapplicable.
    # Patch 98 (v6.9 Rule #52 follow-up, "a more wide rule" — manager,
    # 2026-09-30): confirmed via code read that `_bb_play_gw` was previously
    # sourced ONLY from the isolated Bench Boost advisor's own verdict,
    # never from `chip_portfolio`'s harmonized joint assignment — the same
    # "two disconnected sources of truth" bug class Patch 91/92/93 already
    # fixed elsewhere. recommend.resolve_bb_play_gw() now prefers the
    # harmonized joint-schedule week (chip_portfolio's own pick, which can
    # legitimately differ from Bench Boost's standalone-best week since
    # Patch 96) and falls back to the isolated advisor's verdict only when
    # no joint schedule has one.
    _isolated_bb_play_gw = int(bb_advisor["verdict"].split("gw")[1]) \
        if bb_advisor and bb_advisor["verdict"].startswith("play_gw") else None
    _harmonized_bb_gw = (chip_portfolio or {}).get("assignment", {}).get("bboost")
    _bb_play_gw = recommend.resolve_bb_play_gw(_isolated_bb_play_gw, _harmonized_bb_gw)
    # Patch 93 (v6.9 Rule #49 follow-up — same "two disconnected sources of
    # truth" bug class Patch 91/92 fixed on the chip-expiry window and the
    # chained weekly planner, this time on suggest_transfers()'s own
    # single-decision path used at horizon=1/"Force"): `_chip_capped_gw_list`
    # used to come ONLY from the sidebar's manual "Next planned full-rebuild
    # chip GW" dropdown, so a Wildcard the app's own Rule #49 joint scheduler
    # had already scheduled (`chip_portfolio`, computed above) produced no
    # "Chip-aware alt" note at all unless the manager also happened to set
    # the matching dropdown value by hand. `recommend.resolve_chip_capped_
    # gw_list()` now prefers the manual value when the manager explicitly
    # set one (an override they may have reasons the auto-scheduler doesn't
    # know), falling back to the auto-detected Wildcard GW otherwise.
    _auto_wildcard_gw = (chip_portfolio or {}).get("assignment", {}).get("wildcard")
    _chip_capped_gw_list = recommend.resolve_chip_capped_gw_list(
        transfer_gw_list, manual_planned_chip_gw=planned_chip_gw, auto_wildcard_gw=_auto_wildcard_gw)

    # Post-Patch-34 follow-up (2026-09-14 manager report on a Foden->Damsgaard
    # recommendation that made no sense on its face): a disrupted outgoing
    # player is real justification for a transfer on its own, independent of
    # whether the incoming player reaches the XI — but the model had no idea
    # WHICH squad player was flagged, so it couldn't say so. `disruption`
    # (computed above, Rule #41) is the single source of truth here too.
    _disrupted_codes = {p["code"] for p in disruption["players"]} if disruption["players"] else None

    # Post-Patch-34 follow-up — "we can get Tavernier directly instead of
    # Foden if this required": before reaching for a transfer, show what the
    # worst case already looks like using only players you own (a disruption
    # flag only partially discounts a player's projection — see
    # fpl_engine.free_lineup_fix_check()'s docstring for why this is a
    # downside-risk comparison, not a claimed free upgrade).
    free_fix = eng.free_lineup_fix_check(squad_df, _disrupted_codes, opt_col) if _disrupted_codes else \
        {"entries": []}

    # Patch 92 (v6.9 Rules #44/#48/#49 read together, manager: "let's go" on
    # wiring the Rule #49 joint chip schedule's OWN chosen Wildcard week into
    # the chained transfer planner, rather than leaving them as two
    # disconnected sources of truth) — built from data the Rule #49 scheduler
    # above (`chip_portfolio`, `wc_window_scan`) already computed this run;
    # None/absent whenever no Wildcard is actually scheduled, which
    # reproduces exact pre-Patch-92 behavior (see plan_transfer_schedule()'s
    # own docstring for the omitted-parameter contract).
    # Patch 98 (v6.9 Rule #52 follow-up, "a more wide rule") — `_chip_schedule`
    # now also carries `freehit_gw` when the joint scheduler has one, built
    # independently of whether a Wildcard is also scheduled this run (a
    # Free Hit can be scheduled with no Wildcard in the picture at all).
    # Confirmed via code read of plan_transfer_schedule()'s Patch 98 fix:
    # `freehit_gw` excludes that one GW's illusory value from every week's
    # valuation — the persisted squad never actually plays it, a real
    # temporary rebuild does. `_chip_schedule` stays None only when NEITHER
    # a Wildcard nor a Free Hit is scheduled, reproducing exact
    # pre-Patch-92 behavior for that case.
    _chip_schedule = None
    if chip_portfolio is not None and chip_portfolio.get("assignment", {}).get("wildcard") is not None \
            and wc_window_scan is not None:
        _wc_gw = chip_portfolio["assignment"]["wildcard"]
        _wc_rebuild = (wc_window_scan.get("by_gw", {}) or {}).get(_wc_gw, {}).get("rebuild_squad")
        if _wc_rebuild is not None and not _wc_rebuild.empty:
            _chip_schedule = {"wildcard_gw": _wc_gw, "wildcard_rebuild_squad": _wc_rebuild}
    _harmonized_fh_gw = (chip_portfolio or {}).get("assignment", {}).get("freehit")
    if _harmonized_fh_gw is not None:
        _chip_schedule = dict(_chip_schedule) if _chip_schedule else {}
        _chip_schedule["freehit_gw"] = _harmonized_fh_gw

    transfer_error = None
    try:
        # Patch 99 -- routed through the new @st.cache_data-wrapped
        # _suggest_transfers_calc() instead of calling recommend.
        # suggest_transfers() bare, so an unrelated-widget rerun with
        # unchanged inputs is a cache hit instead of a full re-solve.
        rec = _suggest_transfers_calc(
            squad_df, pool_df, cfg, style_name, hit_stance, ft["free_transfers"], bank, planning_gw,
            tuple(transfer_gw_list), forced_count, meaningful_bar_override, tuple(sorted(bench_df["code"])),
            chip_advisory, _bb_play_gw,
            tuple(_chip_capped_gw_list) if _chip_capped_gw_list is not None else None,
            tuple(sorted(_disrupted_codes)) if _disrupted_codes else None, _chip_schedule)
    except Exception as e:
        transfer_error = str(e)
        rec = {"moves": [], "plan": [], "summary": [], "net_gain": 0.0, "profile_used": style_name,
               "hit_cost_threshold": style_profiles.get_profile(style_name)["hit_cost_threshold"],
               "minimum_meaningful_gain_free": cfg["transfer"].get("minimum_meaningful_gain_free", 2.0),
               "margin_of_error": eng.margin_of_error_threshold(0.0, cfg),
               "hit_stance": hit_stance, "free_transfers": ft["free_transfers"],
               "weekly_plan": [], "is_weekly_schedule": False}

# Patch 46 (2026-09-15, manager report: the Wildcard trigger's own reachable-
# ceiling benchmark ignores what the "Transfer Recommendations" section is
# ALREADY telling you to do -- it's a one-shot rebuild using only today's
# banked free-transfer count (data_pipeline.solve_reachable_ceiling, line
# ~893), never the chained, week-by-week plan (recommend.plan_transfer_
# schedule) that accrues +1 FT/week like the real game does. So the trigger
# can read "gap requires a Wildcard" even when ordinary transfers, simply
# followed as recommended, would already close most or all of that gap on
# their own -- a real methodological blind spot, not a display bug.
#
# Manager confirmed (2026-09-15) the fix scope explicitly: DON'T change what
# "active"/"inactive" means (that stays the doc's plain 79%/15pt read, zero
# extra cost, computed once per run same as before) -- INSTEAD surface a
# visible cross-check showing what your own recommended plan already
# achieves, so you can see for yourself whether the trigger's gap survives
# ordinary play or not, before burning a Wildcard on it. Deliberately reuses
# data already computed this run (rec's own moves, reachable_detect's own
# ceiling squad, proj's already-merged wide columns) -- zero new MILP solves,
# zero new compute_all() calls, so this adds no measurable cost on top of
# Patch 42-45's performance work.
# Patch 82 (v6.9 amended §8c trigger, manager uploaded the v6.9 doc and asked
# what should change) — this cross-check used to compare against a single
# shared `reachable_detect["squad"]` and the retired 79% ceiling. Both are
# gone: `reachable_by_gw` (one accrued-FT reachable squad per GW, see
# data_pipeline.solve_reachable_ceiling_by_gw()) replaces the single squad,
# and the 6% scale-free gap threshold (wildcard_trigger.gap_pct_threshold)
# replaces the retired 79%/15pt dual-leg ceiling this note used to quote
# directly. Same zero-new-solves intent as before: reachable_by_gw was
# already computed for the trigger itself this run, just reused here per GW
# instead of once.
_wc_check_note = None
_rating_by_gw = {}
# Patch 108 (2026-10-04, LIVE CRASH, team 4984023 "Spurs", Wildcard already
# used: "NameError: name '_wc_extend_chip_driven' is not defined"). Everything
# the always-visible extend-button section reads is bound HERE, before the
# conditional block, so a team whose Wildcard trigger isn't evaluated (block
# skipped) can never hit an undefined name. The block below only RE-assigns
# the cross-check-specific ones when it actually runs.
_closes = None
_wc_extend_requested_flag = _extended_mode
_detect_gw_list_base = detect_gw_list[:_detect_window_size_base] if detect_gw_list else detect_gw_list
_wc_check_window = recommend.resolve_wildcard_check_gw_window(transfer_gw_list, _detect_gw_list_base)
_wc_current_max_gw = _wc_check_window[-1] if _wc_check_window else planning_gw
_wc_extend_target_gw = recommend.extended_window_target(planning_gw, _ext_span)
_chip_target_gw = recommend.extended_chip_target(planning_gw, _chip_span)  # Patch 110: furthest GW the extended check reaches
_wc_extend_chip_driven = _auto_wildcard_gw is not None and _auto_wildcard_gw > _wc_current_max_gw
_wc_extend_info = None
_wc_extended_active = False
_wc_extended_reached_target = True
_check_gws = []
if wc_flag and reachable_by_gw and not squad_df.empty:
    # Patch 94 (v6.9 Rule #49 cross-check correctness fix — discussion,
    # 2026-09-30: before extending this cross-check, confirmed via code read
    # that it reconstructed "the squad after your plan" by flattening EVERY
    # week's moves from a chained weekly plan into one list and applying
    # them all in a single pd.concat, regardless of which week each move
    # actually belonged to — already misleading for a multi-week plan, and
    # actively wrong once a scheduled Wildcard injects a ~13-player
    # wholesale rebuild into that same flat list (Patch 92). Replaced with
    # recommend.build_squad_after_by_gw() — the same correct per-GW chained
    # reconstruction the pitch navigator already uses (Patch 84's
    # `_nav_squad_after_by_gw`), now a single shared source of truth instead
    # of two. The single-decision path (horizon=1/"Force", NOT a weekly
    # schedule) never had this bug — one decision has nothing to chain — so
    # it keeps the original one-shot reconstruction unchanged.
    _moves_all = rec.get("moves") or []

    # Patch 95 (v6.9 Rule #49 cross-check extension — manager discussion,
    # 2026-09-30): the transfer-plan horizon (`transfer_gw_list`, capped by
    # the Horizon slider) and the reachable-ceiling window (`detect_gw_list`,
    # a separate, fixed ~4-GW window) are both, confirmed via code read,
    # unrelated to when the Rule #49 scheduler actually plans to play the
    # Wildcard — so this cross-check could silently stop short of the GW
    # that matters for the "is the chip worth it" decision. When a
    # scheduled Wildcard falls beyond whichever of those two windows
    # currently reaches less far, run a SEPARATE, cross-check-only extended
    # computation (capped at +8 GWs, manager-confirmed) that reaches it —
    # never touching `rec`/`reachable_by_gw` themselves, so the Transfer
    # Recommendations tab and the Wildcard trigger's own headline % stay
    # exactly as they are today.
    # Patch 101 (2026-10-01, manager: "it's 4 minutes 46 seconds now ,, i
    # need it below 2 minutes" -- after choosing, last session, to keep this
    # extension running automatically every run) -- this duplicates a full
    # plan_transfer_schedule() solve plus a full solve_reachable_ceiling_by_
    # gw() scan over +8 GWs, real measured cost. Now gated behind an
    # explicit opt-in button (st.button(key="wc_extend_run") below, inside
    # tab_chips) via recommend.wc_extend_requested() -- same button-gated
    # pattern Cross-Tool Reconciliation already uses. st.session_state's
    # value for a widget key is available from the top of the script on
    # every rerun (Streamlit restores it before executing script code), so
    # this read is safe even though the actual st.button(...) call that
    # defines "wc_extend_run" doesn't appear until later in the script, same
    # as the Reconcile button's own key is only ever referenced at its own
    # call site, not read early -- this one is read early BECAUSE the result
    # feeds the Wildcard card's tooltip, which renders before tab_chips'
    # later widgets would otherwise be defined.
    # Patch 103 (2026-10-01, manager screenshot: the extend-button's own text
    # read "based on just GW6-GW6" -- confirmed a real bug, not a rendering
    # glitch: this used to be computed independently as min(transfer_gw_list,
    # detect_gw_list), degenerating to a single GW at Horizon=1, while Patch
    # 102 had already fixed the ACTUAL cross-check calculation
    # (_wc_check_gw_source, below) to use whichever window is LONGER via
    # recommend.resolve_wildcard_check_gw_window(). The two could disagree --
    # this derives _wc_current_max_gw from that SAME resolver so the
    # displayed "normal reach" and the real one can never drift apart again.
    # Patch 107: "normal reach" is always computed from the BASE detection
    # window, even in extended mode (where detect_gw_list itself is widened),
    # so the Patch 95 chained-plan extension still has a real gap to extend.
    # (Patch 108: _detect_gw_list_base / _wc_check_window / _wc_current_max_gw
    # are now bound above the block -- see the Patch 108 note there.)
    # Patch 104 (2026-10-01, manager: "why it's grayed , it should give an
    # option for another 5 GWs beyond the one maximum used for the current
    # run" -- confirmed via AskUserQuestion the button should be an always-
    # available manual control, not conditional on a scheduled chip falling
    # beyond the normal reach). The target is now always computed -- a
    # fixed +5 GWs beyond _wc_current_max_gw, extended further only if an
    # actual scheduled Wildcard sits even beyond that.
    # Patch 107 (2026-10-04): fixed Current+9 (replaces Patch 104's "normal
    # reach + 5", which could drift away from Current+9).
    # Patch 105 (2026-10-01, manager screenshot: "already clicked but i don't
    # have a confirmation!!!!"). While building that confirmation, found that
    # the note text below (and the button's own framing further down) each
    # need to know whether THIS run's extension is chip-driven or
    # exploratory -- computed ONCE here, early, and threaded through to both
    # call sites, rather than letting a second independent copy at the
    # button site drift out of sync the way `_wc_current_max_gw` did across
    # Patch 102/103.
    _wc_extend_info = recommend.resolve_cross_check_horizon(
        planning_gw, _wc_current_max_gw, _wc_extend_target_gw, max_extension=8) \
        if recommend.wc_extend_requested(_wc_extend_requested_flag, _auto_wildcard_gw) else None
    _wc_extended_active = False
    _wc_extended_reached_target = True
    _squad_after_by_gw = None
    _wc_check_reachable_by_gw = None
    _wc_check_gw_source = None
    # Patch 111 (2026-10-04): the hit-inclusive 10-GW extension (a second full plan_transfer_schedule
    # + reachable-ceiling ladder) no longer runs -- the Wildcard decision now comes from
    # _wc_chain_compare_calc, and that extension was the main reason the Extended Check took ~5 minutes.
    _wc_extend_info = None
    # Falls through here both when no extension was needed at all, AND when
    # an attempted extension failed above (_wc_extended_active stays False
    # in both cases) — same unextended logic either way, never a crash.
    if not _wc_extended_active and rec.get("is_weekly_schedule"):
        _squad_after_by_gw = recommend.build_squad_after_by_gw(squad_df, rec.get("weekly_plan") or [], proj)
        _wc_check_reachable_by_gw = reachable_by_gw
        _wc_check_gw_source = recommend.resolve_wildcard_check_gw_window(transfer_gw_list, detect_gw_list)
    elif not _wc_extended_active and _moves_all:
        _out_codes = {m["out_code"] for m in _moves_all}
        _in_codes = {m["in_code"] for m in _moves_all}
        _squad_after_by_gw = {planning_gw: pd.concat(
            [squad_df[~squad_df["code"].isin(_out_codes)], proj[proj["code"].isin(_in_codes)]],
            ignore_index=True, sort=False)}
        _wc_check_reachable_by_gw = reachable_by_gw
        _wc_check_gw_source = recommend.resolve_wildcard_check_gw_window(transfer_gw_list, detect_gw_list)
    elif not _wc_extended_active:
        _squad_after_by_gw = {}
        _wc_check_reachable_by_gw = reachable_by_gw
        _wc_check_gw_source = recommend.resolve_wildcard_check_gw_window(transfer_gw_list, detect_gw_list)

    def _squad_as_of(gw: int) -> pd.DataFrame:
        """The squad as it would stand at gameweek `gw` if the plan were
        followed in order — the latest planned week at or before `gw`,
        exactly the same lookup rule the pitch navigator's GW stepper uses,
        so this cross-check and the navigator can never quote two different
        squads for the same GW. Falls back to today's actual squad if no
        planned week is at/before `gw` yet."""
        applicable = [g for g in _squad_after_by_gw if g is not None and g <= gw]
        return _squad_after_by_gw[max(applicable)] if applicable else squad_df

    _check_gws = [g for g in _wc_check_gw_source
                  if f"xpts_gw{g}" in proj.columns and _wc_check_reachable_by_gw.get(g) is not None
                  and _wc_check_reachable_by_gw[g].get("squad") is not None
                  and f"xpts_gw{g}" in _wc_check_reachable_by_gw[g]["squad"].columns]
    if _check_gws:
        _ratings = []
        for _g in _check_gws:
            _col = f"xpts_gw{_g}"
            _sv = opt.rating_gw_value(_squad_as_of(_g), _col, cfg)["total_realized"]
            _rv = opt.rating_gw_value(_wc_check_reachable_by_gw[_g]["squad"], _col, cfg)["total_realized"]
            _rp = eng.team_rating_pct(_sv, _rv, "")["rating_pct"]
            if _rp is not None:
                _ratings.append(_rp)
                _rating_by_gw[_g] = _rp
        if _ratings:
            _avg_after = round(sum(_ratings) / len(_ratings), 1)
            _wc_gap_threshold = cfg.get("wildcard_trigger", {}).get("gap_pct_threshold", 6.0)
            _avg_after_gap = round(100.0 - _avg_after, 1)
            _closes = _avg_after_gap < _wc_gap_threshold
            _plan_desc = (f"the {len(_moves_all)}-move plan" if _moves_all else "no transfer (this run rolls)")
            # Patch 95 — disclose whenever this note is scored over an
            # auto-extended horizon rather than your normal Horizon-slider/
            # detection window, and separately flag it if the +8 GW cap
            # still fell short of the actual scheduled Wildcard GW (Standing
            # Rule #4 — show the inputs, never silently narrow the claim).
            # Patch 105 (2026-10-01) — this used to be built inline here,
            # unconditionally claiming "to reach your scheduled Wildcard"
            # even for Patch 104's new exploratory extensions (no scheduled
            # chip in range at all) — wrong in that case. Moved to a pure,
            # directly-testable function that branches on the SAME
            # `_wc_extend_chip_driven` flag computed once, early, above.
            _wc_extend_note = recommend.build_wc_extend_note(
                _wc_extend_chip_driven, _wc_extended_reached_target, _check_gws[-1], _auto_wildcard_gw) \
                if _wc_extended_active else ""
            _wc_check_note = (
                f"Cross-check against your own recommended transfer plan ({hit_stance}, {_plan_desc}, "
                f"GW{_check_gws[0]}-GW{_check_gws[-1]}{_wc_extend_note}): if followed in full, your squad's "
                f"average Team "
                f"Rating % over that span is projected to rise to {_avg_after}% ({_avg_after_gap}% gap; currently "
                f"{wc_trigger['avg_rating_pct']}%) — "
                + (f"already below the {_wc_gap_threshold:.0f}% trigger gap, so ordinary transfers may close "
                   f"this gap on their own, without needing the Wildcard — worth checking before committing it."
                   if _closes else
                   f"still at/above the {_wc_gap_threshold:.0f}% trigger gap even after the plan, so this looks "
                   f"like a structural gap ordinary transfers alone won't close, not just a few weeks away.")
                # Patch 49 (2026-09-15, manager report: "where the points we
                # discussed" after seeing 100.0% here but 89.5% in the pitch
                # navigator's own Team Rating % for the same GW/after-plan
                # squad) — both numbers are correct, but silently used
                # different denominators with no disclosure, which is exactly
                # what Standing Rule #16 (Team Rating % Disclosure Rule)
                # exists to prevent. This one is scored against the SAME
                # reachable-ceiling squad the Wildcard trigger itself checks
                # (best squad reachable using only your currently-banked free
                # transfers) — the navigator's own Team Rating % is scored
                # against a fully unconstrained "best XI money could buy"
                # ceiling instead (Patch 20/24), a much harder bar. A high
                # number here plus a lower one there isn't a contradiction —
                # it means your plan is basically already optimal GIVEN the
                # transfers you actually have, even though a truly
                # unconstrained rebuild could still in theory do better.
                + f" (Scored against the Wildcard trigger's own reachable-transfer ceiling — a different, "
                  f"easier-to-reach bar than the pitch navigator's Team Rating %, which compares against a "
                  f"fully unconstrained optimal squad instead; the two aren't meant to match.)")
            # Patch 48 (2026-09-15, manager report: the pill only ever showed
            # "Cross-check against your own recommended transfer plan (Hit if
            # wort…" — _flag_pill truncates any text past 70 chars, and the
            # actual verdict (closes the gap or not) was the LAST clause of a
            # long sentence, so it got cut off in the one place the manager
            # actually looks (the always-visible pill row), only surviving in
            # the hover tooltip nobody saw in a screenshot. Fix: a short,
            # verdict-first headline for the pill itself (fits under 70 chars,
            # so it's never truncated) — the full reasoning above is unchanged
            # and still carried as this pill's hover tooltip and inside "Full
            # chip analysis" / the card tooltip.
            # Patch 49 — "of reachable ceiling" added to the headline itself
            # (not just the tooltip) so it never reads as the same stat as
            # the pitch navigator's own Team Rating % at a glance.
            _wc_check_headline = (
                "Wildcard not needed yet — your transfers close the gap" if _closes else
                "Wildcard looks warranted — your transfers don't close the gap")

# Patch 111 (2026-10-04): the Wildcard CHAIN COMPARISON + the single linked decision (replaces the
# Patch 110 94% bridge rule). Runs only when the 6% trigger is active. The decision object below is
# the ONLY source for the Wildcard card, the top pill and the best-GW table.
_wc_chain = None
_wc_unavail = None
_wc_decision = None
_wc_decision_seq = ((chip_portfolio or {}).get("detail", {}) or {}).get("wildcard") if wc_flag else None
_seq_detail_early = ((chip_portfolio or {}).get("detail", {}) or {}) if chip_portfolio else {}
_wc_first_week = None
if wc_flag and not squad_df.empty and shape_proj is not None:
    try:
        _eval_end = planning_gw + (_ext_span - 1 if _extended_mode else 7)
        _eval_gws = []
        for _g in range(planning_gw, _eval_end + 1):
            if f"xpts_gw{_g}" in shape_proj.columns and f"xpts_gw{_g}" in squad_df.columns:
                _eval_gws.append(_g)
            else:
                break
        if _eval_gws:
            _mv0 = []
            if rec.get("is_weekly_schedule"):
                _w0 = [w for w in (rec.get("weekly_plan") or []) if w.get("gw") == planning_gw]
                _mv0 = (_w0[0].get("moves") if _w0 else []) or []
            else:
                _mv0 = rec.get("moves") or []
            _mv0 = [m for m in _mv0[:max(0, int(ft["free_transfers"]))]
                    if m.get("out_code") is not None and m.get("in_code") is not None]
            _wc_first_week = (planning_gw, [f"{m.get('out', '?')} → {m.get('in', '?')}" for m in _mv0])
            _cand_gws = recommend.chain_candidate_gws(
                (wc_window_scan or {}).get("by_gw"), planning_gw, _eval_gws,
                (_wc_decision_seq or {}).get("gw"), max_n=3 if _extended_mode else 2)
            # Patch 113: a first-half chip lapses at its deadline, so each chip is picked inside the EARLIEST window that
            # is still available (the later set only opens afterwards). Empty tuple = no restriction (calendar missing).
            def _earliest_window(prefix):
                _w = [r["window"] for r in chip_rows if r["chip"].startswith(prefix) and r["status"] == "available"
                      and r["window"][0] is not None and r["window"][1] is not None]
                if not _w:
                    return None
                s0, e0 = min(_w, key=lambda w: w[0])
                return tuple(range(int(s0), int(e0) + 1))
            _earliest_window_items = tuple((k, v) for k, v in (("3xc", _earliest_window("Triple Captain")),
                                                               ("bboost", _earliest_window("Bench Boost"))) if v)
            _ece = cfg.get("chip_extended_check", {})
            def _chain_call(cands, bb_gw, post_cap=None):
                return _wc_chain_compare_calc(
                squad_df, pool_df, shape_proj, cfg, style_name, ft["free_transfers"], bank, planning_gw,
                tuple(_eval_gws), tuple(cands), 8, meaningful_bar_override, bb_gw,
                tuple(sorted(_disrupted_codes)) if _disrupted_codes else (), float(team_value),
                tuple((m["out_code"], m["in_code"]) for m in _mv0),
                ((_seq_detail_early.get("freehit") or {}).get("gw") if _seq_detail_early else None),
                float(cfg.get("chip_extended_check", {}).get("cap_use_bar", 0.25)),
                tuple(range(_eval_gws[-1] + 1, _eval_gws[-1] + 1 + int(cfg.get("chip_extended_check", {}).get("value_tail_gws", 3)))),
                tuple(sorted(k for k in (globals().get("_available_chip_types") or set()) if k != "wildcard")),
                bool(cfg.get("chip_extended_check", {}).get("include_chip_value", False)),
                tuple(chip_adv_gw_list or ()),
                str(cfg.get("chip_extended_check", {}).get("pick_rule", "best_horizon")),
                _earliest_window_items,
                str(cfg.get("chip_extended_check", {}).get("wc_objective", "weekly_xi")),
                bool(_extended_mode and cfg.get("chip_extended_check", {}).get("wc_chip_aware_variant", False)),
                bool(cfg.get("chip_extended_check", {}).get("tc_ft_route", True)),
                int(post_cap if post_cap is not None else _ece.get("post_chip_weeks", 6)),
                float(_ece.get("post_chip_decay", 0.9)),
                str(_ece.get("tc_week_source", "schedule")),
                ((_seq_detail_early.get("3xc") or {}).get("gw") if _seq_detail_early else None))
            _chain_bb_gw_used = _bb_play_gw
            _wc_chain = _chain_call(_cand_gws, _bb_play_gw)
            if _wc_chain is not None and not _wc_chain.get("cands"):
                _wc_unavail = _wc_chain.get("unavailable_reason") or "the chain comparison produced no candidate week"
                _wc_chain = None
    except Exception as _e_chain:
        _wc_chain = None
        _wc_unavail = f"the chain comparison failed ({type(_e_chain).__name__}: {str(_e_chain)[:160]})"
if wc_flag and wc_trigger and wc_trigger.get("avg_gap_pct") is not None:
    # Patch 115 (model v6.12 rulings 2, 3, 6): band = Rule #34 over the compared window total, LATER week inside it, the
    # Rule #52 squad-health guardrail on the in-band weeks, the four-GW value as a cross-check, a later shift vs the
    # previous logged run flagged as a Rule #11 reversal.
    _cal_path0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildcard_calibration_log.csv")
    _moe_f = lambda total: eng.margin_of_error_threshold(total, cfg)
    _prev_gw = recommend.last_logged_wc_gw(_cal_path0, entry_id)
    _gains0 = ({t: c["gain"] for t, c in _wc_chain["cands"].items()} if _wc_chain else None)
    _kw = dict(margin=eng.margin_of_error_threshold(0.0, cfg), floor=float(meaningful_bar_override),
               today_pct=wc_trigger.get("avg_rating_pct"), gap_pct=wc_trigger["avg_gap_pct"],
               seq_gw=(_wc_decision_seq or {}).get("gw"), seq_value=(_wc_decision_seq or {}).get("value"),
               moe_fn=_moe_f, prev_gw=_prev_gw, unavailable_reason=_wc_unavail,
               wait_cap_frac=float(cfg.get("chip_extended_check", {}).get("wc_wait_cap_frac", 0.5)),
               waiting_used=recommend.waiting_budget_used(_cal_path0, entry_id, planning_gw),
               provisional="re-run after the team news and press conferences; pending news is not a reason to defer the chip (Rule #48(c))")
    if _wc_chain:
        _kw.update(gains4=({t: c.get("gain_decide") for t, c in _wc_chain["cands"].items()}
                           if int(cfg.get("chip_extended_check", {}).get("wc_decide_weeks", 4) or 0) else None),
                   checkpoint_gw=_wc_chain.get("checkpoint_gw"))
        _kw.update(window_totals={t: c.get("window_total") for t, c in _wc_chain["cands"].items()},
                   band_totals={t: c.get("band_total") for t, c in _wc_chain["cands"].items()},
                   extra_lines=sorted({c.get("build_note") for c in _wc_chain["cands"].values() if c.get("build_note")}),
                   not_scored=_wc_chain.get("not_scored") or [])
        if bool(cfg.get("chip_extended_check", {}).get("chain_guardrail", True)):
            _kw.update(health={t: c.get("checkpoint") for t, c in _wc_chain["cands"].items()},
                       base_health=_wc_chain["base"].get("checkpoint"), health_band=_wc_chain.get("health_band"))
    _wc_decision = recommend.wildcard_chain_decision(True, _gains0, planning_gw, **_kw)
    if _wc_chain and _wc_decision:
        try:
            _ns4 = set(_wc_chain.get("not_scored") or [])
            _by4 = {t: v.get("gap") for t, v in ((wc_window_scan or {}).get("by_gw") or {}).items() if t not in _ns4}
            _xc = recommend.four_gw_crosscheck(_by4, _wc_decision.get("gw"), moe_fn=_moe_f)
            _dwk = _wc_chain["cands"].get(_wc_decision.get("gw"), {})
            _wk = ([g for g in _wc_chain["gws"] if g >= _wc_decision["gw"]][:int(_wc_chain.get("post_weeks") or 0)]
                   if _wc_decision.get("gw") is not None else [])
            _fb = recommend.fallback_weeks(_wk, planning_gw, int(cfg.get("chip_extended_check", {}).get("market_coverage_gws", 5)))
            _kw.update(crosscheck=_xc, fallback=_fb)
            _wc_decision = recommend.wildcard_chain_decision(True, _gains0, planning_gw, **_kw)
        except Exception:
            pass
    # Patch 115 fix 2: decay sensitivity -- same solved squads, alternative decays; a flip = low confidence (display only)
    if _wc_chain and _wc_decision:
        try:
            _dk = ([] if int(cfg.get("chip_extended_check", {}).get("wc_decide_weeks", 4) or 0) else
                   [float(x) for x in (cfg.get("chip_extended_check", {}).get("decay_check") or [])])   # Patch 117: the 0.9 decay no longer decides
            _sens = recommend.decay_sensitivity(
                _wc_chain["cands"], _wc_chain["base"]["score"], list(_wc_chain["gws"]), int(_wc_chain.get("post_weeks") or 0),
                _wc_chain.get("fh_gw"), _dk, lambda g_: recommend.wildcard_chain_decision(True, g_, planning_gw, **_kw)) if _dk else None
            if _sens:
                _wc_decision["decay_check"] = _sens
                _wc_decision.setdefault("detail_lines", []).append(_sens["text"])
        except Exception:
            pass
# Patch 112 (ONE linked decision): Triple Captain / Bench Boost / Free Hit are picked on the squads the chain actually
# fields each week (plain Wildcard rebuild + later transfers), so the cards, headline, table, expanders, pitch navigator
# and the Transfer page all read the same weeks. Fallback (chain chips unavailable): the Rule #49 re-pick on the
# Wildcard candidate's own 4-GW rebuild.
_chips_repicked = None
_chain_path = None
try:
    if _wc_chain and chip_portfolio is not None:
        _dw0 = (_wc_decision or {}).get("gw")
        _chain_path = _wc_chain["cands"][_dw0] if _dw0 in _wc_chain["cands"] else (_wc_chain.get("base_tc") or _wc_chain["base"])
        # Patch 112 (replan_for_bb): the transfers must PREPARE for the Bench Boost week the plan finally picks. The first
        # pass planned for the joint sequence's Bench Boost week (bb_gw_used); if the plan's own pick differs, re-plan the
        # decided Wildcard path once with the new week (one extra pass, not iterated) and use that path.
        _bbp = (_chain_path.get("chips") or {}).get("bb")
        if (bool(cfg.get("chip_extended_check", {}).get("replan_for_bb", True)) and _bbp and _dw0 in _wc_chain["cands"]
                and _bbp[0] != _chain_bb_gw_used):
            _re2 = _chain_call((_dw0,), int(_bbp[0]), post_cap=int(_wc_chain.get("post_weeks") or 0) or None)
            if _re2 and _dw0 in _re2["cands"]:
                _wc_chain["cands"][_dw0] = _re2["cands"][_dw0]
                _chain_path = _wc_chain["cands"][_dw0]
                _chain_path["replanned_for_bb"] = int(_bbp[0])
        if _chain_path.get("chips"):
            _old_d = (chip_portfolio.get("detail", {}) or {})
            _new_d, _chg = recommend.override_chip_detail(_old_d, _chain_path["chips"], floor=float(meaningful_bar_override))
            chip_portfolio = {**chip_portfolio, "detail": _new_d}
            _chips_repicked = {"old": {k: v.get("gw") for k, v in _old_d.items() if k != "wildcard"},
                               "new": {k: v.get("gw") for k, v in _new_d.items() if k != "wildcard"},
                               "changed": bool(_chg), "src": "chain"}
        else:
            _chain_path = None
except Exception:
    _chain_path, _chips_repicked = None, None
try:
    if _chips_repicked is None:
        _dw = (_wc_decision or {}).get("gw")
        _seq_wc = (((chip_portfolio or {}).get("detail", {}) or {}).get("wildcard") or {}).get("gw")
        _by = (wc_window_scan or {}).get("by_gw") or {}
        if wc_flag and _dw is not None and _dw != _seq_wc and _dw in _by:
            _re = _chip_portfolio_fixed_wc_calc(
                {"by_gw": {_dw: _by[_dw]}}, _fh_gap_table, squad_df_adv, tuple(_portfolio_gw_list), cfg,
                tuple(sorted(_available_chip_types)), _valid_gws_by_type_items, _fh_by_gw,
                _reachable_ceiling_for_guardrail, int(_dw))
            if _re and (_re.get("detail") or {}).get("wildcard", {}).get("gw") == _dw:
                _old = {k: v.get("gw") for k, v in (chip_portfolio or {}).get("detail", {}).items() if k != "wildcard"}
                _new = {k: v.get("gw") for k, v in _re["detail"].items() if k != "wildcard"}
                chip_portfolio = _re
                _chips_repicked = {"old": _old, "new": _new, "changed": _old != _new, "src": "rule49"}
except Exception:
    pass


# Patch 117d: ONE chained plan -- the weekly transfer plan follows the Wildcard decision. The first plan was built before the
# decision existed (with the scan's Wildcard week); rebuild it once on the decided week (Wildcard now = no-Wildcard list).
try:
    _dw_final = (_wc_decision or {}).get("gw") if _wc_decision else None
    _by_rb = {g: (v or {}).get("rebuild_squad") for g, v in ((wc_window_scan or {}).get("by_gw") or {}).items()}
    _cs2 = recommend.reconcile_chip_schedule(_chip_schedule, _dw_final, planning_gw, _by_rb)
    if (_cs2 or {}).get("wildcard_gw") != (_chip_schedule or {}).get("wildcard_gw"):
        _cs2 = _cs2 if _cs2 else None
        rec = _suggest_transfers_calc(
            squad_df, pool_df, cfg, style_name, hit_stance, ft["free_transfers"], bank, planning_gw,
            tuple(transfer_gw_list), forced_count, meaningful_bar_override, tuple(sorted(bench_df["code"])),
            chip_advisory, _bb_play_gw,
            tuple(_chip_capped_gw_list) if _chip_capped_gw_list is not None else None,
            tuple(sorted(_disrupted_codes)) if _disrupted_codes else None, _cs2)
        _chip_schedule = _cs2
except Exception:
    pass


def _edge_txt(chosen, best, gap, what, conf="near-tie", yielded_to=None, tie_set=None):
    """Patch 113: honest edge sentence from recommend.chip_edge_text (one source, no stray punctuation).
    Patch 115: the tie set (weeks inside Rule #34's band of the best) is named when the pick is low confidence."""
    return recommend.chip_edge_text(chosen, best, gap, conf, what, yielded_to=yielded_to, tie_set=tie_set)

def _tbl_h(n_rows):
    return int(35 * (n_rows + 1) + 3)


def _path_squad_at(gw):
    """Patch 112: the squad the Chip Plan fields in `gw` (None when no chain path / outside its span)."""
    try:
        codes = (_chain_path or {}).get("squads", {}).get(gw)
        if not codes:
            return None
        sq = chip_adv_proj[chip_adv_proj["code"].isin(codes)]
        if len(sq) != len(codes):
            sq = shape_proj[shape_proj["code"].isin(codes)] if shape_proj is not None else sq
        return sq if f"xpts_gw{gw}" in sq.columns else None
    except Exception:
        return None


# Patch 112: Captaincy follows the plan. When the Chip Plan says play the Wildcard THIS GW, the armband is picked from the
# Wildcard squad's GW XI (same style/tie-break mechanics as the current-squad pick), not from the squad being replaced.
cap_plan = None
try:
    if _chain_path and (_wc_decision or {}).get("gw") == planning_gw:
        _sq0 = _path_squad_at(planning_gw)
        _col0 = f"xpts_gw{planning_gw}"
        _xi0 = opt.best_starting_xi(_sq0, _col0) if (_sq0 is not None and len(_sq0) >= 15) else None
        if _xi0 and _xi0.get("xi") is not None and not _xi0["xi"].empty:
            _need0 = ["code", "web_name", "team", _col0, "selected_by_percent"]
            if all(c in _xi0["xi"].columns for c in _need0):
                _cands0 = _xi0["xi"].rename(columns={_col0: "xpts_this_gw"})[
                    ["code", "web_name", "team", "xpts_this_gw", "selected_by_percent"]]
                _res0 = eng.captaincy_protocol(_cands0, cfg)
                _tt0 = eng.team_league_table(snap.fixtures, snap.teams)
                _pk0 = style_profiles.captaincy_pick(_res0, style_name, team_table=_tt0, cfg=cfg)
                _alt0, _lab0 = style_profiles.captain_alt_pick(_res0, _pk0["web_name"], style_name)
                _n0 = int(_res0["shortlisted"].sum()) if "shortlisted" in _res0.columns else 1
                _xp0 = _pk0["xpts_this_gw"]
                if _n0 <= 1:
                    _cap0 = f"🎯 Armband: <b>{_pk0['web_name']}</b> — the standout pick on the Wildcard squad ({_xp0:.1f} xPts, clear of the field)."
                elif _alt0 is not None and not _lab0.startswith("Near miss"):
                    _cap0 = (f"🎯 Armband: <b>{_pk0['web_name']}</b> — a coin-flip with {_alt0['web_name']} on the Wildcard squad "
                             f"({_xp0:.1f} xPts); {_lab0.lower()} given per your style profile (<b>{style_name}</b>).")
                else:
                    _cap0 = f"🎯 Armband: <b>{_pk0['web_name']}</b> — a coin-flip within the shortlist on the Wildcard squad ({_xp0:.1f} xPts)."
                cap_plan = {"row": _pk0, "alt": _alt0, "alt_label": _lab0, "caption": _cap0, "xp": _xp0}
except Exception:
    cap_plan = None


_final_chip_gw = {}
try:
    for _k, _v in (((chip_portfolio or {}).get("detail", {}) or {}).items()):
        if _k != "wildcard" and _v.get("gw") is not None:
            _final_chip_gw[_k] = int(_v["gw"])
except Exception:
    _final_chip_gw = {}
_chip_headline_text = None
try:
    if chip_portfolio is not None:
        _cb0, _ = recommend.final_chips_by_gw(
            dict((chip_portfolio.get("detail") or {})), chip_protocol.CHIP_LABELS, _wc_decision,
            "wildcard" in (globals().get("_available_chip_types") or set()))
        _chip_headline_text = recommend.chip_plan_headline(
            _cb0, _wc_decision, tc_player=((_chain_path or {}).get("chips") or {}).get("tc_player"))
except Exception:
    _chip_headline_text = None
if _wc_decision is not None:
    # the top pill now states the same decision as the card (replaces the
    # old "transfers close the gap" headline, which used a hit-inclusive plan).
    _wc_check_headline = _wc_decision["pill"]
    _wc_check_note = ((_wc_decision["pill"] + ". The decision compares the Wildcard chain with the no-chip chain (free transfers only). "
                       + ("Older hit-inclusive plan cross-check, for reference: " + _wc_check_note) if _wc_check_note else "")
                      or _wc_decision["pill"])

# ---------------------------------------------------------------------------
# Patch 66 — top-tab section navigation, matching the project's 5 standing
# output sections (manager confirmation via AskUserQuestion: "Match the
# project's 5 output sections"). st.tabs() renders every tab's body on each
# script run regardless of which tab is visually active, so each `with tabX:`
# block below simply routes already-existing, unmoved code into its tab's
# container — no compute-order dependencies were changed by this refactor.
# Patch 68 (manager, immediately after seeing Patch 66 live: "The pitch
# should be the first tab and by default appears") — Streamlit's st.tabs()
# always opens on whichever tab is listed FIRST, with no separate "default
# tab" setting, so making the pitch the default view means making it the
# first tab. Added as its own tab (⚽ Pitch) ahead of Latest News, and the
# Pitch Navigator's compute+render calls (previously inside Transfer
# Recommendations, Patch 66) now route into this tab instead — see
# `with tab_pitch:` below, where `_compute_scenario_evaluations()` /
# `_render_pitch_navigator()` are actually called.
# ---------------------------------------------------------------------------
tab_pitch, tab_news, tab_transfers, tab_captain, tab_chips, tab_style = st.tabs(
    ["⚽ Pitch", "🗞️ Latest News", "🔄 Transfer Recommendations", "🎯 Captaincy Pick",
     "🗺️ Chip Plan", "🧠 Manager Style Fit"])

with tab_captain:
    # Release 2's captaincy spotlight card, extracted out of the Pitch
    # Navigator's @st.fragment (where it lived through Patches 61-65) into
    # its own standalone tab. It previously only rendered when the navigator
    # happened to be scrolled to planning_gw ("at_planning_gw") — that was
    # only ever a side effect of where the navigator's own cursor sat, never
    # a real dependency, since captaincy is computed once, for planning_gw
    # only. The extraction drops that condition entirely; this tab now has
    # no dependency on the navigator or its fragment state at all.
    _c_row, _c_alt, _c_cap, _c_xp = ((cap_plan["row"], cap_plan["alt"], cap_plan["caption"], cap_plan["xp"]) if cap_plan
                                      else (cap_pick_row, cap_alt_row, cap_caption, cap_xp if cap_pick_row is not None else None))
    if cap_plan:
        _ni("🃏 Wildcard this GW — armband from WC squad", "🃏 Chip Plan: play the Wildcard this GW — the armband below is picked from the **Wildcard squad**. "
                + (f"With your current squad it would be {cap_pick_row['web_name']} ({cap_xp:.1f} xPts)." if cap_pick_row is not None else ""))
    if _c_cap and _c_row is not None:
        _cap_alt_xp = _c_alt.get("xpts_this_gw") if _c_alt is not None else None
        if _cap_alt_xp and _cap_alt_xp > 0:
            _cap_bar_pct = max(40, min(100, round(_c_xp / _cap_alt_xp * 100)))
            _cap_bar_note = "xPts vs. nearest alternative"
        else:
            _cap_bar_pct = 100
            _cap_bar_note = "xPts this week"
        _cap_eo = _c_row.get("selected_by_percent")
        _cap_eo_txt = f"{_cap_eo:.1f}% EO" if _cap_eo is not None and not pd.isna(_cap_eo) else "EO unavailable"
        _cap_photo = _photo_url(_c_row.get("code", 0))
        _cap_initials = "".join([w[0] for w in str(_c_row.get("web_name", "??")).split()][:2]).upper() or "??"
        st.markdown(
            '<div class="cap-spotlight"><div class="row">'
            '<div class="ring"><img src="' + _cap_photo + '" '
            'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">'
            '<div class="avatar-fallback">' + _cap_initials + '</div></div>'
            '<div><span class="tag">Captaincy</span>'
            '<div class="name">' + str(_c_row.get('web_name', '')) + '</div>'
            '<div class="meta">' + str(_c_row.get('team', '')) + '</div></div></div>'
            '<div class="xp-row"><div class="xp">' + f"{_c_xp:.1f}" + '</div>'
            '<div class="xp-u">xPts · captain (doubled)</div></div>'
            '<div class="bar-track"><div class="bar-fill" style="width:' + str(_cap_bar_pct) + '%"></div></div>'
            '<div class="eo-row"><span>' + _cap_eo_txt + '</span><span>' + _cap_bar_note + '</span></div>'
            '<div class="alt-sub">' + _c_cap + '</div>'
            '</div>', unsafe_allow_html=True)
    else:
        _ni("No captain pick yet", "No captaincy pick yet this run — check back once the model has computed your starting XI.")

    # Patch 113 (manager: the Captaincy tab must mention the current squad AND the pick after the recommended chip):
    # one small table -- armband now (current squad), after the Wildcard, and the Triple Captain week. All three read the
    # same plan as the Chip Plan (path squads), top scorer of each week's best XI (the armband protocol itself only runs
    # for the planning GW).
    try:
        _now_row = ({"name": cap_pick_row["web_name"], "xpts": float(cap_xp)} if (cap_pick_row is not None and cap_xp is not None) else None)
        _wcg_c = (_wc_decision or {}).get("gw")
        _wc_top_c = None
        if _wcg_c is not None:
            _wsq = _path_squad_at(_wcg_c)
            _wc_top_c = recommend.top_scorer(_wsq, f"xpts_gw{_wcg_c}") if _wsq is not None else None
        _tcg_c = _final_chip_gw.get("3xc")
        _tc_top_c = None
        if _tcg_c is not None:
            _tp_c = ((_chain_path or {}).get("chips") or {}).get("tc_player")
            if _tp_c and _tp_c.get("gw") == _tcg_c:
                _tc_top_c = {"name": _tp_c["name"], "xpts": _tp_c["xpts"]}
            else:
                _tsq = _path_squad_at(_tcg_c)
                _tc_top_c = recommend.top_scorer(_tsq, f"xpts_gw{_tcg_c}") if _tsq is not None else None
        _cap_rows = recommend.chip_captain_rows(_now_row, planning_gw, _wcg_c, _wc_top_c, _tcg_c, _tc_top_c)
        if len(_cap_rows) > 1 or (_cap_rows and _cap_rows[0]["kind"] != "now"):
            st.markdown("**Captain by chip plan**")
            _tbl = pd.DataFrame([{"When": r["label"], "GW": f"GW{r['gw']}", "Captain": r["name"],
                                  "xPts": (f"{r['xpts']:.1f}" if r["kind"] != "triple_captain"
                                           else f"{r['xpts']:.1f} x3 = {r['total']:.1f}")} for r in _cap_rows])
            _df(_tbl, hide_index=True, use_container_width=True, height=_tbl_h(len(_tbl)))
            _nc("Same plan as Chip Plan tab", "Same plan as the Chip Plan tab. The armband protocol runs for this GW only; later rows show each "
                       "week's top scorer on the squad the plan fields (current squad = what you hold today).")
    except Exception:
        pass

with tab_news:
    # ---------------------------------------------------------------------------
    # Header + verdict
    # ---------------------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f'<div class="verdict-card"><span class="phase-tag">GW{planning_gw}</span>'
                    f'<p class="h">{verdict["headline"]}</p>'
                    f'<p class="b">{verdict["body"]}</p></div>', unsafe_allow_html=True)
    with col2:
        trend = ""
        if len(rank_history_display) >= 2:
            trend = '<span class="trend-up">▲</span>' if rank_history_display[-1] < rank_history_display[-2] \
                else ('<span class="trend-down">▼</span>' if rank_history_display[-1] > rank_history_display[-2] else "")
        rank_disp = f"{rank_history_display[-1]:,}" if rank_history_display else "—"
        # Patch 11 — Standing Rule #4 disclosure: `points_total`/`rank_history[-1]`
        # come straight from the official API's `entry/` and `entry/.../history/`
        # endpoints for GW{squad_gw}. Those are real, live numbers, not stale
        # placeholders — but until FPL itself sets `data_checked=True` on that
        # gameweek (bonus points manually confirmed, scores locked for good),
        # they are PROVISIONAL and can still move, same as on the official site/
        # app during that exact window. Silently showing them as if final is
        # what actually produced the "not up to date" complaint: the numbers
        # were correct-as-of-the-fetch, just not yet the final word from FPL.
        gw_final = getattr(snap, "current_gw_data_checked", False)
        prov_badge = ("" if gw_final else
                      ' <span class="info-dot" title="GW' + str(squad_gw) + ' points/bonus not yet finalized by FPL — '
                      'this number can still move (same as the official site right now).">prov.</span>')
        _rp = fh_auto_rating['rating_pct']
        if _rp is not None:
            _gcol = "var(--accent-strong)" if _rp >= 79 else ("var(--gold)" if _rp >= 65 else "var(--coral)")
            _gauge_html = (f'<div class="gauge-wrap"><div class="gauge-ring" title="{fh_auto_tooltip}" '
                           f'style="background:conic-gradient({_gcol} {_rp*3.6:.0f}deg, var(--rule) 0deg);">'
                           f'<span class="gauge-val">{_rp:.0f}%</span></div></div>')
        else:
            # Patch 72 — was a hardcoded generic "Not enough data this run"
            # with no way to see why; now reuses fh_auto_tooltip, which
            # already carries the Patch 72 diagnostic appended above when
            # fh_auto_result is None.
            _gauge_html = f'<span class="info-dot" title="{fh_auto_tooltip}">—</span>'

        # Patch 51 (2026-09-16, §1a compliance fix) — second, visually distinct
        # badge for the doc-compliant multi-GW "Team Rating %" (compliant_rating,
        # computed above), placed right next to "Quick Team Rating" so neither
        # one can be mistaken for a stray, unlabeled second number. Its tooltip
        # cross-references the quick badge by name (Patch 49 disclosure pattern).
        _compliant_tooltip = (
            f"Team Rating % (GW{compliant_gw_start}-{compliant_gw_end}, full pool) — the model doc's §1a-compliant "
            f"metric: Squad_xPts / Ceiling_xPts over a FIXED {len(compliant_gw_list)}-GW horizon (GW{compliant_gw_start}"
            f"-GW{compliant_gw_end}, independent of the sidebar horizon slider), where Ceiling_xPts is a full-player-"
            f"pool, unconstrained, complete 15-man £100m squad solve (§1a requirements (a) and (b) both satisfied). "
            f"Both sides use the same captain-doubled, bench-autosub-discounted best-XI-per-GW calculation as the "
            f"'{fh_auto_label}' badge above. Squad: {compliant_squad_total:.1f} xPts · Ceiling: "
            f"{compliant_ceiling_total:.1f} xPts · gap: {compliant_gap:.1f} xPts (margin-of-error threshold: "
            f"{compliant_moe:.1f} xPts). NOTE: this is a DIFFERENT, wider-horizon metric than '{fh_auto_label}' "
            f"(single-GW, GW{planning_gw} only) above — the two are not meant to match; see 'Team Rating % "
            f"(GW{compliant_gw_start}-{compliant_gw_end}) — full breakdown' below for the full disclosure."
        )
        _crp = compliant_rating['rating_pct']
        # Patch 72 (manager, 2026-09-27: this badge started showing "—" after
        # Patch 71 — see the fh_auto_tooltip note above for the same pattern).
        # compliant_ceiling_total is 0.0 exactly when theoretical_ceiling
        # (data_pipeline.solve_ceiling(), label="theoretical_ceiling") came
        # back None — surface the real reason here too.
        if _crp is None:
            _ceiling_diag = opt.get_diagnostic("theoretical_ceiling")
            _compliant_tooltip += (f" ⚠ Diagnostic (Patch 72): this run's ceiling solve returned no result — "
                                    f"{_ceiling_diag}" if _ceiling_diag else
                                    " ⚠ Diagnostic (Patch 72): this run's ceiling solve returned no result, but no "
                                    "specific reason was recorded — please report this exact combination so it can "
                                    "be added.")
        if _crp is not None:
            _ccol = "var(--accent-strong)" if _crp >= 79 else ("var(--gold)" if _crp >= 65 else "var(--coral)")
            _compliant_gauge_html = (f'<div class="gauge-wrap"><div class="gauge-ring" title="{_compliant_tooltip}" '
                           f'style="background:conic-gradient({_ccol} {_crp*3.6:.0f}deg, var(--rule) 0deg);">'
                           f'<span class="gauge-val">{_crp:.0f}%</span></div></div>')
        else:
            _compliant_gauge_html = f'<span class="info-dot" title="{_compliant_tooltip}">—</span>'

        st.markdown(f"""<div class="stat-row">
          <div class="stat"><div class="n">{rank_disp} {trend}{prov_badge}</div><div class="l">Overall rank</div></div>
          <div class="stat rating">
            <div class="n">{_gauge_html}</div>
            <div class="l">{fh_auto_label} <span class="info-dot" title="{fh_auto_tooltip}">ⓘ</span></div>
          </div>
          <div class="stat rating">
            <div class="n">{_compliant_gauge_html}</div>
            <div class="l">{compliant_gw_end - compliant_gw_start + 1}-GW Rating <span class="info-dot" title="{_compliant_tooltip}">ⓘ</span></div>
          </div>
          <div class="stat new"><div class="n">{gw_xpts_total:.1f}</div><div class="l">GW{planning_gw} xPts</div></div>
          <div class="stat"><div class="n">{points_total if points_total is not None else '—'}{prov_badge}</div><div class="l">Season points</div></div>
        </div>""", unsafe_allow_html=True)
        if not gw_final:
            st.caption(f"⏳ GW{squad_gw} rank & points are still provisional",
                       help=f"GW{squad_gw} rank & points above are FPL's live provisional numbers — bonus points "
                            f"haven't been finalized yet, so both can still shift (this matches the official app/site "
                            f"during this same window, it isn't a bug in this tool). Use **Refresh live data** in the "
                            f"sidebar to re-pull the latest provisional figures.")
        # Patch 74 (manager, 2026-09-27: after Patch 73, still "same issue" —
        # the manager works from screenshots, which can never show a hover
        # tooltip. Patch 72/73's diagnostics were only ever placed in
        # fh_auto_tooltip/_compliant_tooltip, which only ever reach the page
        # as a `title` attribute on the "—" info-dot spans above — invisible
        # in any screenshot no matter how correct the text was. This was
        # never a second logic bug in the diagnostic itself; the diagnostic
        # text was always right, it just never had a visible home. Fixed by
        # printing it as an ordinary, always-visible st.caption() line right
        # here — the exact same widget already used one line up for the
        # "still provisional" note — instead of only inside a hover title.
        _fh_visible_caption = None
        if _rp is None:
            _fh_diag = opt.get_diagnostic("free_hit_optimal")
            _fh_visible_caption = (f"⚠ {fh_auto_label}: no result this run — "
                                   + (_fh_diag if _fh_diag else "no specific reason was recorded — please report "
                                                                 "this exact combination so it can be added."))
        if _fh_visible_caption:
            _nw("Rating: no result", _fh_visible_caption)

        _ceiling_visible_caption = None
        if _crp is None:
            _ceiling_diag_visible = opt.get_diagnostic("theoretical_ceiling")
            _ceiling_visible_caption = (
                f"⚠ Team Rating % (GW{compliant_gw_start}-{compliant_gw_end}): no result this run — "
                + (_ceiling_diag_visible if _ceiling_diag_visible else
                   "no specific reason was recorded — please report this exact combination so it can be added."))
        if _ceiling_visible_caption:
            _nw("Team Rating: no result", _ceiling_visible_caption)

    gw_status = "confirmed final" if getattr(snap, "current_gw_data_checked", False) else "provisional, not yet finalized"

    # Patch 47 (2026-09-15, manager request: "show the data retrieval timestamp
    # to make sure about the numbers we are seeing and make it user friendly")
    # — replaces the old bare "fetched 11:34" (no date, no timezone, easy to
    # misread as your own local time when this app can run on a server in a
    # different one, and gives no sense of whether that fetch is fresh or long
    # stale) with an explicit-UTC date+time, a live "how long ago" readout
    # computed at render time (not cached, so it's accurate even if you've had
    # the page open a while), and a color-coded freshness badge matching the
    # EXACT 15-minute window `_load_data`'s own st.cache_data(ttl=900) actually
    # uses (Patch reference: the "Refresh live data now" button that clears that
    # same cache) — never a made-up threshold. Green < 5 min, gold 5-15 min
    # (still the SAME cached fetch, just older), coral >= 15 min (that cache
    # entry has actually expired — the next "Run Model"/page action re-fetches
    # automatically, but if you're staring at numbers from well past that mark,
    # the badge says so instead of leaving you to guess).
    _fetch_dt_utc = dt.datetime.fromtimestamp(snap.fetched_at, tz=dt.timezone.utc)
    _age_s = max(0.0, dt.datetime.now(dt.timezone.utc).timestamp() - snap.fetched_at)
    if _age_s < 60:
        _age_txt, _fresh_cls = "just now", "play"
    elif _age_s < 300:
        _age_txt, _fresh_cls = f"{int(_age_s // 60)} min ago", "play"
    elif _age_s < 900:
        _age_txt, _fresh_cls = f"{int(_age_s // 60)} min ago", "active"
    else:
        _age_txt, _fresh_cls = f"{int(_age_s // 60)} min ago — cache expired, will refetch on next run", "caution"
    _fetch_badge = (f'<span class="badge {_fresh_cls}" title="Live official FPL data cached for up to 15 minutes '
                    f'(_load_data\'s own cache window) — exactly matching what the ↑Refresh live data now button '
                    f'in the sidebar clears. This badge is computed fresh every time the page renders, so it always '
                    f'reflects how old the underlying fetch actually is, even if you\'ve had this tab open a while.">'
                    f'{_age_txt}</span>')
    st.markdown(f'<div class="side-note">Data as of <b>{_fetch_dt_utc.strftime("%b %d, %H:%M:%S UTC")}</b> '
                f'{_fetch_badge} · Source: {snap.source} · squad as of GW{squad_gw} ({gw_status}) · '
                f'planning for GW{planning_gw} · '
                f'style profile: <b>{style_name}</b></div>', unsafe_allow_html=True)
    fh_at_ceiling = fh_auto_optimal_val > 0 and fh_auto_gap < fh_auto_moe
    if fh_auto_rating["rating_pct"] is not None:
        st.markdown(recommend.rating_bar_html(fh_auto_rating["rating_pct"], f"GW{planning_gw} rating"),
                    unsafe_allow_html=True)
        if fh_at_ceiling:
            st.caption("✓ Already at this GW's optimal",
                       help=f"The {fh_auto_gap:.1f} xPts gap is inside normal weekly noise (threshold "
                            f"{fh_auto_moe:.1f} xPts), not real room left on the table.")
        with st.expander(f"{fh_auto_label} — single-GW, EST — full breakdown"):
            st.markdown(tier_label)
            _nc("Current vs optimal xPts", f"GW{planning_gw} xPts (your current squad's best XI, captain doubled, bench autosub-discounted): "
                       f"{fh_auto_current_val:.1f} · GW{planning_gw} optimal (a genuinely unconstrained best-possible "
                       f"squad from the full player pool, £{team_value}m proxy budget, no free-transfer limit): "
                       f"{fh_auto_optimal_val:.1f} · gap: {fh_auto_gap:.1f} xPts (margin-of-error threshold: "
                       f"{fh_auto_moe:.1f} xPts)")
            _nc("Rating method", f"Patch 24 methodology (2026-09-08): {fh_auto_label} = current squad GW xPts / GW-optimal xPts, "
                       "both sides using the same best-legal-Starting-XI-plus-captain-bonus-plus-Rule-#12-bench-value "
                       "calculation (Rule #22 Systematic Application) — replacing the prior reachable-ceiling version, "
                       "which the manager flagged as tautologically high whenever few free transfers are banked. This "
                       "is the same figure previously shown as 'Free Hit rating'. "
                       "Patch 51 (2026-09-16): relabeled from 'Team Rating %' to 'Quick Team Rating — single-GW, EST' "
                       "because it is scored against GW{planning_gw} ALONE — it does not satisfy the model doc's §1a "
                       "requirement (a) of a fixed 3-4 GW horizon, so per the doc it 'is not a Team Rating % under this "
                       "clause — it's a bounded estimate and must be labeled as such.' The MATH is unchanged from Patch "
                       "24 — same numbers, label/disclosure only. See 'Team Rating % (GW{compliant_gw_start}-"
                       "{compliant_gw_end}) — full breakdown' below for the doc-compliant metric."
                       .format(planning_gw=planning_gw, compliant_gw_start=compliant_gw_start,
                               compliant_gw_end=compliant_gw_end))

        # Patch 51 (2026-09-16, §1a compliance fix) — the new, genuinely
        # §1a-compliant "Team Rating %" expander: fixed multi-GW horizon,
        # full-pool unconstrained ceiling (reusing `theoretical_ceiling`, no new
        # MILP solve), captaincy-aware, same margin-of-error banding as every
        # other Team Rating variant. Kept in its own expander (not merged into
        # the one above) so the two differently-scoped percentages are always
        # visually and textually separate — Patch 49 precedent.
        if compliant_rating["rating_pct"] is not None:
            if compliant_at_ceiling:
                st.caption(f"✓ Already at the GW{compliant_gw_start}-{compliant_gw_end} full-pool optimal",
                           help=f"The {compliant_gap:.1f} xPts gap is inside normal weekly noise (threshold "
                                f"{compliant_moe:.1f} xPts), not real room left on the table.")
            with st.expander(f"Team Rating % (GW{compliant_gw_start}-{compliant_gw_end}) — full breakdown"):
                st.markdown(tier_label)
                _nc("Squad vs ceiling xPts", f"GW{compliant_gw_start}-GW{compliant_gw_end} Squad_xPts (your current 15, best-XI-per-GW, "
                           f"captain doubled, bench autosub-discounted, summed across all {len(compliant_gw_list)} "
                           f"GWs): {compliant_squad_total:.1f} · GW{compliant_gw_start}-GW{compliant_gw_end} "
                           f"Ceiling_xPts (a genuinely unconstrained, full-player-pool, complete 15-man £100m squad "
                           f"solve — same theoretical_ceiling squad used elsewhere in this app, re-summed over this "
                           f"fixed window, no second solve): {compliant_ceiling_total:.1f} · gap: "
                           f"{compliant_gap:.1f} xPts (margin-of-error threshold: {compliant_moe:.1f} xPts)")
                _nc("Rating method", f"Patch 51 methodology (2026-09-16): Team Rating % = Squad_xPts(horizon "
                           f"{len(compliant_gw_list)}) / Ceiling_xPts(horizon {len(compliant_gw_list)}) × 100, per the "
                           f"model doc's §1a clause — a FIXED 3-4 GW horizon (never the sidebar horizon slider) against "
                           f"a full-player-pool, unconstrained, complete 15-man £100m squad ceiling (never a bounded/"
                           f"shortlist estimate). This is the metric that actually satisfies §1a; '{fh_auto_label}' "
                           f"above is a faster single-GW read for day-to-day use and is labeled as such.")
    if snap.stale_warning:
        _nw("Data may be stale", snap.stale_warning)

    # ---------------------------------------------------------------------------
    # Patch 69 (manager screenshot) — moved BELOW the header/verdict/rating-
    # gauge dashboard: the manager wants the at-a-glance rank/rating card at
    # the top of this tab, with the news feed underneath it, not the other
    # way around (Patch 67 originally put the feed first).
    # Latest News feed (Patch 67, manager question: "what will be there, can
    # we fetch the latest news from the FPL app or website?"). ROOT-CAUSE
    # DISCLOSURE: this tab previously only held the header/verdict/rating-
    # gauge dashboard above — never actual injury/press-conference news, a
    # real mismatch against this project's own output_format spec. Verified
    # in code before building this: the official FPL API (already the sole
    # data source this app uses, via fpl_data.fetch_bootstrap_official())
    # already returns per-player `status` (a/d/i/s/u), `news` (free text,
    # e.g. "Ankle injury - Expected back 12 Oct"), `news_added` (the ISO
    # timestamp FPL itself puts on that text) and `chance_of_playing_
    # next_round` (%) — this is the exact same "News" shown on a player's
    # page in the official app/site, not a scrape of anything else. `status`/
    # `news` were already carried into the player table; `chance_of_playing_
    # next_round`/`news_added` were not (added this patch, data_pipeline.py).
    # Scope + sort per manager confirmation (AskUserQuestion, this session):
    # scans the full squad + transfer pool (not just your 15), and shows
    # every currently-flagged player, newest news_added first — no diffing
    # against a prior run.
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-h">🗞️ Latest News</div>', unsafe_allow_html=True)
    _news_sev_map = {"i": ("sev-injured", "Injured"), "s": ("sev-suspended", "Suspended"),
                      "d": ("sev-doubtful", "Doubtful"), "u": ("sev-unavailable", "Unavailable"),
                      "n": ("sev-unavailable", "Not available")}
    _news_universe = pd.concat(
        [squad_df.assign(_in_squad=True) if not squad_df.empty else squad_df,
         pool_df.assign(_in_squad=False) if not pool_df.empty else pool_df],
        ignore_index=True, sort=False) if ("status" in squad_df.columns or "status" in pool_df.columns) else pd.DataFrame()
    if not _news_universe.empty and "status" in _news_universe.columns:
        _cop = pd.to_numeric(_news_universe.get("chance_of_playing_next_round"), errors="coerce")
        _flagged_mask = (_news_universe["status"].fillna("a") != "a") | (_cop < 100)
        _news_rows = _news_universe[_flagged_mask].copy()
        _news_rows["_cop"] = _cop[_flagged_mask]
        if not _news_rows.empty:
            _news_rows["_ts"] = pd.to_datetime(_news_rows.get("news_added"), errors="coerce", utc=True)
            _news_rows = _news_rows.sort_values("_ts", ascending=False, na_position="last")
            _NEWS_CAP = 40
            _shown = _news_rows.head(_NEWS_CAP)
            for _, nr in _shown.iterrows():
                _sev_cls, _sev_label = _news_sev_map.get(nr.get("status"), ("sev-doubtful", "Flagged"))
                _n_photo = _photo_url(nr.get("code", 0))
                _n_initials = "".join([w[0] for w in str(nr.get("web_name", "??")).split()][:2]).upper() or "??"
                _n_text = (nr.get("news") or "").strip() or "No further detail published by FPL yet."
                _n_ts = nr["_ts"]
                _n_ts_txt = _n_ts.strftime("%d %b %Y, %H:%M UTC") if pd.notna(_n_ts) else "Timestamp unconfirmed"
                _n_cop = nr.get("_cop")
                _n_cop_txt = recommend.chance_bar_html(_n_cop) if pd.notna(_n_cop) else ""
                _n_text = recommend.strip_chance_suffix(_n_text) or _n_text
                _n_scope = "IN YOUR SQUAD" if nr.get("_in_squad") else "WATCHLIST"
                st.markdown(
                    '<div class="news-item ' + _sev_cls + '">'
                    '<div class="ring"><img src="' + _n_photo + '" '
                    'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">'
                    '<div class="avatar-fallback">' + _n_initials + '</div></div>'
                    '<div class="body">'
                    '<div class="top-row">'
                    '<span class="name">' + str(nr.get('web_name', '')) + '</span>'
                    '<span class="team">' + str(nr.get('team', '')) + ' · ' + str(nr.get('position', '')) + '</span>'
                    '<span class="status-badge ' + _sev_cls + '">' + _sev_label + '</span>'
                    '<span class="scope-tag">' + _n_scope + '</span>'
                    + _n_cop_txt +
                    '</div>'
                    '<div class="text">' + html.escape(_n_text) + ' · <span class="ts" style="display:inline">' + _n_ts_txt + '</span></div>'
                    '</div></div>', unsafe_allow_html=True)
            if len(_news_rows) > _NEWS_CAP:
                _nc(f"Top {_NEWS_CAP} of {len(_news_rows)} flagged", f"Showing the {_NEWS_CAP} most recently updated of {len(_news_rows)} flagged players "
                           f"across your squad + the transfer pool.")
            _nc("Source: FPL API", "Source: official FPL API (status/news/chance-of-playing-next-round), the same data shown "
                       "on a player's page in the official app/site — refreshed every model run, not scraped "
                       "from anywhere else. \"IN YOUR SQUAD\" = one of your 15; \"WATCHLIST\" = anyone else in "
                       "the transfer pool, so a target's injury shows up here before you'd notice it manually.")
        else:
            _nc("✓ No injury news", "No live injury/status news for your squad or the transfer pool right now — every "
                       "scanned player is currently 'a' (available) with no doubt flag from FPL.")
    else:
        _nc("News feed unavailable", "News feed unavailable this run — player status/news columns weren't present in this run's "
                   "data (see any staleness warning below).")

with tab_chips:
    # ---------------------------------------------------------------------------
    # Chip status
    # ---------------------------------------------------------------------------
    # Patch 54 (2026-09-17, manager decision) — the old "Chip Rack" section
    # rendered every one of the 8 calendar windows (2 per chip: Wildcard/Bench
    # Boost/Triple Captain/Free Hit) as its own small tile up here, ABOVE the 4
    # signal cards below that already summarize the same chips. Once the signal
    # cards themselves started showing a genuine "USED GW{n}" state (this same
    # patch, see _advisor_card below), that tile row became pure duplication —
    # manager's own call: "there is no need for the chips here because we are
    # using the boxes." Removed entirely; `chip_rows` itself is untouched and
    # still drives the signal cards, the pills below, and the Wildcard trigger
    # exactly as before -- only this tile-row rendering is gone.
    flagged_chip_names = set()
    if wc_flag:
        flagged_chip_names = {r["chip"] for r in chip_rows if r["status"] == "available" and r["chip"].startswith("Wildcard")}
    much_more = wc_trigger["reason"] if (wc_trigger and not wc_flag and wc_trigger.get("avg_rating_pct") is not None) else None


    # Patch 52 (2026-09-16, manager screenshot) — Patch 51 removed _flag_pill's
    # hard 70-char truncation (which used to cut four pill types off mid-
    # sentence), but did so by showing the FULL sentence on every pill, which
    # just traded "truncated" for "giant multi-line coral box" for the same
    # four pill types. The fix Patch 48 already used successfully for the
    # Wildcard cross-check pill is generalized here to the other three: a
    # short, genuinely-authored (never truncated) headline on the pill itself,
    # full original sentence unabridged in its hover tooltip via
    # _flag_pill(short, full). Every helper below is built from the ACTUAL
    # fields/note text the corresponding function returns (verified by reading
    # wildcard_trigger_check(), chip_recommendations(), wildcard_freehit_
    # shape_test() and disruption_check() directly, not guessed from shape) —
    # nothing here fabricates data the note didn't already carry.
    def _wc_active_pill_headline(trigger: dict) -> str:
        """Short headline for the ACTIVE 'Wildcard trigger ACTIVE' pill
        (Patch 78 — manager feedback: this was the one pill Patch 51/52 never
        migrated off the always-full-sentence _flag_pill(text) single-arg
        call, so it stayed a large uneven box next to the three short ones).
        Pulls the same avg_rating_pct/cumulative_gap fields
        wildcard_trigger_check() always returns when active — the identical
        fields _wc_not_active_pill_headline() already uses for the inactive
        case, just phrased for 'active.' Full original wildcard_trigger_flag()
        sentence (with the rank-worsening color note, if any) stays unabridged
        in this pill's tooltip via the existing _flag_pill(short, full) call
        shape."""
        pct = trigger.get("avg_rating_pct")
        gap = trigger.get("cumulative_gap")
        if pct is not None and gap is not None:
            return f"Wildcard trigger ACTIVE ({pct}%, {gap:.1f} xPts gap)"
        return "Wildcard trigger ACTIVE"


    def _wc_not_active_pill_headline(trigger: dict) -> str:
        """Short headline for the 'Wildcard trigger: not active' pill. Pulls
        the two numbers wildcard_trigger_check() always returns when inactive
        (avg_rating_pct / cumulative_gap) rather than the full `reason`
        sentence, which stays as this pill's tooltip unabridged."""
        pct = trigger.get("avg_rating_pct")
        gap = trigger.get("cumulative_gap")
        if pct is not None and gap is not None:
            return f"Wildcard trigger: not active ({pct}%, {gap:.1f} xPts gap — noise band)"
        return "Wildcard trigger: not active (inside noise band)"


    _CHIP_NOTE_RE = re.compile(
        r"^GW(?P<gw>\d+): confirmed (?P<kind>double|blank) (?:for|hits) (?P<n>\d+) of your clubs — "
        r"(?P<chip>.+?) (?:has|is) ")


    def _chip_recommendation_pill_headline(note: str) -> str:
        """Short headline for a chip_recommendations() note (BB/TC/FH DGW/BGW
        exposure). All three sentence shapes that function emits start
        'GW{n}: confirmed double/blank ... — {chip_tag} has/is ...' — parsed
        here rather than hardcoded, so it stays correct if the club count or
        chip tag changes. Full original sentence is unabridged in the tooltip."""
        m = _CHIP_NOTE_RE.match(note)
        if m:
            plural = "s" if m["n"] != "1" else ""
            return f"{m['chip']}: {m['kind']} GW exposure (GW{m['gw']}, {m['n']} club{plural})"
        return "Chip exposure flagged this GW"


    def _shape_test_pill_headline(shape_test: dict) -> str:
        """Short headline for the Step 8c shape-test note. classification is
        always one of wildcard_shaped/freehit_shaped/no_signal by the time this
        is called (insufficient_data is filtered out by the caller)."""
        return f"Shape-test: {shape_test['classification'].replace('_', ' ')}"


    def _disruption_pill_headline(note: str) -> str:
        """Short headline for one disruption_check() note. That function emits
        several distinct note shapes (the live-flag note, the horizon-cap note,
        the price-drop-flow note, and two 'nothing to cap yet' no-op notes) —
        each handled by prefix here (verified against disruption_check()'s
        actual f-strings) so the player/GW count in the headline always reflects
        the same data the underlying note does. Full original sentence is
        unabridged in the tooltip regardless of which shape matched."""
        if note.startswith("Disruption check (Rule #41 auto-trigger):"):
            m = re.search(r": (.+) currently carr", note)
            n = len(m.group(1).split(", ")) if m else "1+"
            return f"Disruption check: {n} player(s) flagged"
        if note.startswith("Price-drop-flow override"):
            m = re.search(r"\): (.+) current transfers-out", note)
            n = len(m.group(1).split(", ")) if m else "1+"
            return f"Price-drop-flow override: {n} player(s) flagged"
        if note.startswith("A full-rebuild chip is planned for"):
            m = re.search(r"planned for (GW\d+)", note)
            return f"Disruption: {m.group(1) if m else 'rebuild'} chip already covers it"
        if note.startswith("Transfer net-gain horizon capped"):
            m = re.search(r"capped at (GW\d+|-)", note)
            return f"Disruption: transfer horizon capped ({m.group(1) if m else 'see detail'})"
        if note.startswith("Next planned full-rebuild chip GW set"):
            m = re.search(r"set to (GW\d+)", note)
            return f"Disruption: planned chip {m.group(1) if m else ''} noted".replace("  ", " ")
        return "Disruption check — see detail"


    pill_items = []
    if wc_flag:
        pill_items.append(_flag_pill(_wc_active_pill_headline(wc_trigger), wc_flag))
        if _wc_check_note:
            pill_items.append(_flag_pill(_wc_check_headline, _wc_check_note))
    elif much_more:
        _wc_not_active_full = f"Wildcard trigger: not active — {much_more}."
        pill_items.append(_flag_pill(_wc_not_active_pill_headline(wc_trigger), _wc_not_active_full))
    for note in chip_notes:
        pill_items.append(_flag_pill(_chip_recommendation_pill_headline(note), note))
    if shape_test and shape_test["classification"] != "insufficient_data":
        for note in shape_test["notes"]:
            pill_items.append(_flag_pill(_shape_test_pill_headline(shape_test), note))
    # Patch 58 / Standing Rule #45 (v6.8, Structural Drift Escalation Rule) —
    # manager-approved implementation: the rule wants the shape-test's
    # "wildcard_shaped" pattern flagged when it persists across the SAME NUMBER
    # OF CONSECUTIVE WEEKLY RUNS as the detection window. This app has no
    # cross-session persistence (Step 2 — confirmed, no new storage added per
    # the manager's explicit choice), so genuine week-over-week tracking isn't
    # available. Disclosed proxy used instead (manager-approved): "wildcard_
    # shaped" is ITSELF only ever classified when overlap already sits at/below
    # the ceiling across the WHOLE current detection window in a single run
    # (chip_protocol.wildcard_freehit_shape_test(), the `all(v <= structural_
    # ceiling for v in overlaps)` check) — that single-run signal stands in for
    # the rule's multi-run one, clearly labeled as a proxy below, not verified
    # history. Informational only, alongside HOLD — never overrides the
    # points-based Wildcard verdict (wildcard_trigger_check()).
    if shape_test and shape_test["classification"] == "wildcard_shaped":
        _drift_note = (
            "Structural drift (Standing Rule #45, v6.8): the shape-test's persistent-gap pattern is present "
            "across the whole current detection window this run. This app has no cross-session persistence, "
            "so this is a disclosed single-run PROXY for Rule #45's 'same number of consecutive weekly runs "
            "as the detection window' condition, not verified week-over-week history — re-check next run to "
            "see if it repeats. Informational only, alongside HOLD — never an automatic override of the "
            "points-based Wildcard verdict above.")
        pill_items.append(_flag_pill("Structural drift flagged (Rule #45, proxy)", _drift_note))
    for note in disruption["notes"]:
        pill_items.append(_flag_pill(_disruption_pill_headline(note), note))
    if pill_items:
        st.markdown(f'<div class="flag-row">{"".join(pill_items)}</div>', unsafe_allow_html=True)

    # Patch 87 (manager report against a live screenshot, 2026-09-29): the
    # Rules #48/49 "Chip Sequence" section (Patch 85) placed a second,
    # independently-computed verdict for the same four chips right below
    # this card grid — and the two genuinely disagreed (e.g. Free Hit card
    # said "PLAY GW7", the sequence said "GW9"), because the cards below
    # come from each chip's own isolated scan (evaluate_bench_boost() /
    # evaluate_triple_captain() / evaluate_free_hit() / wildcard_trigger_
    # check()) while the sequence comes from the JOINT assignment
    # (chip_portfolio_schedule()) that accounts for how the chips compete
    # for weeks — confirmed in code, not a display coincidence. Manager
    # chose "joint sequence drives the cards" as the fix: once >=2 chip
    # types are available and the joint scan actually ran, each card's
    # headline GW/verdict below is now the SEQUENCED one, with its own
    # isolated-scan number folded into the tooltip as context instead of
    # standing as a second, contradicting number. Also restores the Patch
    # 31 "more visuals, more than words" rule the standalone prose section
    # broke — see the compact one-line summary that replaces it below the
    # card grid.
    _seq_detail = (chip_portfolio or {}).get("detail", {})
    _seq_order = {t: i + 1 for i, (t, _) in
                  enumerate(sorted(_seq_detail.items(), key=lambda kv: kv[1]["gw"]))}
    _seq_n = len(_seq_detail)
    _seq_is_active = chip_portfolio is not None

    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _seq_for(chip_key: str) -> dict | None:
        if not _seq_is_active:
            return None
        info = _seq_detail.get(chip_key)
        return {"scheduled": info is not None, "gw": info["gw"] if info else None,
                "value": info["value"] if info else None, "order": _seq_order.get(chip_key),
                "n_scheduled": _seq_n, "total_gain": chip_portfolio.get("total_gain"),
                "tie_band": chip_portfolio.get("tie_band"),
                "near_tie_count": chip_portfolio.get("near_tie_count"),
                # Rule #49(f) -- only ever set on the "wildcard" entry, only
                # when Bench Boost is also scheduled at/after it.
                "bench_term": (info or {}).get("bench_term"),
                "player": (info or {}).get("player")}      # Patch 113: Triple Captain's named player

    # Patch 107 (2026-10-04): compare THIS run's chip schedule with the
    # previous NORMAL run's (saved below), so an extended-check run can tag
    # any card it moved ("updated by extended check (was GWx)"). A normal run
    # just refreshes the saved baseline; an extended run never overwrites it.
    _CARD_CHIP_KEY = {"Bench Boost": "bboost", "Triple Captain": "3xc", "Free Hit": "freehit"}
    _chip_now = {k: (_seq_detail.get(k) or {}).get("gw") for k in ("wildcard", "bboost", "3xc", "freehit")}
    _chip_tags = {}
    try:
        if _extended_mode:
            _chip_tags = recommend.chip_change_tags(st.session_state.get("_chip_baseline"), _chip_now)
        else:
            st.session_state["_chip_baseline"] = dict(_chip_now)
    except Exception:
        _chip_tags = {}

    def _tc_player_tag(label: str, seq) -> str:
        return f" · {seq['player']}" if (label == "Triple Captain" and seq and seq.get("player")) else ""

    def _card_tag(label: str) -> str:
        t = _chip_tags.get(_CARD_CHIP_KEY.get(label, ""), "")
        return f" · {t}" if t else ""

    def _seq_combined_note(seq: dict) -> str:
        """Patch 87b (manager: cards and a separate sequence summary read as
        two things — only one should exist). The combined total/tie-band/
        scope disclosure that used to live in its own always-visible caption
        below the grid now lives ONLY here, in each scheduled card's own
        tooltip — the grid is the single visible surface, hover for the
        rest, same convention as every other card on this tab.

        Patch 89 (Rule #50, Chip-Gain Reporting) — appends the doc-required
        horizon/formula-variant/tier disclosure via the shared gain_
        disclosure() formatter, and (Rule #49d/#49f) updates the scope note
        now that both are partially modeled instead of not at all: (d) uses
        a one-shot reachable-squad approximation, not a full chained
        simulation; (f) is a disclosure-only addition, not a scoring
        change."""
        _gw_list = chip_adv_window["gw_list"] if chip_adv_window else None
        return (f" Combined across this {seq['n_scheduled']}-chip sequence: {seq['total_gain']:+.1f} xPts vs. the "
                f"best transfer path (tie band ±{seq['tie_band']:.1f} xPts, {seq['near_tie_count']} sequence(s) "
                f"inside it, later commitment preferred — Rule #34/#49) "
                f"{chip_protocol.gain_disclosure('best_transfer_path', _gw_list, cfg)}. Rule #49d (preparation "
                f"transfers) uses a one-shot reachable-squad approximation per candidate week, not a full "
                f"week-by-week chained simulation — same performance tradeoff already disclosed for Rule #48's "
                f"own baseline. Rule #49f (Wildcard bench term) is a disclosure only, shown on the Wildcard card "
                f"when Bench Boost is also scheduled — it does not change any reported value. Re-run every "
                f"gameweek — this is a hypothesis, not a commitment (Rule #32).")

    # Chip Signals — Patch 31 visual redesign. Replaces the old paragraph-per-
    # rule Chip Advisor + Chip Strategy expanders with one scannable card grid;
    # every card's rule/step citation and full quantified reasoning lives in its
    # hover tooltip (same hover-hidden pattern as the header's .info-dot),
    # instead of sitting as permanent visible body text.
    def _advisor_card(label: str, adv: dict | None, used_state: dict | None = None,
                       seq: dict | None = None) -> str:
        _win = (f"GW{chip_adv_window['gw_list'][0]}-GW{chip_adv_window['gw_list'][-1]}" if chip_adv_window
                else "the scanned window")
        # Patch 54 — this chip has already been played this half, and the near-
        # term scan window doesn't reach the next available window yet (see
        # _clip_to_available_windows above). Showing "HOLD"/"PLAY" here would be
        # fabricating a verdict over a date range where the chip literally can't
        # be played — show what's actually true instead: when it was used, and
        # when the next one opens.
        if used_state is not None:
            sub = (f"next available GW{used_state['next_open_gw']}" if used_state.get("next_open_gw") is not None
                   else "no further window this season")
            tooltip = (f"{label} was already played at GW{used_state['last_used_gw']}. " +
                       (f"The next available window opens GW{used_state['next_open_gw']} — this card will show a "
                        f"real PLAY/HOLD verdict again once the scan window reaches it."
                        if used_state.get("next_open_gw") is not None else
                        "No further window is available for this chip this season."))
            return _signal_card(label, f"USED GW{used_state['last_used_gw']}", "used", "—", sub, tooltip)
        # Patch 87 — once >=2 chip types are available, the joint Rule #48/49
        # sequence (below) is the authoritative "which week" answer for this
        # chip, not this card's own isolated scan — the two can genuinely
        # disagree (confirmed live) because the sequence accounts for the
        # other available chips competing for the same weeks. When the
        # sequence ran (`seq` is not None), it drives the headline; the
        # isolated scan's own number is kept, but only as tooltip context.
        if adv is None or adv.get("best_gw") is None:
            if seq is not None and seq["scheduled"]:
                return _signal_card(
                    label, f"PLAY GW{seq['gw']}", "play", f"GW{seq['gw']}",
                    f"{_ordinal(seq['order'])} of {seq['n_scheduled']} in sequence · {seq['value']:+.1f} xPts" + _tc_player_tag(label, seq) + _card_tag(label),
                    f"No isolated single-chip scan data for this chip this run, but the joint Rule #48/49 sequence "
                    f"(which accounts for your other available chips) schedules it GW{seq['gw']} "
                    f"({seq['value']:+.1f} xPts vs. the best no-chip transfer path)." + _seq_combined_note(seq),
                    "is-play")
            return _signal_card(label, "N/A", "used", "—", "no data this run",
                                 f"No candidate gameweek available for this chip across {_win}.")
        # Free Hit's by_gw is {gw: {"current":.., "rebuild":.., "gap":..}} (a
        # rebuild-vs-hold comparison, not a single number) — Bench Boost/Triple
        # Captain's is already {gw: float}. Normalize to the bar-chart-relevant
        # number in each case: the rebuild's net gap for Free Hit, the raw value
        # for the other two.
        raw_by_gw = adv.get("by_gw") or {}
        if raw_by_gw and isinstance(next(iter(raw_by_gw.values())), dict):
            by_gw = {gw: v.get("gap", 0.0) for gw, v in raw_by_gw.items()}
        else:
            by_gw = raw_by_gw
        if seq is not None:
            _isolated_gw = adv.get("best_gw")
            _isolated_note = (
                f"Scanned on its own, this chip's best candidate is GW{_isolated_gw} — sequencing it against your "
                f"other available chips (Rule #49) " +
                (f"confirms GW{_isolated_gw}." if seq["scheduled"] and seq["gw"] == _isolated_gw else
                 f"moves it to GW{seq['gw']}." if seq["scheduled"] else
                 "finds no positive slot for it within the current window instead."))
            if seq["scheduled"]:
                return _signal_card(
                    label, f"PLAY GW{seq['gw']}", "play", f"GW{seq['gw']}",
                    f"{_ordinal(seq['order'])} of {seq['n_scheduled']} in sequence · {seq['value']:+.1f} xPts" + _tc_player_tag(label, seq) + _card_tag(label),
                    f"{_isolated_note} Value against the best no-chip transfer path: {seq['value']:+.1f} xPts "
                    f"(Rule #48/#50)." + _seq_combined_note(seq), "is-play", by_gw=by_gw, best_gw=seq["gw"])
            return _signal_card(
                label, "HOLD", "hold", f"GW{_isolated_gw}", "no slot in the current joint sequence — held" + _card_tag(label),
                f"{_isolated_note} Re-run every gameweek — this is a hypothesis, not a commitment (Rule #32).",
                by_gw=by_gw, best_gw=adv.get("best_gw"))
        if adv["verdict"].startswith("play_gw"):
            gw = int(adv["verdict"].split("gw")[1])
            margin = adv.get("margin", adv.get("threshold", 0))
            return _signal_card(
                label, f"PLAY GW{gw}", "play", f"GW{gw}", f"+{margin:.1f} xPts clear of next-best — click for the "
                f"per-GW breakdown",
                f"The single best GW across a {_win} scan (Patch 32 — independent of the sidebar's transfer "
                f"Horizon slider, extended further if a confirmed Double/Blank fell just past it). Clears "
                f"margin-of-error by {margin:.1f} xPts over the next-best GW in that window "
                f"(threshold {adv['threshold']:.1f} xPts) — Standing Rule #34 margin-of-error gate.",
                "is-play", by_gw=by_gw, best_gw=gw)
        return _signal_card(
            label, "HOLD", "hold", f"GW{adv['best_gw']}", "best candidate, statistical tie — click for the breakdown",
            f"No GW across a {_win} scan clears margin-of-error over the others (best candidate GW{adv['best_gw']}, "
            f"threshold {adv['threshold']:.1f} xPts) — Standing Rule #34. A statistical tie, not a reason to rule "
            f"it out later.", by_gw=by_gw, best_gw=adv.get("best_gw"))

    # Patch 87 — Rule #48's window-value scan runs regardless of whether the
    # need-based trigger above fired (the doc: window value is decided
    # "beside the need trigger", not instead of it), so the joint sequence
    # can recommend a Wildcard week even when this card reads HOLD/N/A. That
    # is surfaced here as an addendum to whichever badge state already
    # applies — it never overrides the trigger badge itself, since ACTIVE
    # vs. HOLD is genuinely different information (a mechanical need signal,
    # Rule #24/#41) from a value-only sequencing recommendation (Rule #48).
    _wc_seq = _seq_for("wildcard")
    _wc_seq_sub_suffix, _wc_seq_tooltip_suffix = "", ""
    if _wc_seq is not None:
        if _wc_seq["scheduled"]:
            # Rule #49(f) -- disclose the bench sum this build carries into
            # a Bench Boost scheduled at/after it, when chip_portfolio_
            # schedule() reported one (chip_protocol.py sets bench_term
            # only on the "wildcard" detail entry, only in that case).
            _bench_note = (f" This build also carries a bench worth {_wc_seq['bench_term']:+.1f} xPts for the "
                            f"Bench Boost planned soon after (Rule #49f)."
                            if _wc_seq.get("bench_term") is not None else "")
            _wc_seq_sub_suffix = (f" · sequenced GW{_wc_seq['gw']} ({_ordinal(_wc_seq['order'])} of "
                                   f"{_wc_seq['n_scheduled']}, {_wc_seq['value']:+.1f} xPts)"
                                   + (f" · {_chip_tags['wildcard']}" if "wildcard" in _chip_tags else ""))
            _wc_seq_tooltip_suffix = (f" Rule #48/#49's joint sequence recommends playing it GW{_wc_seq['gw']} "
                                       f"({_wc_seq['value']:+.1f} xPts vs. the best no-chip transfer path) as part "
                                       f"of a {_wc_seq['n_scheduled']}-chip sequence — the exact date remains your "
                                       f"own call (Rule #32)." + _bench_note + _seq_combined_note(_wc_seq))
        else:
            _wc_seq_sub_suffix = " · no positive slot in the current joint sequence"
            _wc_seq_tooltip_suffix = (" Rule #48/#49's joint sequence finds no positive slot for the Wildcard "
                                       "within the current scan window given your other available chips.")

    wc_card_tooltip = (wc_flag or (f"Wildcard trigger: not active — {much_more}." if much_more else
                                    "Insufficient data to evaluate the trigger this run."))
    wc_card_tooltip += " Wildcard's trigger condition is mechanical (Standing Rule #24/#41), but the specific play " \
                        "date is never a mechanical verdict — it's a rolling re-test per Standing Rule #32."
    wc_card_tooltip += _wc_seq_tooltip_suffix
    # Patch 73 (fixes a confirmed-broken Patch 72 check — see the
    # wc_diag_reason note above where it's computed. wc_trigger is NEVER
    # actually None on a "insufficient data" run; it's always a placeholder
    # dict, which is why Patch 72's `wc_trigger is None` check could never
    # fire and the tooltip kept showing the same generic text with no
    # diagnostic appended. wc_diag_reason is the corrected replacement,
    # computed right where each real precondition is known.
    if not wc_flag and not much_more and wc_diag_reason:
        wc_card_tooltip += f" ⚠ Diagnostic (Patch 73): {wc_diag_reason}"
    if wc_flag and _wc_check_note:
        wc_card_tooltip += " " + _wc_check_note
    _wc_verdict = None
    if wc_flag:
        wc_stat = f'{wc_trigger["avg_rating_pct"]}%' if wc_trigger and wc_trigger.get("avg_rating_pct") is not None else "ACTIVE"
        # Patch 49 — the "may not be needed if you take the recommended
        # transfers" finding now lives directly on this card, same prominence
        # as "TRIGGER ACTIVE" itself, not just in the smaller Chip Rack pill
        # row above (that pill stays exactly as Patch 48 left it — this is in
        # addition, not instead).
        _wc_note_html, _wc_note_cls = None, ""
        if _wc_check_note:
            _wc_note_html = ("✓ Your recommended transfers may already close this — the chip may not be needed"
                              if _closes else
                              "Recommended transfers alone don't close this — Wildcard still looks warranted")
            _wc_note_cls = "good" if _closes else "warn"
        # Patch 108 (2026-10-04, manager: "if my team for this week is 90% do
        # you think that triggering the chip is the right call ... is this the
        # best way to present it"): the 6% trigger (Rule #45) is unchanged and
        # still decides that this branch runs; the card now leads with the
        # DECISION (PLAY GWn / MONITOR / HOLD) and the net xPts gain instead
        # of a bare alarm percentage. Falls back to the old card when no gap
        # figure exists.
        # Patch 110: the card reads the SAME decision object as the pill and the table.
        _wc_verdict = _wc_decision
        if _wc_verdict is not None:
            _wc_tag = f" · {_chip_tags['wildcard']}" if "wildcard" in _chip_tags else ""
            wc_card = _signal_card(
                "Wildcard", _wc_verdict["badge"], _wc_verdict["badge_cls"], _wc_verdict["stat"],
                _wc_verdict["sub"] + _wc_tag,
                f"Patch 111 chain comparison: a free-transfers-only chained plan (no hits) with NO chip vs with the "
                f"Wildcard at each candidate week (team rebuilt for the next 8 GWs), summed over the whole checked "
                f"span. Earliest week within the margin of error of the best wins; below the materiality bar = HOLD. "
                f"Thresholds borrowed, not validated. The 6% trigger (Standing Rule #45) is unchanged. " + wc_card_tooltip,
                _wc_verdict["card_cls"], note=_wc_verdict["note"], note_cls=_wc_verdict["note_cls"])
        else:
            wc_card = _signal_card("Wildcard", "TRIGGER ACTIVE", "active", wc_stat,
                                    f"structural gap detected — date is your call{_wc_seq_sub_suffix}",
                                    wc_card_tooltip, "is-active", note=_wc_note_html, note_cls=_wc_note_cls)
    elif not _wc_available_now and _wc_last_used is not None:
        # Patch 55 (manager report, team 1301651: Wildcard played GW4, closing
        # window 1 — window 2 (a real, separate calendar window) isn't reachable
        # yet, so _wc_available_now is correctly False and wc_trigger never even
        # ran. Show the same genuine "USED GW{n}" state the Bench Boost/Triple
        # Captain/Free Hit cards already show in this situation (_advisor_card,
        # Patch 54), instead of falling through to the old "N/A / insufficient
        # data" branch, which read as a data gap rather than "already played."
        _wc_used_sub = (f"next available GW{_wc_next_open}" if _wc_next_open is not None
                        else "no further window this season")
        _wc_used_tooltip = (f"Wildcard was already played at GW{_wc_last_used}. " +
                             (f"The next available window opens GW{_wc_next_open} — this card will show a live "
                              f"trigger check again once planning reaches it."
                              if _wc_next_open is not None else
                              "No further window is available for this chip this season."))
        wc_card = _signal_card("Wildcard", f"USED GW{_wc_last_used}", "used", "—", _wc_used_sub, _wc_used_tooltip)
    elif much_more:
        wc_card = _signal_card("Wildcard", "HOLD", "hold", f'{wc_trigger["avg_rating_pct"]}%',
                                f"inside noise band — no trigger{_wc_seq_sub_suffix}", wc_card_tooltip)
    else:
        # Patch 74 (manager, 2026-09-27: after Patch 73, still "same issue"
        # — because the Patch 72/73 diagnostic was only ever added to
        # `wc_card_tooltip`, the hover-only `title` attribute _signal_card()
        # puts on the whole card (see its docstring: "Rule/step citations
        # ... move into the card's `title` tooltip (hover-hidden)"). The
        # manager works from screenshots, which cannot show a hover state at
        # all — so the diagnostic was invisible no matter how correct it
        # was. This was never a second logic bug: it's the exact same
        # sentence any screenshot ever showed, because the VISIBLE part of
        # this card (the `sub` argument below) was never changed. Fixed by
        # putting the diagnostic directly in `sub` — the always-visible
        # one-line caption every other Chip Signals card already uses for
        # its own explanatory text — instead of only in the tooltip.
        _wc_sub = f"insufficient data this run — {wc_diag_reason}" if wc_diag_reason else "insufficient data this run"
        wc_card = _signal_card("Wildcard", "N/A", "used", "—", f"{_wc_sub}{_wc_seq_sub_suffix}", wc_card_tooltip)

    signal_html = '<div class="signal-grid">' + wc_card + \
        _advisor_card("Bench Boost", bb_advisor, bb_used_state, _seq_for("bboost")) + \
        _advisor_card("Triple Captain", tc_advisor, tc_used_state, _seq_for("3xc")) + \
        _advisor_card("Free Hit", fh_advisor, fh_used_state, _seq_for("freehit")) + '</div>'
    st.markdown(signal_html, unsafe_allow_html=True)

    # Patch 101 (2026-10-01, manager: "it's 4 minutes 46 seconds now ,, i
    # need it below 2 minutes") -- the Patch 95 extended Wildcard cross-check
    # (reaches a scheduled Wildcard's actual GW when it falls beyond the
    # normal transfer-plan/reachable-ceiling windows) used to run
    # automatically every time that condition was met. It's real cost (a
    # full extra transfer-plan solve plus a full extra reachable-ceiling
    # scan over +8 GWs), so it's opt-in now -- same button-gated pattern as
    # Cross-Tool Reconciliation below. Clicking it recomputes the Wildcard
    # card's own tooltip/sub-line above with the deeper check folded in; a
    # later, unrelated widget interaction reverts to the normal (faster)
    # check until clicked again -- identical behavior to how Reconcile's own
    # result only shows for the run right after it's clicked.
    # Patch 102 (2026-10-01, manager: "the button for extended wildcard needs
    # to be more visual and self explained") -- the plain st.button above
    # gave no indication of WHY you'd click it or WHAT GW it would actually
    # reach, and showed identically whether or not it had anything to do --
    # confirmed via code read that `_auto_wildcard_gw` (chip_portfolio's own
    # scheduled Wildcard GW) and `_wc_current_max_gw` (the normal check's own
    # reach) are both already available here, unchanged, from earlier in
    # this same script run -- enough to make this self-explanatory without
    # any new computation.
    # Patch 104 (2026-10-01, manager screenshot + "i didn't get it!!" after
    # this button went correctly-but-confusingly disabled once Patch 103's
    # wider normal reach already covered the one scheduled Wildcard on
    # screen). Confirmed via AskUserQuestion: redesigned from a conditional
    # "only relevant if a scheduled chip needs it" gate into an ALWAYS-ON
    # manual control -- clicking it pushes the cross-check a further fixed
    # +5 GWs beyond the normal reach regardless of whether anything is
    # currently scheduled out there (recommend.resolve_wildcard_extend_target()
    # above already extends the target further still if a real scheduled
    # chip sits beyond even that). The messaging below just changes framing
    # depending on whether a scheduled chip happens to be the reason this
    # run's extension matters, or whether it's purely exploratory.
    # Patch 105 (2026-10-01) — `_wc_extend_chip_driven` is now computed once,
    # early (see above), and reused here rather than recomputed — no second
    # copy of the same condition to drift out of sync.
    with st.container(border=True):
        _span_n = _ext_span - 1
        st.button(recommend.extended_button_label(planning_gw, _chip_target_gw, _chip_span),
                  key="wc_extend_run", type="primary")
        # Patch 105/107: explicit confirmation right where the button is.
        if _wc_extend_requested_flag:
            if _wc_extend_info is None:
                # Patch 108: no Wildcard cross-check to run for this team
                # (e.g. Wildcard already used) -- the chips themselves were
                # still re-evaluated over the wider window.
                _chips_reached = max(chip_adv_gw_list) if chip_adv_gw_list else None
                if _chips_reached is not None and _chips_reached < _chip_target_gw:
                    _nw(f"⚠️ Chips checked to GW{_chips_reached} only", f"⚠️ Finished short — chips re-evaluated over GW{planning_gw}–GW{_chips_reached} "
                               f"(no projection data beyond that).")
                else:
                    st.success(f"✅ Done — all chips re-evaluated over GW{planning_gw}–GW{_chip_target_gw}.")
            elif _wc_extended_active:
                _reach_ok, _reach_msg = recommend.extend_reach_status(_wc_extend_target_gw, _check_gws[-1])
                if _reach_ok:
                    st.success(f"✅ Done — all chips re-evaluated over GW{planning_gw}–GW{_chip_target_gw}.")
                else:
                    st.warning(f"⚠️ Finished short. {_reach_msg}")
            else:
                _nw("Cross-check incomplete", "The transfer-plan cross-check didn't complete (a solve failed) — the chip cards are "
                           "still re-evaluated over the wider window.")
            # Patch 107/108: best-GW answer + what changed, visual first.
            _span_gws = list(range(planning_gw, _chip_target_gw + 1))
            _wc_in_play = "wildcard" in _available_chip_types
            # Patch 110: ONE source -- the Wildcard week is the decision's GW (same object as the card
            # and the pill); BB/TC/FH come from the joint sequence.
            _seq_gws_only = {k: v for k, v in (_seq_detail or {}).items()}
            _chips_by_gw, _chip_collisions = recommend.final_chips_by_gw(
                _seq_gws_only, chip_protocol.CHIP_LABELS, _wc_decision, _wc_in_play)
            _best_tbl = recommend.build_best_gw_table(
                planning_gw, _span_gws, ((wc_window_scan or {}).get("by_gw") if _wc_in_play else None),
                {}, {g: v for g, v in _chips_by_gw.items()})
            st.markdown("🏆 **" + recommend.chip_plan_headline(
                _chips_by_gw, _wc_decision, tc_player=((_chain_path or {}).get("chips") or {}).get("tc_player")) + "**")
            if _chips_repicked and _chips_repicked["changed"]:
                _nm = {"3xc": "Triple Captain", "bboost": "Bench Boost", "freehit": "Free Hit"}
                _nc("🔗 Chips re-picked on WC squad", "🔗 chips re-picked on the Wildcard squad: " + " · ".join(
                    f"{_nm.get(k, k)} GW{_chips_repicked['old'].get(k, '—')} → GW{v}"
                    for k, v in _chips_repicked["new"].items() if _chips_repicked["old"].get(k) != v)
                    + "".join(f" · {_nm.get(k, k)} GW{g} → HOLD (no real edge on the Wildcard path)"
                              for k, g in _chips_repicked["old"].items() if k not in _chips_repicked["new"])
                    + (" (picked on the squads the plan fields each week — Wildcard rebuild + later transfers; chip value "
                       "counts in the Wildcard decision, a Rule #31 deviation pending model-chat approval)"
                       if _chips_repicked.get("src") == "chain" else " (Rule #49 re-pick on the Wildcard candidate's own rebuild)"))
            for _cn in _chip_collisions:
                _nw("Chip clash", _cn)
            # Patch 111: which Wildcard candidate to draw beside the no-chip path -- the decision's week,
            # else the candidate with the biggest chained gain.
            _show_t = None
            if _wc_chain:
                if _wc_decision and _wc_decision.get("gw") in _wc_chain["cands"]:
                    _show_t = _wc_decision["gw"]
                else:
                    _show_t = max(_wc_chain["cands"], key=lambda t: _wc_chain["cands"][t]["gain"])
            _rows = []
            for r in _best_tbl["rows"]:
                row = {"GW": f"GW{r['gw']}", "Chips": r["chips"] or "—"}
                if _wc_in_play:
                    row["Wildcard value if played alone (xPts)"] = r["wc_gap"]
                if _wc_chain:
                    row["No chip %"] = _wc_chain["base"]["pct"].get(r["gw"])
                    if _show_t is not None:
                        row[f"With Wildcard GW{_show_t} %"] = _wc_chain["cands"][_show_t]["pct"].get(r["gw"])
                _rows.append(row)
            _tbl_df = pd.DataFrame(_rows)
            _cfgs = {}
            _wcv = "Wildcard value if played alone (xPts)"
            if _wcv in _tbl_df.columns and _tbl_df[_wcv].notna().any():
                _cfgs[_wcv] = st.column_config.ProgressColumn(
                    _wcv, format="%.1f", min_value=0.0, max_value=float(max(1.0, _tbl_df[_wcv].max())))
            for _pc in [c for c in _tbl_df.columns if c.endswith("%")]:
                _cfgs[_pc] = st.column_config.NumberColumn(
                    _pc, format="%.1f%%",
                    help="Squad you would field vs the best possible squad for that GW (the Pitch tab's yardstick), free transfers only (no hits), "
                         "starting from your recommended move this GW. 'No chip' = without the Wildcard; "
                         "'With Wildcard' = chip played at the stated GW.")
            _df(_tbl_df, hide_index=True, use_container_width=True, column_config=_cfgs)
            _tbl_chips = []
            if _wc_chain:
                _raw_all = dict(_wc_chain["base"].get("pct_raw") or {})
                if _show_t is not None:
                    for _g, _v in (_wc_chain["cands"][_show_t].get("pct_raw") or {}).items():
                        _raw_all[_g] = max(_v, _raw_all.get(_g, 0.0)) if _v is not None else _raw_all.get(_g)
                _cn = recommend.capped_note(_raw_all)
                if _cn:
                    _tbl_chips.append(recommend.note_html("⚠ " + _cn, "warn",
                                      "The best-possible-squad solve for these weeks scored at or just below the squad shown, so the % is held at 100. "
                                      "A small raw excess is solver/bench-weight noise, not a real 100%+."))
            if _wc_in_play:
                _tbl_chips.append(recommend.note_html("Scan column: GW7+ understated baseline (accrual defect)", "info",
                                  "'Wildcard value if played alone' gives the free-transfer baseline too few transfers for windows from GW7 (known accrual defect, "
                                  "model chat M8, fix after the deadline), so later weeks read too high. The week choice uses the chain gains, not this column."))
            if _tbl_chips:
                st.markdown(recommend.chips_row_html(_tbl_chips), unsafe_allow_html=True)
            _scan_alt = recommend.scan_alternative_line(_best_tbl["rows"], (_wc_decision or {}).get("gw")) \
                if (_wc_in_play and _wc_decision) else None
            if _scan_alt:
                _nc("ℹ️ Scan-only week differs", _scan_alt)
            if _wc_chain and _wc_chain.get("diag"):
                _d = _wc_chain["diag"]
                _nc(f"GW{_d['gw']}: now {_d['current']} → after {_d['after_move']} · best {_d['reference']}", 
                    f"GW{_d['gw']} (xPts, best XI): current squad **{_d['current']}** → after your recommended move "
                    f"**{_d['after_move']}**; the best possible GW{_d['gw']} squad scores **{_d['reference']}**. "
                    f"Every % in this table is measured against the best possible squad for THAT GW — the same yardstick "
                    f"as the Pitch tab's GW Rating — so a squad that drifts away from the best team shows up as a lower %.")
                if _d["n_after"] != _d["n_current"]:
                    _nw("Squad size changed — % unreliable", f"Squad size changed after the recommended move ({_d['n_current']} → {_d['n_after']}): "
                               "a recommended incoming player was missing from the pool. Table % is unreliable.")
            if _wc_chain and _wc_decision and _wc_decision.get("detail_lines"):
                # Patch 115 (v6.12): band, tie set, guardrail rejections, not-scored weeks, fallback tier, 4-GW cross-check
                if _wc_decision.get("shift_text"):
                    _nw("Wildcard week shifted", _wc_decision["shift_text"])
                _nc(f"Why this week · {len(_wc_decision['detail_lines'])} notes", " · ".join(_wc_decision["detail_lines"]))
            if _wc_chain and _wc_chain["cands"]:
                _pw = int(_wc_chain.get("post_weeks") or 0)
                _gcol = (f"Gain over {_pw} post-chip weeks, decay-weighted (xPts)" if _pw else "Total gain over span (xPts)")
                _g_df = pd.DataFrame({"Wildcard at": [f"GW{t}" for t in sorted(_wc_chain["cands"])],
                                      _gcol: [_wc_chain["cands"][t]["gain"] for t in sorted(_wc_chain["cands"])]})
                if _wc_decision and _wc_decision.get("gains4"):
                    _chip_in = " · ".join(f"GW{t} {c.get('gain_chips', 0.0):+.1f}" for t, c in sorted(_wc_chain["cands"].items()))
                    _ins = any(c.get("chips_inside") for c in _wc_chain["cands"].values())
                    _nc("4-week gain decides", ("Deciding measure (four-week plain gain + chip value): " if _ins else "Deciding measure (four-week PLAIN gain, chips not counted - model M9): ")
                               + " · ".join(f"GW{t} {v:+.1f}" for t, v in sorted(_wc_decision["gains4"].items()))
                               + (f" — chips inside: {_chip_in}" if _ins else f" — chip effect beside it, not counted: {_chip_in}")
                               + " — the bars below are the six-week decay-weighted gain, shown beside it.")
                if _wc_chain.get("not_scored"):
                    _nc("Not scored: short window", "Not scored (window truncated): " + ", ".join(f"GW{g}" for g in _wc_chain["not_scored"])
                               + f" — every scored week is valued over the same {_pw} post-chip weeks (Rule #48(a)).")
                _df(_g_df, hide_index=True, use_container_width=True, column_config={
                    _gcol: st.column_config.ProgressColumn(
                        _gcol, format="%.1f",
                        min_value=min(0.0, float(_g_df[_gcol].min())),
                        max_value=float(max(1.0, _g_df[_gcol].max())))})
            if _wc_chain and _show_t is not None:
                _tl = recommend.plan_timeline_lines(
                    _wc_chain["cands"][_show_t]["plan"],
                    first_week=(None if _show_t == planning_gw else
                                (_wc_first_week + (_wc_chain.get("first_ft_after"),) if _wc_first_week else None)))
                st.markdown(f"📅 **Plan with the Wildcard at GW{_show_t}** (free transfers only)")
                st.code("\n".join(_tl), language=None)
            if _chip_tags:
                st.markdown("🔁 **Moved vs your last normal run:** " + " · ".join(
                    f"{chip_protocol.CHIP_LABELS.get(k, k)} — {v}" for k, v in _chip_tags.items()))
            elif st.session_state.get("_chip_baseline") is None:
                _nc("Run normally, then extend", "Run the model normally once, then extend, to see which cards moved.")
            else:
                st.caption("No chip moved vs your last normal run.")
            if _wc_check_note:
                with st.expander("Cross-check details"):
                    st.markdown(_wc_check_note)

    # Patch 108 (2026-10-04): Wildcard calibration log. The 6% trigger
    # (Standing Rule #45) is flagged "unvalidated" by the model document
    # itself; one row per NORMAL run (extended runs use a wider window, so
    # they are not logged) lets the threshold be judged from real weeks.
    # DISCLOSED: on Streamlit Cloud the file lives on an ephemeral disk
    # (it resets on reboot/redeploy) -- download it regularly.
    _cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildcard_calibration_log.csv")
    try:
        if (not _extended_mode) and wc_trigger and wc_trigger.get("avg_gap_pct") is not None:
            recommend.append_calibration_row(_cal_path, recommend.calibration_log_row(
                team_id=entry_id, team_name=st.session_state.get("team_name", ""), planning_gw=planning_gw,
                patch=PATCH_VERSION.split(" (")[0], style=style_name, gap_pct=wc_trigger.get("avg_gap_pct"),
                rating_pct=wc_trigger.get("avg_rating_pct"), cumulative_gap=wc_trigger.get("cumulative_gap"),
                active=bool(wc_flag),
                verdict=(_wc_decision["badge"] if (wc_flag and _wc_decision) else
                         (_wc_verdict["badge"] if (wc_flag and _wc_verdict) else ("TRIGGER ACTIVE" if wc_flag else "NOT ACTIVE"))),
                wc_value=(_wc_decision["best_gain"] if _wc_decision else
                          (_wc_seq["value"] if (_wc_seq and _wc_seq.get("scheduled")) else None)),
                wc_gw=(_wc_decision["gw"] if _wc_decision else
                       (_wc_seq["gw"] if (_wc_seq and _wc_seq.get("scheduled")) else None)),
                closes=(_closes if _wc_check_note else None),
                shift=(_wc_decision.get("shift") if _wc_decision else None),
                band=(round(_wc_decision["band"], 2) if (_wc_decision and _wc_decision.get("band") is not None) else None),
                wait_cost=(((_wc_decision.get("wait_cost") or {}).get("cost") or 0.0) if _wc_decision else None),
                deferred=(bool(_wc_decision.get("deferred")) if _wc_decision else None)))
    except Exception:
        pass
    # Full analysis — Patch 29's synthesis narrative, kept in full (nothing
    # deleted per manager instruction) but moved behind an opt-in expander now
    # that the cards above carry the at-a-glance read (2026-09-14 redesign).
    #
    # Patch 52 (2026-09-16, manager decision: "tooltip only, drop from the
    # expander") — the `_wc_check_note` append here used to duplicate, verbatim,
    # the exact same sentence that's already the Wildcard cross-check pill's
    # tooltip (Patch 48) — removed. The disruption notes' own duplication is
    # removed at the source instead (chip_strategy_summary()'s own section (5)
    # in chip_protocol.py), since that function no longer receives them
    # verbatim into its output.
    strategy_lines = chip_protocol.chip_strategy_summary(
        wc_flag, shape_test, bb_advisor, tc_advisor, fh_advisor, chip_rows, disruption["notes"], wc_trigger)
    with st.expander("Full chip analysis — combined narrative, rule-by-rule"):
        for line in strategy_lines:
            st.markdown(f"- {line}")
        _nc("Synthesis — nothing new computed", "A synthesis of the Wildcard trigger, shape-test, Chip Advisor verdicts and any disruption notes "
                   "above — computes nothing new itself. Wildcard's trigger is mechanical (Patch 30) but never names "
                   "a single play GW — the date stays a rolling re-test (Standing Rule #32).")
        # Patch 87b (v6.9 Rules #48-49, manager: cards + a separate sequence
        # summary read as two things when only one should exist). The
        # combined total/tie-band figure now lives only in each scheduled
        # card's own tooltip (_seq_combined_note()); this scope disclosure
        # (what Rule #49 does/doesn't model yet) folds into this ALREADY-
        # EXISTING expander instead of getting its own — one opt-in detail
        # panel for the whole tab, not two.
        if chip_portfolio is not None and chip_portfolio.get("assignment"):
            st.markdown("---")
            st.markdown(recommend.ui_chip(f"🗓️ Chip sequence {chip_portfolio['total_gain']:+.1f} xPts", "ok" if chip_portfolio['total_gain'] > 0 else "info",
                                          tip=f"vs the best transfer path, {_seq_n} chips scheduled; tie band ±{chip_portfolio['tie_band']:.1f} xPts, {chip_portfolio['near_tie_count']} sequence(s) inside it, later commitment preferred (Rules #48-49)")
                        + " " + recommend.ui_chip(f"{_seq_n} chips", "info")
                        + " " + recommend.ui_chip(f"band ±{chip_portfolio['tie_band']:.1f}", "info"),
                        unsafe_allow_html=True)
            # Patch 89 — (d) and (f) are now partially modeled (see
            # chip_protocol.py's _reachable_bb_tc_tables() and the
            # bench_term disclosure), so this bullet list states their real
            # scope instead of claiming "not yet modeled."
            _nc("How chips are valued (Rules #49)",
                "- Bench Boost/Triple Captain values switch to the Wildcard's own rebuild squad for any week at or "
                "after a scheduled Wildcard (Rule #49b) — verified in code, not assumed.\n"
                "- Free Hit's value is a full rebuild vs. your own best XI that week, and doesn't depend on squad "
                "path (Rule #49c).\n"
                "- Pre-Wildcard Bench Boost/Triple Captain values use a one-shot reachable-squad approximation per "
                "candidate week (Rule #49d) — not a full week-by-week chained transfer simulation, same "
                "performance tradeoff already disclosed for Rule #48's own baseline.\n"
                "- When a Wildcard and Bench Boost are scheduled together, the Wildcard card discloses that "
                "build's bench sum at the Bench Boost week (Rule #49f) — a disclosure only, it doesn't change any "
                "reported value.\n"
                "- Re-run every gameweek — this is a hypothesis, not a commitment (Rule #32).")
        elif chip_portfolio is not None:
            st.markdown("---")
            _nc("🗓️ No chip combo adds value — hold", "🗓️ Sequencing your available chips together finds no positive combined assignment this "
                       "run — holding all for now (Rule #32).")

    # Patch 89 (v6.9 Standing Rule #51, Cross-Tool Reconciliation Rule) —
    # confirmed via code read (2026-09-29) that nothing in this app
    # decomposed a disagreement with another tool's quoted figure before
    # this patch; the closest existing thing (the Wildcard what-if captions
    # above) only ever reports THIS app's own number, never compares it
    # input-by-input against a different tool's. Scoped to the Wildcard
    # window-value gain (Rule #48) — the doc's own worked origin example
    # for this rule is specifically a Wildcard-size disagreement, and
    # that's where this app already has the richest machinery
    # (wildcard_window_value_scan) to build a decomposition on. See
    # chip_protocol.reconcile_wildcard_gain() for exactly which of the
    # doc's six listed inputs are automated here (4 of 6) vs. disclosed as
    # not yet modeled (2 of 6 — formula variant, availability assumptions).
    with st.expander("🔍 Cross-Tool Reconciliation (Rule #51) — compare against another tool's Wildcard figure"):
        _nc("Compare another tool's gain", "Enter another tool's quoted Wildcard gain for a candidate week, and this decomposes the "
                   "difference from this app's own figure one input at a time — captain doubling, free-transfer "
                   "accrual, horizon length, and team value — instead of either number being accepted at face "
                   "value.")
        _rc_col1, _rc_col2, _rc_col3 = st.columns(3)
        with _rc_col1:
            _rc_gw = st.number_input("Candidate GW", min_value=int(planning_gw),
                                      max_value=int(planning_gw) + 37, value=int(planning_gw), step=1,
                                      key="reconcile_gw")
        with _rc_col2:
            _rc_other_gain = st.number_input("Other tool's quoted gain (xPts)", value=0.0, step=0.1,
                                              key="reconcile_other_gain")
        with _rc_col3:
            _rc_window_len = st.number_input("This app's window length (GWs)", min_value=1, max_value=10,
                                              value=int(cfg.get("chip_portfolio", {}).get("window_value_len", 4)),
                                              step=1, key="reconcile_window_len")
        _rc_adv_col1, _rc_adv_col2 = st.columns(2)
        with _rc_adv_col1:
            _rc_alt_window = st.number_input("Alt. window length to test (optional, 0 = skip)", min_value=0,
                                              max_value=10, value=0, step=1, key="reconcile_alt_window")
        with _rc_adv_col2:
            _rc_alt_budget = st.number_input("Alt. team value £m to test (optional, 0 = skip)", min_value=0.0,
                                              value=0.0, step=0.5, key="reconcile_alt_budget")
        if st.button("Reconcile", key="reconcile_run") and not squad_df.empty:
            with st.spinner("Reconciling against the current squad and pool — this re-solves several "
                             "squad variants and can take a few seconds..."):
                _rc_result = chip_protocol.reconcile_wildcard_gain(
                    squad_df_adv, pool_df_adv, cfg, ft["free_transfers"], bank, int(_rc_gw),
                    chip_adv_window["gw_list"] if chip_adv_window else gw_list, float(_rc_other_gain),
                    window_len=int(_rc_window_len),
                    alt_window_len=int(_rc_alt_window) if _rc_alt_window else None,
                    alt_budget=float(_rc_alt_budget) if _rc_alt_budget else None)
            if not _rc_result["feasible"]:
                st.info(_rc_result.get("reason", "Couldn't reconcile this run."))
            else:
                st.markdown(recommend.ui_chip(f"App {_rc_result['model_gap']:+.1f}", "info") + " "
                            + recommend.ui_chip(f"Other tool {_rc_result['other_gain']:+.1f}", "info") + " "
                            + recommend.ui_chip(f"Gap {_rc_result['gap_to_explain']:+.1f} xPts", "warn"),
                            unsafe_allow_html=True)
                if _rc_result["components"]:
                    _rc_rows = pd.DataFrame([
                        {"Input tested": c["input"], "Gain under that input": f"{c['variant_gap']:+.1f}",
                         "Share of the gap it explains": f"{c['delta']:+.1f}"}
                        for c in _rc_result["components"]])
                    _df(_rc_rows, hide_index=True, use_container_width=True)
                    for c in _rc_result["components"]:
                        _nc(str(c['input']), str(c['note']))
                st.markdown(recommend.ui_chip(f"Explained {_rc_result['explained_total']:+.1f}", "ok",
                                              tip="Rule #51: a difference that can't be reproduced is reported as unexplained, with its size, never guessed at.")
                            + " " + recommend.ui_chip(f"Unexplained {_rc_result['residual_unexplained']:+.1f}", "warn"),
                            unsafe_allow_html=True)
                with st.expander("Not yet automated (Rule #51 scope)"):
                    for item in _rc_result["not_modeled"]:
                        st.markdown(f"- {item}")

with tab_transfers:
    # ---------------------------------------------------------------------------
    # Team Recommendation — auto-built the moment any chip signal fires (Patch 31,
    # manager request: "if any chip strategy triggered i need a section for the
    # model full analysis and team recommendation"). No manual GW-picking step —
    # this reuses the exact same solves the manual "Evaluate a scenario" panel
    # below already offers, just triggered automatically and shown visually.
    # ---------------------------------------------------------------------------
    _bb_play = bb_advisor and bb_advisor["verdict"].startswith("play_gw")
    _tc_play = tc_advisor and tc_advisor["verdict"].startswith("play_gw")
    _fh_play = fh_advisor and fh_advisor["verdict"].startswith("play_gw")
    # Patch 112: ONE source for the chip weeks -- the Chip Plan's joint sequence (_final_chip_gw). The standalone
    # advisors scan each chip alone and could name a different week (Free Hit GW9 here vs GW13 in the Chip Plan).
    _adv_diff = []
    if _final_chip_gw:
        for _ck, _adv, _nm in (("bboost", bb_advisor, "Bench Boost"), ("3xc", tc_advisor, "Triple Captain"),
                               ("freehit", fh_advisor, "Free Hit")):
            _av = (_adv or {}).get("verdict", "") if _adv else ""
            if _ck in _final_chip_gw and _av.startswith("play_gw") and int(_av.split("gw")[1]) != _final_chip_gw[_ck]:
                _adv_diff.append(f"{_nm}: standalone scan GW{_av.split('gw')[1]}, Chip Plan GW{_final_chip_gw[_ck]}")
        _bb_play, _tc_play, _fh_play = ("bboost" in _final_chip_gw, "3xc" in _final_chip_gw, "freehit" in _final_chip_gw)
    _wc_active = bool(wc_flag)
    if _wc_active or _bb_play or _tc_play or _fh_play:
        st.markdown('<div class="section-h">🎯 Team Recommendation — active signals, auto-built</div>',
                    unsafe_allow_html=True)

        if _chip_headline_text:
            _nc("🔗 Same as Chip Plan + Pitch", "🔗 " + _chip_headline_text + " — same plan as the Chip Plan tab and the Pitch navigator.")
        if _adv_diff:
            _nc("🔗 Weeks follow Chip Plan", "🔗 Chip weeks follow the Chip Plan (joint sequence). Standalone scans differ — " + " · ".join(_adv_diff))
        if _wc_active:
            wc_rebuild_gw = detect_gw_list[0] if detect_gw_list else planning_gw
            _wc_col = f"xpts_gw{wc_rebuild_gw}"
            _full_pool_now = pd.concat([squad_df, pool_df], ignore_index=True, sort=False)
            if "code" in _full_pool_now.columns:
                _full_pool_now = _full_pool_now.drop_duplicates(subset=["code"], keep="first")
            wc_eval_auto = _wc_whatif_calc(
                squad_df, pool_df, cfg, team_value, tuple(detect_gw_list or gw_list))
            # Patch 112: the rebuild shown is the CHAIN's squad for the decision's Wildcard week (same data and
            # 8-GW window the decision used), so the squad you see is the one the Chip Plan scored. HOLD -> the
            # best-gain candidate is shown as a reference. No chain -> the old 'if played now' rebuild.
            _rb_t, _rb_codes, _rb_ref = None, None, False
            if _wc_chain and _wc_chain.get("cands"):
                _rb_t = (_wc_decision or {}).get("gw")
                if _rb_t not in _wc_chain["cands"]:
                    _rb_t = max(_wc_chain["cands"], key=lambda t: _wc_chain["cands"][t]["gain"])
                    _rb_ref = True
                _rb_codes = _wc_chain["cands"][_rb_t].get("rebuild_codes")
            if _rb_codes:
                wc_rebuild_gw = int(_rb_t)
                _wc_col = f"xpts_gw{wc_rebuild_gw}"
                _rb_sq = _full_pool_now[_full_pool_now["code"].isin(_rb_codes)]
                if len(_rb_sq) == 15 and _wc_col in _rb_sq.columns:
                    wc_eval_auto = dict(wc_eval_auto)
                    wc_eval_auto["feasible"] = True
                    wc_eval_auto["rebuild_squad"] = _rb_sq
            with st.expander(f"🃏 Wildcard rebuild — "
                             + (f"best week GW{wc_rebuild_gw}" if _rb_codes and not _rb_ref else
                                f"HOLD · reference GW{wc_rebuild_gw}" if _rb_codes else
                                f"trigger active, shown for GW{wc_rebuild_gw} onward"), expanded=False):
                _row_start()
                if not wc_eval_auto["feasible"]:
                    _ni("No auto-rebuild solved", "Couldn't solve an auto-rebuild this run (projection data may not reach far enough).")
                else:
                    if _rb_codes:
                        _cc = _wc_chain["cands"][_rb_t]
                        _nc("Same squad as Chip Plan", f"Same squad the Chip Plan scored for a Wildcard at GW{_rb_t} (built for GW{_rb_t}–"
                                   f"GW{_rb_t + 7}, current data). Chain gain {_cc['gain_chain']:+.1f} xPts"
                                   + (f" + chips {_cc['gain_chips']:+.1f}" if (_cc.get("chips") and _cc.get("chips_inside")) else "")
                                   + f" = {_cc['gain']:+.1f}."
                                   + (f" Chip effect beside it (not counted): {_cc['gain_chips']:+.1f}." if (_cc.get("chips") and not _cc.get("chips_inside")) else ""))
                        # Patch 114: say how it was built and which variant won
                        _bw_txt = cfg.get("transfer", {}).get("bench_weight_non_bb_gw", 0.08)
                        if _cc.get("objective") == "sum15":
                            _nc("15-man sum build", "Built on the plain 15-man sum (wc_objective = sum15, the Patch 113 setting): "
                                       "the bench counts like the starting XI.")
                        else:
                            _nc("Built week-by-week (8 GWs)", f"Built week by week: the squad and a legal starting XI for each of the 8 weeks are chosen together "
                                       f"(the XI counts in full, the bench at {_bw_txt:g}, the planner's own bench weight; each week's captain "
                                       f"is counted and nearer weeks weigh more, decay {float(cfg.get('chip_extended_check', {}).get('wc_build_decay', 0.9)):g} — Rule #54, estimate-tier), so a "
                                       f"bench player only earns his place by starting some weeks. The autosub curve is not "
                                       f"inside this build (it is applied when the squad is scored).")
                        _vt = _cc.get("variant_totals")
                        if _vt:
                            _bits = [f"{k} {v:+.1f}" for k, v in _vt.items() if v is not None]
                            _tg = _cc.get("tc_target_considered") or {}
                            _nc("Chip-aware check", "Chip-aware check (Bench Boost bench + Triple Captain target"
                                       + (f" {_tg.get('name')}, GW{_tg.get('gw')}" if _tg else "")
                                       + f"): {' · '.join(_bits)} xPts gain vs the no-chip chain (the plain figure is the "
                                       f"ranking WITHOUT the Triple Captain premium; a flip inside the "
                                       f"{_cc.get('variant_band', 0):.1f}-xPts band stays chip-agnostic). Using the "
                                       f"**{_cc.get('variant', 'plain')}** variant"
                                       + (" (plain stays unless another variant is clearly better, i.e. beyond the margin "
                                          "of error)." if _cc.get("variant") == "plain" else "."))
                    styled_wc = recommend.apply_style_to_wildcard_squad(
                        squad_df, wc_eval_auto["rebuild_squad"], _full_pool_now, style_name, cfg, _wc_col)
                    _style_swaps, _style_cost = [], 0.0
                    _win_cols = [f"xpts_gw{g}" for g in range(wc_rebuild_gw, wc_rebuild_gw + 8)
                                 if f"xpts_gw{g}" in _full_pool_now.columns]
                    if _rb_codes:
                        # Patch 112: the Wildcard is decided and SCORED plain; the style profile is shown as a list
                        # of swaps with their cost, never silently applied to the squad that was scored.
                        _style_swaps, _style_cost = recommend.style_swap_overlay(
                            wc_eval_auto["rebuild_squad"], styled_wc, _win_cols)
                        styled_wc = wc_eval_auto["rebuild_squad"].copy()
                    if _win_cols:
                        styled_wc = styled_wc.copy()
                        styled_wc["_win_sum"] = styled_wc[_win_cols].sum(axis=1, skipna=True)
                    gap = wc_eval_auto["gap"]
                    # Patch 89 (Rule #50) — this figure is against HOLDING
                    # the squad, not the best no-chip transfer path Rule #50
                    # requires for a decision; the shared gain_disclosure()
                    # formatter states that plainly plus horizon/variant/
                    # tier, instead of this caption's own ad hoc wording.
                    if _rb_codes:
                        _nc("Date = your call", "Date remains your own call (Standing Rule #32).")
                    else:
                      _nc("Rebuild vs hold", f"Rebuild projects {wc_eval_auto['rebuild_total']:.1f} xPts vs "
                               f"{wc_eval_auto['hold_total']:.1f} xPts holding your current squad over this window "
                               f"({gap:+.1f} xPts) "
                               f"{chip_protocol.gain_disclosure('hold_squad', detect_gw_list or gw_list, cfg)}. "
                               f"Style profile **{style_name}** applied. Date remains your own call (Standing "
                               f"Rule #32) — this is the model's current best rebuild if played now.")
                    if _wc_col in styled_wc.columns:
                        xi_res = opt.best_starting_xi(styled_wc, _wc_col)
                        cols = ["web_name", "team", "position", "price", _wc_col] + (["_win_sum"] if "_win_sum" in styled_wc.columns else [])
                        rn = {"web_name": "Player", "team": "Team", "position": "Pos", "price": "£m",
                              _wc_col: f"xPts GW{wc_rebuild_gw}",
                              "_win_sum": f"xPts GW{wc_rebuild_gw}–GW{wc_rebuild_gw + 7}"}
                        if xi_res is not None:
                            xi_df, wc_bench = xi_res["xi"], styled_wc[~styled_wc["code"].isin(xi_res["xi"]["code"])]
                            d, m, f = xi_res["shape"]
                            if locked_codes:
                                _lc = _lock_cost_calc(_full_pool_now.drop(columns=["_locked"], errors="ignore"), styled_wc, cfg,
                                                      float(team_value), tuple(range(wc_rebuild_gw, wc_rebuild_gw + 8)))
                                _lk_names = ", ".join(locked_names)
                                _row_add(recommend.ui_chip(f"🔒 {_lk_names}", "info",
                                                           None if _lc is None else f"locks cost {_lc:.1f} xPts"))
                            _row_end()
                            st.markdown(f"**Starting XI** (1-{d}-{m}-{f}); 15-man squad total £{styled_wc['price'].sum():.1f}m")
                            _df(xi_df.sort_values(["position", _wc_col], ascending=[True, False])[cols]
                                         .rename(columns=rn), hide_index=True, use_container_width=True,
                                         height=_tbl_h(len(xi_df)), column_config=_xp_colcfg(rn))
                            if not xi_df.empty:
                                cap = xi_df.sort_values(_wc_col, ascending=False).iloc[0]
                                st.caption(f"Suggested captain: **{cap['web_name']}** ({cap[_wc_col]:.1f} xPts).")
                            st.markdown("**Bench**")
                            _df(wc_bench.sort_values(["position", _wc_col], ascending=[True, False])[cols]
                                         .rename(columns=rn), hide_index=True, use_container_width=True,
                                         height=_tbl_h(len(wc_bench)), column_config=_xp_colcfg(rn))
                            if _rb_codes:
                                if _style_swaps:
                                    st.markdown(f"**{style_name} style swaps** (not applied; total cost "
                                                f"{_style_cost:+.1f} xPts over GW{wc_rebuild_gw}–GW{wc_rebuild_gw + 7})")
                                    _df(pd.DataFrame([{"Out": x["out"], "In (style)": x["in"], "Cost (xPts)": x["cost"]}
                                                               for x in _style_swaps]), hide_index=True, use_container_width=True)
                                else:
                                    _nc("No style swaps", f"{style_name}: no style swaps — the plain squad is also the styled squad.")
                                _tl_wc = recommend.plan_timeline_lines(
                                    _wc_chain["cands"][_rb_t]["plan"],
                                    first_week=(None if _rb_t == planning_gw else
                                                (_wc_first_week + (_wc_chain.get("first_ft_after"),) if _wc_first_week else None)))
                                if _tl_wc:
                                    st.markdown(f"📅 **Plan after the Wildcard at GW{_rb_t}** (free transfers only)")
                                    st.code("\n".join(_tl_wc), language=None)
            _row_end()

        if _fh_play:
            _fh_gw = _final_chip_gw["freehit"] if "freehit" in _final_chip_gw else int(fh_advisor["verdict"].split("gw")[1])
            _fh_col = f"xpts_gw{_fh_gw}"
            _fh_proj_auto = proj if _fh_col in proj.columns else _project(snap, hist_df, overrides, cfg, [_fh_gw], fixture_baselines)
            fh_res_auto = _fh_optimal_calc(cfg, _fh_proj_auto, team_value, _fh_gw)
            with st.expander(f"🎟️ Free Hit — PLAY GW{_fh_gw}, optimal squad", expanded=False):
                if fh_res_auto is None:
                    _ni("No Free Hit squad solved", "Couldn't solve an optimal Free Hit squad this run.")
                else:
                    fh_sq, fh_xi = fh_res_auto["squad"], None
                    fh_xi = fh_sq[fh_sq["code"].isin(fh_res_auto["xi_codes"])]
                    fh_bn = fh_sq[~fh_sq["code"].isin(fh_res_auto["xi_codes"])]
                    d, m, f = fh_res_auto["shape"]
                    cols = ["web_name", "team", "position", "price", _fh_col]
                    rn = {"web_name": "Player", "team": "Team", "position": "Pos", "price": "£m",
                          _fh_col: f"xPts GW{_fh_gw}"}
                    st.markdown(f"**Starting XI** (1-{d}-{m}-{f}, £{fh_xi['price'].sum():.1f}m XI, "
                                f"£{fh_res_auto['total_cost']:.1f}m of £{team_value:.1f}m)")
                    _df(fh_xi.sort_values(["position", _fh_col], ascending=[True, False])[cols]
                                 .rename(columns=rn), hide_index=True, use_container_width=True,
                                 height=_tbl_h(len(fh_xi)))
                    if not fh_xi.empty:
                        cap = fh_xi.sort_values(_fh_col, ascending=False).iloc[0]
                        st.caption(f"Suggested captain: **{cap['web_name']}** ({cap[_fh_col]:.1f} xPts).")
                    st.markdown("**Bench** (cheap by design — budget routed to the XI)")
                    _df(fh_bn.sort_values(["position", _fh_col], ascending=[True, False])[cols]
                                 .rename(columns=rn), hide_index=True, use_container_width=True,
                                 height=_tbl_h(len(fh_bn)))

        def _chip_src(gw):
            """Patch 112: the squad the plan fields in `gw` (Wildcard rebuild + later transfers once the Wildcard is
            played); falls back to the current squad only when no plan path exists."""
            sq = _path_squad_at(gw)
            if sq is not None and len(sq) >= 15:
                _wcg = (_wc_decision or {}).get("gw")
                lab = (f"Wildcard squad (GW{_wcg}) + later transfers" if (_wcg is not None and gw >= _wcg)
                       else "your squad on the planned transfers")
                return sq, f"Squad shown: {lab}, as fielded in GW{gw}."
            return squad_df_adv, "Squad shown: your CURRENT squad (no plan path this run)."

        if _bb_play:
            _bb_gw = _final_chip_gw["bboost"] if "bboost" in _final_chip_gw else int(bb_advisor["verdict"].split("gw")[1])
            _bb_col = f"xpts_gw{_bb_gw}"
            with st.expander(f"🛋️ Bench Boost — PLAY GW{_bb_gw}, full 15", expanded=False):
                _bb_sq, _bb_lab = _chip_src(_bb_gw)
                if _bb_col in _bb_sq.columns:
                    cols = ["web_name", "team", "position", "price", "Role", _bb_col]
                    rn = {"web_name": "Player", "team": "Team", "position": "Pos", "price": "£m",
                          _bb_col: f"xPts GW{_bb_gw}"}
                    _bbx = opt.best_starting_xi(_bb_sq, _bb_col)
                    _bb_view = _bb_sq.copy()
                    _bb_view["Role"] = "XI"
                    if _bbx and _bbx.get("xi") is not None:
                        _bb_view.loc[~_bb_view["code"].isin(_bbx["xi"]["code"]), "Role"] = "Bench (boosted)"
                    _cc = (_chain_path or {}).get("chips") or {}
                    _bbv = _cc.get("bb")
                    _bb_same = bool(_bbv and _bbv[0] == _bb_gw)
                    _nc("Bench Boost", f"{_bb_lab} "
                               + (f"Bench adds {_bbv[1]:.1f} xPts. "
                                  + _edge_txt(_bbv, _cc.get("bb_best"), _cc.get("bb_gap"), "Bench Boost week",
                                              _cc.get("bb_conf") or "near-tie", _cc.get("bb_yield"),
                                              tie_set=_cc.get("bb_tie_set")) + " " if _bb_same else "")
                               + "Same weeks and squads as the Chip Plan.")
                    _df(_bb_view.sort_values(["Role", "position", _bb_col], ascending=[False, True, False])[cols]
                                 .rename(columns=rn), hide_index=True, use_container_width=True,
                                 height=_tbl_h(len(_bb_view)))

        if _tc_play:
            _tc_gw = _final_chip_gw["3xc"] if "3xc" in _final_chip_gw else int(tc_advisor["verdict"].split("gw")[1])
            _tc_col = f"xpts_gw{_tc_gw}"
            with st.expander(f"👑 Triple Captain — PLAY GW{_tc_gw}", expanded=False):
                _tc_sq, _tc_lab = _chip_src(_tc_gw)
                _tcx = opt.best_starting_xi(_tc_sq, _tc_col) if _tc_col in _tc_sq.columns else None
                if _tcx and _tcx.get("xi") is not None and not _tcx["xi"].empty:
                    cap_row = _tcx["xi"].sort_values(_tc_col, ascending=False).iloc[0]
                    _cc = (_chain_path or {}).get("chips") or {}
                    _tcv = _cc.get("tc")
                    _tcp = _cc.get("tc_player")                       # Patch 113: the one pick every screen reads
                    _tc_same = bool(_tcv and _tcv[0] == _tc_gw)
                    if _tc_same and _tcp and _tcp.get("gw") == _tc_gw and _tcp.get("name"):
                        cap_row = _tcx["xi"][_tcx["xi"]["code"] == _tcp["code"]].iloc[0] if (
                            _tcx["xi"]["code"] == _tcp["code"]).any() else cap_row
                    st.markdown(f"**{cap_row['web_name']}** ({cap_row.get('team','')}) — "
                                f"{cap_row[_tc_col]:.1f} xPts, tripled to {cap_row[_tc_col]*3:.1f}.")
                    _tgt = (_chain_path or {}).get("tc_target")
                    _buy = ""
                    if _tgt and _tgt.get("gw") == _tc_gw:
                        _bw = next((w["gw"] for w in ((_chain_path or {}).get("plan") or [])
                                    if any(m.get("in_code") == _tgt.get("code") for m in (w.get("moves") or []))), None)
                        _buy = (f"Planned for: {_tgt.get('name')} ({_tgt.get('xpts', 0):.1f} xPts in GW{_tc_gw}) is "
                                + (f"bought in GW{_bw}." if _bw is not None else "part of the Wildcard squad.") + " ")
                    _nc("Triple Captain", f"{_tc_lab} "
                               + (_edge_txt(_tcv, _cc.get("tc_best"), _cc.get("tc_gap"), "Triple Captain week",
                                            _cc.get("tc_conf") or "near-tie", _cc.get("tc_yield"),
                                            tie_set=_cc.get("tc_tie_set")) + " " if _tc_same else "")
                               + _buy + "Same week and squad as the Chip Plan.")

# ---------------------------------------------------------------------------
# Squad pitch + GW navigator (merged, manager report: "having 2 pitches like
# this is too much, i need only one on the above and build the navigator
# inside it"). One pitch section now: it opens on your planning GW with
# "Current squad" selected — visually identical to the pre-navigator pitch —
# and arrows/toggle at the top let it move across the WIDER Chip Advisor
# window (chip_adv_window), aligned with chip strategy, not just the
# narrower sidebar Horizon. Confirmed with the manager: every navigated GW
# shows a simple top-projected-scorer armband (not the full captaincy-
# protocol pick, which is only ever run for planning_gw).
#
# Wrapped in st.fragment (manager report: "the app performance is too slow"
# — Streamlit reruns the ENTIRE page on every widget interaction by default,
# and arrow clicks are clicked far more often than any other control; a
# fragment confines a rerun to just this section instead) and the per-GW
# "optimal squad" solve is cached (@st.cache_data) so revisiting a GW you've
# already viewed this session is instant instead of re-running a fresh MILP.
# Re-solves a fresh best_starting_xi() per (GW, toggle-state) pair rather
# than reusing planning_gw's XI/bench split for every displayed GW — that
# reuse was a real staleness bug (a GW several weeks out can have a totally
# different optimal XI once rotation/fixtures/doubles are accounted for).
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _nav_optimal_squad(_cfg, _proj, team_value, gw):
    return data_pipeline.solve_best_gw_squad(_cfg, _proj, team_value, gw)


# ---------------------------------------------------------------------------
# Evaluate your own scenario — compute/display separation (Patch 60, manager,
# 2026-09-21: "the point related to the scenario requires a model run after
# the scenario was calculated to enable the pitch navigator ... please fix
# that ... once the scenario was calculated the pitch navigation option
# should be enabled"). Real fix, not a disclosure caption this time. Root
# cause (confirmed in code): this whole block used to live entirely AFTER
# the Pitch Navigator's call site, compute-and-render fused together inside
# `if st.button(...)`, so a freshly-evaluated scenario was stored into
# session_state too late for the SAME script pass's navigator render to see
# it — and, as a second problem found while fixing this, that button-gated
# design ALSO meant the displayed results vanished the instant the manager
# touched any other widget on the page (the button's own True state resets
# after the one rerun that processed the click).
#
# Fix: COMPUTE is fully separated from DISPLAY. `_compute_scenario_
# evaluations()` below is defined here (above the Pitch Navigator) and
# CALLED immediately before `_render_pitch_navigator()`'s own call site
# further down, reading the 3 selector widgets' values straight from
# session_state (safe — Streamlit populates a `key=`-backed widget's
# session_state entry before the script body runs on a rerun, not as a side
# effect of reaching that widget's own call site further down the script),
# and — only when "Evaluate scenario" was actually just clicked
# (`scenario_evaluate_btn`, an explicit key) — stores every field the
# display needs into 3 session_state caches (`scenario_target_cache` /
# `scenario_wc_cache` / `scenario_fh_cache`), on top of the existing
# navigator-facing keys (`scenario_nav_moves`, `scenario_squad_wc`,
# `scenario_squad_fh`, etc.). The widgets themselves, and the actual
# `_render_scenario_results()` call, stay in their normal visual position
# further down the page (inside the "Evaluate your own scenario" expander),
# but the display now reads FROM those caches unconditionally (any rerun,
# not just the one right after the click) — so results also no longer
# disappear the moment the manager interacts with anything else on the page.
# ---------------------------------------------------------------------------
def _compute_scenario_evaluations():
    target_choice = st.session_state.get("scenario_target", (None, "— none —"))
    wc_gw_choice = st.session_state.get("scenario_wc_gw")
    fh_gw_choice = st.session_state.get("scenario_fh_gw")
    if not st.session_state.get("scenario_evaluate_btn"):
        return  # not clicked this run — whatever's already cached stays as-is

    if target_choice[0] is None and wc_gw_choice is None and fh_gw_choice is None:
        st.session_state["scenario_nothing_selected"] = True
        return
    st.session_state["scenario_nothing_selected"] = False

    if target_choice[0] is not None:
        target_eval = recommend.evaluate_target_transfer(
            squad_df, pool_df, cfg, style_name, hit_stance, ft["free_transfers"], bank,
            planning_gw, gw_list, target_choice[0], default_net_gain=rec.get("net_gain"),
            disrupted_codes=_disrupted_codes, bb_play_gw=_bb_play_gw)
        st.session_state["scenario_target_cache"] = {"label": target_choice[1], "eval": target_eval}
        if target_eval["moves"]:
            # Patch 40 — the Pitch Navigator's 3rd toggle for this scenario
            # (unchanged mechanism: out/in-code reconstruction against the
            # current squad).
            st.session_state["scenario_nav_moves"] = target_eval["moves"]
            st.session_state["scenario_nav_label"] = target_choice[1]
        else:
            st.session_state.pop("scenario_nav_moves", None)
            st.session_state.pop("scenario_nav_label", None)
    else:
        st.session_state.pop("scenario_target_cache", None)
        st.session_state.pop("scenario_nav_moves", None)
        st.session_state.pop("scenario_nav_label", None)

    if wc_gw_choice is not None:
        wc_horizon = max(3, horizon)
        future_gw_list = list(range(wc_gw_choice, wc_gw_choice + wc_horizon))
        future_proj = _project(snap, hist_df, overrides, cfg, future_gw_list, fixture_baselines)
        future_proj = recommend.apply_sell_prices(future_proj, _sp)      # Patch 117d: same pricing as the main run
        future_squad_proj = future_proj[future_proj["code"].isin(squad_codes)].copy()
        future_pool_proj = future_proj[~future_proj["code"].isin(squad_codes)].copy()
        wc_eval = _wc_whatif_calc(future_squad_proj, future_pool_proj, cfg,
                                   team_value, tuple(future_gw_list))
        if not wc_eval["feasible"]:
            st.session_state["scenario_wc_cache"] = {
                "wc_gw_choice": wc_gw_choice, "feasible": False}
            st.session_state.pop("scenario_squad_wc", None)
            st.session_state.pop("scenario_label_wc", None)
            st.session_state.pop("scenario_gw_list_wc", None)
        else:
            wc_gw_col = f"xpts_gw{wc_gw_choice}"
            full_pool_future = pd.concat([future_squad_proj, future_pool_proj], ignore_index=True, sort=False)
            if "code" in full_pool_future.columns:
                full_pool_future = full_pool_future.drop_duplicates(subset=["code"], keep="first")
            styled_squad = recommend.apply_style_to_wildcard_squad(
                future_squad_proj, wc_eval["rebuild_squad"], full_pool_future, style_name, cfg, wc_gw_col)
            # Patch 58/59 — Pitch Navigator's Wildcard-scenario toggle, its
            # own solved GW window. Now written BEFORE the navigator call
            # below, so it's live on this same render.
            st.session_state["scenario_squad_wc"] = styled_squad
            st.session_state["scenario_label_wc"] = f"GW{wc_gw_choice}"
            st.session_state["scenario_gw_list_wc"] = future_gw_list
            xi_result = opt.best_starting_xi(styled_squad, wc_gw_col) if wc_gw_col in styled_squad.columns else None
            st.session_state["scenario_wc_cache"] = {
                "wc_gw_choice": wc_gw_choice, "feasible": True, "wc_horizon": wc_horizon,
                "future_gw_list": future_gw_list, "gap": wc_eval["gap"], "rebuild_total": wc_eval["rebuild_total"],
                "hold_total": wc_eval["hold_total"], "wc_gw_col": wc_gw_col, "styled_squad": styled_squad,
                "xi_result": xi_result, "style_name": style_name,
            }
    else:
        st.session_state.pop("scenario_wc_cache", None)
        st.session_state.pop("scenario_squad_wc", None)
        st.session_state.pop("scenario_label_wc", None)
        st.session_state.pop("scenario_gw_list_wc", None)

    if fh_gw_choice is not None:
        fh_col = f"xpts_gw{fh_gw_choice}"
        fh_proj = _project(snap, hist_df, overrides, cfg, [fh_gw_choice], fixture_baselines)
        if fh_col not in fh_proj.columns:
            st.session_state["scenario_fh_cache"] = {"fh_gw_choice": fh_gw_choice, "feasible": False,
                                                       "reason": "no_projection"}
            st.session_state.pop("scenario_squad_fh", None)
            st.session_state.pop("scenario_label_fh", None)
            st.session_state.pop("scenario_gw_list_fh", None)
        else:
            fh_result = data_pipeline.solve_free_hit_optimal_squad(cfg, fh_proj, team_value, fh_gw_choice)
            if fh_result is None:
                st.session_state["scenario_fh_cache"] = {"fh_gw_choice": fh_gw_choice, "feasible": False,
                                                           "reason": "infeasible"}
                st.session_state.pop("scenario_squad_fh", None)
                st.session_state.pop("scenario_label_fh", None)
                st.session_state.pop("scenario_gw_list_fh", None)
            else:
                fh_squad = fh_result["squad"]
                st.session_state["scenario_squad_fh"] = fh_squad
                st.session_state["scenario_label_fh"] = f"GW{fh_gw_choice}"
                st.session_state["scenario_gw_list_fh"] = [fh_gw_choice]
                current_squad_at_fh_gw = fh_proj[fh_proj["code"].isin(squad_codes)]
                fh_current_val = opt.rating_gw_value(current_squad_at_fh_gw, fh_col, cfg)["total_realized"]
                fh_optimal_val = opt.rating_gw_value(fh_squad, fh_col, cfg)["total_realized"]
                fh_rating = eng.team_rating_pct(fh_current_val, fh_optimal_val, "")
                st.session_state["scenario_fh_cache"] = {
                    "fh_gw_choice": fh_gw_choice, "feasible": True, "fh_col": fh_col, "fh_squad": fh_squad,
                    "fh_result": fh_result, "team_value": team_value,
                    "fh_current_val": fh_current_val, "fh_optimal_val": fh_optimal_val,
                    "fh_gap": round(fh_optimal_val - fh_current_val, 2),
                    "fh_moe": eng.margin_of_error_threshold(fh_optimal_val, cfg),
                    "fh_rating": fh_rating,
                }
    else:
        st.session_state.pop("scenario_fh_cache", None)
        st.session_state.pop("scenario_squad_fh", None)
        st.session_state.pop("scenario_label_fh", None)
        st.session_state.pop("scenario_gw_list_fh", None)


def _render_scenario_results():
    """Renders whatever is currently cached (see _compute_scenario_
    evaluations() above) — unconditional on every rerun, not gated behind
    the button's one-shot True state, so results stay visible until the
    manager explicitly clears them or evaluates a different scenario."""
    if st.session_state.get("scenario_nothing_selected"):
        _ni("Pick a player / GW first", "Nothing selected — pick a target player, a Wildcard gameweek, and/or a Free Hit "
                "gameweek above first.")

    t_cache = st.session_state.get("scenario_target_cache")
    if t_cache:
        target_eval = t_cache["eval"]
        st.markdown("**Target player scenario**")
        if target_eval["summary"]:
            for line in target_eval["summary"]:
                st.markdown(f'<div class="tx-reco">🧪 {line}</div>', unsafe_allow_html=True)
        if target_eval["moves"]:
            _mv = pd.DataFrame(target_eval["moves"])
            _df(_mv[[c for c in ["out", "in", "position", "xpts_gain", "hit_cost", "net_gain", "justified"]
                              if c in _mv.columns]], hide_index=True, use_container_width=True)
        if target_eval.get("plan"):
            with st.expander("Why — full trace, rule references, and move-by-move detail"):
                for line in target_eval["plan"]:
                    st.markdown(f"- {line}")

    wc_cache = st.session_state.get("scenario_wc_cache")
    if wc_cache:
        wc_gw_choice = wc_cache["wc_gw_choice"]
        st.markdown(f"**Wildcard what-if — GW{wc_gw_choice}**")
        if not wc_cache["feasible"]:
            _ni(f"No rebuild solved for GW{wc_gw_choice}", f"Couldn't solve a rebuild for GW{wc_gw_choice} this run (projection data may not "
                    f"reach that far yet).")
        else:
            if wc_cache["wc_horizon"] != horizon:
                fgl = wc_cache["future_gw_list"]
                _nc(f"Window GW{fgl[0]}–{fgl[-1]}", f"Evaluated over GW{fgl[0]}–GW{fgl[-1]} ({wc_cache['wc_horizon']} GWs) — "
                           f"a 3-GW minimum applies to Wildcard rebuilds regardless of the sidebar horizon "
                           f"(currently {horizon} GW).")
            gap = wc_cache["gap"]
            # Patch 89 (Rule #50) — same "labelled as such, never the
            # decision basis" treatment as the auto-rebuild caption above:
            # this is a hold-squad baseline, not the best no-chip transfer
            # path.
            _dm(f"🧪 GW{wc_gw_choice}: rebuild {wc_cache['rebuild_total']:.1f} vs hold {wc_cache['hold_total']:.1f} ({wc_cache['gap']:+.1f} xPts)", f'<div class="tx-reco">🧪 If played at GW{wc_gw_choice}: a full rebuild projects '
                        f'{wc_cache["rebuild_total"]:.1f} xPts vs {wc_cache["hold_total"]:.1f} xPts holding your '
                        f'current squad, over the same {len(wc_cache["future_gw_list"])}-GW window ({gap:+.1f} '
                        f'xPts) {chip_protocol.gain_disclosure("hold_squad", wc_cache["future_gw_list"], cfg)}. '
                        f'Informational only — this candidate GW is your own choice, and the model never '
                        f'names a single "play" date (Standing Rule #32); see Chip Rack above for whether '
                        f'v6.4\'s own Wildcard trigger is currently active.</div>', unsafe_allow_html=True)
            _nc("📍 Also in Pitch Navigator", "📍 Also available in the Pitch Navigator above, right now — no extra click needed.")

            styled_squad = wc_cache["styled_squad"]
            wc_gw_col = wc_cache["wc_gw_col"]
            xi_result = wc_cache["xi_result"]
            show_cols = ["web_name", "team", "position", "price", wc_gw_col]
            col_rename = {"web_name": "Player", "team": "Team", "position": "Pos",
                          "price": "£m", wc_gw_col: f"xPts GW{wc_gw_choice}"}
            if xi_result is not None:
                xi_df = xi_result["xi"]
                wc_bench_df = styled_squad[~styled_squad["code"].isin(xi_df["code"])]
                d, m, f = xi_result["shape"]
                st.markdown(f"**Recommended Wildcard XI — GW{wc_gw_choice}** "
                            f"(formation 1-{d}-{m}-{f}, squad cost £{styled_squad['price'].sum():.1f}m)")
                xi_show = xi_df.sort_values(["position", wc_gw_col], ascending=[True, False])[show_cols] \
                    .rename(columns=col_rename)
                _df(xi_show, hide_index=True, use_container_width=True)
                if not xi_df.empty:
                    cap_row = xi_df.sort_values(wc_gw_col, ascending=False).iloc[0]
                    _nc(f"© {cap_row['web_name']} GW{wc_gw_choice} ({cap_row[wc_gw_col]:.1f})", f"Suggested captain for GW{wc_gw_choice}: **{cap_row['web_name']}** "
                               f"({cap_row[wc_gw_col]:.1f} projected xPts that week).")
                st.markdown("**Bench**")
                bench_show = wc_bench_df.sort_values(["position", wc_gw_col], ascending=[True, False])[show_cols] \
                    .rename(columns=col_rename)
                _df(bench_show, hide_index=True, use_container_width=True)
            else:
                st.markdown(f"**Recommended Wildcard squad — GW{wc_gw_choice}** (full 15)")
                full_show = styled_squad.sort_values(["position", wc_gw_col], ascending=[True, False])[show_cols] \
                    .rename(columns=col_rename) if wc_gw_col in styled_squad.columns else styled_squad
                _df(full_show, hide_index=True, use_container_width=True)
            _nc(f"Style: {wc_cache['style_name']}", f"Style profile **{wc_cache['style_name']}** applied to this rebuild (same EO-pull "
                       f"tie-break as ordinary transfers). Prices, injuries and fixtures can move before "
                       f"GW{wc_gw_choice} — re-run this closer to the date rather than treating it as locked in.")

    fh_cache = st.session_state.get("scenario_fh_cache")
    if fh_cache:
        fh_gw_choice = fh_cache["fh_gw_choice"]
        st.markdown(f"**Free Hit optimal squad — GW{fh_gw_choice}**")
        if not fh_cache["feasible"]:
            if fh_cache["reason"] == "no_projection":
                _ni(f"No projection for GW{fh_gw_choice} yet", f"No projection reaches GW{fh_gw_choice} yet this run — try a nearer gameweek.")
            else:
                _ni(f"No Free Hit squad solved for GW{fh_gw_choice}", f"Couldn't solve an optimal Free Hit squad for GW{fh_gw_choice} this run "
                        f"(projection data may not reach that far yet, or no feasible squad fit the "
                        f"budget/club constraints).")
        else:
            _nc("📍 Also in Pitch Navigator", "📍 Also available in the Pitch Navigator above, right now — no extra click needed.")
            fh_col = fh_cache["fh_col"]
            fh_squad = fh_cache["fh_squad"]
            fh_result = fh_cache["fh_result"]
            fh_xi = fh_squad[fh_squad["code"].isin(fh_result["xi_codes"])]
            fh_bench = fh_squad[~fh_squad["code"].isin(fh_result["xi_codes"])]
            d, m, f = fh_result["shape"]
            fh_show_cols = ["web_name", "team", "position", "price", fh_col]
            fh_col_rename = {"web_name": "Player", "team": "Team", "position": "Pos",
                              "price": "£m", fh_col: f"xPts GW{fh_gw_choice}"}
            st.markdown(f"Starting XI (formation 1-{d}-{m}-{f}, XI cost "
                        f"£{fh_xi['price'].sum():.1f}m, bench cost £{fh_result['bench_cost']:.1f}m, "
                        f"total £{fh_result['total_cost']:.1f}m of £{fh_cache['team_value']:.1f}m available)")
            fh_xi_show = fh_xi.sort_values(["position", fh_col], ascending=[True, False])[fh_show_cols] \
                .rename(columns=fh_col_rename)
            _df(fh_xi_show, hide_index=True, use_container_width=True)
            if not fh_xi.empty:
                fh_cap_row = fh_xi.sort_values(fh_col, ascending=False).iloc[0]
                _nc(f"© {fh_cap_row['web_name']} GW{fh_gw_choice} ({fh_cap_row[fh_col]:.1f})", f"Suggested captain for GW{fh_gw_choice}: **{fh_cap_row['web_name']}** "
                           f"({fh_cap_row[fh_col]:.1f} projected xPts that week).")
            st.markdown("**Bench**")
            _nc("Cheap by design", "A Free Hit's bench only matters if an autosub fires, so budget is routed to the XI above instead.")
            fh_bench_show = fh_bench.sort_values(["position", fh_col], ascending=[True, False])[fh_show_cols] \
                .rename(columns=fh_col_rename)
            _df(fh_bench_show, hide_index=True, use_container_width=True)
            _nc(f"Optimized for GW{fh_gw_choice} only", f"Optimized for GW{fh_gw_choice} only — re-run closer to the date",
                       help=f"Optimized for GW{fh_gw_choice} only (a Free Hit squad reverts after this "
                            f"gameweek, per Rule #25) — this is the model's single best squad for that "
                            f"week, not season-shaping, so no Style Profile differential pull is applied. "
                            f"Prices, injuries and fixtures can move before GW{fh_gw_choice} — re-run "
                            f"this closer to the date rather than treating it as locked in.")
            st.markdown("**Your squad vs. this Free Hit optimal**")
            if fh_cache["fh_rating"]["rating_pct"] is not None:
                st.markdown(recommend.rating_bar_html(fh_cache['fh_rating']['rating_pct'], f"Now {fh_cache['fh_current_val']:.1f} vs FH {fh_cache['fh_optimal_val']:.1f} xPts"), unsafe_allow_html=True)
                _nc(f"Gap {fh_cache['fh_gap']:.1f} (noise bar {fh_cache['fh_moe']:.1f})", f"Your current squad's best XI this GW: **{fh_cache['fh_current_val']:.1f} xPts** vs. "
                            f"Free Hit optimal: **{fh_cache['fh_optimal_val']:.1f} xPts** → "
                            f"**{fh_cache['fh_rating']['rating_pct']}%** (gap: {fh_cache['fh_gap']:.1f} xPts, "
                            f"margin-of-error threshold: {fh_cache['fh_moe']:.1f} xPts)")
                if fh_cache["fh_gap"] < fh_cache["fh_moe"]:
                    _nc("✓ Gap = noise — FH upside small", "✓ That gap is inside normal weekly noise — your squad is already "
                               "effectively at this week's ceiling; a Free Hit's upside here is limited.")
            else:
                _ni("No valid XI to compare", "Couldn't compute a comparison — your current squad has no valid XI for this GW this run.")


@st.fragment
def _render_pitch_navigator():
    st.markdown(f'<div class="section-h">Squad · planning for GW{planning_gw}</div>', unsafe_allow_html=True)
    # Patch 83 (v6.9 §10 output-format addition: "Every run states the
    # attack-adjustment strength and the tier that fed it") -- one compact,
    # always-visible disclosure line per run. fixture_baselines/_fb_warn are
    # computed once near the top of this run (see the _fixture_baselines()
    # call right after snap loads) and reused everywhere xPts is computed
    # this run (Rule #22), so this one line accurately describes every
    # number on the page, not just this tile.
    _fa_cfg = cfg.get("fixture_adjustment", {})
    _n_cs_override = int(pd.to_numeric(overrides["cs_pct_override"], errors="coerce").notna().sum()) \
        if (not overrides.empty and "cs_pct_override" in overrides.columns) else 0
    if _fa_cfg.get("enabled", True) and fixture_baselines.get("by_id"):
        _fa_n = fixture_baselines.get("n_teams_with_data", 0)
        _tiers = [recommend.ui_chip(f"Attack: team strength {_fa_cfg.get('strength', 0.6):.1f}", "info",
                                    tip="Fixture-adjusted attack, team-strength tier (Rule #46)"),
                  recommend.ui_chip(f"{_fa_n}/20 teams with xG", "info", tip="Teams with real match-xG data this season"),
                  recommend.ui_chip("CS: team-strength tier", "info", tip="Clean sheets: fallback tier (Rule #47), no market-odds source wired")]
        if _n_cs_override:
            _tiers.append(recommend.ui_chip(f"Sheet CS: {_n_cs_override}", "warn", tip=recommend.cs_carry_label(_n_cs_override).strip()))
        st.markdown(" ".join(_tiers), unsafe_allow_html=True)
    else:
        _nc("⚙️ Fixture adjust: none", "⚙️ Fixture-adjusted attack: **no adjustment this run** (Rule #46) — "
                   f"{opt.get_diagnostic('fixture_adjustment') or 'no finished-match data yet.'}")
    if starters_df.empty:
        _nw("No squad data yet", "No squad data returned for this team ID / gameweek yet (common right after a deadline, or if "
                   "this is a brand-new team). Transfer targets and captaincy below still use the full player pool.")
        return

    _nav_gw_list = chip_adv_window["gw_list"] if chip_adv_window else gw_list
    if not _nav_gw_list:
        _nav_gw_list = [planning_gw]

    if "nav_gw_idx" not in st.session_state or st.session_state.get("nav_gw_list") != _nav_gw_list:
        st.session_state.nav_gw_idx = _nav_gw_list.index(planning_gw) if planning_gw in _nav_gw_list else 0
        st.session_state.nav_gw_list = _nav_gw_list
    st.session_state.nav_gw_idx = max(0, min(st.session_state.nav_gw_idx, len(_nav_gw_list) - 1))

    def _apply_moves(base_squad: pd.DataFrame, moves: list) -> pd.DataFrame:
        """Applies one week's out/in moves to a squad DataFrame, pulling the
        incoming player's row from `chip_adv_proj` (the full projection pool
        for this horizon window) exactly like the pre-Patch-84 single-move
        reconstruction did. Returns `base_squad` unchanged if `moves` is
        empty or malformed (no out_code/in_code columns)."""
        if not moves:
            return base_squad
        moves_df = pd.DataFrame(moves)
        if "out_code" not in moves_df.columns or "in_code" not in moves_df.columns:
            return base_squad
        out_codes = set(moves_df["out_code"])
        in_codes = set(moves_df["in_code"])
        result = pd.concat([
            base_squad[~base_squad["code"].isin(out_codes)],
            chip_adv_proj[chip_adv_proj["code"].isin(in_codes)],
        ], ignore_index=True, sort=False)
        if "code" in result.columns:
            result = result.drop_duplicates(subset=["code"], keep="first")
        return result

    # Patch 84 (manager screenshot: paged the navigator to GW7 with "After
    # recommended transfer (this week's move)" selected — the pitch still
    # showed Gomez despite the Transfer Recommendations panel, for the same
    # chained pacing plan, saying Gomez -> Groß lands in GW7. Root cause,
    # confirmed by reading this function as it stood before this patch: the
    # toggle only ever reconstructed `weekly_plan[0]`'s move (this week's
    # move at `planning_gw`) into a single static `_nav_squad_after`, and
    # every later GW the ◀▶ stepper pages to reused that SAME static squad —
    # only the projection column (`xpts_gw{nav_gw}`) changed, never the
    # squad's actual player composition. The prior comment here explicitly
    # scoped this to week 0 on purpose ("later weeks' hypothetical chained
    # moves stay out of scope for this preview") — that was a deliberate
    # simplification, not an oversight, but the manager has now asked for
    # the pitch to auto-update across the full horizon, not just GW1 of the
    # plan, so this replaces that scope with a per-GW CUMULATIVE
    # reconstruction: for the chained weekly pacing plan, `weekly_plan`'s
    # weeks are applied to the squad IN ORDER (each week's `sim_squad` in
    # recommend.py already chains off the previous week's, so replaying the
    # same out/in moves in the same order here reproduces that exact chain),
    # and `_nav_squad_after_by_gw[g]` holds the squad as it would stand once
    # every move up to and including week `g` has been made. The pitch's GW
    # stepper (`nav_gw`, below) then looks up the entry for the latest
    # planned week at or before whatever GW is currently being viewed, so
    # paging from GW6 to GW7 now genuinely reflects GW7's Gomez -> Groß move
    # stacked on top of GW6's Palmer -> Saka move, not a frozen GW6 squad.
    # A week with no move (e.g. a "Roll" week) simply carries the prior
    # week's squad forward unchanged, which is correct: no transfer that
    # week means no squad change that week.
    _nav_squad_after = None
    _nav_squad_after_by_gw = {}
    if rec.get("is_weekly_schedule"):
        # Patch 94: this per-GW chained reconstruction now lives in
        # recommend.build_squad_after_by_gw() — the exact same logic that
        # used to be this function's own private `_apply_moves` + loop,
        # extracted so the Wildcard cross-check note (app.py, computed much
        # earlier in the script than this function is even called) can
        # share it instead of using its own, buggy flat reconstruction. No
        # behavior change here: same chaining, same pool, same result.
        _wk_plan = rec.get("weekly_plan") or []
        _nav_squad_after_by_gw = recommend.build_squad_after_by_gw(squad_df_adv, _wk_plan, chip_adv_proj)
        _any_weekly_moves = any(_wk.get("moves") for _wk in _wk_plan)
        _nav_can_toggle = _any_weekly_moves
        # Kept for anything downstream still expecting "this week's move"
        # specifically (e.g. the pre-Patch-84 single-move fallback path).
        _this_week_moves = _wk_plan[0]["moves"] if _wk_plan and _wk_plan[0].get("gw") == planning_gw else []
        _nav_moves_df = pd.DataFrame(_this_week_moves) if _this_week_moves else pd.DataFrame()
    else:
        _nav_moves_df = pd.DataFrame(rec["moves"]) if rec.get("moves") else pd.DataFrame()
        _nav_can_toggle = (not _nav_moves_df.empty
                           and "out_code" in _nav_moves_df.columns and "in_code" in _nav_moves_df.columns)
        if _nav_can_toggle:
            _nav_squad_after = _apply_moves(squad_df_adv, _nav_moves_df.to_dict("records"))

    # Patch 40 (manager, 2026-09-14: "the navigator can have a 3rd option to
    # read from the scenarios on the section for 'evaluate the scenario'") —
    # a 3rd toggle, built the exact same out_code/in_code reconstruction way
    # as "After recommended transfer" above, but sourced from whatever the
    # manager last evaluated in "Evaluate your own scenario" (a manager-
    # named target, e.g. Tavernier) rather than the model's own default pick.
    # Only offered when that scenario's out-players are still actually in
    # the current squad — a stale scenario from a squad that's since changed
    # (a real transfer made, a new GW loaded) is silently unavailable rather
    # than previewing a squad that no longer makes sense, same caution as
    # every other "as-if" preview on this page.
    _scenario_moves = st.session_state.get("scenario_nav_moves")
    _nav_squad_scenario = None
    _scenario_label = st.session_state.get("scenario_nav_label", "scenario")
    if _scenario_moves:
        _scen_moves_df = pd.DataFrame(_scenario_moves)
        if "out_code" in _scen_moves_df.columns and "in_code" in _scen_moves_df.columns:
            _scen_out_codes = set(_scen_moves_df["out_code"])
            _scen_in_codes = set(_scen_moves_df["in_code"])
            if _scen_out_codes.issubset(set(squad_df_adv["code"])):
                _nav_squad_scenario = pd.concat([
                    squad_df_adv[~squad_df_adv["code"].isin(_scen_out_codes)],
                    chip_adv_proj[chip_adv_proj["code"].isin(_scen_in_codes)],
                ], ignore_index=True, sort=False)
                if "code" in _nav_squad_scenario.columns:
                    _nav_squad_scenario = _nav_squad_scenario.drop_duplicates(subset=["code"], keep="first")

    # Patch 58 (manager: "the team navigation should have an option to
    # navigate the new team evaluated scenario 'Player, Wildcard or FH'") —
    # the single-player target scenario above (`_nav_squad_scenario`) was
    # already wired in by Patch 40, via an out/in-code reconstruction
    # against the current squad. That reconstruction doesn't apply to a
    # Wildcard/Free Hit scenario below — those are full rebuilds with no
    # "out/in legs" against the current 15, just a whole new squad — so
    # those two blocks (further down this script) stash their ALREADY-BUILT
    # squad DataFrame directly in session_state instead, each tagged with
    # its own solved GW window (a Wildcard what-if solves its own 3+ GW
    # horizon; a Free Hit squad is single-GW only, Rule #25), so paging
    # covers whatever GWs THAT scenario actually solved for.
    _nav_scenario_sources = {}
    if _nav_squad_scenario is not None:
        _nav_scenario_sources[f"After evaluated scenario ({_scenario_label})"] = {
            "squad": _nav_squad_scenario, "gw_list": None}
    _wc_scen_squad = st.session_state.get("scenario_squad_wc")
    if _wc_scen_squad is not None:
        _nav_scenario_sources[f"Wildcard scenario ({st.session_state.get('scenario_label_wc', '')})"] = {
            "squad": _wc_scen_squad, "gw_list": st.session_state.get("scenario_gw_list_wc")}
    _fh_scen_squad = st.session_state.get("scenario_squad_fh")
    if _fh_scen_squad is not None:
        _nav_scenario_sources[f"Free Hit scenario ({st.session_state.get('scenario_label_fh', '')})"] = {
            "squad": _fh_scen_squad, "gw_list": st.session_state.get("scenario_gw_list_fh")}

    _nav_options = ["Current squad"]
    # Patch 84: label no longer says "(this week's move)" for the chained
    # weekly plan -- it now shows the CUMULATIVE squad through whichever GW
    # the stepper is on (see _nav_squad_after_by_gw above), not just week 1.
    # Deliberately doesn't embed the current nav_gw in the label text itself
    # (e.g. "through GW7") -- that value changes every time the ◀▶ stepper
    # is clicked, and Streamlit's st.radio keys off the option label, so an
    # option whose text changes out from under it would silently reset the
    # radio back to "Current squad" on every single GW-step click.
    _after_tx_label = ("After recommended transfers (chained plan)" if rec.get("is_weekly_schedule")
                        else "After recommended transfer")
    if _nav_can_toggle:
        _nav_options.append(_after_tx_label)
    _nav_options.extend(_nav_scenario_sources.keys())
    # Patch 112: the Chip Plan's own squads (Wildcard rebuild + later transfers) as a navigator mode, so the Pitch,
    # the Chip Plan and the Transfer page show the same squad in each GW.
    _nav_chain_by_gw = {}
    _chain_label = None
    if _chain_path and _chain_path.get("squads"):
        _cwg = (_wc_decision or {}).get("gw")
        _chain_label = f"Chip Plan path (Wildcard GW{_cwg})" if _cwg is not None else "Chip Plan path (no Wildcard)"
        _nav_options.append(_chain_label)

    # Dynamic options (both the 2nd option's wording and the 3rd option's
    # label can change run to run) — a stale session_state value that no
    # longer matches any current option would otherwise raise a Streamlit
    # exception on the widget below, so reset it defensively rather than
    # let a changed label crash the whole page.
    if st.session_state.get("nav_mode") not in _nav_options:
        st.session_state["nav_mode"] = "Current squad"
    # Patch 117d: Wildcard decided for THIS week -> open on the Chip Plan path (the plan to follow), once per GW.
    if _chain_label and (_wc_decision or {}).get("gw") == planning_gw and not st.session_state.get(f"_nav_default_{planning_gw}"):
        st.session_state["nav_mode"] = _chain_label
        st.session_state[f"_nav_default_{planning_gw}"] = True

    nav_c1, nav_c2 = st.columns([2, 2])
    with nav_c1:
        nav_mode = st.radio("Squad", _nav_options,
                             index=0, horizontal=True, key="nav_mode",
                             disabled=len(_nav_options) == 1,
                             help=None if len(_nav_options) > 1 else
                             "No recommended transfer this run to preview for the current planning GW, "
                             "and no scenario evaluated below yet.")

    # Patch 58 — a Wildcard/Free Hit scenario source (see _nav_scenario_
    # sources above) can carry its OWN solved GW window, different from the
    # default `_nav_gw_list` (current squad / recommended-transfer window):
    # a Wildcard what-if solves a 3+ GW horizon starting at its own
    # candidate GW, and a Free Hit squad only ever has one GW's projections
    # (Rule #25). Falls back to the shared `_nav_gw_list` for "Current
    # squad"/"After recommended transfer" and for the player-target scenario
    # (which reuses the current squad's own window, gw_list=None above).
    _active_src = _nav_scenario_sources.get(nav_mode)
    _active_nav_gw_list = (_active_src["gw_list"] if _active_src and _active_src.get("gw_list") else None) \
        or _nav_gw_list
    if st.session_state.get("nav_active_gw_list") != _active_nav_gw_list:
        st.session_state.nav_gw_idx = _active_nav_gw_list.index(planning_gw) if planning_gw in _active_nav_gw_list \
            else 0
        st.session_state.nav_active_gw_list = _active_nav_gw_list
    st.session_state.nav_gw_idx = max(0, min(st.session_state.nav_gw_idx, len(_active_nav_gw_list) - 1))

    with nav_c2:
        # Patch 84 (found while verifying the transfer-recommendation fix
        # above, not reported by the manager, but confirmed live via
        # Playwright: clicking ▶ showed the RIGHT GW's xPts/Rating/squad
        # below, but the counter label between the two buttons stayed ONE
        # CLICK BEHIND, e.g. reading "GW6 (1/3)" while the metrics under it
        # already said "GW7 xPts"/"GW7 Rating"). Root cause: the counter
        # (originally the middle `with pb2:` block) was rendered BETWEEN the
        # ◀ button's handler (which mutates nav_gw_idx before the counter
        # renders) and the ▶ button's handler (which mutated it AFTER the
        # counter had already rendered) -- so a ◀ click was reflected
        # immediately but a ▶ click only showed up on the NEXT rerun. Fixed
        # by resolving both buttons' clicks first, then rendering the
        # counter once nav_gw_idx is final for this run -- symmetric for
        # both directions.
        pb1, pb2, pb3 = st.columns([1, 3, 1])
        with pb1:
            _prev_clicked = st.button("◀", key="nav_prev", disabled=st.session_state.nav_gw_idx == 0)
        with pb3:
            _next_clicked = st.button("▶", key="nav_next",
                                       disabled=st.session_state.nav_gw_idx == len(_active_nav_gw_list) - 1)
        if _prev_clicked:
            st.session_state.nav_gw_idx -= 1
        if _next_clicked:
            st.session_state.nav_gw_idx += 1
        with pb2:
            st.markdown(f'<div style="text-align:center;font-weight:600;padding-top:0.4rem;color:var(--ink);">'
                        f'GW{_active_nav_gw_list[st.session_state.nav_gw_idx]} '
                        f'({st.session_state.nav_gw_idx + 1}/{len(_active_nav_gw_list)})</div>',
                        unsafe_allow_html=True)

    nav_gw = _active_nav_gw_list[st.session_state.nav_gw_idx]
    nav_col = f"xpts_gw{nav_gw}"
    if _active_src is not None:
        nav_squad = _active_src["squad"]
    elif _chain_label is not None and nav_mode == _chain_label:
        nav_squad = _path_squad_at(nav_gw)
        if nav_squad is None:
            _prior = [g for g in (_chain_path or {}).get("squads", {}) if g <= nav_gw]
            nav_squad = _path_squad_at(max(_prior)) if _prior else None
        if nav_squad is None:
            nav_squad = squad_df_adv
    elif nav_mode.startswith("After recommended transfer") and _nav_squad_after_by_gw:
        # Patch 84: cumulative-through-this-GW lookup -- the latest planned
        # week at or before `nav_gw` (weekly_plan only ever covers
        # planning_gw onward, so a nav_gw before the first planned week
        # can't happen via the stepper, but the empty-list fallback to
        # squad_df_adv below stays defensive rather than assuming that).
        _applicable_gws = [g for g in _nav_squad_after_by_gw if g is not None and g <= nav_gw]
        nav_squad = _nav_squad_after_by_gw[max(_applicable_gws)] if _applicable_gws else squad_df_adv
    elif nav_mode.startswith("After recommended transfer") and _nav_squad_after is not None:
        nav_squad = _nav_squad_after
    else:
        nav_squad = squad_df_adv
    at_planning_gw = (nav_gw == planning_gw and nav_mode == "Current squad")

    if nav_col not in nav_squad.columns:
        st.caption(f"No projection data for GW{nav_gw} this run.")
        return
    nav_xi_result = opt.best_starting_xi(nav_squad, nav_col)
    if nav_xi_result is None:
        _nc(f"No valid XI GW{nav_gw} (blank?)", f"Couldn't solve a valid starting XI for GW{nav_gw} (common for a genuine blank gameweek).")
        return

    nav_starters = nav_xi_result["xi"]
    nav_bench = nav_squad[~nav_squad["code"].isin(nav_starters["code"])]
    nav_gw_xpts = round(nav_xi_result["total"], 1)
    # Patch 112: chip badge — the pitch shows which chip the Chip Plan assigns to the GW being browsed.
    _chips_here = [chip_protocol.CHIP_LABELS.get(k, k) for k, g in _final_chip_gw.items() if g == nav_gw]
    if (_wc_decision or {}).get("gw") == nav_gw:
        _chips_here.insert(0, "Wildcard")
    if _chips_here:
        _nc(f"🎴 Chip Plan GW{nav_gw}: {' + '.join(_chips_here)}", "🎴 Chip Plan for GW" + str(nav_gw) + ": **" + " + ".join(_chips_here) + "**"
                   + (" — switch the Squad view to the Chip Plan path to see the squad it fields."
                      if (_chain_label and nav_mode != _chain_label) else ""))

    # Header tiles — same mechanic as the main Team Rating % (Patch 20/24):
    # current squad's best XI (captain doubled, bench autosub-discounted)
    # over a genuinely unconstrained optimal squad for THIS GW specifically.
    # Cached (see _nav_optimal_squad above) so revisiting a GW is instant.
    nav_optimal_result = _nav_optimal_squad(cfg, chip_adv_proj, team_value, nav_gw)
    nav_current_val = opt.rating_gw_value(nav_squad, nav_col, cfg)["total_realized"]
    nav_optimal_val = opt.rating_gw_value(nav_optimal_result["squad"], nav_col, cfg)["total_realized"] \
        if nav_optimal_result else 0.0
    nav_rating = eng.team_rating_pct_capped(nav_current_val, nav_optimal_val, "")

    # Patch 51 (2026-09-16) -- relabeled from "Team Rating % (GW{n})" to avoid
    # colliding, in name only, with the new §1a-compliant "Team Rating %
    # (GW{start}-{end})" badge added this patch (a fixed multi-GW, full-pool
    # metric -- a completely different calculation from this navigator tile,
    # which is single-GW and scoped to whatever squad/GW you're browsing
    # here). This tile is the SAME calculation the manager already saw
    # produce a real, confusing near-miss against a different metric in
    # Patch 49 ("where the points we discussed!!!!") -- renaming it here is
    # a proactive extension of that same fix, not a new bug report. No
    # change to its underlying math (still current-squad-best-XI over a
    # genuinely unconstrained single-GW optimal squad) -- label/disclosure
    # only, per Standing Rules #16/#18.
    nt1, nt2 = st.columns(2)
    with nt1:
        st.metric(f"GW{nav_gw} xPts (best XI)", f"{nav_gw_xpts:.1f}")
    with nt2:
        st.metric(f"GW{nav_gw} Rating",
                  f"{nav_rating['rating_pct']}%" if nav_rating["rating_pct"] is not None else "—",
                  help=f"Single-GW, EST -- your best XI for GW{nav_gw} (captain doubled, bench autosub-"
                       f"discounted) over a fully unconstrained optimal squad for that GW alone. NOT the "
                       f"doc's §1a Team Rating % (that one needs a fixed 3-4 GW horizon) -- see the header's "
                       f"'Team Rating % (GW{compliant_gw_start}-{compliant_gw_end})' badge for that metric.")
    # Patch 75 (2026-09-27, found during the manager's own "test it from your
    # end before i deploy" request -- verified via a real Playwright render of
    # this exact tile, not inferred): this tile's "GW{n} Rating" is a THIRD,
    # separate render site sharing the exact same underlying solve as the
    # Latest News tab's "GW{n} Rating" card that Patch 74 already fixed --
    # nav_optimal_result above is data_pipeline.solve_free_hit_optimal_squad()
    # (see _nav_optimal_squad, same function fh_auto_result calls), which
    # already calls opt.set_diagnostic("free_hit_optimal", ...) on failure.
    # Patch 74 only added a visible caption at the Latest News tab's call
    # site (~line 1773 _fh_visible_caption) -- it never touched this Pitch-tab
    # navigator tile, so a manager screenshotting THIS tile (as originally
    # reported: "Pitch tab, GW6 Rating: —, 'Where is the rate%'") still saw a
    # bare "—" with no visible reason, even after Patch 74 shipped. Same fix,
    # same diagnostic key, second location -- not a new logic bug.
    if nav_rating.get("over_ceiling"):
        st.markdown(recommend.ui_chip(f"GW{nav_gw} capped at 100%", "warn", f"raw {nav_rating['raw_pct']}%",
                                      tip="The best-squad solve returned a squad below yours: the ceiling is not exact (solver gap)."),
                    unsafe_allow_html=True)
    if nav_rating["rating_pct"] is None:
        _nav_rating_diag = opt.get_diagnostic("free_hit_optimal")
        _nc(f"⚠ GW{nav_gw} Rating: no result", f"⚠ GW{nav_gw} Rating: no result this run — "
                   + (_nav_rating_diag if _nav_rating_diag else
                      "no specific reason was recorded — please report this exact combination so it can be added."))
    if not at_planning_gw:
        st.caption("Projected for this GW only",
                   help="Overall rank and Season points elsewhere on this page are your live actuals and "
                        "don't change with navigation.")

    if nav_starters.empty:
        st.warning("No starting XI data for this GW.")
        return

    # Armband: at planning_gw with the current squad, use the real
    # captaincy-protocol pick (EO-aware) and mark your actual live FPL
    # captain if it differs. Every other navigated GW/toggle state uses a
    # simple top-projected-scorer armband (confirmed with the manager) — a
    # full captaincy-protocol re-run isn't meaningful for a hypothetical
    # future GW or an as-if-transferred squad.
    if at_planning_gw:
        nav_cap_code = cap_pick_row["code"] if cap_pick_row is not None else None
    else:
        nav_cap_code = nav_starters.sort_values(nav_col, ascending=False).iloc[0]["code"]
    nav_show_ticker = at_planning_gw and gw_list and len(gw_list) > 1

    nav_html = '<div class="pitch">'
    for pos in ["GK", "DEF", "MID", "FWD"]:
        rows = nav_starters[nav_starters["position"] == pos].sort_values(nav_col, ascending=False)
        if rows.empty:
            continue
        nav_html += '<div class="prow">'
        for _, r in rows.iterrows():
            rec_cap = nav_cap_code is not None and r["code"] == nav_cap_code
            live_cap_diff = at_planning_gw and (r["code"] == captain_id) and not rec_cap
            nav_html += _player_card(r, is_captain=rec_cap, is_live_captain=live_cap_diff,
                                      xp_col=nav_col, opp_col=f"opp_gw{nav_gw}",
                                      gw_list=gw_list if nav_show_ticker else None)
        nav_html += '</div>'
    if not nav_bench.empty:
        nav_html += '<div class="bench-strip"><div class="side-note">BENCH</div><div class="prow">'
        for _, r in nav_bench.sort_values(nav_col, ascending=False).iterrows():
            nav_html += _player_card(r, xp_col=nav_col, opp_col=f"opp_gw{nav_gw}",
                                      gw_list=gw_list if nav_show_ticker else None)
        nav_html += '</div></div>'
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)
    if nav_show_ticker:
        _nc("● easy · mid · hard (hover)", "Fixture ticker: one dot per GW in your horizon — easy/mid/hard, hover for the opponent. Full opponent + xPts breakdown per GW is in the table below.")
with tab_pitch:
    # Patch 68 — moved here (was inside tab_transfers, Patch 66) so the Pitch
    # Navigator is the app's default-open view (see the Patch 68 note above
    # the st.tabs() call for why "first tab" = "default tab" in Streamlit).
    #
    # Patch 60 — compute the manager's own scenario (if "Evaluate scenario" was
    # just clicked, or was clicked on a prior rerun and is still cached) BEFORE
    # the Pitch Navigator renders, so a freshly-evaluated Wildcard/Free
    # Hit/target scenario is selectable in the navigator on THIS SAME render —
    # not only after one extra, unrelated interaction. See the fix-rationale
    # comment above `_compute_scenario_evaluations()`'s definition for the root
    # cause this replaces.
    _compute_scenario_evaluations()
    _render_pitch_navigator()

with tab_transfers:
    # ---------------------------------------------------------------------------
    # GW Breakdown table (Patch 2) — opponent + per-GW xPts split out instead of
    # blended into one horizon number. Only shown when horizon > 1; at horizon=1
    # the pitch view's opponent chip + xp already tell the whole story. Uses
    # st.dataframe (not custom HTML) so it gets native horizontal scroll on
    # narrow screens for free, same pattern as the Season Ledger / move-by-move
    # tables elsewhere on this page.
    # ---------------------------------------------------------------------------
    if horizon > 1 and not squad_df.empty:
        st.markdown('<div class="section-h">GW Breakdown</div>', unsafe_allow_html=True)
        breakdown_rows = []
        for _, r in squad_df.sort_values(["position", "xpts_horizon_sum"], ascending=[True, False]).iterrows():
            row = {"Player": f"{r.get('web_name','')}", "Pos": r.get("position", "")}
            for gw in gw_list:
                opp = r.get(f"opp_gw{gw}", "") or "—"
                xp = r.get(f"xpts_gw{gw}", 0.0)
                xp = 0.0 if pd.isna(xp) else xp
                row[f"GW{gw}"] = f"{opp} · {xp:.1f}"
            total = r.get("xpts_horizon_sum", 0.0)
            row["Horizon total"] = f"{(0.0 if pd.isna(total) else total):.1f}"
            breakdown_rows.append(row)
        _df(pd.DataFrame(breakdown_rows), hide_index=True, use_container_width=True)
        _nc("Cell = opp (H/A) · xPts", "Each GW cell: opponent (H/A) · projected xPts for that gameweek specifically.")

    # ---------------------------------------------------------------------------
    # Transfer recommendations
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-h">Transfer Recommendations</div>', unsafe_allow_html=True)
    # Patch 112: link the ordinary transfer list to the Wildcard decision (they were separate islands).
    _wg = (_wc_decision or {}).get("gw") if _wc_decision else None
    if _wg is not None and _wg == planning_gw:
        st.markdown(recommend.ui_chip(f"🃏 Wildcard GW{_wg} now", "ok", "list below = no-Wildcard alternative"),
                    unsafe_allow_html=True)
    _chips_row = [recommend.ui_chip(recommend.budget_chip_text(team_value, _budget_src, round(team_value - (bank + _market_sum), 1)),
                                    "ok" if _budget_src == "auto" else "warn")]
    _pcc = recommend.plan_conflict_chip((_chip_schedule or {}).get("wildcard_gw"), _wg)
    if _pcc:
        _chips_row.append(recommend.ui_chip(_pcc, "warn"))
    if locked_names:
        _chips_row.append(recommend.ui_chip("🔒 " + ", ".join(locked_names), "info"))
    for _fn, _fs in _lock_flagged:
        _chips_row.append(recommend.ui_chip(f"{_fn} ({_fs})", "bad", "locked, flagged"))
    st.markdown(" ".join(_chips_row), unsafe_allow_html=True)
    if _wg is not None and _wg != planning_gw:
        st.markdown(recommend.ui_chip(f"🃏 Wildcard GW{_wg}", "info", "list below = path until then"), unsafe_allow_html=True)

    # Post-Patch-34 follow-up — free worst-case comparison, shown before any
    # transfer recommendation: if a flagged squad player is currently starting,
    # this is what your best XI looks like if he truly scores zero, using only
    # players you already own. Deliberately captioned as a downside-risk check,
    # not "free upgrade" — the model's own projection for him already reflects
    # a probability-weighted expectation (see the function's docstring); this is
    # for when the manager's own read is harsher than that.
    # Patch 81 (manager report: 3 simultaneously-flagged players this run --
    # Palmer, Pedro, Isak -- only Isak's worst case showed here; Pedro's was
    # invisible in this tab even though his own xPts already carried his live
    # discount). free_lineup_fix_check() now returns one entry PER currently-
    # starting disrupted player instead of only the single highest-projected
    # one -- loop and show all of them, not just the first.
    for _entry in free_fix.get("entries", []):
        _dm(f"⚠️ If {_entry['player']} scores 0: {_entry['worst_case_total']:.1f} xPts (now {_entry['current_total']:.1f})", f'<div class="tx-preview">⚠️ Worst case if <b>{_entry["player"]}</b> scores 0 this GW '
                    f'(currently started; his own projection already reflects a live chance-of-playing discount, '
                    f'this is the harsher case): best XI with <b>{_entry["worst_case_replacement"] or "—"}</b> '
                    f'instead — <b>{_entry["worst_case_total"]:.1f}</b> xPts (vs {_entry["current_total"]:.1f} '
                    f'if he plays at his current projection). No transfer needed for this alone — compare against '
                    f'any transfer recommended below.</div>', unsafe_allow_html=True)

    if transfer_error:
        _ne("Transfer suggestions failed", f"Couldn't compute transfer suggestions this run ({transfer_error}). Everything else on this page "
                 f"is unaffected — try Run Model again, and if it repeats, this is worth reporting with that message.")

    # Patch 5 — simplified primary display: just the recommendation, in plain
    # language, front and center. All the rule-citation trace and the raw move
    # table that used to be the primary content now live behind one expander,
    # available on demand rather than shown by default.
    if rec.get("is_weekly_schedule"):
        _hs_txt = rec.get("hit_stance", "No hits")
        st.caption(f"{_hs_txt} + {horizon}-GW chained pacing plan",
                   help=f"{_hs_txt} + {horizon}-GW horizon → this is a chained, week-by-week pacing plan (each week's "
                        f"move assumes every earlier week's suggested move already happened), not a single this-week "
                        f"decision. Free-transfer accrual (+1/week, cap 5) is modeled explicitly below."
                        + (" Hits are allowed where a paid move still clears the stricter hit-cost bar."
                           if _hs_txt == "Hit if worth it" else ""))
    # Patch 58 (manager screenshot, red-annotated "no need for this
    # explanation") — correcting an earlier verification: this specific text
    # ("GW{n}: Chip context — Disruption check ...; Price-drop-flow override
    # ...; Shape-test ...") is NOT a `_flag_pill()` hover tooltip (that
    # mechanism was checked and is unrelated) — it's `chip_advisory`
    # (constructed above from disruption/shape-test notes) appended straight
    # into `rec["summary"]` by recommend.py (lines ~736-738 / ~1272-1274) and
    # rendered here as plain, always-visible body text with no truncation and
    # no tooltip. That's the actual gap Patch 57's sweep missed, since Patch 57
    # only covered `st.caption`/`st.markdown` blocks written directly in this
    # file, not text assembled upstream and passed through `rec["summary"]`.
    # Fixed the same way as Patch 57's other blocks: short headline + full text
    # on hover — every other summary line (the actual move recommendation)
    # renders exactly as before, unabridged.
    # Release 2 — Transfer OUT->IN visual card(s), one per actual player move in
    # rec["moves"] (manager: "the ... transfer recommendation to be visuals not
    # written"). Added ABOVE the existing text summary rather than replacing it
    # — every field shown here (out/in player, position, net xPts, hit cost) is
    # already in that text too, so nothing is hidden or lost, this is purely an
    # additional at-a-glance view. "No move" weeks (Roll) have no entries in
    # rec["moves"] and so render no card here, same as before.
    #
    # Patch 65 (manager screenshot: the chained-pacing-plan case — "Hit if
    # worth it" + a 2+ GW horizon, i.e. rec["is_weekly_schedule"] is True —
    # still showed the old plain-text line instead of a photo card). ROOT
    # CAUSE, confirmed in code: this section was originally gated with
    # `not rec.get("is_weekly_schedule")`, which was never actually necessary
    # — recommend.py's weekly-schedule path (suggest_transfers(), ~line 744)
    # builds each week's moves with the exact same `_move_row()` helper used by
    # every single-decision path (~lines 1081/1221/1543/1570), so the dicts in
    # rec["moves"] are IDENTICALLY shaped either way (out/out_code/out_team/
    # in/in_code/in_team/position/hit_cost/net_gain/gw), and recommend.py
    # itself already flattens every week's moves into the top-level
    # rec["moves"] list (`"moves": [m for wk in weekly_plan for m in
    # wk["moves"]]`) specifically so callers don't have to special-case it.
    # The guard was simply wrong — removed, so the chained-plan case gets the
    # same visual cards, one per move across every week in the plan.
    if rec.get("moves"):
        for _mv in rec["moves"]:
            _mv_out_initials = "".join([w[0] for w in str(_mv.get("out", "??")).split()][:2]).upper() or "??"
            _mv_in_initials = "".join([w[0] for w in str(_mv.get("in", "??")).split()][:2]).upper() or "??"
            _mv_hit = _mv.get("hit_cost", 0) or 0
            _mv_hit_html = '<span class="hit-pill free">FREE</span>' if not _mv_hit else \
                f'<span class="hit-pill hit">-{_mv_hit} pts</span>'
            _mv_net = _mv.get("net_gain", 0) or 0
            _mv_net_cls = "" if _mv_net >= 0 else " neg"
            _mv_gw_tag = f"GW{_mv['gw']}" if _mv.get("gw") is not None else f"GW{planning_gw}"
            st.markdown(
                '<div class="tx-card"><div class="top"><span class="tag">' + _mv_gw_tag +
                ' · ' + str(_mv.get('position', '')) + '</span>' + _mv_hit_html + '</div>'
                '<div class="tx-swap">'
                '<div class="tx-player out"><div class="ring"><img src="' + _photo_url(_mv.get('out_code', 0)) + '" '
                'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">'
                '<div class="avatar-fallback">' + _mv_out_initials + '</div></div>'
                '<div class="pname">' + str(_mv.get('out', '')) + '</div>'
                '<div class="pmeta">' + str(_mv.get('out_team', '')) + '</div></div>'
                '<div class="tx-arrow">'
                '<svg width="22" height="14" viewBox="0 0 28 18"><path d="M0 9 H24 M17 2 L24 9 L17 16" '
                'stroke="var(--accent)" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                '<div class="tx-net' + _mv_net_cls + '">' + f"{_mv_net:+.1f}" + '</div>'
                '<div class="tx-net-l">net xPts</div></div>'
                '<div class="tx-player in"><div class="ring"><img src="' + _photo_url(_mv.get('in_code', 0)) + '" '
                'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'flex\';">'
                '<div class="avatar-fallback">' + _mv_in_initials + '</div></div>'
                '<div class="pname">' + str(_mv.get('in', '')) + '</div>'
                '<div class="pmeta">' + str(_mv.get('in_team', '')) + '</div></div>'
                '</div></div>', unsafe_allow_html=True)

    _CHIP_CONTEXT_RE = re.compile(r"^(GW\d+): Chip context — (.+)$")
    if rec.get("summary"):
        _plain_lines = []
        _ctx_chips = []
        for line in rec["summary"]:
            m = _CHIP_CONTEXT_RE.match(line)
            if m:
                _gw_txt, _detail = m.group(1), m.group(2).rstrip(".")
                _n_bits = _detail.count("; ") + 1
                _plural = "s" if _n_bits != 1 else ""
                _ctx_chips.append(recommend.note_html(f"🔗 {_gw_txt}: {_n_bits} factor{_plural}", "info", _detail))
            else:
                _plain_lines.append(line)
        if _plain_lines:
            _dm(f"Move notes ({len(_plain_lines)}) — same as the cards", "<br>".join(f'<div class="tx-reco">{l}</div>' for l in _plain_lines))
        if _ctx_chips:
            st.markdown(recommend.chips_row_html(_ctx_chips), unsafe_allow_html=True)
    else:
        st.info("No squad/pool data to plan against this run.")

    # Patch 33 (manager report: this preview already existed — Patch 23 — but sat
    # buried inside the "Why" expander below, so it read as missing). Promoted
    # to sit directly under the recommendation itself, always visible. Same
    # scope/logic as before: only the single-decision recommendation (not the
    # "No hits" chained weekly schedule, where "which week's squad" is itself
    # ambiguous), and never touches the header stats above — a local preview only.
    _moves_df_preview = pd.DataFrame(rec["moves"]) if rec.get("moves") else pd.DataFrame()
    if not rec.get("is_weekly_schedule") and not _moves_df_preview.empty \
            and "out_code" in _moves_df_preview.columns and "in_code" in _moves_df_preview.columns:
        _move_out_codes = set(_moves_df_preview["out_code"])
        _move_in_codes = set(_moves_df_preview["in_code"])
        _post_transfer_squad = pd.concat([
            squad_df[~squad_df["code"].isin(_move_out_codes)],
            proj[proj["code"].isin(_move_in_codes)],
        ], ignore_index=True, sort=False)
        if "code" in _post_transfer_squad.columns:
            _post_transfer_squad = _post_transfer_squad.drop_duplicates(subset=["code"], keep="first")

        _new_xi_result = opt.best_starting_xi(_post_transfer_squad, fh_auto_col) \
            if fh_auto_col in _post_transfer_squad.columns else None
        _new_gw_xpts = round(_new_xi_result["total"], 1) if _new_xi_result else None
        _new_current_val = opt.rating_gw_value(_post_transfer_squad, fh_auto_col, cfg)["total_realized"]
        _new_rating = eng.team_rating_pct(_new_current_val, fh_auto_optimal_val, "")

        if _new_gw_xpts is not None and _new_rating["rating_pct"] is not None:
            _dm(f"📈 After move: {_new_gw_xpts:.1f} xPts (was {gw_xpts_total:.1f}) · rating {_new_rating['rating_pct']}%", f'<div class="tx-preview">📈 If you make this move — new GW{planning_gw} xPts: '
                        f'<b>{_new_gw_xpts:.1f}</b> (was {gw_xpts_total:.1f}) · new Team Rating: '
                        f'<b>{_new_rating["rating_pct"]}%</b> (was {fh_auto_rating["rating_pct"]}%, vs. the same '
                        f'GW{planning_gw} Free Hit optimal shown at the top). Local preview only — the header stats '
                        f'above are unaffected until you actually make the transfer and re-run.</div>',
                        unsafe_allow_html=True)

    _nc(f"Gate: {rec['minimum_meaningful_gain_free']} xPts bar + {rec.get('margin_of_error', 2.0):.1f} MoE", f"Gated by a **{rec['minimum_meaningful_gain_free']} xPts** materiality bar and a "
               f"**{rec.get('margin_of_error', 2.0):.1f} xPts** margin-of-error floor — both must clear.",
               help=f"Two separate bars gate a transfer: a **{rec['minimum_meaningful_gain_free']} xPts** materiality "
                    f"bar (is the gain worth spending a free transfer at all) and a "
                    f"**{rec.get('margin_of_error', 2.0):.1f} xPts** margin-of-error floor (is the gain distinguishable "
                    f"from this model's own known projection noise — Standing Rule #34, not adjustable via the sidebar "
                    f"slider). A move must clear BOTH to be recommended. xM badges above show each player's "
                    f"expected-minutes multiplier — already priced into their xPts, surfaced here so a rotation risk "
                    f"doesn't hide behind a good net number.")

    with st.expander("Why — full trace, rule references, and move-by-move detail"):
        _nc(f"{style_name} · hit {rec['hit_cost_threshold']} · bar {rec['minimum_meaningful_gain_free']}", f"Style profile: **{style_name}** · hit-cost threshold **{rec['hit_cost_threshold']} xPts** · "
                   f"free-transfer materiality bar **{rec['minimum_meaningful_gain_free']} xPts** · "
                   f"margin-of-error floor **{rec.get('margin_of_error', 2.0):.1f} xPts** · "
                   f"free transfers available: **{ft['free_transfers']}** (bank £{bank}m) · horizon **{horizon} GW**")
        # Patch 14 — Standing Rule #4 disclosure: whether the recency signal
        # behind the xM Floor Rule's Rule #19 check was actually available this
        # run, not just assumed. If it wasn't, any player's "confirmed nailed"
        # floor this run is on the pre-Patch-14 season-total basis only.
        checked_gws = getattr(snap, "recent_start_checked_gws", None)
        if checked_gws:
            _nc("Recency check on", f"Recency check (Standing Rule #19, Bench GK Verification): confirmed-start xM floors this run "
                       f"required an actual start in GW{checked_gws[0]}–GW{checked_gws[-1]} — a player who started "
                       f"earlier this season but not recently no longer gets an automatic 'nailed' floor.")
        else:
            _nc("⚠️ Recency check unavailable", "⚠️ Recency check (Standing Rule #19) unavailable this run — the per-gameweek live data needed "
                       "to confirm RECENT starts couldn't be fetched, so any 'confirmed start' xM floor this run falls "
                       "back to season-total starts only (pre-Patch-14 behavior). Treat a bench/backup-tier transfer "
                       "candidate's projection with extra caution until this resolves.")
        for line in ft["trace"]:
            st.markdown(f"- {line}")
        for line in rec["plan"]:
            st.markdown(f"- {line}")
        if rec["moves"]:
            moves_df = pd.DataFrame(rec["moves"])
            show_cols = [c for c in ["gw", "out", "in", "position", "xpts_gain_this_gw", "xpts_gain", "in_eo",
                                      "hit_cost", "net_gain", "justified", "setpiece_flag"] if c in moves_df.columns]
            _df(moves_df[show_cols], hide_index=True, use_container_width=True)
            # "New numbers if you make this move" (Patch 23) now renders directly
            # under the main recommendation above (Patch 33) instead of here —
            # see `_moves_df_preview`/`_post_transfer_squad` just above this
            # expander. Kept out of this expander to avoid computing it twice.

    # ---------------------------------------------------------------------------
    # Evaluate your own scenario (Patch 6) — manager-directed what-ifs, always
    # shown ALONGSIDE the model's own default recommendation above, never in
    # place of it (Standing Rule #30: a scope-restricted comparison must be
    # stated as one, not presented as if it were the model's own full-pool
    # pick). Nothing chosen here changes anything above — this section is
    # purely additive. Gated behind an explicit button rather than re-running
    # on every widget change, since each evaluation is a fresh MILP solve.
    #
    # Patch 60: the actual compute (`_compute_scenario_evaluations()`) and the
    # actual render (`_render_scenario_results()`) now live ABOVE, before the
    # Pitch Navigator's call site, so a freshly-evaluated scenario is already
    # live in the navigator by the time it renders — see the fix-rationale
    # comment on `_compute_scenario_evaluations()`'s definition for the full
    # root-cause writeup. This block only holds the input widgets themselves
    # (unchanged in behavior) plus the call that displays whatever is cached.
    # ---------------------------------------------------------------------------
    with st.expander("Evaluate your own scenario — a specific target, a candidate Wildcard date, or a Free Hit GW"):
        _nc("Optional what-if", "Optional. Pick a target player, a candidate Wildcard gameweek, and/or a candidate Free Hit "
                   "gameweek below, then click Evaluate. Leave all on \"— none —\" and nothing changes — the "
                   "recommendation above stays the model's own default full-pool pick.")
        scen_col1, scen_col2, scen_col3 = st.columns(3)
        with scen_col1:
            pool_options = [(None, "— none —")]
            if not pool_df.empty:
                pool_sorted = pool_df.sort_values("web_name")
                pool_options += [(r["code"], f"{r['web_name']} ({r.get('team','')}) · £{r.get('price','?')}m")
                                  for _, r in pool_sorted.iterrows()]
            st.selectbox("Target player to bring in", options=pool_options,
                          format_func=lambda t: t[1], key="scenario_target")
        with scen_col2:
            wc_gw_options = [None] + list(range(planning_gw, 39))
            st.selectbox("Candidate Wildcard gameweek", options=wc_gw_options,
                          format_func=lambda g: "— none —" if g is None else f"GW{g}",
                          key="scenario_wc_gw")
        with scen_col3:
            fh_gw_options = [None] + list(range(planning_gw, 39))
            st.selectbox("Candidate Free Hit gameweek", options=fh_gw_options,
                          format_func=lambda g: "— none —" if g is None else f"GW{g}",
                          key="scenario_fh_gw")
        # Patch 60 — explicit key so `_compute_scenario_evaluations()` (called
        # earlier in the script, before the Pitch Navigator) can tell whether
        # THIS click is what triggered the current rerun.
        st.button("Evaluate scenario", key="scenario_evaluate_btn")
        _render_scenario_results()


# ---------------------------------------------------------------------------
# Captaincy — history note: Patch 4 retired the old standalone "Captaincy
# Pick" section (two st.metric boxes) in favor of the pitch card's armband +
# `.cap-caption` line. Patch 66 brings a standalone captaincy section back —
# see `with tab_captain:` above — but as its own top-level tab (matching the
# project's 5 standing output sections) rather than the old always-visible
# metric-box section this comment originally described.
# ---------------------------------------------------------------------------

with tab_style:
    # ---------------------------------------------------------------------------
    # Season ledger
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-h">Season Ledger</div>', unsafe_allow_html=True)
    if cur_hist:
        chip_by_event = {c.get("event"): c.get("name") for c in chips_played}
        ledger_rows = []
        for r in sorted(cur_hist, key=lambda x: x["event"], reverse=True)[:10]:
            gw = r["event"]
            chip = chip_by_event.get(gw)
            cost = r.get("event_transfers_cost", 0) or 0
            n = r.get("event_transfers", 0) or 0
            move = chip_protocol.CHIP_LABELS.get(chip, chip) + " played" if chip else \
                (f"{n} transfer(s) (−{cost}pt)" if cost else (f"{n} transfer(s)" if n else "—"))
            # Patch 15 — same fix as the header stat (Patch 12), applied here
            # too: this table was still reading `history["current"]`'s own
            # per-row `overall_rank`, which is a DIFFERENT official-API field
            # from `entry["summary_overall_rank"]` and can disagree with it for
            # the CURRENT (not-yet-finalized) gameweek — showing two different
            # numbers for "GW3 rank" on the same page. Only the current squad_gw
            # row is corrected to the live field; already-finalized past rows
            # keep their own historical value (both fields should already agree
            # once FPL finalizes a gameweek).
            rank_val = live_overall_rank if (gw == squad_gw and live_overall_rank is not None) else r.get("overall_rank")
            ledger_rows.append({"GW": gw, "Move": move, "Points": r.get("points"), "Overall rank": rank_val})
        _df(pd.DataFrame(ledger_rows), hide_index=True, use_container_width=True)
        if not gw_final:
            st.caption(f"⏳ GW{squad_gw} row is still provisional",
                       help=f"GW{squad_gw}'s Points and Overall rank above are both live, provisional FPL figures "
                            f"(rank uses the same corrected field as the header stat) — bonus points for this "
                            f"gameweek aren't finalized yet, so both can still move. Past rows are each GW's own "
                            f"confirmed, finalized value and won't change.")
    else:
        _nc("No season history yet", "No season history yet — nothing finished before GW1.")

    # ---------------------------------------------------------------------------
    # Manager style fit
    # ---------------------------------------------------------------------------
    st.markdown('<div class="section-h">Manager Style Fit</div>', unsafe_allow_html=True)
    st.markdown(f"**{style_name}** — {style_profiles.get_profile(style_name)['description']}",
                help="Ownership is never a reason on its own to prefer a pick — the EO weighting above only breaks "
                     "ties once xPts is already close, and any differential still has to clear the pool-average "
                     "floor on merit.")

    # -----------------------------------------------------------------------
    # Season rank chart (Patch 66; y-axis reversal Patch 69; Patch 78 —
    # manager: "the season rank graph needs to be changed to be (All FPL
    # Players >> 1 million up till it reaches 500000 then 100000 then 50000
    # then 10000 then 1000 .. also the GW needs to be more visible and the
    # whole chart to be better visuals". This is a genuine milestone-tier
    # ruler request (Top 1M / 500K / 100K / 50K / 10K / 1K are the tiers FPL
    # managers actually talk about), not just "zoom to my own data" — a
    # PLAIN LINEAR scale (the old 0-3,000,000-by-500K axis, per the
    # manager's screenshot) can't show that: it gives equal pixel-space to
    # every rank number, so almost the entire chart is wasted on the part of
    # the range nobody's actual rank ever sits at evenly. Fixed with a LOG
    # scale instead of a hand-rolled fake "banded" axis (which the Patch 78
    # changelog's original addendum flagged as a real risk — a fabricated
    # band scale can visually misstate how close two ranks actually are). A
    # log scale is the standard, honest way to do exactly what was asked:
    # it naturally compresses the top of the range and expands the bottom,
    # so the manager's own named tiers (dataviz-skill "choosing a form" —
    # order-of-magnitude data belongs on a log axis, never a linear one)
    # land as genuine reference gridlines, not an invented visual trick.
    # Y-axis ticks are pinned to exactly the manager's own named bands
    # (1/1,000/10,000/50,000/100,000/500,000/1,000,000 plus a 10,000,000
    # "rest of the pool" ceiling) via `tickValues`, so those are the ONLY
    # labels shown — reversed (Patch 69's "0 at the top" convention kept).
    # GW-axis fix: labelAngle=0 (was auto-rotated/tiny per the manager's
    # screenshot) + larger, bold, unrotated labels. Visual-polish fix: a
    # subtle accent-tint area wash under the line (dataviz-skill mark spec:
    # area fill at ~10% opacity, never a saturated block), a 2px line, and
    # >=8px point markers with a surface-color ring — all pulled from the
    # app's own existing CSS custom-property palette (--ink/--ink-muted/
    # --rule/--accent-strong/--accent-tint), not new colors, so this stays
    # visually consistent with the rest of the app. Verified by rendering
    # standalone via vl-convert (real PNG output inspected, not just
    # "should work") before this was wired into the live app — see
    # test_patch78_season_rank_chart.py.
    # Reuses `cur_hist` (history["current"], already fetched/computed above
    # for the Season Ledger table right above this — not a new API call) so
    # the chart and the ledger table can never disagree.
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-h">Season Rank</div>', unsafe_allow_html=True)
    if cur_hist:
        _rank_rows = [{"GW": r["event"], "Overall rank": (live_overall_rank if (r["event"] == squad_gw and
                                                                                 live_overall_rank is not None)
                                                            else r.get("overall_rank"))}
                      for r in sorted(cur_hist, key=lambda x: x["event"]) if r.get("overall_rank") is not None or
                      (r["event"] == squad_gw and live_overall_rank is not None)]
        if _rank_rows:
            _rank_df = pd.DataFrame(_rank_rows)
            _rank_chart = build_season_rank_chart(_rank_df)
            st.altair_chart(_rank_chart, use_container_width=True)
            _nc("Rank by GW (log, best on top)", "Overall rank by gameweek — log scale, reversed (best rank at the top), with reference "
                       "lines at the Top 1K/10K/50K/100K/500K/1M tiers so you can read your position against "
                       "them at a glance. Uses the same live-corrected GW figure as the Season Ledger table "
                       "above, so the two never disagree on the current, not-yet-finalized gameweek.")
        else:
            _nc("No rank data yet", "No overall-rank data yet this season — chart will populate once a gameweek finishes.")
    else:
        _nc("No season history yet", "No season history yet — nothing finished before GW1.")
