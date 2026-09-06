# RB Model

A live, zero-cost Streamlit app implementing **FPL Projection Model v5.0**
(base formula v3.3 + Step 3c Set-Piece Signal, parameterized §7 Manager
Style Profile, Step 9 Chip Timing Protocol, Rule #34 Margin-of-Error Tie
Rule) plus **Patch 1** (reachable-ceiling Team Rating %, a quantified Chip
Advisor, and a style-aware alt-captain), **Patch 2** (auto-optimized
starting XI, dual captain markers, per-GW opponent + a GW Breakdown table,
a mobile-adaptive pitch view, and a football-terminology UI pass), and
**Patch 3 + Patch 4** (joint multi-transfer optimization with an EO-pull
style tie-break, a dynamic margin-of-error captaincy shortlist, a redesigned
player card with a multi-GW fixture ticker, and captaincy moved onto the
pitch as a single themed caption — see "Patch 3 + Patch 4" below). Enter an
FPL team ID, pick a style profile and hit-stance, click **Run Model**, get
live xPts-driven transfer, captaincy, and chip recommendations pulled
straight from the official FPL API.

For the click-by-click hosting walkthrough, see **DEPLOY.md**. This file
covers what's in the repo and how to change the model later.

## File map

| File | What it does |
|---|---|
| `app.py` | The Streamlit UI — gate screen, sidebar, all display sections. Start here to change layout/wording. |
| `model_config.yaml` | **Every tunable number in the model.** Position multipliers, DEFCON calibration, decay schedule, hit-cost thresholds, set-piece multipliers, chip-timing thresholds. Change a weight here — no code edit needed. |
| `fpl_engine.py` | Core Formula, Decay Schedule, DEFCON probability, CS% (MODEL_POISSON), xM estimation, Team Rating %, base Transfer Net Gain, Captaincy Protocol (Patch 3: shortlist window is now `margin_of_error_threshold()`-driven, plus a `near_miss` fallback column), Rule #34 margin-of-error threshold (Patch 1). |
| `fpl_data.py` | All live-fetch functions — official API first, GitHub mirror fallback. |
| `data_pipeline.py` | Wires a raw API snapshot into a computed player table (Patch 2: per-GW opponent + venue, `opp_gw{n}`; Patch 4: per-GW fixture-difficulty tier, `fdr_gw{n}`, from the official API's difficulty field with a strength-rating fallback); also owns the three squad-solves (`solve_ceiling`, `solve_reachable_ceiling`, `solve_free_hit_rebuild` — Patch 1) that `app.py` and any future CLI/script use. |
| `setpiece.py` | Step 3c — Set-Piece Role Signal. |
| `style_profiles.py` | §7 — the four Manager Style Profile presets and their dials, plus `captain_alt_pick()` (Patch 1; Patch 3 adds a near-miss fallback for clear-standout weeks) which picks the captaincy panel's second slot by each profile's own `eo_pull` direction, and `differential_floor()` (Patch 1, wired in by Patch 3) — the merit bar a low-EO pick must clear before a profile's EO pull is allowed to favour it. |
| `chip_protocol.py` | Step 9 — Chip Timing Protocol: chip status tracking, DGW/BGW detection, Wildcard flag, and (Patch 1) the Chip Advisor's `evaluate_bench_boost` / `evaluate_triple_captain` / `evaluate_free_hit` quantified play/hold verdicts. |
| `transfers.py` | Step 7a addendum — free-transfer count derived from transfer history. |
| `recommend.py` | Transfer suggestions (Patch 3: full rewrite — a joint multi-transfer solve via `optimizer.solve_squad()`, Rules #28/#30/#34/#35/#36, with an EO-pull style tie-break) + the season verdict text (Patch 2: football-commentary phrase bank — `season_verdict()`, was `chess_verdict()`). |
| `optimizer.py` | PuLP/CBC constrained squad solver. `solve_squad()` (Patch 1) takes an `objective_col` (multi-GW horizon sum or a single GW, for Free Hit) and an optional `retain_pool_codes`/`min_retain` constraint — Patch 3 reuses this same mechanism for the joint transfer-count (k=0..5) solve. `best_starting_xi()` (existing, newly wired into `app.py` in Patch 2) picks the model's own best valid formation from a fixed squad for the pitch view. |
| `manual_overrides.csv` | Optional hand-pasted overrides (xM floor, CS% tier-2/3 odds-derived numbers, BPS profile tags, tiny-sample rescue rates). Empty by default; edit and push to use. |
| `requirements.txt` | Python dependencies — this is what Streamlit Cloud installs on deploy. |
| `.streamlit/config.toml` | Theme colors so native widgets (buttons, sliders) match the design. |

## Patch 1 (reachable-ceiling Team Rating %, Chip Advisor, dynamic alt-captain)

Four changes, shipped together because they all lean on the same new
`solve_squad()` capability (a "retain at least N of the current squad"
constraint) and the same new `margin_of_error_threshold()` helper:

- **Team Rating % now compares against a *reachable* ceiling, not an
  unconstrained one.** The old ceiling was the best possible £100m squad
  from the whole pool with zero regard for what you already own or how many
  transfers you have — a fantasy-ideal comparison that reads as
  permanently low almost regardless of squad quality, because it's
  structurally near-impossible to actually reach in one week. The new
  reachable ceiling (`data_pipeline.solve_reachable_ceiling`) keeps at
  least `15 - free_transfers` of your current squad and re-optimizes the
  rest — "the best squad you could actually have this week." The old
  unconstrained number is still shown, labeled "theoretical ceiling," as a
  secondary reference. The headline also now applies Standing Rule #34: a
  gap smaller than the margin-of-error threshold shows as "at ceiling,"
  not a precise-looking decimal implying room that's really just weekly
  projection noise. The tier-disclosure expander also now shows a live
  **researched-tier coverage count** (how many of your 15, and how many of
  the whole pool, have at least one `manual_overrides.csv` entry) instead
  of a static paragraph — it moves as that file gets researched further.
- **Chip Advisor** (new expander under the Chip Rack): quantified
  play-GW{n}/hold verdicts for Bench Boost, Triple Captain, and Free Hit
  within your chosen horizon, each gated by the same margin-of-error
  threshold. Bench Boost and Triple Captain scan your horizon's own
  bench-total / best-starter-per-GW numbers for a genuine standout week.
  Free Hit runs a real full-15-man rebuild solve (`solve_free_hit_rebuild`,
  Standing Rule #25) against your team's total value for each horizon GW
  and compares it to your own best starting XI that week — only solved for
  chips you haven't already played, since each check is a fresh MILP solve.
  Wildcard deliberately has no verdict here (Standing Rule #24) — it's
  flag-only, now with the quantified squad-vs-ceiling gap folded into that
  flag's text as one more disclosed data point, never a trigger.
- **Alt-captain is now style-aware.** It used to always show the
  lowest-EO shortlisted player, which made no sense for "Template Hugger /
  Rank Protector" (a profile about avoiding risk, not surfacing a
  differential). `style_profiles.captain_alt_pick()` now follows each
  profile's own `eo_pull` dial: low-EO profiles get a genuine differential,
  the high-EO profile gets its next-safest pick, and the no-EO-weighting
  profile gets its next-best pick on raw projection. The panel's label
  changes to match ("Differential" / "Next-safest" / "Next-best").
- **`margin_of_error_threshold()`** (`fpl_engine.py`) is the one shared
  Rule #34 implementation behind all of the above — `max(floor_points,
  pct_of_total * compared_total)`, tunable via `model_config.yaml`'s new
  `margin_of_error` section. Not yet wired into the captaincy shortlist
  window itself (`captaincy.shortlist_xpts_window` is still a flat 1.0)
  — that's scoped into Patch 2 alongside the EO-pull transfer-scoring
  rewrite, since both are the same underlying "narrow hardcoded cutoff"
  problem.

Every reachable-ceiling/Free-Hit solve still uses each player's current
price as a stand-in for real sell value (bank + individual sell prices
isn't tracked per-player here) — same disclosed-simplification standard as
the existing Bench Value Rule autosub discount.

## Patch 2 (auto-optimized XI, opponent + GW Breakdown, mobile, football-terminology theme)

Six changes, all UI/presentation-layer plus one small pipeline addition —
no changes to the underlying xPts math itself:

- **Auto-optimized starting XI.** The pitch now renders `optimizer.best_starting_xi()`
  against your actual 15-man squad for the planning gameweek, instead of
  copying whatever XI/bench arrangement your live FPL team happens to have
  set. Bench players are whoever the model itself would leave out.
- **Dual captain markers.** The gold armband marks the model's own
  recommended captain (from `captaincy_protocol()` + your style profile).
  If your actual live FPL captain is a different player, that player gets a
  smaller hollow-ring secondary marker instead — both visible at once, with
  a hover explanation on each, rather than the armband silently disagreeing
  with your real team.
- **Per-GW opponent capture.** `data_pipeline.compute_all()` already looked
  up each player's fixture per GW to compute CS% — it just discarded the
  opponent afterward. Now captured as `opp_gw{n}` ("BOU (H)", or
  `"BOU (H) / NEW (A)"` for a double gameweek, `""` for a blank) and
  surfaced two ways: a compact chip on the pitch card (next GW only — the
  card stays uncluttered) and a full column in the new GW Breakdown table.
- **GW Breakdown table.** New section below the pitch, shown whenever
  horizon > 1: one row per squad player, one column per GW in the horizon,
  each cell showing that GW's opponent + projected xPts, plus a horizon
  total column. Uses `st.dataframe` (not custom HTML) so it gets native
  horizontal scroll on narrow screens for free.
- **Mobile-adaptive pitch.** Card width is now `clamp(58px, 15vw, 104px)`
  instead of a fixed 104px, with a `max-width:480px` media query that drops
  the price line and compresses position/opponent tags so a full 11-man XI
  (including the worst-case 5-defender row) still fits without horizontal
  scrolling on a ~360-375px phone screen. `flex-wrap` was already on the
  card rows, so anything narrower than tested just wraps to more lines
  instead of breaking.
- **Football-terminology theme pass.** The pawn icon is replaced with a
  small inline-SVG crest; `recommend.py`'s season-verdict phrase bank
  (`season_verdict()`, was `chess_verdict()`) now uses genuine football
  commentary language for the same six rank/hits situations (e.g. "Losing
  the Run of Play," "Chasing the Game," "Finding Your XI" for early season);
  and several rule-citation captions (Team Rating, Chip Advisor, Wildcard)
  were reworded into plain language, with the Team Rating stat also gaining
  an inline hover ("i") tooltip so the "what is this measuring" answer
  doesn't require opening the expander below it.

## Patch 3 + Patch 4 (joint transfer optimization, dynamic captaincy, card redesign, fixture ticker)

Delivered together (no file overlap — Patch 3 touches `recommend.py` /
`fpl_engine.py` / `style_profiles.py` / `model_config.yaml`, Patch 4 touches
`app.py` / `data_pipeline.py`) at the user's request, but they're two
conceptually separate changes:

**Patch 3 — transfer/captaincy math rewrite**

- **`recommend.suggest_transfers()` now solves jointly, not pairwise.** The
  old logic ranked the single best replacement per OUT slot and picked the
  top N — it could never find a genuinely better 2-or-3-player reshuffle
  that no individual swap looks best in isolation. It's replaced with a real
  joint solve: for every transfer count k in the hit-stance's allowed range
  ("No hits" tests 0..free transfers; "Hit if worth it" tests 0..5; "Force"
  tests the requested count), `optimizer.solve_squad()` runs against the
  **full squad+pool combined** (Standing Rule #30 — never a pre-filtered
  subset) with `retain_pool_codes=<current squad>, min_retain=15-k`
  (Standing Rule #28 — one MILP per k, not k independent pairwise choices).
  Among the k's whose net gain (after any hit cost) lands within
  `margin_of_error_threshold()` of the best net gain found, the **smallest**
  k wins outright (Standing Rules #34/#35) — a bigger transfer count only
  gets picked when it clears noise by a real margin. `_pair_moves()`
  mechanically asserts the OUT/IN position multisets match (Standing Rule
  #36) before turning the chosen squad diff into a move-by-move plan.
- **EO-pull tie-break, finally wired in.** `style_profiles.differential_floor()`
  existed since Patch 1 but nothing called it. Now, after the joint xPts
  solve has already picked the best squad, `_apply_eo_pull()` looks for a
  same-position, same-or-cheaper, genuinely-tied (margin-of-error) alternative
  to each chosen IN player and substitutes per the active profile's
  `eo_pull` direction — low-EO profiles only substitute a pick that *also*
  clears the differential floor (merit, not a relaxation); the no-EO-weighting
  profile never substitutes at all. This never overrides a real projection
  edge, only breaks a genuine statistical tie in the direction the manager's
  own style profile already implies.
- **The old Bench Value Rule autosub-discount heuristic is retired.** Each
  player's own `xm` already prices in their expected minutes inside
  `xpts_horizon_sum`, so a flat discount on top of that for a bench-slot OUT
  player was double-counting the same signal (the model doc's own worked
  example flagged this as never properly wired in). `bench_codes` is still
  accepted by `suggest_transfers()` for backward compatibility but no longer
  changes the math — the joint solve doesn't need it, since it already
  reasons about the whole 15-man squad shape directly, not per-slot swaps.
- **Captaincy shortlist window is now dynamic.** `captaincy_protocol()`'s
  "statistically tied" window used to be a flat `1.0 xPts`
  (`captaincy.shortlist_xpts_window`, now documented as legacy/superseded
  in `model_config.yaml`, kept in the file but no longer read) regardless of
  how big the week's top score actually was. It's now
  `margin_of_error_threshold(top_xpts, cfg)` (Standing Rule #34) — the same
  shared threshold used everywhere else. A `near_miss` column also flags
  candidates just outside the shortlist but within
  `captaincy.near_miss_multiplier x` the same window; `captain_alt_pick()`
  falls back to that pool (labelled "Near miss – ...") on a genuine
  clear-standout week, so the alt-captain slot isn't just empty when one
  player is miles ahead.

**Patch 4 — card redesign, fixture ticker, captaincy on the pitch**

- **Player card redesign.** Tighter, top-cropped player photo
  (`object-position:center 12%`) in a thin team-color ring, replacing the
  plain centered avatar. Info hierarchy is now name first, a single compact
  "position · opponent" meta line, then xPts as the clearly dominant number
  with price as a quiet mono footnote beside it — not five equal-weight
  stacked lines.
- **Multi-GW fixture ticker.** At horizon 1 the card still shows the single
  opponent chip exactly as Patch 2 shipped it. At horizon > 1, that line is
  replaced by a small row of difficulty-colored dots — one per GW in the
  horizon, hover for the exact opponent — closing the gap where a wider
  horizon still only showed the current gameweek's fixture on the card
  itself (the GW Breakdown table below always had the full text version).
  Sourced from the new `fdr_gw{n}` column (`data_pipeline.compute_all()`):
  the official FPL `team_h_difficulty`/`team_a_difficulty` field (1-5 scale,
  mapped 1-2→easy/3→mid/4-5→hard) when present, falling back to a tier
  derived from the same team-strength ratings `cs_pct_poisson()` already
  reads when it's missing. A double gameweek shows its harder fixture's tier
  (worst case); a blank gameweek shows a dim neutral dot.
- **Captaincy moved onto the pitch.** The standalone "Captaincy Pick"
  section (two `st.metric` boxes) is retired entirely. The gold armband on
  the pitch card stays the primary signal; a single themed caption directly
  under the pitch carries the "why" in plain football language — genuinely
  distinguishing a clear standout week ("the standout pick this week, clear
  of the field") from an actual statistical tie ("a coin-flip with
  {alt}... differential edge given per your style profile"), using the same
  shortlist/near-miss data Patch 3 computes rather than always phrasing it
  as a coin-flip.
- **Fixed: pitch only showed the goalkeeper at horizon = 1.** `_player_card()`
  originally built its HTML as a multi-line f-string with the fixture-ticker
  interpolation (`{ticker_html}`) alone on its own line. That value is only
  ever non-empty when the horizon is 2+ GWs (the ticker doesn't render at
  horizon 1 — see above), so at horizon = 1 that line was blank. Streamlit's
  Markdown renderer treats a blank line inside a raw HTML block as the end
  of that block (a CommonMark HTML-block rule), so everything in the pitch's
  combined HTML string past the first blank line — the rest of that very
  card plus all ten others — silently never rendered. `_player_card()` now
  builds its return value as a single physical line with no embedded blank
  lines possible, regardless of which optional piece (ticker, price, captain
  markers) happens to be empty — verified by rendering a full 11-card pitch
  through an actual CommonMark parser and confirming all 11 survive.

## Updating the model later

- **A weight or threshold changes** (e.g. hit-cost threshold, a decay-schedule
  band, a set-piece multiplier): edit the relevant value in
  `model_config.yaml`, commit, push. Streamlit Cloud auto-redeploys — no
  code change anywhere.
- **A genuinely new calculation step** (the kind of change that produced
  v4.0 itself): edit or add a `.py` module, wire it into `data_pipeline.py`
  or `app.py`, commit, push. Same auto-redeploy.
- **The model's own documentation** (`FPL_Model_v3_3.pdf` /
  `FPL_Model_v4.0_Changelog.md`) doesn't live in this repo — it's a project
  doc, not app code. Keep the two in sync by hand when you change one.

## Known limitations (documented on purpose, not hidden)

- **Budget-fit on transfer suggestions** uses each player's current price,
  not your actual banked sale price (FPL's sell-on-fee mechanic). The
  official API only exposes your exact sale price via an authenticated
  endpoint this zero-cost, no-login tool deliberately doesn't use. Suggested
  budgets run slightly conservative, not optimistic.
- **Transfer suggestions now run a full joint multi-transfer solve**
  (Patch 3 — see above), same MILP (`optimizer.py`) the Team Rating % and
  Chip Advisor's Free Hit check already used for "what's the best
  reachable/rebuildable squad." One thing still not modeled: real banked
  sale price (see the bullet above) feeds the same `team_value` budget the
  joint solve optimizes against, so its budget constraint carries the same
  slightly-conservative bias as the single-swap version did.
- **Player photos hotlink the official `resources.premierleague.com` CDN
  directly** — the same images the official site itself uses. For a player
  who transferred clubs very recently, that CDN photo can still show the
  old kit for a while until the Premier League's own media team refreshes
  it; the club-color card stripe is still correct immediately (it's read
  live from `team_id`, unrelated to the photo), only the headshot can lag.
  Nothing in this app can force that update — check the same player's
  photo on the official FPL site itself as a quick way to confirm it's an
  upstream CDN lag and not something specific to this app.
- **Live news beyond the official API's own `news`/`chance_of_playing_next_round`
  fields** (press-conference quotes, journalist reports) is not something
  this deployed app does — that needs an LLM actively searching the web,
  which costs money per call and would break the zero-cost design. The
  official-source flags (injury status, doubt percentage, official news
  text with timestamp) are live and free and shown in the app; deeper
  news-scanning stays a "run it past Claude" step alongside the app, not
  inside it.
- **Set-piece role "newly confirmed" detection is a decay-schedule proxy**,
  not a remembered role-change timestamp (the free tier has no database).
  See the docstring in `setpiece.py` for the exact mechanism and how to
  upgrade it later if you add persistent storage.
- **Wildcard timing is always a flag, never a verdict** — by design
  (Standing Rule #24), not a limitation to fix.
- **`current_gw` vs `planning_gw`** (fixed, documented here since it's easy
  to reintroduce). `fpl_data.py`'s `FplSnapshot` carries both: `current_gw`
  is the last COMPLETED/locked gameweek (the official API's `is_current`
  flag) — the only one it has an actual picks snapshot for, so it's what
  fetches the manager's squad. `planning_gw` is the next gameweek whose
  deadline hasn't passed (`is_next`) — what xPts, transfer suggestions,
  captaincy, and chip advisories should all target. They're different for
  most of the week: `is_current` stays pointed at a gameweek for a while
  after its deadline passes (through kickoff and results processing), so a
  naive single "current GW" reads as the gameweek that just finished, not
  the one you can still act on. Passing `current_gw` where `planning_gw`
  belongs makes every recommendation look like it's reasoning about a dead
  gameweek — that was a real bug here until this fix, not just a labeling
  issue.
- **Team Rating %'s per-player tier still defaults to MECHANICAL-TIER**
  (MODEL_POISSON CS% + the xM Floor Rule only) until `manual_overrides.csv`
  has an entry for that player. Steps 4 (full Role Multiplier table), 4a
  (Manager Tenure Split), 5 (Pre-Season Evidence) and 6 (Manager System
  Fit) need web research/judgment a zero-cost automated pipeline can't do
  on its own — but they're not unreachable, they route through
  `manual_overrides.csv`: `xm_override` carries a Step 4/4a/5-researched xM
  number, `cs_pct_override` carries a Step 6 blended CS%, and
  `tenure_discount` (0.40-1.00, clamped) applies Step 4a's red-flag scale
  directly inside `estimate_xm()`. Because the file lives in the repo, not
  per-visitor, populating it (e.g. via a periodic Claude-run research pass,
  pasted into GitHub's web editor and committed) upgrades that player's
  numbers for **every** visitor to the app's URL, dynamically, the next
  time they run their team through it — not just whoever asked for the
  research. The headline ratio itself (Patch 1) no longer reads as
  permanently low regardless of coverage, since it's now measured against
  a reachable ceiling instead of an unconstrained one — see "Patch 1"
  above. The tier-disclosure expander shows a live researched-tier
  coverage count (squad and pool-wide) instead of a static paragraph, so
  it's visible exactly how much of the pool still needs research.
- **No persistent week-over-week Team Rating % trend.** Streamlit
  Community Cloud has no database/persistent storage in the free tier, so
  each run is a fresh computation with nothing to compare against last
  week's number except what the manager remembers or screenshots. Adding a
  real trend line would need an external, still-zero-cost store (e.g. a
  small committed CSV log file, appended on each research pass) — not
  built yet, flagged here so it isn't mistaken for a missing feature bug.

## Running locally (optional — mainly for testing changes before pushing)

```bash
pip install -r requirements.txt
streamlit run app.py
```
