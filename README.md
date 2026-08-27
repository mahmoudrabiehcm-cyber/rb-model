# RB Model

A live, zero-cost Streamlit app implementing **FPL Projection Model v4.0**
(base formula v3.3 + Step 3c Set-Piece Signal, parameterized §7 Manager
Style Profile, Step 9 Chip Timing Protocol). Enter an FPL team ID, pick a
style profile and hit-stance, click **Run Model**, get live xPts-driven
transfer, captaincy, and chip recommendations pulled straight from the
official FPL API.

For the click-by-click hosting walkthrough, see **DEPLOY.md**. This file
covers what's in the repo and how to change the model later.

## File map

| File | What it does |
|---|---|
| `app.py` | The Streamlit UI — gate screen, sidebar, all display sections. Start here to change layout/wording. |
| `model_config.yaml` | **Every tunable number in the model.** Position multipliers, DEFCON calibration, decay schedule, hit-cost thresholds, set-piece multipliers, chip-timing thresholds. Change a weight here — no code edit needed. |
| `fpl_engine.py` | Core Formula, Decay Schedule, DEFCON probability, CS% (MODEL_POISSON), xM estimation, Team Rating %, base Transfer Net Gain, Captaincy Protocol (v3.3, unchanged). |
| `fpl_data.py` | All live-fetch functions — official API first, GitHub mirror fallback. |
| `data_pipeline.py` | Wires a raw API snapshot into a computed player table (the same pipeline `app.py` and any future CLI/script use). |
| `setpiece.py` | Step 3c — Set-Piece Role Signal (new in v4.0). |
| `style_profiles.py` | §7 — the four Manager Style Profile presets and their dials (new in v4.0). |
| `chip_protocol.py` | Step 9 — Chip Timing Protocol: chip status tracking, DGW/BGW detection, Wildcard flag (new in v4.0). |
| `transfers.py` | Step 7a addendum — free-transfer count derived from transfer history. |
| `recommend.py` | Transfer-swap suggestions + the chess-themed season verdict text. |
| `optimizer.py` | PuLP/CBC constrained squad solver — used for Team Rating %'s Ceiling_xPts. |
| `manual_overrides.csv` | Optional hand-pasted overrides (xM floor, CS% tier-2/3 odds-derived numbers, BPS profile tags, tiny-sample rescue rates). Empty by default; edit and push to use. |
| `requirements.txt` | Python dependencies — this is what Streamlit Cloud installs on deploy. |
| `.streamlit/config.toml` | Theme colors so native widgets (buttons, sliders) match the design. |

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
- **Transfer suggestions are single-slot swaps** (best replacement per
  position, ranked by net gain), not a full multi-transfer squad
  re-optimization. The Team Rating %'s Ceiling_xPts *does* run the full
  constrained MILP solver (`optimizer.py`) — that's the "what's the
  theoretical best squad for this budget" number, just not turned into a
  turn-by-turn multi-transfer path yet.
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
- **Team Rating % defaults to MECHANICAL-TIER** (MODEL_POISSON CS% + the xM
  Floor Rule only). Steps 4 (full Role Multiplier table), 4a (Manager
  Tenure Split), 5 (Pre-Season Evidence) and 6 (Manager System Fit) need
  web research/judgment a zero-cost automated pipeline can't do on its own
  — but they're not unreachable, they route through `manual_overrides.csv`:
  `xm_override` carries a Step 4/4a/5-researched xM number, `cs_pct_override`
  carries a Step 6 blended CS%, and `tenure_discount` (0.40-1.00, clamped)
  applies Step 4a's red-flag scale directly inside `estimate_xm()`. Because
  the file lives in the repo, not per-visitor, populating it (e.g. via a
  periodic Claude-run research pass, pasted into GitHub's web editor and
  committed) upgrades that player's numbers for **every** visitor to the
  app's URL, dynamically, the next time they run their team through it —
  not just whoever asked for the research. `tenure_discount` was loaded by
  `load_overrides()` but silently never consumed anywhere before this fix —
  now wired into `estimate_xm()` in `fpl_engine.py`.

## Running locally (optional — mainly for testing changes before pushing)

```bash
pip install -r requirements.txt
streamlit run app.py
```
