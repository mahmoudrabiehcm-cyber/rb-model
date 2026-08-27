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

## Running locally (optional — mainly for testing changes before pushing)

```bash
pip install -r requirements.txt
streamlit run app.py
```
