# RB Model

A free Streamlit app for Fantasy Premier League. Enter any FPL team ID and it reads the squad, bank, free transfers and
chips from the official FPL API, projects xPts for every player over the coming gameweeks, and recommends transfers,
captaincy and chip timing. It is built to help any team; it is not tuned to one squad.

- Model document: FPL Maverick v6.13 (project PDF). The app follows its rules; departures are named in the app text
  and in `logs/LOG_APP.md`.
- Current release: **Release 2, Patch 1** (badge "r2 patch 1"). Release 1 ended at Patch 117l. The full history is
  the `PATCH_VERSION` string at the top of `app.py`.
- How to host it: see `DEPLOY.md`.

## Screens

Pitch (squad per gameweek, GW rating), Latest News, Transfer Recommendations (weekly plan, hit check), Captaincy Pick,
Chip Plan (Wildcard decision, Triple Captain / Bench Boost / Free Hit weeks, Extended Check), Manager Style Fit.
Sidebar: team, style profile, hit stance, Locked players, Excluded players, Download run snapshot, Advanced.

## File map

| File | What it does |
|---|---|
| `app.py` | The Streamlit app: gate screen, sidebar, every tab, and the order the model steps run in. |
| `model_config.yaml` | Every tunable number (weights, thresholds, chip and Wildcard switches). Change a number here, no code edit. |
| `fpl_data.py` | Live data from the official FPL API, with a GitHub mirror fallback. |
| `data_pipeline.py` | Builds the per-player table (xPts per gameweek, clean sheets, fixture factors, sheet overrides) and the best-possible-squad solves. |
| `fpl_engine.py` | Core formulas: xM (expected minutes), decay schedule, DEFCON, Team Rating %, margin of error (Rule #34), Wildcard trigger. |
| `optimizer.py` | PuLP/CBC squad solver (budget, 2-5-5-3, max 3 per club; weekly XI + captain build for the Wildcard). Works on PuLP 3.x and 4.x. |
| `recommend.py` | Transfer planner, Wildcard chain decision, chip placement on the plan, card and label text. |
| `chip_protocol.py` | Chip status per window, double/blank detection, Rule #48 Wildcard scan, Rule #49 joint chip sequence. |
| `setpiece.py` | Set-piece role signal. |
| `style_profiles.py` | The manager style presets (default "Calculated Maverick"). |
| `transfers.py` | Free transfers from the transfer history (banked to 5). |
| `scalar_reference.py` | Old reference copy used by one test only; not imported by the app. |
| `manual_overrides_template.csv` | Column layout for the weekly sheet. The manager uploads the real `manual_overrides.csv` himself. |
| `my_squad_template.csv` | Not used (the squad is read from the FPL API by team ID). |
| `requirements.txt` | What Streamlit Cloud installs. |
| `.streamlit/config.toml` | Theme colours for native widgets. Must sit in a `.streamlit` folder in the repo; patch zips do not include it. |

## The weekly sheet (`manual_overrides.csv`, key `player_code`)

Columns read: `xm_override`, `cs_pct_override`, `bps_profile`, `tenure_discount`, `npxg90_rescue`, `xa90_rescue`,
`dc90_rescue`, `note`, `cs_pct_gw6`..`cs_pct_gw10` (any `cs_pct_gwN` column is picked up), `cs_gw`. A blank per-GW cell,
or a week with no column, uses `cs_pct_override`.

## Tests (not part of the deploy)

The tests live in `tests_bundle.zip`. Unzip it beside these files in a folder named `rb_model_app`; two tests read that
path. Then:

```
pip install -r requirements.txt -r requirements-dev.txt
python3 -m pytest -q -p no:cacheprovider $(ls test_*.py | grep -v "driver\|playwright\|harness")
```

Release 2 Patch 1: 590 tests, all passing on PuLP 3.3.2 and on PuLP 4.0.0 (with the repo's Turf Green `config.toml`).
Rule: write the failing test first, then the fix, then the full regression.
