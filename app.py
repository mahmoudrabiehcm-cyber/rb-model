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

st.set_page_config(page_title="RB Model", page_icon="⚽", layout="wide")

# Crest mark (Patch 2) — replaces the pawn icon everywhere it appeared inline
# in the brand wordmark. A small geometric badge rather than an emoji glyph.
_CREST_SVG = ('<svg viewBox="0 0 100 100" width="26" height="26" style="flex:none;">'
              '<polygon points="50,4 90,26 90,68 50,96 10,68 10,26" fill="none" stroke="#0E3D26" stroke-width="6"/>'
              '<polygon points="50,22 74,36 74,64 50,80 26,64 26,36" fill="#1F6D45"/>'
              '<circle cx="50" cy="50" r="9" fill="#F5F6F0"/></svg>')

# ---------------------------------------------------------------------------
# Style — a considered, precise identity (terse micro-copy, restraint, one
# accent per screen) expressed through football's own vocabulary (Patch 2)
# rather than chess — same underlying "soul," different, more natural words.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,900&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#F5F6F0; --surface:#FFFFFF; --surface-2:#EDF0E7;
  --ink:#16211B; --ink-muted:#4B564E; --ink-faint:#8B9686;
  --rule:#DBE1D3; --accent:#1F6D45; --accent-strong:#12492E; --accent-tint:#E4EFE6;
  --gold:#B8842A; --gold-tint:#F6ECD9;
  --blue:#2E6E93; --blue-tint:#E1EDF3;
  --coral:#C1502E; --coral-tint:#F7E3DA;
  --warn:#B23B1E; --bench:#EFF1E9;
  --fdr-easy:#375A2E; --fdr-mid:#B8842A; --fdr-hard:#B23B1E;
  --shadow:0 1px 2px rgba(20,30,22,.07), 0 8px 22px -12px rgba(20,30,22,.24);
}
html, body, [class*="css"]{ font-family:"IBM Plex Sans",sans-serif; color:var(--ink); }
.mono{ font-family:"IBM Plex Mono",monospace; }
.stApp{ background:var(--bg); }

.brand-row{ display:flex; align-items:center; gap:8px; }
.brand-mark{ font-family:"Fraunces"; font-weight:900; font-size:2rem; line-height:1; margin-bottom:2px; }
.brand-mark .b2{ color:var(--accent-strong); }
.brand-tag{ font-family:"IBM Plex Mono"; font-size:10.5px; color:var(--ink-faint); letter-spacing:.06em; text-transform:uppercase; }

.verdict-card{
  background:linear-gradient(160deg, var(--surface) 55%, var(--accent-tint));
  border:1px solid var(--rule); border-left:4px solid var(--accent);
  box-shadow:var(--shadow); padding:18px 20px; margin-bottom:6px;
}
.verdict-card .phase-tag{ font-family:"IBM Plex Mono"; font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); margin-bottom:4px; display:block; }
.verdict-card .h{ font-family:"Fraunces"; font-weight:800; font-size:1.35rem; margin:0 0 6px; color:var(--accent-strong); }
.verdict-card .b{ margin:0; color:var(--ink-muted); font-size:.94rem; font-style:italic; }

.stat-row{ display:flex; gap:28px; font-family:"IBM Plex Mono"; margin:14px 0 26px; flex-wrap:wrap; align-items:flex-start; }
.stat .n{ font-size:1.5rem; font-weight:600; }
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

.section-h{ font-family:"Fraunces"; font-weight:700; font-size:1.15rem; margin:30px 0 14px; padding-bottom:8px; border-bottom:1px solid var(--rule); }

.chip-rack{ display:flex; gap:10px; flex-wrap:wrap; margin:0 0 8px; }
.chip{ display:flex; align-items:center; gap:7px; background:var(--surface); border:1px solid var(--rule);
  padding:6px 11px; font-family:"IBM Plex Mono"; font-size:11.5px; box-shadow:var(--shadow); }
.chip .dot{ width:7px; height:7px; border-radius:50%; flex:none; }
.chip.available .dot{ background:var(--accent); }
.chip.used .dot{ background:var(--ink-faint); }
.chip.flagged .dot{ background:var(--gold); }
.chip.used{ color:var(--ink-faint); }
.chip.used .name{ text-decoration:line-through; }

.pitch{ background:linear-gradient(180deg, var(--accent-tint), var(--surface-2) 70%);
  border:1px solid var(--rule); padding:22px 14px 10px; }
.prow{ display:flex; justify-content:center; gap:14px; margin-bottom:18px; flex-wrap:wrap; }

/* Patch 4 — card redesign: tighter top-cropped photo in a team-color ring,
   name-first info hierarchy (name -> compact pos+opponent meta line -> xPts
   as the dominant stat with price as a quiet footnote), replacing the old
   five-equal-weight stacked-line layout. */
.card{ background:var(--surface); border:1px solid var(--rule); box-shadow:var(--shadow);
  width:clamp(64px, 15vw, 112px); padding:10px 8px 9px; text-align:center; position:relative; }
.card .cap{ position:absolute; top:-9px; right:-9px; width:20px; height:20px; border-radius:50%;
  background:var(--gold); color:#241A05; font-family:"IBM Plex Mono"; font-size:10.5px; font-weight:700;
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
  background:linear-gradient(160deg, var(--team,var(--accent)), #12492E); color:#fff; font-family:"Fraunces";
  font-weight:700; font-size:13px; align-items:center; justify-content:center; border:2px solid var(--surface); }
.card .name{ font-weight:700; font-size:12.5px; margin-bottom:2px; }
.card .meta{ font-family:"IBM Plex Mono"; font-size:8.5px; color:var(--ink-muted); margin-bottom:6px; }
.card .meta .pos{ font-weight:700; }
.card .meta .pos.gk{ color:var(--gold); } .card .meta .pos.def{ color:var(--blue); }
.card .meta .pos.mid{ color:var(--accent-strong); } .card .meta .pos.fwd{ color:var(--coral); }
.card .ticker{ display:flex; justify-content:center; gap:3px; margin-bottom:6px; }
.card .fdr-dot{ width:7px; height:7px; border-radius:50%; cursor:help; flex:none; }
.card .fdr-dot.easy{ background:var(--fdr-easy); } .card .fdr-dot.mid{ background:var(--fdr-mid); }
.card .fdr-dot.hard{ background:var(--fdr-hard); } .card .fdr-dot.blank{ background:var(--ink-faint); opacity:.4; }
.card .stats-row{ display:flex; align-items:baseline; justify-content:center; gap:6px; }
.card .xp{ font-family:"IBM Plex Mono"; font-size:14px; font-weight:700; color:var(--accent-strong); }
.card .xp-l{ font-family:"IBM Plex Mono"; font-size:7px; color:var(--ink-faint); text-transform:uppercase; display:block; margin-top:-2px; }
.card .price{ font-family:"IBM Plex Mono"; font-size:8.5px; color:var(--ink-faint); }
.bench-strip{ background:var(--bench); margin:0 -14px; padding:12px 14px 4px; border-top:1px dashed var(--rule); }
.bench-strip .card{ opacity:.68; width:clamp(56px, 13vw, 96px); }
.side-note{ font-size:11.5px; color:var(--ink-faint); font-family:"IBM Plex Mono"; line-height:1.5; }

/* Patch 4 — captaincy-on-pitch caption, replacing the old standalone
   "Captaincy Pick" metric section entirely. */
.cap-caption{ font-size:13.5px; color:var(--ink); background:var(--surface); border-left:3px solid var(--gold);
  box-shadow:var(--shadow); padding:9px 14px; margin-top:10px; }
.cap-caption b{ color:var(--accent-strong); }

/* Patch 5 — Transfer Recommendations simplification: the primary display is
   now just the recommendation sentence(s), styled the same as the
   captaincy caption for visual consistency; the old rule-citation trace and
   raw move table moved into an on-demand expander. */
.tx-reco{ font-size:13.5px; color:var(--ink); background:var(--surface); border-left:3px solid var(--accent-strong);
  box-shadow:var(--shadow); padding:9px 14px; margin-top:6px; margin-bottom:6px; }

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
}
</style>
""", unsafe_allow_html=True)


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
    sp_html = '<div class="sp">SP</div>' if row.get("setpiece_flag") else ""
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
def _project(_snap, hist_df, overrides_df, cfg, gw_list):
    players = data_pipeline.build_player_table(cfg, _snap, hist_df, overrides_df)
    proj = data_pipeline.compute_all(cfg, _snap, players, gw_list)
    return proj


@st.cache_data(ttl=900, show_spinner=False)
def _picks(entry_id: int, gw: int):
    return fpl_data.fetch_entry_picks_official(entry_id, gw)


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
            st.error("Team ID should be numbers only — find it in the URL when you open 'Points' on the official FPL site.")
        else:
            with st.spinner("Checking team ID against the official API..."):
                test_entry = fpl_data.fetch_entry_official(int(entry_input.strip()))
            if test_entry is None or "id" not in test_entry:
                st.error("Couldn't find that team ID on the official FPL API. Double-check it and try again.")
            else:
                st.session_state.unlocked = True
                st.session_state.team_id = int(entry_input.strip())
                st.session_state.team_name = test_entry.get("name", "")
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
                f'<div class="brand-tag">v5.0 engine · live · zero-cost</div><br>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="side-note">TEAM ID</div>'
                f'<div style="font-family:\'IBM Plex Mono\'; color:var(--accent-strong); '
                f'font-weight:600; margin-bottom:14px;">{entry_id} · {st.session_state.get("team_name","")}</div>',
                unsafe_allow_html=True)
    if st.button("Change team", use_container_width=True):
        st.session_state.unlocked = False
        st.rerun()

    style_name = st.selectbox("Style", list(style_profiles.PROFILES.keys()), index=0)
    st.caption(style_profiles.get_profile(style_name)["description"])

    hit_stance = st.radio("Hit stance", ["No hits", "Hit if worth it", "Force"], index=1)
    forced_count = None
    if hit_stance == "Force":
        forced_count = st.number_input("Transfers to force", min_value=1, max_value=5, value=2, step=1)

    horizon = st.slider("Horizon (gameweeks)", min_value=1, max_value=6, value=1,
                         help="xPts are always shown per-GW too — widen this when you want a multi-week transfer plan view, not just this week's picture.")

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
    st.info("Set your style and hit stance in the sidebar, then click **Run Model** to fetch live data and build your recommendations.")
    st.stop()

with st.spinner("Fetching live data and computing xPts..."):
    recency_window = cfg.get("xm_heuristic", {}).get("recency_window_gws", 4)
    snap, hist_df, entry, history = _load_data(entry_id, cfg["meta"]["season"], cfg["meta"]["previous_season"],
                                                recency_window)

    if snap is None or entry is None or history is None:
        st.error("Couldn't reach the official FPL API right now. It's normally free and open with no key required — this is "
                  "most likely a transient outage or a network policy on wherever this app is currently running. Try again shortly.")
        st.stop()

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

    proj = _project(snap, hist_df, overrides, cfg, gw_list)
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
    team_value = round(bank + (squad_df["price"].sum() if not squad_df.empty else 0.0), 1)
    reachable = data_pipeline.solve_reachable_ceiling(cfg, proj, squad_codes, ft["free_transfers"])
    theoretical_ceiling = data_pipeline.solve_ceiling(cfg, proj)
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
    fh_auto_result = data_pipeline.solve_free_hit_optimal_squad(cfg, proj, team_value, planning_gw)
    fh_auto_current_val = opt.rating_gw_value(squad_df, fh_auto_col, cfg)["total_realized"] \
        if not squad_df.empty else 0.0
    fh_auto_optimal_val = opt.rating_gw_value(fh_auto_result["squad"], fh_auto_col, cfg)["total_realized"] \
        if fh_auto_result else 0.0
    fh_auto_rating = eng.team_rating_pct(fh_auto_current_val, fh_auto_optimal_val, "")
    fh_auto_gap = round(fh_auto_optimal_val - fh_auto_current_val, 2)
    fh_auto_moe = eng.margin_of_error_threshold(fh_auto_optimal_val, cfg) if fh_auto_optimal_val else 0.0
    fh_auto_tooltip = (f"Free Hit rating = your current squad's best XI this GW (captain doubled, bench "
                       f"autosub-discounted), divided by a genuinely unconstrained optimal squad for GW"
                       f"{planning_gw} only (full player pool, no free-transfer limit — a true from-scratch "
                       f"rebuild, unlike Team Rating %'s reachable-ceiling comparison above). Gap: "
                       f"{fh_auto_gap:.1f} xPts (margin-of-error threshold: {fh_auto_moe:.1f} xPts). "
                       f"Explore a different candidate gameweek in 'Evaluate your own scenario' below.")

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
        cap_pick = style_profiles.captaincy_pick(cap_result, style_name)
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

    # chip status + timing — computed before transfer suggestions so the
    # transfer plan can factor in "a chip is coming, banking may beat spending"
    boot_chips = fpl_data.fetch_bootstrap_chips(snap.raw_boot) if snap.raw_boot else []
    chips_played = history.get("chips", []) if history else []
    chip_rows = chip_protocol.chip_status(boot_chips, chips_played)
    fixture_counts = chip_protocol.fixture_counts_by_team(snap.fixtures, gw_list)
    all_team_ids = snap.teams["id"].tolist() if "id" in snap.teams.columns else []
    dgw_bgw = chip_protocol.dgw_bgw_flags(fixture_counts, all_team_ids)
    squad_team_ids = squad_df["team_id"].tolist() if "team_id" in squad_df.columns else []
    chip_notes = chip_protocol.chip_recommendations(chip_rows, dgw_bgw, squad_team_ids, max(len(squad_df), 1))
    flagged_players = squad_df[(squad_df["status"] != "a") | (squad_df["est_rescue_needed"])]
    wc_flag = chip_protocol.wildcard_flag(rank_history_display, len(flagged_players),
                                           squad_xpts_total=squad_total,
                                           reachable_ceiling_total=reachable_total,
                                           moe_threshold=moe)

    # Chip Advisor (v5.0 / Patch 1) — quantified play/hold verdicts within the
    # chosen horizon for the three chips that actually have a "which GW"
    # question (Bench Boost, Triple Captain, Free Hit). Only solved for chips
    # that are actually still available this season — each Free Hit check is
    # a fresh MILP solve per horizon GW, so it's skipped entirely once that
    # chip is used, rather than burning compute on a verdict nobody can act on.
    available_chip_names = {r["chip"] for r in chip_rows if r["status"] == "available"}
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

    bb_advisor = None
    if any(c.startswith("Bench Boost") for c in available_chip_names):
        bb_advisor = chip_protocol.evaluate_bench_boost(bench_df, gw_list, _chip_moe_fn("bench_boost"))
    tc_advisor = None
    if any(c.startswith("Triple Captain") for c in available_chip_names):
        tc_advisor = chip_protocol.evaluate_triple_captain(starters_df, gw_list, _chip_moe_fn("triple_captain"))
    fh_advisor = None
    if any(c.startswith("Free Hit") for c in available_chip_names) and not squad_df.empty:
        fh_advisor = chip_protocol.evaluate_free_hit(
            squad_df, gw_list,
            lambda gw: data_pipeline.solve_free_hit_rebuild(cfg, proj, team_value, gw),
            _chip_moe_fn("free_hit"))

    # Chip-aware transfer advisory (Standing Rule #24: only from signals
    # already computed mechanically above — never a guess at the manager's
    # intent). Wildcard: only fires when the objective wc_flag is already
    # active AND a Wildcard is currently available — both real, not invented.
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
    chip_advisory = f"GW{planning_gw}: Chip context — " + "; ".join(advisory_bits) + "." if advisory_bits else None

    # transfer suggestions — isolated so a bad row here can't take down the
    # rest of the page (pitch view, chip rack, captaincy, ledger all still
    # render even if this section fails).
    transfer_error = None
    try:
        rec = recommend.suggest_transfers(squad_df, pool_df, cfg, style_name, hit_stance,
                                           ft["free_transfers"], bank, planning_gw, gw_list, forced_count,
                                           meaningful_bar_override, set(bench_df["code"]), chip_advisory)
    except Exception as e:
        transfer_error = str(e)
        rec = {"moves": [], "plan": [], "summary": [], "net_gain": 0.0, "profile_used": style_name,
               "hit_cost_threshold": style_profiles.get_profile(style_name)["hit_cost_threshold"],
               "minimum_meaningful_gain_free": cfg["transfer"].get("minimum_meaningful_gain_free", 2.0),
               "margin_of_error": eng.margin_of_error_threshold(0.0, cfg),
               "hit_stance": hit_stance, "free_transfers": ft["free_transfers"],
               "weekly_plan": [], "is_weekly_schedule": False}

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
    rating_tooltip = ("Team Rating % = your optimized XI's projected xPts over this horizon (captain doubled, "
                       "bench valued at its real autosub-discounted rate — Patch 20), divided by the best squad "
                       "actually reachable using your free transfers right now (not an unlimited-budget fantasy "
                       "ideal). See the disclosure expander below for the full researched-tier breakdown.")
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
    st.markdown(f"""<div class="stat-row">
      <div class="stat"><div class="n">{rank_disp} {trend}{prov_badge}</div><div class="l">Overall rank</div></div>
      <div class="stat rating">
        <div class="n">{rating['rating_pct'] if rating['rating_pct'] is not None else '—'}% <span class="info-dot" title="{rating_tooltip}">i</span></div>
        <div class="l">Team rating</div>
        <div class="rating-basis">vs. reachable ceiling</div>
      </div>
      <div class="stat rating">
        <div class="n">{fh_auto_rating['rating_pct'] if fh_auto_rating['rating_pct'] is not None else '—'}% <span class="info-dot" title="{fh_auto_tooltip}">i</span></div>
        <div class="l">Free Hit rating</div>
        <div class="rating-basis">vs. GW{planning_gw} optimal</div>
      </div>
      <div class="stat new"><div class="n">{gw_xpts_total:.1f}</div><div class="l">GW{planning_gw} xPts</div></div>
      <div class="stat"><div class="n">{points_total if points_total is not None else '—'}{prov_badge}</div><div class="l">Season points</div></div>
    </div>""", unsafe_allow_html=True)
    if not gw_final:
        st.caption(f"⏳ GW{squad_gw} rank & points above are FPL's live provisional numbers — bonus points "
                   f"haven't been finalized yet, so both can still shift (this matches the official app/site "
                   f"during this same window, it isn't a bug in this tool). Use **Refresh live data** in the "
                   f"sidebar to re-pull the latest provisional figures.")

gw_status = "confirmed final" if getattr(snap, "current_gw_data_checked", False) else "provisional, not yet finalized"
st.markdown(f'<div class="side-note">Source: {snap.source} · squad as of GW{squad_gw} ({gw_status}) · '
            f'planning for GW{planning_gw} · '
            f'fetched {dt.datetime.fromtimestamp(snap.fetched_at).strftime("%H:%M")} · '
            f'style profile: <b>{style_name}</b></div>', unsafe_allow_html=True)
if rating["rating_pct"] is not None:
    if at_ceiling:
        st.caption(f"✓ Already at your reachable ceiling this week — the {rating_gap:.1f} xPts gap is inside "
                   f"normal weekly noise (threshold {moe:.1f} xPts), not real room left on the table.")
    with st.expander("Team Rating % — full breakdown"):
        st.markdown(tier_label)
        st.caption(f"Squad horizon xPts: {squad_total:.1f} · Reachable ceiling: {reachable_total:.1f} "
                   f"(best squad gettable using your {ft['free_transfers']} free transfer(s) right now, "
                   f"£{cfg['squad_rules']['budget']}m proxy budget, {horizon}-GW horizon) · "
                   f"gap to reachable ceiling: {rating_gap:.1f} xPts (margin-of-error threshold: {moe:.1f} xPts)")
        st.caption(f"Theoretical ceiling (secondary reference, unconstrained — ignores what you currently own or "
                   f"how many transfers you have): {theoretical_total:.1f} xPts")
        st.caption("Patch 20 methodology: each of the three totals above is your best legal Starting XI per GW in "
                   "the horizon, plus a captain bonus (that XI's own top scorer counted a second time — the real "
                   "doubling effect, per §1a's captaincy requirement), plus the 4 bench slots valued at their "
                   "Rule #12 autosub-discounted rate rather than full raw value — computed identically for the "
                   "squad and both ceiling sides (Rule #22 Systematic Application), never a flat 15-man sum. "
                   "The underlying squad SELECTION for the two ceiling solves still optimizes a simpler raw-sum "
                   "objective (a disclosed approximation — the true joint optimum across squad+XI+captain+bench "
                   "for a multi-GW horizon is a materially harder combinatorial problem); only the reported score "
                   "for whichever squad each solve returns uses the corrected calculation above.")
if snap.stale_warning:
    st.warning(snap.stale_warning)

# ---------------------------------------------------------------------------
# Chip rack
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">Chip Rack</div>', unsafe_allow_html=True)
flagged_chip_names = set()
if wc_flag:
    flagged_chip_names = {r["chip"] for r in chip_rows if r["status"] == "available" and r["chip"].startswith("Wildcard")}
chip_html = '<div class="chip-rack">'
for r in chip_rows:
    cls = "used" if r["status"] == "used" else ("flagged" if r["chip"] in flagged_chip_names else "available")
    win = f'used GW{r["event"]}' if r["status"] == "used" else f'GW{r["window"][0]}–{r["window"][1]}'
    chip_html += f'<div class="chip {cls}"><span class="dot"></span><span class="name">{r["chip"]}</span><span class="win">&nbsp;{win}</span></div>'
chip_html += '</div>'
st.markdown(chip_html, unsafe_allow_html=True)
if wc_flag:
    st.markdown(f'<p class="side-note">{wc_flag}</p>', unsafe_allow_html=True)
for note in chip_notes:
    st.markdown(f'<p class="side-note">{note}</p>', unsafe_allow_html=True)

# Chip Advisor — quantified verdicts (Standing Rule #34 margin-of-error gated)
def _advisor_line(label: str, adv: dict | None) -> str | None:
    if adv is None or adv.get("best_gw") is None:
        return None
    if adv["verdict"].startswith("play_gw"):
        gw = adv["verdict"].split("gw")[1]
        return (f"**{label}: play in GW{gw}** — clears margin-of-error by "
                f"{adv.get('margin', adv.get('threshold', 0)):.1f} xPts over the next-best GW in your horizon "
                f"(threshold {adv['threshold']:.1f} xPts).")
    return (f"{label}: hold — no GW in your horizon clears margin-of-error over the others "
            f"(best candidate GW{adv['best_gw']}, threshold {adv['threshold']:.1f} xPts). Statistical tie, not a "
            f"reason to rule it out later.")

advisor_lines = [l for l in (
    _advisor_line("Bench Boost", bb_advisor),
    _advisor_line("Triple Captain", tc_advisor),
    _advisor_line("Free Hit", fh_advisor),
) if l]
if advisor_lines:
    with st.expander("Chip Advisor — quantified play/hold verdicts for this horizon"):
        for line in advisor_lines:
            st.markdown(line)
        st.caption("Wildcard timing is always your call, never a verdict — see the flag above instead. "
                   "Verdicts here only compute for chips you haven't already played this season.")

# ---------------------------------------------------------------------------
# Pitch view
# ---------------------------------------------------------------------------
st.markdown(f'<div class="section-h">Squad · planning for GW{planning_gw}</div>', unsafe_allow_html=True)
if starters_df.empty:
    st.warning("No squad data returned for this team ID / gameweek yet (common right after a deadline, or if this is a brand-new team). "
               "Transfer targets and captaincy below still use the full player pool.")
else:
    pitch_html = '<div class="pitch">'
    for pos in ["GK", "DEF", "MID", "FWD"]:
        rows = starters_df[starters_df["position"] == pos].sort_values("xpts_horizon_sum", ascending=False)
        if rows.empty:
            continue
        pitch_html += '<div class="prow">'
        for _, r in rows.iterrows():
            rec_cap = cap_pick_row is not None and r["code"] == cap_pick_row["code"]
            live_cap_diff = (r["code"] == captain_id) and not rec_cap
            pitch_html += _player_card(r, is_captain=rec_cap, is_live_captain=live_cap_diff,
                                        xp_col=opt_col, opp_col=f"opp_gw{planning_gw}", gw_list=gw_list)
        pitch_html += '</div>'
    if not bench_df.empty:
        pitch_html += '<div class="bench-strip"><div class="side-note">BENCH</div><div class="prow">'
        for _, r in bench_df.sort_values("xpts_horizon_sum", ascending=False).iterrows():
            pitch_html += _player_card(r, xp_col=opt_col, opp_col=f"opp_gw{planning_gw}", gw_list=gw_list)
        pitch_html += '</div></div>'
    pitch_html += '</div>'
    st.markdown(pitch_html, unsafe_allow_html=True)
    if gw_list and len(gw_list) > 1:
        st.markdown('<p class="side-note">Fixture ticker: one dot per GW in your horizon — '
                    'easy/mid/hard, hover for the opponent. Full opponent + xPts breakdown per GW is in the '
                    'table below.</p>', unsafe_allow_html=True)
    st.markdown('<p class="side-note">SP tag = newly confirmed set-piece role, decaying out as current-season minutes accrue.</p>',
                unsafe_allow_html=True)

# Patch 4 — captaincy-on-pitch caption, replacing the old standalone
# "Captaincy Pick" section. Rendered here (own top-level block, not nested
# inside the pitch if/else above) so it still shows even in the rare case
# starters_df is empty but the pool-only captaincy protocol still ran.
if cap_caption:
    st.markdown(f'<div class="cap-caption">{cap_caption}</div>', unsafe_allow_html=True)

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
    st.dataframe(pd.DataFrame(breakdown_rows), hide_index=True, use_container_width=True)
    st.markdown('<p class="side-note">Each GW cell: opponent (H/A) · projected xPts for that gameweek specifically.</p>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Transfer recommendations
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">Transfer Recommendations</div>', unsafe_allow_html=True)
if transfer_error:
    st.error(f"Couldn't compute transfer suggestions this run ({transfer_error}). Everything else on this page "
             f"is unaffected — try Run Model again, and if it repeats, this is worth reporting with that message.")

# Patch 5 — simplified primary display: just the recommendation, in plain
# language, front and center. All the rule-citation trace and the raw move
# table that used to be the primary content now live behind one expander,
# available on demand rather than shown by default.
if rec.get("is_weekly_schedule"):
    st.caption(f"No-hits + {horizon}-GW horizon → this is a chained, week-by-week pacing plan (each week's move "
               f"assumes every earlier week's suggested move already happened), not a single this-week decision. "
               f"Free-transfer accrual (+1/week, cap 5) is modeled explicitly below.")
if rec.get("summary"):
    for line in rec["summary"]:
        st.markdown(f'<div class="tx-reco">{line}</div>', unsafe_allow_html=True)
else:
    st.info("No squad/pool data to plan against this run.")

st.caption(f"Two separate bars gate a transfer: a **{rec['minimum_meaningful_gain_free']} xPts** materiality bar "
           f"(is the gain worth spending a free transfer at all) and a **{rec.get('margin_of_error', 2.0):.1f} xPts** "
           f"margin-of-error floor (is the gain distinguishable from this model's own known projection noise — "
           f"Standing Rule #34, not adjustable via the sidebar slider). A move must clear BOTH to be recommended.")

with st.expander("Why — full trace, rule references, and move-by-move detail"):
    st.caption(f"Style profile: **{style_name}** · hit-cost threshold **{rec['hit_cost_threshold']} xPts** · "
               f"free-transfer materiality bar **{rec['minimum_meaningful_gain_free']} xPts** · "
               f"margin-of-error floor **{rec.get('margin_of_error', 2.0):.1f} xPts** · "
               f"free transfers available: **{ft['free_transfers']}** (bank £{bank}m) · horizon **{horizon} GW**")
    # Patch 14 — Standing Rule #4 disclosure: whether the recency signal
    # behind the xM Floor Rule's Rule #19 check was actually available this
    # run, not just assumed. If it wasn't, any player's "confirmed nailed"
    # floor this run is on the pre-Patch-14 season-total basis only.
    checked_gws = getattr(snap, "recent_start_checked_gws", None)
    if checked_gws:
        st.caption(f"Recency check (Standing Rule #19, Bench GK Verification): confirmed-start xM floors this run "
                   f"required an actual start in GW{checked_gws[0]}–GW{checked_gws[-1]} — a player who started "
                   f"earlier this season but not recently no longer gets an automatic 'nailed' floor.")
    else:
        st.caption("⚠️ Recency check (Standing Rule #19) unavailable this run — the per-gameweek live data needed "
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
        st.dataframe(moves_df[show_cols], hide_index=True, use_container_width=True)

        # "New numbers if you make this move" (2026-09-07 discussion, Patch
        # 23). Scoped to the single-decision recommendation only for now
        # ("Hit if worth it"/"Force" at any horizon, or "No hits" at
        # horizon=1) — the "No hits" chained multi-week schedule
        # (rec["is_weekly_schedule"]) isn't covered yet, since "which
        # week's squad" is itself ambiguous there in a way it isn't here.
        # Per the manager's explicit design: the header stats above are
        # NEVER touched by this — this is a separate, local preview. The
        # post-transfer squad's GW xPts and its rating are both re-derived
        # from scratch (best XI re-solved on the changed 15, per Standing
        # Rule #18 -- never assume the old XI just carries over), and the
        # rating is compared against the SAME Free Hit optimal total
        # already computed for planning_gw this run (Patch 22) -- not a
        # fresh reachable-ceiling solve, per the manager's own direction.
        if not rec.get("is_weekly_schedule") and "out_code" in moves_df.columns and "in_code" in moves_df.columns:
            move_out_codes = set(moves_df["out_code"])
            move_in_codes = set(moves_df["in_code"])
            post_transfer_squad = pd.concat([
                squad_df[~squad_df["code"].isin(move_out_codes)],
                proj[proj["code"].isin(move_in_codes)],
            ], ignore_index=True, sort=False)
            if "code" in post_transfer_squad.columns:
                post_transfer_squad = post_transfer_squad.drop_duplicates(subset=["code"], keep="first")

            new_xi_result = opt.best_starting_xi(post_transfer_squad, fh_auto_col) \
                if fh_auto_col in post_transfer_squad.columns else None
            new_gw_xpts = round(new_xi_result["total"], 1) if new_xi_result else None
            new_current_val = opt.rating_gw_value(post_transfer_squad, fh_auto_col, cfg)["total_realized"]
            new_rating = eng.team_rating_pct(new_current_val, fh_auto_optimal_val, "")

            if new_gw_xpts is not None and new_rating["rating_pct"] is not None:
                st.markdown(f'<div class="tx-reco">📈 If you make this move — new GW{planning_gw} xPts: '
                            f'**{new_gw_xpts:.1f}** (was {gw_xpts_total:.1f}) · new Free Hit rating: '
                            f'**{new_rating["rating_pct"]}%** (was {fh_auto_rating["rating_pct"]}%, vs. the same '
                            f'GW{planning_gw} Free Hit optimal shown at the top). These are a local preview for '
                            f'this recommendation only — the header stats above are unaffected until you actually '
                            f'make the transfer and re-run.</div>', unsafe_allow_html=True)
            else:
                st.info("Couldn't compute the post-transfer preview this run (no valid XI for the resulting "
                        "squad this gameweek).")

# ---------------------------------------------------------------------------
# Evaluate your own scenario (Patch 6) — manager-directed what-ifs, always
# shown ALONGSIDE the model's own default recommendation above, never in
# place of it (Standing Rule #30: a scope-restricted comparison must be
# stated as one, not presented as if it were the model's own full-pool
# pick). Nothing chosen here changes anything above — this section is
# purely additive. Gated behind an explicit button rather than re-running
# on every widget change, since each evaluation is a fresh MILP solve.
# ---------------------------------------------------------------------------
with st.expander("Evaluate your own scenario — a specific target, a candidate Wildcard date, or a Free Hit GW"):
    st.caption("Optional. Pick a target player, a candidate Wildcard gameweek, and/or a candidate Free Hit "
               "gameweek below, then click Evaluate. Leave all on \"— none —\" and nothing changes — the "
               "recommendation above stays the model's own default full-pool pick.")
    scen_col1, scen_col2, scen_col3 = st.columns(3)
    with scen_col1:
        pool_options = [(None, "— none —")]
        if not pool_df.empty:
            pool_sorted = pool_df.sort_values("web_name")
            pool_options += [(r["code"], f"{r['web_name']} ({r.get('team','')}) · £{r.get('price','?')}m")
                              for _, r in pool_sorted.iterrows()]
        target_choice = st.selectbox("Target player to bring in", options=pool_options,
                                      format_func=lambda t: t[1], key="scenario_target")
    with scen_col2:
        wc_gw_options = [None] + list(range(planning_gw, 39))
        wc_gw_choice = st.selectbox("Candidate Wildcard gameweek", options=wc_gw_options,
                                     format_func=lambda g: "— none —" if g is None else f"GW{g}",
                                     key="scenario_wc_gw")
    with scen_col3:
        fh_gw_options = [None] + list(range(planning_gw, 39))
        fh_gw_choice = st.selectbox("Candidate Free Hit gameweek", options=fh_gw_options,
                                     format_func=lambda g: "— none —" if g is None else f"GW{g}",
                                     key="scenario_fh_gw")
    if st.button("Evaluate scenario"):
        if target_choice[0] is None and wc_gw_choice is None and fh_gw_choice is None:
            st.info("Nothing selected — pick a target player, a Wildcard gameweek, and/or a Free Hit "
                    "gameweek above first.")
        if target_choice[0] is not None:
            target_eval = recommend.evaluate_target_transfer(
                squad_df, pool_df, cfg, style_name, hit_stance, ft["free_transfers"], bank,
                planning_gw, gw_list, target_choice[0], default_net_gain=rec.get("net_gain"))
            st.markdown("**Target player scenario**")
            if target_eval["summary"]:
                for line in target_eval["summary"]:
                    st.markdown(f'<div class="tx-reco">🧪 {line}</div>', unsafe_allow_html=True)
            if target_eval["moves"]:
                st.dataframe(pd.DataFrame(target_eval["moves"])[
                    [c for c in ["out", "in", "position", "xpts_gain", "hit_cost", "net_gain", "justified"]
                     if c in pd.DataFrame(target_eval["moves"]).columns]], hide_index=True, use_container_width=True)
        if wc_gw_choice is not None:
            # Wildcard-list feature (2026-09-07 discussion): a Wildcard
            # resets your whole squad for the rest of the season, so
            # evaluating it against a 1-GW window (whatever the sidebar
            # horizon happens to be set to) is a bad basis for a decision
            # this size — always use at least 3 GWs, disclosed explicitly
            # whenever that overrides the sidebar's own setting.
            wc_horizon = max(3, horizon)
            future_gw_list = list(range(wc_gw_choice, wc_gw_choice + wc_horizon))
            future_proj = _project(snap, hist_df, overrides, cfg, future_gw_list)
            future_squad_proj = future_proj[future_proj["code"].isin(squad_codes)].copy()
            future_pool_proj = future_proj[~future_proj["code"].isin(squad_codes)].copy()
            wc_eval = chip_protocol.evaluate_wildcard_whatif(future_squad_proj, future_pool_proj, cfg,
                                                              team_value, future_gw_list)
            st.markdown(f"**Wildcard what-if — GW{wc_gw_choice}**")
            if wc_horizon != horizon:
                st.caption(f"Evaluated over GW{future_gw_list[0]}–GW{future_gw_list[-1]} ({wc_horizon} GWs) — "
                           f"a 3-GW minimum applies to Wildcard rebuilds regardless of the sidebar horizon "
                           f"(currently {horizon} GW).")
            if not wc_eval["feasible"]:
                st.info(f"Couldn't solve a rebuild for GW{wc_gw_choice} this run (projection data may not "
                        f"reach that far yet).")
            else:
                gap = wc_eval["gap"]
                st.markdown(f'<div class="tx-reco">🧪 If played at GW{wc_gw_choice}: a full rebuild projects '
                            f'{wc_eval["rebuild_total"]:.1f} xPts vs {wc_eval["hold_total"]:.1f} xPts holding your '
                            f'current squad, over the same {len(future_gw_list)}-GW window ({gap:+.1f} xPts). '
                            f'Informational only — Wildcard timing stays your own call (Standing Rule #24), '
                            f'never a play/hold verdict from this model.</div>', unsafe_allow_html=True)

                wc_gw_col = f"xpts_gw{wc_gw_choice}"
                full_pool_future = pd.concat([future_squad_proj, future_pool_proj], ignore_index=True, sort=False)
                if "code" in full_pool_future.columns:
                    full_pool_future = full_pool_future.drop_duplicates(subset=["code"], keep="first")
                styled_squad = recommend.apply_style_to_wildcard_squad(
                    future_squad_proj, wc_eval["rebuild_squad"], full_pool_future, style_name, cfg, wc_gw_col)

                xi_result = opt.best_starting_xi(styled_squad, wc_gw_col) if wc_gw_col in styled_squad.columns \
                    else None
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
                    st.dataframe(xi_show, hide_index=True, use_container_width=True)
                    if not xi_df.empty:
                        cap_row = xi_df.sort_values(wc_gw_col, ascending=False).iloc[0]
                        st.caption(f"Suggested captain for GW{wc_gw_choice}: **{cap_row['web_name']}** "
                                   f"({cap_row[wc_gw_col]:.1f} projected xPts that week).")
                    st.markdown("**Bench**")
                    bench_show = wc_bench_df.sort_values(["position", wc_gw_col], ascending=[True, False])[show_cols] \
                        .rename(columns=col_rename)
                    st.dataframe(bench_show, hide_index=True, use_container_width=True)
                else:
                    st.markdown(f"**Recommended Wildcard squad — GW{wc_gw_choice}** (full 15)")
                    full_show = styled_squad.sort_values(["position", wc_gw_col], ascending=[True, False])[show_cols] \
                        .rename(columns=col_rename) if wc_gw_col in styled_squad.columns else styled_squad
                    st.dataframe(full_show, hide_index=True, use_container_width=True)
                st.caption(f"Style profile **{style_name}** applied to this rebuild (same EO-pull tie-break as "
                           f"ordinary transfers). Prices, injuries and fixtures can move before GW{wc_gw_choice} "
                           f"— re-run this closer to the date rather than treating it as locked in.")

        if fh_gw_choice is not None:
            # Free Hit "optimal team for this GW" feature (2026-09-07
            # discussion, Patch 19). Unlike the Wildcard what-if above (a
            # non-reverting rebuild evaluated over a 3-GW-minimum horizon,
            # Rule #24 flag-only), a Free Hit squad reverts after one week
            # (Standing Rule #25 / Horizon-Matching Rule) — so this is
            # single-GW only, and it deliberately optimizes differently:
            # highest-scoring legal Starting XI + cheapest legal bench
            # (optimizer.solve_xi_first_squad via
            # data_pipeline.solve_free_hit_optimal_squad), not a raw
            # 15-man-sum rebuild like the Chip Advisor's own play/hold
            # verdict solve uses. No Style Profile EO-pull is applied here
            # (unlike the Wildcard squad above) — this shows the model's
            # single best squad for one specific week, not a season-shaping
            # decision the manager's differential-risk profile should bend.
            # proj only covers the sidebar horizon's gw_list — fh_gw_choice can
            # be well beyond that (same reason the Wildcard block above
            # re-projects onto its own future_gw_list rather than reusing
            # proj), so re-project fresh for just this one target GW.
            fh_col = f"xpts_gw{fh_gw_choice}"
            fh_proj = _project(snap, hist_df, overrides, cfg, [fh_gw_choice])
            if fh_col not in fh_proj.columns:
                st.info(f"No projection reaches GW{fh_gw_choice} yet this run — try a nearer gameweek.")
            else:
                fh_result = data_pipeline.solve_free_hit_optimal_squad(cfg, fh_proj, team_value, fh_gw_choice)
                st.markdown(f"**Free Hit optimal squad — GW{fh_gw_choice}**")
                if fh_result is None:
                    st.info(f"Couldn't solve an optimal Free Hit squad for GW{fh_gw_choice} this run "
                            f"(projection data may not reach that far yet, or no feasible squad fit the "
                            f"budget/club constraints).")
                else:
                    fh_squad = fh_result["squad"]
                    fh_xi = fh_squad[fh_squad["code"].isin(fh_result["xi_codes"])]
                    fh_bench = fh_squad[~fh_squad["code"].isin(fh_result["xi_codes"])]
                    d, m, f = fh_result["shape"]
                    fh_show_cols = ["web_name", "team", "position", "price", fh_col]
                    fh_col_rename = {"web_name": "Player", "team": "Team", "position": "Pos",
                                      "price": "£m", fh_col: f"xPts GW{fh_gw_choice}"}
                    st.markdown(f"Starting XI (formation 1-{d}-{m}-{f}, XI cost "
                                f"£{fh_xi['price'].sum():.1f}m, bench cost £{fh_result['bench_cost']:.1f}m, "
                                f"total £{fh_result['total_cost']:.1f}m of £{team_value:.1f}m available)")
                    fh_xi_show = fh_xi.sort_values(["position", fh_col], ascending=[True, False])[fh_show_cols] \
                        .rename(columns=fh_col_rename)
                    st.dataframe(fh_xi_show, hide_index=True, use_container_width=True)
                    if not fh_xi.empty:
                        fh_cap_row = fh_xi.sort_values(fh_col, ascending=False).iloc[0]
                        st.caption(f"Suggested captain for GW{fh_gw_choice}: **{fh_cap_row['web_name']}** "
                                   f"({fh_cap_row[fh_col]:.1f} projected xPts that week).")
                    st.markdown("**Bench** (deliberately cheap — a Free Hit's bench only matters if an "
                                "autosub fires, so budget is routed to the XI above instead)")
                    fh_bench_show = fh_bench.sort_values(["position", fh_col], ascending=[True, False])[fh_show_cols] \
                        .rename(columns=fh_col_rename)
                    st.dataframe(fh_bench_show, hide_index=True, use_container_width=True)
                    st.caption(f"Optimized for GW{fh_gw_choice} only (a Free Hit squad reverts after this "
                               f"gameweek, per Rule #25) — this is the model's single best squad for that week, "
                               f"not season-shaping, so no Style Profile differential pull is applied. Prices, "
                               f"injuries and fixtures can move before GW{fh_gw_choice} — re-run this closer to "
                               f"the date rather than treating it as locked in.")

                    # Rating vs. FH optimal (2026-09-07 discussion) — same
                    # rating_gw_value() mechanic Patch 20 uses for the main
                    # Team Rating % (best XI + captain doubled + Rule #12
                    # bench discount), applied here to a genuinely
                    # unconstrained single-GW ceiling instead of the main
                    # rating's free-transfer-limited "reachable ceiling."
                    # That sidesteps the exact distortion flagged earlier
                    # this session: this comparison isn't capped to "the one
                    # best swap available," it's your actual current squad
                    # against a true from-scratch optimal for this one week
                    # — a cleaner read on "how far off is my squad, really."
                    current_squad_at_fh_gw = fh_proj[fh_proj["code"].isin(squad_codes)]
                    fh_current_val = opt.rating_gw_value(current_squad_at_fh_gw, fh_col, cfg)["total_realized"]
                    fh_optimal_val = opt.rating_gw_value(fh_squad, fh_col, cfg)["total_realized"]
                    fh_rating = eng.team_rating_pct(fh_current_val, fh_optimal_val, "")
                    fh_gap = round(fh_optimal_val - fh_current_val, 2)
                    fh_moe = eng.margin_of_error_threshold(fh_optimal_val, cfg)
                    st.markdown("**Your squad vs. this Free Hit optimal**")
                    if fh_rating["rating_pct"] is not None:
                        st.markdown(f"Your current squad's best XI this GW: **{fh_current_val:.1f} xPts** vs. "
                                    f"Free Hit optimal: **{fh_optimal_val:.1f} xPts** → "
                                    f"**{fh_rating['rating_pct']}%** (gap: {fh_gap:.1f} xPts, "
                                    f"margin-of-error threshold: {fh_moe:.1f} xPts)")
                        if fh_gap < fh_moe:
                            st.caption(f"✓ That gap is inside normal weekly noise — your squad is already "
                                       f"effectively at this week's ceiling; a Free Hit's upside here is limited.")
                    else:
                        st.info("Couldn't compute a comparison — your current squad has no valid XI for this GW "
                                "this run.")

# ---------------------------------------------------------------------------
# Captaincy — Patch 4: the standalone "Captaincy Pick" section (two st.metric
# boxes) has been retired. The armband on the pitch card is the primary
# signal; the themed `.cap-caption` line rendered directly under the pitch
# (see the Pitch view section above) carries the "why" — EO%/tier detail is
# still available via the alt-pick's underlying data, just not surfaced as
# its own section any more.
# ---------------------------------------------------------------------------

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
    st.dataframe(pd.DataFrame(ledger_rows), hide_index=True, use_container_width=True)
    if not gw_final:
        st.caption(f"GW{squad_gw}'s Points and Overall rank above are both live, provisional FPL figures "
                   f"(rank uses the same corrected field as the header stat) — bonus points for this gameweek "
                   f"aren't finalized yet, so both can still move. Past rows are each GW's own confirmed, "
                   f"finalized value and won't change.")
else:
    st.caption("No season history yet — nothing finished before GW1.")

# ---------------------------------------------------------------------------
# Manager style fit
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">Manager Style Fit</div>', unsafe_allow_html=True)
st.markdown(f"**{style_name}** — {style_profiles.get_profile(style_name)['description']} "
            f"Ownership is never a reason on its own to prefer a pick — the EO weighting above only breaks ties "
            f"once xPts is already close, and any differential still has to clear the pool-average floor on merit.")
