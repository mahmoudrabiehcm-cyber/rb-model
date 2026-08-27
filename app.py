"""
app.py — RB Model
Streamlit front end for the FPL Projection Model v4.0. Zero-cost: official
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
import style_profiles
import chip_protocol
import transfers
import recommend

st.set_page_config(page_title="RB Model", page_icon="♟️", layout="wide")

# ---------------------------------------------------------------------------
# Style — same palette/type system as the design mockup ("chess soul": terse
# notation-style micro-copy, restraint, one accent per screen).
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
  --shadow:0 1px 2px rgba(20,30,22,.07), 0 8px 22px -12px rgba(20,30,22,.24);
}
html, body, [class*="css"]{ font-family:"IBM Plex Sans",sans-serif; color:var(--ink); }
.mono{ font-family:"IBM Plex Mono",monospace; }
.stApp{ background:var(--bg); }

.brand-mark{ font-family:"Fraunces"; font-weight:900; font-size:2rem; line-height:1; margin-bottom:2px; }
.brand-mark .b2{ color:var(--accent-strong); }
.brand-tag{ font-family:"IBM Plex Mono"; font-size:10.5px; color:var(--ink-faint); letter-spacing:.06em; text-transform:uppercase; }

.verdict-card{
  background:linear-gradient(160deg, var(--surface) 55%, var(--accent-tint));
  border:1px solid var(--rule); border-left:4px solid var(--accent);
  box-shadow:var(--shadow); padding:18px 20px; margin-bottom:6px;
}
.verdict-card .h{ font-family:"Fraunces"; font-weight:800; font-size:1.35rem; margin:0 0 6px; color:var(--accent-strong); }
.verdict-card .b{ margin:0; color:var(--ink-muted); font-size:.94rem; font-style:italic; }

.stat-row{ display:flex; gap:28px; font-family:"IBM Plex Mono"; margin:14px 0 26px; flex-wrap:wrap; }
.stat .n{ font-size:1.5rem; font-weight:600; }
.stat .l{ font-size:10.5px; color:var(--ink-faint); text-transform:uppercase; letter-spacing:.06em; }
.trend-up{ color:var(--accent); } .trend-down{ color:var(--warn); }

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
.card{ background:var(--surface); border:1px solid var(--rule); border-top:4px solid var(--team,var(--accent));
  box-shadow:var(--shadow); width:104px; padding:10px 6px 8px; text-align:center; position:relative; }
.card .cap{ position:absolute; top:-9px; right:-9px; width:20px; height:20px; border-radius:50%;
  background:var(--gold); color:#241A05; font-family:"IBM Plex Mono"; font-size:10.5px; font-weight:700;
  display:flex; align-items:center; justify-content:center; box-shadow:var(--shadow); z-index:2; }
.card .sp{ position:absolute; top:5px; left:5px; font-family:"IBM Plex Mono"; font-size:7.5px; font-weight:700;
  color:var(--gold); border:1px solid var(--gold); border-radius:2px; padding:0 3px; }
.avatar-wrap{ width:42px; height:42px; margin:0 auto 6px; position:relative; }
.avatar-wrap img{ width:42px; height:42px; border-radius:50%; object-fit:cover; box-shadow:var(--shadow); }
.avatar-fallback{ width:42px; height:42px; border-radius:50%; background:linear-gradient(160deg, var(--team,var(--accent)), #12492E);
  color:#fff; font-family:"Fraunces"; font-weight:700; font-size:14px; align-items:center; justify-content:center;
  position:absolute; top:0; left:0; }
.card .pos{ display:inline-block; font-family:"IBM Plex Mono"; font-size:9px; font-weight:600; padding:1px 5px; border-radius:2px; margin-bottom:4px; }
.card .pos.gk{ background:var(--gold-tint); color:var(--gold); } .card .pos.def{ background:var(--blue-tint); color:var(--blue); }
.card .pos.mid{ background:var(--accent-tint); color:var(--accent-strong); } .card .pos.fwd{ background:var(--coral-tint); color:var(--coral); }
.card .name{ font-weight:600; font-size:12px; } .card .xp{ font-family:"IBM Plex Mono"; font-size:11px; color:var(--accent-strong); font-weight:600; margin-top:2px; }
.card .price{ font-family:"IBM Plex Mono"; font-size:9.5px; color:var(--ink-faint); }
.bench-strip{ background:var(--bench); margin:0 -14px; padding:12px 14px 4px; border-top:1px dashed var(--rule); }
.bench-strip .card{ opacity:.68; width:92px; }
.side-note{ font-size:11.5px; color:var(--ink-faint); font-family:"IBM Plex Mono"; line-height:1.5; }
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


def _player_card(row: pd.Series, is_captain: bool = False, xp_col: str | None = None) -> str:
    team_color = _team_color(row.get("team", ""))
    initials = "".join([w[0] for w in str(row.get("web_name", "??")).split()][:2]).upper() or "??"
    xp = row.get(xp_col) if xp_col else row.get("xpts_horizon_sum")
    cap_html = '<div class="cap">C</div>' if is_captain else ""
    sp_html = '<div class="sp">SP</div>' if row.get("setpiece_flag") else ""
    pos = str(row.get("position", "")).lower()
    price = row.get("price")
    price_html = f'<div class="price mono">£{price}m</div>' if price is not None else ""
    return f"""<div class="card" style="--team:{team_color}">{cap_html}{sp_html}
      <div class="avatar-wrap">
        <img src="{_photo_url(row.get('code', 0))}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
        <div class="avatar-fallback" style="display:none;">{initials}</div>
      </div>
      <span class="pos {pos}">{row.get('position','')}</span>
      <div class="name">{row.get('web_name','')}</div>
      {price_html}
      <div class="xp">{xp:.1f} xPts</div>
    </div>"""


# ---------------------------------------------------------------------------
# Cached data / compute layers — keyed so a sidebar widget change (style,
# hit-stance) never re-hits the network; only a new team ID or a fresh
# "Run Model" click does.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _load_data(entry_id: int, season: str, prev_season: str):
    snap = fpl_data.load_snapshot(season)
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
    st.markdown('<div class="brand-mark">RB <span class="b2">Model</span></div>'
                '<div class="brand-tag">v4.0 engine · live · zero-cost</div>'
                '<p style="margin:22px 0 14px; color:var(--ink-muted);">Enter your FPL team ID to begin.</p>',
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
    st.markdown('<div class="brand-mark">RB <span class="b2">Model</span></div>'
                '<div class="brand-tag">v4.0 engine · live · zero-cost</div><br>',
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

    run_clicked = st.button("Run Model →", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Run / render
# ---------------------------------------------------------------------------
if run_clicked:
    st.session_state.has_run = True

if not st.session_state.get("has_run"):
    st.markdown('<div class="brand-mark">RB <span class="b2">Model</span></div>', unsafe_allow_html=True)
    st.info("Set your style and hit stance in the sidebar, then click **Run Model** to fetch live data and build your recommendations.")
    st.stop()

with st.spinner("Fetching live data and computing xPts..."):
    snap, hist_df, entry, history = _load_data(entry_id, cfg["meta"]["season"], cfg["meta"]["previous_season"])

    if snap is None or entry is None or history is None:
        st.error("Couldn't reach the official FPL API right now. It's normally free and open with no key required — this is "
                  "most likely a transient outage or a network policy on wherever this app is currently running. Try again shortly.")
        st.stop()

    overrides = eng.load_overrides()
    current_gw = cfg["meta"].get("current_gw_override") or snap.current_gw
    gw_list = list(range(current_gw, current_gw + horizon))

    proj = _project(snap, hist_df, overrides, cfg, gw_list)
    picks = _picks(entry_id, current_gw)

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
    bench_df = squad_df[squad_df["code"].isin(bench_codes)]
    starters_df = squad_df[~squad_df["code"].isin(bench_codes)]
    pool_df = proj[~proj["code"].isin(squad_codes)].copy()

    # Team Rating % (§1a)
    ceiling = data_pipeline.solve_ceiling(cfg, proj)
    squad_total = squad_df["xpts_horizon_sum"].sum() if not squad_df.empty else 0.0
    ceiling_total = ceiling["total_xpts"] if ceiling else 0.0
    rating = eng.team_rating_pct(squad_total, ceiling_total,
                                  "MECHANICAL-TIER — MODEL_POISSON CS%, xM Floor Rule only. "
                                  "Steps 4 (full Role Multiplier table), 4a (Manager Tenure Split), "
                                  "5 (Pre-Season Evidence) and 6 (Manager System Fit) are NOT automated here "
                                  "(they need web research/judgment) — a chat-run review that applies those "
                                  "by hand will read differently. Standing Rules #16/#18 disclosure.")

    # rank history + points from entry history
    cur_hist = history.get("current", []) if history else []
    rank_history = [r.get("overall_rank") for r in cur_hist if r.get("overall_rank") is not None]
    points_total = entry.get("summary_overall_points") if entry else None
    hits_last_3 = sum(1 for r in cur_hist[-3:] if (r.get("event_transfers_cost") or 0) > 0)

    verdict = recommend.chess_verdict(rank_history, hits_last_3, current_gw)

    # free transfers + bank
    ft = transfers.derive_free_transfers(cur_hist, history.get("chips", []) if history else [])
    bank = (entry.get("last_deadline_bank", 0) or 0) / 10.0 if entry else 0.0

    # captaincy — starting XI only, never the bench
    cap_pick_row, cap_alt_row = None, None
    if not starters_df.empty:
        cap_col = f"xpts_gw{gw_list[0]}"
        cap_candidates = starters_df.rename(columns={cap_col: "xpts_this_gw"})[
            ["web_name", "team", "xpts_this_gw", "selected_by_percent"]]
        cap_result = eng.captaincy_protocol(cap_candidates, cfg)
        cap_pick = style_profiles.captaincy_pick(cap_result, style_name)
        shortlist = cap_result[cap_result["shortlisted"]]
        alt_pool = shortlist[shortlist["web_name"] != cap_pick["web_name"]]
        cap_alt_row = alt_pool.sort_values("eo", ascending=True).iloc[0] if not alt_pool.empty else None
        cap_pick_row = cap_pick

    # transfer suggestions — isolated so a bad row here can't take down the
    # rest of the page (pitch view, chip rack, captaincy, ledger all still
    # render even if this section fails).
    transfer_error = None
    try:
        rec = recommend.suggest_transfers(squad_df, pool_df, cfg, style_name, hit_stance,
                                           ft["free_transfers"], bank, current_gw, gw_list, forced_count)
    except Exception as e:
        transfer_error = str(e)
        rec = {"moves": [], "plan": [], "profile_used": style_name,
               "hit_cost_threshold": style_profiles.get_profile(style_name)["hit_cost_threshold"],
               "minimum_meaningful_gain_free": cfg["transfer"].get("minimum_meaningful_gain_free", 1.5),
               "hit_stance": hit_stance, "free_transfers": ft["free_transfers"]}

    # chip status + timing
    boot_chips = fpl_data.fetch_bootstrap_chips(snap.raw_boot) if snap.raw_boot else []
    chips_played = history.get("chips", []) if history else []
    chip_rows = chip_protocol.chip_status(boot_chips, chips_played)
    fixture_counts = chip_protocol.fixture_counts_by_team(snap.fixtures, gw_list)
    all_team_ids = snap.teams["id"].tolist() if "id" in snap.teams.columns else []
    dgw_bgw = chip_protocol.dgw_bgw_flags(fixture_counts, all_team_ids)
    squad_team_ids = squad_df["team_id"].tolist() if "team_id" in squad_df.columns else []
    chip_notes = chip_protocol.chip_recommendations(chip_rows, dgw_bgw, squad_team_ids, max(len(squad_df), 1))
    flagged_players = squad_df[(squad_df["status"] != "a") | (squad_df["est_rescue_needed"])]
    wc_flag = chip_protocol.wildcard_flag(rank_history, len(flagged_players))

# ---------------------------------------------------------------------------
# Header + verdict
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f'<div class="verdict-card"><p class="h">{verdict["headline"]}</p>'
                f'<p class="b">{verdict["body"]}</p></div>', unsafe_allow_html=True)
with col2:
    trend = ""
    if len(rank_history) >= 2:
        trend = '<span class="trend-up">▲</span>' if rank_history[-1] < rank_history[-2] \
            else ('<span class="trend-down">▼</span>' if rank_history[-1] > rank_history[-2] else "")
    rank_disp = f"{rank_history[-1]:,}" if rank_history else "—"
    st.markdown(f"""<div class="stat-row">
      <div class="stat"><div class="n">{rank_disp} {trend}</div><div class="l">Overall rank</div></div>
      <div class="stat"><div class="n">{rating['rating_pct'] if rating['rating_pct'] is not None else '—'}%</div><div class="l">Team rating</div></div>
      <div class="stat"><div class="n">{points_total if points_total is not None else '—'}</div><div class="l">Points</div></div>
    </div>""", unsafe_allow_html=True)

st.markdown(f'<div class="side-note">Source: {snap.source} · GW{current_gw} · '
            f'fetched {dt.datetime.fromtimestamp(snap.fetched_at).strftime("%H:%M")} · '
            f'style profile: <b>{style_name}</b></div>', unsafe_allow_html=True)
if rating["rating_pct"] is not None:
    with st.expander("Team Rating % — data-source tier disclosure (Standing Rules #16/#18)"):
        st.markdown(rating["tier"])
        st.caption(f"Squad horizon xPts: {squad_total:.1f} · Ceiling horizon xPts: {ceiling_total:.1f} "
                   f"(unconstrained £{cfg['squad_rules']['budget']}m, {horizon}-GW horizon)")
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

# ---------------------------------------------------------------------------
# Pitch view
# ---------------------------------------------------------------------------
st.markdown(f'<div class="section-h">Squad · GW{current_gw}</div>', unsafe_allow_html=True)
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
            pitch_html += _player_card(r, is_captain=(r["code"] == captain_id))
        pitch_html += '</div>'
    if not bench_df.empty:
        pitch_html += '<div class="bench-strip"><div class="side-note">BENCH</div><div class="prow">'
        for _, r in bench_df.sort_values("xpts_horizon_sum", ascending=False).iterrows():
            pitch_html += _player_card(r)
        pitch_html += '</div></div>'
    pitch_html += '</div>'
    st.markdown(pitch_html, unsafe_allow_html=True)
    st.markdown('<p class="side-note">SP tag = newly confirmed set-piece role, decaying out as current-season minutes accrue.</p>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Transfer recommendations
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">Transfer Recommendations</div>', unsafe_allow_html=True)
if transfer_error:
    st.error(f"Couldn't compute transfer suggestions this run ({transfer_error}). Everything else on this page "
             f"is unaffected — try Run Model again, and if it repeats, this is worth reporting with that message.")
st.caption(f"Style profile: **{style_name}** · hit-cost threshold **{rec['hit_cost_threshold']} xPts** · "
           f"free-transfer materiality bar **{rec['minimum_meaningful_gain_free']} xPts** · "
           f"free transfers available: **{ft['free_transfers']}** (bank £{bank}m) · horizon **{horizon} GW**")
with st.expander("How the free-transfer count was derived"):
    for line in ft["trace"]:
        st.markdown(f"- {line}")

st.markdown("**Plan**")
if rec["plan"]:
    for line in rec["plan"]:
        st.markdown(f"- {line}")
else:
    st.info("No squad/pool data to plan against this run.")

if rec["moves"]:
    with st.expander("Move-by-move detail"):
        moves_df = pd.DataFrame(rec["moves"])
        show_cols = [c for c in ["out", "in", "position", "xpts_gain_this_gw", "xpts_gain", "in_eo",
                                  "hit_cost", "net_gain", "justified", "setpiece_flag"] if c in moves_df.columns]
        st.dataframe(moves_df[show_cols], hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Captaincy
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">Captaincy Pick</div>', unsafe_allow_html=True)
if cap_pick_row is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"{cap_pick_row['web_name']} ({cap_pick_row['team']})", f"{cap_pick_row['xpts_this_gw']:.1f} xPts",
                   help=f"EO {cap_pick_row['eo']:.1f}% · tier: {cap_pick_row['eo_tier']}")
    with c2:
        if cap_alt_row is not None:
            st.metric(f"Differential: {cap_alt_row['web_name']} ({cap_alt_row['team']})", f"{cap_alt_row['xpts_this_gw']:.1f} xPts",
                       help=f"EO {cap_alt_row['eo']:.1f}% · tier: {cap_alt_row['eo_tier']}")
        else:
            st.caption("No lower-EO alternative inside the shortlist window this week.")
else:
    st.info("No squad data to run the captaincy protocol against this run.")

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
        ledger_rows.append({"GW": gw, "Move": move, "Points": r.get("points"), "Overall rank": r.get("overall_rank")})
    st.dataframe(pd.DataFrame(ledger_rows), hide_index=True, use_container_width=True)
else:
    st.caption("No season history yet — nothing finished before GW1.")

# ---------------------------------------------------------------------------
# Manager style fit
# ---------------------------------------------------------------------------
st.markdown('<div class="section-h">Manager Style Fit</div>', unsafe_allow_html=True)
st.markdown(f"**{style_name}** — {style_profiles.get_profile(style_name)['description']} "
            f"Ownership is never a reason on its own to prefer a pick — the EO weighting above only breaks ties "
            f"once xPts is already close, and any differential still has to clear the pool-average floor on merit.")
