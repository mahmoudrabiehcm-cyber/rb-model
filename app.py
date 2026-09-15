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

# Patch 38 (2026-09-14, manager report: "we spent the whole day explaining
# the logic and still the same issue" — the real cause across that whole day
# was never being able to tell, from a screenshot alone, whether a fix had
# actually been redeployed or whether a number was legitimately different
# live data): a permanent, visible version stamp so that question is
# answerable at a glance, without another round of screenshots. Bump this
# with every patch that ships to the manager.
PATCH_VERSION = "Patch 47"

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

.stat-row{ display:flex; gap:28px; font-family:"IBM Plex Mono"; margin:14px 0 26px; flex-wrap:nowrap; align-items:flex-start; overflow-x:auto; }
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
.signal-card .stat{ font-family:"Fraunces"; font-weight:800; font-size:1.4rem; color:var(--accent-strong); line-height:1; }
.signal-card .sub{ font-size:10.5px; color:var(--ink-faint); margin-top:4px; }
.badge{ font-family:"IBM Plex Mono"; font-size:9px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  padding:2px 7px; border-radius:20px; display:inline-block; white-space:nowrap; }
.badge.play{ background:var(--accent-tint); color:var(--accent-strong); }
.badge.active{ background:var(--gold-tint); color:#7A5A16; }
.badge.hold{ background:var(--surface-2); color:var(--ink-faint); }
.badge.caution{ background:var(--coral-tint); color:var(--coral); }
.badge.used{ background:var(--surface-2); color:var(--ink-faint); }

.flag-row{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 16px; }
.flag-pill{ display:flex; align-items:center; gap:6px; background:var(--coral-tint); border:1px solid var(--coral);
  color:#7A2E18; font-family:"IBM Plex Mono"; font-size:11px; padding:4px 10px; border-radius:20px; cursor:help; }

/* Team Rating % radial gauge (Patch 31) — conic-gradient ring, no SVG/JS
   library needed. Percentage is still the same number the tooltip/expander
   already computed; this only changes how it's presented. */
.gauge-wrap{ display:flex; align-items:center; gap:10px; }
.gauge-ring{ width:44px; height:44px; border-radius:50%; flex:none;
  display:flex; align-items:center; justify-content:center; position:relative; }
.gauge-ring::before{ content:""; position:absolute; inset:5px; border-radius:50%; background:var(--bg); }
.gauge-ring .gauge-val{ position:relative; z-index:1; font-family:"IBM Plex Mono"; font-size:10.5px; font-weight:700; }

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


def _signal_card(name: str, badge_text: str, badge_cls: str, stat: str, sub: str,
                  tooltip: str, card_cls: str = "", by_gw: dict | None = None,
                  best_gw: int | None = None) -> str:
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
    this GW" is answered by the numbers themselves, not just a sentence."""
    body = (f'<div class="top"><span class="name">{name}</span>'
            f'<span class="badge {badge_cls}">{badge_text}</span></div>'
            f'<div class="stat">{stat}</div>'
            f'<div class="sub">{sub}</div>')
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


def _flag_pill(text: str, tooltip: str = "") -> str:
    """Patch 31 — a single compact pill for Chip Rack notes (Wildcard flag,
    shape-test, disruption, price-drop-flow) that used to render as a full
    <p class="side-note"> paragraph each. Full detail stays available on
    hover rather than being deleted."""
    icon = "⚠️" if any(k in text for k in ("CAUTION", "flagged", "Disruption")) else "●"
    short = text if len(text) <= 70 else text[:67] + "…"
    return f'<span class="flag-pill" title="{tooltip or text}">{icon} {short}</span>'


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
                f'<div class="brand-tag">v5.0 engine · {PATCH_VERSION.lower()} · live · zero-cost</div><br>',
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
    _wc_available_now = any(r["status"] == "available" and r["chip"].startswith("Wildcard") for r in chip_rows)
    _fh_available_now = any(r["status"] == "available" and r["chip"].startswith("Free Hit") for r in chip_rows)
    available_chip_names = {r["chip"] for r in chip_rows if r["status"] == "available"}

    detect_gw_list = None
    if _wc_available_now or _fh_available_now:
        shape_cfg = cfg.get("chip_shape_test", {})
        detect_window = shape_cfg.get("detection_window_gws", 4)
        detect_gw_list = list(range(planning_gw, planning_gw + detect_window))

    chip_adv_window = None
    chip_adv_gw_list = None
    if any(c.startswith(("Bench Boost", "Triple Captain", "Free Hit")) for c in available_chip_names):
        chip_adv_window = chip_protocol.chip_advisor_gw_window(planning_gw, snap.fixtures, all_team_ids, cfg)
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
    _gw_union = sorted(set(gw_list) | set(detect_gw_list or []) | set(chip_adv_gw_list or [])
                        | {_tie_break_lookahead_gw})
    proj = _project(snap, hist_df, overrides, cfg, _gw_union)
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
    # Patch 24 (2026-09-08 discussion) — this FH-optimal-based ratio now IS
    # the header's "Team Rating" (manager's explicit call: it's a truer
    # current-vs-optimal read than the old reachable-ceiling version, which
    # is tautologically high whenever few free transfers are banked). The
    # old reachable-ceiling calc (`rating`/squad_total/reachable_total/moe)
    # is kept as-is for the Wildcard-flag trigger only (chip_protocol.wildcard_flag
    # below) — untouched, not displayed anywhere any more.
    fh_auto_tooltip = (f"Team Rating % = your current squad's best XI this GW (captain doubled, bench "
                       f"autosub-discounted), divided by a genuinely unconstrained optimal squad for GW"
                       f"{planning_gw} only (full player pool, no free-transfer limit — a true from-scratch "
                       f"rebuild). Gap: {fh_auto_gap:.1f} xPts (margin-of-error threshold: {fh_auto_moe:.1f} xPts). "
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

    if _wc_available_now and shape_proj is not None:
        squad_detect = shape_proj[shape_proj["code"].isin(squad_codes)]
        reachable_detect = data_pipeline.solve_reachable_ceiling(cfg, shape_proj, squad_codes, ft["free_transfers"])
        wc_trigger = eng.wildcard_trigger_check(
            squad_detect, reachable_detect["squad"] if reachable_detect else None, detect_gw_list, cfg)

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
        shape_test = chip_protocol.wildcard_freehit_shape_test(
            squad_df, shape_proj, cfg, detect_gw_list, team_value)

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

    bb_advisor = None
    if any(c.startswith("Bench Boost") for c in available_chip_names):
        bb_advisor = chip_protocol.evaluate_bench_boost(
            bench_df_adv, chip_adv_window["gw_list"], _chip_moe_fn("bench_boost"))
    tc_advisor = None
    if any(c.startswith("Triple Captain") for c in available_chip_names):
        tc_advisor = chip_protocol.evaluate_triple_captain(
            starters_df_adv, chip_adv_window["gw_list"], _chip_moe_fn("triple_captain"))
    fh_advisor = None
    if any(c.startswith("Free Hit") for c in available_chip_names) and not squad_df.empty:
        fh_advisor = chip_protocol.evaluate_free_hit(
            squad_df_adv, chip_adv_window["gw_list"],
            lambda gw: data_pipeline.solve_free_hit_rebuild(cfg, chip_adv_proj, team_value, gw),
            _chip_moe_fn("free_hit"))

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
    _bb_play_gw = int(bb_advisor["verdict"].split("gw")[1]) \
        if bb_advisor and bb_advisor["verdict"].startswith("play_gw") else None
    _chip_capped_gw_list = None
    if planned_chip_gw is not None and transfer_gw_list and planned_chip_gw <= transfer_gw_list[-1]:
        _chip_capped_gw_list = [g for g in transfer_gw_list if g < planned_chip_gw]

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
        {"flagged_starting": False}

    transfer_error = None
    try:
        rec = recommend.suggest_transfers(squad_df, pool_df, cfg, style_name, hit_stance,
                                           ft["free_transfers"], bank, planning_gw, transfer_gw_list, forced_count,
                                           meaningful_bar_override, set(bench_df["code"]), chip_advisory,
                                           bb_play_gw=_bb_play_gw, chip_capped_gw_list=_chip_capped_gw_list,
                                           disrupted_codes=_disrupted_codes)
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
_wc_check_note = None
if wc_flag and reachable_detect is not None and not squad_df.empty:
    _moves_all = rec.get("moves") or []
    if _moves_all:
        _out_codes = {m["out_code"] for m in _moves_all}
        _in_codes = {m["in_code"] for m in _moves_all}
        _after_plan_squad = pd.concat(
            [squad_df[~squad_df["code"].isin(_out_codes)], proj[proj["code"].isin(_in_codes)]],
            ignore_index=True, sort=False)
    else:
        _after_plan_squad = squad_df
    _check_gws = [g for g in transfer_gw_list
                  if f"xpts_gw{g}" in _after_plan_squad.columns and f"xpts_gw{g}" in reachable_detect["squad"].columns]
    if _check_gws:
        _ratings = []
        for _g in _check_gws:
            _col = f"xpts_gw{_g}"
            _sv = opt.rating_gw_value(_after_plan_squad, _col, cfg)["total_realized"]
            _rv = opt.rating_gw_value(reachable_detect["squad"], _col, cfg)["total_realized"]
            _rp = eng.team_rating_pct(_sv, _rv, "")["rating_pct"]
            if _rp is not None:
                _ratings.append(_rp)
        if _ratings:
            _avg_after = round(sum(_ratings) / len(_ratings), 1)
            _wc_ceiling = cfg.get("wildcard_trigger", {}).get("team_rating_pct_ceiling", 79.0)
            _closes = _avg_after >= _wc_ceiling
            _plan_desc = (f"the {len(_moves_all)}-move plan" if _moves_all else "no transfer (this run rolls)")
            _wc_check_note = (
                f"Cross-check against your own recommended transfer plan ({hit_stance}, {_plan_desc}, "
                f"GW{_check_gws[0]}-GW{_check_gws[-1]}): if followed in full, your squad's average Team "
                f"Rating % over that span is projected to rise to {_avg_after}% (currently "
                f"{wc_trigger['avg_rating_pct']}%) — "
                + (f"already at/above the {_wc_ceiling:.0f}% trigger ceiling, so ordinary transfers may close "
                   f"this gap on their own, without needing the Wildcard — worth checking before committing it."
                   if _closes else
                   f"still below the {_wc_ceiling:.0f}% trigger ceiling even after the plan, so this looks like "
                   f"a structural gap ordinary transfers alone won't close, not just a few weeks away."))

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
        _gauge_html = '<span class="info-dot" title="Not enough data this run">—</span>'
    st.markdown(f"""<div class="stat-row">
      <div class="stat"><div class="n">{rank_disp} {trend}{prov_badge}</div><div class="l">Overall rank</div></div>
      <div class="stat rating">
        <div class="n">{_gauge_html}</div>
        <div class="l">Team rating</div>
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
    if fh_at_ceiling:
        st.caption(f"✓ Already at this GW's optimal — the {fh_auto_gap:.1f} xPts gap is inside normal weekly "
                   f"noise (threshold {fh_auto_moe:.1f} xPts), not real room left on the table.")
    with st.expander("Team Rating % — full breakdown"):
        st.markdown(tier_label)
        st.caption(f"GW{planning_gw} xPts (your current squad's best XI, captain doubled, bench autosub-discounted): "
                   f"{fh_auto_current_val:.1f} · GW{planning_gw} optimal (a genuinely unconstrained best-possible "
                   f"squad from the full player pool, £{team_value}m proxy budget, no free-transfer limit): "
                   f"{fh_auto_optimal_val:.1f} · gap: {fh_auto_gap:.1f} xPts (margin-of-error threshold: "
                   f"{fh_auto_moe:.1f} xPts)")
        st.caption("Patch 24 methodology (2026-09-08): Team Rating % = current squad GW xPts / GW-optimal xPts, "
                   "both sides using the same best-legal-Starting-XI-plus-captain-bonus-plus-Rule-#12-bench-value "
                   "calculation (Rule #22 Systematic Application) — replacing the prior reachable-ceiling version, "
                   "which the manager flagged as tautologically high whenever few free transfers are banked. This "
                   "is the same figure previously shown as 'Free Hit rating'; it is now the sole Team Rating stat.")
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
much_more = wc_trigger["reason"] if (wc_trigger and not wc_flag and wc_trigger.get("avg_rating_pct") is not None) else None
pill_items = []
if wc_flag:
    pill_items.append(_flag_pill(wc_flag))
    if _wc_check_note:
        pill_items.append(_flag_pill(_wc_check_note))
elif much_more:
    pill_items.append(_flag_pill(f"Wildcard trigger: not active — {much_more}."))
for note in chip_notes:
    pill_items.append(_flag_pill(note))
if shape_test and shape_test["classification"] != "insufficient_data":
    for note in shape_test["notes"]:
        pill_items.append(_flag_pill(note))
for note in disruption["notes"]:
    pill_items.append(_flag_pill(note))
if pill_items:
    st.markdown(f'<div class="flag-row">{"".join(pill_items)}</div>', unsafe_allow_html=True)

# Chip Signals — Patch 31 visual redesign. Replaces the old paragraph-per-
# rule Chip Advisor + Chip Strategy expanders with one scannable card grid;
# every card's rule/step citation and full quantified reasoning lives in its
# hover tooltip (same hover-hidden pattern as the header's .info-dot),
# instead of sitting as permanent visible body text.
def _advisor_card(label: str, adv: dict | None) -> str:
    _win = (f"GW{chip_adv_window['gw_list'][0]}-GW{chip_adv_window['gw_list'][-1]}" if chip_adv_window
            else "the scanned window")
    if adv is None or adv.get("best_gw") is None:
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

wc_card_tooltip = (wc_flag or (f"Wildcard trigger: not active — {much_more}." if much_more else
                                "Insufficient data to evaluate the trigger this run."))
wc_card_tooltip += " Wildcard's trigger condition is mechanical (Standing Rule #24/#41), but the specific play " \
                    "date is never a mechanical verdict — it's a rolling re-test per Standing Rule #32."
if wc_flag and _wc_check_note:
    wc_card_tooltip += " " + _wc_check_note
if wc_flag:
    wc_stat = f'{wc_trigger["avg_rating_pct"]}%' if wc_trigger and wc_trigger.get("avg_rating_pct") is not None else "ACTIVE"
    wc_card = _signal_card("Wildcard", "TRIGGER ACTIVE", "active", wc_stat,
                            "structural gap detected — date is your call", wc_card_tooltip, "is-active")
elif much_more:
    wc_card = _signal_card("Wildcard", "HOLD", "hold", f'{wc_trigger["avg_rating_pct"]}%',
                            "inside noise band — no trigger", wc_card_tooltip)
else:
    wc_card = _signal_card("Wildcard", "N/A", "used", "—", "insufficient data this run", wc_card_tooltip)

signal_html = '<div class="signal-grid">' + wc_card + _advisor_card("Bench Boost", bb_advisor) + \
    _advisor_card("Triple Captain", tc_advisor) + _advisor_card("Free Hit", fh_advisor) + '</div>'
st.markdown(signal_html, unsafe_allow_html=True)

# Full analysis — Patch 29's synthesis narrative, kept in full (nothing
# deleted per manager instruction) but moved behind an opt-in expander now
# that the cards above carry the at-a-glance read (2026-09-14 redesign).
strategy_lines = chip_protocol.chip_strategy_summary(
    wc_flag, shape_test, bb_advisor, tc_advisor, fh_advisor, chip_rows, disruption["notes"], wc_trigger)
if _wc_check_note:
    strategy_lines.append(_wc_check_note)
with st.expander("Full chip analysis — combined narrative, rule-by-rule"):
    for line in strategy_lines:
        st.markdown(f"- {line}")
    st.caption("A synthesis of the Wildcard trigger, shape-test, Chip Advisor verdicts and any disruption notes "
               "above — computes nothing new itself. Wildcard's trigger is mechanical (Patch 30) but never names "
               "a single play GW — the date stays a rolling re-test (Standing Rule #32).")

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
_wc_active = bool(wc_flag)
if _wc_active or _bb_play or _tc_play or _fh_play:
    st.markdown('<div class="section-h">🎯 Team Recommendation — active signals, auto-built</div>',
                unsafe_allow_html=True)

    if _wc_active:
        wc_rebuild_gw = detect_gw_list[0] if detect_gw_list else planning_gw
        _wc_col = f"xpts_gw{wc_rebuild_gw}"
        _full_pool_now = pd.concat([squad_df, pool_df], ignore_index=True, sort=False)
        if "code" in _full_pool_now.columns:
            _full_pool_now = _full_pool_now.drop_duplicates(subset=["code"], keep="first")
        wc_eval_auto = chip_protocol.evaluate_wildcard_whatif(
            squad_df, pool_df, cfg, team_value, detect_gw_list or gw_list)
        with st.expander(f"🃏 Wildcard rebuild — trigger active, shown for GW{wc_rebuild_gw} onward", expanded=False):
            if not wc_eval_auto["feasible"]:
                st.info("Couldn't solve an auto-rebuild this run (projection data may not reach far enough).")
            else:
                styled_wc = recommend.apply_style_to_wildcard_squad(
                    squad_df, wc_eval_auto["rebuild_squad"], _full_pool_now, style_name, cfg, _wc_col)
                gap = wc_eval_auto["gap"]
                st.caption(f"Rebuild projects {wc_eval_auto['rebuild_total']:.1f} xPts vs "
                           f"{wc_eval_auto['hold_total']:.1f} xPts holding your current squad over this window "
                           f"({gap:+.1f} xPts). Style profile **{style_name}** applied. Date remains your own "
                           f"call (Standing Rule #32) — this is the model's current best rebuild if played now.")
                if _wc_col in styled_wc.columns:
                    xi_res = opt.best_starting_xi(styled_wc, _wc_col)
                    cols = ["web_name", "team", "position", "price", _wc_col]
                    rn = {"web_name": "Player", "team": "Team", "position": "Pos", "price": "£m",
                          _wc_col: f"xPts GW{wc_rebuild_gw}"}
                    if xi_res is not None:
                        xi_df, wc_bench = xi_res["xi"], styled_wc[~styled_wc["code"].isin(xi_res["xi"]["code"])]
                        d, m, f = xi_res["shape"]
                        st.markdown(f"**Starting XI** (1-{d}-{m}-{f}, £{styled_wc['price'].sum():.1f}m)")
                        st.dataframe(xi_df.sort_values(["position", _wc_col], ascending=[True, False])[cols]
                                     .rename(columns=rn), hide_index=True, use_container_width=True)
                        if not xi_df.empty:
                            cap = xi_df.sort_values(_wc_col, ascending=False).iloc[0]
                            st.caption(f"Suggested captain: **{cap['web_name']}** ({cap[_wc_col]:.1f} xPts).")
                        st.markdown("**Bench**")
                        st.dataframe(wc_bench.sort_values(["position", _wc_col], ascending=[True, False])[cols]
                                     .rename(columns=rn), hide_index=True, use_container_width=True)

    if _fh_play:
        _fh_gw = int(fh_advisor["verdict"].split("gw")[1])
        _fh_col = f"xpts_gw{_fh_gw}"
        _fh_proj_auto = proj if _fh_col in proj.columns else _project(snap, hist_df, overrides, cfg, [_fh_gw])
        fh_res_auto = data_pipeline.solve_free_hit_optimal_squad(cfg, _fh_proj_auto, team_value, _fh_gw)
        with st.expander(f"🎟️ Free Hit — PLAY GW{_fh_gw}, optimal squad", expanded=False):
            if fh_res_auto is None:
                st.info("Couldn't solve an optimal Free Hit squad this run.")
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
                st.dataframe(fh_xi.sort_values(["position", _fh_col], ascending=[True, False])[cols]
                             .rename(columns=rn), hide_index=True, use_container_width=True)
                if not fh_xi.empty:
                    cap = fh_xi.sort_values(_fh_col, ascending=False).iloc[0]
                    st.caption(f"Suggested captain: **{cap['web_name']}** ({cap[_fh_col]:.1f} xPts).")
                st.markdown("**Bench** (cheap by design — budget routed to the XI)")
                st.dataframe(fh_bn.sort_values(["position", _fh_col], ascending=[True, False])[cols]
                             .rename(columns=rn), hide_index=True, use_container_width=True)

    if _bb_play:
        _bb_gw = int(bb_advisor["verdict"].split("gw")[1])
        _bb_col = f"xpts_gw{_bb_gw}"
        with st.expander(f"🛋️ Bench Boost — PLAY GW{_bb_gw}, full 15", expanded=False):
            if _bb_col in squad_df_adv.columns:
                cols = ["web_name", "team", "position", "price", _bb_col]
                rn = {"web_name": "Player", "team": "Team", "position": "Pos", "price": "£m",
                      _bb_col: f"xPts GW{_bb_gw}"}
                st.caption(f"Clears margin-of-error by {bb_advisor.get('margin', bb_advisor['threshold']):.1f} "
                           f"xPts — every one of your 15 scores this week, bench included. Found by scanning "
                           f"GW{chip_adv_window['gw_list'][0]}-GW{chip_adv_window['gw_list'][-1]}, independent of "
                           f"the sidebar's transfer Horizon.")
                st.dataframe(squad_df_adv.sort_values(["position", _bb_col], ascending=[True, False])[cols]
                             .rename(columns=rn), hide_index=True, use_container_width=True)

    if _tc_play:
        _tc_gw = int(tc_advisor["verdict"].split("gw")[1])
        _tc_col = f"xpts_gw{_tc_gw}"
        with st.expander(f"👑 Triple Captain — PLAY GW{_tc_gw}", expanded=False):
            if _tc_col in starters_df_adv.columns and not starters_df_adv.empty:
                cap_row = starters_df_adv.sort_values(_tc_col, ascending=False).iloc[0]
                st.markdown(f"**{cap_row['web_name']}** ({cap_row.get('team','')}) — "
                            f"{cap_row[_tc_col]:.1f} xPts, tripled to {cap_row[_tc_col]*3:.1f}.")
                st.caption(f"Clears margin-of-error by {tc_advisor.get('margin', tc_advisor['threshold']):.1f} xPts "
                           f"over the next-best captaincy GW in a GW{chip_adv_window['gw_list'][0]}-"
                           f"GW{chip_adv_window['gw_list'][-1]} scan, independent of the sidebar's transfer "
                           f"Horizon.")

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
    return data_pipeline.solve_free_hit_optimal_squad(_cfg, _proj, team_value, gw)


@st.fragment
def _render_pitch_navigator():
    st.markdown(f'<div class="section-h">Squad · planning for GW{planning_gw}</div>', unsafe_allow_html=True)
    if starters_df.empty:
        st.warning("No squad data returned for this team ID / gameweek yet (common right after a deadline, or if "
                   "this is a brand-new team). Transfer targets and captaincy below still use the full player pool.")
        return

    _nav_gw_list = chip_adv_window["gw_list"] if chip_adv_window else gw_list
    if not _nav_gw_list:
        _nav_gw_list = [planning_gw]

    if "nav_gw_idx" not in st.session_state or st.session_state.get("nav_gw_list") != _nav_gw_list:
        st.session_state.nav_gw_idx = _nav_gw_list.index(planning_gw) if planning_gw in _nav_gw_list else 0
        st.session_state.nav_gw_list = _nav_gw_list
    st.session_state.nav_gw_idx = max(0, min(st.session_state.nav_gw_idx, len(_nav_gw_list) - 1))

    # "After recommended transfer" needs an UNAMBIGUOUS single set of moves
    # for the CURRENT planning GW specifically. For a single-decision
    # recommendation that's just `rec["moves"]`. For the chained weekly
    # pacing plan (Patch 39 routes "Hit if worth it" through this too, not
    # just "No hits" — manager report 2026-09-15: this made the toggle
    # disappear far more often than before, since that combination is now
    # common, not rare), using the FULL flattened `rec["moves"]` would wrongly
    # merge every week's swaps together as if they all happened at once — so
    # this only ever previews THIS WEEK's move (`weekly_plan[0]`), which is
    # itself a concrete, unambiguous move exactly like the single-decision
    # case; later weeks' hypothetical chained moves stay out of scope for
    # this preview, same as they always were. Fixed (Patch 34 follow-up):
    # _move_row() previously dropped out_code/in_code entirely, which
    # silently disabled this toggle every run regardless of whether a
    # transfer was recommended.
    if rec.get("is_weekly_schedule"):
        _wk_plan = rec.get("weekly_plan") or []
        _this_week_moves = _wk_plan[0]["moves"] if _wk_plan and _wk_plan[0].get("gw") == planning_gw else []
        _nav_moves_df = pd.DataFrame(_this_week_moves) if _this_week_moves else pd.DataFrame()
    else:
        _nav_moves_df = pd.DataFrame(rec["moves"]) if rec.get("moves") else pd.DataFrame()
    _nav_can_toggle = (not _nav_moves_df.empty
                       and "out_code" in _nav_moves_df.columns and "in_code" in _nav_moves_df.columns)
    _nav_squad_after = None
    if _nav_can_toggle:
        _nav_out_codes = set(_nav_moves_df["out_code"])
        _nav_in_codes = set(_nav_moves_df["in_code"])
        _nav_squad_after = pd.concat([
            squad_df_adv[~squad_df_adv["code"].isin(_nav_out_codes)],
            chip_adv_proj[chip_adv_proj["code"].isin(_nav_in_codes)],
        ], ignore_index=True, sort=False)
        if "code" in _nav_squad_after.columns:
            _nav_squad_after = _nav_squad_after.drop_duplicates(subset=["code"], keep="first")

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

    _nav_options = ["Current squad"]
    _after_tx_label = ("After recommended transfer (this week's move)" if rec.get("is_weekly_schedule")
                        else "After recommended transfer")
    if _nav_can_toggle:
        _nav_options.append(_after_tx_label)
    if _nav_squad_scenario is not None:
        _nav_options.append(f"After evaluated scenario ({_scenario_label})")

    # Dynamic options (both the 2nd option's wording and the 3rd option's
    # label can change run to run) — a stale session_state value that no
    # longer matches any current option would otherwise raise a Streamlit
    # exception on the widget below, so reset it defensively rather than
    # let a changed label crash the whole page.
    if st.session_state.get("nav_mode") not in _nav_options:
        st.session_state["nav_mode"] = "Current squad"

    nav_c1, nav_c2 = st.columns([2, 2])
    with nav_c1:
        nav_mode = st.radio("Squad", _nav_options,
                             index=0, horizontal=True, key="nav_mode",
                             disabled=len(_nav_options) == 1,
                             help=None if len(_nav_options) > 1 else
                             "No recommended transfer this run to preview for the current planning GW, "
                             "and no scenario evaluated below yet.")
    with nav_c2:
        pb1, pb2, pb3 = st.columns([1, 3, 1])
        with pb1:
            if st.button("◀", key="nav_prev", disabled=st.session_state.nav_gw_idx == 0):
                st.session_state.nav_gw_idx -= 1
        with pb2:
            st.markdown(f'<div style="text-align:center;font-weight:600;padding-top:0.4rem;">'
                        f'GW{_nav_gw_list[st.session_state.nav_gw_idx]} '
                        f'({st.session_state.nav_gw_idx + 1}/{len(_nav_gw_list)})</div>', unsafe_allow_html=True)
        with pb3:
            if st.button("▶", key="nav_next", disabled=st.session_state.nav_gw_idx == len(_nav_gw_list) - 1):
                st.session_state.nav_gw_idx += 1

    nav_gw = _nav_gw_list[st.session_state.nav_gw_idx]
    nav_col = f"xpts_gw{nav_gw}"
    if nav_mode.startswith("After evaluated scenario") and _nav_squad_scenario is not None:
        nav_squad = _nav_squad_scenario
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
        st.caption(f"Couldn't solve a valid starting XI for GW{nav_gw} (common for a genuine blank gameweek).")
        return

    nav_starters = nav_xi_result["xi"]
    nav_bench = nav_squad[~nav_squad["code"].isin(nav_starters["code"])]
    nav_gw_xpts = round(nav_xi_result["total"], 1)

    # Header tiles — same mechanic as the main Team Rating % (Patch 20/24):
    # current squad's best XI (captain doubled, bench autosub-discounted)
    # over a genuinely unconstrained optimal squad for THIS GW specifically.
    # Cached (see _nav_optimal_squad above) so revisiting a GW is instant.
    nav_optimal_result = _nav_optimal_squad(cfg, chip_adv_proj, team_value, nav_gw)
    nav_current_val = opt.rating_gw_value(nav_squad, nav_col, cfg)["total_realized"]
    nav_optimal_val = opt.rating_gw_value(nav_optimal_result["squad"], nav_col, cfg)["total_realized"] \
        if nav_optimal_result else 0.0
    nav_rating = eng.team_rating_pct(nav_current_val, nav_optimal_val, "")

    nt1, nt2 = st.columns(2)
    with nt1:
        st.metric(f"GW{nav_gw} xPts (best XI)", f"{nav_gw_xpts:.1f}")
    with nt2:
        st.metric(f"Team Rating % (GW{nav_gw})",
                  f"{nav_rating['rating_pct']}%" if nav_rating["rating_pct"] is not None else "—")
    if not at_planning_gw:
        st.caption("Projected for this GW only — Overall rank and Season points elsewhere on this page are "
                   "your live actuals and don't change with navigation.")

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
        st.markdown('<p class="side-note">Fixture ticker: one dot per GW in your horizon — '
                    'easy/mid/hard, hover for the opponent. Full opponent + xPts breakdown per GW is in the '
                    'table below.</p>', unsafe_allow_html=True)
    st.markdown('<p class="side-note">SP tag = newly confirmed set-piece role, decaying out as current-season '
                'minutes accrue.</p>', unsafe_allow_html=True)
    if at_planning_gw and cap_caption:
        # Patch 4 — captaincy-on-pitch caption, only meaningful at
        # planning_gw where the real captaincy protocol (not the simple
        # top-scorer armband) actually ran.
        st.markdown(f'<div class="cap-caption">{cap_caption}</div>', unsafe_allow_html=True)


_render_pitch_navigator()

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

# Post-Patch-34 follow-up — free worst-case comparison, shown before any
# transfer recommendation: if a flagged squad player is currently starting,
# this is what your best XI looks like if he truly scores zero, using only
# players you already own. Deliberately captioned as a downside-risk check,
# not "free upgrade" — the model's own projection for him already reflects
# a probability-weighted expectation (see the function's docstring); this is
# for when the manager's own read is harsher than that.
if free_fix.get("flagged_starting"):
    st.markdown(f'<div class="tx-preview">⚠️ Worst case if <b>{free_fix["player"]}</b> scores 0 this GW '
                f'(currently started; his own projection already reflects a live chance-of-playing discount, '
                f'this is the harsher case): best XI with <b>{free_fix["worst_case_replacement"] or "—"}</b> '
                f'instead — <b>{free_fix["worst_case_total"]:.1f}</b> xPts (vs {free_fix["current_total"]:.1f} '
                f'if he plays at his current projection). No transfer needed for this — compare against any '
                f'transfer recommended below.</div>', unsafe_allow_html=True)

if transfer_error:
    st.error(f"Couldn't compute transfer suggestions this run ({transfer_error}). Everything else on this page "
             f"is unaffected — try Run Model again, and if it repeats, this is worth reporting with that message.")

# Patch 5 — simplified primary display: just the recommendation, in plain
# language, front and center. All the rule-citation trace and the raw move
# table that used to be the primary content now live behind one expander,
# available on demand rather than shown by default.
if rec.get("is_weekly_schedule"):
    _hs_txt = rec.get("hit_stance", "No hits")
    st.caption(f"{_hs_txt} + {horizon}-GW horizon → this is a chained, week-by-week pacing plan (each week's move "
               f"assumes every earlier week's suggested move already happened), not a single this-week decision. "
               f"Free-transfer accrual (+1/week, cap 5) is modeled explicitly below."
               + (" Hits are allowed where a paid move still clears the stricter hit-cost bar." if _hs_txt == "Hit if worth it" else ""))
if rec.get("summary"):
    for line in rec["summary"]:
        st.markdown(f'<div class="tx-reco">{line}</div>', unsafe_allow_html=True)
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
        st.markdown(f'<div class="tx-preview">📈 If you make this move — new GW{planning_gw} xPts: '
                    f'<b>{_new_gw_xpts:.1f}</b> (was {gw_xpts_total:.1f}) · new Team Rating: '
                    f'<b>{_new_rating["rating_pct"]}%</b> (was {fh_auto_rating["rating_pct"]}%, vs. the same '
                    f'GW{planning_gw} Free Hit optimal shown at the top). Local preview only — the header stats '
                    f'above are unaffected until you actually make the transfer and re-run.</div>',
                    unsafe_allow_html=True)

st.caption(f"Two separate bars gate a transfer: a **{rec['minimum_meaningful_gain_free']} xPts** materiality bar "
           f"(is the gain worth spending a free transfer at all) and a **{rec.get('margin_of_error', 2.0):.1f} xPts** "
           f"margin-of-error floor (is the gain distinguishable from this model's own known projection noise — "
           f"Standing Rule #34, not adjustable via the sidebar slider). A move must clear BOTH to be recommended. "
           f"xM badges above show each player's expected-minutes multiplier — already priced into their xPts, "
           f"surfaced here so a rotation risk doesn't hide behind a good net number.")

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
                planning_gw, gw_list, target_choice[0], default_net_gain=rec.get("net_gain"),
                disrupted_codes=_disrupted_codes, bb_play_gw=_bb_play_gw)
            st.markdown("**Target player scenario**")
            if target_eval["summary"]:
                for line in target_eval["summary"]:
                    st.markdown(f'<div class="tx-reco">🧪 {line}</div>', unsafe_allow_html=True)
            if target_eval["moves"]:
                st.dataframe(pd.DataFrame(target_eval["moves"])[
                    [c for c in ["out", "in", "position", "xpts_gain", "hit_cost", "net_gain", "justified"]
                     if c in pd.DataFrame(target_eval["moves"]).columns]], hide_index=True, use_container_width=True)
                # Patch 40 (manager, 2026-09-14: "the navigator can have a 3rd
                # option to read from the scenarios on the section for
                # 'evaluate the scenario'"). Stash this evaluated scenario's
                # moves in session_state so the pitch navigator above (it
                # renders earlier in the script, but this is a full rerun —
                # not confined to the navigator's own st.fragment — so the
                # next run picks this up) can offer a 3rd "After evaluated
                # scenario" toggle alongside "Current squad" / "After
                # recommended transfer", built the same out_code/in_code
                # reconstruction way as that existing toggle.
                st.session_state["scenario_nav_moves"] = target_eval["moves"]
                st.session_state["scenario_nav_label"] = target_choice[1]
            elif "scenario_nav_moves" in st.session_state:
                # Feasible search ran but found no moves to preview (e.g. the
                # already-owned / no-legal-way branches) — don't leave a
                # stale, unrelated scenario sitting in the navigator toggle.
                del st.session_state["scenario_nav_moves"]
                st.session_state.pop("scenario_nav_label", None)
            if target_eval.get("plan"):
                with st.expander("Why — full trace, rule references, and move-by-move detail"):
                    for line in target_eval["plan"]:
                        st.markdown(f"- {line}")
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
                            f'Informational only — this candidate GW is your own choice, and the model never '
                            f'names a single "play" date (Standing Rule #32); see Chip Rack above for whether '
                            f'v6.4\'s own Wildcard trigger is currently active.</div>', unsafe_allow_html=True)

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
            # date always the manager's own choice per Rule #32), a Free Hit
            # squad reverts after one week
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
