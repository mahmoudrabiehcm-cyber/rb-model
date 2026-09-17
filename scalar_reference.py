"""Stashed copy of the ORIGINAL scalar compute_all() (pre-Patch-50), kept
only as ground truth for test_patch50_vectorized_correctness.py's diff.
Not imported by the app."""
from __future__ import annotations
import numpy as np
import pandas as pd

import fpl_engine as eng
import setpiece
from data_pipeline import team_short_name, comp_discount_for_team, get_fixture_for_gw, \
    _fdr_tier_from_official, _fdr_tier_from_strength, price


def compute_all_scalar_reference(cfg: dict, snap, players: pd.DataFrame,
                                  gw_list: list[int]) -> pd.DataFrame:
    teams = snap.teams.copy()
    if "id" not in teams.columns:
        teams["id"] = teams.index

    rows = []
    for _, p in players.iterrows():
        pos = p.get("position", "MID")
        if pos not in ("GK", "DEF", "MID", "FWD"):
            continue
        team_id = p.get("team")
        trow = teams.loc[teams["id"] == team_id]
        trow = trow.iloc[0] if not trow.empty else pd.Series(dtype=float)
        short = team_short_name(teams, team_id)
        comp_disc = comp_discount_for_team(cfg, short)

        override_row = p if pd.notna(p.get("player_code", np.nan)) else None
        xm_override = p.get("xm_override", np.nan)
        xm = eng.estimate_xm(p, cfg, xm_override if pd.notna(xm_override) else None)

        gw_xpts = {}
        gw_opp = {}
        gw_fdr = {}
        fdr_rank = {"easy": 0, "mid": 1, "hard": 2}
        for gw in gw_list:
            fixtures = get_fixture_for_gw(snap.fixtures, team_id, gw)
            if not fixtures:
                gw_xpts[gw] = 0.0
                gw_opp[gw] = ""
                gw_fdr[gw] = ""
                continue
            gw_total = 0.0
            opp_labels = []
            fdr_tiers = []
            for opp_id, is_home, official_diff in fixtures:
                opp_row = teams.loc[teams["id"] == opp_id]
                opp_row = opp_row.iloc[0] if not opp_row.empty else pd.Series(dtype=float)
                opp_labels.append(f"{team_short_name(teams, opp_id)} ({'H' if is_home else 'A'})")
                tier = _fdr_tier_from_official(official_diff)
                if tier is None:
                    tier = _fdr_tier_from_strength(opp_row, is_home)
                fdr_tiers.append(tier)

                cs_override = p.get("cs_pct_override", np.nan)
                if pd.notna(cs_override):
                    cs_pct = float(cs_override)
                else:
                    cs_pct = eng.cs_pct_poisson(trow, opp_row, is_home)

                npxg = eng.blend_rate(p.get("npxg90_hist"), p.get("npxg90_cur"), gw, cfg,
                                       metric="npxg",
                                       current_sample_matches=int(p.get("starts", 0) or 0))
                npxg, _sp_mult_this_gw = setpiece.apply_to_npxg(npxg, p, gw, cfg)
                xa = eng.blend_rate(p.get("xa90_hist"), p.get("xa90_cur"), gw, cfg,
                                     metric="xa",
                                     current_sample_matches=int(p.get("starts", 0) or 0))
                dc90 = eng.blend_rate(p.get("dc90_hist"), p.get("dc90_cur"), gw, cfg,
                                       metric="dc",
                                       current_sample_matches=int(p.get("starts", 0) or 0))
                p_for_bonus = p.copy()
                p_for_bonus["bonus_per_start_hist"] = (
                    p.get("bonus_per_start_cur") if pd.notna(p.get("bonus_per_start_cur"))
                    else p.get("bonus_per_start_hist")
                )
                res = eng.compute_player_gw_xpts(p_for_bonus, pos, cs_pct, xm, npxg, xa,
                                                  dc90, comp_disc, cfg, override_row)
                gw_total += res["xpts"]
            gw_xpts[gw] = round(gw_total, 3)
            gw_opp[gw] = " / ".join(opp_labels)
            gw_fdr[gw] = max(fdr_tiers, key=lambda t: fdr_rank.get(t, 1)) if fdr_tiers else ""

        sp_mult_ref = setpiece.setpiece_multiplier(p, gw_list[0], cfg) if gw_list else 1.0

        rec = {
            "code": p.get("code"), "id": p.get("id"), "web_name": p.get("web_name"),
            "team": short, "team_id": team_id, "position": pos, "price": price(p.get("now_cost")),
            "status": p.get("status"), "news": p.get("news"),
            "selected_by_percent": p.get("selected_by_percent"),
            "transfers_in_event": p.get("transfers_in_event"),
            "transfers_out_event": p.get("transfers_out_event"),
            "net_transfers_event": (
                (p.get("transfers_in_event") if pd.notna(p.get("transfers_in_event")) else 0)
                - (p.get("transfers_out_event") if pd.notna(p.get("transfers_out_event")) else 0)
            ) if pd.notna(p.get("transfers_in_event")) or pd.notna(p.get("transfers_out_event")) else None,
            "xm": round(xm, 3),
            "est_rescue_needed": bool(p.get("est_rescue_needed", False)),
            "setpiece_flag": sp_mult_ref > 1.001,
            "setpiece_multiplier": round(sp_mult_ref, 3),
        }
        for gw in gw_list:
            rec[f"xpts_gw{gw}"] = gw_xpts[gw]
            rec[f"opp_gw{gw}"] = gw_opp[gw]
            rec[f"fdr_gw{gw}"] = gw_fdr[gw]
        rec["xpts_horizon_sum"] = round(sum(gw_xpts.values()), 3)
        rows.append(rec)

    return pd.DataFrame(rows)
