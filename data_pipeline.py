"""
data_pipeline.py
Shared player-table build + xPts computation, used by app.py. Ported from
the CLI tool's run_weekly_report.py so the Streamlit app and the CLI stay
on the exact same pipeline — this is the one place that logic lives.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import fpl_data
import fpl_engine as eng
import optimizer as opt
import setpiece


def build_player_table(cfg: dict, snap: fpl_data.FplSnapshot, hist_df: pd.DataFrame,
                        overrides: pd.DataFrame) -> pd.DataFrame:
    df = snap.players.copy()

    if "element_type" in df.columns:
        df["position"] = df["element_type"].map(eng.POSITION_MAP)
    else:
        df["position"] = "MID"

    numcols = ["expected_goals", "expected_assists", "expected_goals_per_90",
               "expected_assists_per_90", "defensive_contribution",
               "defensive_contribution_per_90", "minutes", "starts",
               "starts_per_90", "bonus", "now_cost", "selected_by_percent",
               "chance_of_playing_next_round", "total_points",
               "penalties_order", "corners_and_indirect_freekicks_order",
               "direct_freekicks_order"]
    for c in numcols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    games90 = (df.get("minutes", 0) / 90.0).replace(0, np.nan)
    df["npxg90_cur"] = df.get("expected_goals", np.nan) / games90
    df["xa90_cur"] = df.get("expected_assists", np.nan) / games90
    df["dc90_cur"] = df.get("defensive_contribution", np.nan) / games90
    df["bonus_per_start_cur"] = df.get("bonus", 0) / df.get("starts", np.nan).replace(0, np.nan)

    if hist_df is not None and not hist_df.empty and "code" in hist_df.columns:
        h = hist_df[["code", "expected_goals_per_90", "expected_assists_per_90",
                      "defensive_contribution_per_90", "bonus", "starts", "minutes"]].copy()
        for c in ["expected_goals_per_90", "expected_assists_per_90",
                  "defensive_contribution_per_90", "bonus", "starts", "minutes"]:
            h[c] = pd.to_numeric(h[c], errors="coerce")
        h["bonus_per_start_hist"] = h["bonus"] / h["starts"].replace(0, np.nan)
        h = h.rename(columns={
            "expected_goals_per_90": "npxg90_hist",
            "expected_assists_per_90": "xa90_hist",
            "defensive_contribution_per_90": "dc90_hist",
            "minutes": "minutes_hist",
        })[["code", "npxg90_hist", "xa90_hist", "dc90_hist", "bonus_per_start_hist", "minutes_hist"]]
        df = df.merge(h, on="code", how="left")
    else:
        df["npxg90_hist"] = np.nan
        df["xa90_hist"] = np.nan
        df["dc90_hist"] = np.nan
        df["bonus_per_start_hist"] = np.nan
        df["minutes_hist"] = np.nan

    # Standing Rule #21 / Step 2b — tiny-sample gate (the "Dowman bug" fix).
    min_min = cfg["data_quality"]["min_sample_minutes"]
    df["hist_sample_ok"] = df["minutes_hist"].fillna(0) >= min_min
    df["cur_sample_ok"] = df.get("minutes", pd.Series(0, index=df.index)).fillna(0) >= min_min
    for col in ["npxg90_hist", "xa90_hist", "dc90_hist"]:
        df.loc[~df["hist_sample_ok"], col] = np.nan
    for col in ["npxg90_cur", "xa90_cur", "dc90_cur"]:
        df.loc[~df["cur_sample_ok"], col] = np.nan
    df["est_rescue_needed"] = ~df["hist_sample_ok"] & ~df["cur_sample_ok"]

    df = df.merge(overrides, left_on="code", right_on="player_code", how="left")

    for src, dst in [("npxg90_rescue", "npxg90_hist"), ("xa90_rescue", "xa90_hist"),
                      ("dc90_rescue", "dc90_hist")]:
        if src in df.columns:
            df[src] = pd.to_numeric(df[src], errors="coerce")
            mask = df[src].notna()
            if mask.any():
                df.loc[mask, dst] = df.loc[mask, src]
                df.loc[mask, "est_rescue_needed"] = False

    return df


def price(now_cost) -> float | None:
    return round(now_cost / 10.0, 1) if pd.notna(now_cost) else None


def team_short_name(teams: pd.DataFrame, team_id) -> str:
    row = teams.loc[teams["id"] == team_id]
    return row["short_name"].iloc[0] if not row.empty and "short_name" in row.columns else "?"


def comp_discount_for_team(cfg: dict, short_name: str) -> float:
    euro = cfg["european_competition_teams"]
    disc = cfg["competition_load_discount"]
    if short_name in euro.get("champions_league", []):
        return disc["champions_league"]
    if short_name in euro.get("europa_league", []):
        return disc["europa_league"]
    if short_name in euro.get("conference_league", []):
        return disc["conference_league"]
    return disc["domestic_only"]


def get_fixture_for_gw(fixtures: pd.DataFrame, team_id: int, gw: int):
    if fixtures.empty or "event" not in fixtures.columns:
        return []
    rows = fixtures[fixtures["event"] == gw]
    out = []
    for _, r in rows.iterrows():
        if r.get("team_h") == team_id:
            out.append((int(r["team_a"]), True))
        elif r.get("team_a") == team_id:
            out.append((int(r["team_h"]), False))
    return out


def compute_all(cfg: dict, snap: fpl_data.FplSnapshot, players: pd.DataFrame,
                 gw_list: list[int]) -> pd.DataFrame:
    """Same Core Formula pipeline as the CLI tool, plus Step 3c's set-piece
    multiplier applied to npxG_blend right after Step 3's decay blend."""
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
        sp_mult_last = 1.0
        for gw in gw_list:
            fixtures = get_fixture_for_gw(snap.fixtures, team_id, gw)
            if not fixtures:
                gw_xpts[gw] = 0.0
                gw_opp[gw] = ""  # blank gameweek — no fixture, surfaced as-is in the UI
                continue
            gw_total = 0.0
            opp_labels = []
            for opp_id, is_home in fixtures:
                opp_row = teams.loc[teams["id"] == opp_id]
                opp_row = opp_row.iloc[0] if not opp_row.empty else pd.Series(dtype=float)
                opp_labels.append(f"{team_short_name(teams, opp_id)} ({'H' if is_home else 'A'})")

                cs_override = p.get("cs_pct_override", np.nan)
                if pd.notna(cs_override):
                    cs_pct = float(cs_override)
                else:
                    cs_pct = eng.cs_pct_poisson(trow, opp_row, is_home)

                npxg = eng.blend_rate(p.get("npxg90_hist"), p.get("npxg90_cur"), gw, cfg,
                                       current_sample_matches=int(p.get("starts", 0) or 0))
                npxg, sp_mult_last = setpiece.apply_to_npxg(npxg, p, gw, cfg)
                xa = eng.blend_rate(p.get("xa90_hist"), p.get("xa90_cur"), gw, cfg,
                                     current_sample_matches=int(p.get("starts", 0) or 0))
                dc90 = eng.blend_rate(p.get("dc90_hist"), p.get("dc90_cur"), gw, cfg,
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
            gw_opp[gw] = " / ".join(opp_labels)  # "/"-joined for a double gameweek, single label otherwise

        rec = {
            "code": p.get("code"), "id": p.get("id"), "web_name": p.get("web_name"),
            "team": short, "team_id": team_id, "position": pos, "price": price(p.get("now_cost")),
            "status": p.get("status"), "news": p.get("news"),
            "selected_by_percent": p.get("selected_by_percent"),
            "xm": round(xm, 3),
            "est_rescue_needed": bool(p.get("est_rescue_needed", False)),
            "setpiece_flag": sp_mult_last > 1.001,
            "setpiece_multiplier": round(sp_mult_last, 3),
        }
        for gw in gw_list:
            rec[f"xpts_gw{gw}"] = gw_xpts[gw]
            rec[f"opp_gw{gw}"] = gw_opp[gw]
        rec["xpts_horizon_sum"] = round(sum(gw_xpts.values()), 3)
        rows.append(rec)

    return pd.DataFrame(rows)


def solve_ceiling(cfg: dict, proj: pd.DataFrame):
    """§1a's unconstrained Ceiling_xPts — the best possible £100m/2-5-5-3/
    max-3-per-club squad from the FULL pool, ignoring what you currently own
    or how many transfers you have. Kept as the "theoretical" reference
    number (Standing Rules #16/#18 disclosure) alongside the reachable
    ceiling below, which is what the Team Rating % headline now uses."""
    return opt.solve_squad(proj, cfg, budget=cfg["squad_rules"]["budget"])


def solve_reachable_ceiling(cfg: dict, proj: pd.DataFrame, current_squad_codes: list,
                             free_transfers: int):
    """The constrained counterpart to solve_ceiling(): the best squad
    actually reachable FROM the current squad using only the free transfers
    available right now (min_retain = 15 - free transfers, so up to that
    many transfers can freely change; the rest of the 15 must be kept).
    This is what makes Team Rating % answer "how close am I to the best
    *reachable* squad this week", not "how close am I to a fantasy ideal
    that assumes I own nobody and have unlimited transfers" — the latter is
    structurally near-impossible to score well on regardless of squad
    quality, which is why it read as permanently low no matter how much
    manual_overrides.csv research went into the pool.
    Same £100m budget proxy as solve_ceiling — real budget is bank + each
    player's individual sell price, which isn't tracked per-player here, so
    current price stands in for it (disclosed simplification, same standard
    as the Bench Value Rule's autosub discount in model_config.yaml)."""
    min_retain = max(0, min(15, 15 - max(0, free_transfers)))
    return opt.solve_squad(proj, cfg, budget=cfg["squad_rules"]["budget"],
                            retain_pool_codes=current_squad_codes, min_retain=min_retain)


def solve_free_hit_rebuild(cfg: dict, proj: pd.DataFrame, total_value: float, gw: int):
    """Step 8b Free Hit Evaluation (Standing Rule #25): a full 15-man
    rebuild against the manager's TOTAL team value (bank + current squad's
    sell-value proxy — see solve_reachable_ceiling's note on that proxy),
    optimized for the single target gameweek only, never the multi-GW
    horizon sum, since a Free Hit's squad reverts after that one week
    (Horizon-Matching Rule). Never a marginal swap-budget check — always the
    full rebuild, per Rule #25."""
    col = f"xpts_gw{gw}"
    if col not in proj.columns:
        return None
    return opt.solve_squad(proj, cfg, budget=total_value, objective_col=col)
