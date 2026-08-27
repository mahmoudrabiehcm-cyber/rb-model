"""
transfers.py
Step 7a addendum — free-transfer count, derived automatically from the
manager's transfer history rather than asked (v4.0 doc). Mirrors the
official 2026/27 rule: +1 free transfer per gameweek from GW2 onward,
banked up to a max of 5; a played Wildcard resets the bank to 1 the
following gameweek; a played Free Hit doesn't touch the bank at all
(its transfers are free and don't draw down or reset anything).
"""
from __future__ import annotations

MAX_BANK = 5


def derive_free_transfers(history_current: list[dict], chips_played: list[dict]) -> dict:
    """history_current: entry history's ['current'] list (per finished GW —
    event, event_transfers, ...). chips_played: entry history's ['chips']
    list (name, event). Returns {"free_transfers": int, "trace": [str,...]}
    — the trace is shown in the UI so the derivation is never a black box."""
    chip_by_event = {c.get("event"): c.get("name") for c in (chips_played or [])}
    ft = 1
    trace = ["GW2 baseline: 1 free transfer (GW1 pre-season moves are unlimited and don't count)."]

    rows = sorted([r for r in (history_current or []) if r.get("event", 1) >= 2],
                  key=lambda r: r["event"])

    for r in rows:
        gw = r["event"]
        made = int(r.get("event_transfers", 0) or 0)
        chip = chip_by_event.get(gw)

        if chip in ("wildcard", "freehit"):
            trace.append(f"GW{gw}: {chip} played — transfers that week are free, bank untouched.")
        else:
            spent = min(ft, made)
            ft = max(0, ft - made)
            if made > spent:
                trace.append(f"GW{gw}: made {made} transfers on a bank of {spent + ft}+{ft} "
                              f"-> {made - spent} paid.")
            elif made:
                trace.append(f"GW{gw}: used {made} of the banked free transfer(s).")

        if chip == "wildcard":
            ft = 1
            trace.append(f"GW{gw}: Wildcard resets the bank to 1 for GW{gw + 1}.")
        else:
            ft = min(MAX_BANK, ft + 1)

    trace.append(f"-> {ft} free transfer(s) available now (cap {MAX_BANK}).")
    return {"free_transfers": ft, "trace": trace}
