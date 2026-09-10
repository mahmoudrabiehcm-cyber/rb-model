"""
transfers.py
Step 7a addendum — free-transfer count, derived automatically from the
manager's transfer history rather than asked (v4.0 doc). Mirrors the
official 2026/27 rule: +1 free transfer per gameweek from GW2 onward,
banked up to a max of 5; a played Wildcard draws from its own separate
unlimited-transfer pool that week and does NOT touch the banked count at
all -- the bank carries forward untouched and still gets its normal +1
accrual, same as any other gameweek; a played Free Hit likewise doesn't
touch the bank (its transfers are free and don't draw down or reset
anything).

Patch 26 (v6.2 / Standing Rule #39, 2026-09-10 discussion) -- this
previously reset the bank to 1 on a Wildcard gameweek, on the mistaken
premise that the Wildcard's unlimited moves made the existing bank
irrelevant. Corrected: a banked transfer survives a Wildcard untouched
and remains available immediately after it as a genuine, deliberate tool
(fixing a Wildcard-build miss, chasing a fixture swing the Wildcard
didn't reach, or setting up a subsequent chip) -- it is never zeroed out
just because a Wildcard was played.
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

        # Patch 26 (v6.2 / Standing Rule #39) — a Wildcard draws from its own
        # unlimited pool, not from the bank, so the bank is NEVER reset here.
        # Both branches now get the same normal +1 accrual regardless of
        # whether a Wildcard was played this GW.
        if chip == "wildcard":
            trace.append(f"GW{gw}: Wildcard played — bank carries forward untouched (Rule #39), "
                          f"then gets its normal +1 accrual for GW{gw + 1} like any other week.")
        ft = min(MAX_BANK, ft + 1)

    trace.append(f"-> {ft} free transfer(s) available now (cap {MAX_BANK}).")
    return {"free_transfers": ft, "trace": trace}
