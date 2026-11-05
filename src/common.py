"""Shared paths, dates and the NYSE trading calendar."""
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

PRICE_START = date(2023, 1, 1)
FILING_START = date(2024, 1, 1)

# SEC asks automated clients to identify themselves. Set SEC_USER_AGENT to "Name email" to follow that fully.
# Collapse whitespace so a pasted secret with a trailing newline is still a valid header.
SEC_USER_AGENT = (" ".join((os.environ.get("SEC_USER_AGENT") or "").split())
                  or "market-data-quality research project github.com/satwik-kotta/market-data-quality")


@lru_cache(maxsize=1)
def sessions(start: date = PRICE_START, end: date | None = None) -> pl.Series:
    """NYSE trading days from start to end (inclusive)."""
    import exchange_calendars as xc

    cal = xc.get_calendar("XNYS")
    if end is None:
        # Only sessions that have closed, so a run during market hours never reads a half-finished bar.
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        end = now.date() if now.hour >= 17 else now.date() - timedelta(days=1)
        end = min(end, cal.last_session.date())
    days = cal.sessions_in_range(start.isoformat(), end.isoformat())
    return pl.Series("date", [d.date() for d in days], dtype=pl.Date)


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")
