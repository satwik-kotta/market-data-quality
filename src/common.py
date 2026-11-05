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
SEC_USER_AGENT = (os.environ.get("SEC_USER_AGENT")
                  or "market-data-quality research project github.com/satwik-kotta/market-data-quality")


@lru_cache(maxsize=1)
def sessions(start: date = PRICE_START, end: date | None = None) -> pl.Series:
    """NYSE trading days from start to end (inclusive)."""
    import exchange_calendars as xc

    cal = xc.get_calendar("XNYS")
    end = end or min(date.today(), cal.last_session.date())
    days = cal.sessions_in_range(start.isoformat(), end.isoformat())
    return pl.Series("date", [d.date() for d in days], dtype=pl.Date)


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def stooq_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").lower() + ".us"
