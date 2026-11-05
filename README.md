# Market Data Quality Monitor

Two free vendors, the same 500 stocks, three years of daily prices. How often do they disagree, why, and what does a clean table look like? Then a test of whether the text of 8-K filings carries information about how a stock moves.

**Status:** first run in progress. Findings will be added here.

## What it does

1. **Collects** daily prices for the S&P 500 from Yahoo Finance and Stooq, and every 8-K filed by those companies since January 2024 from SEC EDGAR (`src/fetch.py`, run weekly on GitHub Actions).
2. **Normalizes** both vendors into one schema in polars: types, ticker symbols, duplicates and missing closes (`src/quality.py`).
3. **Validates** each vendor: non positive prices, open, high, low and close that contradict each other, zero volume, stale prices, moves over 25% that are not splits, rows on days the NYSE was closed, and missing trading days.
4. **Compares** the vendors: works out per ticker whether Stooq's close is adjusted for dividends or only for splits, then flags every day the two disagree by more than 0.5%.
5. **Builds a clean table** where each row is confirmed by both vendors, from one vendor only, in conflict, or rejected. Returns are only computed between two usable rows.
6. **Event study** (`src/events.py`): abnormal returns around each 8-K, by item type (earnings, executive changes, agreements and others), and a test of whether Loughran-McDonald tone of the press release predicts the move. Tone cut offs are set on the earlier period and tested on the later one, with the 5 days before each filing as a placebo.

## Run it

```bash
pip install -r requirements.txt
python src/fetch.py prices filings   # needs internet access to Yahoo, Stooq and SEC EDGAR
python src/run_all.py
python -m pytest -q
```

Raw downloads are kept on the `data` branch exactly as received, so every cleaning step can be checked.
