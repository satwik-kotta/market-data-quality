"""Download the raw inputs. Runs on GitHub Actions, where these hosts are reachable.

    universe   S&P 500 members with CIK numbers (datasets/s-and-p-500-companies on GitHub)
    yahoo      daily OHLCV, split adjusted close, adjusted close, dividends and splits (yfinance)
    nasdaq     daily OHLCV from Nasdaq's public quote API
    filings    every 8-K filed by those companies since FILING_START, from SEC EDGAR, with
               Loughran-McDonald word counts taken from the filing text (the text itself is not stored)

Each source is written to data/raw as parquet. Vendor data is stored exactly as received, so all
cleaning happens later and can be checked.
"""
import io
import json
import re
import sys
import time
import urllib.request
from datetime import date, timedelta
from html import unescape

import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import FILING_START, PRICE_START, RAW, SEC_USER_AGENT, yahoo_symbol  # noqa: E402

UNIVERSE_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def http_get(url: str, headers: dict | None = None, retries: int = 4, timeout: int = 30) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "market-data-quality"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1} {url[:80]}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt * 2)
    return b""


def universe() -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(http_get(UNIVERSE_URL)))
    df = df.select(
        pl.col("Symbol").alias("ticker"), pl.col("Security").alias("name"),
        pl.col("GICS Sector").alias("sector"), pl.col("CIK").cast(pl.Int64).alias("cik"),
    )
    df.write_parquet(RAW / "universe.parquet")
    return df


def yahoo(tickers: list[str]) -> pl.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        # threads=False avoids yfinance's shared cache database locking up under parallel downloads
        wide = yf.download([yahoo_symbol(t) for t in batch], start=PRICE_START.isoformat(),
                           end=(date.today() + timedelta(days=1)).isoformat(), auto_adjust=False, actions=True,
                           group_by="ticker", threads=False, progress=False)
        for t in batch:
            sym = yahoo_symbol(t)
            if sym not in wide.columns.get_level_values(0):
                continue
            g = wide[sym].reset_index()
            g.columns = [str(c).lower().replace(" ", "_") for c in g.columns]
            g = g.dropna(subset=["close"]) if "close" in g.columns else g
            if len(g):
                frames.append(pl.from_pandas(g).with_columns(pl.lit(t).alias("ticker")))
        time.sleep(1)
    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.rename({"stock_splits": "splits"}) if "stock_splits" in df.columns else df
    df.write_parquet(RAW / "yahoo.parquet")
    return df


def nasdaq(tickers: list[str]) -> pl.DataFrame:
    """Daily history from Nasdaq's public quote API. Prices arrive as text like "$1,234.56"."""
    import json

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
               "Accept": "application/json, text/plain, */*", "Origin": "https://www.nasdaq.com",
               "Referer": "https://www.nasdaq.com/"}
    frames, failed = [], []
    for t in tickers:
        url = (f"https://api.nasdaq.com/api/quote/{t.replace('.', '%5E')}/historical?assetclass=stocks"
               f"&fromdate={PRICE_START.isoformat()}&todate={date.today().isoformat()}&limit=9999")
        try:
            body = json.loads(http_get(url, headers, retries=3, timeout=40))
            rows = ((body.get("data") or {}).get("tradesTable") or {}).get("rows") or []
        except Exception as e:
            if not failed:
                print(f"nasdaq sample error for {t}: {e}")
            failed.append(t)
            continue
        if not rows:
            if not failed:
                print(f"nasdaq sample response for {t}: {str(body)[:300]}")
            failed.append(t)
            continue
        frames.append(pl.DataFrame(rows).with_columns(pl.lit(t).alias("ticker")))
        time.sleep(0.5)
    print(f"nasdaq: {len(frames)} tickers, {len(failed)} failed: {failed[:20]}")
    if not frames:
        raise SystemExit("Nasdaq returned no data")
    df = pl.concat(frames, how="diagonal_relaxed")
    df.write_parquet(RAW / "nasdaq.parquet")
    return df


TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
# Press release exhibits are named many ways: ex99.htm, ex-99_1.htm, exhibit991.htm, a991pressrelease.htm ...
EX99 = re.compile(r"(ex|exh|exhibit)[-_ ]?99|(^|[^0-9])99[-_.]?1[^0-9]|press[-_ ]?release|earnings[-_ ]?release", re.I)


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return SPACE.sub(" ", unescape(TAG.sub(" ", html))).strip()


class RateLimit:
    """Shared limit across threads: at most `per_second` requests start each second."""

    def __init__(self, per_second: float):
        import threading

        self.gap = 1.0 / per_second
        self.lock = threading.Lock()
        self.next = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.gap
        time.sleep(max(0.0, start - now))


def _one_filing(meta: dict, headers: dict, limit: RateLimit, lm) -> dict:
    row = meta | {"text_source": None, "n_words": None, "lm_positive": None, "lm_negative": None}
    folder = f"https://www.sec.gov/Archives/edgar/data/{meta['cik']}/{meta['accession'].replace('-', '')}"
    try:
        # The news is usually in the attached press release (exhibit 99), not the 8-K cover form.
        limit.wait()
        listing = json.loads(http_get(f"{folder}/index.json", headers, retries=3))
        names = [f["name"] for f in listing["directory"]["item"]]
        ex99 = [n for n in names if n != meta["document"] and EX99.search(n)
                and n.lower().endswith((".htm", ".html", ".txt"))]
        doc = ex99[0] if ex99 else meta["document"]
        limit.wait()
        text = html_to_text(http_get(f"{folder}/{doc}", headers, retries=3).decode("utf-8", "ignore"))
        tokens = lm.tokenize(text)
        score = lm.get_score(tokens)
        row.update(text_source="exhibit_99" if ex99 else "form", n_words=len(tokens),
                   lm_positive=int(score["Positive"]), lm_negative=int(score["Negative"]))
    except Exception as e:
        print(f"document failed {meta['ticker']} {meta['accession']}: {e}", file=sys.stderr)
    return row


def filings(uni: pl.DataFrame, max_per_company: int = 60, workers: int = 8) -> pl.DataFrame:
    from concurrent.futures import ThreadPoolExecutor

    import pysentiment2 as ps

    lm = ps.LM()
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "identity"}
    limit = RateLimit(9)  # SEC allows 10 requests per second
    metas = []
    for ticker, cik in uni.select("ticker", "cik").iter_rows():
        limit.wait()
        try:
            r = json.loads(http_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers))["filings"]["recent"]
        except Exception as e:
            print(f"submissions failed {ticker}: {e}", file=sys.stderr)
            continue
        taken = 0
        for i, form in enumerate(r["form"]):
            if form == "8-K" and r["filingDate"][i] >= FILING_START.isoformat() and taken < max_per_company:
                metas.append({"ticker": ticker, "cik": cik, "accession": r["accessionNumber"][i],
                              "filing_date": r["filingDate"][i], "accepted": r["acceptanceDateTime"][i],
                              "items": r["items"][i], "document": r["primaryDocument"][i]})
                taken += 1
    print(f"filings: {len(metas)} 8-Ks listed, downloading text")
    with ThreadPoolExecutor(workers) as pool:
        rows = []
        for k, row in enumerate(pool.map(lambda m: _one_filing(m, headers, limit, lm), metas)):
            rows.append(row)
            if k % 1000 == 0:
                print(f"filings: {k} done")
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_parquet(RAW / "filings.parquet")
    return df


def main(parts: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    uni = universe()
    limit = int(__import__("os").environ.get("LIMIT") or 0)
    if limit:  # small trial runs
        uni = uni.head(limit)
        uni.write_parquet(RAW / "universe.parquet")
    tickers = uni["ticker"].to_list()
    print(f"universe: {len(tickers)} tickers")
    if "prices" in parts:
        print(f"yahoo: {yahoo(tickers).height} rows")
        print(f"nasdaq: {nasdaq(tickers).height} rows")
    if "filings" in parts:
        print(f"filings: {filings(uni).height} 8-Ks")


if __name__ == "__main__":
    main(sys.argv[1:] or ["prices", "filings"])
