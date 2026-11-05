"""Download the raw inputs. Runs on GitHub Actions, where these hosts are reachable.

    universe   S&P 500 members with CIK numbers (datasets/s-and-p-500-companies on GitHub)
    yahoo      daily OHLCV, split adjusted close, adjusted close, dividends and splits (yfinance)
    stooq      daily OHLCV from Stooq's CSV endpoint
    filings    every 8-K filed by those companies since FILING_START, from SEC EDGAR, with
               Loughran-McDonald word counts taken from the filing text (the text itself is not stored)

Each source is written to data/raw as parquet. Vendor data is stored exactly as received, so all
cleaning happens later and can be checked.
"""
import io
import re
import sys
import time
import urllib.request
from datetime import date, timedelta
from html import unescape

import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import FILING_START, PRICE_START, RAW, SEC_USER_AGENT, stooq_symbol, yahoo_symbol  # noqa: E402

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
        wide = yf.download([yahoo_symbol(t) for t in batch], start=PRICE_START.isoformat(),
                           end=(date.today() + timedelta(days=1)).isoformat(), auto_adjust=False, actions=True,
                           group_by="ticker", threads=True, progress=False)
        for t in batch:
            sym = yahoo_symbol(t)
            if sym not in wide.columns.get_level_values(0):
                continue
            g = wide[sym].reset_index()
            g.columns = [str(c).lower().replace(" ", "_") for c in g.columns]
            frames.append(pl.from_pandas(g).with_columns(pl.lit(t).alias("ticker")))
        time.sleep(1)
    df = pl.concat(frames, how="diagonal_relaxed")
    df = df.rename({"stock_splits": "splits"}) if "stock_splits" in df.columns else df
    df.write_parquet(RAW / "yahoo.parquet")
    return df


def stooq(tickers: list[str]) -> pl.DataFrame:
    frames, failed = [], []
    for t in tickers:
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol(t)}&i=d&d1={PRICE_START:%Y%m%d}&d2={date.today():%Y%m%d}"
        try:
            body = http_get(url, retries=2)
        except Exception:
            failed.append(t)
            continue
        if not body.startswith(b"Date"):
            failed.append(t)  # "No data" or a rate limit message
            if b"limit" in body.lower():
                print("Stooq rate limit reached", file=sys.stderr)
                break
            continue
        frames.append(pl.read_csv(io.BytesIO(body)).with_columns(pl.lit(t).alias("ticker")))
        time.sleep(0.4)
    print(f"stooq: {len(frames)} tickers, {len(failed)} failed")
    df = pl.concat(frames, how="diagonal_relaxed")
    df.write_parquet(RAW / "stooq.parquet")
    return df


TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
EX99 = re.compile(r"ex-?_?99", re.I)


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return SPACE.sub(" ", unescape(TAG.sub(" ", html))).strip()


def filings(uni: pl.DataFrame, max_per_company: int = 60) -> pl.DataFrame:
    import pysentiment2 as ps

    lm = ps.LM()
    headers = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "identity"}
    rows = []
    for k, (ticker, cik) in enumerate(uni.select("ticker", "cik").iter_rows()):
        try:
            sub = __import__("json").loads(http_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers))
        except Exception as e:
            print(f"submissions failed {ticker}: {e}", file=sys.stderr)
            continue
        time.sleep(0.12)
        r = sub["filings"]["recent"]
        taken = 0
        for i, form in enumerate(r["form"]):
            if form != "8-K" or r["filingDate"][i] < FILING_START.isoformat() or taken >= max_per_company:
                continue
            acc = r["accessionNumber"][i]
            row = {
                "ticker": ticker, "cik": cik, "accession": acc, "filing_date": r["filingDate"][i],
                "accepted": r["acceptanceDateTime"][i], "items": r["items"][i], "document": r["primaryDocument"][i],
                "text_source": None, "n_words": None, "lm_positive": None, "lm_negative": None,
            }
            folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}"
            try:
                # The news is usually in the attached press release (exhibit 99), not the 8-K cover form.
                listing = __import__("json").loads(http_get(f"{folder}/index.json", headers, retries=2))
                time.sleep(0.12)
                names = [f["name"] for f in listing["directory"]["item"]]
                ex99 = [n for n in names if EX99.search(n) and n.lower().endswith((".htm", ".html", ".txt"))]
                doc = ex99[0] if ex99 else r["primaryDocument"][i]
                row["text_source"] = "exhibit_99" if ex99 else "form"
                text = html_to_text(http_get(f"{folder}/{doc}", headers, retries=2).decode("utf-8", "ignore"))
                tokens = lm.tokenize(text)
                score = lm.get_score(tokens)
                row.update(n_words=len(tokens), lm_positive=int(score["Positive"]), lm_negative=int(score["Negative"]))
            except Exception as e:
                print(f"document failed {ticker} {acc}: {e}", file=sys.stderr)
            rows.append(row)
            taken += 1
            time.sleep(0.12)  # stay under SEC's 10 requests per second
        if k % 50 == 0:
            print(f"filings: {k} companies, {len(rows)} 8-Ks")
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_parquet(RAW / "filings.parquet")
    return df


def main(parts: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    uni = universe()
    tickers = uni["ticker"].to_list()
    print(f"universe: {len(tickers)} tickers")
    if "prices" in parts:
        print(f"yahoo: {yahoo(tickers).height} rows")
        print(f"stooq: {stooq(tickers).height} rows")
    if "filings" in parts:
        print(f"filings: {filings(uni).height} 8-Ks")


if __name__ == "__main__":
    main(sys.argv[1:] or ["prices", "filings"])
