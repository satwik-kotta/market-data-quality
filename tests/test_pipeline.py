import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import events  # noqa: E402
import quality  # noqa: E402
from common import sessions  # noqa: E402

SESS = sessions(date(2024, 1, 1), date(2024, 6, 28))


def fake_yahoo(ticker="AAA", seed=0, div_yield_daily=0.0001):
    rng = np.random.default_rng(seed)
    n = len(SESS)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # Adjusted close scales history down for dividends, so it ends equal to close.
    adj = close * np.exp(-div_yield_daily * np.arange(n)[::-1])
    return pl.DataFrame({
        "date": SESS, "ticker": [ticker] * n, "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "adj_close": adj, "volume": rng.integers(1_000, 5_000, n).astype(float),
        "dividends": [0.0] * n, "splits": [0.0] * n,
    })


def fake_nasdaq(y: pl.DataFrame, series="adj_close"):
    money = lambda c: pl.format("${}", pl.col(c).round(4))  # noqa: E731
    return y.select(pl.col("date").dt.strftime("%m/%d/%Y").alias("Date"), pl.col("ticker"),
                    money("open").alias("Open"), money("high").alias("High"), money("low").alias("Low"),
                    money(series).alias("Close/Last"), pl.col("volume").cast(pl.Int64).cast(pl.Utf8).alias("Volume"))


def test_normalize_drops_duplicates_and_nulls():
    raw = fake_yahoo()
    raw = pl.concat([raw, raw.head(3)]).with_columns(
        pl.when(pl.int_range(pl.len()) == 10).then(None).otherwise(pl.col("close")).alias("close"))
    df, log = quality.normalize_yahoo(raw)
    assert log["duplicate_rows"] == 3 and log["rows_without_close"] == 1
    assert df.select(["ticker", "date"]).is_unique().all()


def test_row_checks_find_each_planted_problem():
    y, _ = quality.normalize_yahoo(fake_yahoo())
    d = y["date"]
    y = y.with_columns(
        pl.when(pl.col("date") == d[5]).then(pl.col("close") * 2).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("date") == d[5]).then(pl.col("close") * 1.5).otherwise(pl.col("low")).alias("low"),
        pl.when(pl.col("date") == d[40]).then(pl.col("close") * 1.5).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("date") == d[60]).then(0.0).otherwise(pl.col("volume")).alias("volume"),
    )
    y = y.with_columns(  # 6 identical closes in a row
        pl.when(pl.col("date").is_between(d[80], d[85])).then(123.0).otherwise(pl.col("close")).alias("close"))
    y = y.filter(pl.col("date") != d[100])  # a missing trading day
    y = pl.concat([y, y.head(1).with_columns(pl.lit(date(2024, 1, 6)).alias("date"))])  # a Saturday
    issues = quality.row_checks(y, SESS)
    found = {(r["check"], r["date"]) for r in issues.iter_rows(named=True)}
    assert ("ohlc_inconsistent", d[5]) in found
    assert ("return_outlier", d[40]) in found
    assert ("zero_volume", d[60]) in found
    assert ("stale_price", d[83]) in found
    assert ("missing_session", d[100]) in found
    assert ("off_calendar", date(2024, 1, 6)) in found


def test_split_day_is_not_an_outlier():
    y, _ = quality.normalize_yahoo(fake_yahoo())
    d = y["date"]
    y = y.with_columns(pl.when(pl.col("date") >= d[30]).then(pl.col("close") / 4).otherwise(pl.col("close")).alias("close"))
    splits = pl.DataFrame({"ticker": ["AAA"], "date": [d[30]], "splits": [4.0]})
    issues = quality.row_checks(y, SESS, splits)
    assert issues.filter(pl.col("check") == "return_outlier").height == 0


def test_vendor_convention_and_mismatch():
    y1, _ = quality.normalize_yahoo(fake_yahoo("AAA", 1, div_yield_daily=0.0002))
    y2, _ = quality.normalize_yahoo(fake_yahoo("BBB", 2, div_yield_daily=0.0002))
    # CCC had a 2 for 1 split. Yahoo's close is split adjusted, Nasdaq shows the price as it traded.
    y3, _ = quality.normalize_yahoo(fake_yahoo("CCC", 3))
    split_day = y3["date"][60]
    y3 = y3.with_columns(pl.when(pl.col("date") == split_day).then(2.0).otherwise(0.0).alias("splits"))
    traded = y3.with_columns(pl.when(pl.col("date") < split_day).then(pl.col("close") * 2)
                             .otherwise(pl.col("close")).alias("traded"))
    y = pl.concat([y1, y2, y3])
    raw = pl.concat([fake_nasdaq(y1, "adj_close"), fake_nasdaq(y2, "close"), fake_nasdaq(traded, "traded")])
    bad_day = y1["date"][50]
    raw = raw.with_columns(
        pl.when((pl.col("ticker") == "AAA") & (pl.col("Date") == bad_day.strftime("%m/%d/%Y")))
        .then(pl.lit("$999.00")).otherwise(pl.col("Close/Last")).alias("Close/Last"))
    n, _ = quality.normalize_nasdaq(raw.rename({"Close/Last": "Close"}))
    assert n["close"].dtype == pl.Float64 and n["volume"].null_count() == 0
    conv, daily = quality.compare_vendors(y, n)
    c = dict(conv.select("ticker", "nasdaq_convention").iter_rows())
    assert c == {"AAA": "dividend_adjusted", "BBB": "split_adjusted", "CCC": "unadjusted"}
    mism = daily.filter(pl.col("mismatch"))
    assert mism.select("ticker", "date").rows() == [("AAA", bad_day)]

    clean = quality.clean_table(y, n, quality.row_checks(y, SESS), daily)
    row = clean.filter((pl.col("ticker") == "AAA") & (pl.col("date") == bad_day)).row(0, named=True)
    assert row["status"] == "conflict" and row["ret"] is None


def test_event_day_respects_market_close():
    f = pl.DataFrame({"ticker": ["A", "A", "A"], "accepted": [
        "2024-03-01T09:15:00.000Z",  # Friday morning, same day
        "2024-03-01T16:30:00.000Z",  # Friday after the close, next trading day is Monday
        "2024-03-29T10:00:00.000Z",  # Good Friday, market closed, next trading day
    ]})
    ev = events.event_days(f, SESS).sort("accepted")
    assert ev["event_date"].to_list() == [date(2024, 3, 1), date(2024, 3, 4), date(2024, 4, 1)]


def test_event_panel_aligns_window():
    ar = pl.DataFrame({"ticker": ["A"] * 20, "t": list(range(20)), "ar": [k / 100 for k in range(20)]})
    ev = pl.DataFrame({"ticker": ["A"], "t0": [10], "n_words": [200], "lm_positive": [5], "lm_negative": [1],
                       "items": ["2.02"]})
    p = events.event_panel(ev, ar).row(0, named=True)
    assert p["ar_0"] == 0.10 and p["ar_m1"] == 0.09 and p["ar_5"] == 0.15
    assert abs(p["car_post"] - sum(k / 100 for k in range(11, 16))) < 1e-12
    assert abs(p["tone"] - 2.0) < 1e-12
