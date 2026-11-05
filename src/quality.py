"""Normalize two price vendors into one schema, validate each, compare them, and build a clean table.

Checks on each vendor
    nonpositive_price     any of open, high, low, close at or below zero
    ohlc_inconsistent     low above open or close, or high below them
    zero_volume           a trading day with no volume
    stale_price           the same close for 5 or more trading days in a row
    return_outlier        a daily move over 25% that is not a split
    off_calendar          a row on a day the NYSE was closed
    missing_session       an NYSE trading day with no row, inside the ticker's own date range

Checks across vendors
    adjustment convention which Yahoo series the Nasdaq close matches, decided per ticker: unadjusted
                          (rebuilt from Yahoo's split history), split adjusted, or split and dividend adjusted
    price_mismatch        the two vendors disagree by more than 0.5% on the same day
"""
import polars as pl

PRICE_COLS = ["open", "high", "low", "close"]
MISMATCH_TOL = 0.005
STALE_RUN = 5
OUTLIER = 0.25
SEVERE = {"nonpositive_price", "ohlc_inconsistent", "off_calendar"}


def _finish(df: pl.DataFrame, source: str) -> tuple[pl.DataFrame, dict]:
    n_in = df.height
    df = df.filter(pl.col("close").is_not_null())
    n_null = n_in - df.height
    before = df.height
    df = df.unique(["ticker", "date"], keep="last", maintain_order=True).sort(["ticker", "date"])
    return df.with_columns(pl.lit(source).alias("source")), {
        "rows_received": n_in, "rows_without_close": n_null, "duplicate_rows": before - df.height,
        "rows_kept": df.height, "tickers": df["ticker"].n_unique(),
    }


def normalize_yahoo(raw: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    df = raw.select(
        pl.col("date").cast(pl.Date), pl.col("ticker"),
        *[pl.col(c).cast(pl.Float64) for c in PRICE_COLS],
        pl.col("adj_close").cast(pl.Float64), pl.col("volume").cast(pl.Float64),
        pl.col("dividends").cast(pl.Float64).fill_null(0), pl.col("splits").cast(pl.Float64).fill_null(0),
    )
    return _finish(df, "yahoo")


def _num(col: str) -> pl.Expr:
    """Nasdaq sends numbers as text like "$1,234.56"; "N/A" becomes null."""
    return pl.col(col).cast(pl.Utf8).str.replace_all(r"[$,]", "").str.strip_chars().cast(pl.Float64, strict=False)


def normalize_nasdaq(raw: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    df = raw.rename({c: c.lower() for c in raw.columns}).select(
        pl.col("date").cast(pl.Utf8).str.to_date("%m/%d/%Y"), pl.col("ticker"),
        *[_num(c).alias(c) for c in PRICE_COLS], _num("volume").alias("volume"),
    )
    return _finish(df, "nasdaq")


def row_checks(df: pl.DataFrame, sessions: pl.Series, splits: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (ticker, date, check) that failed."""
    c = pl.col
    d = df.sort(["ticker", "date"]).with_columns(
        (c("close") / c("close").shift(1).over("ticker")).log().alias("_lr"),
        (c("close") != c("close").shift(1).over("ticker")).fill_null(True).cum_sum().over("ticker").alias("_run"),
    )
    d = d.with_columns(pl.len().over(["ticker", "_run"]).alias("_run_len"))
    if splits is not None:  # use another vendor's split dates, e.g. Yahoo's for Nasdaq
        d = d.drop("splits", strict=False).join(splits, on=["ticker", "date"], how="left", maintain_order="left")
    elif "splits" not in d.columns:
        d = d.with_columns(pl.lit(0.0).alias("splits"))
    split_day = c("splits").fill_null(0) > 0
    sess = sessions.implode()
    checks = {
        "nonpositive_price": pl.any_horizontal([c(x) <= 0 for x in PRICE_COLS]),
        "ohlc_inconsistent": (c("low") > pl.min_horizontal("open", "close") * (1 + 1e-6))
        | (c("high") < pl.max_horizontal("open", "close") * (1 - 1e-6)),
        "zero_volume": c("volume") == 0,
        "stale_price": c("_run_len") >= STALE_RUN,
        "return_outlier": (c("_lr").abs() > OUTLIER) & ~split_day,
        "off_calendar": ~c("date").is_in(sess),
    }
    flagged = pl.concat([
        d.filter(expr.fill_null(False)).select("ticker", "date", pl.lit(name).alias("check")) for name, expr in checks.items()
    ])
    # Missing sessions: NYSE trading days inside each ticker's first to last date with no row.
    span = df.group_by("ticker").agg(c("date").min().alias("first"), c("date").max().alias("last"))
    expected = span.join(pl.DataFrame({"date": sessions}), how="cross").filter(
        (c("date") >= c("first")) & (c("date") <= c("last"))).select("ticker", "date")
    missing = expected.join(df.select("ticker", "date"), on=["ticker", "date"], how="anti").with_columns(
        pl.lit("missing_session").alias("check"))
    return pl.concat([flagged, missing]).sort(["ticker", "date", "check"])


# Order matters for ties: with no splits, unadjusted and split adjusted are identical, so call it split adjusted.
CONVENTIONS = {"split_adjusted": "lr_close", "unadjusted": "lr_raw", "dividend_adjusted": "lr_adj"}


def compare_vendors(y: pl.DataFrame, n: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per ticker adjustment convention and daily agreement between the two vendors."""
    c = pl.col
    # Yahoo's close is split adjusted. Undo later splits to get the price as it actually traded.
    sf = pl.when(c("splits") > 0).then(c("splits")).otherwise(1.0)
    y = y.sort(["ticker", "date"]).with_columns(
        (c("close") * sf.reverse().cum_prod().reverse().over("ticker") / sf).alias("raw_close"))
    both = y.select("ticker", "date", "close", "adj_close", "raw_close", "volume").join(
        n.select("ticker", "date", c("close").alias("n_close"), c("volume").alias("n_volume")),
        on=["ticker", "date"], how="inner",
    ).with_columns(
        (c("n_close") / c("raw_close")).log().alias("lr_raw"),
        (c("n_close") / c("close")).log().alias("lr_close"),
        (c("n_close") / c("adj_close")).log().alias("lr_adj"),
    )
    conv = both.group_by("ticker").agg(
        *[c(col).abs().median().alias(f"med_gap_{name}") for name, col in CONVENTIONS.items()],
        (c("n_volume") / c("volume")).filter(c("volume") > 0).median().alias("volume_ratio"),
        pl.len().alias("days_both"),
    )
    gaps = [f"med_gap_{k}" for k in CONVENTIONS]
    best = pl.concat_list(gaps).list.arg_min()
    conv = conv.with_columns(
        pl.when(pl.min_horizontal(gaps) < 0.002)
        .then(best.replace_strict(dict(enumerate(CONVENTIONS)), return_dtype=pl.Utf8))
        .otherwise(pl.lit("unclear")).alias("nasdaq_convention")
    ).sort("ticker")
    gap = pl.lit(None, dtype=pl.Float64)
    for name, col in CONVENTIONS.items():
        gap = pl.when(c("nasdaq_convention") == name).then(c(col)).otherwise(gap)
    daily = both.join(conv.select("ticker", "nasdaq_convention"), on="ticker", how="left").with_columns(
        pl.when(c("nasdaq_convention") == "unclear").then(c("lr_close")).otherwise(gap).alias("gap")
    ).sort(["ticker", "date"]).with_columns(
        (c("gap").abs() > MISMATCH_TOL).alias("mismatch"),
        # A constant gap means one vendor scaled its history (a spin-off, say) and the daily moves still agree.
        (c("gap") - c("gap").shift(1).over("ticker")).alias("return_gap"),
    ).with_columns((c("return_gap").abs() > MISMATCH_TOL).fill_null(True).alias("return_mismatch"))
    return conv, daily


def clean_table(y: pl.DataFrame, n: pl.DataFrame, issues_y: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Yahoo prices with a status per row, and returns only where both ends of the move are usable.

    confirmed      Nasdaq agrees within 0.5%
    offset         the price levels differ but the daily move agrees within 0.5%, so one vendor scaled its history
    single_source  Nasdaq has no row for that day
    conflict       the vendors disagree by more than 0.5% on the level and on the daily move
    rejected       a severe check failed on the Yahoo row
    """
    c = pl.col
    severe = issues_y.filter(c("check").is_in(list(SEVERE))).select("ticker", "date").unique().with_columns(
        pl.lit(True).alias("_severe"))
    out = (y.join(daily.select("ticker", "date", "gap", "mismatch", "return_mismatch"), on=["ticker", "date"], how="left", maintain_order="left")
            .join(severe, on=["ticker", "date"], how="left", maintain_order="left")
            .with_columns(
                pl.when(c("_severe").fill_null(False)).then(pl.lit("rejected"))
                .when(c("mismatch").is_null()).then(pl.lit("single_source"))
                .when(c("mismatch") & ~c("return_mismatch")).then(pl.lit("offset"))
                .when(c("mismatch")).then(pl.lit("conflict"))
                .otherwise(pl.lit("confirmed")).alias("status"))
            .drop("_severe", "return_mismatch")
            .sort(["ticker", "date"]))
    usable = c("status").is_in(["confirmed", "offset", "single_source"])
    return out.with_columns(
        pl.when(usable & usable.shift(1).over("ticker"))
        .then(c("adj_close") / c("adj_close").shift(1).over("ticker") - 1).alias("ret")
    )
