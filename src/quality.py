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
    adjustment convention which Yahoo series (split adjusted close, or close adjusted for splits and
                          dividends) the Stooq close matches, decided per ticker
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


def normalize_stooq(raw: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    df = raw.rename({c: c.lower() for c in raw.columns}).select(
        pl.col("date").cast(pl.Utf8).str.to_date("%Y-%m-%d"), pl.col("ticker"),
        *[pl.col(c).cast(pl.Float64) for c in PRICE_COLS], pl.col("volume").cast(pl.Float64),
    )
    return _finish(df, "stooq")


def row_checks(df: pl.DataFrame, sessions: pl.Series, splits: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per (ticker, date, check) that failed."""
    c = pl.col
    d = df.sort(["ticker", "date"]).with_columns(
        (c("close") / c("close").shift(1).over("ticker")).log().alias("_lr"),
        (c("close") != c("close").shift(1).over("ticker")).fill_null(True).cum_sum().over("ticker").alias("_run"),
    )
    d = d.with_columns(pl.len().over(["ticker", "_run"]).alias("_run_len"))
    if splits is not None:  # use another vendor's split dates, e.g. Yahoo's for Stooq
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


def compare_vendors(y: pl.DataFrame, s: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per ticker adjustment convention and daily agreement between the two vendors."""
    c = pl.col
    both = y.select("ticker", "date", "close", "adj_close", "volume").join(
        s.select("ticker", "date", c("close").alias("s_close"), c("volume").alias("s_volume")),
        on=["ticker", "date"], how="inner",
    ).with_columns(
        (c("s_close") / c("close")).log().alias("lr_close"),
        (c("s_close") / c("adj_close")).log().alias("lr_adj"),
    )
    conv = both.group_by("ticker").agg(
        c("lr_close").abs().median().alias("med_gap_close"),
        c("lr_adj").abs().median().alias("med_gap_adj"),
        (c("s_volume") / c("volume")).filter(c("volume") > 0).median().alias("volume_ratio"),
        pl.len().alias("days_both"),
    ).with_columns(
        pl.when((c("med_gap_adj") < c("med_gap_close")) & (c("med_gap_adj") < 0.002)).then(pl.lit("dividend_adjusted"))
        .when(c("med_gap_close") < 0.002).then(pl.lit("split_adjusted"))
        .otherwise(pl.lit("unclear")).alias("stooq_convention")
    ).sort("ticker")
    daily = both.join(conv.select("ticker", "stooq_convention"), on="ticker", how="left").with_columns(
        pl.when(c("stooq_convention") == "split_adjusted").then(c("lr_close")).otherwise(c("lr_adj")).alias("gap")
    ).with_columns((c("gap").abs() > MISMATCH_TOL).alias("mismatch"))
    return conv, daily


def clean_table(y: pl.DataFrame, s: pl.DataFrame, issues_y: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Yahoo prices with a status per row, and returns only where both ends of the move are usable.

    confirmed      Stooq agrees within 0.5%
    single_source  Stooq has no row for that day
    conflict       the vendors disagree by more than 0.5%
    rejected       a severe check failed on the Yahoo row
    """
    c = pl.col
    severe = issues_y.filter(c("check").is_in(list(SEVERE))).select("ticker", "date").unique().with_columns(
        pl.lit(True).alias("_severe"))
    out = (y.join(daily.select("ticker", "date", "gap", "mismatch"), on=["ticker", "date"], how="left", maintain_order="left")
            .join(severe, on=["ticker", "date"], how="left", maintain_order="left")
            .with_columns(
                pl.when(c("_severe").fill_null(False)).then(pl.lit("rejected"))
                .when(c("mismatch").is_null()).then(pl.lit("single_source"))
                .when(c("mismatch")).then(pl.lit("conflict"))
                .otherwise(pl.lit("confirmed")).alias("status"))
            .drop("_severe")
            .sort(["ticker", "date"]))
    usable = c("status").is_in(["confirmed", "single_source"])
    return out.with_columns(
        pl.when(usable & usable.shift(1).over("ticker"))
        .then(c("adj_close") / c("adj_close").shift(1).over("ticker") - 1).alias("ret")
    )
