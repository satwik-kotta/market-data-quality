"""Process the raw downloads: normalize, validate, compare vendors, build the clean table, run the event study,
and write outputs/results.json plus the issue tables."""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events  # noqa: E402
import quality  # noqa: E402
from common import OUT, PROC, RAW, sessions  # noqa: E402


def main() -> None:
    OUT.mkdir(exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    sess = sessions()
    uni = pl.read_parquet(RAW / "universe.parquet")

    y, y_log = quality.normalize_yahoo(pl.read_parquet(RAW / "yahoo.parquet"))
    s, s_log = quality.normalize_nasdaq(pl.read_parquet(RAW / "nasdaq.parquet"))
    y = y.filter(pl.col("date") <= sess.max())
    s = s.filter(pl.col("date") <= sess.max())
    splits = y.filter(pl.col("splits") > 0).select("ticker", "date", "splits")
    issues_y = quality.row_checks(y, sess).with_columns(pl.lit("yahoo").alias("vendor"))
    issues_s = quality.row_checks(s, sess, splits).with_columns(pl.lit("nasdaq").alias("vendor"))
    conv, daily = quality.compare_vendors(y, s)
    clean = quality.clean_table(y, s, issues_y, daily)

    issues = pl.concat([issues_y, issues_s])
    issues.write_csv(OUT / "issues.csv")
    conv.write_csv(OUT / "vendor_conventions.csv")
    mism = daily.filter(pl.col("mismatch")).sort(pl.col("gap").abs(), descending=True)
    mism.write_csv(OUT / "price_mismatches.csv")
    clean.write_parquet(PROC / "clean_prices.parquet")

    ar = events.abnormal_returns(clean, sess)
    filings = pl.read_parquet(RAW / "filings.parquet")
    ev = events.event_days(filings, sess)
    panel = events.event_panel(ev, ar)
    panel.write_parquet(PROC / "event_panel.parquet")

    # Where the vendors disagree on levels, find the session the gap ends and what Yahoo recorded the next day.
    nxt = pl.DataFrame({"date": sess, "next_session": sess.shift(-1)})
    odd_splits = splits.filter((pl.col("splits") - pl.col("splits").round()).abs() > 1e-9)
    breaks = (daily.filter(pl.col("mismatch")).group_by("ticker")
              .agg(pl.len().alias("days"), pl.col("date").max().alias("last_mismatch"),
                   pl.col("gap").median().alias("median_gap"), pl.col("return_mismatch").sum().alias("return_conflicts"))
              .join(nxt, left_on="last_mismatch", right_on="date", how="left")
              .join(odd_splits.rename({"date": "next_session", "splits": "yahoo_ratio"}),
                    on=["ticker", "next_session"], how="left")
              .sort("days", descending=True))
    breaks.write_csv(OUT / "level_breaks.csv")

    by_check = issues.group_by(["vendor", "check"]).agg(pl.len().alias("rows"), pl.col("ticker").n_unique().alias("tickers")).sort(["vendor", "rows"], descending=[False, True])
    status = clean.group_by("status").agg(pl.len().alias("rows")).sort("rows", descending=True)
    results = {
        "universe": uni.height,
        "sessions": {"first": str(sess.min()), "last": str(sess.max()), "count": len(sess)},
        "vendors": {"yahoo": y_log, "nasdaq": s_log},
        "tickers_both_vendors": conv.height,
        "row_issues": by_check.to_dicts(),
        "nasdaq_convention": conv.group_by("nasdaq_convention").agg(pl.len().alias("tickers")).sort("tickers", descending=True).to_dicts(),
        "days_compared": daily.height,
        "mismatch_days": int(daily["mismatch"].sum()),
        "mismatch_tickers": int(daily.filter(pl.col("mismatch"))["ticker"].n_unique()),
        "return_conflict_days": int((daily["mismatch"] & daily["return_mismatch"]).sum()),
        "yahoo_fractional_split_records": odd_splits.height,
        "breaks_ending_before_fractional_split": int(breaks["yahoo_ratio"].is_not_null().sum()),
        "breaks_with_100_plus_days": int((breaks["days"] >= 100).sum()),
        "volume_ratio_median": float(conv["volume_ratio"].median()),
        "clean_status": status.to_dicts(),
        "returns_usable": int(clean["ret"].is_not_null().sum()),
        "filings": {
            "eight_ks": filings.height,
            "with_text": int(filings["n_words"].is_not_null().sum()),
            "press_release_text": int((filings["text_source"] == "exhibit_99").sum()),
            "matched_to_returns": int(panel["ar_0"].is_not_null().sum()),
        },
        "item_effects": events.item_effects(panel, ar),
        "tone_all": events.tone_test(panel),
        "tone_earnings": events.tone_test(panel, "2.02"),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
