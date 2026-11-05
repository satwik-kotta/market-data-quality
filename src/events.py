"""8-K event study: how much do stocks move when a company files an 8-K, and does the text tell you which way?

Abnormal return (AR) is the stock's return minus the equal weighted return of all clean stocks that day.
The event day is the first trading day the market could react: the filing day if EDGAR accepted it
before 4 pm Eastern, otherwise the next trading day.
"""
import numpy as np
import polars as pl

WINDOW = range(-5, 6)
MIN_ITEM_COUNT = 150
ITEM_NAMES = {
    "1.01": "Material agreement", "1.02": "Agreement ended", "2.01": "Acquisition or sale completed",
    "2.02": "Earnings results", "2.03": "New debt", "2.05": "Restructuring costs", "2.06": "Impairment",
    "3.01": "Listing notice", "5.02": "Executive or director change", "5.03": "Bylaws change",
    "5.07": "Shareholder vote results", "7.01": "Reg FD disclosure", "8.01": "Other events",
    "9.01": "Exhibits attached",
}


def abnormal_returns(clean: pl.DataFrame, sessions: pl.Series) -> pl.DataFrame:
    c = pl.col
    r = clean.filter(c("ret").is_not_null()).select("ticker", "date", "ret")
    mkt = r.group_by("date").agg(c("ret").mean().alias("mkt"), pl.len().alias("n_stocks"))
    idx = pl.DataFrame({"date": sessions, "t": np.arange(len(sessions))})
    return (r.join(mkt, on="date").with_columns((c("ret") - c("mkt")).alias("ar"))
             .join(idx, on="date", how="inner"))


def event_days(filings: pl.DataFrame, sessions: pl.Series) -> pl.DataFrame:
    """Map EDGAR acceptance times to the first trading day the market could react."""
    c = pl.col
    # EDGAR acceptance times are Eastern wall clock time, even though the string ends in Z.
    f = filings.with_columns(c("accepted").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("_ts"))
    f = f.with_columns(
        pl.when(c("_ts").dt.hour() >= 16).then(c("_ts").dt.date() + pl.duration(days=1))
        .otherwise(c("_ts").dt.date()).alias("_target")
    ).sort("_target")
    idx = pl.DataFrame({"event_date": sessions, "t0": np.arange(len(sessions))}).sort("event_date")
    return f.join_asof(idx, left_on="_target", right_on="event_date", strategy="forward").drop("_ts", "_target")


def event_panel(events: pl.DataFrame, ar: pl.DataFrame) -> pl.DataFrame:
    """One row per filing with AR on each day from 5 before to 5 after the event day."""
    c = pl.col
    e = events.filter(c("t0").is_not_null())
    out = e
    for k in WINDOW:
        col = f"ar_{k}".replace("-", "m")
        out = out.join(
            ar.select("ticker", (c("t") - k).alias("t0"), c("ar").alias(col)),
            on=["ticker", "t0"], how="left", maintain_order="left")
    return out.with_columns(
        c("ar_0").abs().alias("abs_ar0"),
        pl.sum_horizontal([c(f"ar_{k}") for k in range(1, 6)]).alias("car_post"),
        pl.sum_horizontal([c(f"ar_m{k}") for k in range(1, 6)]).alias("car_pre"),
        pl.when(c("n_words") > 0).then((c("lm_positive") - c("lm_negative")) / c("n_words") * 100).alias("tone"),
    )


def item_dummies(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    items = panel.select(pl.col("items").fill_null("").str.split(",").alias("i")).explode("i")["i"].str.strip_chars()
    counts = items.filter(items != "").value_counts().sort("count", descending=True)
    keep = [i for i, n in counts.iter_rows() if n >= MIN_ITEM_COUNT and i in ITEM_NAMES and i != "9.01"]
    return panel.with_columns([
        pl.col("items").fill_null("").str.contains(i.replace(".", r"\."), literal=False).cast(pl.Int64).alias(f"item_{i}")
        for i in keep
    ]), keep


def ols(y: np.ndarray, X: np.ndarray, names: list[str], cluster: np.ndarray | None = None) -> dict:
    import statsmodels.api as sm

    X = sm.add_constant(X, has_constant="add")
    if cluster is not None:
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    else:
        fit = sm.OLS(y, X).fit(cov_type="HC1")
    ci = fit.conf_int()
    return {n: {"coef": float(fit.params[i]), "ci": [float(ci[i][0]), float(ci[i][1])], "p": float(fit.pvalues[i])}
            for i, n in enumerate(["const"] + names)} | {"n": int(fit.nobs), "r2": float(fit.rsquared)}


def item_effects(panel: pl.DataFrame, ar: pl.DataFrame) -> dict:
    """How much larger is the event day move, in absolute terms, for each kind of 8-K."""
    d, keep = item_dummies(panel.filter(pl.col("abs_ar0").is_not_null()))
    if not keep:
        return {"items": [], "note": "too few filings per item type"}
    cols = [f"item_{i}" for i in keep]
    fit = ols(d["abs_ar0"].to_numpy() * 100, d.select(cols).to_numpy().astype(float), keep,
              cluster=d["ticker"].to_numpy())
    normal_day = float(ar["ar"].abs().mean() * 100)
    rows = []
    for i in keep:
        sub = d.filter(pl.col(f"item_{i}") == 1)
        rows.append({"item": i, "name": ITEM_NAMES[i], "filings": sub.height,
                     "mean_abs_move_pct": float(sub["abs_ar0"].mean() * 100),
                     "times_normal_day": float(sub["abs_ar0"].mean() * 100 / normal_day),
                     "extra_move_pts": fit[i]["coef"], "ci": fit[i]["ci"]})
    return {"normal_day_abs_ar_pct": normal_day, "all_8k_abs_ar_pct": float(d["abs_ar0"].mean() * 100),
            "items": sorted(rows, key=lambda r: -r["mean_abs_move_pct"]), "fit_n": fit["n"]}


def split_periods(panel: pl.DataFrame, frac: float = 0.6) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    dates = panel["event_date"].sort()
    cut = dates[int(len(dates) * frac)]
    return panel.filter(pl.col("event_date") < cut), panel.filter(pl.col("event_date") >= cut), str(cut)


def tercile_spread(d: pl.DataFrame, x: str, y: str, edges: tuple[float, float], n_boot: int = 2000, seed: int = 0) -> dict:
    lo, hi = edges
    top = d.filter(pl.col(x) >= hi)[y].to_numpy()
    bot = d.filter(pl.col(x) <= lo)[y].to_numpy()
    rng = np.random.default_rng(seed)
    boots = [rng.choice(top, len(top)).mean() - rng.choice(bot, len(bot)).mean() for _ in range(n_boot)]
    return {"spread_pct": float((top.mean() - bot.mean()) * 100),
            "ci_pct": [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)],
            "n_top": len(top), "n_bottom": len(bot)}


def tone_test(panel: pl.DataFrame, item: str | None = None) -> dict:
    """Does a more positive filing go with a higher event day return? Terciles are set on the earlier
    period only and then applied to the later period, so the later period is a clean out of sample test.
    The 5 days before the filing are a placebo: text should not explain moves that already happened,
    unless the news leaked or was expected."""
    c = pl.col
    d = panel.filter(c("tone").is_not_null() & c("ar_0").is_not_null() & (c("n_words") >= 100))
    if item:
        d = d.filter(c("items").fill_null("").str.contains(item.replace(".", r"\.")))
    train, test, cut = split_periods(d)
    edges = (float(train["tone"].quantile(1 / 3)), float(train["tone"].quantile(2 / 3)))
    out = {"item": item, "filings": d.height, "split_date": cut, "tercile_edges": edges}
    for name, part in [("earlier", train), ("later", test)]:
        out[name] = {
            "event_day": tercile_spread(part, "tone", "ar_0", edges),
            "next_5_days": tercile_spread(part.filter(c("car_post").is_not_null()), "tone", "car_post", edges),
            "placebo_5_days_before": tercile_spread(part.filter(c("car_pre").is_not_null()), "tone", "car_pre", edges),
            "rank_corr_event_day": float(part.select(pl.corr("tone", "ar_0", method="spearman")).item()),
        }
    return out
