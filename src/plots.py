"""Figures for the report (saved as PNG in results/figures/)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

# palette (colour-blind validated categorical order) and recessive chart ink
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BASELINE_GREY = "#b9b8b1"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

TRIVIAL = {"Previous-day volatility (naive)", "Historical mean", "Coin flip",
           "Most frequent label", "Previous direction (momentum)"}


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _model_colour(name):
    if name in TRIVIAL:
        return BASELINE_GREY
    if "sentiment" in name.lower() or "hawkish" in name.lower():
        return ORANGE
    return BLUE


def _legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=BASELINE_GREY, label="Trivial baseline"),
            Patch(color=BLUE, label="Model without text"),
            Patch(color=ORANGE, label="Model using central bank text")]


# ------------------------------------------------------------------ data overview
def plot_overview(prices, panel):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    p = prices.loc[panel.index.min():]
    axes[0].plot(p.index, p["Close"], color=BLUE, lw=1.1)
    axes[0].set_title("GBP/USD daily close (Yahoo Finance)")
    axes[0].set_ylabel("USD per GBP")
    vol = np.sqrt(p["var_gk"].rolling(5).mean() * config.TRADING_DAYS) * 100
    axes[1].plot(vol.index, vol, color=INK2, lw=0.9, label="5-day Garman-Klass volatility")
    boe_days = panel.index[panel["boe_event"] == 1]  # BoE decision + minutes days
    for d in boe_days:
        axes[1].axvline(d, color=ORANGE, lw=0.6, alpha=0.5)
    axes[1].plot([], [], color=ORANGE, lw=1, label="BoE announcement")
    axes[1].set_ylabel("annualised vol (%)")
    axes[1].set_title("Realised volatility and Bank of England announcement days")
    axes[1].legend(loc="upper left")
    return _save(fig, "01_overview.png")


def plot_sentiment(sent):
    """Document-level FinBERT index for every set of minutes (step 5 output)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    from src.features import index_docs
    for bank, c in [("BoE", BLUE), ("Fed", ORANGE)]:
        d, t = index_docs(sent, bank)
        d = d.sort_values("date")
        name = f"{'Bank of England' if bank == 'BoE' else 'Federal Reserve'} {t}"
        if t == "speech":                       # many speeches: show a 90-day rolling mean
            d = d.set_index("date")[["fin_net", "hawk_score"]].rolling("90D").mean().reset_index()
        ms = 0 if t == "speech" else 3
        axes[0].plot(d["date"], d["fin_net"], color=c, marker="o", ms=ms, lw=1.2, label=name)
        axes[1].plot(d["date"], d["hawk_score"], color=c, marker="o", ms=ms, lw=1.2, label=name)
    axes[0].set_title("FinBERT-FOMC sentiment index per document (mean of +1 / 0 / -1 sentence scores)")
    axes[0].set_ylabel("net sentiment")
    meth = sent["hawk_method"].dropna().astype(str) if "hawk_method" in sent else pd.Series(dtype=str)
    meth = meth[meth != ""]
    meth = {"dictionary": "terminology dictionary"}.get(meth.iloc[0], meth.iloc[0]) if len(meth) else "FOMC-RoBERTa"
    axes[1].set_title(f"Extension: hawkish-dovish score ({meth})")
    axes[1].set_ylabel("P(hawkish) - P(dovish)")
    for ax in axes:
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.legend(loc="upper left")
    return _save(fig, "02_sentiment_index_documents.png")


def plot_ts_index(q):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    for c, col, name in [("boe_sent", BLUE, "Bank of England"), ("fed_sent", ORANGE, "Federal Reserve")]:
        axes[0].plot(q.index, q[c], color=col, lw=0.9, alpha=0.45)
        axes[0].plot(q.index, q[f"{c}_hp"], color=col, lw=2.0, label=f"{name} (HP trend)")
    axes[0].axhline(0, color=AXIS, lw=0.8)
    axes[0].set_title("Quarterly sentiment index, raw (thin) and HP-filtered (thick, lambda = 1600)")
    axes[0].set_ylabel("net sentiment")
    axes[0].legend(loc="upper left")
    axes[1].plot(q.index, q["rolling_corr_8q"], color=INK2, lw=1.4)
    axes[1].axhline(0, color=AXIS, lw=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("Rolling 2-year correlation between BoE and Fed sentiment")
    axes[1].set_ylabel("correlation")
    return _save(fig, "10_sentiment_index_quarterly.png")


def plot_local_projections(lp, fname="11_local_projections.png"):
    shocks = [("boe_sent_hp", "BoE sentiment (HP)"), ("fed_sent_hp", "Fed sentiment (HP)")]
    outcomes = [("gbp_vol", "GBP/USD volatility (log)"), ("gbp_ret", "GBP/USD return (%)")]
    fig, axes = plt.subplots(len(outcomes), len(shocks), figsize=(10, 6), sharex=True)
    for i, (o, oname) in enumerate(outcomes):
        for j, (sh, sname) in enumerate(shocks):
            ax = axes[i, j]
            d = lp[(lp["shock"] == sh) & (lp["outcome"] == o)]
            if d.empty:
                continue
            ax.fill_between(d["h"], d["beta"] - 1.96 * d["se"], d["beta"] + 1.96 * d["se"],
                            color=BLUE, alpha=0.15, lw=0)
            ax.plot(d["h"], d["beta"], color=BLUE, marker="o", ms=5)
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.set_title(f"{sname} -> {oname}", fontsize=9.5)
            if i == len(outcomes) - 1:
                ax.set_xlabel("quarters ahead (h)")
            if j == 0:
                ax.set_ylabel("effect of +1 s.d. sentiment")
    fig.suptitle("Local projections (Jorda, 2005), 95% White confidence bands",
                 fontsize=10.5, color=INK, y=1.01)
    return _save(fig, fname)


def plot_cross_corr(cc, fname="12_cross_correlations.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colours = {"boe_sent": BLUE, "fed_sent": ORANGE, "div_sent": AQUA}
    names = {"boe_sent": "BoE", "fed_sent": "Fed", "div_sent": "BoE - Fed"}
    for ax, (o, oname) in zip(axes, [("gbp_vol", "GBP/USD volatility"), ("gbp_ret", "GBP/USD return")]):
        for s_, c in colours.items():
            d = cc[(cc["sentiment"] == s_) & (cc["outcome"] == o)]
            ax.plot(d["lead_k"], d["corr"], color=c, marker="o", ms=4, label=names[s_])
        n = 130
        ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=GRID, alpha=0.6, lw=0)
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"Sentiment vs {oname}")
        ax.set_xlabel("months (k > 0: sentiment leads)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("correlation(sentiment_t, outcome_t+k)")
    return _save(fig, fname)


# ------------------------------------------------------------------ results
def plot_metric_bars(table, metric, title, fname, lower_is_better=True, ref=None):
    t = table[metric].sort_values(ascending=not lower_is_better)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t.values, color=[_model_colour(m) for m in t.index], height=0.62)
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lo = min(t.min(), ref if ref is not None else t.min())
    hi = max(t.max(), ref if ref is not None else t.max())
    pad = (hi - lo) * 0.25 if hi > lo else 0.1
    left = lo - pad if (metric not in ("RMSE", "MAE", "QLIKE", "LogLoss", "Brier")) else 0
    ax.set_xlim(left, hi + pad)
    for i, v in enumerate(t.values):
        ax.text(v, i, f" {v:.3f}", va="center", ha="left", color=INK2, fontsize=8.5)
    if ref is not None:
        ax.axvline(ref, color=MUTED, ls="--", lw=0.9)
        ax.text(ref, len(t) - 0.4, f"  {ref:g}", color=MUTED, fontsize=8, va="bottom")
    ax.set_title(title, pad=26)
    ax.set_xlabel(metric + ("  (lower is better)" if lower_is_better else "  (higher is better)"))
    ax.legend(handles=_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=8)          # above the bars, so no value label is covered
    return _save(fig, fname)


def plot_vol_forecasts(preds, models, fname="05_vol_forecasts.png"):
    last_year = preds.index.year.max()
    p = preds[preds.index.year == last_year]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(p.index, np.exp(p["y_true"]) * 100, color=MUTED, lw=0.8, label="Actual next-day volatility")
    colours = [INK2, BLUE, ORANGE]
    for m, c in zip(models, colours):
        ax.plot(p.index, np.exp(p[m]) * 100, color=c, lw=1.3, label=m)
    ax.set_title(f"Next-day GBP/USD volatility forecasts, test year {last_year}")
    ax.set_ylabel("annualised vol (%)")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, fname)


def plot_event_study(es_by_bank, fname="07_event_study_daily.png"):
    banks = [b for b, es in es_by_bank.items() if es]
    if not banks:
        return None
    fig, axes = plt.subplots(1, len(banks), figsize=(5.2 * len(banks), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    style = {"more positive": BLUE, "more negative": ORANGE, "neutral": MUTED}
    for ax, bank in zip(axes, banks):
        for g, df in es_by_bank[bank].items():
            c = style[g]
            ax.plot(df.index, df["mean_bp"], color=c, marker="o", ms=4,
                    label=f"{g} (n={int(df['n'].iloc[0])})")
            if g != "neutral":
                ax.fill_between(df.index, df["mean_bp"] - 1.96 * df["se_bp"],
                                df["mean_bp"] + 1.96 * df["se_bp"], color=c, alpha=0.12, lw=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{bank} minutes releases")
        ax.set_xlabel("trading days after release (0 = release day)")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("cumulative GBP/USD return (bp)")
    fig.suptitle("Does GBP/USD keep drifting after more positive / negative minutes?  "
                 "(mean, 95% CI)", fontsize=10.5, color=INK, y=1.02)
    return _save(fig, fname)


def plot_intraday(es_by_bank, fname="08_event_study_intraday.png"):
    banks = [b for b, es in es_by_bank.items() if es]
    if not banks:
        return None
    fig, axes = plt.subplots(1, len(banks), figsize=(5.2 * len(banks), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bank in zip(axes, banks):
        for (g, df), c in zip(es_by_bank[bank].items(), [BLUE, ORANGE]):
            ax.plot(df.index, df["mean_bp"], color=c, label=f"{g} (n={int(df['n'].iloc[0])})")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{bank} minutes: hourly path (last ~2 years)")
        ax.set_xlabel("hours after release")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cumulative GBP/USD return (bp)")
    return _save(fig, fname)


def plot_importance(imp, title, fname, top=15):
    s = imp.sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
    colours = [ORANGE if c.startswith(("boe_", "fed_", "div_")) and not c.endswith(
        ("_event", "_event_ahead", "_days_since")) else BLUE for c in s.index]
    ax.barh(np.arange(len(s)), s.values, color=colours, height=0.62)
    ax.set_yticks(np.arange(len(s)), s.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel("permutation importance (increase in loss when shuffled)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Price / market / calendar feature"),
                       Patch(color=ORANGE, label="Central bank text feature")],
              loc="lower right", fontsize=8)
    return _save(fig, fname)


def plot_period_comparison(core_periods, core_models, fname="05b_core_by_period.png"):
    """MAE of the three core experiments in each regime (pre-COVID / COVID / post-COVID)."""
    if core_periods is None or core_periods.empty:
        return None
    d = core_periods.reset_index()
    periods = list(dict.fromkeys(d["period"]))
    colours = [BASELINE_GREY, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.26
    x = np.arange(len(periods))
    for i, (m, c) in enumerate(zip(core_models, colours)):
        vals = [d[(d["period"] == p) & (d["model"] == m)]["MAE"].mean() for p in periods]
        ax.bar(x + (i - 1) * width, vals, width=width - 0.03, color=c, label=m)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    labels = [f"{p}\n(n={int(d[d['period'] == p]['n_days'].iloc[0])} days)" for p in periods]
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("MAE on log volatility (lower is better)")
    ax.set_ylim(0, np.nanmax(d["MAE"].values) * 1.18)
    ax.set_title("Core experiments by regime: does central bank sentiment help more post-COVID?",
                 pad=30)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return _save(fig, fname)
