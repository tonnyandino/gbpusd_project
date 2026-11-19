"""
Step 5b - walk-forward evaluation, metrics, significance tests and the event study.

Chronological walk-forward test (expanding window): for every test year Y >=
FIRST_TEST_YEAR, models are trained on all data before 1 January Y and tested on year Y,
so a model never sees the future. Because targets look ahead, the last few training rows
overlap the test period and are purged from the training set (Lopez de Prado, 2018).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, mean_absolute_error, mean_squared_error,
                             roc_auc_score)

import config

H = config.HORIZON
PURGE = max(config.HORIZON, config.DIR_HORIZON)


# ------------------------------------------------------------------ walk-forward
def walk_forward(panel, model_factory, target, tuning_log=None):
    """Return a DataFrame of out-of-sample predictions (rows = test days, cols = models)
    plus the list of models fitted on the final fold (these are saved to disk).
    If `tuning_log` is a list, the hyper-parameters chosen in every fold are appended."""
    data = panel.dropna(subset=[target])
    years = sorted(y for y in data.index.year.unique() if y >= config.FIRST_TEST_YEAR)
    preds, last_models = [], None
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        train = data[data.index < start].iloc[:-PURGE]      # purge overlapping targets
        test = data[(data.index >= start) & (data.index < pd.Timestamp(f"{y + 1}-01-01"))]
        if len(test) == 0 or len(train) < 250:
            continue
        models = model_factory()
        fold = pd.DataFrame(index=test.index)
        fold["y_true"] = test[target]
        for m in models:
            m.fit(train)
            fold[m.name] = m.predict(test)
            if tuning_log is not None:
                tuning_log.extend(tuning_rows(m, y, target))
        preds.append(fold)
        last_models = models
        print(f"   fold {y}: train {len(train):5d} rows | test {len(test):4d} rows")
    return pd.concat(preds), last_models


def tuning_rows(model, year, target):
    """Hyper-parameters chosen for one model in one walk-forward fold."""
    rows = []
    if getattr(model, "grid_scores", None):
        best = min(sc for _, sc in model.grid_scores)
        for params, score in model.grid_scores:
            rows.append({"task": target, "test_year": year, "model": model.name,
                         "params": ", ".join(f"{k}={v}" for k, v in params.items()),
                         "validation_loss": score, "chosen": score == best})
    elif hasattr(model, "params") and model.name.startswith("GARCH"):
        mu, omega, alpha, beta = model.params
        rows.append({"task": target, "test_year": year, "model": model.name,
                     "params": f"omega={omega:.4f}, alpha={alpha:.3f}, beta={beta:.3f}, "
                               f"persistence={alpha + beta:.3f}",
                     "validation_loss": np.nan, "chosen": True})
    return rows


def tuning_summary(log):
    """How often each setting was chosen per model across the folds."""
    df = pd.DataFrame(log)
    if df.empty:
        return df
    ch = df[df["chosen"]]
    return (ch.groupby(["task", "model", "params"]).size().rename("folds_chosen")
              .reset_index().sort_values(["task", "model", "folds_chosen"], ascending=[True, True, False]))


# ------------------------------------------------------------------ regression metrics
def qlike(true_logvol, pred_logvol):
    """QLIKE loss on variances (Patton, 2011): robust to noise in the volatility proxy."""
    s2 = np.exp(2 * true_logvol)
    h = np.exp(2 * pred_logvol)
    return np.mean(s2 / h - np.log(s2 / h) - 1)


def diebold_mariano(e1, e2, h=max(H, 1)):
    """DM test of equal predictive accuracy on squared errors, with Newey-West variance
    (h-1 lags, overlapping forecasts) and the Harvey-Leybourne-Newbold correction.
    Negative statistic = model 1 is more accurate."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


NAIVE = "Previous-day volatility (naive)"


def regression_table(preds, benchmark=NAIVE, naive=NAIVE):
    y = preds["y_true"].values
    rows = []
    for m in preds.columns.drop("y_true"):
        p = preds[m].values
        e = y - p
        dm, pval = (diebold_mariano(e, y - preds[benchmark].values)
                    if m != benchmark else (np.nan, np.nan))
        rows.append({
            "model": m,
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "MAE": mean_absolute_error(y, p),
            "QLIKE": qlike(y, p),
            "R2_oos_vs_naive": 1 - np.sum(e ** 2) / np.sum((y - preds[naive].values) ** 2),
            "MAE_vol_pts": np.mean(np.abs(np.exp(y) - np.exp(p))) * 100,   # in vol % points
            f"DM_vs_{benchmark}": dm,
            "DM_p": pval,
        })
    return pd.DataFrame(rows).set_index("model")


def core_table(preds, core_models):
    """The lecturer's experiment table: MAE and RMSE on the chronological test set, plus a
    Diebold-Mariano test of 'price + sentiment' against 'price only'."""
    y = preds["y_true"].values
    rows = []
    for m in core_models:
        p = preds[m].values
        rows.append({"model": m, "MAE": mean_absolute_error(y, p),
                     "RMSE": np.sqrt(mean_squared_error(y, p)),
                     "MAE_vol_pts": np.mean(np.abs(np.exp(y) - np.exp(p))) * 100})
    t = pd.DataFrame(rows).set_index("model")
    naive = core_models[0]
    t["MAE_improvement_vs_naive_%"] = 100 * (1 - t["MAE"] / t.loc[naive, "MAE"])
    t["RMSE_improvement_vs_naive_%"] = 100 * (1 - t["RMSE"] / t.loc[naive, "RMSE"])
    if len(core_models) >= 3:
        dm, p = diebold_mariano(y - preds[core_models[2]].values, y - preds[core_models[1]].values)
        t["DM_sent_vs_price"] = np.nan
        t["DM_p"] = np.nan
        t.loc[core_models[2], ["DM_sent_vs_price", "DM_p"]] = [dm, p]
    return t


# ------------------------------------------------------------------ classification metrics
def classification_table(preds):
    y = preds["y_true"].astype(int).values
    rows = []
    every_h = np.arange(0, len(y), config.DIR_HORIZON)   # non-overlapping windows
    for m in preds.columns.drop("y_true"):
        p = preds[m].values
        label = (p >= 0.5).astype(int)
        # the coin flip produces hard 0/1 guesses; for probability metrics use p = 0.5
        p_prob = np.full_like(p, 0.5) if m == "Coin flip" else np.clip(p, 1e-6, 1 - 1e-6)
        auc = roc_auc_score(y, p_prob) if len(np.unique(p_prob)) > 1 else 0.5
        acc_no = accuracy_score(y[every_h], label[every_h])
        rows.append({
            "model": m,
            "Accuracy": accuracy_score(y, label),
            "Balanced_acc": balanced_accuracy_score(y, label),
            "F1_up": f1_score(y, label, zero_division=0),
            "ROC_AUC": auc,
            "LogLoss": log_loss(y, p_prob, labels=[0, 1]),
            "Brier": brier_score_loss(y, p_prob),
            "Acc_nonoverlap": acc_no,
            "Binom_p_nonoverlap": stats.binomtest(int(round(acc_no * len(every_h))),
                                                  len(every_h), 0.5,
                                                  alternative="greater").pvalue,
        })
    return pd.DataFrame(rows).set_index("model")


def by_period(preds, table_fn, *args):
    """Apply a metric table to each sub-period in config.PERIODS that has test data."""
    out = []
    for name, (a, b) in config.PERIODS.items():
        sub = preds.loc[(preds.index >= pd.Timestamp(a)) & (preds.index <= pd.Timestamp(b))]
        if len(sub) < 30:
            continue
        t = table_fn(sub, *args)
        t.insert(0, "n_days", len(sub))
        t.insert(0, "period", name)
        out.append(t)
    return pd.concat(out) if out else pd.DataFrame()


def subset(preds, mask_index):
    return preds.loc[preds.index.isin(mask_index)]


# ------------------------------------------------------------------ event study
def event_study(prices, panel, bank, window=config.EVENT_WINDOW, score="fin_chg"):
    """Average cumulative GBP/USD log return from the minutes release day to +window days,
    split into positive vs negative sentiment surprises (change in the FinBERT index vs
    the previous minutes). Tests whether the market keeps drifting after the news."""
    prefix = bank.lower()
    ev = panel[panel[f"{prefix}_text_event"] == 1]
    chg = ev[f"{prefix}_{score}"].dropna()
    if len(chg) < 6:
        return None
    hi, lo = chg.quantile(2 / 3), chg.quantile(1 / 3)
    groups = {"more positive": chg[chg >= hi].index,
              "more negative": chg[chg <= lo].index,
              "neutral": chg[(chg > lo) & (chg < hi)].index}
    logp = np.log(prices["Close"])
    idx = prices.index
    out = {}
    for g, dates in groups.items():
        paths = []
        for d in dates:
            i = idx.get_loc(d)
            if i - 1 < 0 or i + window >= len(idx):
                continue
            base = logp.iloc[i - 1]                       # close the day before
            paths.append((logp.iloc[i - 1:i + window + 1].values - base) * 1e4)   # pips-ish (bp)
        if paths:
            arr = np.vstack(paths)
            out[g] = pd.DataFrame({"mean_bp": arr.mean(0),
                                   "se_bp": arr.std(0, ddof=1) / np.sqrt(len(arr)),
                                   "n": len(arr)},
                                  index=range(-1, window + 1))
    return out


def event_study_table(es, bank):
    rows = []
    if not es:
        return pd.DataFrame()
    for g, df in es.items():
        for day in [0, 1, 3, 5, config.EVENT_WINDOW]:
            if day in df.index:
                m, se = df.loc[day, "mean_bp"], df.loc[day, "se_bp"]
                rows.append({"bank": bank, "group": g, "day": day, "n": int(df.loc[day, "n"]),
                             "cum_return_bp": m, "t_stat": m / se if se > 0 else np.nan})
    return pd.DataFrame(rows)


def intraday_event_study(hourly, sent, bank, hours_after=72):
    """Hourly GBP/USD path around announcements over the ~2 years Yahoo provides.
    Release times: BoE minutes 12:00 UK, FOMC minutes 14:00 US Eastern."""
    if hourly is None or hourly.empty:
        return None
    from src.features import index_docs
    s, doc_type = index_docs(sent, bank)
    s = s.groupby("date", as_index=False)["fin_net"].mean().sort_values("date")
    s["chg"] = s["fin_net"].diff()
    s = s[(s["date"] >= hourly.index.min()) & (s["date"] <= hourly.index.max())]
    s = s.dropna(subset=["chg"])
    if len(s) < 4:
        return None
    if doc_type != "minutes":
        return None                     # speeches have no fixed release time
    if bank == "BoE":
        ts = (pd.to_datetime(s["date"]) + pd.Timedelta(hours=12)).dt.tz_localize(
            "Europe/London", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        ts = (pd.to_datetime(s["date"]) + pd.Timedelta(hours=14)).dt.tz_localize(
            "America/New_York", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
    s["ts"] = ts.dt.tz_localize(None)
    logp = np.log(hourly["Close"])
    med = s["chg"].median()
    out = {}
    for g, sub in [("more positive", s[s["chg"] > med]), ("more negative", s[s["chg"] <= med])]:
        paths = []
        for t in sub["ts"].dropna():
            pos = logp.index.searchsorted(t)
            if pos < 1 or pos + hours_after >= len(logp):
                continue
            base = logp.iloc[pos - 1]
            paths.append((logp.iloc[pos - 1:pos + hours_after].values - base) * 1e4)
        if paths:
            L = min(len(p) for p in paths)
            arr = np.vstack([p[:L] for p in paths])
            out[g] = pd.DataFrame({"mean_bp": arr.mean(0), "n": len(arr)},
                                  index=np.arange(-1, L - 1))
    return out or None
