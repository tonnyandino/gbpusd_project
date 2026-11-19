"""
Key findings - turns the result tables in results/ into a plain-English summary.

The text is generated from the numbers of the latest run (nothing is hard-coded), so it
stays correct after every re-run. Saved to results/key_findings.txt.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config

R = config.RESULTS_DIR
NAIVE, RF_P, RF_S = "Previous-day volatility (naive)", "RF: price only", "RF: price + sentiment"


def _csv(name, **kw):
    path = R / name
    return pd.read_csv(path, **kw) if path.exists() else None


def _p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _cap(text):
    return text[:1].upper() + text[1:]


def _sig(p, level=0.05):
    if p is None or pd.isna(p):
        return "not tested"
    return f"significant ({_p(p)})" if p < level else f"not significant ({_p(p)})"


def _corr_p(r, n):
    if pd.isna(r) or n < 4 or abs(r) >= 1:
        return np.nan
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(abs(t), n - 2)


def _dm_sentence(dm, p):
    """DM statistic: negative = sentiment model more accurate."""
    if pd.isna(dm):
        return "no Diebold-Mariano test available"
    if p >= 0.05:
        return f"the difference is not statistically significant (DM = {dm:+.2f}, {_p(p)})"
    better = "better" if dm < 0 else "worse"
    return f"the sentiment model is significantly {better} (DM = {dm:+.2f}, {_p(p)})"


# ====================================================================== sections
def data_section():
    p = R / "data_summary.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    docs = ", ".join(f"{k.replace('documents_', '').replace('_', ' ')} {v}"
                     for k, v in d.items() if k.startswith("documents_") and k != "documents_total")
    out = ["DATA",
           f"- {d.get('gbpusd_daily_rows', '?'):,} daily GBP/USD prices, {d.get('cross_market_tickers', '?')} "
           f"related markets and {d.get('gbpusd_hourly_rows', 0):,} hourly prices (Yahoo Finance).",
           f"- {d.get('documents_total', '?')} central bank documents ({docs}), "
           f"{d.get('sentences_scored', 0):,} sentences scored with FinBERT-FOMC.",
           f"- Chronological test period: {d.get('test_period', '?')}."]
    if d.get("demo_mode"):
        out.insert(1, "- !!! DEMO MODE: synthetic data - do not report these numbers !!!")
    return out


def core_section():
    core = _csv("core_experiments.csv", index_col=0)
    if core is None or not {NAIVE, RF_P, RF_S} <= set(core.index):
        return []
    imp = core.loc[RF_P, "MAE_improvement_vs_naive_%"]
    sent_change = 100 * (core.loc[RF_S, "MAE"] / core.loc[RF_P, "MAE"] - 1)
    out = ["1. DO THE AI MODELS BEAT THE TRIVIAL BASELINE?",
           f"- Random forest (price only): MAE {core.loc[RF_P, 'MAE']:.3f} vs {core.loc[NAIVE, 'MAE']:.3f} for "
           f"'tomorrow = today' -> {imp:.1f}% {'lower' if imp > 0 else 'higher'} error."]
    allm = _csv("vol_metrics_all_models.csv", index_col=0)
    if allm is not None and RF_P in allm.index:
        dmcol = [c for c in allm.columns if c.startswith("DM_vs")]
        if dmcol:
            out.append(f"- Versus the naive baseline: {_sig(allm.loc[RF_P, 'DM_p'])} "
                       "(Diebold-Mariano test).")
    out += ["", "2. DOES CENTRAL BANK SENTIMENT IMPROVE THE VOLATILITY FORECAST?",
            f"- RF + sentiment: MAE {core.loc[RF_S, 'MAE']:.4f} vs {core.loc[RF_P, 'MAE']:.4f} without "
            f"sentiment ({sent_change:+.2f}%, {'worse' if sent_change > 0 else 'better'}).",
            f"- {_cap(_dm_sentence(core.loc[RF_S, 'DM_sent_vs_price'], core.loc[RF_S, 'DM_p']))}."]
    ev = _csv("core_experiments_announcement_days.csv", index_col=0)
    if ev is not None and RF_S in ev.index:
        ch = 100 * (ev.loc[RF_S, "MAE"] / ev.loc[RF_P, "MAE"] - 1)
        out.append(f"- Announcement days only: {ch:+.2f}% MAE; "
                   f"{_dm_sentence(ev.loc[RF_S, 'DM_sent_vs_price'], ev.loc[RF_S, 'DM_p'])}.")
    return out


def regime_section():
    per = _csv("core_experiments_by_period.csv")
    out = []
    if per is not None and "period" in per:
        per = per.rename(columns={per.columns[0]: "model"})
        out = ["", "3. PRE- VS POST-COVID: DOES SENTIMENT HELP MORE AFTER COVID?"]
        for period in dict.fromkeys(per["period"]):
            p = per[per["period"] == period].set_index("model")
            if RF_S not in p.index:
                continue
            ch = 100 * (p.loc[RF_S, "MAE"] / p.loc[RF_P, "MAE"] - 1)
            out.append(f"- {period} ({int(p.loc[RF_S, 'n_days'])} test days): sentiment changes MAE by "
                       f"{ch:+.2f}%; {_dm_sentence(p.loc[RF_S, 'DM_sent_vs_price'], p.loc[RF_S, 'DM_p'])}.")
    reg = _csv("ts_regime_comparison.csv")
    if reg is not None and not reg.empty:
        out.append("- Relationship between sentiment CHANGES and GBP/USD (monthly correlation):")
        for (s_, o), g in reg.groupby(["sentiment", "outcome"], sort=False):
            parts = []
            for _, r in g.iterrows():
                pv = _corr_p(r["corr_sent_change_vs_outcome"], int(r["months"]))
                parts.append(f"{r['period']} {r['corr_sent_change_vs_outcome']:+.2f} "
                             f"(n={int(r['months'])}, p={pv:.2f})")
            out.append(f"    {s_} vs {o}: " + " | ".join(parts))
        pre = reg[reg["period"] == "pre-COVID"].set_index(["sentiment", "outcome"])
        post = reg[reg["period"] == "post-COVID"].set_index(["sentiment", "outcome"])
        both = pre.index.intersection(post.index)
        if len(both):
            diff = (pre.loc[both, "corr_sent_change_vs_outcome"]
                    - post.loc[both, "corr_sent_change_vs_outcome"]).abs()
            k = diff.idxmax()
            out.append(f"- Largest change after COVID: {k[0]} vs {k[1]} "
                       f"({pre.loc[k, 'corr_sent_change_vs_outcome']:+.2f} before, "
                       f"{post.loc[k, 'corr_sent_change_vs_outcome']:+.2f} after) - evidence that the "
                       "market reacted differently after COVID (treat as suggestive: small samples).")
    return out


def benchmark_section():
    allm = _csv("vol_metrics_all_models.csv", index_col=0)
    if allm is None:
        return []
    best = allm["MAE"].idxmin()
    out = ["", "4. WHICH MODEL WAS BEST OVERALL?",
           f"- Lowest error: {best} (MAE {allm.loc[best, 'MAE']:.4f})."]
    if best == "HAR-RV":
        out.append("- A simple econometric model (HAR-RV, Corsi 2009) beat the machine learning models - "
                   "a common finding in volatility forecasting and a key point for critical evaluation.")
    rank = allm["MAE"].sort_values()
    out.append("- Ranking (MAE): " + ", ".join(f"{m} {v:.3f}" for m, v in rank.items()))
    return out


def _dm_pair(a, b):
    """Diebold-Mariano test of model a vs model b on the saved test predictions
    (negative statistic = a more accurate). Returns (stat, p) or (nan, nan)."""
    preds = _csv("vol_predictions.csv", index_col=0)
    if preds is None or a not in preds or b not in preds:
        return np.nan, np.nan
    from src.evaluation import diebold_mariano
    y = preds["y_true"].values
    return diebold_mariano(y - preds[a].values, y - preds[b].values)


def _vs(a, b):
    stat, p = _dm_pair(a, b)
    if pd.isna(p):
        return "not tested"
    if p >= 0.05:
        return f"no significant difference (DM = {stat:+.2f}, {_p(p)})"
    return f"significantly {'better' if stat < 0 else 'worse'} (DM = {stat:+.2f}, {_p(p)})"


def hybrid_section():
    allm = _csv("vol_metrics_all_models.csv", index_col=0)
    if allm is None:
        return []
    names = [m for m in allm.index if m.startswith("Hybrid") or "key sentiment" in m]
    if not names:
        return []
    out = ["", "5. MODIFICATIONS & HYBRID MODELS"]
    ref = {m: allm.loc[m, "MAE"] for m in ("HAR-RV", "RF: price only", "RF: price + sentiment")
           if m in allm.index}
    for m in names:
        cmp = ", ".join(f"{100 * (allm.loc[m, 'MAE'] / v - 1):+.1f}% vs {k}" for k, v in ref.items())
        out.append(f"- {m}: MAE {allm.loc[m, 'MAE']:.4f} ({cmp}).")
    if "RF: price + key sentiment" in allm.index and "RF: price + sentiment" in allm.index:
        out.append("- RF + key sentiment (9 features) vs RF + full sentiment (~29): "
                   + _vs("RF: price + key sentiment", "RF: price + sentiment") + ".")
        out.append("- RF + key sentiment vs RF price only: "
                   + _vs("RF: price + key sentiment", "RF: price only") + ".")
    for m in [n for n in names if n.startswith("Hybrid")]:
        if "HAR-RV" in allm.index:
            out.append(f"- {m} vs HAR-RV alone: {_vs(m, 'HAR-RV')}.")
    return out


def tuning_section():
    t = _csv("hyperparameters_summary.csv")
    if t is None or t.empty:
        return []
    out = ["", "HYPER-PARAMETERS (chosen on the last year of each training window)"]
    for (task, model), g in t.groupby(["task", "model"], sort=False):
        if task != "y_vol" or not (model.startswith("RF") or model.startswith("Hybrid")
                                   or model.startswith("GBM")):
            continue
        top = g.iloc[0]
        out.append(f"- {model}: most often {top['params']} ({int(top['folds_chosen'])} of "
                   f"{int(g['folds_chosen'].sum())} folds)")
    return out


def direction_section():
    d = _csv("dir_metrics_all_days.csv", index_col=0)
    if d is None:
        return []
    feats = d.drop(index=[m for m in ("Coin flip", "Most frequent label", "Previous direction (momentum)",
                                      "Sentiment-surprise rule") if m in d.index])
    best = feats["ROC_AUC"].idxmax()
    out = ["", f"6. EXTENSION - DIRECTION OF GBP/USD OVER {config.DIR_HORIZON} DAYS",
           f"- Best model: {best}: accuracy {d.loc[best, 'Accuracy']:.1%}, AUC {d.loc[best, 'ROC_AUC']:.3f} "
           f"(coin flip {d.loc['Coin flip', 'Accuracy']:.1%}, AUC 0.500).",
           f"- On non-overlapping windows: accuracy {d.loc[best, 'Acc_nonoverlap']:.1%}, "
           f"{_sig(d.loc[best, 'Binom_p_nonoverlap'])} vs 50%."]
    for fam in ("RF", "GBM", "Logistic"):
        a, b = f"{fam}: price only", f"{fam}: price + sentiment"
        if a in d.index and b in d.index:
            out.append(f"- {fam}: AUC {d.loc[a, 'ROC_AUC']:.3f} without sentiment vs "
                       f"{d.loc[b, 'ROC_AUC']:.3f} with sentiment.")
    per = _csv("dir_metrics_by_period.csv")
    if per is not None and "period" in per:
        per = per.rename(columns={per.columns[0]: "model"})
        for period in dict.fromkeys(per["period"]):
            p = per[per["period"] == period].set_index("model")
            if {RF_P, RF_S} <= set(p.index):
                out.append(f"- {period}: RF accuracy {p.loc[RF_P, 'Accuracy']:.1%} -> "
                           f"{p.loc[RF_S, 'Accuracy']:.1%} with sentiment "
                           f"(AUC {p.loc[RF_P, 'ROC_AUC']:.3f} -> {p.loc[RF_S, 'ROC_AUC']:.3f}).")
    return out


def timeseries_section():
    out = ["", "7. TIME-SERIES ANALYSIS (REPRODUCING HILSCHER ET AL., 2026)"]
    co = _csv("ts_comovement.csv")
    if co is not None:
        v = co.set_index("measure")["value"]
        out.append(f"- BoE and Fed sentiment move together: correlation {v.get('corr_raw_quarterly', np.nan):.2f} "
                   f"(quarterly), {v.get('corr_hp_quarterly', np.nan):.2f} after HP smoothing - compare with "
                   "the paper's Fed-ECB co-movement.")
    gr = _csv("ts_granger.csv")
    if gr is not None:
        sig = gr[gr["p"] < 0.05]
        if sig.empty:
            out.append("- Granger causality: no significant relationships at 5% in either direction.")
        else:
            for _, r in sig.iterrows():
                out.append(f"- Granger causality: {r['cause']} -> {r['effect']} ({int(r['lags'])} lags, "
                           f"{_p(r['p'])}).")
            market_first = sig["cause"].isin(["gbp_ret", "gbp_vol"]).all()
            if market_first:
                out.append("  Direction: market moves help predict LATER central bank sentiment, not the "
                           "reverse - central bank language reacts to markets.")
    return out


def event_section():
    es = _csv("event_study_daily.csv")
    if es is None or es.empty:
        return []
    out = ["", "8. EVENT STUDY - GBP/USD AFTER MINUTES RELEASES"]
    sig = es[(es["t_stat"].abs() >= 2) & (es["day"] > 0)]
    if sig.empty:
        out.append("- No cumulative move is statistically significant (|t| >= 2).")
    for _, r in sig.iterrows():
        out.append(f"- {r['bank']} '{r['group']}' minutes: {r['cum_return_bp']:+.1f} bp after {int(r['day'])} "
                   f"days (t = {r['t_stat']:+.2f}, n = {int(r['n'])}).")
    return out


def limitations_section():
    out = ["", "LIMITATIONS"]
    sent = config.SENTIMENT_CSV
    if sent.exists():
        s = pd.read_csv(sent)
        m = s.get("hawk_method")
        if m is not None and (m.astype(str) == "dictionary").any():
            out.append("- Hawkish/dovish score from the terminology dictionary (FOMC-RoBERTa is gated "
                       "on Hugging Face).")
        n_min = s[s["doc_type"] == "minutes"].groupby("bank").size()
        out.append("- Minutes per bank: " + ", ".join(f"{b} {n}" for b, n in n_min.items())
                   + " - few events relative to the number of sentiment features (overfitting risk).")
    out.append("- Volatility is measured with the Garman-Klass range estimator on daily data "
               "(no intraday realised volatility).")
    return out


def generate(print_it=True):
    lines = ["KEY FINDINGS (generated from the latest results)", "=" * 60]
    for section in (data_section, core_section, regime_section, benchmark_section, hybrid_section,
                    direction_section, timeseries_section, event_section, tuning_section,
                    limitations_section):
        try:
            part = section()
        except Exception as exc:                      # never let one table break the summary
            part = [f"[{section.__name__} skipped: {type(exc).__name__}: {exc}]"]
        while part and part[0] == "":
            part = part[1:]
        if part:
            lines += part + [""]
    text = "\n".join(lines)
    R.mkdir(parents=True, exist_ok=True)
    (R / "key_findings.txt").write_text(text, encoding="utf-8")
    if print_it:
        print("\n" + text)
        print(f"Saved to {(R / 'key_findings.txt').relative_to(config.ROOT)}")
    return text
