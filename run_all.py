"""
Reproducing and extending a FinBERT central bank sentiment index (Hilscher, Nabors &
Raviv, 2026) to forecast next-day GBP/USD volatility.

    python run_all.py                  # full run on real data (Yahoo Finance, BoE, Fed)
    python run_all.py --demo           # quick offline check with synthetic data (numbers meaningless)
    python run_all.py --from train     # re-use earlier outputs, start at a later step

Steps
  1 prices      Collecting financial market data from Yahoo Finance
  2 texts       Collecting Federal Reserve and Bank of England communications
  3 sentences   Splitting documents into sentences using NLTK
  4-5 sentiment FinBERT-FOMC sentence classification (positive / neutral / negative),
                aggregated into document-level sentiment indices
  6 features    Building the daily modelling panel
  7 timeseries  Time-series analysis of sentiment vs financial variables
  8 train       Forecasting experiments and evaluation
Outputs: data/ (datasets), models/ (trained models), results/ (tables + figures)
"""
import argparse
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import config

sys.path.insert(0, str(config.ROOT))
from src import (data_prices, data_texts, evaluation, features, models, plots,  # noqa: E402
                 sentiment, timeseries)

STEPS = ["prices", "texts", "sentences", "sentiment", "features", "timeseries", "train"]
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _fmt(df, digits=4):
    return df.to_string(float_format=lambda v: f"{v:.{digits}f}")


def permutation_importance_table(model, test, task):
    from sklearn.inspection import permutation_importance
    test = test.dropna(subset=[model.target])
    X, y = test[model.features], test[model.target]
    scoring = "neg_mean_squared_error" if task == "reg" else "neg_log_loss"
    if task == "clf":
        y = y.astype(int)
    res = permutation_importance(model.est, X, y, scoring=scoring, n_repeats=5,
                                 random_state=config.RANDOM_SEED, n_jobs=-1)
    return pd.Series(res.importances_mean, index=model.features)


def time_series_step():
    res = timeseries.run_analysis(data_prices.load_prices(), data_prices.load_market(),
                                  sentiment.load_sentiment())
    plots.plot_ts_index(res["quarterly"])
    plots.plot_local_projections(res["local_projections"])
    plots.plot_cross_corr(res["cross_corr"])
    return res


def train_and_evaluate(demo):
    print("=== Step 8: Forecasting experiments (chronological walk-forward test) ===")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prices = data_prices.load_prices()
    panel = features.load_panel()
    sent = sentiment.load_sentiment()
    groups = features.feature_groups(panel)

    report = []
    if demo:
        report.append("!!! DEMO MODE - SYNTHETIC DATA - DO NOT REPORT THESE NUMBERS !!!\n")

    # ---------------------------------------------------------- Task A: next-day volatility
    print("\n[Task A] next-day volatility (regression)")
    tuning_log = []
    vol_preds, vol_models = evaluation.walk_forward(
        panel, lambda: models.volatility_models(groups, prices["ret"]), "y_vol", tuning_log)
    core = evaluation.core_table(vol_preds, models.CORE_VOL)
    vol_tab = evaluation.regression_table(vol_preds)
    ann_days = panel.index[(panel["boe_event"] == 1) | (panel["fed_event"] == 1)
                           | (panel["fed_text_event"] == 1)]
    vol_ev = evaluation.subset(vol_preds, ann_days)
    vol_tab_ev = evaluation.regression_table(vol_ev)
    core_ev = evaluation.core_table(vol_ev, models.CORE_VOL)
    vol_preds.to_csv(config.RESULTS_DIR / "vol_predictions.csv")
    core.to_csv(config.RESULTS_DIR / "core_experiments.csv")
    core_ev.to_csv(config.RESULTS_DIR / "core_experiments_announcement_days.csv")
    core_periods = evaluation.by_period(vol_preds, evaluation.core_table, models.CORE_VOL)
    core_periods.to_csv(config.RESULTS_DIR / "core_experiments_by_period.csv")
    vol_tab.to_csv(config.RESULTS_DIR / "vol_metrics_all_models.csv")
    vol_tab_ev.to_csv(config.RESULTS_DIR / "vol_metrics_announcement_days.csv")
    period = f"{vol_preds.index.min().date()} -> {vol_preds.index.max().date()}"
    report += [
        "CORE EXPERIMENTS - next-day GBP/USD volatility (target: log annualised vol)",
        f"Chronological test set {period}, {len(vol_preds)} trading days",
        "MAE / RMSE on log volatility (lower is better); MAE_vol_pts = MAE in volatility "
        "percentage points; DM = Diebold-Mariano test, price + sentiment vs price only "
        "(negative = sentiment model more accurate)",
        _fmt(core), "",
        f"Same, central bank release / announcement days only (n={len(vol_ev)})",
        _fmt(core_ev), "",
        "REGIME COMPARISON - core experiments by period (does sentiment help more post-COVID?)",
        _fmt(core_periods), "",
        "ALL VOLATILITY MODELS incl. hybrids (DM test vs the naive previous-day baseline)",
        _fmt(vol_tab), "",
    ]

    # ---------------------------------------------------------- Task B: direction (extension)
    print(f"\n[Task B] {config.DIR_HORIZON}-day direction (classification, extension)")
    dir_preds, dir_models = evaluation.walk_forward(
        panel, lambda: models.direction_models(groups), "y_dir", tuning_log)
    tuning = pd.DataFrame(tuning_log)
    tuning.to_csv(config.RESULTS_DIR / "hyperparameters_all_folds.csv", index=False)
    tuning_sum = evaluation.tuning_summary(tuning_log)
    tuning_sum.to_csv(config.RESULTS_DIR / "hyperparameters_summary.csv", index=False)
    dir_tab = evaluation.classification_table(dir_preds)
    dir_preds.to_csv(config.RESULTS_DIR / "dir_predictions.csv")
    dir_tab.to_csv(config.RESULTS_DIR / "dir_metrics_all_days.csv")
    dir_periods = evaluation.by_period(dir_preds, evaluation.classification_table)
    dir_periods.to_csv(config.RESULTS_DIR / "dir_metrics_by_period.csv")
    report += [f"EXTENSION - GBP/USD direction over the next {config.DIR_HORIZON} days "
               "(classification)",
               f"Share of 'up' windows in test period: {dir_preds['y_true'].mean():.3f}",
               _fmt(dir_tab), "", "Direction accuracy by period",
               _fmt(dir_periods[["period", "n_days", "Accuracy", "ROC_AUC"]]), ""]

    # ---------------------------------------------------------- event studies
    print("\n[event study] daily and intraday")
    es = {b: evaluation.event_study(prices, panel, b) for b in ["BoE", "Fed"]}
    es_tab = pd.concat([evaluation.event_study_table(es[b], b) for b in es], ignore_index=True)
    es_tab.to_csv(config.RESULTS_DIR / "event_study_daily.csv", index=False)
    hourly = data_prices.load_hourly()
    es_intra = {b: evaluation.intraday_event_study(hourly, sent, b) for b in ["BoE", "Fed"]}
    report += ["EVENT STUDY - cumulative GBP/USD return (bp) after minutes releases, grouped by",
               "the change in the FinBERT index vs the previous minutes (terciles)",
               _fmt(es_tab, 2) if not es_tab.empty else "(not enough events)", ""]

    # ---------------------------------------------------------- feature importance
    last_year = vol_preds.index.year.max()
    test_last = panel[panel.index.year == last_year]
    imp_vol = permutation_importance_table(
        [m for m in vol_models if m.name == "RF: price + sentiment"][0], test_last, "reg")
    imp_dir = permutation_importance_table(
        [m for m in dir_models if m.name == "GBM: price + sentiment"][0], test_last, "clf")
    pd.DataFrame({"volatility_rf": imp_vol, "direction_gbm": imp_dir}).to_csv(
        config.RESULTS_DIR / "feature_importance.csv")

    # ---------------------------------------------------------- figures
    print("\n[figures]")
    figs = [
        plots.plot_overview(prices, panel),
        plots.plot_sentiment(sent),
        plots.plot_metric_bars(vol_tab, "MAE", "Next-day volatility: MAE by model",
                               "03_vol_mae.png"),
        plots.plot_metric_bars(vol_tab, "RMSE", "Next-day volatility: RMSE by model",
                               "04_vol_rmse.png"),
        plots.plot_vol_forecasts(vol_preds, models.CORE_VOL),
        plots.plot_period_comparison(core_periods, models.CORE_VOL),
        plots.plot_metric_bars(dir_tab, "Accuracy", "Extension: 5-day direction accuracy",
                               "06a_dir_accuracy.png", lower_is_better=False, ref=0.5),
        plots.plot_metric_bars(dir_tab, "ROC_AUC", "Extension: 5-day direction ROC AUC",
                               "06b_dir_auc.png", lower_is_better=False, ref=0.5),
        plots.plot_event_study(es),
        plots.plot_intraday(es_intra),
        plots.plot_importance(imp_vol, f"What drives the volatility forecast? "
                              f"(RF: price + sentiment, {last_year})", "09a_importance_vol.png"),
        plots.plot_importance(imp_dir, f"What drives the direction forecast? "
                              f"(GBM: price + sentiment, {last_year})", "09b_importance_dir.png"),
    ]
    for f in sorted(config.FIG_DIR.glob("*.png")):
        print(f"   {f.relative_to(config.ROOT)}")
    del figs

    # ---------------------------------------------------------- save trained models
    joblib.dump({"volatility": vol_models, "direction": dir_models},
                config.MODEL_DIR / "final_fold_models.joblib")
    purge = max(config.HORIZON, config.DIR_HORIZON)
    # the volatility models used by the forecasting tool (main.py option 4), refitted on
    # ALL available data; the tool recommends whichever was most accurate in testing
    full = {}
    for m in [models.FeatureModel("HAR-RV", groups["har"], models.ols, "reg"),
              models.FeatureModel("RF: price only", groups["price_only"], models.rf_reg, "reg",
                                  models.RF_GRID),
              models.FeatureModel("RF: price + sentiment", groups["all"], models.rf_reg, "reg",
                                  models.RF_GRID),
              models.FeatureModel("RF: price + key sentiment",
                                  groups["price_only"] + models.key_sentiment(groups),
                                  models.rf_reg, "reg", models.RF_GRID),
              models.HybridHAR("Hybrid HAR+RF: price only", groups["har"], groups["price_only"]),
              models.HybridHAR("Hybrid HAR+RF: price + sentiment", groups["har"], groups["all"])]:
        full[m.name] = m.fit(panel.iloc[:-purge])
    # direction models for the prediction tool (option 4): price only vs price + sentiment
    full_dir = {}
    for m in [models.FeatureModel("RF: price only", groups["price_only"], models.rf_clf, "clf",
                                  models.RF_GRID),
              models.FeatureModel("RF: price + sentiment", groups["all"], models.rf_clf, "clf",
                                  models.RF_GRID)]:
        full_dir[m.name] = m.fit(panel.iloc[:-purge])
    joblib.dump({"volatility_models": full, "direction_models": full_dir, "feature_groups": groups,
                 "trained_until": str(panel.index[-purge - 1].date())},
                config.MODEL_DIR / "full_data_models.joblib")
    print(f"   saved models to {config.MODEL_DIR.relative_to(config.ROOT)}/")

    # ---------------------------------------------------------- dataset size summary
    texts = data_texts.load_texts()
    market = data_prices.load_market()
    sents = sentiment.load_sentences()
    summary = {
        "gbpusd_daily_rows": int(len(prices)),
        "cross_market_tickers": len(config.MARKET_TICKERS),
        "cross_market_cells": int(market.notna().sum().sum()),
        "gbpusd_hourly_rows": 0 if hourly is None else int(len(hourly)),
        "documents_total": int(len(texts)),
        **{f"documents_{b}_{t}": int(n) for (b, t), n in
           texts.groupby(["bank", "doc_type"]).size().items()},
        "words": int(sent["n_words"].sum()),
        "sentences_scored": int(len(sents)),
        "panel_rows": int(len(panel)),
        "panel_features": len(groups["all"]),
        "test_period": period,
        "demo_mode": demo,
    }
    (config.RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    report += ["HYPER-PARAMETERS chosen by inner chronological validation (folds chosen out of "
               f"{tuning['test_year'].nunique() if len(tuning) else 0}; full detail in "
               "results/hyperparameters_all_folds.csv)", tuning_sum.to_string(index=False), ""]
    ts_reg = config.RESULTS_DIR / "ts_regime_comparison.csv"
    if ts_reg.exists():
        reg = pd.read_csv(ts_reg)
        report += ["TIME-SERIES REGIME COMPARISON (monthly; full tables in results/ts_*.csv)",
                   _fmt(reg.set_index(["period", "sentiment", "outcome"])
                        [["months", "corr_sent_change_vs_outcome", "granger_p"]], 3), ""]
    report += ["DATA SUMMARY", json.dumps(summary, indent=2)]
    (config.RESULTS_DIR / "summary.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    from src import findings
    findings.generate(print_it=False)       # plain-English results/key_findings.txt for the report
    print(f"\n   key findings saved to {(config.RESULTS_DIR / 'key_findings.txt').relative_to(config.ROOT)}")


def run_pipeline(start="prices", demo=False):
    """Run the project from step `start` onwards (callable from Python, e.g. main.py)."""
    if start not in STEPS:
        raise ValueError(f"start must be one of {STEPS}")
    np.random.seed(config.RANDOM_SEED)
    t0 = time.time()
    todo = STEPS[STEPS.index(start):]

    if "prices" in todo:
        print("=== Step 1: Collecting financial market data from Yahoo Finance ===")
        data_prices.build_price_file(demo=demo)
    if "texts" in todo:
        data_texts.build_text_file(demo=demo)
    if "sentences" in todo:
        sentiment.split_documents(data_texts.load_texts(), demo=demo)
    if "sentiment" in todo:
        sentiment.build_sentiment_file(data_texts.load_texts(), demo=demo)
    if "features" in todo:
        features.build_panel(data_prices.load_prices(), data_prices.load_market(),
                             sentiment.load_sentiment())
    if "timeseries" in todo:
        time_series_step()
    if "train" in todo:
        train_and_evaluate(demo)
    print(f"\nDone in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="synthetic data, offline smoke test")
    ap.add_argument("--from", dest="start", choices=STEPS, default=config.RUN_FROM,
                    help="start from this step, re-using earlier outputs")
    args = ap.parse_args()
    run_pipeline(args.start, args.demo)


if __name__ == "__main__":
    main()
