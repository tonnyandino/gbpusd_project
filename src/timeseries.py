"""
Step 7 - analysing relationships between central bank sentiment and financial variables
using time-series methods.

Reproduces the analysis design of Hilscher, Nabors & Raviv (2026) and applies it to
GBP/USD:
  * quarterly sentiment indices smoothed with the Hodrick-Prescott filter (lambda = 1600)
  * co-movement of the two central banks' sentiment (full-sample and rolling 2-year
    correlation; the paper finds Fed and ECB sentiment move together)
  * local projections (Jorda, 2005): outcome at t+h regressed on current and lagged
    sentiment plus the lagged outcome, with White heteroskedasticity-robust errors
Extensions at monthly frequency (more observations):
  * augmented Dickey-Fuller stationarity tests (Dickey & Fuller, 1979)
  * lead-lag cross-correlations between sentiment and GBP/USD returns / volatility
  * Granger causality F-tests (Granger, 1969) in both directions

Everything is implemented with numpy/scipy so each step is transparent.
Note: the HP filter is two-sided (it uses future data), which is fine for this
descriptive analysis but is NEVER used in the forecasting models.
"""
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import spsolve

import config


# ====================================================================== building blocks
def hp_filter(y, lam=config.HP_LAMBDA_QUARTERLY):
    """Hodrick-Prescott trend: argmin sum (y - tau)^2 + lam * sum (second diff of tau)^2."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return y.copy()
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    a = sparse.eye(n) + lam * (d.T @ d)
    return spsolve(a.tocsc(), y)


def ols(y, X, cov="white"):
    """OLS with a constant. cov = 'white' (HC1) or 'classic'. Returns a dict."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    if cov == "white":
        meat = (X * resid[:, None] ** 2).T @ X
        vcov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    else:
        vcov = xtx_inv * (resid @ resid) / (n - k)
    se = np.sqrt(np.clip(np.diag(vcov), 0, None))
    t = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), df=n - k)
    tss = np.sum((y - y.mean()) ** 2)
    return {"beta": beta, "se": se, "t": t, "p": p, "resid": resid, "n": n, "k": k,
            "ssr": float(resid @ resid), "r2": 1 - resid @ resid / tss if tss > 0 else np.nan}


def _lags(s, k):
    if k <= 0:
        return pd.DataFrame(index=s.index)
    return pd.concat({f"L{i}": s.shift(i) for i in range(1, k + 1)}, axis=1)


# MacKinnon (2010) response-surface critical values, ADF with a constant (no trend)
_ADF_CV = {"1%": (-3.43035, -6.5393, -16.786, -79.433),
           "5%": (-2.86154, -2.8903, -4.234, -40.040),
           "10%": (-2.56677, -1.5384, -2.809, 0.0)}


def adf_test(x, max_lags=6):
    """ADF test with a constant; lag length chosen by AIC. H0: unit root (non-stationary)."""
    x = pd.Series(np.asarray(x, dtype=float)).dropna().reset_index(drop=True)
    dx = x.diff()
    best = None
    for k in range(0, max_lags + 1):
        df = pd.concat([dx.rename("dx"), x.shift(1).rename("lag_level"), _lags(dx, k)], axis=1)
        df = df.iloc[max_lags + 1:].dropna()          # same sample for every k
        if len(df) < 15:
            continue
        r = ols(df["dx"], df.drop(columns="dx"), cov="classic")
        aic = r["n"] * np.log(r["ssr"] / r["n"]) + 2 * r["k"]
        if best is None or aic < best[0]:
            best = (aic, k, r)
    if best is None:
        return {"adf_stat": np.nan, "lags": np.nan, "n": len(x)}
    _, k, r = best
    T = r["n"]
    cv = {lvl: b0 + b1 / T + b2 / T ** 2 + b3 / T ** 3 for lvl, (b0, b1, b2, b3) in _ADF_CV.items()}
    stat = r["t"][1]                                   # t-stat on the lagged level
    return {"adf_stat": stat, "lags": k, "n": T, "cv_1%": cv["1%"], "cv_5%": cv["5%"],
            "cv_10%": cv["10%"], "stationary_5%": bool(stat < cv["5%"])}


def granger_test(y, x, lags):
    """F-test of H0: lags of x do not help predict y beyond y's own lags."""
    d = pd.concat([y.rename("y"), _lags(y, lags).add_prefix("y_"), _lags(x, lags).add_prefix("x_")],
                  axis=1).dropna()
    yv = d["y"]
    restricted = ols(yv, d[[c for c in d if c.startswith("y_")]], cov="classic")
    full = ols(yv, d.drop(columns="y"), cov="classic")
    df1, df2 = lags, full["n"] - full["k"]
    f = ((restricted["ssr"] - full["ssr"]) / df1) / (full["ssr"] / df2)
    return f, stats.f.sf(f, df1, df2), full["n"]


def local_projection(y, shock, horizons=4, lags=1):
    """Jorda (2005): y_{t+h} = a + b_h * shock_t + c * shock_{t-1} + d * y_{t-1} + e,
    White standard errors - the specification described in Hilscher et al. (2026)."""
    rows = []
    for h in range(0, horizons + 1):
        d = pd.concat([y.shift(-h).rename("y_lead"), shock.rename("shock"),
                       _lags(shock, lags).add_prefix("shock_"), _lags(y, lags).add_prefix("y_")],
                      axis=1).dropna()
        if len(d) < 12:
            continue
        r = ols(d["y_lead"], d.drop(columns="y_lead"), cov="white")
        rows.append({"h": h, "beta": r["beta"][1], "se": r["se"][1], "p": r["p"][1], "n": r["n"]})
    return pd.DataFrame(rows)


# ====================================================================== datasets
def _bank_index(sent, bank, freq):
    """Average of the bank's FinBERT document index within each period (minutes, or
    speeches if minutes were unavailable), carried forward through periods without one."""
    from src.features import index_docs
    s, _ = index_docs(sent, bank)
    return s.set_index("date")["fin_net"].resample(freq).mean().ffill()


def build_frequency_data(prices, market, sent, freq):
    """freq: 'QE' (quarter end) or 'ME' (month end)."""
    try:
        pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-31"])).resample(freq).mean()
    except ValueError:                              # pandas < 2.2 uses 'Q' / 'M'
        freq = freq.rstrip("E")
    out = pd.DataFrame({
        "boe_sent": _bank_index(sent, "BoE", freq),
        "fed_sent": _bank_index(sent, "Fed", freq),
        "gbp_ret": prices["ret"].resample(freq).sum() * 100,                       # %
        "gbp_vol": np.log(np.sqrt(prices["var_gk"].resample(freq).mean() * config.TRADING_DAYS)),
    })
    if "vix_close" in market:
        out["vix"] = np.log(market["vix_close"].resample(freq).mean())
    if "us10y_close" in market:
        out["us10y_chg"] = market["us10y_close"].resample(freq).last().diff()
    out = out.dropna()
    out["div_sent"] = out["boe_sent"] - out["fed_sent"]
    return out


# ====================================================================== main analysis
def run_analysis(prices, market, sent):
    print("=== Step 7: Analysing relationships between sentiment and financial variables "
          "(time-series methods) ===")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    q = build_frequency_data(prices, market, sent, "QE")
    m = build_frequency_data(prices, market, sent, "ME")
    res = {"quarterly": q, "monthly": m}

    # ---- (1) quarterly HP-smoothed indices and co-movement (paper reproduction)
    for c in ["boe_sent", "fed_sent"]:
        q[f"{c}_hp"] = hp_filter(q[c].values)
    corr_raw = q["boe_sent"].corr(q["fed_sent"])
    corr_hp = q["boe_sent_hp"].corr(q["fed_sent_hp"])
    q["rolling_corr_8q"] = q["boe_sent"].rolling(8).corr(q["fed_sent"])
    print(f"[ts] quarterly sample {q.index.min().date()} -> {q.index.max().date()} ({len(q)} quarters)")
    print(f"[ts] correlation BoE vs Fed sentiment: raw {corr_raw:.2f}, HP-smoothed {corr_hp:.2f}")
    comove = pd.DataFrame({"measure": ["corr_raw_quarterly", "corr_hp_quarterly",
                                       "corr_raw_monthly"],
                           "value": [corr_raw, corr_hp, m["boe_sent"].corr(m["fed_sent"])]})
    comove.to_csv(config.RESULTS_DIR / "ts_comovement.csv", index=False)
    q.to_csv(config.RESULTS_DIR / "ts_quarterly_data.csv")
    m.to_csv(config.RESULTS_DIR / "ts_monthly_data.csv")

    # ---- (2) local projections at quarterly frequency (paper's method). The shock is
    # standardised, so each coefficient = effect of a one-standard-deviation rise in sentiment.
    lp_rows = []
    for shock in ["boe_sent_hp", "fed_sent_hp", "boe_sent", "fed_sent"]:
        z = (q[shock] - q[shock].mean()) / q[shock].std()
        for outcome in ["gbp_vol", "gbp_ret"] + (["vix"] if "vix" in q else []):
            lp = local_projection(q[outcome], z, horizons=4, lags=1)
            lp.insert(0, "outcome", outcome)
            lp.insert(0, "shock", shock)
            lp_rows.append(lp)
    lp_all = pd.concat(lp_rows, ignore_index=True)
    lp_all.to_csv(config.RESULTS_DIR / "ts_local_projections.csv", index=False)
    res["local_projections"] = lp_all

    # ---- (3) stationarity (monthly)
    adf = pd.DataFrame({c: adf_test(m[c]) for c in m.columns}).T
    adf.index.name = "variable"
    adf.to_csv(config.RESULTS_DIR / "ts_adf.csv")
    res["adf"] = adf

    # ---- (4) lead-lag cross-correlations (monthly): corr(sentiment_t, outcome_{t+k})
    cc_rows = []
    for s in ["boe_sent", "fed_sent", "div_sent"]:
        for o in ["gbp_ret", "gbp_vol"]:
            for k in range(-6, 7):
                cc_rows.append({"sentiment": s, "outcome": o, "lead_k": k,
                                "corr": m[s].corr(m[o].shift(-k))})
    cc = pd.DataFrame(cc_rows)
    cc.to_csv(config.RESULTS_DIR / "ts_cross_correlations.csv", index=False)
    res["cross_corr"] = cc

    # ---- (5) Granger causality (monthly), both directions; non-stationary series differenced
    def _prep(c):
        st = adf.loc[c, "stationary_5%"] if c in adf.index else True
        return m[c] if bool(st) else m[c].diff()
    g_rows = []
    for s in ["boe_sent", "fed_sent", "div_sent"]:
        for o in ["gbp_ret", "gbp_vol"]:
            for lags in (1, 2, 3):
                f1, p1, n1 = granger_test(_prep(o), _prep(s), lags)
                f2, p2, n2 = granger_test(_prep(s), _prep(o), lags)
                g_rows.append({"cause": s, "effect": o, "lags": lags, "F": f1, "p": p1, "n": n1})
                g_rows.append({"cause": o, "effect": s, "lags": lags, "F": f2, "p": p2, "n": n2})
    gr = pd.DataFrame(g_rows)
    gr.to_csv(config.RESULTS_DIR / "ts_granger.csv", index=False)
    res["granger"] = gr

    # ---- (6) regime comparison: does the sentiment-market link change after COVID?
    reg_rows = []
    for name, (a, b) in config.PERIODS.items():
        sub = m.loc[(m.index >= pd.Timestamp(a)) & (m.index <= pd.Timestamp(b))]
        if len(sub) < 24:                       # need ~2 years of months for Granger tests
            continue
        for s in ["boe_sent", "fed_sent", "div_sent"]:
            for o in ["gbp_ret", "gbp_vol"]:
                row = {"period": name, "months": len(sub), "sentiment": s, "outcome": o,
                       "corr_same_month": sub[s].corr(sub[o]),
                       "corr_sent_leads_1m": sub[s].corr(sub[o].shift(-1)),
                       "corr_sent_change_vs_outcome": sub[s].diff().corr(sub[o])}
                f, p, n = granger_test(sub[o], sub[s].diff() if s != "div_sent" else sub[s], 2)
                row.update({"granger_F_sent_to_outcome": f, "granger_p": p})
                reg_rows.append(row)
    regimes = pd.DataFrame(reg_rows)
    regimes.to_csv(config.RESULTS_DIR / "ts_regime_comparison.csv", index=False)
    res["regimes"] = regimes
    if not regimes.empty:
        piv = regimes.pivot_table(index=["sentiment", "outcome"], columns="period",
                                  values="corr_sent_change_vs_outcome")
        print("[ts] correlation of sentiment CHANGES with GBP/USD, by period:")
        print(piv.round(3).to_string())

    sig = gr[(gr["p"] < 0.05)]
    print(f"[ts] ADF: stationary at 5% -> "
          + ", ".join(f"{c}={'yes' if v else 'no'}" for c, v in adf["stationary_5%"].items()))
    print(f"[ts] Granger tests run: {len(gr)} | significant at 5%: {len(sig)}")
    if not sig.empty:
        print(sig.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return res
