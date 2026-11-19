"""
Step 5a - every forecasting model and baseline, behind one small interface:

    model.fit(train_df)      train_df/test_df are slices of the daily panel
    model.predict(test_df)   -> regression: predicted y_vol (log volatility)
                             -> classification: predicted P(y_dir = 1)

Task A - next-day volatility (regression)
    Core experiments (lecturer table)          Additional benchmarks
    - Previous-day volatility (naive)          - Historical mean
    - Random forest: price only                - GARCH(1,1)  (Bollerslev, 1986)
    - Random forest: price + sentiment         - HAR-RV      (Corsi, 2009), HAR + sentiment
                                               - Gradient boosting: price only / + sentiment
Task B - 5-day direction (classification, extension)
    Coin flip, most frequent label, previous direction, sentiment-surprise rule,
    logistic regression / gradient boosting / random forest, each price only vs + sentiment
"""
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

H = config.HORIZON
PURGE = max(config.HORIZON, config.DIR_HORIZON)
SEED = config.RANDOM_SEED


# ====================================================================== GARCH(1,1)
def garch_filter(r, mu, omega, alpha, beta):
    """Conditional variances sigma2[t] = Var(r[t] | info up to t-1)."""
    eps = r - mu
    sigma2 = np.empty_like(r)
    sigma2[0] = np.var(r)
    for t in range(1, len(r)):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2, eps


def garch_nll(params, r):
    mu, omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    sigma2, eps = garch_filter(r, mu, omega, alpha, beta)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)


def fit_garch(r):
    """Gaussian maximum likelihood for GARCH(1,1); r in percent."""
    v = np.var(r)
    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95)]:
        x0 = [np.mean(r), v * (1 - a0 - b0), a0, b0]
        res = minimize(garch_nll, x0, args=(r,), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


class GarchModel:
    """GARCH(1,1) on daily close-to-close returns, H-day-ahead average variance.

    GARCH forecasts close-to-close variance whereas our target is range-based, so the
    log-forecast is mapped to the target scale with a linear calibration fitted on the
    training period only (a Mincer-Zarnowitz style regression).
    """
    name = "GARCH(1,1)"

    def __init__(self, returns):
        self.r = returns.dropna() * 100.0          # percent returns, full history

    def _forecasts(self, params, r):
        mu, omega, alpha, beta = params
        sigma2, eps = garch_filter(r.values, mu, omega, alpha, beta)
        next_var = omega + alpha * eps ** 2 + beta * sigma2       # sigma2_{t+1 | t}
        persist = alpha + beta
        long_run = omega / (1 - persist)
        k = np.arange(H)
        mean_var = long_run + (next_var[:, None] - long_run) * (persist ** k)[None, :]
        mean_var = mean_var.mean(axis=1)
        lv = np.log(np.sqrt(mean_var / 1e4 * config.TRADING_DAYS))
        return pd.Series(lv, index=r.index)

    def fit(self, train):
        r_train = self.r[self.r.index <= train.index.max()]
        self.params = fit_garch(r_train.values)
        raw = self._forecasts(self.params, r_train).reindex(train.index)
        ok = raw.notna() & train["y_vol"].notna()
        self.calib = LinearRegression().fit(raw[ok].values.reshape(-1, 1), train.loc[ok, "y_vol"])
        return self

    def predict(self, test):
        r_hist = self.r[self.r.index <= test.index.max()]     # filtering only uses the past
        raw = self._forecasts(self.params, r_hist).reindex(test.index).ffill().bfill()
        return self.calib.predict(raw.values.reshape(-1, 1))


# ====================================================================== simple baselines
class NaivePrevious:
    """Trivial baseline: tomorrow's volatility = today's volatility (random walk)."""
    name = "Previous-day volatility (naive)"

    def fit(self, train):
        return self

    def predict(self, test):
        return test["naive_vol"].values


class HistoricalMean:
    name = "Historical mean"

    def fit(self, train):
        self.mu = train["y_vol"].mean()
        return self

    def predict(self, test):
        return np.full(len(test), self.mu)


class CoinFlip:
    name = "Coin flip"

    def fit(self, train):
        self.rng = np.random.default_rng(SEED)
        return self

    def predict(self, test):
        # hard random 0/1 guesses; as a probability forecast this is exactly 0 or 1,
        # so we also report it with p = 0.5 for AUC/log-loss (see evaluation.py)
        return self.rng.integers(0, 2, len(test)).astype(float)


class MostFrequent:
    name = "Most frequent label"

    def fit(self, train):
        self.p = float(train["y_dir"].mean())
        self.label = float(self.p >= 0.5)
        return self

    def predict(self, test):
        return np.full(len(test), self.p)


class PreviousDirection:
    """'Previous value' baseline for direction: the next 5 days repeat the last 5."""
    name = "Previous direction (momentum)"

    def fit(self, train):
        return self

    def predict(self, test):
        return np.where(test["ret_5"].values > 0, 0.6, 0.4)


class HawkishRule:
    """Theory-only rule, no fitted weights: a fresh hawkish BoE surprise should lift
    GBP, a fresh hawkish Fed surprise should lift USD (i.e. push GBP/USD down)."""
    name = "Sentiment-surprise rule"

    def fit(self, train):
        self.p_default = float(train["y_dir"].mean())
        return self

    def predict(self, test):
        signal = test["boe_hawk_surprise"].values - test["fed_hawk_surprise"].values
        p = np.full(len(test), self.p_default)
        p[signal > 1e-3] = 0.6
        p[signal < -1e-3] = 0.4
        return p


# ====================================================================== feature-based models
class FeatureModel:
    """Wraps an sklearn estimator with a fixed feature list and an optional small
    hyper-parameter grid tuned on the last year of the training window."""

    def __init__(self, name, features, make_estimator, task, grid=None):
        self.name = name
        self.features = list(features)
        self.make_estimator = make_estimator
        self.task = task                      # "reg" or "clf"
        self.grid = grid or [{}]

    @property
    def target(self):
        return "y_vol" if self.task == "reg" else "y_dir"

    def _score(self, est, X, y):
        if self.task == "reg":
            return mean_squared_error(y, est.predict(X))
        p = np.clip(est.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return log_loss(y, p, labels=[0, 1])

    def fit(self, train):
        train = train.dropna(subset=[self.target])
        X, y = train[self.features], train[self.target]
        if self.task == "clf":
            y = y.astype(int)
        params = self.grid[0]
        self.grid_scores = []
        if len(self.grid) > 1 and len(train) > 500:
            # inner time split: last 252 days = validation, purged gap before it
            cut = len(train) - 252
            Xtr, ytr = X.iloc[:cut - PURGE], y.iloc[:cut - PURGE]
            Xva, yva = X.iloc[cut:], y.iloc[cut:]
            scores = []
            for g in self.grid:
                est = self.make_estimator(**g).fit(Xtr, ytr)
                scores.append(self._score(est, Xva, yva))
            params = self.grid[int(np.argmin(scores))]
            self.grid_scores = list(zip(self.grid, scores))
        self.best_params = params
        self.est = self.make_estimator(**params).fit(X, y)
        return self

    def predict(self, test):
        X = test[self.features]
        if self.task == "reg":
            return self.est.predict(X)
        return self.est.predict_proba(X)[:, 1]


def _grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


def gbm_reg(**p):
    return HistGradientBoostingRegressor(random_state=SEED, max_iter=300, **p)


def gbm_clf(**p):
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=300, **p)


def rf_reg(**p):
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **p)


def rf_clf(**p):
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **p)


def ols(**p):
    return make_pipeline(StandardScaler(), LinearRegression())


def logit(**p):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **p))


# Hyper-parameter grids. Each candidate is scored on the LAST YEAR of the training window
# (inner chronological validation, purged gap) and the best one is refitted on all
# training data - so the test year is never used for tuning.
#   GBM: learning_rate = step size of each tree, max_depth = interactions per tree,
#        min_samples_leaf = minimum days per leaf (regularisation)
#   RF : max_depth = tree depth (complexity), max_features = share of features tried at
#        each split (de-correlates trees), min_samples_leaf = regularisation
#   Logistic: C = inverse L2 penalty (smaller = stronger shrinkage)
GBM_GRID = _grid(learning_rate=[0.03, 0.1], max_depth=[2, 3], min_samples_leaf=[40])
RF_GRID = _grid(max_depth=[4, 8], max_features=[0.33, 1.0], min_samples_leaf=[20])
LOGIT_GRID = _grid(C=[0.01, 0.1, 1.0])

# Compact sentiment set: the paper's index (FinBERT tone) and the hawkish score for each
# bank, their divergence and the "fresh news" surprises - 9 features instead of ~29,
# to test whether the full sentiment set over-fits.
KEY_SENTIMENT = ["boe_fin_net", "fed_fin_net", "boe_hawk_score", "fed_hawk_score",
                 "div_fin_net", "boe_fin_surprise", "fed_fin_surprise",
                 "boe_hawk_surprise", "fed_hawk_surprise"]


class HybridHAR:
    """Hybrid econometric + machine learning model (residual learning):
         step 1  HAR-RV (Corsi, 2009) fitted by OLS captures the long-memory structure
         step 2  a random forest learns what HAR misses: it is trained on HAR's residuals
         forecast = HAR forecast + random forest residual forecast
    HAR gives a strong, stable baseline; the forest adds non-linear effects (markets,
    calendar, sentiment) without having to re-learn volatility persistence."""

    def __init__(self, name, har_features, rf_features, grid=None):
        self.name = name
        self.har = FeatureModel(f"{name} [HAR part]", har_features, ols, "reg")
        self.rf = FeatureModel(f"{name} [RF part]", rf_features, rf_reg, "reg", grid or RF_GRID)
        self.features = sorted(set(har_features) | set(rf_features))
        self.target = "y_vol"

    def fit(self, train):
        train = train.dropna(subset=["y_vol"])
        self.har.fit(train)
        resid = train.copy()
        resid["y_vol"] = train["y_vol"] - self.har.predict(train)
        self.rf.fit(resid)
        self.best_params = self.rf.best_params
        self.grid_scores = self.rf.grid_scores
        return self

    def predict(self, test):
        return self.har.predict(test) + self.rf.predict(test)


# ====================================================================== model zoos
CORE_VOL = ["Previous-day volatility (naive)", "RF: price only", "RF: price + sentiment"]


def volatility_models(groups, returns):
    har, sent = groups["har"], groups["sentiment"]
    price, allf = groups["price_only"], groups["all"]
    return [
        # --- core experiments
        NaivePrevious(),
        FeatureModel("RF: price only", price, rf_reg, "reg", RF_GRID),
        FeatureModel("RF: price + sentiment", allf, rf_reg, "reg", RF_GRID),
        # --- additional benchmarks
        HistoricalMean(),
        GarchModel(returns),
        FeatureModel("HAR-RV", har, ols, "reg"),
        FeatureModel("HAR + sentiment", har + sent, ols, "reg"),
        FeatureModel("GBM: price only", price, gbm_reg, "reg", GBM_GRID),
        FeatureModel("GBM: price + sentiment", allf, gbm_reg, "reg", GBM_GRID),
        # --- modifications / hybrids (extensions)
        FeatureModel("RF: price + key sentiment", price + key_sentiment(groups), rf_reg, "reg", RF_GRID),
        HybridHAR("Hybrid HAR+RF: price only", har, price),
        HybridHAR("Hybrid HAR+RF: price + sentiment", har, allf),
    ]


def key_sentiment(groups):
    return [c for c in KEY_SENTIMENT if c in groups["sentiment"]]


def direction_models(groups):
    price, allf = groups["price_only"], groups["all"]
    return [
        CoinFlip(),
        MostFrequent(),
        PreviousDirection(),
        HawkishRule(),
        FeatureModel("Logistic: price only", price, logit, "clf", LOGIT_GRID),
        FeatureModel("Logistic: price + sentiment", allf, logit, "clf", LOGIT_GRID),
        FeatureModel("GBM: price only", price, gbm_clf, "clf", GBM_GRID),
        FeatureModel("GBM: price + sentiment", allf, gbm_clf, "clf", GBM_GRID),
        FeatureModel("RF: price only", price, rf_clf, "clf", RF_GRID),
        FeatureModel("RF: price + sentiment", allf, rf_clf, "clf", RF_GRID),
    ]
