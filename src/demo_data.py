"""
Synthetic data used ONLY by `python run_all.py --demo`.

It lets you (or a marker) check that the whole pipeline runs end-to-end in about a
minute without internet access, transformers or a GPU. Results from demo mode are
meaningless and must not be reported.
"""
import numpy as np
import pandas as pd

import config

RNG_SEED = 7
START, END = config.PRICE_START, "2026-10-30"


def _event_dates():
    days = pd.bdate_range("2015-08-01", END)
    boe = [d for d in days if d.dayofweek == 3 and d.month in (2, 3, 5, 6, 8, 9, 11, 12)
           and d.day <= 7]
    fed = [d for d in days if d.dayofweek == 2 and d.month in (1, 3, 4, 6, 7, 9, 10, 12)
           and 15 <= d.day <= 21]
    return pd.DatetimeIndex(boe), pd.DatetimeIndex(fed)


def synthetic_prices():
    rng = np.random.default_rng(RNG_SEED)
    idx = pd.bdate_range(START, END)
    boe, fed = _event_dates()
    n = len(idx)
    omega, alpha, beta = 0.02, 0.06, 0.92        # GARCH(1,1) in percent units
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = omega / (1 - alpha - beta)
    shock = np.where(idx.isin(boe), 2.0, 1.0) * np.where(idx.isin(fed), 1.5, 1.0)
    for t in range(n):
        if t:
            s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t] * shock[t]) * rng.standard_normal()
    close = 1.30 * np.exp(np.cumsum(r / 100))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.0003, n))
    span = np.abs(rng.normal(0, 1, n)) * np.sqrt(s2 * shock) / 100 + 0.001
    high = np.maximum(open_, close) * np.exp(span * 0.6)
    low = np.minimum(open_, close) * np.exp(-span * 0.6)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=pd.DatetimeIndex(idx, name="date"))


def synthetic_market(index):
    rng = np.random.default_rng(RNG_SEED + 1)
    frames = []
    for i, name in enumerate(config.MARKET_TICKERS):
        level = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, len(index))))
        op = level * np.exp(rng.normal(0, 0.002, len(index)))
        span = np.abs(rng.normal(0, 0.006, len(index))) + 0.001
        df = pd.DataFrame({"Open": op, "High": np.maximum(op, level) * (1 + span),
                           "Low": np.minimum(op, level) * (1 - span), "Close": level},
                          index=index)
        if name == "vix":
            df = df * 0.2
        from src.data_prices import market_columns
        frames.append(market_columns(name, df))
    return pd.concat(frames, axis=1)


def synthetic_hourly():
    rng = np.random.default_rng(RNG_SEED + 2)
    end = pd.Timestamp(END) + pd.Timedelta(hours=23)
    idx = pd.date_range(end - pd.Timedelta(days=730), end, freq="h")
    idx = idx[idx.dayofweek < 5]
    close = 1.27 * np.exp(np.cumsum(rng.normal(0, 0.0012, len(idx))))
    return pd.DataFrame({"Open": close, "High": close * 1.0008, "Low": close * 0.9992,
                         "Close": close}, index=pd.DatetimeIndex(idx, name="datetime_utc"))


HAWK = ["The Committee voted to raise Bank Rate as inflationary pressures persist.",
        "Upside risks to inflation remain and further tightening may be required.",
        "Although growth slowed, the labour market remained tight and wage growth was strong."]
DOVE = ["The Committee voted to cut Bank Rate given weak demand.",
        "Inflation fell back, but downside risks to growth have increased materially.",
        "Disinflation is under way and unemployment has risen across most regions."]
NEUTRAL = ["The Committee will continue to monitor developments closely.",
           "Energy prices and trade developments were discussed at length by members.",
           "Growth in the second quarter was broadly in line with expectations overall."]


def _doc(rng, stance, n_neutral):
    n_h = max(0, int(round(4 + 3 * stance + rng.normal())))
    n_d = max(0, int(round(4 - 3 * stance + rng.normal())))
    sents = (list(rng.choice(HAWK, n_h)) + list(rng.choice(DOVE, n_d))
             + list(rng.choice(NEUTRAL, n_neutral)))
    rng.shuffle(sents)
    return " ".join(sents)


def _fed_fmt(x):
    """4.25 -> '4-1/4' (the way FOMC statements write rates)."""
    whole, frac = int(x), round((x - int(x)) * 4)
    tail = {0: "", 1: "1/4", 2: "1/2", 3: "3/4"}[frac]
    if whole == 0:
        return tail or "0"
    return f"{whole}-{tail}" if tail else str(whole)


def synthetic_texts():
    rng = np.random.default_rng(RNG_SEED + 3)
    boe, fed = _event_dates()
    rows = []
    stance, rate = 0.0, 0.5
    for d in boe:                                   # BoE: decision + minutes the same day
        stance = 0.8 * stance + rng.normal(0, 0.6)
        step = 0.25 if stance > 0.7 else (-0.25 if stance < -0.7 else 0.0)
        rate = max(0.1, rate + step)
        verb = ("increase Bank Rate by 0.25 percentage points, to" if step > 0 else
                "reduce Bank Rate by 0.25 percentage points, to" if step < 0 else "maintain Bank Rate at")
        votes = "unanimously" if step == 0 else f"by a majority of {rng.integers(5, 8)}-{9 - rng.integers(5, 8)}"
        lead = f"The Monetary Policy Committee voted {votes} to {verb} {rate:g}%. "
        rows.append({"date": d, "bank": "BoE", "doc_type": "minutes", "url": "synthetic",
                     "text": lead + _doc(rng, stance, 12)})
    stance, lo = 0.0, 0.25
    for d in fed:                                   # Fed: statement now, minutes 3 weeks later
        stance = 0.8 * stance + rng.normal(0, 0.6)
        step = 0.25 if stance > 0.7 else (-0.25 if stance < -0.7 and lo > 0 else 0.0)
        lo = max(0.0, lo + step)
        verb = "raise" if step > 0 else "lower" if step < 0 else "maintain"
        prep = "at" if step == 0 else "to"
        lead = (f"The Committee decided to {verb} the target range for the federal funds rate "
                f"{prep} {_fed_fmt(lo)} to {_fed_fmt(lo + 0.25)} percent. ")
        rows.append({"date": d, "bank": "Fed", "doc_type": "statement", "url": "synthetic",
                     "text": lead + _doc(rng, stance, 3)})
        release = d + pd.Timedelta(days=config.FED_MINUTES_LAG_DAYS)
        if release <= pd.Timestamp(END):
            rows.append({"date": release, "bank": "Fed", "doc_type": "minutes",
                         "url": "synthetic", "text": _doc(rng, stance, 12)})
    return pd.DataFrame(rows)
