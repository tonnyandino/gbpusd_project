"""
Step 6 - build the daily modelling panel (one row per GBP/USD trading day).

Timing: a row dated t only uses information available at the end of trading day t.
BoE decisions + minutes are published at 12:00 UK time; FOMC statements at 14:00 US
Eastern (19:00 UK); FOMC minutes at 14:00 US Eastern three weeks after the meeting.
All fall inside the London-time daily bar of their release day.

Targets (strictly forward-looking):
    y_vol : log annualised Garman-Klass volatility of the NEXT trading day (H = 1)
            (average over t+1..t+H if HORIZON is changed in config.py)       -> regression
    y_dir : 1 if GBP/USD is higher DIR_HORIZON days ahead, else 0            -> classification
"""
import numpy as np
import pandas as pd

import config

H = config.HORIZON
HD = config.DIR_HORIZON


# ------------------------------------------------------------------ targets
def _log_vol(mean_var):
    return np.log(np.sqrt(mean_var * config.TRADING_DAYS))


def add_targets(df):
    var = df["var_gk"]
    fwd_mean_var = sum(var.shift(-k) for k in range(1, H + 1)) / H
    df["y_vol"] = _log_vol(fwd_mean_var)
    df["fwd_ret"] = np.log(df["Close"].shift(-HD) / df["Close"])
    df["y_dir"] = (df["fwd_ret"] > 0).astype(float).where(df["fwd_ret"].notna())
    return df


# ------------------------------------------------------------------ price features
def add_price_features(df):
    v = df["var_gk"]
    df["lvol_d"] = _log_vol(v)                         # today's volatility
    df["lvol_w"] = _log_vol(v.rolling(5).mean())       # last week
    df["lvol_m"] = _log_vol(v.rolling(22).mean())      # last month  (HAR: Corsi, 2009)
    df["naive_vol"] = _log_vol(v.rolling(H).mean())    # "previous value" of the target
    df["abs_ret"] = df["ret"].abs()
    df["ret_1"] = df["ret"]
    df["ret_5"] = df["ret"].rolling(5).sum()           # 1-week momentum
    df["ret_22"] = df["ret"].rolling(22).sum()         # 1-month momentum
    df["dow"] = df.index.dayofweek                     # day-of-week effect in FX volatility
    return df


HAR_COLS = ["lvol_d", "lvol_w", "lvol_m"]
PRICE_COLS = HAR_COLS + ["abs_ret", "ret_1", "ret_5", "ret_22", "dow"]


# ------------------------------------------------------------------ cross-market features
def market_features(market, index):
    m = market.reindex(market.index.union(index)).sort_index().ffill(limit=3).reindex(index)
    out = pd.DataFrame(index=index)
    for col in m.columns:
        name, kind = col.rsplit("_", 1)
        if name in ("vix", "us10y"):
            continue                                   # levels, handled below
        if kind == "ret":
            out[f"mkt_{name}_ret"] = m[col]
            out[f"mkt_{name}_ret5"] = m[col].rolling(5).sum()
        elif kind == "lvar":
            out[f"mkt_{name}_lvar"] = m[col]
            out[f"mkt_{name}_lvar5"] = m[col].rolling(5).mean()
    if "vix_close" in m:
        out["mkt_vix_level"] = np.log(m["vix_close"])
        out["mkt_vix_chg5"] = out["mkt_vix_level"].diff(5)
    if "us10y_close" in m:
        out["mkt_us10y_chg1"] = m["us10y_close"].diff()
        out["mkt_us10y_chg5"] = m["us10y_close"].diff(5)
    return out


# ------------------------------------------------------------------ central bank features
TEXT_VARS = ["fin_net", "fin_prob_net", "fin_neg_share", "hawk_score", "dict_net", "dict_uncert"]


def _snap(dates, index):
    """Map release dates to the same or next GBP/USD trading day."""
    pos = np.clip(index.searchsorted(pd.DatetimeIndex(dates)), 0, len(index) - 1)
    return index[pos]


def _days_since(flag):
    n = np.arange(len(flag))
    last = pd.Series(np.where(flag.values == 1, n, np.nan), index=flag.index).ffill()
    return pd.Series(n - last.values, index=flag.index)


def index_docs(sent, bank):
    """Documents behind a bank's sentiment index: minutes if enough were collected,
    otherwise speeches (config.INDEX_DOC_PRIORITY). Returns (documents, doc_type)."""
    b = sent[sent["bank"] == bank]
    for t in config.INDEX_DOC_PRIORITY:
        d = b[b["doc_type"] == t]
        if len(d) >= config.INDEX_MIN_DOCS:
            return d, t
    d = b[b["doc_type"] != "statement"]
    return (d if len(d) else b), "all documents"


def bank_features(sent, bank, index):
    """Text features from the bank's minutes + announcement-day calendar features."""
    p = bank.lower()
    out = pd.DataFrame(index=index)
    b = sent[sent["bank"] == bank].copy()

    # ---- text: sentiment index from minutes (or speeches, see index_docs)
    docs, _ = index_docs(sent, bank)
    docs = docs.copy()
    docs["tday"] = _snap(docs["date"], index)
    num = [c for c in docs.columns if c not in ("date", "bank", "doc_type", "tday")
           and pd.api.types.is_numeric_dtype(docs[c])]
    docs = docs.groupby("tday")[num].mean()          # several documents on one day -> average
    docs["fin_chg"] = docs["fin_net"].diff()
    docs["hawk_chg"] = docs["hawk_score"].diff()
    out[f"{p}_text_event"] = index.isin(docs.index).astype(int)
    ev = docs.reindex(index)
    for v in TEXT_VARS + ["fin_chg", "hawk_chg"]:
        out[f"{p}_{v}"] = ev[v].ffill()
    if bank == "BoE":
        for kw in [c for c in docs.columns if c.startswith("kw_")]:
            out[f"{p}_{kw}"] = ev[kw].ffill()
    decay = np.exp(-_days_since(out[f"{p}_text_event"]) / config.SURPRISE_DECAY_DAYS)
    out[f"{p}_fin_surprise"] = out[f"{p}_fin_chg"] * decay          # signed "fresh news"
    out[f"{p}_abs_surprise"] = out[f"{p}_fin_chg"].abs() * decay
    out[f"{p}_hawk_surprise"] = out[f"{p}_hawk_chg"] * decay

    # ---- calendar: policy announcement days (BoE: same day as minutes; Fed: statements)
    ann = b[b["doc_type"] == "statement"]["date"] if bank == "Fed" else docs.index.to_series()
    if len(ann) == 0:
        ann = docs.index.to_series()
    ann_days = pd.DatetimeIndex(_snap(ann, index)).unique()
    out[f"{p}_event"] = index.isin(ann_days).astype(int)
    out[f"{p}_days_since"] = _days_since(out[f"{p}_event"])
    # Meetings are pre-scheduled and public, so "is there a decision tomorrow / within the
    # forecast window" is known in advance (rare unscheduled meetings are a caveat).
    window = max(H, 1)
    out[f"{p}_event_ahead"] = (sum(out[f"{p}_event"].shift(-k) for k in range(1, window + 1))
                               .fillna(0).clip(upper=1))
    return out


def build_panel(prices, market, sent):
    print("=== Step 6: Building the daily modelling panel ===")
    df = prices.copy()
    df = add_targets(df)
    df = add_price_features(df)
    df = df.join(market_features(market, df.index))
    df = df.join(bank_features(sent, "BoE", df.index)).join(bank_features(sent, "Fed", df.index))
    df["div_fin_net"] = df["boe_fin_net"] - df["fed_fin_net"]      # sentiment divergence
    df["div_hawk"] = df["boe_hawk_score"] - df["fed_hawk_score"]
    for bank in ("BoE", "Fed"):
        d, t = index_docs(sent, bank)
        print(f"[features] {bank} sentiment index built from {len(d)} {t}")
        if t != "minutes" and not (bank == "Fed" and (sent["doc_type"] == "statement").any()):
            print(f"           note: no {bank} policy-decision dates available, so its calendar "
                  "features use document dates instead")
    feature_cols = feature_groups(df)["all"]
    df = df.dropna(subset=feature_cols)
    df.to_csv(config.PANEL_CSV)
    g = feature_groups(df)
    print(f"[features] panel {df.shape[0]} days x {len(feature_cols)} features "
          f"({df.index.min().date()} -> {df.index.max().date()}) | price {len(g['price'])}, "
          f"market {len(g['market'])}, calendar {len(g['events'])}, sentiment {len(g['sentiment'])}")
    return df


def feature_groups(df):
    cols = list(df.columns)
    market = [c for c in cols if c.startswith("mkt_")]
    events = [c for c in cols if c.endswith(("_event", "_event_ahead", "_days_since"))]
    sentiment = [c for c in cols if c.startswith(("boe_", "fed_", "div_")) and c not in events]
    groups = {"har": HAR_COLS, "price": PRICE_COLS, "market": market,
              "events": events, "sentiment": sentiment}
    # "price only" = everything that is NOT central bank text: GBP/USD history, other
    # markets and the public meeting calendar. The only difference to "price + sentiment"
    # is the text, so the comparison isolates the value of central bank sentiment.
    groups["price_only"] = PRICE_COLS + market + events
    groups["all"] = groups["price_only"] + sentiment
    groups["base"] = groups["price_only"]
    return groups


def load_panel():
    return pd.read_csv(config.PANEL_CSV, index_col="date", parse_dates=["date"])
