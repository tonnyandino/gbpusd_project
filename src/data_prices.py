"""
Step 1 - download daily GBP/USD OHLC prices and compute a daily variance estimate.

Realised volatility proper needs intraday data (Andersen & Bollerslev, 1998). Free
intraday FX history is hard to redistribute, so we use the Garman & Klass (1980)
range-based estimator on daily OHLC bars, which is far more efficient than squared
close-to-close returns (Garman & Klass, 1980; Parkinson, 1980).
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import config


def _tidy(hist):
    hist = hist[["Open", "High", "Low", "Close"]].copy()
    idx = pd.to_datetime(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    hist.index = idx.normalize()
    hist.index.name = "date"
    return hist[~hist.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------- browser-style session
# Yahoo answers non-browser requests with HTTP 429 "Too Many Requests". yfinance therefore
# connects through curl_cffi, which imitates Chrome. curl_cffi uses its OWN certificate
# list, so on networks / antivirus that inspect HTTPS traffic it fails with SSLError.
# Fix: give curl_cffi the operating system's certificates (macOS keychain) as well.
def system_ca_bundle():
    """PEM file with the OS's trusted certificates + certifi's (None if unavailable)."""
    path = config.DATA_DIR / "system_ca_bundle.pem"
    if path.exists() and path.stat().st_size > 1000:
        return str(path)
    pem = ""
    if sys.platform == "darwin":
        for keychain in ["/System/Library/Keychains/SystemRootCertificates.keychain",
                         "/Library/Keychains/System.keychain",
                         os.path.expanduser("~/Library/Keychains/login.keychain-db")]:
            try:
                pem += subprocess.run(["security", "find-certificate", "-a", "-p", keychain],
                                      capture_output=True, text=True, timeout=60).stdout
            except Exception:
                pass
    elif sys.platform.startswith("win"):
        try:
            import ssl
            for store in ("ROOT", "CA"):
                for cert, enc, _ in ssl.enum_certificates(store):
                    if enc == "x509_asn":
                        pem += ssl.DER_cert_to_PEM_cert(cert)
        except Exception:
            pass
    else:
        for f in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
            if os.path.exists(f):
                pem += open(f, encoding="utf-8", errors="ignore").read()
    try:
        import certifi
        pem += open(certifi.where(), encoding="utf-8").read()
    except Exception:
        pass
    if pem.count("BEGIN CERTIFICATE") < 10:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pem, encoding="utf-8")
    return str(path)


_BROWSER = {"session": None}


def browser_session():
    """curl_cffi session that looks like Chrome AND trusts the system certificates.
    Passed to yfinance (yf.Ticker(..., session=...)) and used for direct API calls."""
    if _BROWSER["session"] is None:
        try:
            from curl_cffi import requests as cffi_requests
        except ImportError:
            return None
        ca = system_ca_bundle()
        try:
            _BROWSER["session"] = cffi_requests.Session(impersonate="chrome", verify=ca or True)
        except TypeError:                                  # older curl_cffi
            _BROWSER["session"] = cffi_requests.Session(impersonate="chrome")
    return _BROWSER["session"]


YAHOO_CHART_URLS = ["https://query2.finance.yahoo.com/v8/finance/chart/{s}",
                    "https://query1.finance.yahoo.com/v8/finance/chart/{s}"]
_HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
            "Accept": "application/json,text/plain,*/*"}


LAST_YAHOO_ERROR = []          # reasons for failed direct requests (shown by check_setup.py)
_YAHOO = {"session": None, "crumb": None}


def _yahoo_session():
    """requests session holding Yahoo's cookie + 'crumb' token (the handshake yfinance
    normally does with curl_cffi, which fails with SSLError on some networks)."""
    import requests
    if _YAHOO["session"] is None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        crumb = None
        try:
            s.get("https://fc.yahoo.com", timeout=20, allow_redirects=True)   # sets cookie
            c = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=20)
            if c.status_code == 200 and c.text and "<" not in c.text and len(c.text) < 50:
                crumb = c.text.strip()
        except Exception as exc:
            LAST_YAHOO_ERROR.append(f"cookie handshake: {type(exc).__name__}: {str(exc)[:120]}")
        _YAHOO.update(session=s, crumb=crumb)
    return _YAHOO["session"], _YAHOO["crumb"]


def yahoo_chart(symbol, interval="1d", start=None, range_=None, local_dates=True):
    """Yahoo Finance chart API called directly with `requests` (the same data yfinance
    reads). Used first because on some networks yfinance's own connection library
    (curl_cffi) fails with SSLError while ordinary HTTPS requests work fine.
    Tries without, then with Yahoo's cookie/crumb. Returns an OHLC DataFrame or None."""
    import requests
    from urllib.parse import quote

    params = {"interval": interval, "includePrePost": "false"}
    if range_:
        params["range"] = range_
    else:
        params["period1"] = int(pd.Timestamp(start or "2000-01-01").timestamp())
        params["period2"] = int(pd.Timestamp.now().timestamp()) + 86400
    plain = requests.Session()
    plain.headers.update(_HEADERS)
    browser = browser_session()
    for base in YAHOO_CHART_URLS:
        for attempt in range(3):
            if attempt == 0:                      # 1st: Chrome-like session + system certs
                if browser is None:
                    continue
                sess, crumb = browser, None
            elif attempt == 2:                    # 3rd: plain requests with cookie + crumb
                sess, crumb = _yahoo_session()
            else:                                 # 2nd: plain requests
                sess, crumb = plain, None
            p = dict(params, **({"crumb": crumb} if crumb else {}))
            try:
                r = sess.get(base.format(s=quote(symbol, safe="")), params=p, timeout=30)
            except Exception as exc:
                LAST_YAHOO_ERROR.append(f"{symbol}: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            if r.status_code != 200:
                LAST_YAHOO_ERROR.append(f"{symbol}: HTTP {r.status_code} {r.text[:80]!r}")
                continue
            df = _parse_chart(r, local_dates)
            if df is not None:
                return df
    return None


def _parse_chart(r, local_dates):
    """Turn a chart-API JSON response into an OHLC DataFrame (None if it has no data)."""
    try:
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        ts = np.asarray(res["timestamp"], dtype="int64")
    except Exception as exc:
        LAST_YAHOO_ERROR.append(f"unexpected response: {type(exc).__name__}")
        return None
    if local_dates:          # daily bars: shift to the exchange's local calendar day
        ts = ts + int(res.get("meta", {}).get("gmtoffset", 0) or 0)
    idx = pd.to_datetime(ts, unit="s")
    df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                       "Close": q["close"]}, index=idx).astype(float).dropna(how="all")
    return df if len(df) else None


def _flatten(df):
    """yf.download returns (field, ticker) column pairs in some versions: keep the field."""
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        level = 0 if "Close" in df.columns.get_level_values(0) else 1
        df.columns = df.columns.get_level_values(level)
    return df


def _plain_download(symbol, start, end, interval="1d", period=None):
    """yfinance called the standard way - exactly like a normal yfinance script."""
    import yfinance as yf
    kw = dict(interval=interval, auto_adjust=False, progress=False)
    if period:
        kw["period"] = period
    else:
        kw.update(start=start, end=end)
    try:
        return _flatten(yf.download(symbol, multi_level_index=False, **kw))
    except TypeError:                         # older yfinance without multi_level_index
        return _flatten(yf.download(symbol, **kw))


def _yahoo_daily(symbol, start=config.PRICE_START, end=config.PRICE_END):
    """Download daily OHLC from Yahoo Finance, trying several request styles in order:

    1. yfinance the standard way (yf.download / yf.Ticker) - works on most setups.
    2. Yahoo's chart API directly (Chrome-style session with the system certificates,
       then plain requests) - helps where yfinance's own connection fails.
    3. yfinance with that Chrome-style session, then period='max' / '10y'.
    """
    import time

    import yfinance as yf  # imported here so --demo mode does not need it

    def keep(hist, label):
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        hist = _tidy(hist.dropna(subset=["Close"]))
        hist = hist[hist.index >= pd.Timestamp(start)]
        if end is not None:
            hist = hist[hist.index < pd.Timestamp(end)]
        if hist.empty:
            return None
        if label != "yf.download":
            print(f"   {symbol}: downloaded with '{label}'")
        return hist

    def custom_ticker():
        sess = browser_session()
        try:
            return yf.Ticker(symbol, session=sess) if sess is not None else yf.Ticker(symbol)
        except Exception:
            return yf.Ticker(symbol)

    attempts = [
        ("yf.download", lambda: _plain_download(symbol, start, end)),
        ("yf.Ticker period=max", lambda: yf.Ticker(symbol).history(period="max", interval="1d",
                                                                     auto_adjust=False)),
        ("direct Yahoo API", lambda: yahoo_chart(symbol, "1d", start=start)),
        ("yfinance + Chrome session", lambda: custom_ticker().history(period="max", interval="1d",
                                                                      auto_adjust=False)),
        ("yf.download period=10y", lambda: _plain_download(symbol, None, None, period="10y")),
    ]
    for label, fetch in attempts:
        try:
            hist = keep(fetch(), label)
        except Exception as exc:
            print(f"   ! {symbol}: {label} failed ({type(exc).__name__})")
            hist = None
        if hist is not None:
            return hist
        time.sleep(1)
    raise RuntimeError(f"Yahoo Finance returned no data for {symbol}. Run `python check_setup.py` "
                       "or use colab_yahoo_download.py.")


def download_prices(start=config.PRICE_START, end=config.PRICE_END):
    """Download daily OHLC for GBP/USD from Yahoo Finance via yfinance."""
    return _yahoo_daily(config.TICKER, start, end)


def clean_ohlc(df):
    """Fix the small inconsistencies that are common in free FX data."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    # High must be >= max(Open, Close) and Low <= min(Open, Close)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    # drop weekend rows that occasionally appear
    df = df[df.index.dayofweek < 5]
    return df


def add_variance_measures(df):
    """Add log return, Garman-Klass daily variance and annualised daily vol."""
    df = df.copy()
    df["ret"] = np.log(df["Close"]).diff()
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2

    # Fallback: if the bar has no range (bad data) use the squared close-to-close return
    bad = (df["High"] <= df["Low"]) | (gk <= 0) | gk.isna()
    gk = gk.where(~bad, df["ret"] ** 2)
    df["var_gk"] = gk.clip(lower=config.MIN_DAILY_VAR)
    df["vol_ann"] = np.sqrt(df["var_gk"] * config.TRADING_DAYS)
    df["bad_bar"] = bad.astype(int)
    return df.iloc[1:]


def download_market(start=config.PRICE_START, end=config.PRICE_END):
    """Download every cross-market ticker and return one wide daily table."""
    frames = []
    for name, symbol in config.MARKET_TICKERS.items():
        try:
            df = download_prices_for(symbol, start, end)
        except Exception as exc:
            print(f"   ! could not download {symbol} ({name}): {exc} - skipped")
            continue
        frames.append(market_columns(name, df))
        print(f"[market] {name:6s} {symbol:10s} {len(df)} rows")
    if not frames:
        raise RuntimeError("No cross-market data downloaded.")
    return pd.concat(frames, axis=1).sort_index()


def download_prices_for(symbol, start, end):
    return clean_ohlc(_yahoo_daily(symbol, start, end))


def market_columns(name, df):
    """Close level, log return and log Garman-Klass variance for one ticker."""
    out = pd.DataFrame(index=df.index)
    out[f"{name}_close"] = df["Close"]
    out[f"{name}_ret"] = np.log(df["Close"]).diff()
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    gk = (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)
    gk = gk.where(gk > 0, out[f"{name}_ret"] ** 2).clip(lower=config.MIN_DAILY_VAR)
    out[f"{name}_lvar"] = np.log(gk)
    return out


def download_hourly():
    """Last ~730 days of hourly GBP/USD bars (Yahoo's limit for 1h data), in UTC."""
    try:
        h = _plain_download(config.TICKER, None, None, interval="1h", period="730d")
        if h is not None and len(h) > 100:
            h = h[["Open", "High", "Low", "Close"]].copy()
            idx = pd.to_datetime(h.index)
            h.index = (idx.tz_convert("UTC") if idx.tz is not None
                       else idx.tz_localize("UTC")).tz_localize(None)
            h.index.name = "datetime_utc"
            return h[~h.index.duplicated(keep="last")].sort_index()
    except Exception:
        pass
    direct = yahoo_chart(config.TICKER, "1h", range_="730d", local_dates=False)
    if direct is not None and len(direct) > 100:
        direct.index.name = "datetime_utc"
        return direct[~direct.index.duplicated(keep="last")].sort_index()
    import yfinance as yf
    sess = browser_session()
    t = yf.Ticker(config.TICKER, session=sess) if sess is not None else yf.Ticker(config.TICKER)
    h = t.history(period="730d", interval="1h", auto_adjust=False)
    if h.empty:
        raise RuntimeError("no hourly data")
    h = h[["Open", "High", "Low", "Close"]].copy()
    idx = pd.to_datetime(h.index)
    h.index = (idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")).tz_localize(None)
    h.index.name = "datetime_utc"
    return h[~h.index.duplicated(keep="last")].sort_index()


def build_price_file(demo=False):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if demo:
        from src.demo_data import synthetic_hourly, synthetic_market, synthetic_prices
        raw = synthetic_prices()
        market = synthetic_market(raw.index)
        hourly = synthetic_hourly()
    else:
        raw = download_prices()
        market = download_market()
        try:
            hourly = download_hourly()
        except Exception as exc:
            print(f"   ! hourly download failed ({exc}); intraday event study will be skipped")
            hourly = None
    df = add_variance_measures(clean_ohlc(raw))
    df.to_csv(config.PRICES_CSV)
    market.to_csv(config.MARKET_CSV)
    if hourly is not None:
        hourly.to_csv(config.HOURLY_CSV)
    n_hourly = 0 if hourly is None else len(hourly)
    print(f"[prices] GBP/USD {len(df)} trading days {df.index.min().date()} -> "
          f"{df.index.max().date()} | fallback bars: {int(df['bad_bar'].sum())}")
    print(f"[prices] cross-market table {market.shape[0]} rows x {market.shape[1]} columns "
          f"| hourly GBP/USD bars: {n_hourly}")
    return df


def load_prices():
    return pd.read_csv(config.PRICES_CSV, index_col="date", parse_dates=["date"])


def load_market():
    return pd.read_csv(config.MARKET_CSV, index_col="date", parse_dates=["date"])


def load_hourly():
    if not config.HOURLY_CSV.exists():
        return None
    return pd.read_csv(config.HOURLY_CSV, index_col="datetime_utc", parse_dates=["datetime_utc"])
