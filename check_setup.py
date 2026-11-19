"""
Quick check (about 30 seconds) that every data source and package works on this computer.
Run it before the long run:   python check_setup.py
"""
import importlib
import sys

OK, FAIL, WARN = "OK  ", "FAIL", "WARN"
results = []


def report(status, what, detail=""):
    results.append(status)
    print(f"  [{status}] {what}" + (f"  -> {detail}" if detail else ""))


print(f"Python {sys.version.split()[0]}")
print("\n1. Packages")
for pkg in ["pandas", "numpy", "scipy", "sklearn", "matplotlib", "requests", "bs4",
            "yfinance", "nltk", "transformers", "torch"]:
    try:
        importlib.import_module(pkg)
        report(OK, pkg)
    except ImportError:
        report(FAIL, pkg, "run: pip install -r requirements.txt")

print("\n2. Yahoo Finance (market data)")
sys.path.insert(0, ".")
import src  # noqa: E402,F401  (uses the operating system's certificates)
from src import data_prices  # noqa: E402


def tls_issuer(host):
    """Who signed the HTTPS certificate we receive for this host. Yahoo's real
    certificate is issued by DigiCert; anything else means the network or antivirus
    software is inspecting HTTPS traffic."""
    import socket
    import ssl
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=15) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                issuer = dict(x[0] for x in tls.getpeercert()["issuer"])
        return issuer.get("organizationName") or issuer.get("commonName") or "unknown"
    except Exception as exc:
        return f"could not check ({type(exc).__name__})"


issuer = tls_issuer("query2.finance.yahoo.com")
if "digicert" in issuer.lower():
    report(OK, "HTTPS to Yahoo is not intercepted", f"certificate issued by {issuer}")
else:
    report(WARN, "HTTPS to Yahoo looks intercepted",
           f"certificate issued by '{issuer}' (network filter or antivirus web protection)")
ca = data_prices.system_ca_bundle()
report(OK if ca else WARN, "System certificates exported for yfinance",
       "" if ca else "could not read the system certificate store")

data_prices.LAST_YAHOO_ERROR.clear()
import yfinance as yf  # noqa: E402
print(f"  yfinance version {yf.__version__}")
try:
    h = data_prices._plain_download("GBPUSD=X", "2014-06-01", None)
    if h is not None and len(h) > 2000:
        report(OK, "GBP/USD daily history (standard yfinance)", f"{len(h)} days")
    else:
        raise RuntimeError("standard yfinance returned no data")
except Exception as exc:
    print(f"  standard yfinance: {type(exc).__name__}: {str(exc)[:100]}")
    h = None
try:
    if h is not None and len(h) > 2000:
        raise StopIteration
    h = data_prices.yahoo_chart("GBPUSD=X", "1d", start="2014-06-01")
    if h is not None and len(h) > 2000:
        report(OK, "GBP/USD daily history (direct Yahoo API)",
               f"{len(h)} days from {h.index.min().date()}")
    else:
        import yfinance as yf
        sess = data_prices.browser_session()
        t = yf.Ticker("GBPUSD=X", session=sess) if sess is not None else yf.Ticker("GBPUSD=X")
        h = t.history(period="max", interval="1d")
        report(OK if len(h) > 2000 else FAIL, "GBP/USD daily history (yfinance + system certificates)",
               f"{len(h)} days" if len(h) else "no data")
except StopIteration:
    pass
except Exception as exc:
    report(FAIL, "GBP/USD daily history", f"{type(exc).__name__}: {str(exc)[:100]}")
if data_prices.LAST_YAHOO_ERROR and FAIL in results:
    print("     why the Yahoo requests failed (send this to your helper):")
    for line in list(dict.fromkeys(data_prices.LAST_YAHOO_ERROR))[:6]:
        print(f"       - {line}")
try:
    hv = data_prices.yahoo_chart("^VIX", "1d", start="2024-01-01")
    report(OK if hv is not None and len(hv) else WARN, "VIX (cross-market example)",
           "" if hv is not None and len(hv) else "cross-market data may be incomplete")
except Exception as exc:
    report(WARN, "VIX (cross-market example)", type(exc).__name__)

print("\n3. Central bank text sources (at least one per bank must work)")
from src import data_texts  # noqa: E402

sites_ok = []
for name, url in [("Federal Reserve website", data_texts.FED_PROBE),
                  ("Bank of England website", data_texts.BOE_PROBE)]:
    ok = data_texts.site_reachable(url)
    sites_ok.append(ok)
    report(OK if ok else WARN, name, "" if ok else "blocked on this network - BIS speeches will be used")

import config  # noqa: E402

blob = data_texts._download_bytes(data_texts.BIS_YEAR_URL.format(y=2024),
                                  config.RAW_DIR / "bis" / "speeches-2024.zip")
if blob:
    tidy = data_texts._standardise_bis(data_texts._read_bis_zip(blob))
    banks = tidy["meta"].map(data_texts._bank_of).value_counts()
    report(OK if len(tidy) else FAIL, "BIS central bankers' speeches (2024)",
           f"{len(tidy)} speeches, BoE {int(banks.get('BoE', 0))}, Fed {int(banks.get('Fed', 0))}")
elif all(sites_ok):
    report(WARN, "BIS central bankers' speeches",
           "not reachable, but not needed: both central bank websites work")
else:
    report(FAIL, "BIS central bankers' speeches",
           "download failed AND a central bank website is blocked - try another network")

print("\n4. NLTK sentence model")
try:
    from src.sentiment import get_sentence_splitter
    split, method = get_sentence_splitter()
    report(OK if method == "nltk" else WARN, "NLTK Punkt", method)
except Exception as exc:
    report(FAIL, "NLTK Punkt", type(exc).__name__)

print()
if FAIL in results:
    print("Some checks FAILED - fix those first (see the arrows above).")
else:
    print("All essential checks passed - you can run:  python run_all.py")
