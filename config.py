"""
Central configuration for the project
"Reproducing and extending a FinBERT central bank sentiment index (Hilscher, Nabors &
Raviv, 2026) to forecast GBP/USD volatility".
Change values here rather than inside the individual modules.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                    # cached HTML pages
MANUAL_TEXT_DIR = DATA_DIR / "texts_manual"   # optional: drop extra .txt files here
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

PRICES_CSV = DATA_DIR / "gbpusd_daily.csv"
MARKET_CSV = DATA_DIR / "market_daily.csv"
HOURLY_CSV = DATA_DIR / "gbpusd_hourly.csv"   # Yahoo keeps ~730 days of 1h bars
TEXTS_CSV = DATA_DIR / "cb_texts.csv"                   # step 2: one row per document
SENTENCES_CSV = DATA_DIR / "cb_sentences.csv"           # step 3: one row per sentence
SENTENCE_SCORES_CSV = DATA_DIR / "cb_sentence_scores.csv"   # step 4: + model labels
SENTIMENT_CSV = DATA_DIR / "cb_sentiment.csv"           # step 5: document-level indices
FINBERT_CACHE_CSV = DATA_DIR / "cb_finbert_cache.csv"   # saved FinBERT scores (never recomputed)
PANEL_CSV = DATA_DIR / "model_panel.csv"                # daily modelling panel

# ---------------------------------------------------------------- market data (Yahoo Finance)
TICKER = "GBPUSD=X"            # target
PRICE_START = "2014-06-01"     # earlier than the texts so rolling windows are warm
PRICE_END = None               # None = up to today
MARKET_TICKERS = {             # cross-market features (name -> Yahoo symbol)
    "eurusd": "EURUSD=X", "usdjpy": "USDJPY=X", "eurgbp": "EURGBP=X",
    "dxy": "DX-Y.NYB",         # US dollar index
    "vix": "^VIX",             # S&P 500 implied volatility
    "spx": "^GSPC", "ftse": "^FTSE",
    "us10y": "^TNX",           # US 10-year Treasury yield (x10)
    "gold": "GC=F",
}

# ---------------------------------------------------------------- central bank texts
BOE_FIRST_YEAR = 2015          # combined "Monetary Policy Summary and minutes" pages start Aug 2015
FED_FIRST_YEAR = 2015
TEXT_LAST_YEAR = None          # None = current year
FED_MINUTES_LAG_DAYS = 21      # FOMC minutes are released three weeks after the decision
# Document type behind each bank's sentiment index: minutes when enough could be downloaded
# (the paper's main document type), otherwise BIS speeches (also used in the paper).
INDEX_DOC_PRIORITY = ["minutes", "speech"]
INDEX_MIN_DOCS = 20
REQUEST_PAUSE_SEC = 0.6        # be polite to the central bank websites
USER_AGENT = ("Mozilla/5.0 (academic research project; GBP/USD volatility study) "
              "python-requests")

# ---------------------------------------------------------------- sentiment (paper reproduction)
# Hilscher et al. (2026) use the FinBERT model fine-tuned on FOMC minutes by Goessi et al.
# (2023). The paper names no checkpoint; FinBERT-FOMC is the public release of that
# research group (Chen, Goessi, Kim, Bermeitinger & Handschuh, 2023).
FINBERT_MODEL = "ZiweiChen/FinBERT-FOMC"
HAWKISH_MODEL = "gtfintechlab/FOMC-RoBERTa"   # extension: Shah, Paturi & Chava (2023)
# FOMC-RoBERTa is a gated model on Hugging Face (account + access request needed). If it
# cannot be downloaded, the hawkish/dovish score falls back to the terminology dictionary.
SENTIMENT_FOCUS = True         # trim clauses around "but / although / though" (Goessi et al., 2023)
REMOVE_STOPWORDS = True        # the paper removes punctuation and stop words before scoring
SENT_BATCH_SIZE = 16
SENT_MAX_TOKENS = 256
HP_LAMBDA_QUARTERLY = 1600     # Hodrick-Prescott smoothing used in the paper
SURPRISE_DECAY_DAYS = 5        # decay of the "fresh news" features after a release

# ---------------------------------------------------------------- forecasting targets
HORIZON = 1                    # volatility target: next trading day
DIR_HORIZON = 5                # direction target (extension): next 5 trading days
EVENT_WINDOW = 10              # days after an announcement shown in the event study
TRADING_DAYS = 252             # annualisation factor
MIN_DAILY_VAR = 1e-7           # floor for the daily variance estimator (avoids log(0))

# ---------------------------------------------------------------- running
# Step the Run button / `python run_all.py` starts from. Use "train" to re-run only the
# models once data and sentiment scores exist (options: prices, texts, sentences,
# sentiment, features, timeseries, train).
RUN_FROM = "prices"

# ---------------------------------------------------------------- evaluation
FIRST_TEST_YEAR = 2019         # chronological walk-forward: refit each January, test that year
# Regime comparison: results are also reported separately for these sub-periods
PERIODS = {
    "pre-COVID": ("2015-01-01", "2019-12-31"),
    "COVID year": ("2020-01-01", "2020-12-31"),
    "post-COVID": ("2021-01-01", "2100-12-31"),
}
RANDOM_SEED = 42
