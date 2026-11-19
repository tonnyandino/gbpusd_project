# Central Bank Sentiment and GBP/USD Volatility

Reproducing a FinBERT central bank sentiment index (Hilscher, Nabors & Raviv, 2026) and extending it to forecast next-day GBP/USD volatility.

**Primary paper:** Hilscher, J., Nabors, K. & Raviv, A. (2026). Information in central bank sentiment: An analysis of Fed and ECB communication. *Journal of International Financial Markets, Institutions and Money*, 110, 102335.

The paper builds a sentiment index from FOMC minutes and ECB press conferences with a FinBERT model fine-tuned on FOMC minutes. It then studies the index with time-series methods. Its data are available only on request, so this project **rebuilds the index from public sources** and then **extends** it:

| | Paper | This project |
|---|---|---|
| Central banks | Fed and ECB | Fed and **Bank of England** (the two banks behind GBP/USD) |
| Documents | FOMC minutes, ECB press conferences (plus speeches) | FOMC minutes and BoE MPC minutes from the official websites; **BIS central bankers' speeches** for both banks (used automatically when a website blocks downloads) |
| Sentence splitting | NLTK | NLTK (Punkt) |
| Pre-processing | Sentiment focus, punctuation and stop-word removal | Same (the dependency-parse comma rule is omitted) |
| Model | FinBERT fine-tuned on FOMC minutes (Gössi et al., 2023) | **FinBERT-FOMC**, the public release of that research group |
| Sentence score | +1 / 0 / −1 | +1 / 0 / −1 |
| Document index | Mean of sentence scores | Mean of sentence scores |
| Time series | Quarterly, HP-filtered (λ = 1600); local projections | Same, plus ADF, cross-correlations and Granger causality |
| **Extension** | — | **Forecasting next-day GBP/USD volatility** with random forests on Yahoo Finance data, plus a pre-/post-COVID regime comparison |

---

## Reproducing the results (for markers): 2 commands, no downloads

The package includes all collected data, the FinBERT sentence scores and the trained models, so the results can be reproduced **offline** and without a GPU:

```bash
pip install -r requirements.txt
python run_all.py --from features      # rebuilds every model, table and figure (about 5-10 min)
```

Then run `python main.py` (or open it and press Run) for the interactive tool. Option 4 can also re-train the models on the saved data (`retrain`), or download fresh data and re-run everything (`update`). Output tables are in `results/` (start with `results/summary.txt` and `results/key_findings.txt`), figures in `results/figures/`, and trained models in `models/`.

Re-collecting everything from the internet is also possible (`python run_all.py`). It needs about 2 GB for the language models, and some networks rate-limit Yahoo Finance; see *How to run* below.

## Interactive tool (`main.py`)

```
  CAN CENTRAL BANK SENTIMENT HELP PREDICT GBP/USD VOLATILITY?
   1) Look up a Fed or BoE meeting: decision, what was said, sentiment
   2) GBP/USD, volatility and central bank sentiment
   3) How GBP/USD reacted to chosen meetings (first hour + next days)
   4) Sentiment summary -> GBP/USD prediction
```

Option 4 works in three steps. Step 1 summarises the latest Fed and BoE documents: FinBERT tone, hawkish/dovish stance, change vs the previous document, and key sentences. Step 2 predicts next-day volatility and the 5-day direction with the trivial baseline, the model without sentiment and the model with sentiment. For a past day (2019 onwards) it uses the walk-forward test predictions, which were made only with earlier data, and shows what actually happened. Step 3 gives the decision, how much the sentiment changed it, and the test-period track record against the baselines (Diebold-Mariano test).

## Pipeline (what the program prints)

```
=== Step 1: Collecting financial market data from Yahoo Finance ===
=== Step 2: Collecting Federal Reserve and Bank of England communications ===
=== Step 3: Splitting documents into sentences using NLTK ===
=== Step 4: Processing sentences with FinBERT-FOMC (FinBERT adapted to central bank communications) ===
    Classifying sentiment as positive, neutral or negative
=== Step 5: Aggregating sentence scores into document-level sentiment indices ===
=== Step 6: Building the daily modelling panel ===
=== Step 7: Analysing relationships between sentiment and financial variables (time-series methods) ===
=== Step 8: Forecasting experiments (chronological walk-forward test) ===
```

## Experiments (lecturer requirements)

| Experiment | What it tests |
|---|---|
| Previous-day volatility baseline | Whether the AI beats a trivial forecast |
| Random forest, price only | Whether historical market features predict volatility |
| FinBERT sentiment analysis (steps 3–5, 7) | Whether the paper's sentiment method can be reproduced |
| Random forest, price + sentiment | Whether central bank sentiment adds forecasting value |

**Metrics:** MAE and RMSE on log volatility (regression metrics), plus a Diebold–Mariano test of price + sentiment against price only.

**Chronological test set:** walk-forward with an expanding window. Each year from 2019 is predicted by models trained only on earlier years, with overlapping rows purged.

**Regime comparison:** every result is also reported for pre-COVID (2019), the COVID year (2020) and post-COVID (2021 onwards) test periods. The time-series tests are likewise split into 2015–2019 and 2021 onwards.

**Additional benchmarks:** historical mean, GARCH(1,1), HAR-RV, and gradient boosting (price only and price + sentiment).

**Modifications and hybrids:**
- **Hybrid HAR+RF (residual learning):** HAR-RV (OLS) models volatility persistence, then a random forest is trained on HAR's residuals; the forecast is the sum of the two. One version uses price features, one adds sentiment.
- **RF + key sentiment:** 9 core sentiment features instead of about 29, to test whether the full sentiment set over-fits.

**Hyper-parameter tuning:** every random forest, gradient boosting and logistic model picks its settings from a grid (RF: `max_depth`, `max_features`; GBM: `learning_rate`, `max_depth`; logistic: `C`). The grid is scored on the **last year of each training window**, with a purged gap so overlapping targets can't leak, and is never scored on the test year. The chosen settings are saved for every fold (`results/hyperparameters_all_folds.csv`, `results/hyperparameters_summary.csv`), along with the fitted GARCH parameters.

**Extensions:**
- Hawkish/dovish scores (FOMC-RoBERTa)
- 5-day direction classification (coin flip / most-frequent-label baselines; accuracy, AUC)
- Event study of GBP/USD after more-positive or more-negative minutes

---

## How to run

Python 3.10–3.12. **No API keys are needed:** Yahoo Finance, the Fed, the BoE and the BIS are all free public sources.

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python check_setup.py                  # 30-second check of packages and data sources
python run_all.py                      # full run on real data, about 30-50 minutes the first time
```

**Without the terminal:** in VS Code select the `.venv` interpreter (Cmd/Ctrl + Shift + P, then "Python: Select Interpreter"), open `run_all.py` and press the Run button. To re-run only the models later, set `RUN_FROM = "train"` in `config.py`.

Other options: `python run_all.py --demo` (offline test with synthetic data; the numbers are meaningless) and `python run_all.py --from <step>` with steps `prices, texts, sentences, sentiment, features, timeseries, train`.

### Hawkish/dovish model (optional login)
`gtfintechlab/FOMC-RoBERTa` is a **gated** model on Hugging Face. Without access, the program automatically scores hawkish/dovish language with the terminology dictionary in `src/sentiment.py`, and every output says which method was used. To use the model instead:
1. Create a free account at huggingface.co, open the model page and click **"Agree and access repository"**.
2. Create an access token under *Settings → Access Tokens* (type "Read").
3. Run `hf auth login` (older versions: `huggingface-cli login`), paste the token, then `python run_all.py --from sentiment`. Saved FinBERT scores are reused, so only the hawkish model runs.

### Where the central bank texts come from
1. **Official websites:** FOMC minutes and statements (federalreserve.gov) and BoE Monetary Policy Summary and minutes (bankofengland.co.uk). Some networks block these sites; the program tests each one first.
2. **BIS central bankers' speeches:** free bulk files for research (https://www.bis.org/cbspeeches/download.htm), always collected for both banks.

Each bank's index uses its **minutes** when at least 20 were downloaded, otherwise its **speeches**. The terminal prints which source was used (`[features] BoE sentiment index built from ...`), so you can state it in the report.

**Mac users:** if Yahoo or NLTK downloads fail with `SSLError`, run `open "/Applications/Python 3.12/Install Certificates.command"` once, or switch networks (for example, a phone hotspot).

**Manual texts:** if a central bank page can't be scraped, paste the text into `data/texts_manual/boe/YYYY-MM-DD.txt` or `data/texts_manual/fed/YYYY-MM-DD.txt`, named by release date, and run `python run_all.py --from texts`.

---

## Project structure

```
main.py              interactive tool (START HERE)
menu.py              the views used by main.py
make_submission.py   builds the submission zip (code + data + models + results)
check_setup.py       30-second check of packages and data sources
config.py            all settings (tickers, dates, models, periods, RUN_FROM)
run_all.py           runs every step in order
src/data_prices.py   Step 1  Yahoo Finance: GBP/USD daily + hourly, 9 cross-market tickers
src/data_texts.py    Step 2  scrape FOMC minutes/statements and BoE summaries/minutes
src/sentiment.py     Steps 3-5  NLTK sentences, sentiment focus, FinBERT-FOMC, document indices
src/features.py      Step 6  daily panel: targets and features, no look-ahead
src/timeseries.py    Step 7  HP filter, local projections, ADF, cross-correlation, Granger
src/models.py        Step 8  baselines, random forests, GARCH, HAR, gradient boosting
src/evaluation.py    Step 8  walk-forward test, MAE/RMSE, Diebold-Mariano, regimes, event study
src/plots.py         figures
src/demo_data.py     synthetic data for --demo only
```

## Outputs

| File | Content |
|---|---|
| `results/summary.txt` | All headline tables |
| `results/core_experiments.csv` | **The lecturer table:** naive vs RF price only vs RF price + sentiment |
| `results/core_experiments_by_period.csv` | The same, split pre-COVID / COVID / post-COVID |
| `results/vol_metrics_all_models.csv` | Every volatility model |
| `results/ts_*.csv` | Time-series results (local projections, ADF, Granger, regimes) |
| `results/figures/*.png` | 15 figures for the report |
| `data/cb_sentences.csv`, `data/cb_sentence_scores.csv` | Every sentence with its FinBERT label |
| `data/cb_sentiment.csv` | Document-level sentiment indices |
| `models/*.joblib` | Trained models |

---

## References

- Andersen, T. G. & Bollerslev, T. (1998). Answering the skeptics: Yes, standard volatility models do provide accurate forecasts. *International Economic Review*, 39(4).
- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.
- Bird, S., Klein, E. & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly (NLTK).
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3).
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1).
- Chen, Z., Gössi, S., Kim, W., Bermeitinger, B. & Handschuh, S. (2023). FinBERT-FOMC: Fine-tuned FinBERT model with sentiment focus method for enhancing sentiment analysis of FOMC minutes. *Proceedings of the 4th ACM International Conference on AI in Finance (ICAIF '23)*. Model: `ZiweiChen/FinBERT-FOMC`.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2).
- Dickey, D. A. & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3).
- Garman, M. B. & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1).
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3).
- Hilscher, J., Nabors, K. & Raviv, A. (2026). Information in central bank sentiment: An analysis of Fed and ECB communication. *Journal of International Financial Markets, Institutions and Money*, 110, 102335.
- Hodrick, R. J. & Prescott, E. C. (1997). Postwar U.S. business cycles: An empirical investigation. *Journal of Money, Credit and Banking*, 29(1).
- Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review*, 95(1).
- Kiss, T. & Strunk, J. (2006). Unsupervised multilingual sentence boundary detection. *Computational Linguistics*, 32(4) (Punkt).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- MacKinnon, J. G. (2010). Critical values for cointegration tests. *Queen's Economics Department Working Paper* No. 1227.
- Shah, A., Paturi, S. & Chava, S. (2023). Trillion Dollar Words: A new financial dataset, task & market analysis. *ACL 2023*. Model: `gtfintechlab/FOMC-RoBERTa`.
- Bank for International Settlements (2024). Central bank speeches, 1996–2026. https://www.bis.org/cbspeeches/download.htm
- Data: Yahoo Finance (via `yfinance`); Board of Governors of the Federal Reserve System; Bank of England.
