"""
The views behind main.py (run main.py, not this file).

  1) meeting_view    a Fed or BoE meeting: decision, what was said, sentiment (+ chart)
  2) market_view     GBP/USD, volatility and central bank sentiment
  3) reaction_view   how GBP/USD reacted to chosen meetings (first hour + next days)
  4) decision_view   central bank sentiment summary -> GBP/USD prediction -> decision

Tables are printed in the terminal, charts are saved as PNG files in results/menu/.
"""
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

import config

sys.path.insert(0, str(config.ROOT))
import src  # noqa: E402,F401  (certificate fix)
from src.sentiment import DOVISH_TERMS, HAWKISH_TERMS  # noqa: E402

warnings.filterwarnings("ignore")
OUT = config.RESULTS_DIR / "menu"
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

BLUE, ORANGE, GREY, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#b9b8b1", "#52514e", "#898781", "#e1e0d9"
STANCE_COLOUR = {"Hawkish": ORANGE, "Dovish": BLUE, "Neutral": GREY}


# ====================================================================== helpers
def _need(path, what="python run_all.py"):
    if not path.exists():
        print(f"\n  Missing {path.relative_to(config.ROOT)} - run `{what}` first.\n")
        return False
    return True


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "font.size": 9.5, "axes.titleweight": "bold",
                         "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False})
    return plt


def _show(fig, name):
    plt = _plt()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"  chart saved: {path.relative_to(config.ROOT)}")
    plt.close(fig)
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        return
    try:                       # open the saved PNG in the computer's image viewer (always in front)
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.run(["xdg-open", str(path)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  (the chart opened in a separate window - the menu carries on here)")
    except Exception:
        print("  (open the PNG above to see the chart)")


def _shade_covid(ax):
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"), color=GRID, alpha=0.5, lw=0)


def load_documents():
    """Texts + their sentiment scores (rows are saved in the same order)."""
    if not (_need(config.TEXTS_CSV) and _need(config.SENTIMENT_CSV)):
        return None
    texts = pd.read_csv(config.TEXTS_CSV, parse_dates=["date"])
    sent = pd.read_csv(config.SENTIMENT_CSV, parse_dates=["date"])
    if len(texts) != len(sent):
        print("  Texts and sentiment files do not match - run `python run_all.py --from sentences`.")
        return None
    return pd.concat([texts, sent.drop(columns=["date", "bank", "doc_type"])], axis=1)


def _method(df):
    """Which method produced the hawkish score (FOMC-RoBERTa or the dictionary)."""
    if "hawk_method" in df:
        m = df["hawk_method"].dropna().astype(str)
        m = m[m != ""]
        if len(m):
            return {"dictionary": "terminology dictionary"}.get(m.iloc[0], m.iloc[0])
    return "FOMC-RoBERTa"


def key_terms(text, top=2):
    """Most frequent hawkish and dovish phrases actually used in a document."""
    t = text.lower()

    def found(patterns):
        counts = {}
        for p in patterns:
            for m in re.findall(p, t):
                m = m if isinstance(m, str) else m[0]
                counts[m] = counts.get(m, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]
    hawk, dove = found(HAWKISH_TERMS), found(DOVISH_TERMS)
    parts = []
    if hawk:
        parts.append("hawk: " + ", ".join(hawk))
    if dove:
        parts.append("dove: " + ", ".join(dove))
    return " | ".join(parts) or "-"


def add_stance(df):
    """Relative stance labels: top third of a bank's hawkish scores = Hawkish,
    bottom third = Dovish, middle = Neutral; plus the move vs the previous meeting."""
    lo, hi = df["hawk_score"].quantile([1 / 3, 2 / 3])
    df["stance"] = np.where(df["hawk_score"] >= hi, "Hawkish",
                            np.where(df["hawk_score"] <= lo, "Dovish", "Neutral"))
    chg = df["hawk_score"].diff()
    df["vs_previous"] = np.where(chg > 0.02, "more hawkish",
                                 np.where(chg < -0.02, "more dovish", "similar"))
    df.loc[df.index[0], "vs_previous"] = "-"
    df["tone"] = np.where(df["fin_net"] > 0.05, "positive",
                          np.where(df["fin_net"] < -0.05, "negative", "neutral"))
    return df


# ====================================================================== rate parsers
_DASH = r"[-‐‑‒–]"
_FEDNUM = rf"\d+{_DASH}\d/\d|\d/\d|\d+(?:\.\d+)?"
FED_RE = re.compile(rf"target range for the federal funds rate (?:at|to) ({_FEDNUM}) to ({_FEDNUM}) percent",
                    flags=re.I)
BOE_RE = re.compile(r"Bank Rate (?:by [\d.]+ percentage points?,? )?(?:at|to|unchanged at) "
                    r"(\d+(?:\.\d+)?)\s?(?:%|per ?cent)", flags=re.I)
VOTE_RE = re.compile(rf"voted (unanimously|by a majority of \d+\s?{_DASH}\s?\d+)", flags=re.I)


def fed_number(tok):
    tok = re.sub(_DASH, "-", tok)
    if "/" in tok:
        whole, frac = (tok.split("-") if "-" in tok else ("0", tok))
        num, den = frac.split("/")
        return float(whole) + float(num) / float(den)
    return float(tok)


def fed_rate(text):
    m = FED_RE.search(text)
    return (fed_number(m.group(1)), fed_number(m.group(2))) if m else (np.nan, np.nan)


def boe_rate(text):
    m = BOE_RE.search(text)
    v = VOTE_RE.search(text)
    vote = (v.group(1).replace("by a majority of ", "") if v else "")
    return (float(m.group(1)) if m else np.nan), vote


# the decision as the statement itself words it ("decided to maintain ...", "voted ... to
# increase Bank Rate"); dissents later in the text ("preferred to raise") are not matched
ACTION_RE = re.compile(r"(?:decided|voted[^.]{0,60}?) to (maintain|keep|leave|raise|increase|lower|"
                       r"reduce|cut)\b", flags=re.I)
ACTION = {"maintain": "hold", "keep": "hold", "leave": "hold", "raise": "raise", "increase": "raise",
          "lower": "cut", "reduce": "cut", "cut": "cut"}


def _moves(rate, texts=None):
    """raise / cut / hold: read from the decision sentence; if it cannot be found, compare the
    rate with the previous decision."""
    d = rate.diff()
    by_rate = np.where(d > 1e-9, "raise", np.where(d < -1e-9, "cut", "hold"))
    if texts is None:
        return by_rate
    out = []
    for fallback, text in zip(by_rate, texts):
        m = ACTION_RE.search(str(text))
        out.append(ACTION[m.group(1).lower()] if m else fallback)
    return np.array(out)


# ====================================================================== options 1 + 2
def decision_table(bank, quiet=False):
    """One row per policy decision: rate, move, vote (BoE), stance, tone, key words."""
    docs = load_documents()
    if docs is None:
        return None
    b = docs[docs["bank"] == bank]
    decision = b[b["doc_type"] == ("statement" if bank == "Fed" else "minutes")]
    if decision.empty:
        decision = b[b["doc_type"] == "minutes"]
    if decision.empty:
        print(f"\n  No {bank} decision documents were downloaded (only speeches), so the rate "
              "history cannot be read. Run step 2 on a network where the website works.\n")
        return None
    df = decision.sort_values("date").copy()
    df["doc_id"] = df.index                      # row in cb_texts.csv = doc_id in sentence files
    if bank == "Fed":
        rates = df["text"].map(fed_rate)
        df["rate_low"] = [r[0] for r in rates]
        df["rate_high"] = [r[1] for r in rates]
        df = df.dropna(subset=["rate_high"])
        df["rate"] = df["rate_high"]
        df["rate_text"] = df.apply(lambda r: f"{r.rate_low:.2f}-{r.rate_high:.2f}%", axis=1)
    else:
        parsed = df["text"].map(boe_rate)
        df["rate"] = [p[0] for p in parsed]
        df["vote"] = [p[1] for p in parsed]
        df = df.dropna(subset=["rate"])
        df["rate_text"] = df["rate"].map(lambda x: f"{x:.2f}%")
    if df.empty:
        print("\n  Could not find interest rate sentences in the documents.\n")
        return None
    df["decision"] = _moves(df["rate"], df["text"])
    df = add_stance(df.reset_index(drop=True))
    df["key_words"] = df["text"].map(key_terms)
    return df


def _month_end():
    try:
        pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-31"])).resample("ME").mean()
        return "ME"
    except ValueError:                      # pandas < 2.2
        return "M"


def _monthly_sentiment(sent, bank, col):
    from src.features import index_docs
    d, t = index_docs(sent, bank)
    return d.set_index("date")[col].resample(_month_end()).mean().ffill(), t


def _short(text, n=46):
    text = str(text)
    return text if len(text) <= n else text[:n - 1] + "…"


def market_view():
    """Option 2: GBP/USD price + volatility next to central bank sentiment."""
    if not (_need(config.PRICES_CSV) and _need(config.SENTIMENT_CSV)):
        return
    p = pd.read_csv(config.PRICES_CSV, index_col="date", parse_dates=["date"])
    sent = pd.read_csv(config.SENTIMENT_CSV, parse_dates=["date"])
    last = p.index[-1]
    yr = p.loc[last - pd.Timedelta(days=365):, "Close"]
    vol5 = np.sqrt(p["var_gk"].rolling(5).mean() * config.TRADING_DAYS) * 100
    print(f"\n  GBP/USD on {last.date()}: {p['Close'].iloc[-1]:.4f} | 12-month range {yr.min():.4f}-{yr.max():.4f} "
          f"({100 * (yr.iloc[-1] / yr.iloc[0] - 1):+.1f}%)")
    print(f"  Volatility now (5-day, annualised): {vol5.iloc[-1]:.1f}% vs long-run average {vol5.mean():.1f}%")

    me = _month_end()
    vol = np.sqrt(p["var_gk"].resample(me).mean() * config.TRADING_DAYS) * 100
    data = {"GBP/USD volatility": vol}
    for bank in ("BoE", "Fed"):
        data[f"{bank} hawkishness"], _ = _monthly_sentiment(sent, bank, "hawk_score")
        data[f"{bank} FinBERT tone"], _ = _monthly_sentiment(sent, bank, "fin_net")
    df = pd.DataFrame(data).dropna()
    rows = []
    for c in df.columns[1:]:
        for period, (a_, b_) in config.PERIODS.items():
            sub = df.loc[a_:b_]
            if len(sub) >= 12:
                rows.append({"sentiment": c, "period": period,
                             "corr": sub[c].corr(sub["GBP/USD volatility"])})
    corr = pd.DataFrame(rows).pivot_table(index="sentiment", columns="period", values="corr", sort=False)
    print("\n  Correlation of monthly central bank sentiment with GBP/USD volatility:\n")
    print(corr.to_string(float_format=lambda v: f"{v:+.2f}"))
    print("  (+1 = rise together, -1 = opposite, ~0 = no relationship)")

    plt = _plt()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.2, 1]})
    a1.plot(p.index, p["Close"], color=BLUE, lw=1.0)
    a1.set_ylabel("USD per GBP")
    a1.set_title("GBP/USD (Yahoo Finance)")
    z = ((df - df.mean()) / df.std()).rolling(3, min_periods=1).mean()
    a2.fill_between(z.index, 0, z["GBP/USD volatility"], color=GREY, alpha=0.55, lw=0, label="GBP/USD volatility")
    a2.plot(z.index, z["BoE hawkishness"], color=BLUE, lw=1.8, label="BoE hawkishness")
    a2.plot(z.index, z["Fed hawkishness"], color=ORANGE, lw=1.8, label="Fed hawkishness")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("standardised")
    a2.set_title("Volatility vs central bank hawkishness (monthly, 3-month average)")
    a2.legend(fontsize=8, loc="upper left", ncol=3)
    for ax in (a1, a2):
        _shade_covid(ax)
    _show(fig, "gbpusd_and_sentiment.png")


# (options 4 and 5 are defined after the meeting tools below)


# ====================================================================== interactive meeting tools
import textwrap  # noqa: E402

BANK_NAME = {"Fed": "Federal Reserve", "BoE": "Bank of England"}
MEETING_COLOURS = [BLUE, ORANGE, "#1baf7a", "#eda100", "#6250d6", "#d6409f", "#2a9fd6", INK2]


def _ask(prompt):
    return input(prompt).strip().lower()


def _fix_text(text):
    """Repair dashes garbled by the web pages' encoding (e.g. '10 â 2' -> '10–2')."""
    text = str(text)
    try:
        text = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return re.sub(r"\s*â[\x80-\x9f]*\s*", "–", text)


def _wrap(text, indent="    ", width=98, bullet=""):
    return textwrap.fill(_fix_text(text), width=width, initial_indent=indent + bullet,
                         subsequent_indent=indent + " " * len(bullet))


def choose_bank():
    a = _ask("\n  Which central bank?  1) Federal Reserve   2) Bank of England : ")
    return {"1": "Fed", "fed": "Fed", "2": "BoE", "boe": "BoE"}.get(a)


def _list_meetings(df, rows):
    print()
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        vote = f" {r['vote']:>11s}" if "vote" in r and isinstance(r.get("vote"), str) else ""
        print(f"   {i:>2}) {r['date']:%Y-%m-%d}  {r['rate_text']:>11s}  {r['decision']:<5s}{vote}  {r['stance']}")


def select_meetings(df, multi):
    """Ask for a year / 'recent' / a date (and, if multi, 'hawkish' / 'dovish'), then let the
    user pick from a numbered list. Returns a DataFrame of chosen meetings (or None)."""
    first, last = df["date"].min().year, df["date"].max().year
    extra = ", 'hawkish' or 'dovish' (all meetings with that tone)" if multi else ""
    a = _ask(f"\n  Type a year ({first}-{last}), 'recent', or a date (YYYY-MM-DD){extra}: ")
    if multi and a in ("hawkish", "dovish"):
        return df[df["stance"] == a.capitalize()]
    if len(a) == 10 and a[4] == "-":
        try:
            d = pd.Timestamp(a)
        except ValueError:
            print("  Not a valid date.")
            return None
        near = df.iloc[[int((df["date"] - d).abs().argmin())]]
        if near["date"].iloc[0] != d:
            print(f"  No meeting on {d.date()} - nearest is {near['date'].iloc[0].date()}.")
        return near
    if a == "recent" or a == "":
        rows = df.tail(12)
    elif a.isdigit() and first <= int(a) <= last:
        rows = df[df["date"].dt.year == int(a)]
    else:
        print("  Please type a year, 'recent' or a date.")
        return None
    if rows.empty:
        print("  No meetings found.")
        return None
    _list_meetings(df, rows)
    how = "numbers separated by commas (e.g. 1,3) or 'all'" if multi else "a number"
    pick = _ask(f"\n  Choose {how}: ")
    if multi and pick == "all":
        return rows
    try:
        idx = [int(x) - 1 for x in pick.replace(" ", "").split(",") if x]
        chosen = rows.iloc[[i for i in idx if 0 <= i < len(rows)]]
    except ValueError:
        chosen = rows.iloc[[]]
    if chosen.empty:
        print("  Nothing selected.")
        return None
    return chosen if multi else chosen.iloc[[0]]


def _announcement_utc(bank, day):
    """Fed statements 14:00 New York, BoE decisions 12:00 London -> naive UTC timestamp."""
    tz, hour = ("America/New_York", 14) if bank == "Fed" else ("Europe/London", 12)
    return (pd.Timestamp(day).normalize() + pd.Timedelta(hours=hour)).tz_localize(tz).tz_convert("UTC").tz_localize(None)


def reaction(bank, day, prices, hourly, before=3, after=24, days=10):
    """GBP/USD reaction to one announcement: hourly path (pips) and daily path (%)."""
    out = {"hourly": None, "first_hour_pips": np.nan, "first_hour_pct": np.nan, "daily": None}
    if hourly is not None and len(hourly):
        t0 = _announcement_utc(bank, day)
        pos = hourly.index.searchsorted(t0)          # first bar starting at/after the announcement
        if 1 <= pos < len(hourly) and abs((hourly.index[pos - 1] - (t0 - pd.Timedelta(hours=1)))
                                          / pd.Timedelta(hours=1)) <= 1:
            base = hourly["Close"].iloc[pos - 1]     # price at the moment of the announcement
            lo, hi = max(pos - 1 - before, 0), min(pos + after, len(hourly))
            seg = hourly["Close"].iloc[lo:hi]
            hrs = ((seg.index - t0) / pd.Timedelta(hours=1)).astype(float) + 1
            out["hourly"] = pd.Series((seg.values - base) * 1e4, index=hrs)
            out["first_hour_pips"] = (hourly["Close"].iloc[pos] - base) * 1e4
            out["first_hour_pct"] = 100 * (hourly["Close"].iloc[pos] / base - 1)
    idx = prices.index
    i = idx.searchsorted(pd.Timestamp(day).normalize())
    if 1 <= i < len(idx):
        base = prices["Close"].iloc[i - 1]           # close before the announcement day
        seg = prices["Close"].iloc[i:i + days + 1]
        out["daily"] = pd.Series(100 * (seg.values / base - 1), index=range(len(seg)))
    return out


def _load_prices():
    if not _need(config.PRICES_CSV):
        return None, None
    p = pd.read_csv(config.PRICES_CSV, index_col="date", parse_dates=["date"])
    h = None
    if config.HOURLY_CSV.exists():
        h = pd.read_csv(config.HOURLY_CSV, index_col="datetime_utc", parse_dates=["datetime_utc"])
    return p, h


def _sentences(doc_id):
    if not config.SENTENCE_SCORES_CSV.exists():
        return None
    s = pd.read_csv(config.SENTENCE_SCORES_CSV, usecols=lambda c: c in (
        "doc_id", "sent_id", "sentence", "fin_label", "p_pos", "p_neg", "p_hawk", "p_dove"))
    return s[s["doc_id"] == doc_id].sort_values("sent_id")


def _print_document(title, row, n_open=4, full=False):
    sents = _sentences(row["doc_id"])
    print(f"\n  --- {title} ---")
    if full:
        body = str(row["text"])
        print(_wrap(body[:3000] + (" ..." if len(body) > 3000 else "")))
    elif sents is not None and len(sents):
        print("  Opening:")
        for t in sents["sentence"].head(n_open):
            print(_wrap(t, bullet="• "))
    if sents is None or sents.empty:
        return
    shares = sents["fin_label"].value_counts(normalize=True)
    print(f"\n  Sentiment: FinBERT tone {row['fin_net']:+.2f} ({row.get('tone', '')}) | "
          f"{len(sents)} sentences: {shares.get('positive', 0):.0%} positive, "
          f"{shares.get('neutral', 0):.0%} neutral, {shares.get('negative', 0):.0%} negative")
    sents = sents.assign(net=sents["p_pos"] - sents["p_neg"])
    pos = sents[sents["fin_label"] == "positive"].nlargest(3, "net")
    neg = sents[sents["fin_label"] == "negative"].nsmallest(3, "net")
    if len(pos):
        print("  Most positive sentences:")
        for t in pos["sentence"]:
            print(_wrap(t, bullet="+ "))
    if len(neg):
        print("  Most negative sentences:")
        for t in neg["sentence"]:
            print(_wrap(t, bullet="- "))


def _meeting_chart(bank, df, r, n_show=12):
    """Option 1 chart: how the chosen statement was scored, and its tone next to recent meetings."""
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 2]})
    shares = [r.get("fin_pos_share", np.nan), r.get("fin_neu_share", np.nan), r.get("fin_neg_share", np.nan)]
    labels = ["positive", "neutral", "negative"]
    bars = a1.barh(labels, [100 * x for x in shares], color=[BLUE, GREY, ORANGE])
    for b, x in zip(bars, shares):
        a1.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{100 * x:.0f}%", va="center", fontsize=9)
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel("share of sentences (%)")
    a1.set_title(f"FinBERT scores of the {BANK_NAME[bank]} text\n"
                 f"{r['date']:%d %b %Y}: tone {r['fin_net']:+.2f}", fontsize=10)

    i = df.index.get_loc(r.name)
    recent = df.iloc[max(0, i - n_show + 1): i + 1]
    x = np.arange(len(recent))
    cols = [BLUE if t > 0.05 else (ORANGE if t < -0.05 else GREY) for t in recent["fin_net"]]
    a2.bar(x, recent["fin_net"], color=cols, alpha=0.55)
    a2.bar(x[-1], recent["fin_net"].iloc[-1], color=cols[-1], edgecolor=INK2, linewidth=1.5)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("FinBERT tone (net)")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{d:%b %y}" for d in recent["date"]], rotation=45, ha="right", fontsize=8)
    ax2 = a2.twinx()
    ax2.plot(x, recent["rate"], color=INK2, lw=1.4, marker="o", ms=4, label="interest rate")
    ax2.grid(False)
    ax2.set_ylabel("interest rate (%)" if bank == "BoE" else "interest rate, upper bound (%)")
    ax2.legend(loc="upper center", fontsize=8, frameon=False)
    a2.set_title(f"Tone of the last {len(recent)} decisions (bars) and the interest rate", fontsize=10)
    fig.tight_layout()
    _show(fig, f"{bank.lower()}_{r['date']:%Y-%m-%d}_sentiment.png")


def meeting_view():
    """Option 1: choose a Fed or BoE meeting and see the decision, what was said and the sentiment."""
    bank = choose_bank()
    if bank is None:
        print("  Please type 1 or 2.")
        return
    df = decision_table(bank)
    if df is None:
        return
    chosen = select_meetings(df, multi=False)
    if chosen is None:
        return
    r = chosen.iloc[0]
    i = df.index.get_loc(chosen.index[0])
    prev = df.iloc[i - 1] if i > 0 else None
    print("\n" + "=" * 70)
    print(f"  {BANK_NAME[bank].upper()} - DECISION OF {r['date']:%d %B %Y}")
    print("=" * 70)
    move = {"raise": "RAISED", "cut": "CUT", "hold": "HELD"}[r["decision"]]
    was = f" (previously {prev['rate_text']})" if prev is not None and r["decision"] != "hold" else ""
    vote = f" | vote: {r['vote']}" if bank == "BoE" and r.get("vote") else ""
    print(f"  Decision: rate {move} - {r['rate_text']}{was}{vote}")
    vs = r["vs_previous"]
    vs = {"similar": "similar to the previous meeting", "-": "first meeting in the data"}.get(
        vs, f"{vs} than the previous meeting")
    print(f"  Stance: {r['stance']} (hawkish-dovish score {r['hawk_score']:+.2f}, {vs}; "
          f"method: {_method(df)})")
    print(f"  Key phrases: {r['key_words']}")

    if bank == "Fed":
        _print_document("FOMC statement (released on the decision day)", r, full=True)
        docs = load_documents()
        key = f"fomcminutes{r['date']:%Y%m%d}"
        mins = docs[(docs["bank"] == "Fed") & (docs["doc_type"] == "minutes")
                    & docs["url"].astype(str).str.contains(key)]
        if len(mins):
            m = mins.iloc[0].copy()
            m["doc_id"] = mins.index[0]
            m["tone"] = "positive" if m["fin_net"] > 0.05 else ("negative" if m["fin_net"] < -0.05 else "neutral")
            _print_document(f"FOMC minutes of this meeting (released {m['date']:%d %B %Y})", m)
        else:
            print("\n  (Minutes of this meeting not available.)")
    else:
        _print_document("Monetary Policy Summary and minutes (released with the decision)", r)

    prices, hourly = _load_prices()
    if prices is not None:
        rx = reaction(bank, r["date"], prices, hourly)
        parts = []
        if not np.isnan(rx["first_hour_pips"]):
            parts.append(f"first hour {rx['first_hour_pips']:+.0f} pips ({rx['first_hour_pct']:+.2f}%)")
        if rx["daily"] is not None:
            d = rx["daily"]
            parts += [f"decision day {d.iloc[0]:+.2f}%"] + [
                f"after {k} day{'s' if k > 1 else ''} {d.iloc[k]:+.2f}%" for k in (1, 5) if k < len(d)]
        if parts:
            print("\n  GBP/USD reaction: " + " | ".join(parts) + "   (more detail in option 3)")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n  Full text saved: results/menu/{bank.lower()}_{r['date']:%Y-%m-%d}.txt")
    (OUT / f"{bank.lower()}_{r['date']:%Y-%m-%d}.txt").write_text(str(r["text"]), encoding="utf-8")
    _meeting_chart(bank, df, r)


def reaction_view():
    """Option 3: choose meetings and see how GBP/USD reacted in the first hour and next days."""
    bank = choose_bank()
    if bank is None:
        print("  Please type 1 or 2.")
        return
    df = decision_table(bank)
    if df is None:
        return
    chosen = select_meetings(df, multi=True)
    if chosen is None or chosen.empty:
        return
    prices, hourly = _load_prices()
    if prices is None:
        return
    rows, paths = [], []
    for _, r in chosen.iterrows():
        rx = reaction(bank, r["date"], prices, hourly)
        d = rx["daily"]
        rows.append({"date": r["date"].date(), "decision": f"{r['decision']} {r['rate_text']}",
                     "stance": r["stance"],
                     "1st hour (pips)": rx["first_hour_pips"],
                     "day 0 %": d.iloc[0] if d is not None else np.nan,
                     "day 1 %": d.iloc[1] if d is not None and len(d) > 1 else np.nan,
                     "day 3 %": d.iloc[3] if d is not None and len(d) > 3 else np.nan,
                     "day 5 %": d.iloc[5] if d is not None and len(d) > 5 else np.nan,
                     "day 10 %": d.iloc[10] if d is not None and len(d) > 10 else np.nan})
        paths.append((r, rx))
    tab = pd.DataFrame(rows)
    print(f"\n  GBP/USD REACTION TO {BANK_NAME[bank].upper()} DECISIONS "
          "(moves from the price just before the announcement)\n")
    show = tab if len(tab) <= 25 else tab.tail(25)
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.2f}", na_rep="n/a"))
    if len(tab) > 25:
        print(f"  ... showing the last 25 of {len(tab)} meetings (all saved to CSV)")
    if len(tab) > 1:
        avg = tab.groupby("stance")[["1st hour (pips)", "day 1 %", "day 5 %", "day 10 %"]].mean()
        print("\n  Average by tone:")
        print(avg.to_string(float_format=lambda v: f"{v:+.2f}", na_rep="n/a"))
    if tab["1st hour (pips)"].isna().any() and hourly is not None:
        print(f"\n  n/a in the first-hour column: Yahoo hourly prices only start on "
              f"{hourly.index.min():%d %b %Y}.")
    print("  Intuition: a hawkish BoE should lift GBP (+); a hawkish Fed lifts USD, so GBP/USD falls (-).")
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / f"{bank.lower()}_reaction.csv", index=False)

    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    many = len(paths) > len(MEETING_COLOURS)
    for k, (r, rx) in enumerate(paths):
        if many:
            c, lw, lab = STANCE_COLOUR[r["stance"]], 0.6, None
        else:
            c, lw, lab = MEETING_COLOURS[k], 1.6, f"{r['date']:%Y-%m-%d} ({r['stance']})"
        if rx["hourly"] is not None:
            a1.plot(rx["hourly"].index, rx["hourly"].values, color=c, lw=lw, alpha=0.5 if many else 1,
                    marker=None if many else "o", ms=2.5, label=lab)
        if rx["daily"] is not None:
            a2.plot(rx["daily"].index, rx["daily"].values, color=c, lw=lw, alpha=0.5 if many else 1,
                    marker=None if many else "o", ms=3.5, label=lab)
    if many:                                          # bold average per tone
        for st, c in STANCE_COLOUR.items():
            dd = [rx["daily"] for r, rx in paths if r["stance"] == st and rx["daily"] is not None
                  and len(rx["daily"]) == 11]
            if dd:
                a2.plot(range(11), np.mean(dd, axis=0), color=c, lw=2.6, label=f"{st} average (n={len(dd)})")
            hh = [rx["hourly"].reindex(range(-3, 25)) for r, rx in paths
                  if r["stance"] == st and rx["hourly"] is not None]
            if hh:
                a1.plot(range(-3, 25), pd.concat(hh, axis=1).mean(axis=1).values, color=c, lw=2.6,
                        label=f"{st} average (n={len(hh)})")
    for ax in (a1, a2):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    a1.axvspan(0, 1, color=GRID, alpha=0.6, lw=0)
    a1.set_title("First hours (shaded = first hour)")
    a1.set_xlabel("hours after the announcement")
    a1.set_ylabel("GBP/USD move (pips)")
    a2.set_title("Following days")
    a2.set_xlabel("trading days after the announcement (0 = decision day)")
    a2.set_ylabel("GBP/USD move (%)")
    if not any(rx["hourly"] is not None for _, rx in paths):
        a1.text(0.5, 0.5, "no hourly data for these meetings\n(Yahoo keeps ~2 years)",
                transform=a1.transAxes, ha="center", va="center", color=MUTED)
    for ax in (a1, a2):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7.5, loc="best")
    fig.suptitle(f"GBP/USD after {BANK_NAME[bank]} decisions", y=1.02)
    _show(fig, f"{bank.lower()}_reaction.png")


# ====================================================================== options 4 + 5
CORE = {"naive": "Previous-day volatility (naive)", "price": "RF: price only",
        "sent": "RF: price + sentiment"}
LABEL = {"naive": "Previous-day value (baseline)", "price": "Model WITHOUT sentiment",
         "sent": "Model WITH sentiment"}


def _csv(name):
    p = config.RESULTS_DIR / name
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None


def _pct(logvol):
    return float(np.exp(logvol) * 100)


def _retrain(ans):
    try:
        import run_all
        run_all.run_pipeline("features" if ans == "retrain" else "prices")
    except Exception as exc:
        print(f"\n  ! Update failed ({type(exc).__name__}: {exc}) - using the saved models.")


def _choose_day():
    """Ask which day to predict. Returns a Timestamp, 'latest', or None."""
    print("\n  Which day do you want to predict?")
    print("    Enter       = the next trading day (live forecast)")
    print("    YYYY-MM-DD  = a past day (2019 onwards) to compare the prediction with what happened")
    print("    m           = pick a Fed or BoE meeting day")
    print("    update      = download the latest data and re-train first (10-15 min)")
    print("    retrain     = re-train on the saved data first (5-10 min)")
    a = _ask("  Your choice: ")
    if a in ("update", "retrain"):
        _retrain(a)
        return "latest"
    if a == "":
        return "latest"
    if a == "m":
        bank = choose_bank()
        df = decision_table(bank) if bank else None
        chosen = select_meetings(df, multi=False) if df is not None else None
        return None if chosen is None else pd.Timestamp(chosen["date"].iloc[0])
    try:
        return pd.Timestamp(a)
    except ValueError:
        print("  Not a valid date - use YYYY-MM-DD, e.g. 2024-08-01.")
        return None


def make_forecast(day):
    """Predictions for one target day from the baseline, the model without sentiment and
    the model with sentiment, plus what actually happened (if known).
    Past days use the walk-forward TEST predictions (each made only with earlier data)."""
    if not _need(config.PANEL_CSV):
        return None
    panel = pd.read_csv(config.PANEL_CSV, index_col="date", parse_dates=["date"])
    idx = panel.index
    if day == "latest":
        pos = len(idx)
    else:
        pos = int(idx.searchsorted(pd.Timestamp(day).normalize()))
        if pos == 0:
            print(f"  The data starts on {idx[0].date()} - choose a later day.")
            return None
        if pos >= len(idx) and pd.Timestamp(day) > idx[-1] + pd.offsets.BDay(3):
            print(f"  The model predicts one trading day ahead; the latest data is {idx[-1].date()}, "
                  "so the furthest day is the next trading day.")
            return None
    t = idx[pos - 1]                                  # forecast is made at the close of day t
    target = idx[pos] if pos < len(idx) else t + pd.offsets.BDay(1)
    row = panel.loc[[t]]
    fc = {"origin": t, "target": target, "row": row, "vol": {}, "up": {}, "actual_vol": None,
          "actual_ret": None, "mode": None}

    vp, dp = _csv("vol_predictions.csv"), _csv("dir_predictions.csv")
    if vp is not None and t in vp.index:
        fc["mode"] = "test"
        fc["vol"] = {k: _pct(vp.loc[t, m]) for k, m in CORE.items() if m in vp}
        fc["actual_vol"] = _pct(vp.loc[t, "y_true"])
        if dp is not None and t in dp.index:
            fc["up"] = {k: float(dp.loc[t, m]) for k, m in
                        {"freq": "Most frequent label", **CORE}.items() if m in dp}
    else:
        mp = config.MODEL_DIR / "full_data_models.joblib"
        if not _need(mp):
            return None
        import joblib
        bundle = joblib.load(mp)
        trained_until = pd.Timestamp(bundle.get("trained_until", "1900-01-01"))
        if t <= trained_until:
            first = vp.index.min().date() if vp is not None else config.FIRST_TEST_YEAR
            print(f"  {target.date()} is before the test period (starts {first}) - the model was "
                  "trained on those days, so it would not be a fair prediction. Choose a later day.")
            return None
        fc["mode"] = "live"
        fc["vol"]["naive"] = _pct(row["naive_vol"].iloc[0])
        for k in ("price", "sent"):
            if CORE[k] in bundle["volatility_models"]:
                fc["vol"][k] = _pct(bundle["volatility_models"][CORE[k]].predict(row)[0])
        dmods = bundle.get("direction_models")
        if dmods is None and (config.MODEL_DIR / "final_fold_models.joblib").exists():
            dmods = {m.name: m for m in joblib.load(config.MODEL_DIR / "final_fold_models.joblib")["direction"]}
        for k in ("price", "sent"):
            if dmods and CORE[k] in dmods:
                fc["up"][k] = float(dmods[CORE[k]].predict(row)[0])
        fc["up"]["freq"] = float(panel["y_dir"].dropna().iloc[:-config.DIR_HORIZON].mean() >= 0.5)
        if pos < len(idx) and not np.isnan(panel["y_vol"].loc[t]):
            fc["actual_vol"] = _pct(panel["y_vol"].loc[t])
    if "fwd_ret" in panel and not np.isnan(panel.loc[t, "fwd_ret"]):
        fc["actual_ret"] = float(panel.loc[t, "fwd_ret"])
    fc["dir_end"] = t + pd.offsets.BDay(config.DIR_HORIZON)
    return fc


def _event_text(row):
    banks = [BANK_NAME[b] for b in ("BoE", "Fed") if f"{b.lower()}_event_ahead" in row
             and row[f"{b.lower()}_event_ahead"].iloc[0] > 0]
    return " and ".join(banks) + " decision day" if banks else ""


def _forecast_chart(fc):
    if not config.PANEL_CSV.exists():
        return
    panel = pd.read_csv(config.PANEL_CSV, index_col="date", parse_dates=["date"])
    t, target = fc["origin"], fc["target"]
    hist = np.exp(panel["lvol_d"].loc[t - pd.Timedelta(days=90):t]) * 100
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(hist.index, hist, color=INK2, lw=1.1, marker="o", ms=2.2, label="daily volatility before the forecast")
    style = {"naive": (GREY, "o", 45), "price": (BLUE, "s", 60), "sent": (ORANGE, "D", 60)}
    for k, v in fc["vol"].items():
        c, mk, s = style[k]
        ax.scatter([target], [v], color=c, marker=mk, s=s, zorder=3, label=LABEL[k])
    if fc["actual_vol"]:
        ax.scatter([target], [fc["actual_vol"]], color="#1baf7a", marker="*", s=180, zorder=4, label="actual")
    ax.set_ylabel("annualised volatility (%)")
    ax.set_title(f"GBP/USD volatility prediction for {target:%d %b %Y}")
    ax.legend(fontsize=8, loc="upper left")
    _show(fig, f"prediction_{target:%Y-%m-%d}.png")


def _latest_doc(docs, bank, t):
    from src.features import index_docs
    d, kind = index_docs(docs, bank)
    d = d[d["date"] <= t]
    if d.empty:
        return None, kind
    return d.iloc[-1], kind


def _sentiment_summary(fc):
    """Step 1: what the Fed and BoE had said by the time the prediction is made."""
    t, row = fc["origin"], fc["row"]
    panel = pd.read_csv(config.PANEL_CSV, index_col="date", parse_dates=["date"])
    docs = load_documents()
    print(f"\n  STEP 1 - CENTRAL BANK SENTIMENT (what had been said by {t:%d %b %Y})")
    cols = {}
    for bank in ("BoE", "Fed"):
        p = bank.lower()
        doc, kind = _latest_doc(docs, bank, t) if docs is not None else (None, "minutes")
        hawk = row[f"{p}_hawk_score"].iloc[0]
        lo, hi = panel[f"{p}_hawk_score"].quantile([1 / 3, 2 / 3])
        tone = row[f"{p}_fin_net"].iloc[0]
        chg = row[f"{p}_hawk_chg"].iloc[0]
        cols[bank] = {
            "Latest document": "-" if doc is None else f"{kind} {doc['date']:%d %b %Y}",
            "Tone (FinBERT)": f"{tone:+.2f} {'positive' if tone > 0.05 else 'negative' if tone < -0.05 else 'neutral'}",
            "Stance": f"{'HAWKISH' if hawk >= hi else 'DOVISH' if hawk <= lo else 'NEUTRAL'} ({hawk:+.2f})",
            "vs previous document": "more hawkish" if chg > 0.02 else ("more dovish" if chg < -0.02 else "similar"),
            "Decision on forecast day": "YES" if row[f"{p}_event_ahead"].iloc[0] > 0 else "no",
            "_doc": doc,
        }
    print(f"\n    {'':26s}{'Bank of England':30s}Federal Reserve")
    for k in ["Latest document", "Tone (FinBERT)", "Stance", "vs previous document", "Decision on forecast day"]:
        print(f"    {k:26s}{cols['BoE'][k]:30s}{cols['Fed'][k]}")
    for bank in ("BoE", "Fed"):
        doc = cols[bank]["_doc"]
        sents = None if doc is None else _sentences(doc.name)
        if sents is None or sents.empty:
            continue
        sents = sents.assign(net=sents["p_pos"] - sents["p_neg"])
        print(f"\n    Key sentences ({BANK_NAME[bank]}):")
        for pick, mark in ((sents.nlargest(1, "net"), "+ "), (sents.nsmallest(1, "net"), "- ")):
            for s in pick["sentence"]:
                print(_wrap(_short(s, 240), indent="      ", bullet=mark))
    gap = row["div_hawk"].iloc[0] - panel["div_hawk"].median()
    print("\n    Summary: " + ("the BoE sounds more hawkish than the Fed -> points to GBP/USD UP"
                              if gap > 0 else "the Fed sounds more hawkish than the BoE -> points to GBP/USD DOWN"))


def _prediction(fc):
    """Step 2: the prediction, with the baseline and the model without sentiment for comparison."""
    t, target = fc["origin"], fc["target"]
    print(f"\n  STEP 2 - PREDICTION FOR {target:%a %d %b %Y}")
    print("    " + (f"(made with data up to {t:%d %b %Y} only - the models never saw this day)"
                    if fc["mode"] == "test" else f"(live forecast from the latest data, {t:%d %b %Y})"))
    if fc["mode"] == "live" and (pd.Timestamp.today().normalize() - t).days > 4:
        print("    ! The data is a few days old - type 'update' next time for a fresh forecast.")
    act = fc["actual_vol"]
    print(f"\n    How much will it move? (volatility %)  {'predicted':>9s}" + (f"{'actual':>9s}" if act else ""))
    for k in ("naive", "price", "sent"):
        if k in fc["vol"]:
            print(f"      {LABEL[k]:36s}{fc['vol'][k]:8.1f}%" + (f"{act:8.1f}%" if act else ""))
    ret = fc["actual_ret"]
    print(f"\n    Which way over the next {config.DIR_HORIZON} days? (to {fc['dir_end']:%d %b})")
    names = {"freq": "Most frequent label (baseline)", "price": LABEL["price"], "sent": LABEL["sent"]}
    for k in ("freq", "price", "sent"):
        if k in fc["up"]:
            p = fc["up"][k]
            conf = "" if k == "freq" else f" ({max(p, 1 - p):.0%} sure)"
            tick = "" if ret is None else ("  correct" if (p >= 0.5) == (ret > 0) else "  wrong")
            print(f"      {names[k]:36s}{'UP' if p >= 0.5 else 'DOWN':5s}{conf:12s}{tick}")
    if ret is not None:
        print(f"      {'What actually happened':36s}{'UP' if ret > 0 else 'DOWN':5s}({100 * (np.exp(ret) - 1):+.2f}%)")


def _decision(fc):
    """Step 3: did the sentiment change the decision, and does it help in general?"""
    v, u = fc["vol"], fc["up"]
    print("\n  STEP 3 - DECISION")
    if "sent" in v and "sent" in u:
        level = fc["vol"]["sent"]
        panel = pd.read_csv(config.PANEL_CSV, index_col="date", parse_dates=["date"])
        avg = float((np.exp(panel["lvol_d"]) * 100).mean())
        calm = "a MORE volatile day than usual" if level > avg * 1.1 else (
            "a CALMER day than usual" if level < avg * 0.9 else "a normal day")
        print(f"    Expect {calm} ({level:.1f}% vs average {avg:.1f}%), "
              f"GBP/USD more likely {'UP' if u['sent'] >= 0.5 else 'DOWN'} over the next {config.DIR_HORIZON} days.")
    if "price" in v and "sent" in v:
        d = v["sent"] - v["price"]
        print(f"    Effect of the sentiment: volatility {v['price']:.1f}% -> {v['sent']:.1f}% ({d:+.1f} points)"
              + (f", chance of UP {u['price']:.0%} -> {u['sent']:.0%}" if "price" in u and "sent" in u else ""))
    core, dm = _csv("core_experiments.csv"), _csv("dir_metrics_all_days.csv")
    if core is not None and CORE["sent"] in core.index:
        a, b = core.loc[CORE["price"], "MAE_vol_pts"], core.loc[CORE["sent"], "MAE_vol_pts"]
        n = core.loc[CORE["naive"], "MAE_vol_pts"]
        pval, stat = core.loc[CORE["sent"], "DM_p"], core.loc[CORE["sent"], "DM_sent_vs_price"]
        verdict = ("made no significant difference" if pval >= 0.05 else
                   "made forecasts significantly " + ("BETTER" if stat < 0 else "WORSE"))
        pv = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
        print(f"\n    Track record 2019 onwards - typical volatility error: baseline ±{n:.1f}, "
              f"without sentiment ±{a:.1f}, with sentiment ±{b:.1f} points")
        if dm is not None:
            acc = [(nm, dm.loc[m, "Accuracy"]) for nm, m in [("coin flip", "Coin flip"),
                   ("most frequent", "Most frequent label"), ("without sentiment", CORE["price"]),
                   ("with sentiment", CORE["sent"])] if m in dm.index]
            print("    Direction accuracy: " + ", ".join(f"{nm} {x:.0%}" for nm, x in acc))
        print(f"    Answer: adding central bank sentiment {verdict} for next-day volatility "
              f"(Diebold-Mariano {pv}).")


def decision_view():
    """Option 4: central bank sentiment summary -> GBP/USD prediction -> decision."""
    day = _choose_day()
    if day is None:
        return
    fc = make_forecast(day)
    if fc is None:
        return
    ev = _event_text(fc["row"])
    print("\n" + "=" * 74)
    print(f"  SENTIMENT -> PREDICTION FOR {fc['target']:%a %d %b %Y}" + (f"  ({ev})" if ev else ""))
    print("=" * 74)
    _sentiment_summary(fc)
    _prediction(fc)
    _decision(fc)
    _forecast_chart(fc)


def main():
    """menu.py is the library behind main.py - run main.py instead."""
    import main as app
    app.main()


if __name__ == "__main__":
    main()
