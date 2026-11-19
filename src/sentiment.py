"""
Steps 3-5 - from raw central bank documents to document-level sentiment indices.

    Step 3  Split every document into sentences with NLTK's Punkt tokenizer
            (Kiss & Strunk, 2006)                       -> data/cb_sentences.csv
    Step 4  Classify every sentence as positive / neutral / negative with
            FinBERT-FOMC (Chen et al., 2023), a FinBERT (Araci, 2019) model fine-tuned
            on FOMC minutes, i.e. adapted to central bank language. As a second,
            policy-specific view each sentence is also scored hawkish / dovish /
            neutral with FOMC-RoBERTa (Shah, Paturi & Chava, 2023).
                                                        -> data/cb_sentence_scores.csv
    Step 5  Aggregate sentence scores into document-level sentiment indices
                                                        -> data/cb_sentiment.csv

Document-level indices (one row per document):
    fin_net      = (#positive - #negative) / #sentences            in [-1, 1]
    fin_prob_net = mean over sentences of P(positive) - P(negative) in [-1, 1]
    fin_pos_share, fin_neg_share, fin_neu_share  = label shares
    hawk_score   = mean over sentences of P(hawkish) - P(dovish)    in [-1, 1]
    dict_*       = transparent terminology counts per 1,000 words (no ML)

Both transformer models were trained on Federal Reserve text; applying them to the Bank
of England is a cross-institution transfer and one of this project's extensions.
"""
import re

import numpy as np
import pandas as pd

import config

# ====================================================================== dictionary
HAWKISH_TERMS = [
    r"rais\w* (?:bank rate|interest rates?|the (?:federal funds )?(?:rate|target range))",
    r"increas\w* (?:bank rate|interest rates?|the target range)",
    r"tighten\w*", r"restrictive", r"inflationary pressures?", r"upside risks?",
    r"persistent inflation", r"inflation persistence", r"second[- ]round effects?",
    r"overheat\w*", r"tight labou?r market", r"wage growth (?:remains|has been) (?:strong|elevated)",
    r"further (?:increases|tightening)", r"elevated inflation", r"price pressures?",
]
DOVISH_TERMS = [
    r"cut\w* (?:bank rate|interest rates?|the (?:federal funds )?(?:rate|target range))",
    r"(?:lower\w*|reduc\w*) (?:bank rate|interest rates?|the target range)",
    r"loosen\w*", r"eas\w* (?:of )?(?:monetary )?policy", r"accommodat\w*",
    r"downside risks?", r"weak\w* (?:demand|growth|activity)", r"slack", r"recession\w*",
    r"disinflation\w*", r"stimul\w*", r"asset purchases?", r"quantitative easing",
    r"unemployment (?:has )?(?:risen|increased|rising)", r"subdued",
]
UNCERTAINTY_TERMS = [
    r"uncertain\w*", r"volatil\w*", r"unclear", r"risks?", r"could", r"might",
    r"depend\w* on", r"data[- ]dependent", r"vigilant", r"monitor\w* closely",
]
KEY_TERMS = {
    "kw_inflation": r"\binflation\b", "kw_growth": r"\bgrowth\b",
    "kw_labour": r"\blabou?r\b", "kw_energy": r"\benergy\b",
    "kw_brexit_trade": r"\b(?:brexit|tariffs?|trade)\b",
}


def _rate(patterns, text, n_words):
    hits = sum(len(re.findall(p, text)) for p in patterns)
    return 1000.0 * hits / max(n_words, 1)


def dictionary_scores(text):
    t = text.lower()
    n = len(t.split())
    hawk = _rate(HAWKISH_TERMS, t, n)
    dove = _rate(DOVISH_TERMS, t, n)
    out = {
        "dict_hawk": hawk,
        "dict_dove": dove,
        "dict_net": (hawk - dove) / (hawk + dove + 1e-6),
        "dict_uncert": _rate(UNCERTAINTY_TERMS, t, n),
        "n_words": n,
    }
    for name, pat in KEY_TERMS.items():
        out[name] = _rate([pat], t, n)
    return out


# ====================================================================== step 3: sentences
def _regex_split(text):
    return re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9\"'(])", text)


def get_sentence_splitter():
    """NLTK Punkt sentence tokenizer; downloads its model on first use.
    Falls back to a regular-expression splitter only if NLTK data cannot be downloaded."""
    try:
        import nltk
    except ImportError:
        print("   ! NLTK is not installed (pip install nltk) - using a regex splitter instead")
        return _regex_split, "regex"
    try:
        nltk.sent_tokenize("Test one. Test two.")
    except LookupError:
        print("   downloading NLTK Punkt sentence model (one-off)...")
        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        try:
            nltk.sent_tokenize("Test one. Test two.")
        except LookupError:
            print("   ! could not download NLTK Punkt data - using a regex splitter instead")
            return _regex_split, "regex"
    return nltk.sent_tokenize, "nltk"


def clean_sentence(s):
    s = re.sub(r"^\d+\s*[.)]\s*", "", s.strip())      # drop paragraph numbers like "12."
    return re.sub(r"\s+", " ", s)


def sentiment_focus(s):
    """Sentiment Focus pre-processing (Goessi et al., 2023; used by Hilscher et al., 2026):
    in contrast sentences keep only the clause that carries the sentiment.
      "Although X, Y."   -> "Y."      (main clause after the concessive clause)
      "X, but Y."        -> "Y."      (clause after the last 'but')
      "X, although Y."   -> "X."      (main clause before 'although / though')
    The paper's extra comma rule needs a dependency parser and is not reproduced."""
    t = s.strip()
    m = re.match(r"^(?:although|though|even though)\b[^,]*,\s*(.+)$", t, flags=re.I)
    if m and len(m.group(1).split()) >= 3:
        t = m.group(1)
    parts = re.split(r",?\s+\bbut\b\s+", t, flags=re.I)
    if len(parts) > 1 and len(parts[-1].split()) >= 3:
        t = parts[-1]
    m = re.match(r"^(.+?),?\s+\b(?:although|though|even though)\b\s+.+$", t, flags=re.I)
    if m and len(m.group(1).split()) >= 3:
        t = m.group(1)
    t = t[0].upper() + t[1:] if t else t
    return t


_FALLBACK_STOPWORDS = set("""a an the and or of to in on at for by with from as is are was were be
been being this that these those it its which who whom whose there their they them he she his her
we our us you your i me my mine so than then too very can will just do does did has have had
having into over under again further once here when where why how all any both each few more
most other some such own same s t don should now""".split())
_KEEP_WORDS = {"not", "no", "nor", "against", "but", "below", "above", "up", "down",
               "off", "out", "more", "most", "very", "further"}   # negation / direction words


def get_stopwords():
    try:
        import nltk
        try:
            words = set(nltk.corpus.stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(nltk.corpus.stopwords.words("english"))
    except Exception:
        words = _FALLBACK_STOPWORDS
    # Keep negation and direction words: dropping "not" would flip a sentence's meaning.
    return words - _KEEP_WORDS


def model_text(s, stopwords):
    """Text actually passed to FinBERT, following the paper's pre-processing."""
    if config.SENTIMENT_FOCUS:
        s = sentiment_focus(s)
    if config.REMOVE_STOPWORDS:
        tokens = re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?%?", s)
        s = " ".join(w for w in tokens if w.lower() not in stopwords)
    return s


def split_documents(texts, demo=False, min_words=5):
    print("=== Step 3: Splitting documents into sentences using NLTK ===")
    splitter, method = get_sentence_splitter()
    stopwords = get_stopwords() if config.REMOVE_STOPWORDS else set()
    rows = []
    for doc_id, r in texts.reset_index(drop=True).iterrows():
        for i, s in enumerate(splitter(r["text"])):
            s = clean_sentence(s)
            if len(s.split()) < min_words:
                continue
            mt = model_text(s, stopwords)
            if not mt:
                continue
            rows.append({"doc_id": doc_id, "date": r["date"], "bank": r["bank"],
                         "doc_type": r["doc_type"], "sent_id": i, "sentence": s,
                         "model_text": mt})
    sents = pd.DataFrame(rows, columns=["doc_id", "date", "bank", "doc_type", "sent_id",
                                        "sentence", "model_text"])
    print(f"[sentences] pre-processing: sentiment focus={config.SENTIMENT_FOCUS}, "
          f"stop-word removal={config.REMOVE_STOPWORDS} (negation words kept)")
    sents.to_csv(config.SENTENCES_CSV, index=False)
    per_doc = sents.groupby("doc_id").size()
    print(f"[sentences] tokenizer: {method} | {len(sents):,} sentences from "
          f"{sents['doc_id'].nunique()} documents (median {per_doc.median():.0f} per document)")
    for bank in ["BoE", "Fed"]:
        n = int((sents["bank"] == bank).sum())
        print(f"            {bank}: {n:,} sentences")
    return sents


def load_sentences():
    return pd.read_csv(config.SENTENCES_CSV, parse_dates=["date"])


# ====================================================================== step 4: classification
class TransformerScorer:
    """Sentence classifier returning class probabilities (runs on GPU/Apple MPS if available)."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            from transformers import BertTokenizer
            self.tok = BertTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        print(f"[model] loaded {model_name} on {self.device} | labels {self.id2label}")

    def probs(self, sentences, show_progress=False):
        out = []
        bs = config.SENT_BATCH_SIZE
        with self.torch.no_grad():
            for i in range(0, len(sentences), bs):
                enc = self.tok(list(sentences[i:i + bs]), padding=True, truncation=True,
                               max_length=config.SENT_MAX_TOKENS, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
                done = min(i + bs, len(sentences))
                if show_progress and (done // bs) % 50 == 0:
                    print(f"   {done:,}/{len(sentences):,} sentences")
        return np.vstack(out) if out else np.zeros((0, len(self.id2label)))


def _check_mapping(scorer, names, defaults, test_sentences):
    """Find which output column is which class. Use label names when the model has them;
    otherwise use the documented order and confirm it with clear-cut test sentences."""
    found = {}
    for key, frag in names.items():
        hits = [k for k, v in scorer.id2label.items() if frag in v]
        if hits:
            found[key] = hits[0]
    if len(found) == len(names):
        return found
    mapping = dict(defaults)
    p = scorer.probs(test_sentences)
    a, b = list(names)[:2]                  # first two classes, e.g. positive / negative
    if p[0, mapping[a]] - p[0, mapping[b]] < p[1, mapping[a]] - p[1, mapping[b]]:
        print(f"   sanity check: swapping '{a}' and '{b}' columns")
        mapping[a], mapping[b] = mapping[b], mapping[a]
    return mapping


def finbert_mapping(scorer):
    return _check_mapping(
        scorer, {"positive": "pos", "negative": "neg", "neutral": "neu"},
        {"positive": 0, "negative": 1, "neutral": 2},
        ["Economic activity expanded strongly and the labour market improved further.",
         "Economic activity contracted sharply and unemployment rose significantly."])


def hawkish_mapping(scorer):
    # model card: LABEL_0 = dovish, LABEL_1 = hawkish, LABEL_2 = neutral
    return _check_mapping(
        scorer, {"hawkish": "hawk", "dovish": "dov", "neutral": "neu"},
        {"hawkish": 1, "dovish": 0, "neutral": 2},
        ["The Committee decided to raise the target range for the federal funds rate "
         "to combat persistently high inflation.",
         "The Committee decided to lower the target range for the federal funds rate "
         "because economic activity has weakened sharply."])


def dictionary_sentence_stance(sentences):
    """Hawkish / dovish / neutral per sentence from the terminology lists above:
    more hawkish than dovish phrases -> hawkish, the reverse -> dovish, else neutral.
    Returned as 0/1 'probabilities' so it plugs into the same aggregation."""
    out = np.zeros((len(sentences), 3))
    for i, sent in enumerate(sentences):
        t = str(sent).lower()
        nh = sum(len(re.findall(p, t)) for p in HAWKISH_TERMS)
        nd = sum(len(re.findall(p, t)) for p in DOVISH_TERMS)
        out[i, 0 if nh > nd else 1 if nd > nh else 2] = 1.0
    return out


def _read_cache(path, cols):
    """Saved sentence scores as a DataFrame indexed by sentence text (empty if none)."""
    if path.exists():
        try:
            c = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"model_text": str})
            c = c.dropna(subset=cols).drop_duplicates("model_text", keep="last")
            return c.set_index("model_text")[cols].astype(float)
        except Exception:
            pass
    return pd.DataFrame(columns=cols, dtype=float)


def _write_cache(path, cache):
    cache.rename_axis("model_text").reset_index().to_csv(path, index=False)


def classify_sentences(sents, demo=False):
    print("=== Step 4: Processing sentences with FinBERT-FOMC "
          "(FinBERT adapted to central bank communications) ===")
    print("    Classifying sentiment as positive, neutral or negative")
    s = sents.copy()
    if demo:
        rng = np.random.default_rng(config.RANDOM_SEED)
        low = s["sentence"].str.lower()
        pos_hint = low.str.contains("rais|upside|tight|persistent").astype(float)
        neg_hint = low.str.contains("cut|weak|downside|risen|disinflation").astype(float)
        logits = np.column_stack([1.5 * pos_hint, 1.5 * neg_hint, np.full(len(s), 0.8)])
        logits += rng.normal(0, 0.5, logits.shape)
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        s["p_pos"], s["p_neg"], s["p_neu"] = p[:, 0], p[:, 1], p[:, 2]
        s["p_hawk"], s["p_dove"], s["p_hneu"] = p[:, 0], p[:, 1], p[:, 2]
        s["hawk_method"] = "demo"
    else:
        # ---- FinBERT-FOMC: saved scores are reused; only NEW sentences are scored
        cols = ["p_pos", "p_neg", "p_neu"]
        cache = _read_cache(config.FINBERT_CACHE_CSV, cols)
        todo = [t for t in pd.unique(s["model_text"]) if t not in cache.index]
        if todo:
            print(f"    {len(todo):,} sentences to score "
                  f"({len(cache):,} already saved in {config.FINBERT_CACHE_CSV.name})")
            fin = TransformerScorer(config.FINBERT_MODEL)
            m = finbert_mapping(fin)
            raw = fin.probs(todo, show_progress=True)
            new = pd.DataFrame(np.column_stack([raw[:, m["positive"]], raw[:, m["negative"]],
                                                raw[:, m["neutral"]]]), columns=cols, index=todo)
            cache = pd.concat([cache, new])
            _write_cache(config.FINBERT_CACHE_CSV, cache)
            del fin
        else:
            print("    all sentences already scored - using saved FinBERT-FOMC scores")
        p = cache.reindex(s["model_text"].values)[cols].values
        s["p_pos"], s["p_neg"], s["p_neu"] = p[:, 0], p[:, 1], p[:, 2]

        # ---- extension: hawkish / dovish / neutral
        print("    Second view: hawkish / dovish / neutral")
        h, method = None, None
        try:
            hk = TransformerScorer(config.HAWKISH_MODEL)
            mm = hawkish_mapping(hk)
            raw = hk.probs(s["sentence"].tolist(), show_progress=True)
            h = np.column_stack([raw[:, mm["hawkish"]], raw[:, mm["dovish"]], raw[:, mm["neutral"]]])
            method = "FOMC-RoBERTa"
        except Exception as exc:
            print(f"    ! {config.HAWKISH_MODEL} not available ({type(exc).__name__}: it is a gated "
                  "model that needs a Hugging Face login).")
            print("      Using the hawkish/dovish terminology dictionary instead (see README).")
        if h is None:
            h = dictionary_sentence_stance(s["sentence"])
            method = "dictionary"
        s["p_hawk"], s["p_dove"], s["p_hneu"] = h[:, 0], h[:, 1], h[:, 2]
        s["hawk_method"] = method
    s["fin_label"] = np.array(["positive", "negative", "neutral"])[
        np.argmax(s[["p_pos", "p_neg", "p_neu"]].values, axis=1)]
    s["hawk_label"] = np.array(["hawkish", "dovish", "neutral"])[
        np.argmax(s[["p_hawk", "p_dove", "p_hneu"]].values, axis=1)]
    s.to_csv(config.SENTENCE_SCORES_CSV, index=False)
    shares = s.groupby("bank")["fin_label"].value_counts(normalize=True).unstack().round(3)
    print("[classify] share of sentences per label:")
    print(shares.to_string())
    return s


# ====================================================================== step 5: aggregation
def aggregate_documents(texts, scored):
    print("=== Step 5: Aggregating sentence scores into document-level sentiment indices ===")
    s = scored.copy()
    # sentence score as in Hilscher et al. (2026): +1 positive, 0 neutral, -1 negative
    s["score"] = s["fin_label"].map({"positive": 1, "neutral": 0, "negative": -1})
    s["is_pos"] = (s["fin_label"] == "positive").astype(float)
    s["is_neg"] = (s["fin_label"] == "negative").astype(float)
    s["is_neu"] = (s["fin_label"] == "neutral").astype(float)
    s["prob_net"] = s["p_pos"] - s["p_neg"]
    s["hawk_net"] = s["p_hawk"] - s["p_dove"]
    g = s.groupby("doc_id")
    idx = pd.DataFrame({
        "n_sentences": g.size(),
        "fin_net": g["score"].mean(),            # the paper's document-level index
        "fin_pos_share": g["is_pos"].mean(),
        "fin_neg_share": g["is_neg"].mean(),
        "fin_neu_share": g["is_neu"].mean(),
        "fin_prob_net": g["prob_net"].mean(),
        "hawk_score": g["hawk_net"].mean(),
        "hawk_method": g["hawk_method"].first() if "hawk_method" in s else "FOMC-RoBERTa",
    })

    docs = texts.reset_index(drop=True)
    dict_feats = pd.DataFrame([dictionary_scores(t) for t in docs["text"]])
    out = pd.concat([docs[["date", "bank", "doc_type"]], dict_feats], axis=1)
    out = out.join(idx, how="left")
    fill = {"n_sentences": 0, "fin_pos_share": 0, "fin_neg_share": 0, "fin_neu_share": 1,
            "fin_prob_net": 0, "hawk_score": 0, "fin_net": 0, "hawk_method": ""}
    out = out.fillna(value=fill)
    out.to_csv(config.SENTIMENT_CSV, index=False)
    summary = (out.groupby(["bank", "doc_type"])[["fin_net", "hawk_score"]]
               .agg(["mean", "std", "min", "max"]).round(3))
    print(f"[indices] {len(out)} documents -> {config.SENTIMENT_CSV.name}")
    print(summary.to_string())
    return out


def build_sentiment_file(texts, demo=False, sentences=None):
    """Steps 4 + 5 (step 3 output can be passed in, otherwise it is loaded from disk)."""
    if sentences is None:
        sentences = load_sentences()
    scored = classify_sentences(sentences, demo=demo)
    return aggregate_documents(texts, scored)


def load_sentiment():
    return pd.read_csv(config.SENTIMENT_CSV, parse_dates=["date"])
