"""
Step 2 - collecting Federal Reserve and Bank of England communications.

Hilscher, Nabors & Raviv (2026) build their index from FOMC minutes (and ECB press
conferences). For GBP/USD the relevant pair of central banks is the Fed and the BoE:

* Federal Reserve - FOMC minutes, released three weeks after each meeting
  (doc_type "minutes", dated by RELEASE day, so the model never sees them early), plus
  the FOMC statements released on the decision day (doc_type "statement", used for
  announcement dates). Meetings are found from the yearly index
  https://www.federalreserve.gov/newsevents/pressreleases/<year>-press-fomc.htm
* Bank of England - "Monetary Policy Summary and minutes", published together at 12:00
  UK time on MPC announcement days (combined format since August 2015)
  https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/<year>/<month>-<year>

* BIS central bankers' speeches - free bulk download for research, collected for both
  banks and used for a bank's index whenever its website blocks the download.

Every downloaded page is cached in data/raw/ so re-running never re-downloads.
If a page cannot be scraped (website changes, blocking), you can add the text by hand:
    data/texts_manual/boe/YYYY-MM-DD.txt   or   data/texts_manual/fed/YYYY-MM-DD.txt
where the date is the publication (release) date. Manual files are treated as minutes
and override scraped ones.
"""
import re
import time
from datetime import date, datetime

import pandas as pd

import config

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

BOE_URL = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/{y}/{m}-{y}"
FED_INDEX_URL = "https://www.federalreserve.gov/newsevents/pressreleases/{y}-press-fomc.htm"
FED_BASE = "https://www.federalreserve.gov"


# ------------------------------------------------------------------ helpers
# Some sites (notably the Bank of England) answer 403 Forbidden to requests that look like
# scripts. We send ordinary browser headers and, if still refused, retry with curl_cffi
# (installed with yfinance), which reproduces a real browser's connection fingerprint.
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}
BLOCKED = []          # URLs that stayed blocked, reported at the end of step 2


def _session():
    import requests
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s


def _fetch(session, url):
    """Return (status_code, text) trying plain requests first, then curl_cffi."""
    status, text = None, ""
    try:
        r = session.get(url, timeout=30)
        status, text = r.status_code, r.text
    except Exception as exc:
        print(f"   ! request failed for {url}: {type(exc).__name__}")
    if status in (200, 404):
        return status, text
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(url, impersonate="chrome", timeout=30,
                              headers={"Accept-Language": "en-GB,en;q=0.9"})
        if r.status_code in (200, 404):
            return r.status_code, r.text
        status = r.status_code
    except ImportError:
        pass
    except Exception as exc:
        print(f"   ! browser-style retry failed for {url}: {type(exc).__name__}")
    return status, text


def _get_cached(session, url, cache_name):
    """Return page HTML (or None for 404 / blocked), caching successful pages on disk.
    Blocked pages are NOT cached, so simply re-running step 2 tries them again."""
    cache = config.RAW_DIR / cache_name
    if cache.exists():
        txt = cache.read_text(encoding="utf-8")
        return None if txt == "__404__" else txt
    time.sleep(config.REQUEST_PAUSE_SEC)
    status, text = _fetch(session, url)
    if status == 404:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("__404__", encoding="utf-8")
        return None
    if status != 200:
        BLOCKED.append((status, url))
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def _report_blocked():
    if not BLOCKED:
        return
    by_site = {}
    for status, url in BLOCKED:
        site = url.split("/")[2]
        by_site.setdefault(site, []).append(status)
    for site, codes in by_site.items():
        print(f"   ! {len(codes)} pages from {site} could not be downloaded "
              f"(HTTP {sorted(set(c for c in codes if c is not None)) or 'no response'}). "
              "For the BoE this count includes months without an MPC meeting, which is normal.")
    print("     Re-running the step retries them (successful pages are cached). If a site keeps "
          "blocking, try another network (e.g. phone hotspot) or add texts manually "
          f"in {config.MANUAL_TEXT_DIR}.")


def _page_text(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip(), soup


def _clean_body(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------------------------------------ Bank of England
def parse_boe_page(html):
    """Return (publication_date, text) or (None, None) if the page cannot be parsed."""
    text, _ = _page_text(html)
    m = re.search(r"Published on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", text)
    if not m:
        return None, None
    pub = datetime.strptime(m.group(1), "%d %B %Y").date()
    after = text[m.end():]

    start = after.find("Monetary Policy Summary")
    body = after[start:] if start >= 0 else after
    for end_marker in ["Other Monetary Policy Committee news", "Other MPC news",
                       "Follow us", "Our use of cookies"]:
        end = body.find(end_marker)
        if end > 0:
            body = body[:end]
            break
    body = _clean_body(body)
    if len(body) < 500:
        return pub, None
    return pub, body


def scrape_boe(first_year, last_year):
    s = _session()
    rows = []
    for y in range(first_year, last_year + 1):
        found = 0
        for m in MONTHS:
            html = _get_cached(s, BOE_URL.format(y=y, m=m), f"boe/{y}-{m}.html")
            if html is None:
                continue
            pub, body = parse_boe_page(html)
            if body is None:
                print(f"   ! could not extract BoE text for {m} {y}")
                continue
            rows.append({"date": pub, "bank": "BoE", "doc_type": "minutes",
                         "url": BOE_URL.format(y=y, m=m), "text": body})
            found += 1
        print(f"[texts] BoE {y}: {found} documents")
    return rows


# ------------------------------------------------------------------ Federal Reserve
def fed_statement_links(session, year):
    from bs4 import BeautifulSoup
    html = _get_cached(session, FED_INDEX_URL.format(y=year), f"fed/index-{year}.html")
    if html is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        title = a.get_text(" ", strip=True).lower()
        href = a["href"]
        if (re.search(r"monetary\d{8}a\.htm$", href) and "fomc statement" in title
                and "longer-run" not in title):
            links.add(href if href.startswith("http") else FED_BASE + href)
    return sorted(links)


def parse_fed_page(html):
    text, _ = _page_text(html)
    start = None
    for marker in ["For release at", "For immediate release", "For Release at"]:
        i = text.find(marker)
        if i >= 0:
            nl = text.find("\n", i)          # skip the rest of that line
            start = nl if nl > 0 else i
            break
    body = text[start:] if start is not None else text
    for end_marker in ["Voting for the", "Voting for this action", "For media inquiries",
                       "Implementation Note", "Last Update"]:
        e = body.find(end_marker)
        if e > 0:
            body = body[:e]
            break
    body = re.sub(r"^(?:Share\s+)+", "", _clean_body(body))
    return body if len(body) > 300 else None


FED_MINUTES_URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes{d}.htm"


def parse_fed_minutes(html, meeting_date):
    """Return (release_date, text). The minutes page carries no release-date line, but the
    'Last Update' stamp equals the release date unless the page was corrected later, so it
    is only trusted when it falls 2-5 weeks after the meeting (otherwise: meeting + 21 days)."""
    text, _ = _page_text(html)
    release = meeting_date + pd.Timedelta(days=config.FED_MINUTES_LAG_DAYS)
    m = re.search(r"Last Update:\s*([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    if m:
        lu = pd.Timestamp(datetime.strptime(m.group(1), "%B %d, %Y"))
        if pd.Timedelta(days=14) <= lu - pd.Timestamp(meeting_date) <= pd.Timedelta(days=35):
            release = lu
    start = text.find("Minutes of the Federal Open Market Committee")
    body = text[start:] if start >= 0 else text
    end = len(body)
    for marker in ["Notation Vote", "_____", "Last Update", "[Back to Top]"]:
        e = body.find(marker, 2000)
        if 0 < e < end:
            end = e
    body = _clean_body(body[:end])
    return pd.Timestamp(release).date(), (body if len(body) > 2000 else None)


FED_CALENDAR_PAGES = [("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                       "fed/calendar-current.html")]
FED_HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
MIN_FED_MEETINGS_PER_YEAR = 6         # a normal year has 8 scheduled FOMC meetings


def _dates_in(html):
    """Meeting dates linked from a Fed page (minutes or statement links)."""
    out = set()
    for m in re.finditer(r"fomcminutes(\d{8})\.htm|monetary(\d{8})a\.htm", html or ""):
        try:
            out.add(datetime.strptime(m.group(1) or m.group(2), "%Y%m%d").date())
        except ValueError:
            pass
    return out


def fed_meeting_dates(session, first_year, last_year):
    """FOMC meeting dates from three sources, so that older years are not missed:
    1. the yearly press-release index pages (recent years),
    2. the FOMC calendar page and the historical calendar pages (older years),
    3. for a year still incomplete: every Wednesday is checked for a minutes page."""
    dates = set()
    for y in range(first_year, last_year + 1):
        for url in fed_statement_links(session, y):
            dates |= _dates_in(url)
    for url, cache in FED_CALENDAR_PAGES:
        dates |= _dates_in(_get_cached(session, url, cache))
    for y in range(first_year, last_year + 1):
        if sum(d.year == y for d in dates) < MIN_FED_MEETINGS_PER_YEAR:
            dates |= _dates_in(_get_cached(session, FED_HISTORICAL_URL.format(y=y),
                                           f"fed/historical-{y}.html"))
    for y in range(first_year, last_year + 1):
        if sum(d.year == y for d in dates) >= MIN_FED_MEETINGS_PER_YEAR:
            continue
        print(f"[texts] Fed {y}: few meetings listed - checking each Wednesday for minutes")
        for wed in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="W-WED"):
            if wed.date() > date.today():
                break
            d = wed.strftime("%Y%m%d")
            if _get_cached(session, FED_MINUTES_URL.format(d=d), f"fed/minutes-{d}.html"):
                dates.add(wed.date())
    return sorted(d for d in dates if first_year <= d.year <= last_year)


def scrape_fed(first_year, last_year):
    """FOMC statements (released on the decision day, used for announcement dates) and
    FOMC minutes (the document type used for the sentiment index in Hilscher et al., 2026)."""
    s = _session()
    rows = []
    counts = {}
    for meeting in fed_meeting_dates(s, first_year, last_year):
        d = meeting.strftime("%Y%m%d")
        c = counts.setdefault(meeting.year, [0, 0])
        url = f"{FED_BASE}/newsevents/pressreleases/monetary{d}a.htm"
        html = _get_cached(s, url, f"fed/{d}.html")
        if html is not None:
            body = parse_fed_page(html)
            # keep only real policy statements (other press releases share the URL pattern)
            if body is not None and "federal funds rate" in body.lower():
                rows.append({"date": meeting, "bank": "Fed", "doc_type": "statement",
                             "url": url, "text": body})
                c[0] += 1
        murl = FED_MINUTES_URL.format(d=d)
        mhtml = _get_cached(s, murl, f"fed/minutes-{d}.html")
        if mhtml is None:
            continue                          # e.g. unscheduled meetings have no own minutes
        release, mbody = parse_fed_minutes(mhtml, meeting)
        if mbody is None:
            print(f"   ! could not extract FOMC minutes for meeting {meeting}")
            continue
        if pd.Timestamp(release) > pd.Timestamp(date.today()):
            continue                          # not yet published
        rows.append({"date": release, "bank": "Fed", "doc_type": "minutes",
                     "url": murl, "text": mbody})
        c[1] += 1
    for y in range(first_year, last_year + 1):
        n_stmt, n_min = counts.get(y, [0, 0])
        print(f"[texts] Fed {y}: {n_stmt} statements, {n_min} minutes")
    return rows


# ------------------------------------------------------------------ manual overrides
def load_manual_texts():
    rows = []
    for bank_dir, bank in [("boe", "BoE"), ("fed", "Fed")]:
        folder = config.MANUAL_TEXT_DIR / bank_dir
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.txt")):
            try:
                d = datetime.strptime(f.stem, "%Y-%m-%d").date()
            except ValueError:
                print(f"   ! ignoring {f.name}: name must be YYYY-MM-DD.txt")
                continue
            rows.append({"date": d, "bank": bank, "doc_type": "minutes",
                         "url": f"manual:{f.name}",
                         "text": _clean_body(f.read_text(encoding="utf-8"))})
    if rows:
        print(f"[texts] {len(rows)} manual text files loaded")
    return rows


# ------------------------------------------------------------------ BIS central bankers' speeches
# The Bank for International Settlements publishes every central banker's speech as a free
# bulk download for research (non-commercial use), one zip per year:
#   https://www.bis.org/cbspeeches/download.htm
# Hilscher et al. (2026) also build sentiment indices from Fed and ECB speeches, so speeches
# are a document type used in the paper. They are collected for both banks; when a bank's
# minutes cannot be downloaded (e.g. the website blocks the network), its speeches become
# the basis of its sentiment index (see features.index_docs).
BIS_YEAR_URL = "https://www.bis.org/pages/download-central-bankers-speeches/speeches-{y}.zip"
BANK_PATTERNS = {
    "BoE": r"Bank of England",
    "Fed": r"Federal Reserve|Board of Governors of the Federal Reserve|Federal Open Market Committee",
}


def _download_bytes(url, cache_path):
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    blob = None
    try:
        import requests
        r = requests.get(url, headers=BROWSER_HEADERS, timeout=120)
        if r.status_code == 200:
            blob = r.content
    except Exception:
        pass
    if blob is None:
        try:
            from curl_cffi import requests as cffi_requests
            r = cffi_requests.get(url, impersonate="chrome", timeout=120)
            if r.status_code == 200:
                blob = r.content
        except Exception:
            pass
    if blob:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(blob)
    return blob


def _read_bis_zip(blob):
    """Read every CSV inside a BIS zip, whatever its exact column names."""
    import io
    import zipfile
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            if not name.lower().endswith((".csv", ".txt")):
                continue
            raw = z.read(name)
            for enc in ("utf-8", "latin-1"):
                try:
                    frames.append(pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str))
                    break
                except Exception:
                    continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _standardise_bis(df):
    """Find the date, text and metadata columns by name/content, return tidy rows."""
    if df.empty:
        return pd.DataFrame(columns=["date", "meta", "text", "url"])
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if "date" in k), None)
    text_col = next((cols[k] for k in ("text", "content", "speech", "body") if k in cols), None)
    if text_col is None:                                  # fall back to the longest column
        lengths = {c: df[c].fillna("").str.len().mean() for c in df.columns}
        text_col = max(lengths, key=lengths.get)
    meta_cols = [cols[k] for k in cols if any(t in k for t in
                 ("title", "description", "author", "speaker", "info", "subtitle"))]
    url_col = next((cols[k] for k in cols if "url" in k or "link" in k), None)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
        "meta": df[meta_cols].fillna("").agg(" ".join, axis=1) if meta_cols else "",
        "text": df[text_col].fillna(""),
        "url": df[url_col].fillna("") if url_col else "bis",
    })
    if date_col is None:          # try to read the date from the metadata text
        out["date"] = pd.to_datetime(out["meta"].str.extract(
            r"(\d{1,2} [A-Z][a-z]+ \d{4})")[0], errors="coerce", dayfirst=True)
    return out.dropna(subset=["date"])


def _bank_of(meta):
    hits = {b: m.start() for b, pat in BANK_PATTERNS.items()
            if (m := re.search(pat, meta, flags=re.I))}
    return min(hits, key=hits.get) if hits else None


def collect_bis_speeches(first_year, last_year, min_words=300):
    rows = []
    failures = 0
    for y in range(first_year, last_year + 1):
        blob = _download_bytes(BIS_YEAR_URL.format(y=y), config.RAW_DIR / "bis" / f"speeches-{y}.zip")
        if not blob:
            failures += 1
            print(f"   ! BIS speeches {y}: download failed")
            if failures >= 2 and not rows:
                print("   ! BIS site not reachable from this network - skipping speeches")
                break
            continue
        try:
            tidy = _standardise_bis(_read_bis_zip(blob))
        except Exception as exc:
            print(f"   ! BIS speeches {y}: could not read file ({type(exc).__name__})")
            continue
        tidy["bank"] = tidy["meta"].map(_bank_of)
        # if the metadata does not name the bank, look at the first lines of the speech
        unknown = tidy["bank"].isna()
        tidy.loc[unknown, "bank"] = tidy.loc[unknown, "text"].str[:600].map(_bank_of)
        tidy = tidy[tidy["bank"].notna() & (tidy["text"].str.split().str.len() >= min_words)]
        n = tidy["bank"].value_counts()
        print(f"[texts] BIS speeches {y}: BoE {int(n.get('BoE', 0))}, Fed {int(n.get('Fed', 0))}")
        for _, r in tidy.iterrows():
            rows.append({"date": r["date"].date(), "bank": r["bank"], "doc_type": "speech",
                         "url": r["url"] or "bis", "text": _clean_body(r["text"])})
    return rows


def site_reachable(url):
    """Quick probe: can we download one known page from this site?"""
    status, text = _fetch(_session(), url)
    return status == 200 and len(text) > 1000


COLUMNS = ["date", "bank", "doc_type", "url", "text"]
BOE_PROBE = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2024/may-2024"
FED_PROBE = "https://www.federalreserve.gov/newsevents/pressreleases/2024-press-fomc.htm"


def build_text_file(demo=False):
    print("=== Step 2: Collecting Federal Reserve and Bank of England communications ===")
    if demo:
        from src.demo_data import synthetic_texts
        df = synthetic_texts()
    else:
        last = config.TEXT_LAST_YEAR or date.today().year
        rows = []
        for name, probe, scraper, first in [("Bank of England", BOE_PROBE, scrape_boe, config.BOE_FIRST_YEAR),
                                             ("Federal Reserve", FED_PROBE, scrape_fed, config.FED_FIRST_YEAR)]:
            if site_reachable(probe):
                print(f"[texts] {name} website reachable - downloading minutes")
                rows += scraper(first, last)
            else:
                print(f"[texts] {name} website blocks this network - its BIS speeches will be used")
        _report_blocked()
        rows += load_manual_texts()
        # Speeches are only needed for a bank whose minutes could not be downloaded:
        # scoring ~1,500 speeches would take a laptop many hours and a lot of disk space.
        minutes = pd.Series([r["bank"] for r in rows if r["doc_type"] == "minutes"]).value_counts()
        need = [b for b in ("BoE", "Fed") if minutes.get(b, 0) < config.INDEX_MIN_DOCS]
        if need:
            print(f"[texts] Not enough minutes for {', '.join(need)} - downloading BIS speeches")
            speeches = collect_bis_speeches(min(config.BOE_FIRST_YEAR, config.FED_FIRST_YEAR), last)
            rows += [r for r in speeches if r["bank"] in need]
        else:
            print(f"[texts] Minutes collected for both banks (BoE {minutes.get('BoE', 0)}, "
                  f"Fed {minutes.get('Fed', 0)}) - BIS speeches not needed")
        df = pd.DataFrame(rows, columns=COLUMNS)
        for bank in ("BoE", "Fed"):
            if not (df["bank"] == bank).any():
                raise RuntimeError(f"No {bank} texts could be downloaded from any source. "
                                   "Run `python check_setup.py` to see which sites are reachable.")
        # manual files win over scraped ones for the same bank / type / date
        df["is_manual"] = df["url"].astype(str).str.startswith("manual:")
        df = (df.sort_values("is_manual")
                .drop_duplicates(subset=["bank", "doc_type", "date", "text"], keep="last")
                .drop(columns="is_manual"))
    df["date"] = pd.to_datetime(df["date"])
    df = df[COLUMNS].sort_values(["date", "bank", "doc_type"]).reset_index(drop=True)
    df.to_csv(config.TEXTS_CSV, index=False)
    counts = df.groupby(["bank", "doc_type"]).size()
    print(f"[texts] saved {len(df)} documents: "
          + ", ".join(f"{b} {t} {n}" for (b, t), n in counts.items()))
    return df


def load_texts():
    return pd.read_csv(config.TEXTS_CSV, parse_dates=["date"])
