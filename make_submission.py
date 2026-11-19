"""
Create the submission package: code + data + trained models + results in ONE zip file,
ready to upload to OneDrive / Google Drive and link in the report.

    Press Run on this file (or: python make_submission.py)

Output: submission/gbpusd_project_submission.zip
Left out (not needed to reproduce, or machine-specific): .venv, __pycache__, the raw
HTML/zip download cache (data/raw), the exported certificate bundle and old zips.
"""
import json
import zipfile
from datetime import datetime

import config

ROOT = config.ROOT
OUT_DIR = ROOT / "submission"
ZIP_PATH = OUT_DIR / "gbpusd_project_submission.zip"

SKIP_DIRS = {".venv", "__pycache__", "submission", ".git", ".idea", ".vscode", "raw"}
SKIP_FILES = {"system_ca_bundle.pem", ".DS_Store"}
SKIP_SUFFIX = {".pyc", ".zip"}
REQUIRED = ["main.py", "run_all.py", "config.py", "requirements.txt", "README.md",
            "data/gbpusd_daily.csv", "data/market_daily.csv", "data/cb_texts.csv",
            "data/cb_sentiment.csv", "data/model_panel.csv",
            "models/full_data_models.joblib", "results/summary.txt"]


def included(path):
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts[:-1]) or rel.parts[0] in SKIP_DIRS:
        return False
    return path.is_file() and path.name not in SKIP_FILES and path.suffix not in SKIP_SUFFIX


def main():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    if missing:
        print("  ! These files are missing - run the project first (main.py option 4 with 'y', "
              "or `python run_all.py`):")
        for f in missing:
            print("     -", f)
        return None
    summary = json.loads((ROOT / "results" / "data_summary.json").read_text()) \
        if (ROOT / "results" / "data_summary.json").exists() else {}
    if summary.get("demo_mode"):
        print("  ! The saved results come from DEMO data - run the real project before submitting.")
        return None
    OUT_DIR.mkdir(exist_ok=True)
    files = sorted(p for p in ROOT.rglob("*") if included(p))
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=f"gbpusd_project/{p.relative_to(ROOT)}")
    size = ZIP_PATH.stat().st_size / 1e6
    groups = {}
    for p in files:
        top = p.relative_to(ROOT).parts[0] if len(p.relative_to(ROOT).parts) > 1 else "(code)"
        groups[top] = groups.get(top, 0) + 1
    print(f"\n  Created {ZIP_PATH.relative_to(ROOT)} ({size:.1f} MB, {len(files)} files, "
          f"{datetime.now():%d %b %Y %H:%M})")
    print("  Contents: " + ", ".join(f"{k} {v}" for k, v in sorted(groups.items())))
    print("\n  Next: upload this zip to OneDrive or Google Drive, set sharing to 'Anyone with the link")
    print("  can view' (or your university), and paste the link in the report, clearly labelled.")
    return ZIP_PATH


if __name__ == "__main__":
    main()
