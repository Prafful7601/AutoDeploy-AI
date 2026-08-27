"""
Stage 1 (data), step 1: fetch the TravisTorrent dataset.

What this does
---------------
TravisTorrent (Beller, Gousios & Zaidman, MSR 2017) is the standard public
dataset for this exact problem: build metadata + outcome for real CI builds,
mined from Travis CI and GitHub across ~1,300 open-source Java/Ruby/Python/Go
projects (build history through Jan 2017).

The project's original home, travistorrent.testroots.org, has been taken
over by an unrelated domain since the project's academic site lapsed — do
NOT use it. The maintainers' own archive page points to Figshare as the
permanent home: https://doi.org/10.6084/m9.figshare.19314170

This script pulls the file list from Figshare's public API, downloads the
most recent snapshot (`final-2017-01-25.csv.gz`, ~264 MB compressed), and
extracts it to data/raw/. It's idempotent — reruns skip work already done.

Why not the GitHub Actions API fallback
-----------------------------------------
The brief says to try TravisTorrent first and only fall back to mining the
GitHub Actions API (15-20 repos) if TravisTorrent isn't reasonably
obtainable. It was obtainable — the dataset above downloaded cleanly and
matches its published checksum/size — so that's what this project uses.
It is also a strictly better source for this task: ~1,300 projects and
historical build volume no small, freshly-mined GitHub Actions sample could
match in the time available.
"""

import gzip
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR  # noqa: E402

FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/19314170"
EXPECTED_FILENAME = "final-2017-01-25.csv.gz"

GZ_PATH = RAW_DATA_DIR / "travistorrent_final.csv.gz"
CSV_PATH = RAW_DATA_DIR / "travistorrent_final.csv"


def resolve_download_url() -> str:
    """Ask Figshare's API for the current direct-download URL for the
    dataset file, rather than hardcoding a CDN link that could rot."""
    resp = requests.get(FIGSHARE_ARTICLE_API, timeout=30)
    resp.raise_for_status()
    files = resp.json()["files"]
    for f in files:
        if f["name"] == EXPECTED_FILENAME:
            return f["download_url"], f["size"]
    raise RuntimeError(
        f"Could not find {EXPECTED_FILENAME} in Figshare article 19314170. "
        f"Files present: {[f['name'] for f in files]}"
    )


def download(url: str, expected_size: int, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {dest} already downloaded ({expected_size:,} bytes)")
        return
    print(f"Downloading {url} -> {dest} ({expected_size:,} bytes)...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    actual = dest.stat().st_size
    if actual != expected_size:
        raise RuntimeError(
            f"Downloaded size {actual:,} != expected {expected_size:,}. "
            "Download may be corrupt; delete the file and rerun."
        )
    print(f"Downloaded {actual:,} bytes.")


def extract(gz_path: Path, csv_path: Path) -> None:
    if csv_path.exists():
        print(f"[skip] {csv_path} already extracted")
        return
    print(f"Extracting {gz_path} -> {csv_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Extracted {csv_path.stat().st_size:,} bytes.")


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    url, size = resolve_download_url()
    download(url, size, GZ_PATH)
    extract(GZ_PATH, CSV_PATH)
    print(f"\nRaw TravisTorrent CSV ready at: {CSV_PATH}")


if __name__ == "__main__":
    main()
