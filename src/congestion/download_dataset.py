"""
Downloads and extracts the UCI Metro Interstate Traffic Volume dataset
(~48k hourly I-94 traffic records with weather, 2012-2018) that
preprocess.py reads from data/raw/.

Dataset: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

Run:  python -m src.congestion.download_dataset
"""
import gzip
import io
import os
import shutil
import sys
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

DATASET_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"


def download(force=False):
    if os.path.exists(config.METRO_CSV) and not force:
        print(f"Already downloaded: {config.METRO_CSV}")
        return config.METRO_CSV

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    print(f"Downloading UCI Metro Interstate Traffic Volume dataset from {DATASET_URL} ...")
    resp = requests.get(DATASET_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        gz_name = next(n for n in zf.namelist() if n.endswith(".csv.gz"))
        zf.extract(gz_name, config.RAW_DATA_DIR)

    gz_path = os.path.join(config.RAW_DATA_DIR, gz_name)
    with gzip.open(gz_path, "rb") as f_in, open(config.METRO_CSV, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Saved -> {config.METRO_CSV}")
    return config.METRO_CSV


if __name__ == "__main__":
    download()
