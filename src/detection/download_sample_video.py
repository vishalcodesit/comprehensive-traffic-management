"""
Downloads a short (~1 min, 480p, ~10MB) real traffic-intersection clip from
Wikimedia Commons for the vehicle-detection demo.

Why this exists: YOLOv8 is trained on real photographs, so it does not
reliably fire on the flat-shaded procedural renders in synthetic_video.py
(verified empirically -- 0 detections across 200 synthetic frames vs. ~5
vehicles/frame on this real clip). Use this real clip to demonstrate that
detection/tracking genuinely work; use synthetic_video.py's deterministic
strobing-light-bar "ambulance" to demonstrate the emergency-preemption
feature specifically (a real clip can't guarantee an ambulance passes
through it).

License: CC BY-SA 4.0, "Intersection Pie-IX-Sherbrooke" by Thomas1313 --
see config.SAMPLE_VIDEO_ATTRIBUTION.

Run:  python -m src.detection.download_sample_video
"""
import os
import sys

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def download(url=config.SAMPLE_VIDEO_URL, path=config.SAMPLE_VIDEO_PATH, force=False):
    if os.path.exists(path) and not force:
        print(f"Already downloaded: {path}")
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading real traffic sample clip...\n  {config.SAMPLE_VIDEO_ATTRIBUTION}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Saved -> {path}")
    return path


if __name__ == "__main__":
    download()
