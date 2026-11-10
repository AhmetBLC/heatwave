"""Aşama 1b - NASA POWER günlük nokta verisini indirir (doğrulama ve ek değişkenler).

Her nokta için bir CSV:
    data/raw/nasapower/<nokta_id>.csv
Sütunlar: date, <parametreler>..., lat, lon, elevation_m
-999 değerleri boş (NaN) olarak kaydedilir.

Kullanım:
    python 01b_download_nasapower.py
    python 01b_download_nasapower.py --force   # var olanları da yeniden indir
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import requests

from src.common import get_logger, load_config, resolve


def fetch_point(cfg: dict, lat: float, lon: float, retries: int = 3) -> dict:
    n = cfg["nasapower"]
    params = {
        "parameters": ",".join(n["parameters"]),
        "community": n["community"],
        "latitude": lat,
        "longitude": lon,
        "start": f"{cfg['period']['start_year']}0101",
        "end": f"{cfg['period']['end_year']}1231",
        "format": "JSON",
        "time-standard": n["time_standard"],
    }
    for attempt in range(1, retries + 1):
        r = requests.get(n["url"], params=params, timeout=300)
        if r.status_code == 200:
            return r.json()
        if attempt == retries:
            r.raise_for_status()
        time.sleep(30 * attempt)


def to_frame(js: dict) -> pd.DataFrame:
    par = js["properties"]["parameter"]
    df = pd.DataFrame(par)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.replace(-999.0, np.nan)
    lon, lat, elev = js["geometry"]["coordinates"]
    df["lat"], df["lon"], df["elevation_m"] = lat, lon, elev
    return df.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("01b_nasapower", cfg)
    out_dir = resolve(cfg, "nasapower_raw")
    for p in cfg["points"]:
        target = out_dir / f"{p['id']}.csv"
        if target.exists() and not args.force:
            log.info("Zaten var, atlanıyor: %s", target.name)
            continue
        log.info("İndiriliyor: %s (%.3f, %.3f)", p["name"], p["lat"], p["lon"])
        df = to_frame(fetch_point(cfg, p["lat"], p["lon"]))
        df.insert(0, "point_id", p["id"])
        df.to_csv(target, index=False)
        log.info("  %d gün, %d eksik değer", len(df), int(df.isna().sum().sum()))
        time.sleep(2)  # API'ye nazik davran
    log.info("Bitti.")


if __name__ == "__main__":
    main()
