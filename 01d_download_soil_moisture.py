"""Aşama 1d - Toprak nemini standart saatlik ERA5-Land arşivinden indirir.

Neden ayrı: nokta zaman serisi ürünü (reanalysis-era5-land-timeseries) toprak nemi için
veri döndürmüyor. Toprak nemi gün içinde yavaş değiştiği için günde tek saat (12:00 TR) alınır.
Her yıl tek istek (2 değişken x tüm Nisan-Eylül günleri x 1 saat, küçük alan):
    data/raw/era5land_soil/soil_<yıl>.nc
İnmiş yıllar atlanır.

Kullanım:
    python 01d_download_soil_moisture.py --years 1981     # önce dene
    python 01d_download_soil_moisture.py                  # tüm yıllar
"""
from __future__ import annotations

import argparse
import shutil
import time
import zipfile
from pathlib import Path

from src.common import ROOT, get_logger, load_config

QUEUE_FULL = "Maximum number queued requests"


def build_request(cfg: dict, year: int) -> dict:
    s = cfg["soil_moisture"]
    return {
        "variable": s["variables"],
        "year": str(year),
        "month": [f"{m:02d}" for m in cfg["period"]["months"]],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [s["time_utc"]],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": s["area"],
    }


def finalize(tmp: Path, target: Path) -> None:
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            ncs = [n for n in z.namelist() if n.endswith(".nc")]
            if len(ncs) != 1:
                raise RuntimeError(f"Zip içinde beklenmeyen içerik: {z.namelist()}")
            with z.open(ncs[0]) as s_, open(target, "wb") as d:
                shutil.copyfileobj(s_, d)
        tmp.unlink()
    else:
        tmp.replace(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-wait-min", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("01d_soil", cfg)
    out = ROOT / "data" / "raw" / "era5land_soil"
    out.mkdir(parents=True, exist_ok=True)
    years = args.years or list(range(cfg["period"]["start_year"], cfg["period"]["end_year"] + 1))

    client = None
    if not args.dry_run:
        import cdsapi
        client = cdsapi.Client()

    for y in years:
        target = out / f"soil_{y}.nc"
        if target.exists() and target.stat().st_size > 0:
            continue
        req = build_request(cfg, y)
        if args.dry_run:
            log.info("%d -> %s", y, req); continue
        waited = 0
        while True:
            try:
                t0 = time.time()
                log.info("İndiriliyor: toprak nemi %d", y)
                tmp = out / f"soil_{y}.part"
                client.retrieve(cfg["soil_moisture"]["dataset"], req).download(str(tmp))
                finalize(tmp, target)
                log.info("  ✓ %d (%.0f sn, %.1f MB)", y, time.time() - t0, target.stat().st_size / 1e6)
                break
            except Exception as exc:
                msg = str(exc)
                if QUEUE_FULL in msg and waited < args.max_wait_min:
                    log.warning("  Kuyruk dolu; 5 dk bekleniyor"); time.sleep(300); waited += 5; continue
                log.error("  ✗ %d başarısız: %s", y, msg[:500]); break
    log.info("Bitti.")


if __name__ == "__main__":
    main()
