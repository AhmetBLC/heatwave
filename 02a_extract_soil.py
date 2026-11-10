"""Aşama 2a - Toprak nemi NetCDF dosyalarından 9 ilin en yakın kara hücresini çıkarır.

Girdi : data/raw/era5land_soil/soil_<yıl>.nc
Çıktı : data/interim/soil_points.csv  (date, point_id, swvl1, swvl2, cell_lat, cell_lon, dist_km)
        data/interim/soil_cells.csv   (her il için seçilen hücre ve mesafe)
Tarih, Türkiye saatine göre gündür (12:00 TR = 09:00 UTC olduğu için UTC tarihi ile aynıdır).

Kullanım:
    python 02a_extract_soil.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.common import ROOT, get_logger, load_config


def nearest_land(mask2d: np.ndarray, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float):
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    d = np.sqrt((la - lat) ** 2 + ((lo - lon) * np.cos(np.radians(lat))) ** 2) * 111.0
    d = np.where(mask2d, d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return i, j, float(la[i, j]), float(lo[i, j]), float(d[i, j])


def main() -> None:
    cfg = load_config()
    log = get_logger("02a_soil", cfg)
    src = ROOT / "data" / "raw" / "era5land_soil"
    out = ROOT / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("soil_*.nc"))
    log.info("%d yıl dosyası bulundu", len(files))

    cells, rows = None, []
    for f in files:
        with xr.open_dataset(f) as ds:
            tname = "valid_time" if "valid_time" in ds.dims else "time"
            latn = "latitude" if "latitude" in ds.dims else "lat"
            lonn = "longitude" if "longitude" in ds.dims else "lon"
            if cells is None:
                log.info("Dosya yapısı: değişkenler=%s, boyutlar=%s", list(ds.data_vars), dict(ds.sizes))
                lats, lons = ds[latn].values, ds[lonn].values
                mask = ~np.isnan(ds["swvl1"].isel({tname: 0}).values)
                cells = []
                for p in cfg["points"]:
                    i, j, cla, clo, km = nearest_land(mask, lats, lons, p["lat"], p["lon"])
                    cells.append({"point_id": p["id"], "name": p["name"], "lat": p["lat"], "lon": p["lon"],
                                  "i": i, "j": j, "cell_lat": cla, "cell_lon": clo, "dist_km": round(km, 2)})
                    log.info("  %-11s -> (%.2f, %.2f) %.1f km", p["name"], cla, clo, km)
            times = pd.to_datetime(ds[tname].values)
            for c in cells:
                sel = {latn: c["i"], lonn: c["j"]}
                v1 = ds["swvl1"].isel(sel).values
                v2 = ds["swvl2"].isel(sel).values
                rows.append(pd.DataFrame({"date": times.normalize(), "point_id": c["point_id"], "swvl1": v1, "swvl2": v2}))
        log.info("  %s işlendi (%d gün)", f.name, len(times))

    df = pd.concat(rows, ignore_index=True).sort_values(["point_id", "date"])
    df.to_csv(out / "soil_points.csv", index=False)
    pd.DataFrame(cells).to_csv(out / "soil_cells.csv", index=False)
    log.info("Kaydedildi: %d satır | eksik swvl1: %d | swvl1 aralığı [%.3f, %.3f]",
             len(df), int(df.swvl1.isna().sum()), df.swvl1.min(), df.swvl1.max())


if __name__ == "__main__":
    main()
