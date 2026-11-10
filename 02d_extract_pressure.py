"""Aşama 2d - ERA5 basınç seviyesi dosyalarından günlük büyük ölçekli öznitelikleri çıkarır.

Girdi : data/raw/era5_pl/pl_<yıl>.nc   (z, t; 500 ve 850 hPa; 12:00 TR)
Çıktı : data/interim/pl_daily.csv
  Nokta bazında (en yakın 0,25° hücre): z500, z850 (gpm), t500, t850 (°C), thick (z500 - z850, gpm)
  Bölge ortalamaları (enlem ağırlıklı): z500_<bölge>, t850_<bölge>  (bölgeler config.yaml -> pressure_levels.regions)
  Gradyan: z500_grad_we = z500_emed - z500_iran
Satır: date x point_id (bölge sütunları her noktada aynıdır).

Kullanım:
    python 02d_extract_pressure.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.common import ROOT, get_logger, load_config

G = 9.80665


def pick(ds, *names):
    for n in names:
        if n in ds.variables or n in ds.dims:
            return n
    raise KeyError(f"{names} bulunamadı: {list(ds.variables)}")


def main() -> None:
    cfg = load_config()
    log = get_logger("02d_pressure", cfg)
    files = sorted((ROOT / "data" / "raw" / "era5_pl").glob("pl_*.nc"))
    log.info("%d yıl dosyası", len(files))
    regions = cfg["pressure_levels"]["regions"]
    rows = []
    for k, f in enumerate(files):
        with xr.open_dataset(f) as ds:
            tn, ln = pick(ds, "valid_time", "time"), pick(ds, "pressure_level", "level", "isobaricInhPa")
            la, lo = pick(ds, "latitude", "lat"), pick(ds, "longitude", "lon")
            zv, tv = pick(ds, "z"), pick(ds, "t")
            if k == 0:
                log.info("Yapı: değişkenler=%s, boyutlar=%s", list(ds.data_vars), dict(ds.sizes))
            levs = ds[ln].values.astype(float)
            i5, i8 = int(np.argmin(np.abs(levs - 500))), int(np.argmin(np.abs(levs - 850)))
            z5 = ds[zv].isel({ln: i5}).values / G
            z8 = ds[zv].isel({ln: i8}).values / G
            t5 = ds[tv].isel({ln: i5}).values - 273.15
            t8 = ds[tv].isel({ln: i8}).values - 273.15
            lats, lons = ds[la].values, ds[lo].values
            dates = pd.to_datetime(ds[tn].values).normalize()
        w = np.cos(np.radians(lats))[:, None] * np.ones((1, len(lons)))
        reg = {}
        for name, (s, n, wst, e) in regions.items():
            m = ((lats >= s) & (lats <= n))[:, None] & ((lons >= wst) & (lons <= e))[None, :]
            ww = np.where(m, w, 0.0)
            reg[f"z500_{name}"] = (z5 * ww).sum(axis=(1, 2)) / ww.sum()
            reg[f"t850_{name}"] = (t8 * ww).sum(axis=(1, 2)) / ww.sum()
        reg["z500_grad_we"] = reg["z500_emed"] - reg["z500_iran"]
        for p in cfg["points"]:
            i = int(np.argmin(np.abs(lats - p["lat"]))); j = int(np.argmin(np.abs(lons - p["lon"])))
            d = pd.DataFrame({"date": dates, "point_id": p["id"],
                              "z500": z5[:, i, j], "z850": z8[:, i, j], "t500": t5[:, i, j], "t850": t8[:, i, j]})
            d["thick"] = d["z500"] - d["z850"]
            for c, v in reg.items():
                d[c] = v
            rows.append(d)
        log.info("  %s (%d gün)", f.name, len(dates))
    out = pd.concat(rows, ignore_index=True).sort_values(["point_id", "date"])
    path = ROOT / "data" / "interim" / "pl_daily.csv"
    out.round(3).to_csv(path, index=False)
    log.info("Kaydedildi: %d satır, %d sütun -> %s | eksik: %d", len(out), out.shape[1], path, int(out.isna().sum().sum()))
    jul = out[out.date.dt.month == 7]
    log.info("Kontrol (Temmuz ort.): z500_gda %.0f gpm, t850_gda %.1f °C", jul.z500_gda.mean(), jul.t850_gda.mean())


if __name__ == "__main__":
    main()
