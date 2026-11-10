"""Aşama 1c - İndirmelerin eksiksiz ve mantıklı olduğunu kontrol eder.

- Beklenen tüm ERA5-Land dosyaları var mı?
- Her dosyada beklenen gün sayısı var mı?
- 9 noktanın en yakın kara hücresi bulunuyor mu (deniz/göl hücresi değil)?
- Değerler fiziksel olarak makul mü (ör. Tmax -30..60 °C)?
- NASA POWER CSV'leri tam mı?
Sonuç: logs/01c_check.log ve ekrana özet.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import xarray as xr

from src.common import get_logger, load_config, resolve

# NetCDF içindeki kısa adlar -> makul aralık (ham birimler: K, m3/m3, Pa, m/s)
PLAUSIBLE = {
    "t2m": (240, 335), "d2m": (220, 310),
    "swvl1": (0, 0.8), "swvl2": (0, 0.8),
    "sp": (70000, 105000), "u10": (-40, 40), "v10": (-40, 40),
}


def expected_days(year: int, months: list[int]) -> int:
    return sum(calendar.monthrange(year, m)[1] for m in months)


def data_var(ds: xr.Dataset) -> str:
    names = [v for v in ds.data_vars if ds[v].ndim >= 2]
    return names[0]


def nearest_land(da2d: xr.DataArray, lat: float, lon: float):
    """En yakın NaN olmayan hücreyi bulur (ERA5-Land'de deniz/göl hücreleri NaN)."""
    latn = "latitude" if "latitude" in da2d.dims else "lat"
    lonn = "longitude" if "longitude" in da2d.dims else "lon"
    la, lo = np.meshgrid(da2d[latn].values, da2d[lonn].values, indexing="ij")
    d = (la - lat) ** 2 + ((lo - lon) * np.cos(np.radians(lat))) ** 2
    d = np.where(np.isnan(da2d.values), np.inf, d)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return float(la[i, j]), float(lo[i, j]), float(np.sqrt(d[i, j]) * 111.0)


def main() -> None:
    cfg = load_config()
    e = cfg["era5land"]
    log = get_logger("01c_check", cfg)
    raw = resolve(cfg, "era5land_raw")
    years = range(cfg["period"]["start_year"], cfg["period"]["end_year"] + 1)
    size = e.get("months_per_request", 1)
    months = cfg["period"]["months"]
    groups = [months[i:i + size] for i in range(0, len(months), size)]
    problems = 0

    # 1) Eksik istek klasörleri
    missing = []
    for r in e["requests"]:
        for y in years:
            for g in groups:
                tag = f"{y}_" + "-".join(f"{m:02d}" for m in g)
                for v in r["variables"]:
                    if not (raw / r["statistic"] / tag / f"{v}.nc").exists():
                        missing.append(f"{r['statistic']}/{tag}/{v}")
    log.info("ERA5-Land: %d dosya eksik", len(missing))
    for m in missing[:20]:
        log.info("  eksik: %s", m)
    problems += len(missing)

    # 2) İnen dosyaların içeriği
    checked_points = False
    n_files = 0
    for folder in sorted(p for p in raw.glob("*/*") if p.is_dir()):
        stat = folder.parent.name
        year = int(folder.name.split("_")[0])
        g = [int(x) for x in folder.name.split("_")[1].split("-")]
        exp = expected_days(year, g)
        for f in sorted(folder.glob("*.nc")):
            n_files += 1
            with xr.open_dataset(f) as ds:
                v = data_var(ds)
                tname = next(d for d in ds[v].dims if "time" in d)
                n = ds.sizes[tname]
                lo, hi = PLAUSIBLE.get(v, (-np.inf, np.inf))
                vmin, vmax = float(ds[v].min()), float(ds[v].max())
                if n != exp:
                    log.warning("%s/%s: %d gün var, %d bekleniyordu", folder.name, f.name, n, exp); problems += 1
                if vmin < lo or vmax > hi:
                    log.warning("%s/%s (%s): değer aralığı şüpheli [%.2f, %.2f]", folder.name, f.name, v, vmin, vmax); problems += 1
                if not checked_points and v == "t2m":
                    first = ds[v].isel({tname: 0})
                    log.info("Nokta -> en yakın ERA5-Land kara hücresi:")
                    for p in cfg["points"]:
                        la, lo_, km = nearest_land(first, p["lat"], p["lon"])
                        log.info("  %-11s (%.3f, %.3f) -> (%.2f, %.2f)  ~%.1f km", p["name"], p["lat"], p["lon"], la, lo_, km)
                        if km > 15:
                            log.warning("  %s için en yakın hücre uzak!", p["name"]); problems += 1
                    log.info("Örnek dosya: %s | değişken: %s | boyutlar: %s", f.name, v, dict(ds[v].sizes))
                    checked_points = True
    log.info("ERA5-Land: %d .nc dosyası kontrol edildi", n_files)

    # 3) NASA POWER
    nraw = resolve(cfg, "nasapower_raw")
    exp_days = (pd.Timestamp(f"{cfg['period']['end_year']}-12-31") - pd.Timestamp(f"{cfg['period']['start_year']}-01-01")).days + 1
    for p in cfg["points"]:
        f = nraw / f"{p['id']}.csv"
        if not f.exists():
            log.warning("NASA POWER eksik: %s", f.name); problems += 1; continue
        df = pd.read_csv(f)
        miss = df.drop(columns=["point_id", "date", "lat", "lon", "elevation_m"]).isna().mean() * 100
        log.info("NASA POWER %-11s %d/%d gün | en çok eksik: %s %.1f%%", p["name"], len(df), exp_days, miss.idxmax(), miss.max())
        if len(df) != exp_days:
            problems += 1

    log.info("KONTROL SONUCU: %s", "SORUN YOK" if problems == 0 else f"{problems} sorun bulundu, yukarıya bakın")


if __name__ == "__main__":
    main()
