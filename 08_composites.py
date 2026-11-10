"""Aşama 8 - Sıcak hava dalgası öncesi kompozit Z500 ve T850 anomali alanları (+ harita altlığı).

Ne yapar:
  1) Harita altlığı: ERA5 kara-deniz maskesi ve yüzey jeopotansiyelini (yükselti) bir kez indirir
     -> data/raw/era5_invariant.nc   (internet ve CDS hesabı gerekir; varsa atlanır)
  2) Kompozitler: Her ilin Mayıs-Eylül'de başlayan sıcak hava dalgalarının başlangıç günlerini (gün 0)
     alır, aynı gün birden çok ilde başlayan olayları tek tarih sayar ve gün 0'dan 7, 5, 3, 1 gün önce
     ile gün 0'daki ortalama Z500 ve T850 anomalisini her grid noktası için hesaplar.
     Anomali: 1981-2010 takvim günü klimatolojisine göre (±7 gün pencere).
     Eğilimden arındırma (_dt): ısınma eğilimi kompozitlere sızmasın diye her grid noktasında
     yıllık mevsim ortalaması anomalisinin doğrusal eğilimi çıkarılır; böylece kompozit yalnızca
     sinoptik (hava durumu ölçeğindeki) sinyali gösterir.
     Anlamlılık: her grid noktasında tek örneklem t-testi (H0: ortalama anomali = 0).
Çıktı:
  results/analysis/composites.csv     (lag, lat, lon, z500_an, t850_an, z500_p, t850_p ve eğilimden arındırılmış _dt sürümleri)
  results/analysis/composites_meta.csv (lag, n_dates)
  results/analysis/basemap.csv        (lat, lon, lsm, orog_m)

Kullanım:
    python 08_composites.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from src.common import ROOT, get_logger, load_config

G = 9.80665
LAGS = [-7, -5, -3, -1, 0]


def pick(ds, *names):
    for n in names:
        if n in ds.variables or n in ds.dims:
            return n
    raise KeyError(f"{names} bulunamadı: {list(ds.variables)}")


def download_invariants(cfg, log):
    target = ROOT / "data" / "raw" / "era5_invariant.nc"
    if target.exists() and target.stat().st_size > 0:
        return target
    import cdsapi
    req = {"product_type": ["reanalysis"], "variable": ["land_sea_mask", "geopotential"],
           "year": ["2024"], "month": ["01"], "day": ["01"], "time": ["00:00"],
           "data_format": "netcdf", "download_format": "unarchived", "area": cfg["pressure_levels"]["area"]}
    log.info("Harita altlığı indiriliyor (kara-deniz maskesi, yükselti)...")
    tmp = target.with_suffix(".part")
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", req).download(str(tmp))
    import zipfile, shutil
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            n = [x for x in z.namelist() if x.endswith(".nc")][0]
            with z.open(n) as s, open(target, "wb") as d:
                shutil.copyfileobj(s, d)
        tmp.unlink()
    else:
        tmp.replace(target)
    return target


def main():
    cfg = load_config()
    log = get_logger("08_composites", cfg)
    out = ROOT / "results" / "analysis"; out.mkdir(parents=True, exist_ok=True)

    # 1) harita altlığı (zaten varsa atlanır)
    try:
        f = download_invariants(cfg, log)
        with xr.open_dataset(f) as ds:
            la, lo = pick(ds, "latitude", "lat"), pick(ds, "longitude", "lon")
            tn = pick(ds, "valid_time", "time")
            lsm = ds[pick(ds, "lsm")].isel({tn: 0}).values
            orog = ds[pick(ds, "z")].isel({tn: 0}).values / G
            LA, LO = np.meshgrid(ds[la].values, ds[lo].values, indexing="ij")
        pd.DataFrame({"lat": LA.ravel(), "lon": LO.ravel(), "lsm": lsm.ravel().round(3), "orog_m": orog.ravel().round(1)}) \
            .to_csv(out / "basemap.csv", index=False)
        log.info("Harita altlığı kaydedildi (%d nokta)", LA.size)
    except Exception as e:
        log.warning("Harita altlığı alınamadı (%s); kompozitlere devam ediliyor", e)

    # 2) basınç seviyesi alanlarını yükle
    files = sorted((ROOT / "data" / "raw" / "era5_pl").glob("pl_*.nc"))
    Z, T, D = [], [], []
    for k, f in enumerate(files):
        with xr.open_dataset(f) as ds:
            tn, ln = pick(ds, "valid_time", "time"), pick(ds, "pressure_level", "level")
            la, lo = pick(ds, "latitude", "lat"), pick(ds, "longitude", "lon")
            levs = ds[ln].values.astype(float)
            i5, i8 = int(np.argmin(abs(levs - 500))), int(np.argmin(abs(levs - 850)))
            Z.append((ds["z"].isel({ln: i5}).values / G).astype("float32"))
            T.append((ds["t"].isel({ln: i8}).values - 273.15).astype("float32"))
            D.append(pd.to_datetime(ds[tn].values).normalize())
            if k == 0:
                lats, lons = ds[la].values, ds[lo].values
    Z, T = np.concatenate(Z), np.concatenate(T)
    dates = pd.DatetimeIndex(np.concatenate([d.values for d in D]))
    log.info("Alanlar yüklendi: %s gün x %d x %d", len(dates), len(lats), len(lons))

    # 3) klimatoloji (1981-2010, takvim günü, ±7 gün) ve anomali
    doy = dates.dayofyear.to_numpy()
    base = (dates.year >= cfg["heatwave"]["baseline"][0]) & (dates.year <= cfg["heatwave"]["baseline"][1])
    clim = {}
    for d in np.unique(doy):                     # önce tüm klimatolojiler (bellek için yerinde çıkarma)
        w = base & (np.abs(doy - d) <= 7)
        clim[d] = (Z[w].mean(axis=0), T[w].mean(axis=0))
    for d, (cz, ct) in clim.items():
        idx = doy == d
        Z[idx] -= cz; T[idx] -= ct
    za, ta = Z, T
    # eğilimden arındırılmış kopyalar: her grid noktasında yıllık mevsim ortalaması anomalisine doğrusal eğilim
    yrs = dates.year.to_numpy(); uy = np.unique(yrs)
    zdt, tdt = za.copy(), ta.copy()
    for A, out_ in ((za, zdt), (ta, tdt)):
        ym = np.stack([A[yrs == y].mean(axis=0) for y in uy])            # yıl x lat x lon
        X = np.vstack([uy - uy.mean(), np.ones(len(uy))]).T
        coef, *_ = np.linalg.lstsq(X, ym.reshape(len(uy), -1), rcond=None)
        trend = (X @ coef).reshape(ym.shape)
        for k, y in enumerate(uy):
            out_[yrs == y] -= trend[k]
    log.info("Eğilimden arındırma tamam")

    # 4) olay başlangıç tarihleri
    ev = pd.read_csv(ROOT / "data" / "processed" / "events.csv", parse_dates=["start"])
    ev = ev[ev.start.dt.month.isin(cfg["heatwave"]["season_months"])]
    onsets = pd.DatetimeIndex(sorted(ev.start.unique()))
    log.info("%d olay, %d farklı başlangıç tarihi", len(ev), len(onsets))
    pos = pd.Series(np.arange(len(dates)), index=dates)

    rows, meta = [], []
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    for L in LAGS:
        sel = [pos[d + pd.Timedelta(days=L)] for d in onsets if (d + pd.Timedelta(days=L)) in pos.index]
        zs, ts, zd, td = za[sel], ta[sel], zdt[sel], tdt[sel]
        p = lambda a: stats.ttest_1samp(a, 0.0, axis=0).pvalue.ravel().round(5)
        rows.append(pd.DataFrame({"lag": L, "lat": LA.ravel(), "lon": LO.ravel(),
                                  "z500_an": zs.mean(0).ravel().round(2), "t850_an": ts.mean(0).ravel().round(3),
                                  "z500_p": p(zs), "t850_p": p(ts),
                                  "z500_an_dt": zd.mean(0).ravel().round(2), "t850_an_dt": td.mean(0).ravel().round(3),
                                  "z500_p_dt": p(zd), "t850_p_dt": p(td)}))
        meta.append({"lag": L, "n_dates": len(sel)})
        box = (LA >= 36.5) & (LA <= 38.5) & (LO >= 36.5) & (LO <= 43)
        log.info("Gecikme %+d gün: %d tarih | GDA ort. Z500 anomalisi %.1f gpm (eğilimden arındırılmış %.1f)",
                 L, len(sel), zs.mean(0)[box].mean(), zd.mean(0)[box].mean())
    pd.concat(rows).to_csv(out / "composites.csv", index=False)
    pd.DataFrame(meta).to_csv(out / "composites_meta.csv", index=False)
    log.info("Bitti: results/analysis/composites.csv")


if __name__ == "__main__":
    main()
