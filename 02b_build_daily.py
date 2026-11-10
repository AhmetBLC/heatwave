"""Aşama 2b - Saatlik ERA5-Land nokta serilerinden Türkiye saatine (UTC+3) göre günlük veri üretir.

Girdi : data/raw/era5land_ts/<nokta>/*.csv      (saatlik: t2m, d2m, skt, sp, u10, v10, ssrd, tp)
        data/interim/soil_points.csv           (günlük 12:00 TR toprak nemi, 02a çıktısı; varsa)
Çıktı : data/interim/daily/<nokta>.csv          (nokta başına)
        data/interim/era5_daily.csv             (--merge ile hepsi tek dosya)

Günlük değişkenler:
  tmax, tmin, tmean      2 m sıcaklık (°C)
  dtr                    günlük sıcaklık farkı (tmax - tmin)
  tdew_mean              çiy noktası ortalaması (°C)
  rh_mean, rh_min        bağıl nem (%), saatlik t2m ve d2m'den Magnus formülüyle
  vpd_max                en yüksek buhar basıncı açığı (kPa)
  skt_max                en yüksek yüzey (deri) sıcaklığı (°C)
  sp_mean                yüzey basıncı ortalaması (hPa)
  ws_mean, ws_max        10 m rüzgâr hızı (m/s)
  ssrd_sum               günlük güneş radyasyonu (MJ/m²)
  tp_sum                 günlük yağış (mm); küçük negatif saatlik değerler 0'a çekilir
  swvl1, swvl2           toprak nemi 0-7 cm ve 7-28 cm (m³/m³), 12:00 TR
24 saati tam olmayan günler (serinin ilk/son günü) atılır.

Kullanım:
    python 02b_build_daily.py                    # tüm noktalar + birleştir
    python 02b_build_daily.py --points gaziantep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, load_config

UTC_OFFSET_H = 3


def es_kpa(t_c):
    """Doymuş buhar basıncı (kPa), Magnus (Alduchov & Eskridge, 1996)."""
    return 0.61094 * np.exp(17.625 * t_c / (t_c + 243.04))


def load_hourly(folder: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f, parse_dates=["valid_time"]).drop(columns=["latitude", "longitude"], errors="ignore")
        parts.append(d.set_index("valid_time"))
    df = pd.concat(parts, axis=1).sort_index()
    need = {"t2m", "d2m", "skt", "sp", "u10", "v10", "ssrd", "tp"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"{folder.name}: eksik sütunlar {miss}")
    return df


def to_daily(h: pd.DataFrame) -> pd.DataFrame:
    t = h["t2m"] - 273.15
    td = h["d2m"] - 273.15
    ea, es = es_kpa(td), es_kpa(t)
    x = pd.DataFrame({
        "t": t, "td": td,
        "rh": (100 * ea / es).clip(upper=100),
        "vpd": (es - ea).clip(lower=0),
        "skt": h["skt"] - 273.15,
        "sp": h["sp"] / 100,
        "ws": np.hypot(h["u10"], h["v10"]),
        "ssrd": h["ssrd"].clip(lower=0) / 1e6,
        "tp": h["tp"].clip(lower=0) * 1000,
    }, index=h.index)
    local_day = (x.index + pd.Timedelta(hours=UTC_OFFSET_H)).normalize()
    g = x.groupby(local_day)
    d = pd.DataFrame({
        "n_hours": g["t"].size(),
        "tmax": g["t"].max(), "tmin": g["t"].min(), "tmean": g["t"].mean(),
        "tdew_mean": g["td"].mean(),
        "rh_mean": g["rh"].mean(), "rh_min": g["rh"].min(),
        "vpd_max": g["vpd"].max(),
        "skt_max": g["skt"].max(),
        "sp_mean": g["sp"].mean(),
        "ws_mean": g["ws"].mean(), "ws_max": g["ws"].max(),
        "ssrd_sum": g["ssrd"].sum(),
        "tp_sum": g["tp"].sum(),
    })
    d["dtr"] = d["tmax"] - d["tmin"]
    d = d[d["n_hours"] == 24].drop(columns="n_hours")
    d.index.name = "date"
    return d.round(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="*")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("02b_daily", cfg)
    raw = ROOT / "data" / "raw" / "era5land_ts"
    out = ROOT / "data" / "interim" / "daily"
    out.mkdir(parents=True, exist_ok=True)
    soil_f = ROOT / "data" / "interim" / "soil_points.csv"
    soil = pd.read_csv(soil_f, parse_dates=["date"]) if soil_f.exists() else None
    if soil is None:
        log.warning("soil_points.csv yok; toprak nemi sütunları boş kalacak (önce 02a çalıştır)")
    else:
        log.info("Toprak nemi: %s – %s", soil.date.min().date(), soil.date.max().date())

    points = [p for p in cfg["points"] if not args.points or p["id"] in args.points]
    for p in points:
        h = load_hourly(raw / p["id"])
        d = to_daily(h)
        if soil is not None:
            s = soil[soil.point_id == p["id"]].set_index("date")[["swvl1", "swvl2"]]
            d = d.join(s, how="left")
        else:
            d["swvl1"] = np.nan; d["swvl2"] = np.nan
        d.insert(0, "point_id", p["id"])
        d.to_csv(out / f"{p['id']}.csv")
        jul = d[d.index.month == 7]
        log.info("%-11s %d gün | Temmuz ort. Tmax %.1f °C | maks Tmax %.1f °C (%s) | yıllık yağış %.0f mm | toprak nemi dolu gün: %d",
                 p["name"], len(d), jul.tmax.mean(), d.tmax.max(), d.tmax.idxmax().date(),
                 d.tp_sum.groupby(d.index.year).sum().mean(), int(d.swvl1.notna().sum()))

    if not args.no_merge:
        files = sorted(out.glob("*.csv"))
        allc = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
        allc.to_csv(ROOT / "data" / "interim" / "era5_daily.csv", index=False)
        log.info("Birleştirildi: %d nokta, %d satır -> data/interim/era5_daily.csv", allc.point_id.nunique(), len(allc))


if __name__ == "__main__":
    main()
