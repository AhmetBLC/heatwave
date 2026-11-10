"""Aşama 3 - Model öznitelikleri.

Her öznitelik yalnızca t günü ve öncesindeki bilgiyle hesaplanır (geleceğe bakılmaz).
Anomaliler, EĞİTİM dönemi (1981-2010) klimatolojisine göre hesaplanır; böylece
doğrulama/test dönemlerinin bilgisi özniteliklere sızmaz.

Girdi : data/interim/era5_daily.csv, data/processed/labels.csv
        data/interim/pl_daily.csv (varsa; 02d çıktısı, büyük ölçekli öncüller)
Çıktı : data/processed/features.csv   (yalnızca mevsim günleri; hedefler y1, y3, y5, y7 dahil)
Büyük ölçekli öznitelikler "pl_" önekini taşır; 04'te "yalnız yerel" ve "yerel + büyük ölçekli" karşılaştırması bu önekle yapılır.

Öznitelik grupları:
  ham        : t günü değerleri (tmax, tmin, rh_mean, vpd_max, swvl1, ...)
  anomali    : <değişken>_an = değer - takvim günü klimatolojisi (±7 gün, 1981-2010)
  eşik       : excess (tmax - eşik), exceed, son 3/7 gündeki aşım sayısı
  gecikme    : anomalilerin t-1, t-2, t-3 değerleri ve 3/7/14 günlük ortalamaları
  eğilim     : tmax'ın son 3 gündeki değişimi
  yağış      : son 7/14/30 günlük toplam yağış, son yağıştan bu yana geçen gün
  bellek     : son sıcak hava dalgasının bitişinden bu yana geçen gün
  mevsim/yer : doy_sin, doy_cos, lat, lon
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, load_config

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

ANOM_VARS = ["tmax", "tmin", "tmean", "dtr", "tdew_mean", "rh_mean", "rh_min", "vpd_max",
             "skt_max", "sp_mean", "ws_mean", "ssrd_sum", "swvl1", "swvl2"]
PL_VARS = ["z500", "z850", "t500", "t850", "thick", "z500_gda", "z500_anat", "z500_emed", "z500_iran", "z500_north",
           "t850_gda", "t850_anat", "t850_emed", "t850_iran", "t850_north", "z500_grad_we"]
PL_LAG_VARS = ["pl_z500_an", "pl_t850_an", "pl_thick_an", "pl_z500_gda_an", "pl_z500_emed_an", "pl_z500_iran_an",
               "pl_z500_north_an", "pl_z500_anat_an", "pl_t850_gda_an", "pl_z500_grad_we_an"]
LAG_VARS = ["tmax_an", "tmin_an", "rh_mean_an", "vpd_max_an", "sp_mean_an", "swvl1_an", "swvl2_an", "skt_max_an"]


def doy365(idx: pd.DatetimeIndex) -> np.ndarray:
    md = np.where((idx.month == 2) & (idx.day == 29), 28, idx.day)
    return pd.to_datetime({"year": 2001, "month": idx.month, "day": md}).dt.dayofyear.to_numpy()


def climatology(s: pd.Series, years: tuple[int, int], half: int = 7) -> np.ndarray:
    b = s[(s.index.year >= years[0]) & (s.index.year <= years[1])].dropna()
    doy = doy365(b.index)
    clim = np.full(365, np.nan)
    for k in range(1, 366):
        dist = np.abs(((doy - k) + 182) % 365 - 182)
        v = b.values[dist <= half]
        if len(v):
            clim[k - 1] = v.mean()
    return clim


def days_since(flag: np.ndarray, cap: int = 365) -> np.ndarray:
    out = np.full(len(flag), cap, dtype=float)
    last = None
    for i, f in enumerate(flag):
        if f:
            last = i
        if last is not None:
            out[i] = min(i - last, cap)
    return out


def main() -> None:
    cfg = load_config()
    log = get_logger("03_features", cfg)
    train = tuple(cfg["split"]["train"])
    season = cfg["heatwave"]["season_months"]
    daily = pd.read_csv(ROOT / "data" / "interim" / "era5_daily.csv", parse_dates=["date"])
    lab = pd.read_csv(ROOT / "data" / "processed" / "labels.csv", parse_dates=["date"])
    ev = pd.read_csv(ROOT / "data" / "processed" / "events.csv", parse_dates=["start", "end"])
    thr_tab = pd.read_csv(ROOT / "data" / "processed" / "thresholds.csv")
    coords = {p["id"]: (p["lat"], p["lon"]) for p in cfg["points"]}
    ycols = [f"y{k}" for k in cfg["heatwave"]["horizons"]]
    pl_f = ROOT / "data" / "interim" / "pl_daily.csv"
    has_pl = pl_f.exists()
    if has_pl:
        pl = pd.read_csv(pl_f, parse_dates=["date"])
        pl = pl.rename(columns={c: f"pl_{c}" for c in PL_VARS})
        daily = daily.merge(pl, on=["date", "point_id"], how="left")
        log.info("Büyük ölçekli öncüller eklendi (%s)", pl_f.name)
    else:
        log.warning("pl_daily.csv yok: yalnızca yerel öznitelikler üretilecek")

    out = []
    for pid, d in daily.groupby("point_id"):
        d = d.set_index("date").sort_index()
        f = pd.DataFrame(index=d.index)
        doy = doy365(d.index)
        for v in ANOM_VARS:
            f[v] = d[v]
            f[f"{v}_an"] = d[v] - climatology(d[v], train)[doy - 1]

        # eşik: tüm yıl için 02c'nin kaydettiği takvim günü eşikleri
        T = thr_tab[thr_tab.point_id == pid].set_index("doy")["thr90"]
        thr = T.reindex(doy).to_numpy()
        L = lab[lab.point_id == pid].set_index("date")
        f["excess"] = d["tmax"].to_numpy() - thr
        f["exceed"] = (f["excess"] > 0).astype(int)
        f["run_len"] = f["exceed"].groupby((f["exceed"] == 0).cumsum()).cumsum()
        f["exceed_cnt3"] = f["exceed"].rolling(3, min_periods=1).sum()
        f["exceed_cnt7"] = f["exceed"].rolling(7, min_periods=1).sum()

        for v in LAG_VARS:
            for k in (1, 2, 3):
                f[f"{v}_lag{k}"] = f[v].shift(k)
            for w in (3, 7, 14):
                f[f"{v}_mean{w}"] = f[v].rolling(w, min_periods=max(1, w // 2)).mean()
        if has_pl:
            for v in PL_VARS:
                c = f"pl_{v}"
                f[c] = d[c]
                f[f"{c}_an"] = d[c] - climatology(d[c], train)[doy - 1]
            for v in PL_LAG_VARS:
                for k in (1, 2, 3):
                    f[f"{v}_lag{k}"] = f[v].shift(k)
                for w in (3, 7):
                    f[f"{v}_mean{w}"] = f[v].rolling(w, min_periods=max(1, w // 2)).mean()
            f["pl_z500_trend3"] = d["pl_z500"] - d["pl_z500"].shift(3)
            f["pl_z500_gda_trend3"] = d["pl_z500_gda"] - d["pl_z500_gda"].shift(3)
            f["pl_t850_trend3"] = d["pl_t850"] - d["pl_t850"].shift(3)
        f["tmax_trend3"] = d["tmax"] - d["tmax"].shift(3)
        f["swvl1_change7"] = d["swvl1"] - d["swvl1"].shift(7)

        for w in (7, 14, 30):
            f[f"tp_sum{w}"] = d["tp_sum"].rolling(w, min_periods=1).sum()
        f["days_since_rain"] = days_since((d["tp_sum"] > 1.0).to_numpy())

        # son sıcak hava dalgasının bitişinden bu yana (yalnızca t'den önce BİTMİŞ olaylar)
        ended = np.zeros(len(d), dtype=bool)
        e = ev[ev.point_id == pid]
        pos = d.index.get_indexer(e["end"])
        ended[pos[pos >= 0]] = True
        f["days_since_hw"] = np.concatenate([[365], days_since(ended)[:-1]])

        f["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365)
        f["lat"], f["lon"] = coords[pid]

        f = f[f.index.month.isin(season)]
        f = f.join(L[["in_event", *ycols]], how="left")
        f.insert(0, "point_id", pid)
        out.append(f)

    F = pd.concat(out).reset_index().rename(columns={"index": "date"})
    F["year"] = F.date.dt.year
    path = ROOT / "data" / "processed" / "features.csv"
    F.to_csv(path, index=False)
    feats = [c for c in F.columns if c not in ("date", "point_id", "year", "in_event", *ycols)]
    na = F[feats].isna().mean().sort_values(ascending=False)
    log.info("Öznitelik tablosu: %d satır, %d öznitelik (%d büyük ölçekli) -> %s",
             len(F), len(feats), sum(c.startswith("pl_") for c in feats), path)
    log.info("En çok eksik olan öznitelikler: %s", na.head(5).round(4).to_dict())


if __name__ == "__main__":
    main()
