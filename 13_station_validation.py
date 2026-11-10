"""Aşama 13 - ERA5-Land sıcak hava dalgalarının istasyon gözlemleriyle doğrulanması (NOAA GHCN-Daily).

Girdi : data/raw/ghcn/<istasyon>.csv (01f çıktısı), data/interim/era5_daily.csv, data/processed/labels.csv
Yöntem:
  - İl merkezine en yakın GHCN-Daily istasyonu eşleştirilir (aşağıdaki STATIONS sözlüğü).
  - Kalite bayrağı (Q-flag) taşıyan TMAX değerleri atılır; Mayıs-Eylül'de %80'den az verisi olan yıllar dışarıda kalır.
  - Her iki veri için de AYNI eşik dönemi ve AYNI günler kullanılır (istasyon verisi olmayan günler ERA5-Land'den de çıkarılır).
  - İstasyonun kendi 1981-2010 takvim günü 90. yüzdeliği (15 günlük pencere) ile, ERA5-Land'deki tanımın aynısı
    uygulanarak istasyon sıcak hava dalgaları bulunur. 1981-2010'da 20 yıldan az verisi olan istasyonlarda eşik,
    istasyonun tüm kullanılabilir yıllarından hesaplanır (tabloda belirtilir).
  - Günlük Tmax: korelasyon, ortalama fark, RMSE, anomali korelasyonu (Mayıs-Eylül).
  - Olay günleri: isabet oranı (istasyon olay günlerinin ERA5-Land'de de olay günü olma oranı), yanlış alarm oranı, CSI.
  - Olay düzeyi: istasyon olaylarının ±1 gün içinde bir ERA5-Land olayıyla örtüşme oranı (ve tersi).
Çıktı : results/revision/station_validation.csv

Kullanım:
    python 13_station_validation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, load_config

STATIONS = {"gaziantep": "TUM00017260", "adiyaman": "TUM00017265", "sanliurfa": "TUM00017270",
            "diyarbakir": "TUM00017280", "batman": "TUM00017282", "siirt": "TUM00017210"}
SEASON = [5, 6, 7, 8, 9]


def doy365(idx):
    md = np.where((idx.month == 2) & (idx.day == 29), 28, idx.day)
    return pd.to_datetime({"year": 2001, "month": idx.month, "day": md}).dt.dayofyear.to_numpy()


def thresholds(s, years, q=90, half=7):
    b = s[(s.index.year >= years[0]) & (s.index.year <= years[1])].dropna()
    doy = doy365(b.index); thr = np.full(365, np.nan)
    for k in range(1, 366):
        dist = np.abs(((doy - k) + 182) % 365 - 182)
        v = b.values[dist <= half]
        if len(v) > 50:
            thr[k - 1] = np.percentile(v, q)
    return thr


def events(ex, min_len=3):
    """Ardışık en az min_len günlük aşım dizileri; NaN dizileri böler."""
    out, start = [], None
    for i, e in enumerate(ex):
        if e:
            start = i if start is None else start
        else:
            if start is not None and i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(ex) - start >= min_len:
        out.append((start, len(ex) - 1))
    return out


def load_station(sid):
    f = ROOT / "data" / "raw" / "ghcn" / f"{sid}.csv"
    d = pd.read_csv(f, usecols=["DATE", "TMAX", "TMAX_ATTRIBUTES"], parse_dates=["DATE"], low_memory=False)
    d = d.dropna(subset=["TMAX"])
    q = d.TMAX_ATTRIBUTES.fillna(",,").str.split(",").str[1].fillna("")
    d = d[q == ""]
    s = pd.Series(d.TMAX.astype(float).values / 10.0, index=d.DATE).sort_index()
    return s[~s.index.duplicated()]


def main():
    cfg = load_config()
    log = get_logger("13_station", cfg)
    E = pd.read_csv(ROOT / "data" / "interim" / "era5_daily.csv", parse_dates=["date"], usecols=["date", "point_id", "tmax"])
    L = pd.read_csv(ROOT / "data" / "processed" / "labels.csv", parse_dates=["date"], usecols=["date", "point_id", "in_event"])
    rows = []
    for pid, sid in STATIONS.items():
        s = load_station(sid)
        s = s[(s.index.year >= 1981) & (s.index.year <= 2024)]
        full = pd.date_range("1981-01-01", "2024-12-31", freq="D")
        s = s.reindex(full)
        seas = s[s.index.month.isin(SEASON)]
        cov = seas.notna().groupby(seas.index.year).mean()
        good = cov[cov >= 0.8].index
        nb = int(((good >= 1981) & (good <= 2010)).sum())
        base = (1981, 2010) if nb >= 20 else (int(good.min()), int(good.max()))
        thr = thresholds(s, base)
        st = pd.DataFrame({"tmax_st": s})
        st["thr"] = thr[doy365(st.index) - 1]
        ex = (st.tmax_st > st.thr).to_numpy()
        ex_nan = st.tmax_st.isna().to_numpy()
        ev_st = events(np.where(ex_nan, False, ex))
        st["ev_st"] = 0
        for a, b in ev_st:
            st.iloc[a:b + 1, st.columns.get_loc("ev_st")] = 1
        e = E[E.point_id == pid].set_index("date").tmax.reindex(full)
        # Adil karşılaştırma: ERA5-Land olayları da AYNI eşik dönemi ve AYNI günlerle (istasyon verisi olan günler) yeniden bulunur
        e_m = e.where(s.notna())
        thr_e = thresholds(e_m, base)
        ex_e = (e_m > thr_e[doy365(e_m.index) - 1]).to_numpy()
        st["ev_era"] = 0
        for a_, b_ in events(np.where(e_m.isna().to_numpy(), False, ex_e)):
            st.iloc[a_:b_ + 1, st.columns.get_loc("ev_era")] = 1
        lab = L[L.point_id == pid].set_index("date").in_event.reindex(full)
        D = st.assign(tmax_era=e, ev_era_main=lab)
        D = D[D.index.month.isin(SEASON) & D.index.year.isin(good) & D.tmax_st.notna()]
        # günlük Tmax karşılaştırması
        r = np.corrcoef(D.tmax_st, D.tmax_era)[0, 1]
        bias = (D.tmax_era - D.tmax_st).mean(); rmse = np.sqrt(((D.tmax_era - D.tmax_st) ** 2).mean())
        doy = doy365(D.index)
        an_st = D.tmax_st - pd.Series(D.tmax_st.values).groupby(doy).transform("mean").values
        an_er = D.tmax_era - pd.Series(D.tmax_era.values).groupby(doy).transform("mean").values
        ra = np.corrcoef(an_st, an_er)[0, 1]
        # olay günleri
        a, b = D.ev_st.to_numpy() == 1, D.ev_era.to_numpy() == 1
        hit = (a & b).sum() / max(a.sum(), 1); far = (~a & b).sum() / max(b.sum(), 1)
        csi = (a & b).sum() / max((a | b).sum(), 1)
        # olay düzeyi (±1 gün örtüşme)
        idx = D.index
        def ev_list(col):
            v = D[col].to_numpy() == 1; out = []; i = 0
            while i < len(v):
                if v[i]:
                    j = i
                    while j + 1 < len(v) and v[j + 1] and (idx[j + 1] - idx[j]).days == 1:
                        j += 1
                    out.append((idx[i], idx[j])); i = j + 1
                else:
                    i += 1
            return out
        es, ee = ev_list("ev_st"), ev_list("ev_era")
        ov = lambda x, Y: any(not (x[1] + pd.Timedelta(days=1) < y[0] or y[1] + pd.Timedelta(days=1) < x[0]) for y in Y)
        m_st = np.mean([ov(x, ee) for x in es]) if es else np.nan
        m_er = np.mean([ov(y, es) for y in ee]) if ee else np.nan
        # on yıllık olay sayısı eğilimi (yıl başına, iki veri)
        yrs = np.array(sorted(D.index.year.unique()))
        n_st = np.array([sum(1 for x in es if x[0].year == y) for y in yrs]); n_er = np.array([sum(1 for x in ee if x[0].year == y) for y in yrs])
        tr_st = 10 * np.polyfit(yrs, n_st, 1)[0] if len(yrs) > 10 else np.nan
        tr_er = 10 * np.polyfit(yrs, n_er, 1)[0] if len(yrs) > 10 else np.nan
        rows.append({"point_id": pid, "station": sid, "years_used": len(good), "first_year": int(good.min()),
                     "baseline": f"{base[0]}-{base[1]}", "n_days": len(D), "r_tmax": r, "bias_era_minus_st": bias, "rmse": rmse,
                     "r_anomaly": ra, "event_days_st": int(a.sum()), "event_days_era": int(b.sum()), "hit_rate": hit,
                     "false_alarm_ratio": far, "csi": csi, "n_events_st": len(es), "n_events_era": len(ee),
                     "share_st_events_matched": m_st, "share_era_events_matched": m_er,
                     "trend_st_per_decade": tr_st, "trend_era_per_decade": tr_er})
        log.info("%-10s %s (%d yıl, eşik %s-%s): r=%.3f, anomali r=%.3f, fark %+.2f °C, RMSE %.2f | olay günü isabet %.2f, "
                 "yanlış alarm %.2f, CSI %.2f | istasyon olaylarının %%%.0f'i, ERA5-Land olaylarının %%%.0f'i eşleşiyor | "
                 "eğilim (olay/on yıl) istasyon %.2f, ERA5-Land %.2f",
                 pid, sid, len(good), *base, r, ra, bias, rmse, hit, far, csi, 100 * m_st, 100 * m_er, tr_st, tr_er)
    R = pd.DataFrame(rows)
    out = ROOT / "results" / "revision"; out.mkdir(parents=True, exist_ok=True)
    R.round(4).to_csv(out / "station_validation.csv", index=False)
    log.info("Ortalama (il): r=%.3f, anomali r=%.3f, isabet %.2f, CSI %.2f, olay eşleşme %%%.0f",
             R.r_tmax.mean(), R.r_anomaly.mean(), R.hit_rate.mean(), R.csi.mean(), 100 * R.share_st_events_matched.mean())


if __name__ == "__main__":
    main()
