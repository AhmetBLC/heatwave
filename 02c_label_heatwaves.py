"""Aşama 2c - Sıcak hava dalgalarını tespit eder ve tahmin hedeflerini (etiketleri) üretir.

Tanım (config.yaml -> heatwave):
  - Eşik: her nokta ve takvim günü için, baseline döneminde (1981-2010) o günü ortalayan
    15 günlük penceredeki Tmax değerlerinin 90. yüzdeliği (Perkins & Alexander, 2013 yaklaşımı).
  - Sıcak hava dalgası: Tmax > eşik koşulunun en az 3 gün üst üste sağlanması.
Hedef (yalnızca t gününe kadar bilinen bilgiyle tanımlanır; geleceğe bakan örnek seçimi yok):
  - Teyit günü: bir sıcak hava dalgasının 3. günü (koşulun ilk kez sağlandığı gün).
  - y_k(t) = 1, eğer t+1 ... t+k günlerinden birinde bir sıcak hava dalgası TEYİT ediliyorsa
    (sürmekte olan 1-2 günlük bir aşım serisinin dalgaya dönüşmesi de buna dahildir).
  - Örnek kuralı: t gününe kadar üst üste aşım sayısı < 3 ise (yani t'de dalga henüz
    teyit edilmemiş ya da sürmüyorsa) t bir örnektir; aksi halde y_k = boş.
  - Örnekler yalnızca mevsim aylarındaki (Mayıs-Eylül) günlerdir.

Girdi : data/interim/era5_daily.csv
Çıktı : data/processed/labels.csv     (günlük: eşik, aşım, seri uzunluğu, olay içinde mi, y1, y3, y7)
        data/processed/thresholds.csv (nokta x takvim günü eşikleri, tüm yıl)
        data/processed/events.csv     (her sıcak hava dalgası: başlangıç, bitiş, süre, şiddet)
        data/processed/validation_summary.csv  (il ve on yıl bazında yıllık olay sayısı)

Kullanım:
    python 02c_label_heatwaves.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, load_config


def doy365(dates: pd.DatetimeIndex) -> np.ndarray:
    """29 Şubat'ı 28 Şubat'a eşleyen 1..365 gün indeksi."""
    d = pd.DatetimeIndex(dates)
    md = np.where((d.month == 2) & (d.day == 29), 28, d.day)
    ref = pd.to_datetime({"year": 2001, "month": d.month, "day": md})
    return ref.dt.dayofyear.to_numpy()


def thresholds(tmax: pd.Series, base: tuple[int, int], pct: float, window: int) -> np.ndarray:
    """365 elemanlı eşik dizisi (takvim günü başına)."""
    b = tmax[(tmax.index.year >= base[0]) & (tmax.index.year <= base[1])]
    doy = doy365(b.index)
    half = window // 2
    thr = np.empty(365)
    for k in range(1, 366):
        dist = np.abs(((doy - k) + 182) % 365 - 182)   # dairesel uzaklık
        thr[k - 1] = np.percentile(b.values[dist <= half], pct)
    return thr


def find_events(exceed: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    events, i, n = [], 0, len(exceed)
    while i < n:
        if exceed[i]:
            j = i
            while j + 1 < n and exceed[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                events.append((i, j))
            i = j + 1
        else:
            i += 1
    return events


def main() -> None:
    cfg = load_config()
    hw = cfg["heatwave"]
    log = get_logger("02c_labels", cfg)
    daily = pd.read_csv(ROOT / "data" / "interim" / "era5_daily.csv", parse_dates=["date"])
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    names = {p["id"]: p["name"] for p in cfg["points"]}
    season = set(hw["season_months"])
    H = hw["horizons"]

    lab_all, ev_all, thr_rows = [], [], []
    for pid, d in daily.groupby("point_id"):
        d = d.set_index("date").sort_index()
        # tarih sürekliliği kontrolü
        full = pd.date_range(d.index.min(), d.index.max(), freq="D")
        if len(full) != len(d):
            log.warning("%s: %d gün eksik", pid, len(full) - len(d))
        thr = thresholds(d["tmax"], tuple(hw["baseline"]), hw["percentile"], hw["window_days"])
        t = pd.DataFrame(index=d.index)
        t["tmax"] = d["tmax"]
        t["thr90"] = thr[doy365(d.index) - 1]
        t["excess"] = t["tmax"] - t["thr90"]
        ex = (t["excess"] > 0).to_numpy()
        t["exceed"] = ex.astype(int)
        events = find_events(ex, hw["min_length"])

        in_ev = np.zeros(len(t), dtype=bool)
        start = np.zeros(len(t), dtype=bool)
        for a, b in events:
            in_ev[a:b + 1] = True
            start[a] = True
            seg = t.iloc[a:b + 1]
            ev_all.append({"point_id": pid, "start": seg.index[0].date(), "end": seg.index[-1].date(),
                           "length": b - a + 1, "peak_tmax": round(seg.tmax.max(), 2),
                           "mean_excess": round(seg.excess.mean(), 2), "sum_excess": round(seg.excess.sum(), 2)})
        t["in_event"] = in_ev.astype(int)
        t["event_start"] = start.astype(int)

        # seri uzunluğu: t gününde biten ardışık aşım sayısı (t'de bilinir)
        run = np.zeros(len(t), dtype=int)
        for i in range(len(t)):
            run[i] = (run[i - 1] + 1 if i else 1) if ex[i] else 0
        t["run_len"] = run
        # teyit günleri: her olayın (min_length)'inci günü
        conf = np.zeros(len(t), dtype=int)
        for a, b in events:
            conf[a + hw["min_length"] - 1] = 1
        t["event_confirm"] = conf
        cs = np.concatenate([[0], np.cumsum(conf)])
        n = len(conf)
        idx = np.arange(n)
        sample = run < hw["min_length"]
        for k in H:
            hi = np.minimum(idx + k, n - 1)
            y = (cs[hi + 1] - cs[idx + 1]) > 0
            valid = (idx + k <= n - 1) & sample
            t[f"y{k}"] = np.where(valid, y.astype(float), np.nan)
        thr_rows.append(pd.DataFrame({"point_id": pid, "doy": np.arange(1, 366), "thr90": np.round(thr, 4)}))
        t = t[t.index.month.isin(season)]
        t.insert(0, "point_id", pid)
        lab_all.append(t)

    lab = pd.concat(lab_all).reset_index().rename(columns={"index": "date"})
    ev = pd.DataFrame(ev_all)
    ev["start"] = pd.to_datetime(ev["start"])
    num = lab.select_dtypes("number").columns
    lab[num] = lab[num].round(4)
    lab.to_csv(out / "labels.csv", index=False)
    ev.to_csv(out / "events.csv", index=False)
    pd.concat(thr_rows).to_csv(out / "thresholds.csv", index=False)

    # --- doğrulama özeti: mevsim içinde başlayan olaylar ---
    evs = ev[ev.start.dt.month.isin(season)].copy()
    evs["year"] = evs.start.dt.year
    years = range(daily.date.dt.year.min(), daily.date.dt.year.max() + 1)
    rows = []
    for pid in sorted(lab.point_id.unique()):
        e = evs[evs.point_id == pid]
        cnt = e.groupby("year").size().reindex(years, fill_value=0)
        days = e.groupby("year")["length"].sum().reindex(years, fill_value=0)
        slope = np.polyfit(np.array(list(years)), cnt.values, 1)[0] * 10
        row = {"point_id": pid, "name": names.get(pid, pid),
               "events_per_year": round(cnt.mean(), 2), "hw_days_per_year": round(days.mean(), 1),
               "trend_events_per_decade": round(slope, 2)}
        for dec in range(1980, 2030, 10):
            row[f"{dec}s"] = round(cnt[(cnt.index >= dec) & (cnt.index < dec + 10)].mean(), 2)
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "validation_summary.csv", index=False)

    log.info("Toplam %d sıcak hava dalgası (mevsim içinde başlayan: %d)", len(ev), len(evs))
    log.info("\n%s", summ.to_string(index=False))
    for k in H:
        v = lab[f"y{k}"]
        log.info("y%d: %d örnek | pozitif oranı %.1f%% | pozitif sayısı %d", k, int(v.notna().sum()), 100 * v.mean(), int(v.sum()))
    base = lab[(lab.date.dt.year >= hw["baseline"][0]) & (lab.date.dt.year <= hw["baseline"][1])]
    log.info("Kontrol: baseline döneminde mevsim içi eşik aşım oranı %.1f%% (yaklaşık %%10 beklenir)", 100 * base.exceed.mean())


if __name__ == "__main__":
    main()
