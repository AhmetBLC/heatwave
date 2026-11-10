"""Aşama 1f - Bağımsız doğrulama için NOAA GHCN-Daily istasyon verilerini indirir.

Hakem önerisi: NASA POWER da bir yeniden analiz (MERRA-2) olduğundan bağımsız doğrulama sayılmaz;
ERA5-Land'den elde edilen sıcak hava dalgaları istasyon gözlemleriyle karşılaştırılmalı.

Ne yapar:
  1) GHCN-Daily istasyon listesini ve envanterini indirir.
  2) Çalışma alanını kapsayan kutudaki (36,0-39,0°K, 36,0-43,5°D) ve 1981 sonrasında TMAX
     kaydı olan Türkiye istasyonlarını seçer.
  3) Her istasyonun günlük CSV dosyasını indirir.
Çıktı:
  data/raw/ghcn/stations.csv     (seçilen istasyonlar: id, ad, enlem, boylam, yükselti, TMAX yılları)
  data/raw/ghcn/<istasyon_id>.csv (NOAA biçiminde günlük kayıt; TMAX onda bir °C)

Kullanım:
    python 01f_download_ghcn.py
İnternet gerekir, hesap gerekmez (NOAA NCEI açık erişim).
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from src.common import ROOT, get_logger, load_config

BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
ACCESS = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"
BOX = (36.0, 39.0, 36.0, 43.5)   # Güney, Kuzey, Batı, Doğu


def get(url: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def main() -> None:
    cfg = load_config()
    log = get_logger("01f_ghcn", cfg)
    out = ROOT / "data" / "raw" / "ghcn"; out.mkdir(parents=True, exist_ok=True)

    log.info("İstasyon listesi indiriliyor...")
    st = pd.read_fwf(io.StringIO(get(f"{BASE}/ghcnd-stations.txt")),
                     colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (41, 71)],
                     names=["id", "lat", "lon", "elev", "name"])
    inv = pd.read_fwf(io.StringIO(get(f"{BASE}/ghcnd-inventory.txt")),
                      colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
                      names=["id", "lat", "lon", "element", "first", "last"])
    inv = inv[inv.element == "TMAX"][["id", "first", "last"]]
    s, n, w, e = BOX
    sel = st[(st.lat >= s) & (st.lat <= n) & (st.lon >= w) & (st.lon <= e) & st.id.str.startswith("TU")]
    sel = sel.merge(inv, on="id")
    sel = sel[sel["last"] >= 1990]
    sel.to_csv(out / "stations.csv", index=False)
    log.info("%d istasyon seçildi:\n%s", len(sel), sel.to_string(index=False))

    for sid in sel.id:
        f = out / f"{sid}.csv"
        if f.exists() and f.stat().st_size > 0:
            log.info("  %s zaten var, atlanıyor", sid)
            continue
        try:
            txt = get(f"{ACCESS}/{sid}.csv")
            f.write_text(txt, encoding="utf-8")
            log.info("  %s indirildi (%d satır)", sid, txt.count("\n"))
        except Exception as ex:
            log.warning("  %s indirilemedi: %s", sid, ex)
    log.info("Bitti: %s", out)


if __name__ == "__main__":
    main()
