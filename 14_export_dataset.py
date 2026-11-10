"""Aşama 14 - Zenodo'ya yüklenecek işlenmiş veri paketini hazırlar.

Çıktı: zenodo/heatwave_gda_processed_v1.zip
İçerik:
  era5_daily.csv       9 il, Türkiye saatine göre günlük ERA5-Land değişkenleri (1981-01-02 - 2024-12-31)
  pl_daily.csv         il hücresi ve bölge ortalaması Z500/Z850/T500/T850 serileri (Nisan-Eylül, 12:00 TR)
  soil_points.csv      0-7 ve 7-28 cm toprak nemi (12:00 TR)
  thresholds.csv       il ve takvim günü bazında 90. yüzdelik Tmax eşikleri (1981-2010)
  events.csv           sıcak hava dalgası olayları (başlangıç, bitiş, süre, şiddet)
  labels.csv           günlük etiketler (olay içi, teyit günü, y1/y3/y5/y7)
  features.csv.gz      modelde kullanılan 177 öznitelik ve hedefler (sıcak mevsim)
  README_data.md       sütun açıklamaları ve kaynak/lisans bilgisi

Kullanım:
    python 14_export_dataset.py
"""
from __future__ import annotations

import gzip
import shutil
import zipfile

from src.common import ROOT

FILES = {
    "era5_daily.csv": ROOT / "data" / "interim" / "era5_daily.csv",
    "pl_daily.csv": ROOT / "data" / "interim" / "pl_daily.csv",
    "soil_points.csv": ROOT / "data" / "interim" / "soil_points.csv",
    "thresholds.csv": ROOT / "data" / "processed" / "thresholds.csv",
    "events.csv": ROOT / "data" / "processed" / "events.csv",
    "labels.csv": ROOT / "data" / "processed" / "labels.csv",
}

README = """# Güneydoğu Anadolu sıcak hava dalgası tahmini - işlenmiş veri seti (v1)

Bu paket, "Güneydoğu Anadolu'da Sıcak Hava Dalgalarının Makine Öğrenmesi ile Erken Tahmini" makalesinde
kullanılan işlenmiş verileri içerir. Kod: https://github.com/AhmetBLC/heatwave

## Dosyalar
- era5_daily.csv: date, point_id, tmax, tmin, tmean (°C), tdew_mean (°C), rh_mean, rh_min (%), vpd_max (kPa),
  skt_max (°C), sp_mean (hPa), ws_mean, ws_max (m/s), ssrd_sum (MJ/m2), tp_sum (mm), dtr (°C), swvl1, swvl2 (m3/m3).
  Günler Türkiye saatine (UTC+3) göredir.
- pl_daily.csv: date, point_id, il hücresinde z500, z850 (gpm), t500, t850 (°C), thick (gpm) ve bölge ortalamaları
  z500_<bölge>, t850_<bölge> (gda, anat, emed, iran, north, anat_n), z500_grad_we. 12:00 TR, Nisan-Eylül.
- soil_points.csv: il hücresinde 12:00 TR toprak nemi.
- thresholds.csv: point_id, doy (1-365), thr90 (°C).
- events.csv: point_id, start, end, length, peak_tmax, mean_excess, sum_excess.
- labels.csv: date, point_id, tmax, thr90, excess, exceed, in_event, event_start, run_len, event_confirm, y1, y3, y5, y7.
- features.csv.gz: modelde kullanılan 177 öznitelik (pl_ önekliler büyük ölçekli), y1-y7 hedefleri, sıcak mevsim günleri.

## Kaynaklar ve lisans
Contains modified Copernicus Climate Change Service information (1981-2024). Neither the European Commission nor
ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
ERA5 ve ERA5-Land verileri Copernicus lisansına tabidir. Bu türetilmiş veri seti CC BY 4.0 ile paylaşılmıştır.
"""


def main():
    out = ROOT / "zenodo"; out.mkdir(exist_ok=True)
    feat = ROOT / "data" / "processed" / "features.csv"
    gz = out / "features.csv.gz"
    with open(feat, "rb") as s, gzip.open(gz, "wb", compresslevel=9) as d:
        shutil.copyfileobj(s, d)
    (out / "README_data.md").write_text(README, encoding="utf-8")
    z = out / "heatwave_gda_processed_v1.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, path in FILES.items():
            zf.write(path, name)
            print("eklendi:", name)
        zf.write(gz, "features.csv.gz"); zf.write(out / "README_data.md", "README_data.md")
    print(f"Hazır: {z} ({z.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
