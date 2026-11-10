# Güneydoğu Anadolu Sıcak Hava Dalgası Tahmini — Aşama 1: Veri İndirme

## Klasör yapısı
```
heatwave-gda/
├── config.yaml                 # TÜM ayarlar (noktalar, dönem, değişkenler)
├── requirements.txt
├── 01a_download_era5land.py    # ERA5-Land günlük istatistikleri (Copernicus CDS)
├── 01b_download_nasapower.py   # NASA POWER günlük nokta verisi
├── 01c_check_downloads.py      # indirme kontrolü
├── src/common.py               # ortak yardımcılar
├── data/raw/era5land/          # <istatistik>/<yıl>_<ay>/*.nc olarak iner
├── data/raw/nasapower/         # .csv dosyaları buraya iner
└── logs/                       # her betiğin log dosyası
```

## 1. Kurulum (bir kez)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Copernicus CDS hesabı (bir kez)
1. https://cds.climate.copernicus.eu adresinde ücretsiz hesap aç.
2. Giriş yap → sağ üstte profil → **API Token**'ı kopyala.
3. Ana klasörüne (`C:\Users\<kullanıcı>\` veya `~/`) `.cdsapirc` adlı bir dosya oluştur:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <BURAYA-API-TOKEN>
   ```
   Not: Eski rehberlerdeki `/api/v2` adresi ve "UID:anahtar" biçimi artık çalışmıyor.
4. Veri seti sayfasını aç: https://cds.climate.copernicus.eu/datasets/derived-era5-land-daily-statistics
   → **Download** sekmesinin altındaki **lisansı kabul et** (CC-BY). Kabul etmezsen API 403 hatası verir.

## 3. Önce küçük bir deneme yap
```bash
python 01a_download_era5land.py --dry-run --years 1981   # istekleri yazdırır, indirmez
python 01a_download_era5land.py --years 1981             # tek yılı indir (18 istek, 6 ay x 3 istatistik)
python 01b_download_nasapower.py                         # 9 nokta, birkaç dakika
python 01c_check_downloads.py                            # kontrol (1981 dışındakiler "eksik" görünür, normal)
```
`01c` çıktısında her ilin en yakın ERA5-Land hücresine olan mesafesi yazılır; hepsi ~10 km'nin altında olmalı.

## 4. Tüm veriyi indir
```bash
python 01a_download_era5land.py
```
- 44 yıl × 6 ay × 3 istatistik (max, min, ortalama) = **792 istek**. Aynı istatistikteki değişkenler tek istekte gelir. CDS kuyruğuna göre her istek birkaç saniye ile birkaç dakika sürebilir;
  toplamda birkaç saat ile bir gün arası beklenebilir. Bilgisayar açık kalsın.
- Yarıda kesilirse aynı komutu tekrar çalıştır: inmiş dosyaları atlar, kaldığı yerden devam eder.
- Durumu https://cds.climate.copernicus.eu/requests sayfasından da izleyebilirsin.
- Toplam boyut küçüktür (alan ~2°×6°): birkaç yüz MB civarı.

Bittikten sonra: `python 01c_check_downloads.py` → "SORUN YOK" görmelisin.

Not: CDS web formu tek seferde bir ay seçtiriyor; betik de bu yüzden her ayı ayrı istiyor (`config.yaml` → `months_per_request: 1`).
Formda saat dilimini elle seçersen **UTC+03:00** olduğundan emin ol (UTC-03:00 değil).

## İndirilen veriler
**ERA5-Land (0.1°, Nisan–Eylül, Türkiye saati UTC+3'e göre günlük):**
Tmax, Tmin, Tmean (2 m sıcaklık), çiy noktası, toprak nemi (0–7 cm ve 7–28 cm), yüzey basıncı, rüzgâr (u, v).
Birikimli değişkenler (yağış, radyasyon) bu üründe sunulmadığı için NASA POWER'dan alınıyor.

**NASA POWER (0.5°, 1 Ocak 1981 – 31 Aralık 2024, tüm yıl):**
T2M, T2M_MAX, T2M_MIN, T2MDEW, RH2M, QV2M, WS2M, PS, ALLSKY_SFC_SW_DWN (1984'ten itibaren), PRECTOTCORR, GWETTOP, GWETROOT.

## Makalede atıf
- ERA5-Land post-processed daily statistics from 1950 to present, Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.e9c9c792
  (Yazar/yıl dâhil tam atıf metnini veri seti sayfasındaki "Citation" bölümünden kopyala.)
- NASA POWER: "The data was obtained from the National Aeronautics and Space Administration (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resource (POWER) Project funded through the NASA Earth Science/Applied Science Program."

## Sık karşılaşılan hatalar
| Hata | Çözüm |
|---|---|
| 401 / "authentication" | `.cdsapirc` içindeki key yanlış veya dosya yanlış yerde |
| 403 / "licence" | Veri seti sayfasında lisansı kabul et |
| 404 | `.cdsapirc` url'si `/api/v2` olarak kalmış; `/api` yap |
| Çok uzun bekleme | CDS kuyruğu yoğun; betik bekler, sorun değil |
