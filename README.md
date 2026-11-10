# Güneydoğu Anadolu'da sıcak hava dalgalarının makine öğrenmesiyle tahmini

Bu depo, aşağıdaki makalenin tüm analiz kodunu ve sonuç dosyalarını içerir:

> Balcı, A. ve Aslan, Z. *Güneydoğu Anadolu'da sıcak hava dalgalarının 1–7 gün önceden tahmini: büyük ölçekli atmosfer öncüllerinin katkısı ve fiziksel tutarlılığı.* (değerlendirme aşamasında)

Çalışma, Güneydoğu Anadolu'nun 9 il merkezi için 1981–2024 dönemindeki yaz sıcak hava dalgalarını 1, 3, 5 ve 7 gün önceden tahmin eder. Yalnızca açık erişimli verilerle (ERA5-Land, ERA5, NASA POWER) çalışır. Yerel yüzey öncülleri ile büyük ölçekli atmosfer öncüllerinin (Z500, T850) katkısını karşılaştırır, SHAP ile yorumlar ve sonucu modelden bağımsız bir sinoptik kompozit analiziyle sınar.

*English summary: Code and results for a machine-learning study of 1–7-day heatwave prediction in Southeastern Anatolia (Türkiye), 1981–2024, using ERA5-Land, ERA5 pressure-level and NASA POWER data. The pipeline covers data download, leakage-free labelling, five ML models with persistence and climatology baselines, paired block-bootstrap significance tests, calibration, SHAP, detrended composite analysis and sensitivity tests. Run the numbered scripts in order; all settings are in `config.yaml`.*

## Klasör yapısı

```
heatwave-gda/
├── config.yaml              # Tüm ayarlar: iller, dönem, değişkenler, olay tanımı, bölme, bölge kutuları
├── requirements.txt
├── src/common.py            # Ortak yardımcılar (ayar okuma, log)
├── 01a ... 01e              # Veri indirme
├── 02a ... 02d              # Ham veriden günlük veri, etiketler ve büyük ölçekli öznitelikler
├── 03_build_features.py     # Öznitelik üretimi
├── 04_train_models.py       # Baseline ve ML modelleri
├── 05_significance.py       # Eşleştirilmiş blok bootstrap anlamlılık testleri
├── 06_analysis.py           # Kalibrasyon, SHAP, grup önemi, il bazında sonuçlar
├── 07_paper_figures.py      # Eğilim ve tahmin ufku şekilleri
├── 08_composites.py         # Eğilimden arındırılmış Z500/T850 kompozitleri
├── 09_sensitivity.py        # Olay tanımı ve test dönemi duyarlılık analizi
├── 10_maps.py               # Çalışma alanı ve kompozit haritaları
├── results/                 # Metrikler, tahminler, anlamlılık ve analiz tabloları (CSV)
├── figures/                 # Makaledeki şekiller (PNG ve PDF)
└── legacy/                  # Geliştirme sürecinde kullanılan eski betikler (güncel akışta gerekmez)
```

Ham ve işlenmiş veriler (`data/`, yaklaşık 1,8 GB) boyutları nedeniyle depoda yoktur. Aşağıdaki adımlarla yeniden indirilip üretilebilir.

## Kurulum

Python 3.10 veya üzeri gerekir.

```bash
pip install -r requirements.txt
```

### Copernicus CDS hesabı

ERA5 ve ERA5-Land verileri için ücretsiz bir [Copernicus CDS](https://cds.climate.copernicus.eu) hesabı gerekir.

1. Hesabınızla giriş yapın, profil sayfasından **API Token**'ı kopyalayın.
2. Ana klasörünüzde (`C:\Users\<kullanıcı>\` veya `~/`) `.cdsapirc` dosyası oluşturun:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <API-TOKEN>
   ```
3. Kullanılan üç veri setinin sayfasında lisansı kabul edin: `reanalysis-era5-land-timeseries`, `reanalysis-era5-land`, `reanalysis-era5-pressure-levels`.

`.cdsapirc` dosyasını depoya eklemeyin; `.gitignore` bunu engeller.

## Çalıştırma sırası

| Adım | Komut | Ne yapar | Çıktı |
|---|---|---|---|
| 1a | `python 01a_download_era5land.py` | ERA5-Land saatlik nokta serileri (9 il) | `data/raw/era5land_ts/` |
| 1b | `python 01b_download_nasapower.py` | NASA POWER günlük nokta verisi (doğrulama) | `data/raw/nasapower/` |
| 1d | `python 01d_download_soil_moisture.py` | ERA5-Land toprak nemi, 12:00 TR | `data/raw/era5land_soil/` |
| 1e | `python 01e_download_pressure_levels.py` | ERA5 500/850 hPa jeopotansiyel ve sıcaklık | `data/raw/era5_pl/` |
| 2a | `python 02a_extract_soil.py` | İllerin toprak nemi serileri | `data/interim/soil_points.csv` |
| 2b | `python 02b_build_daily.py` | Türkiye saatine göre günlük değişkenler | `data/interim/era5_daily.csv` |
| 2c | `python 02c_label_heatwaves.py` | Eşikler, olaylar ve tahmin hedefleri | `data/processed/` |
| 2d | `python 02d_extract_pressure.py` | İl ve bölge ölçeğinde Z500/T850 | `data/interim/pl_daily.csv` |
| 3 | `python 03_build_features.py` | 177 öznitelik (92 yerel + 85 büyük ölçekli) | `data/processed/features.csv` |
| 4 | `python 04_train_models.py` | Persistence, klimatoloji, LR, RF, HGB, XGBoost, LightGBM | `results/metrics_*.csv`, `results/predictions_test.csv` |
| 5 | `python 05_significance.py` | Model farklarının anlamlılığı | `results/significance.csv` |
| 6 | `python 06_analysis.py` | Kalibrasyon, SHAP, grup önemi, il bazında sonuçlar | `results/analysis/`, `figures/` |
| 7 | `python 07_paper_figures.py` | Eğilim ve ufuk şekilleri | `figures/` |
| 8 | `python 08_composites.py` | Kompozit analizi ve harita altlığı | `results/analysis/composites.csv` |
| 9 | `python 09_sensitivity.py` | Duyarlılık analizi (p95, ≥2 gün, ≥4 gün, alternatif bölünme) | `results/analysis/sensitivity.csv` |
| 10 | `python 10_maps.py` | Çalışma alanı ve kompozit haritaları | `figures/` |

İndirme adımları CDS kuyruğuna bağlı olarak birkaç saat sürebilir. Yarıda kesilirse aynı komut tekrar çalıştırıldığında inmiş dosyaları atlar. Her betik ayrıntılı kaydını `logs/` altına yazar.

`results/` ve `figures/` klasörleri depoda hazır bulunduğundan, veriyi indirmeden makaledeki tablolar ve şekiller incelenebilir. `06`, `07` ve `10` numaralı betiklerin çizim bölümleri yalnızca bu klasördeki CSV dosyalarına ihtiyaç duyar.

## Temel yöntem seçimleri

- **Sıcak hava dalgası:** Günlük Tmax'ın, 1981–2010 döneminde o takvim gününü ortalayan 15 günlük penceredeki 90. yüzdeliği en az 3 gün üst üste aşması.
- **Tahmin hedefi:** `y_k(t) = 1`, eğer bir olayın teyit günü (üçüncü günü) `(t, t+k]` aralığına düşüyorsa. Tahmin anında olayın teyit edilmediği günler örneklem olarak kullanılır; böylece geleceğe ait bilgi modele sızmaz.
- **Zaman bölmesi:** Eğitim 1981–2010, doğrulama 2011–2016, test 2017–2024. Hiperparametreler ve karar eşiği doğrulama setinde seçilir, model eğitim+doğrulama verisiyle yeniden eğitilir.
- **Değerlendirme:** Birincil ölçüt PR-AUC. Farklar, test yılları üzerinde eşleştirilmiş blok bootstrap (1.000 tekrar) ile sınanır.

Ayrıntılar makalenin Yöntem bölümündedir. Olay tanımı, bölme veya bölge kutuları `config.yaml` üzerinden değiştirilebilir.

## Veri kaynakları ve lisanslar

- **ERA5-Land ve ERA5** (Hersbach vd., 2020; Muñoz-Sabater vd., 2021), Copernicus İklim Değişikliği Hizmeti (C3S) Climate Data Store. Copernicus lisansı kapsamında kullanılmıştır. *Contains modified Copernicus Climate Change Service information (1981–2024). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.*
- **NASA POWER** (https://power.larc.nasa.gov). *Data were obtained from the NASA Langley Research Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program.*

## Lisans

Kod MIT lisansı ile paylaşılmaktadır (bkz. `LICENSE`). Veriler kendi kaynaklarının lisanslarına tabidir.

## Atıf

Bu kodu kullanırsanız lütfen makaleye atıf yapın. Atıf bilgileri `CITATION.cff` dosyasındadır ve makale yayımlandığında güncellenecektir.
