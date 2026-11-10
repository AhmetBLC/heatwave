"""Aşama 1a - ERA5-Land saatlik nokta zaman serilerini Copernicus CDS'den indirir.

Veri seti: reanalysis-era5-land-timeseries (nokta için uzun saatlik seri, tek istekte).
İstekler SIRAYLA gönderilir (CDS kuyruk sınırı nedeniyle paralel istek yok).
Dosyalar:
    data/raw/era5land_ts/<nokta_id>/<parça>.csv   (CDS zip döndürürse içi açılır)
İnmiş noktalar atlanır; betik yarıda kalırsa tekrar çalıştırmak yeterli.

Kullanım:
    python 01a_download_era5land.py --points gaziantep     # önce tek nokta ile dene
    python 01a_download_era5land.py                        # tüm noktalar
    python 01a_download_era5land.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import time
import zipfile
from pathlib import Path

from src.common import ROOT, get_logger, load_config

QUEUE_FULL = "Maximum number queued requests"
SHORT = {
    "2m_temperature": "t2m", "2m_dewpoint_temperature": "d2m", "skin_temperature": "skt",
    "surface_pressure": "sp", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "surface_solar_radiation_downwards": "ssrd", "total_precipitation": "tp",
    "volumetric_soil_water_layer_1": "swvl1", "volumetric_soil_water_layer_2": "swvl2",
}


def columns_found(folder: Path, names: list[str]) -> set[str]:
    cols = set()
    for n in names:
        f = folder / n
        if f.suffix == ".csv":
            with open(f, encoding="utf-8") as fh:
                cols |= set(fh.readline().strip().split(","))
    return cols


def snap(x: float, step: float) -> float:
    return round(round(x / step) * step, 4)


def chunks(seq: list, n: int) -> list[list]:
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def build_request(cfg: dict, lat: float, lon: float, variables: list[str]) -> dict:
    p = cfg["period"]
    return {
        "variable": variables,
        "location": {"latitude": lat, "longitude": lon},
        "date": [f"{p['start_year']}-01-01/{p['end_year']}-12-31"],
        "data_format": cfg["era5land"].get("data_format", "csv"),
    }


def unpack(tmp: Path, folder: Path, stem: str) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                dst = folder / f"{stem}__{Path(n).name}"
                with z.open(n) as s, open(dst, "wb") as d:
                    shutil.copyfileobj(s, d)
                out.append(dst.name)
        tmp.unlink()
    else:
        head = tmp.read_bytes()[:4]
        ext = ".nc" if head.startswith(b"CDF") or head.startswith(b"\x89HDF") else ".csv"
        dst = folder / f"{stem}{ext}"
        tmp.replace(dst)
        out.append(dst.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="*", help="sadece bu nokta id'leri (ör. gaziantep)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-wait-min", type=int, default=120, help="kuyruk doluysa toplam bekleme süresi")
    args = ap.parse_args()

    cfg = load_config()
    e = cfg["era5land"]
    log = get_logger("01a_era5land", cfg)
    base = ROOT / "data" / "raw" / "era5land_ts"
    step = e.get("snap_to_grid", 0.1)
    groups = e.get("variable_groups") or chunks(e["variables"], e.get("variables_per_request", len(e["variables"])))
    points = [p for p in cfg["points"] if not args.points or p["id"] in args.points]

    client = None
    if not args.dry_run:
        import cdsapi
        client = cdsapi.Client()

    for p in points:
        lat, lon = snap(p["lat"], step), snap(p["lon"], step)
        folder = base / p["id"]
        for gi, vars_ in enumerate(groups):
            stem = f"part{gi + 1:02d}"
            done = folder / f"_DONE_{stem}"
            if done.exists():
                log.info("%s %s zaten inmiş, atlanıyor", p["name"], stem)
                continue
            req = build_request(cfg, lat, lon, vars_)
            if args.dry_run:
                log.info("%s (%.1f, %.1f) %s -> %s", p["name"], lat, lon, stem, req)
                continue
            waited = 0
            while True:
                try:
                    t0 = time.time()
                    log.info("İndiriliyor: %s (%.1f, %.1f) %s, %d değişken", p["name"], lat, lon, stem, len(vars_))
                    folder.mkdir(parents=True, exist_ok=True)
                    tmp = folder / f"{stem}.part"
                    client.retrieve(e["dataset"], req).download(str(tmp))
                    names = unpack(tmp, folder, stem)
                    missing = [v for v in vars_ if SHORT.get(v) and SHORT[v] not in columns_found(folder, names)]
                    if missing:
                        log.error("  ✗ %s %s: şu değişkenler gelmedi: %s (dosyalar: %s)", p["name"], stem, missing, names)
                        break
                    done.write_text("\n".join(names), encoding="utf-8")
                    log.info("  ✓ %s %s: %s (%.0f sn)", p["name"], stem, ", ".join(names), time.time() - t0)
                    break
                except Exception as exc:
                    msg = str(exc)
                    if QUEUE_FULL in msg and waited < args.max_wait_min:
                        log.warning("  Kuyruk dolu; 5 dk bekleyip tekrar denenecek (toplam bekleme %d dk)", waited)
                        time.sleep(300); waited += 5
                        continue
                    log.error("  ✗ %s %s başarısız: %s", p["name"], stem, msg[:500])
                    break
    log.info("Bitti.")


if __name__ == "__main__":
    main()
