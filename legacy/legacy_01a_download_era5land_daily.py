"""Aşama 1a - ERA5-Land günlük istatistiklerini Copernicus CDS'den indirir.

İstek birimi: (değişken, istatistik, yıl, ay). Her değişken AYRI istenir: CDS günlük istatistikleri
anlık hesapladığı için küçük istekler kuyrukta çok daha hızlı ilerler. Birkaç istek paralel çalışır.
Dosyalar:
    data/raw/era5land/<istatistik>/<yıl>_<ay>/<değişken>.nc
Var olan dosyalar atlanır; betik yarıda kalırsa tekrar çalıştırmak yeterli.

Kullanım:
    python 01a_download_era5land.py                       # hepsini indir (4 paralel istek)
    python 01a_download_era5land.py --workers 6           # paralel istek sayısı
    python 01a_download_era5land.py --years 1981          # belirli yıllar (deneme için)
    python 01a_download_era5land.py --dry-run             # sadece istekleri yazdır
"""
from __future__ import annotations

import argparse
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common import get_logger, load_config, resolve

_local = threading.local()


def month_groups(months: list[int], size: int) -> list[list[int]]:
    return [months[i:i + size] for i in range(0, len(months), size)]


def build_request(cfg: dict, variable: str, statistic: str, year: int, months: list[int]) -> dict:
    e = cfg["era5land"]
    m = [f"{x:02d}" for x in months]
    return {
        "year": str(year),
        "month": m[0] if len(m) == 1 else m,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": statistic,
        "time_zone": e["time_zone"],
        "frequency": e["frequency"],
        "variable": [variable],
        "area": e["area"],
    }


def save_single(tmp: Path, target: Path) -> None:
    """Zip ise içindeki tek .nc'yi, değilse dosyanın kendisini hedefe koyar."""
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            ncs = [n for n in z.namelist() if n.endswith(".nc")]
            if len(ncs) != 1:
                raise RuntimeError(f"Zip içinde beklenmeyen içerik: {z.namelist()}")
            with z.open(ncs[0]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp.unlink()
    else:
        tmp.replace(target)


def migrate_old_layout(folder: Path, variables: list[str]) -> None:
    """Önceki sürümün data.nc dosyasını (tek değişkenli isteklerde) yeni ada taşır."""
    old = folder / "data.nc"
    if old.exists() and len(variables) == 1 and not (folder / f"{variables[0]}.nc").exists():
        old.rename(folder / f"{variables[0]}.nc")


def client():
    if not hasattr(_local, "c"):
        import cdsapi
        _local.c = cdsapi.Client(quiet=True)
    return _local.c


def run_job(cfg, log, job, retries):
    var, stat, year, months, target = job
    req = build_request(cfg, var, stat, year, months)
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(f".part{threading.get_ident()}")
            client().retrieve(cfg["era5land"]["dataset"], req).download(str(tmp))
            save_single(tmp, target)
            log.info("  ✓ %s/%s/%s (%.0f sn)", stat, target.parent.name, var, time.time() - t0)
            return True
        except Exception as exc:
            log.warning("  Hata (%s/%s/%s, deneme %d): %s", stat, target.parent.name, var, attempt, exc)
            if attempt < retries:
                time.sleep(60 * attempt)
    log.error("  ✗ Başarısız, sonraki çalıştırmada tekrar denenecek: %s/%s/%s", stat, target.parent.name, var)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    e = cfg["era5land"]
    log = get_logger("01a_era5land", cfg)
    out_dir = resolve(cfg, "era5land_raw")
    years = args.years or list(range(cfg["period"]["start_year"], cfg["period"]["end_year"] + 1))
    groups = month_groups(cfg["period"]["months"], e.get("months_per_request", 1))

    jobs, skipped = [], 0
    for y in years:
        for g in groups:
            tag = f"{y}_" + "-".join(f"{m:02d}" for m in g)
            for r in e["requests"]:
                folder = out_dir / r["statistic"] / tag
                migrate_old_layout(folder, r["variables"])
                for v in r["variables"]:
                    target = folder / f"{v}.nc"
                    if target.exists() and target.stat().st_size > 0:
                        skipped += 1
                    else:
                        jobs.append((v, r["statistic"], y, g, target))
    log.info("Toplam %d istek | zaten inmiş: %d | paralel: %d", len(jobs) + skipped, skipped, args.workers)
    if args.dry_run:
        for j in jobs[:10]:
            log.info("  %s", build_request(cfg, j[0], j[1], j[2], j[3]))
        return

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, cfg, log, j, args.retries) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            ok += f.result()
            fail += not f.result()
            if i % 10 == 0 or i == len(futs):
                el = time.time() - t0
                log.info("İlerleme: %d/%d | geçen %.0f dk | tahmini kalan %.0f dk", i, len(futs), el / 60, el / i * (len(futs) - i) / 60)
    log.info("Bitti. Yeni: %d | Zaten vardı: %d | Başarısız: %d", ok, skipped, fail)


if __name__ == "__main__":
    main()
