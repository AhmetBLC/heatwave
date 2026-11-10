"""Aşama 9 - Duyarlılık analizi: sonuçlar tanım ve bölme seçimlerine ne kadar bağlı?

Varyantlar (her biri ayrı klasörde, ana sonuçlara dokunmadan):
  p90_L3 (ana tanım), p95_L3, p90_L2, p90_L4  -> sıcak hava dalgası eşiği ve en kısa süre
  split_alt                                   -> alternatif zaman bölmesi: eğitim 1981-2004, doğrulama 2005-2010, test 2011-2016
Her varyantta 02c (etiketler), 03 (öznitelikler), 04 (yalnızca lojistik regresyon + baseline'lar, y3 ve y7)
ve 05 (anlamlılık) yeniden çalıştırılır.

Girdi : data/interim/era5_daily.csv, data/interim/pl_daily.csv
Çıktı : sensitivity/<varyant>/...  ve  results/analysis/sensitivity.csv

Kullanım:
    python 09_sensitivity.py                 # tüm varyantlar
    python 09_sensitivity.py --only p95_L3
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import pandas as pd
import yaml

from src.common import ROOT, get_logger, load_config

VARIANTS = {
    "p90_L3": {"percentile": 90, "min_length": 3},
    "p95_L3": {"percentile": 95, "min_length": 3},
    "p90_L2": {"percentile": 90, "min_length": 2},
    "p90_L4": {"percentile": 90, "min_length": 4},
    "split_alt": {"percentile": 90, "min_length": 3, "split": {"train": [1981, 2004], "val": [2005, 2010], "test": [2011, 2016]}},
}
SCRIPTS = ["02c_label_heatwaves.py", "03_build_features.py", "04_train_models.py", "05_significance.py"]


def run(cmd, cwd, log):
    r = subprocess.run([sys.executable, *cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("%s başarısız:\n%s", cmd[0], r.stderr[-2000:])
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg0 = load_config()
    log = get_logger("09_sensitivity", cfg0)
    rows = []
    for name, v in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        d = ROOT / "sensitivity" / name
        (d / "data" / "interim").mkdir(parents=True, exist_ok=True)
        for f in ("era5_daily.csv", "pl_daily.csv"):
            if not (d / "data" / "interim" / f).exists():
                shutil.copy(ROOT / "data" / "interim" / f, d / "data" / "interim" / f)
        shutil.copytree(ROOT / "src", d / "src", dirs_exist_ok=True)
        for s in SCRIPTS:
            shutil.copy(ROOT / s, d / s)
        cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
        cfg["heatwave"]["percentile"] = v["percentile"]
        cfg["heatwave"]["min_length"] = v["min_length"]
        cfg["heatwave"]["horizons"] = [3, 7]
        if "split" in v:
            cfg["split"] = v["split"]
        yaml.safe_dump(cfg, open(d / "config.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
        log.info("== %s: etiketler, öznitelikler, modeller, testler", name)
        run(["02c_label_heatwaves.py"], d, log)
        run(["03_build_features.py"], d, log)
        run(["04_train_models.py", "--models", "logreg", "--no-importance"], d, log)
        run(["05_significance.py"], d, log)
        m = pd.read_csv(d / "results" / "metrics_test.csv")
        s = pd.read_csv(d / "results" / "significance.csv")
        lab = pd.read_csv(d / "data" / "processed" / "labels.csv", parse_dates=["date"])
        ev = pd.read_csv(d / "data" / "processed" / "events.csv")
        for t in ("y3", "y7"):
            g = lambda mdl, fs: float(m[(m.target == t) & (m.model == mdl) & (m.feature_set == fs)].pr_auc.iloc[0])
            sg = lambda a, b: s[(s.target == t) & (s.model_a == a) & (s.model_b == b)].iloc[0]
            fl, fp, lp = sg("logreg_full", "logreg_local"), sg("logreg_full", "persistence"), sg("logreg_local", "persistence")
            rows.append({"variant": name, "target": t, "n_events": len(ev), "pos_rate": lab[t].mean(),
                         "persistence": g("persistence", "-"), "lr_local": g("logreg", "local"), "lr_full": g("logreg", "full"),
                         "gain_full_local": fl.pr_auc_diff, "p_full_local": fl.p_value,
                         "gain_full_pers": fp.pr_auc_diff, "p_full_pers": fp.p_value,
                         "gain_local_pers": lp.pr_auc_diff, "p_local_pers": lp.p_value})
            log.info("  %s %s: pers %.3f | yerel %.3f | tam %.3f | tam-yerel %+.3f (p=%.3f) | yerel-pers %+.3f (p=%.3f)",
                     name, t, rows[-1]["persistence"], rows[-1]["lr_local"], rows[-1]["lr_full"],
                     fl.pr_auc_diff, fl.p_value, lp.pr_auc_diff, lp.p_value)
    out = ROOT / "results" / "analysis" / "sensitivity.csv"
    old = pd.read_csv(out) if out.exists() and args.only else None
    R = pd.DataFrame(rows)
    if old is not None:
        R = pd.concat([old[~old.variant.isin(R.variant)], R])
    R.round(4).to_csv(out, index=False)
    log.info("Kaydedildi: %s", out)


if __name__ == "__main__":
    main()
