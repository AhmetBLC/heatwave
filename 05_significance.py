"""Aşama 5 - Model farklarının anlamlılığı: test yılları üzerinde eşleştirilmiş blok bootstrap.

Her karşılaştırma için PR-AUC farkı, %95 güven aralığı ve iki yönlü p değeri hesaplanır.
Aynı bootstrap örneği iki modele birlikte uygulanır (eşleştirilmiş), yıllar blok olarak örneklenir
(yıl içi otokorelasyonu korumak için).

Girdi : results/predictions_test.csv (04 çıktısı)
Çıktı : results/significance.csv

Kullanım:
    python 05_significance.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from src.common import ROOT, get_logger, load_config

PAIRS = [("logreg_full", "logreg_local"), ("logreg_full", "persistence"), ("logreg_local", "persistence"),
         ("lgbm_full", "lgbm_local"), ("xgb_full", "xgb_local"), ("hgb_full", "hgb_local"), ("rf_full", "rf_local")]


def main(n_boot: int = 1000, seed: int = 42) -> None:
    cfg = load_config()
    log = get_logger("05_significance", cfg)
    P = pd.read_csv(ROOT / "results" / "predictions_test.csv")
    rng = np.random.default_rng(seed)
    rows = []
    for t, d in P.groupby("target"):
        years = d.year.unique()
        g = {y: x for y, x in d.groupby("year")}
        for a, b in PAIRS:
            if a not in d or b not in d or d[a].isna().all() or d[b].isna().all():
                continue
            obs = ap(d.y_true, d[a]) - ap(d.y_true, d[b])
            diffs = np.empty(n_boot)
            for i in range(n_boot):
                s = pd.concat([g[y] for y in rng.choice(years, len(years))])
                diffs[i] = ap(s.y_true, s[a]) - ap(s.y_true, s[b])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            p = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / n_boot)
            rows.append({"target": t, "model_a": a, "model_b": b, "pr_auc_diff": obs, "ci_lo": lo, "ci_hi": hi, "p_value": p})
            log.info("%s %-13s - %-13s: %+.3f [%+.3f, %+.3f] p=%.3f", t, a, b, obs, lo, hi, p)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results" / "significance.csv", index=False)
    log.info("results/significance.csv yazıldı (p=0.001, %d tekrarda gözlenebilecek en küçük değerdir)", n_boot)


if __name__ == "__main__":
    main()
