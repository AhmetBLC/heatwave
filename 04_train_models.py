"""Aşama 4 - Baseline ve makine öğrenmesi modelleri.

Bölme (config.yaml -> split): eğitim 1981-2010, doğrulama 2011-2016, test 2017-2024.
  - Hiperparametre ve karar eşiği DOĞRULAMA setinde seçilir; test seti yalnızca en sonda bir kez kullanılır.
  - Seçimden sonra en iyi model eğitim+doğrulama (1981-2016) verisiyle yeniden eğitilir (--no-refit ile kapatılır).
    Klimatoloji baseline'ı da aynı şekilde 1981-2016'dan hesaplanır.
Modeller:
  climatology   : eğitim dönemindeki takvim günü pozitif oranı (±7 gün)
  persistence   : bugünkü eşik aşımı (excess = tmax - eşik) -> skor; "sıcaksa sürer" varsayımı
  logreg        : lojistik regresyon (standartlaştırılmış, ağırlıksız; bkz. 06 kalibrasyon)
  rf            : Random Forest (balanced_subsample)
  hgb           : HistGradientBoosting (LightGBM ile aynı histogram tabanlı GBM ailesi)
  xgb / lgbm    : kuruluysa XGBoost / LightGBM
Öznitelik setleri (--feature-sets): local = yalnız yerel yüzey değişkenleri; full = yerel + büyük ölçekli
  (ERA5 basınç seviyeleri, "pl_" önekli). İki setin karşılaştırması makalenin ana bulgusunu test eder.
Metrikler: ROC-AUC, PR-AUC, Brier, BSS (klimatolojiye göre), F1, POD (recall), FAR (1-precision), CSI.
Belirsizlik: test yıllarının blok bootstrap'ı ile %95 güven aralıkları.

Çıktı: results/metrics_test.csv, results/metrics_val.csv, results/predictions_test.csv,
       results/importance_<model>_y<k>.csv

Kullanım:
    python 04_train_models.py
    python 04_train_models.py --targets y3 y7 --models logreg rf hgb
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, get_logger, load_config

META = ["date", "point_id", "year", "in_event"]
FEATURE_SETS = {"local": lambda c: not c.startswith("pl_"), "full": lambda c: True}
SEED = 42


def split(F, cfg):
    s = cfg["split"]
    yr = F.year
    return (F[(yr >= s["train"][0]) & (yr <= s["train"][1])],
            F[(yr >= s["val"][0]) & (yr <= s["val"][1])],
            F[(yr >= s["test"][0]) & (yr <= s["test"][1])])


def climatology_scores(train, other, y):
    doy = train.date.dt.dayofyear
    rate = train.groupby(doy)[y].mean()
    full = pd.Series(np.nan, index=range(1, 367))
    full.loc[rate.index] = rate.values
    sm = full.rolling(15, center=True, min_periods=1).mean()
    return other.date.dt.dayofyear.map(sm).fillna(train[y].mean()).to_numpy()


def candidates(pos_weight):
    c = {
        # Ağırlıksız: 06 kalibrasyon analizinde hem daha iyi kalibre hem de PR-AUC'si daha yüksek çıktı.
        "logreg": [make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
                   for C in (0.01, 0.1, 1.0)],
        "rf": [RandomForestClassifier(n_estimators=400, min_samples_leaf=l, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
               for l in (10, 30)],
        "hgb": [HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=ln, min_samples_leaf=50,
                                               l2_regularization=1.0, max_iter=600, early_stopping=True,
                                               validation_fraction=0.15, class_weight="balanced", random_state=SEED)
                for lr in (0.03, 0.08) for ln in (15, 31)],
    }
    try:
        from xgboost import XGBClassifier
        c["xgb"] = [XGBClassifier(n_estimators=600, learning_rate=lr, max_depth=d, subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=SEED, n_jobs=-1)
                    for lr in (0.03, 0.08) for d in (4, 6)]
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        c["lgbm"] = [LGBMClassifier(n_estimators=600, learning_rate=lr, num_leaves=nl, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, is_unbalance=True, random_state=SEED, verbose=-1)
                     for lr in (0.03, 0.08) for nl in (15, 31)]
    except ImportError:
        pass
    return c


def best_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 300)))
    f1s = []
    for q in qs:
        pr = p >= q
        tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(qs[int(np.argmax(f1s))])


def scores(y, p, thr, base_rate):
    pr = p >= thr
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum())
    pp = np.clip(p, 0, 1)
    bs = brier_score_loss(y, pp)
    bs_ref = brier_score_loss(y, np.full(len(y), base_rate))
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": bs, "bss": 1 - bs / bs_ref,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "pod": tp / max(tp + fn, 1),
            "far": fp / max(tp + fp, 1), "csi": tp / max(tp + fp + fn, 1), "threshold": thr}


def block_bootstrap(df, y, col, n=500, seed=SEED):
    rng = np.random.default_rng(seed)
    years = df.year.unique()
    groups = {yr: g for yr, g in df.groupby("year")}
    vals = []
    for _ in range(n):
        s = pd.concat([groups[yr] for yr in rng.choice(years, len(years), replace=True)])
        if s[y].nunique() == 2:
            vals.append(average_precision_score(s[y], s[col]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--feature-sets", nargs="*", default=["local", "full"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--no-importance", action="store_true")
    ap.add_argument("--no-refit", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("04_models", cfg)
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    F = pd.read_csv(ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])
    ycols = [c for c in F.columns if c.startswith("y") and c[1:].isdigit()]
    targets = args.targets or [f"y{k}" for k in cfg["heatwave"]["horizons"]]
    feats_all = [c for c in F.columns if c not in META and c not in ycols]
    has_pl = any(c.startswith("pl_") for c in feats_all)
    fsets = [fs for fs in args.feature_sets if fs == "local" or has_pl]
    FS = {fs: [c for c in feats_all if FEATURE_SETS[fs](c)] for fs in fsets}
    log.info("Öznitelik setleri: %s", {k: len(v) for k, v in FS.items()})

    rows_val, rows_test, preds = [], [], []
    for y in targets:
        D = F[F[y].notna()].copy()
        D[y] = D[y].astype(int)
        tr, va, te = split(D, cfg)
        base = tr[y].mean()
        log.info("=== %s | eğitim %d (pozitif %.1f%%) | doğrulama %d | test %d (pozitif %.1f%%)",
                 y, len(tr), 100 * base, len(va), len(te), 100 * te[y].mean())
        P = te[["date", "point_id", "year", y]].copy()

        def record(name, pv, pt, params, fs="-"):
            thr = best_threshold(va[y].to_numpy(), pv)
            mv = scores(va[y].to_numpy(), pv, thr, base); mt = scores(te[y].to_numpy(), pt, thr, base)
            col = name if fs == "-" else f"{name}_{fs}"
            P[col] = pt
            lo, hi = block_bootstrap(P, y, col)
            rows_val.append({"target": y, "model": name, "feature_set": fs, **mv, "params": params})
            rows_test.append({"target": y, "model": name, "feature_set": fs, **mt, "pr_auc_ci_lo": lo, "pr_auc_ci_hi": hi, "params": params})
            log.info("  %-17s test ROC-AUC %.3f | PR-AUC %.3f [%.3f-%.3f] | F1 %.3f | POD %.2f | FAR %.2f | BSS %.3f",
                     col, mt["roc_auc"], mt["pr_auc"], lo, hi, mt["f1"], mt["pod"], mt["far"], mt["bss"])

        trva = tr if args.no_refit else pd.concat([tr, va])
        record("climatology", climatology_scores(tr, va, y), climatology_scores(trva, te, y), "")
        record("persistence", va["excess"].to_numpy(), te["excess"].to_numpy(), "score=tmax-esik")

        pos_w = (1 - base) / base
        for fs, feats in FS.items():
          for name, cands in candidates(pos_w).items():
            if args.models and name not in args.models:
                continue
            t0 = time.time()
            best, best_ap = None, -1
            for m in cands:
                m.fit(tr[feats], tr[y])
                ap_ = average_precision_score(va[y], m.predict_proba(va[feats])[:, 1])
                if ap_ > best_ap:
                    best, best_ap = m, ap_
            pv = best.predict_proba(va[feats])[:, 1]
            if not args.no_refit:
                from sklearn.base import clone
                best = clone(best).fit(trva[feats], trva[y])
            pt = best.predict_proba(te[feats])[:, 1]
            prm = best.get_params()
            prm = {k: v for k, v in (prm.items() if not hasattr(best, "steps") else best.steps[-1][1].get_params().items())
                   if k in ("C", "min_samples_leaf", "learning_rate", "max_leaf_nodes", "max_depth", "num_leaves")}
            record(name, pv, pt, json.dumps(prm), fs)
            log.info("     (%s/%s: %d aday, %.0f sn)", name, fs, len(cands), time.time() - t0)
            if not args.no_importance and fs == list(FS)[-1] and name in ("hgb", "xgb", "lgbm"):
                pi = permutation_importance(best, te[feats], te[y], scoring="average_precision",
                                            n_repeats=5, random_state=SEED, n_jobs=1)
                imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean, "std": pi.importances_std})
                imp.sort_values("importance", ascending=False).to_csv(res / f"importance_{name}_{fs}_{y}.csv", index=False)
        preds.append(P.rename(columns={y: "y_true"}).assign(target=y))

    MV, MT, PR = pd.DataFrame(rows_val), pd.DataFrame(rows_test), pd.concat(preds)
    if args.models and (res / "metrics_test.csv").exists():
        # Yalnız seçilen modeller yeniden eğitildi: diğer modellerin önceki sonuçlarını koru
        keep = lambda df: df[~df.model.isin(list(args.models) + ["climatology", "persistence"]) | ~df.target.isin(targets)]
        MV = pd.concat([keep(pd.read_csv(res / "metrics_val.csv")), MV], ignore_index=True)
        MT = pd.concat([keep(pd.read_csv(res / "metrics_test.csv")), MT], ignore_index=True)
        old = pd.read_csv(res / "predictions_test.csv")
        newcols = [c for c in PR.columns if c not in ("date", "point_id", "year", "y_true", "target")]
        old = old.drop(columns=[c for c in newcols if c in old.columns])
        PR["date"] = PR["date"].astype(str).str[:10]; old["date"] = old["date"].astype(str).str[:10]
        PR = old.merge(PR.drop(columns=["year", "y_true"]), on=["date", "point_id", "target"], how="left")
        log.info("Önceki sonuçlarla birleştirildi (yeniden eğitilen: %s)", args.models)
    MV.round(4).to_csv(res / "metrics_val.csv", index=False)
    MT.round(4).to_csv(res / "metrics_test.csv", index=False)
    PR.to_csv(res / "predictions_test.csv", index=False)
    log.info("Sonuçlar results/ klasörüne yazıldı.")


if __name__ == "__main__":
    main()
