"""Aşama 11 - Hakem değerlendirmesine yanıt analizleri.

Her bölüm bağımsız çalışır (--only ile seçilebilir) ve sonuçlarını results/revision/ altına yazar.

  boot      : Ortak bootstrap altyapısı. Test yılları (9 il birlikte) blok olarak, 10.000 tekrarla,
              SABİT tohumla örneklenir; tüm karşılaştırmalar AYNI bootstrap örneklerini kullanır.
              p = 2·min(P(Δ ≤ 0), P(Δ ≥ 0)), en küçük değer 1/B. Holm düzeltmesi raporlanır.
  select    : (Majör 1) Ağırlıklı / ağırlıksız / ağırlıklı+Platt LR seçimi DOĞRULAMA setinde.
  detrend   : (Majör 2a) Büyük ölçekli öznitelikler eğilimden arındırılarak LR yeniden eğitilir:
              (i) eğilim yalnız eğitim döneminden (1981-2010) kestirilip ileri taşınır,
              (ii) geçmişe dönük: her yıl için önceki 5 sezonun ortalama anomalisi çıkarılır.
  trendvars : (Majör 2b) Yerel ve tam sete yıl ve geçmiş 30/60/90 günlük bölgesel Tmax anomalisi eklenir.
  early     : (Majör 3) t gününde aşım olan pozitiflerin oranı; t'de aşım olmayan alt kümede PR-AUC.
  sig       : (Majör 4-5) Ana karşılaştırmalar 10.000 ortak bootstrap ile; Holm düzeltmesi.
  sens      : (Majör 5) Duyarlılık analizi p değerleri aynı bootstrap altyapısıyla (varsa sensitivity/ klasörü).
  baselines : (Majör 9) Yakın dönem (2007-2016) ve eğilimli klimatoloji, 3 değişkenli basit LR;
              BSS'nin farklı referanslara göre değeri.
  trees     : (Majör 10) Ağırlıksız ağaç modelleri, eğitim aralığı dışındaki test örnekleri,
              alternatif bölünmede (test 2011-2016) LR ile ağaçların farkı.
  threshold : (Majör 11) Yeniden eğitilmeyen (yalnız 1981-2010) modelle eşik aktarımı.
  dryness   : (Majör 8) Kuruluk grubunun ayrıştırılması (VPD/RH, nem içeriği, toprak nemi/yağış),
              özgül nem öznitelikleri, bootstrap güven aralıklı ve göreli permütasyon önemi.
  boxes     : (Majör 7) Güneydoğu kutusuyla çakışmayan Orta Anadolu kutusu (anat_n) ile LR ve SHAP.
              data/interim/pl_daily.csv içinde z500_anat_n sütunu gerekir (02d yeniden çalıştırılmalı).

Kullanım:
    python 11_revision.py                 # tüm bölümler
    python 11_revision.py --only early sig
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, get_logger, load_config

B = 10_000
SEED = 2026
TARGETS = ["y3", "y5", "y7"]
META = ["date", "point_id", "year", "in_event"]
OUT = ROOT / "results" / "revision"


# ----------------------------------------------------------------------------- ortak araçlar
def load_features():
    F = pd.read_csv(ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])
    ycols = [c for c in F.columns if c.startswith("y") and c[1:].isdigit()]
    feats = [c for c in F.columns if c not in META and c not in ycols]
    return F, feats


def split(D, s):
    yr = D.year
    return D[yr.between(*s["train"])], D[yr.between(*s["val"])], D[yr.between(*s["test"])]


def lr(C, weighted=False):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000,
                                                              class_weight="balanced" if weighted else None))


def fit_select(make, grid, tr, va, feats, y, refit_on=None):
    """Izgaradan doğrulama PR-AUC'sine göre seçer; refit_on verilirse onunla yeniden eğitir."""
    best, best_ap, best_g = None, -1, None
    for g in grid:
        m = make(g).fit(tr[feats], tr[y])
        a = average_precision_score(va[y], m.predict_proba(va[feats])[:, 1])
        if a > best_ap:
            best, best_ap, best_g = m, a, g
    pv = best.predict_proba(va[feats])[:, 1]
    if refit_on is not None:
        best = clone(best).fit(refit_on[feats], refit_on[y])
    return best, pv, best_g, best_ap


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


class Boot:
    """Yıl bloğu bootstrap'ı: B x Y sayım matrisi; tüm modeller ve karşılaştırmalar aynı örnekleri kullanır.

    Ağırlıklı PR-AUC (sklearn average_precision_score ile aynı tanım) vektörel hesaplanır.
    """

    def __init__(self, years, n=B, seed=SEED):
        self.years = np.array(sorted(np.unique(years)))
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.years), size=(n, len(self.years)))
        self.C = np.zeros((n, len(self.years)), dtype=np.float32)
        for j in range(len(self.years)):
            self.C[:, j] = (draws == j).sum(1)

    def ap(self, y, s, yrs, chunk=250):
        yi = np.searchsorted(self.years, yrs)
        o = np.argsort(-s, kind="mergesort")
        ys, ss, yo = y[o].astype(np.float32), s[o], yi[o]
        ends = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
        out = np.empty(len(self.C))
        for a in range(0, len(self.C), chunk):
            W = self.C[a:a + chunk][:, yo]
            tp = np.cumsum(W * ys, axis=1)[:, ends]
            tot = np.cumsum(W, axis=1)[:, ends]
            prec = np.where(tot > 0, tp / np.maximum(tot, 1e-9), 0)
            rec = tp / np.maximum(tp[:, -1:], 1e-9)
            out[a:a + chunk] = (np.diff(rec, axis=1, prepend=0) * prec).sum(1)
        return out

    def compare(self, y, sa, sb, yrs):
        da = self.ap(y, sa, yrs); db = self.ap(y, sb, yrs)
        d = da - db
        obs = average_precision_score(y, sa) - average_precision_score(y, sb)
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"diff": obs, "ci_lo": lo, "ci_hi": hi, "p": p}

    def ci(self, y, s, yrs):
        return np.percentile(self.ap(y, s, yrs), [2.5, 97.5])


def save(df, name, log):
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(OUT / f"{name}.csv", index=False)
    log.info("-> results/revision/%s.csv", name)


def lr_full_local(F, feats, s, y, feats_full=None, feats_local=None):
    """Ana kurguyla aynı: C doğrulamada seçilir, eğitim+doğrulama ile yeniden eğitilir."""
    D = F[F[y].notna()].copy(); D[y] = D[y].astype(int)
    tr, va, te = split(D, s); trva = pd.concat([tr, va])
    fl = feats_local or [c for c in feats if not c.startswith("pl_")]
    ff = feats_full or feats
    out = {}
    for name, fs in (("local", fl), ("full", ff)):
        m, pv, g, _ = fit_select(lambda C: lr(C), (0.01, 0.1, 1.0), tr, va, fs, y, refit_on=trva)
        out[name] = m.predict_proba(te[fs])[:, 1]
        out[f"C_{name}"] = g
    return te, out


# ----------------------------------------------------------------------------- bölümler
def sec_select(cfg, F, feats, log):
    """Majör 1: LR varyantı doğrulama setinde seçilir. Platt, doğrulama yılları içinde çapraz uydurulur."""
    rows = []
    s = cfg["split"]
    for y in TARGETS:
        D = F[F[y].notna()].copy(); D[y] = D[y].astype(int)
        tr, va, te = split(D, s)
        yv = va[y].to_numpy()
        res = {}
        for wname, w in (("agirliksiz", False), ("agirlikli", True)):
            m, pv, C, apv = fit_select(lambda C, w=w: lr(C, w), (0.01, 0.1, 1.0), tr, va, feats, y)
            res[wname] = (pv, C)
        # Platt: doğrulama yıllarını iki yarıya böl, birinde uydur diğerinde değerlendir (çapraz)
        pv_w, C_w = res["agirlikli"]
        vy = va.year.to_numpy(); half = np.isin(vy, sorted(np.unique(vy))[:3])
        pp = np.empty_like(pv_w)
        for a, b in ((half, ~half), (~half, half)):
            cal = LogisticRegression(C=1e6).fit(logit(pv_w[a]).reshape(-1, 1), yv[a])
            pp[b] = cal.predict_proba(logit(pv_w[b]).reshape(-1, 1))[:, 1]
        res["agirlikli_platt"] = (pp, C_w)
        for name, (p, C) in res.items():
            rows.append({"target": y, "variant": name, "C": C, "val_pr_auc": average_precision_score(yv, p),
                         "val_brier": brier_score_loss(yv, p), "val_ece": ece(yv, p),
                         "val_mean_pred": p.mean(), "val_obs": yv.mean()})
            log.info("select %s %-16s C=%-5s val PR-AUC %.3f | Brier %.4f | ECE %.3f | ort. %.3f (gözl. %.3f)",
                     y, name, C, rows[-1]["val_pr_auc"], rows[-1]["val_brier"], rows[-1]["val_ece"], p.mean(), yv.mean())
    R = pd.DataFrame(rows)
    # seçim kuralı: en düşük doğrulama Brier skoru (hem ayırt etme hem kalibrasyonu içerir)
    R["selected_by_val_brier"] = R.groupby("target").val_brier.transform("min") == R.val_brier
    save(R, "select_lr_variant", log)


def detrend_frame(F, s, mode):
    """Büyük ölçekli öznitelikleri eğilimden arındırır (il ve değişken bazında).

    Eğilim, her yılın mevsim ortalaması anomalisi (pl_<v>_an) üzerinden yıllık bir kayma olarak kestirilir:
      mode='train'     : yalnız 1981-2010 yıllarına doğrusal eğilim uydurulur, tüm yıllara uzatılır;
      mode='trailing5' : her yıl için önceki 5 sezonun ortalama anomalisi (yalnız geçmiş bilgi; ilk yıl 0).
    Kayma yıl içinde sabit olduğundan aynı değer ham değerden (pl_<v>), anomaliden ve onun gecikme/
    hareketli ortalama türevlerinden çıkarılır; 3 günlük eğilim (fark) öznitelikleri değişmez.
    """
    F = F.copy()
    tr0, tr1 = s["train"]
    for v in PL_BASE:
        an = f"pl_{v}_an"
        targets = [f"pl_{v}", an] + [c for c in F.columns if c.startswith(an + "_lag") or c.startswith(an + "_mean")]
        ym = F.groupby(["point_id", "year"])[an].mean().unstack(0)          # yıl x il
        if mode == "train":
            t = ym.loc[tr0:tr1]
            off = pd.DataFrame({p: np.polyval(np.polyfit(t.index.values, t[p].values, 1), ym.index.values) for p in ym.columns},
                               index=ym.index)
            off = off - off.loc[tr0:tr1].mean()          # eğitim dönemi ortalaması değişmez
        else:
            off = ym.shift(1).rolling(5, min_periods=1).mean() - ym.loc[tr0:tr1].mean()
            off = off.fillna(0.0)
        o = off.stack().rename("off").reset_index().rename(columns={"level_1": "point_id"})
        o.columns = ["year", "point_id", "off"]
        k = F[["year", "point_id"]].merge(o, on=["year", "point_id"], how="left")["off"].fillna(0.0).to_numpy()
        for c in targets:
            F[c] = F[c].to_numpy() - k
    return F


PL_BASE = ["z500", "z850", "t500", "t850", "thick", "z500_gda", "z500_anat", "z500_emed", "z500_iran", "z500_north",
           "t850_gda", "t850_anat", "t850_emed", "t850_iran", "t850_north", "z500_grad_we"]


def sec_detrend(cfg, F, feats, log, boot):
    """Majör 2a."""
    s = cfg["split"]
    rows = []
    for mode in ("train", "trailing5"):
        G = detrend_frame(F, s, mode)
        chk = G[G.year.between(2017, 2024)].pl_z500_anat_an.mean(), F[F.year.between(2017, 2024)].pl_z500_anat_an.mean()
        log.info("detrend %s: test dönemi ort. Orta Anadolu Z500 anomalisi %.1f gpm (önce %.1f)", mode, *chk)
        for y in TARGETS:
            te, o = lr_full_local(G, feats, s, y)
            yt = te[y].to_numpy(); yrs = te.year.to_numpy()
            pers = te["excess"].to_numpy()
            c1 = boot.compare(yt, o["full"], o["local"], yrs)
            c2 = boot.compare(yt, o["full"], pers, yrs)
            rows.append({"mode": mode, "target": y, "pr_auc_local": average_precision_score(yt, o["local"]),
                         "pr_auc_full_detrended": average_precision_score(yt, o["full"]),
                         "pr_auc_persistence": average_precision_score(yt, pers),
                         "full_minus_local": c1["diff"], "fl_ci_lo": c1["ci_lo"], "fl_ci_hi": c1["ci_hi"], "fl_p": c1["p"],
                         "full_minus_pers": c2["diff"], "fp_p": c2["p"], "C_full": o["C_full"],
                         "test_mean_z500_anat_an": chk[0]})
            log.info("detrend %-9s %s: yerel %.3f | tam (arındırılmış) %.3f | tam-yerel %+.3f [%+.3f, %+.3f] p=%.4f | tam-pers %+.3f p=%.4f",
                     mode, y, rows[-1]["pr_auc_local"], rows[-1]["pr_auc_full_detrended"], c1["diff"], c1["ci_lo"],
                     c1["ci_hi"], c1["p"], c2["diff"], c2["p"])
    save(pd.DataFrame(rows), "detrend_pl", log)


def regional_tmax_trailing():
    """Geçmiş 30/60/90 günlük, 9 il ortalaması Tmax anomalisi (1981-2010 takvim günü klimatolojisi; t dahil)."""
    D = pd.read_csv(ROOT / "data" / "interim" / "era5_daily.csv", parse_dates=["date"], usecols=["date", "point_id", "tmax"])
    D["doy"] = D.date.dt.dayofyear.clip(upper=365)
    base = D[D.date.dt.year.between(1981, 2010)]
    clim = base.groupby(["point_id", "doy"]).tmax.mean().groupby(level=0).transform(
        lambda x: x.rolling(15, center=True, min_periods=1).mean())
    D["an"] = D.tmax - clim.reindex(list(zip(D.point_id, D.doy))).values
    R = D.groupby("date").an.mean().sort_index()
    out = pd.DataFrame({f"reg_tmax_an_past{w}": R.rolling(w, min_periods=w // 2).mean() for w in (30, 60, 90)})
    return out.reset_index()


def sec_trendvars(cfg, F, feats, log, boot):
    """Majör 2b."""
    s = cfg["split"]
    T = regional_tmax_trailing()
    G = F.merge(T, on="date", how="left")
    G["year_lin"] = G.year.astype(float)
    extra = ["year_lin", "reg_tmax_an_past30", "reg_tmax_an_past60", "reg_tmax_an_past90"]
    fl = [c for c in feats if not c.startswith("pl_")] + extra
    ff = feats + extra
    rows = []
    for y in TARGETS:
        te, o = lr_full_local(G, ff, s, y, feats_full=ff, feats_local=fl)
        yt = te[y].to_numpy(); yrs = te.year.to_numpy(); pers = te["excess"].to_numpy()
        c1 = boot.compare(yt, o["full"], o["local"], yrs); c2 = boot.compare(yt, o["local"], pers, yrs)
        rows.append({"target": y, "pr_auc_local_plus": average_precision_score(yt, o["local"]),
                     "pr_auc_full_plus": average_precision_score(yt, o["full"]),
                     "full_minus_local": c1["diff"], "fl_ci_lo": c1["ci_lo"], "fl_ci_hi": c1["ci_hi"], "fl_p": c1["p"],
                     "local_minus_pers": c2["diff"], "lp_p": c2["p"]})
        log.info("trendvars %s: yerel+ %.3f | tam+ %.3f | tam-yerel %+.3f [%+.3f, %+.3f] p=%.4f | yerel+-pers %+.3f p=%.4f",
                 y, rows[-1]["pr_auc_local_plus"], rows[-1]["pr_auc_full_plus"], c1["diff"], c1["ci_lo"], c1["ci_hi"],
                 c1["p"], c2["diff"], c2["p"])
    save(pd.DataFrame(rows), "trend_variables", log)


def load_preds(F):
    P = pd.read_csv(ROOT / "results" / "predictions_test.csv", parse_dates=["date"])
    R = F[["date", "point_id", "run_len", "exceed", "excess", "exceed_cnt3", "exceed_cnt7"]]
    return P.merge(R, on=["date", "point_id"], how="left")


def sec_early(cfg, F, feats, log, boot):
    """Majör 3."""
    P = load_preds(F)
    rows = []
    for y in TARGETS:
        d = P[P.target == y]
        pos = d[d.y_true == 1]
        share = pos.run_len.value_counts(normalize=True).sort_index()
        sub = d[d.run_len == 0]
        yt, yrs = sub.y_true.to_numpy(), sub.year.to_numpy()
        r = {"target": y, "n_pos": len(pos), "pos_share_run0": share.get(0, 0), "pos_share_run1": share.get(1, 0),
             "pos_share_run2": share.get(2, 0), "n_sub": len(sub), "pos_rate_sub": yt.mean()}
        for m in ("persistence", "logreg_local", "logreg_full", "lgbm_full", "climatology"):
            r[f"pr_auc_{m}"] = average_precision_score(yt, sub[m])
            lo, hi = boot.ci(yt, sub[m].to_numpy(), yrs); r[f"ci_lo_{m}"], r[f"ci_hi_{m}"] = lo, hi
        for a, b in (("logreg_full", "logreg_local"), ("logreg_full", "persistence"), ("logreg_local", "persistence")):
            c = boot.compare(yt, sub[a].to_numpy(), sub[b].to_numpy(), yrs)
            r[f"{a}-{b}"] = c["diff"]; r[f"{a}-{b}_p"] = c["p"]; r[f"{a}-{b}_lo"] = c["ci_lo"]; r[f"{a}-{b}_hi"] = c["ci_hi"]
        rows.append(r)
        log.info("early %s: pozitiflerin %%%.0f'inde t'de aşım yok (run 1: %%%.0f, run 2: %%%.0f) | alt küme n=%d pozitif %%%.1f | "
                 "PR-AUC pers %.3f, LR yerel %.3f, LR tam %.3f | tam-yerel %+.3f p=%.4f | tam-pers %+.3f p=%.4f | yerel-pers %+.3f p=%.4f",
                 y, 100 * r["pos_share_run0"], 100 * r["pos_share_run1"], 100 * r["pos_share_run2"], len(sub), 100 * yt.mean(),
                 r["pr_auc_persistence"], r["pr_auc_logreg_local"], r["pr_auc_logreg_full"],
                 r["logreg_full-logreg_local"], r["logreg_full-logreg_local_p"], r["logreg_full-persistence"],
                 r["logreg_full-persistence_p"], r["logreg_local-persistence"], r["logreg_local-persistence_p"])
    save(pd.DataFrame(rows), "early_warning_subset", log)


PAIRS = [("logreg_full", "logreg_local"), ("logreg_full", "persistence"), ("logreg_local", "persistence"),
         ("lgbm_full", "lgbm_local"), ("xgb_full", "xgb_local"), ("hgb_full", "hgb_local"), ("rf_full", "rf_local"),
         ("logreg_full", "lgbm_full"), ("logreg_full", "xgb_full"), ("logreg_full", "hgb_full"), ("logreg_full", "rf_full")]


def sec_sig(cfg, F, feats, log, boot):
    """Majör 4-5: 10.000 ortak bootstrap, Holm düzeltmesi."""
    P = pd.read_csv(ROOT / "results" / "predictions_test.csv")
    rows = []
    for y in ["y1", "y3", "y5", "y7"]:
        d = P[P.target == y]
        if d.empty:
            continue
        yt, yrs = d.y_true.to_numpy(), d.year.to_numpy()
        for a, b in PAIRS:
            if a not in d or b not in d or d[a].isna().any() or d[b].isna().any():
                continue
            c = boot.compare(yt, d[a].to_numpy(), d[b].to_numpy(), yrs)
            rows.append({"target": y, "model_a": a, "model_b": b, **c})
    R = pd.DataFrame(rows)
    fam = R.target.isin(TARGETS)
    R["p_holm"] = np.nan
    R.loc[fam, "p_holm"] = holm(R.loc[fam, "p"])
    for _, r in R.iterrows():
        log.info("sig %s %-12s - %-13s %+.3f [%+.3f, %+.3f] p=%.4f Holm=%.4f", r.target, r.model_a, r.model_b,
                 r["diff"], r.ci_lo, r.ci_hi, r.p, r.p_holm if r.p_holm == r.p_holm else np.nan)
    save(R, "significance_10k", log)
    # CI'lar (tek model)
    ci = []
    for y in TARGETS:
        d = P[P.target == y]; yt, yrs = d.y_true.to_numpy(), d.year.to_numpy()
        for m in ("climatology", "persistence", "logreg_local", "logreg_full", "rf_full", "hgb_full", "xgb_full", "lgbm_full",
                  "rf_local", "hgb_local", "xgb_local", "lgbm_local"):
            if m in d and not d[m].isna().any():
                lo, hi = boot.ci(yt, d[m].to_numpy(), yrs)
                ci.append({"target": y, "model": m, "pr_auc": average_precision_score(yt, d[m]), "ci_lo": lo, "ci_hi": hi})
    save(pd.DataFrame(ci), "pr_auc_ci_10k", log)


def sec_sens(cfg, F, feats, log, boot):
    root = ROOT / "sensitivity"
    if not root.exists():
        log.warning("sens: sensitivity/ klasörü yok, atlanıyor"); return
    rows = []
    for v in sorted(p.name for p in root.iterdir() if p.is_dir()):
        f = root / v / "results" / "predictions_test.csv"
        if not f.exists():
            continue
        P = pd.read_csv(f)
        for y in ("y3", "y7"):
            d = P[P.target == y]
            if d.empty:
                continue
            yt, yrs = d.y_true.to_numpy(), d.year.to_numpy()
            bt = boot if set(np.unique(yrs)) <= set(boot.years) and len(np.unique(yrs)) == len(boot.years) else Boot(yrs)
            r = {"variant": v, "target": y, "pos_rate": yt.mean()}
            for m in ("persistence", "logreg_local", "logreg_full"):
                r[m] = average_precision_score(yt, d[m])
            for a, b in PAIRS[:3]:
                c = bt.compare(yt, d[a].to_numpy(), d[b].to_numpy(), yrs)
                r[f"{a}-{b}"] = c["diff"]; r[f"{a}-{b}_p"] = c["p"]; r[f"{a}-{b}_lo"] = c["ci_lo"]; r[f"{a}-{b}_hi"] = c["ci_hi"]
            rows.append(r)
            log.info("sens %-10s %s: tam-yerel %+.3f p=%.4f | tam-pers %+.3f p=%.4f | yerel-pers %+.3f p=%.4f", v, y,
                     r["logreg_full-logreg_local"], r["logreg_full-logreg_local_p"], r["logreg_full-persistence"],
                     r["logreg_full-persistence_p"], r["logreg_local-persistence"], r["logreg_local-persistence_p"])
    save(pd.DataFrame(rows), "sensitivity_10k", log)


def sec_baselines(cfg, F, feats, log, boot):
    """Majör 9: daha adil referanslar."""
    s = cfg["split"]
    P = load_preds(F)
    rows = []
    simple = ["excess", "run_len", "exceed_cnt7"]
    for y in TARGETS:
        D = F[F[y].notna()].copy(); D[y] = D[y].astype(int)
        tr, va, te = split(D, s); trva = pd.concat([tr, va])
        yt, yrs = te[y].to_numpy(), te.year.to_numpy()
        doy = lambda X: X.date.dt.dayofyear
        # (a) yakın dönem klimatolojisi: 2007-2016 takvim günü oranı (±7 gün)
        rec = trva[trva.year.between(2007, 2016)]
        rate = rec.groupby(doy(rec))[y].mean().reindex(range(1, 367)).rolling(15, center=True, min_periods=1).mean()
        p_rec = doy(te).map(rate).fillna(rec[y].mean()).to_numpy()
        # (b) eğilimli klimatoloji: yıl + mevsim harmonikleri ile lojistik model (1981-2016)
        def H(X):
            a = 2 * np.pi * doy(X) / 365.25
            return np.c_[X.year - 2000, np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)]
        tc = LogisticRegression(C=1e6, max_iter=2000).fit(H(trva), trva[y])
        p_tc = tc.predict_proba(H(te))[:, 1]
        # (c) basit LR: sıcaklık fazlası, ardışık aşım sayısı, son 7 gündeki aşım sayısı
        m, _, C, _ = fit_select(lambda C: lr(C), (0.01, 0.1, 1.0, 10.0), tr, va, simple, y, refit_on=trva)
        p_simple = m.predict_proba(te[simple])[:, 1]
        d = P[P.target == y].set_index(["date", "point_id"]).loc[list(zip(te.date, te.point_id))]
        main = d.logreg_full.to_numpy()
        ref_const = np.full(len(yt), tr[y].mean())
        bs = lambda p: brier_score_loss(yt, p)
        r = {"target": y, "obs_rate_test": yt.mean(), "rate_train": tr[y].mean(), "rate_2007_2016": rec[y].mean(),
             "pr_auc_clim_1981_2016": average_precision_score(yt, d.climatology),
             "pr_auc_clim_recent": average_precision_score(yt, p_rec), "pr_auc_clim_trend": average_precision_score(yt, p_tc),
             "pr_auc_simple_lr": average_precision_score(yt, p_simple), "simple_lr_C": C,
             "pr_auc_persistence": average_precision_score(yt, d.persistence), "pr_auc_main": average_precision_score(yt, main),
             "bss_vs_const_train_rate": 1 - bs(main) / bs(ref_const), "bss_vs_clim_recent": 1 - bs(main) / bs(p_rec),
             "bss_vs_clim_trend": 1 - bs(main) / bs(p_tc), "bss_simple_vs_clim_trend": 1 - bs(p_simple) / bs(p_tc)}
        for name, p in (("simple_lr", p_simple), ("clim_trend", p_tc), ("clim_recent", p_rec)):
            c = boot.compare(yt, main, p, yrs); r[f"main-{name}"] = c["diff"]; r[f"main-{name}_p"] = c["p"]
        c = boot.compare(yt, d.logreg_local.to_numpy(), p_simple, yrs)
        r["local-simple_lr"] = c["diff"]; r["local-simple_lr_p"] = c["p"]
        rows.append(r)
        log.info("baselines %s: gözl. %.3f | klim(2007-16) PR %.3f | eğilimli klim PR %.3f | basit LR %.3f | pers %.3f | ana %.3f | "
                 "BSS sabit %.3f, yakın klim %.3f, eğilimli klim %.3f | ana-basit %+.3f p=%.4f | yerel-basit %+.3f p=%.4f",
                 y, yt.mean(), r["pr_auc_clim_recent"], r["pr_auc_clim_trend"], r["pr_auc_simple_lr"], r["pr_auc_persistence"],
                 r["pr_auc_main"], r["bss_vs_const_train_rate"], r["bss_vs_clim_recent"], r["bss_vs_clim_trend"],
                 r["main-simple_lr"], r["main-simple_lr_p"], r["local-simple_lr"], r["local-simple_lr_p"])
    save(pd.DataFrame(rows), "baselines", log)


def tree_models(weighted):
    cw = "balanced" if weighted else None
    return {
        "rf": (lambda l: RandomForestClassifier(n_estimators=400, min_samples_leaf=l, max_features="sqrt",
                                                class_weight="balanced_subsample" if weighted else None,
                                                n_jobs=-1, random_state=42), (10, 30)),
        "hgb": (lambda g: HistGradientBoostingClassifier(learning_rate=g[0], max_leaf_nodes=g[1], min_samples_leaf=50,
                                                         l2_regularization=1.0, max_iter=600, early_stopping=True,
                                                         validation_fraction=0.15, class_weight=cw, random_state=42),
                [(0.03, 15), (0.03, 31), (0.08, 15), (0.08, 31)]),
    }


def opt_models(weighted, pos_w):
    out = {}
    try:
        from xgboost import XGBClassifier
        out["xgb"] = (lambda g: XGBClassifier(n_estimators=600, learning_rate=g[0], max_depth=g[1], subsample=0.8,
                                              colsample_bytree=0.8, scale_pos_weight=pos_w if weighted else 1.0,
                                              eval_metric="aucpr", random_state=42, n_jobs=-1),
                      [(0.03, 4), (0.03, 6), (0.08, 4), (0.08, 6)])
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        out["lgbm"] = (lambda g: LGBMClassifier(n_estimators=600, learning_rate=g[0], num_leaves=g[1], subsample=0.8,
                                                subsample_freq=1, colsample_bytree=0.8, is_unbalance=weighted,
                                                random_state=42, verbose=-1),
                       [(0.03, 15), (0.03, 31), (0.08, 15), (0.08, 31)])
    except ImportError:
        pass
    return out


def sec_trees(cfg, F, feats, log, boot):
    """Majör 10."""
    rows, oor = [], []
    splits = {"ana (test 2017-2024)": cfg["split"],
              "alternatif (test 2011-2016)": {"train": [1981, 2004], "val": [2005, 2010], "test": [2011, 2016]}}
    for sname, s in splits.items():
        for y in TARGETS:
            D = F[F[y].notna()].copy(); D[y] = D[y].astype(int)
            tr, va, te = split(D, s); trva = pd.concat([tr, va])
            yt = te[y].to_numpy(); yrs = te.year.to_numpy()
            bt = boot if sname.startswith("ana") else Boot(yrs)
            base = tr[y].mean(); pos_w = (1 - base) / base
            preds = {}
            m, _, C, _ = fit_select(lambda C: lr(C), (0.01, 0.1, 1.0), tr, va, feats, y, refit_on=trva)
            preds["logreg"] = m.predict_proba(te[feats])[:, 1]
            for weighted in (False, True):
                if sname.startswith("ana") and weighted:
                    continue          # ağırlıklı ağaçların ana sonuçları 04'te zaten var
                models = {**tree_models(weighted), **opt_models(weighted, pos_w)}
                for name, (make, grid) in models.items():
                    t0 = time.time()
                    mm, _, g, _ = fit_select(make, grid, tr, va, feats, y, refit_on=trva)
                    preds[f"{name}_{'w' if weighted else 'u'}"] = mm.predict_proba(te[feats])[:, 1]
                    log.info("trees %s %s %s (%s): %.0f sn", sname, y, name, "ağırlıklı" if weighted else "ağırlıksız", time.time() - t0)
            for k, p in preds.items():
                r = {"split": sname, "target": y, "model": k, "pr_auc": average_precision_score(yt, p),
                     "brier": brier_score_loss(yt, p)}
                if k != "logreg":
                    c = bt.compare(yt, preds["logreg"], p, yrs); r.update({"lr_minus": c["diff"], "p": c["p"]})
                rows.append(r)
                log.info("trees %s %s %-7s PR-AUC %.3f%s", sname, y, k, r["pr_auc"],
                         f" | LR-model {r['lr_minus']:+.3f} p={r['p']:.4f}" if "p" in r else "")
            if y == "y7":
                lo, hi = tr[feats].min(), tr[feats].max()
                out = (te[feats] < lo) | (te[feats] > hi)
                oor.append({"split": sname, "share_rows_any_feature_outside": out.any(axis=1).mean(),
                            "share_rows_any_pl_outside": out[[c for c in feats if c.startswith("pl_")]].any(axis=1).mean(),
                            "share_rows_tmax_an_outside": out["tmax_an"].mean(),
                            "share_rows_z500_anat_an_outside": out["pl_z500_anat_an"].mean(),
                            "mean_share_per_feature": out.mean().mean(),
                            "top_features": json.dumps(out.mean().sort_values(ascending=False).head(8).round(4).to_dict())})
                log.info("trees %s: eğitim aralığı dışında en az bir özniteliği olan test satırı %%%.1f", sname,
                         100 * oor[-1]["share_rows_any_feature_outside"])
    save(pd.DataFrame(rows), "trees", log)
    save(pd.DataFrame(oor), "out_of_range", log)


def sec_threshold(cfg, F, feats, log, boot):
    """Majör 11: eşik, yeniden eğitilmeyen modelle aktarılır; ana yaklaşımla karşılaştırılır."""
    s = cfg["split"]; rows = []

    def best_threshold(y, p):
        qs = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 300)))
        f1 = [2 * ((p >= q) & (y == 1)).sum() / max(2 * ((p >= q) & (y == 1)).sum() + ((p >= q) & (y == 0)).sum()
                                                     + ((p < q) & (y == 1)).sum(), 1) for q in qs]
        return float(qs[int(np.argmax(f1))])

    def cat(y, p, thr):
        pr = p >= thr; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = (~pr & (y == 1)).sum()
        return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "pod": tp / max(tp + fn, 1), "far": fp / max(tp + fp, 1)}

    for y in TARGETS:
        D = F[F[y].notna()].copy(); D[y] = D[y].astype(int)
        tr, va, te = split(D, s); trva = pd.concat([tr, va])
        yv, yt = va[y].to_numpy(), te[y].to_numpy()
        m, pv, C, _ = fit_select(lambda C: lr(C), (0.01, 0.1, 1.0), tr, va, feats, y)
        thr = best_threshold(yv, pv)
        p_norefit = m.predict_proba(te[feats])[:, 1]
        m2 = clone(m).fit(trva[feats], trva[y]); p_refit = m2.predict_proba(te[feats])[:, 1]
        thr_opt_test = best_threshold(yt, p_refit)   # yalnız bilgi amaçlı (test üzerinde en iyi eşik; kullanılmaz)
        for name, p in (("yeniden_egitilmeyen", p_norefit), ("yeniden_egitilen", p_refit)):
            rows.append({"target": y, "model": name, "threshold": thr, "pr_auc": average_precision_score(yt, p),
                         **cat(yt, p, thr), "val_rate": yv.mean(), "test_rate": yt.mean(), "mean_pred_test": p.mean(),
                         "f1_at_test_optimal_threshold_info": cat(yt, p_refit, thr_opt_test)["f1"] if name == "yeniden_egitilen" else np.nan})
            log.info("threshold %s %-20s eşik %.3f | PR-AUC %.3f | F1 %.3f POD %.2f FAR %.2f | ort. tahmin %.3f (doğr. %.3f, test %.3f)",
                     y, name, thr, rows[-1]["pr_auc"], rows[-1]["f1"], rows[-1]["pod"], rows[-1]["far"], p.mean(), yv.mean(), yt.mean())
    save(pd.DataFrame(rows), "threshold_transfer", log)


def add_q(F):
    """Özgül nem (g/kg) ve anomalisi + türevleri: sıcaklıktan bağımsız nem içeriği."""
    F = F.copy()
    td = F["tdew_mean"]; p = F["sp_mean"]          # sp_mean hPa
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    F["q"] = 1000 * 0.622 * e / (p - 0.378 * e)
    F["doy"] = F.date.dt.dayofyear.clip(upper=365)
    base = F[F.year.between(1981, 2010)]
    clim = base.groupby(["point_id", "doy"]).q.mean().groupby(level=0).transform(
        lambda x: x.rolling(15, center=True, min_periods=1).mean())
    F["q_an"] = F.q - clim.reindex(list(zip(F.point_id, F.doy))).values
    F = F.sort_values(["point_id", "date"])
    g = F.groupby("point_id")
    for k in (1, 2, 3):
        F[f"q_an_lag{k}"] = g["q_an"].shift(k)
    for w in (3, 7, 14):
        F[f"q_an_mean{w}"] = g["q_an"].transform(lambda x, w=w: x.rolling(w, min_periods=max(1, w // 2)).mean())
    qcols = ["q", "q_an"] + [f"q_an_lag{k}" for k in (1, 2, 3)] + [f"q_an_mean{w}" for w in (3, 7, 14)]
    F[qcols] = F[qcols].fillna(F.groupby("point_id")[qcols].transform("mean"))
    return F.drop(columns="doy"), qcols


def fam(c):
    c = re.sub(r"_(lag\d|mean\d+|trend3|change7)$", "", c); c = re.sub(r"_an$", "", c)
    c = re.sub(r"^(tp_sum)\d+$", r"\1", c); c = re.sub(r"^doy_(sin|cos)$", "doy", c)
    return c


GROUPS2 = {
    "Sıcaklık durumu": {"excess", "exceed", "run_len", "exceed_cnt3", "exceed_cnt7", "tmax", "tmin", "tmean", "dtr", "skt_max", "days_since_hw"},
    "VPD ve bağıl nem (sıcaklığa bağlı)": {"vpd_max", "rh_mean", "rh_min"},
    "Nem içeriği (çiy noktası, özgül nem)": {"tdew_mean", "q"},
    "Toprak nemi ve yağış": {"swvl1", "swvl2", "tp_sum", "days_since_rain"},
    "Yüzey basıncı, rüzgâr, radyasyon": {"sp_mean", "ws_mean", "ws_max", "ssrd_sum"},
    "Mevsim ve konum": {"doy", "lat", "lon"},
}


def group2(c):
    f = fam(c)
    if f.startswith("pl_t"):
        return "T500/T850 ve kalınlık (büyük ölçekli)"
    if f.startswith("pl_"):
        return "Z500 (büyük ölçekli)"
    for g, mem in GROUPS2.items():
        if f in mem:
            return g
    return "Diğer"


def sec_dryness(cfg, F, feats, log, boot, n_boot=200):
    """Majör 8: ayrıştırılmış gruplar, bootstrap GA'lı mutlak ve göreli düşüş; özgül nemle model."""
    s = cfg["split"]
    G, qcols = add_q(F)
    feats_q = feats + qcols
    rng = np.random.default_rng(SEED)
    rows, perf = [], []
    for y in TARGETS:
        D = G[G[y].notna()].copy(); D[y] = D[y].astype(int)
        tr, va, te = split(D, s); trva = pd.concat([tr, va])
        yt = te[y].to_numpy(); yrs = te.year.to_numpy(); uy = np.unique(yrs)
        for fsname, fs in (("ana", feats), ("ana+ozgul_nem", feats_q)):
            m, _, C, _ = fit_select(lambda C: lr(C), (0.01, 0.1, 1.0), tr, va, fs, y, refit_on=trva)
            p0 = m.predict_proba(te[fs])[:, 1]
            perf.append({"target": y, "feature_set": fsname, "pr_auc": average_precision_score(yt, p0), "C": C})
            log.info("dryness %s %s: PR-AUC %.3f", y, fsname, perf[-1]["pr_auc"])
            if fsname != "ana":           # önem analizi ana modelde (özgül nem modeli yalnız performans karşılaştırması)
                continue
            gm = {}
            for c in fs:
                gm.setdefault(group2(c), []).append(c)
            X = te[fs]
            # her grup için bir kez karıştırılmış tahmin (5 tekrar ortalaması), sonra yıl bloğu bootstrap'ı
            perm_p = {}
            for g, cols in gm.items():
                ps = []
                for _ in range(5):
                    Xp = X.copy(); Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
                    ps.append(m.predict_proba(Xp)[:, 1])
                perm_p[g] = ps
            idx_by_year = {u: np.nonzero(yrs == u)[0] for u in uy}
            for g, ps in perm_p.items():
                drops, rel = [], []
                for _ in range(n_boot):
                    ii = np.concatenate([idx_by_year[u] for u in rng.choice(uy, len(uy))])
                    b0 = average_precision_score(yt[ii], p0[ii])
                    bp = np.mean([average_precision_score(yt[ii], p[ii]) for p in ps])
                    drops.append(b0 - bp); rel.append((b0 - bp) / b0)
                b0 = average_precision_score(yt, p0); bp = np.mean([average_precision_score(yt, p) for p in ps])
                rows.append({"target": y, "group": g, "n_features": len(gm[g]), "base": b0, "drop": b0 - bp,
                             "drop_ci_lo": np.percentile(drops, 2.5), "drop_ci_hi": np.percentile(drops, 97.5),
                             "rel_drop": (b0 - bp) / b0, "rel_ci_lo": np.percentile(rel, 2.5), "rel_ci_hi": np.percentile(rel, 97.5)})
                log.info("dryness %s %-40s düşüş %.3f [%.3f, %.3f] | göreli %.3f [%.3f, %.3f]", y, g, rows[-1]["drop"],
                         rows[-1]["drop_ci_lo"], rows[-1]["drop_ci_hi"], rows[-1]["rel_drop"], rows[-1]["rel_ci_lo"], rows[-1]["rel_ci_hi"])
    save(pd.DataFrame(rows), "group_importance_boot", log)
    save(pd.DataFrame(perf), "specific_humidity_model", log)


def sec_boxes(cfg, F, feats, log, boot):
    """Majör 7: çakışmayan Orta Anadolu kutusu."""
    f = ROOT / "data" / "interim" / "pl_daily.csv"
    pl = pd.read_csv(f, parse_dates=["date"]) if f.exists() else None
    if pl is None or "z500_anat_n" not in pl:
        log.warning("boxes: pl_daily.csv içinde z500_anat_n yok (02d'yi yeni config ile çalıştırın); atlanıyor"); return
    s = cfg["split"]
    pl = pl[["date", "point_id", "z500_anat_n", "t850_anat_n"]].rename(columns=lambda c: f"pl_{c}" if c.endswith("anat_n") else c)
    G = F.merge(pl, on=["date", "point_id"], how="left")
    G["doy"] = G.date.dt.dayofyear.clip(upper=365)
    for v in ("pl_z500_anat_n", "pl_t850_anat_n"):
        base = G[G.year.between(1981, 2010)]
        clim = base.groupby("doy")[v].mean().rolling(15, center=True, min_periods=1).mean()
        G[f"{v}_an"] = G[v] - G.doy.map(clim)
    G = G.sort_values(["point_id", "date"]); g = G.groupby("point_id")
    v = "pl_z500_anat_n_an"
    for k in (1, 2, 3):
        G[f"{v}_lag{k}"] = g[v].shift(k)
    for w in (3, 7):
        G[f"{v}_mean{w}"] = g[v].transform(lambda x, w=w: x.rolling(w, min_periods=max(1, w // 2)).mean())
    newc = [c for c in G.columns if "anat_n" in c]
    G[newc] = G[newc].fillna(G.groupby("point_id")[newc].transform("mean"))
    old = [c for c in feats if "_anat" in c]
    ff = [c for c in feats if c not in old] + newc
    rows = []
    for y in TARGETS:
        te, o = lr_full_local(G, ff, s, y, feats_full=ff)
        yt, yrs = te[y].to_numpy(), te.year.to_numpy()
        c = boot.compare(yt, o["full"], o["local"], yrs)
        # doğrusal SHAP aile sıralaması
        D = G[G[y].notna()].copy(); D[y] = D[y].astype(int)
        tr, va, te2 = split(D, s); trva = pd.concat([tr, va])
        sc = StandardScaler().fit(trva[ff]); L = LogisticRegression(C=o["C_full"], max_iter=3000).fit(sc.transform(trva[ff]), trva[y])
        S = pd.DataFrame(sc.transform(te2[ff]) * L.coef_[0], columns=ff)
        imp = S.T.groupby(fam).sum().T.abs().mean().sort_values(ascending=False)
        rows.append({"target": y, "pr_auc_local": average_precision_score(yt, o["local"]),
                     "pr_auc_full_anat_n": average_precision_score(yt, o["full"]), "full_minus_local": c["diff"],
                     "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"], "p": c["p"], "top_families": json.dumps(imp.head(6).round(4).to_dict()),
                     "rank_anat_n": int(list(imp.index).index("pl_z500_anat_n") + 1) if "pl_z500_anat_n" in imp.index else -1,
                     "rank_local_z500": int(list(imp.index).index("pl_z500") + 1) if "pl_z500" in imp.index else -1})
        log.info("boxes %s: tam(anat_n) %.3f | yerel %.3f | fark %+.3f p=%.4f | anat_n SHAP sırası %d | il hücresi Z500 sırası %d | ilk aileler %s",
                 y, rows[-1]["pr_auc_full_anat_n"], rows[-1]["pr_auc_local"], c["diff"], c["p"], rows[-1]["rank_anat_n"],
                 rows[-1]["rank_local_z500"], rows[-1]["top_families"])
    save(pd.DataFrame(rows), "nonoverlap_box", log)


SECTIONS = {"select": sec_select, "early": sec_early, "sig": sec_sig, "sens": sec_sens, "baselines": sec_baselines,
            "threshold": sec_threshold, "detrend": sec_detrend, "trendvars": sec_trendvars, "dryness": sec_dryness,
            "boxes": sec_boxes, "trees": sec_trees}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    log = get_logger("11_revision", cfg)
    F, feats = load_features()
    te_years = F[F.year.between(*cfg["split"]["test"])].year.unique()
    boot = Boot(te_years)
    log.info("Bootstrap: %d tekrar, tohum %d, %d test yılı (9 il birlikte örneklenir)", B, SEED, len(te_years))
    for name, fn in SECTIONS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        if name == "select":
            fn(cfg, F, feats, log)
        else:
            fn(cfg, F, feats, log, boot)
        log.info("[%s] %.0f sn", name, time.time() - t0)


if __name__ == "__main__":
    main()
