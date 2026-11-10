"""Aşama 6 - Kalibrasyon, SHAP ve il bazında sonuçlar (makale şekilleri ve tabloları).

Bölümler (hepsi varsayılan olarak çalışır; --only ile tek bölüm seçilebilir):
  calibration : Lojistik regresyonun (tam öznitelik seti) olasılık kalibrasyonu.
                Üç varyant: (a) 04'teki sınıf ağırlıklı model, (b) aynı model + Platt ölçekleme
                (eğitimde öğrenilir, doğrulamada kalibre edilir), (c) ağırlıksız model.
                Brier, BSS (referans: eğitim+doğrulama pozitif oranı), ECE; güvenilirlik diyagramı.
  groups      : Grup bazında permütasyon önemi (bir öncül grubunun tüm sütunları birlikte karıştırılır;
                birbirine bağlı değişkenlerde tek tek önemden daha sağlam). Model: ağırlıksız LR, tam set.
  shap        : Doğrusal SHAP (ağırlıksız lojistik regresyon, standartlaştırılmış uzayda katsayı x değer).
                Gecikmeli/ortalama türevler aynı "aile" altında toplanır (ör. pl_z500_anat_*),
                çünkü birbirine bağlı öznitelikler tek tek ele alınınca etki bölünür.
                LightGBM kuruluysa ağaç SHAP'ı (pred_contrib) da hesaplanır.
  regional    : İl bazında PR-AUC (persistence, LR yerel, LR tam).
  lgbm        : Yalnızca LightGBM ağaç SHAP'ı (matplotlib gerekmez; tablolar yazılır, şekil varsa çizilir).

Girdi : data/processed/features.csv, results/predictions_test.csv, results/metrics_test.csv
Çıktı : results/analysis/*.csv, figures/*.png ve *.pdf

Kullanım:
    python 06_analysis.py
    python 06_analysis.py --only shap
"""
from __future__ import annotations

import argparse
import json
import re
import warnings

import numpy as np
try:  # şekiller için; yoksa yalnızca tablolar üretilir
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    plt, HAVE_MPL = None, False
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, get_logger, load_config

warnings.filterwarnings("ignore")
META = ["date", "point_id", "year", "in_event"]
NAMES = {"gaziantep": "Gaziantep", "adiyaman": "Adıyaman", "kilis": "Kilis", "sanliurfa": "Şanlıurfa",
         "diyarbakir": "Diyarbakır", "mardin": "Mardin", "batman": "Batman", "siirt": "Siirt", "sirnak": "Şırnak"}
# Şekil renkleri (gri öncelikli; vurgu = mavi)
INK, INK2, GRID, MUTED, ACCENT, ACCENT2 = "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29b", "#2a78d6", "#eb6834"
FAMILY_TR = {
    "excess": "Bugünkü sıcaklık fazlası (Tmax − eşik)", "exceed": "Bugün eşik aşıldı mı", "run_len": "Ardışık aşım günü",
    "exceed_cnt3": "Son 3 günde aşım sayısı", "exceed_cnt7": "Son 7 günde aşım sayısı",
    "tmax": "Tmax", "tmin": "Tmin", "tmean": "Tmean", "dtr": "Günlük sıcaklık farkı", "tdew_mean": "Çiy noktası",
    "rh_mean": "Bağıl nem (ort.)", "rh_min": "Bağıl nem (min.)", "vpd_max": "Buhar basıncı açığı",
    "skt_max": "Yüzey sıcaklığı", "sp_mean": "Yüzey basıncı", "ws_mean": "Rüzgâr hızı (ort.)", "ws_max": "Rüzgâr hızı (maks.)",
    "ssrd_sum": "Güneş radyasyonu", "swvl1": "Toprak nemi 0–7 cm", "swvl2": "Toprak nemi 7–28 cm",
    "tp_sum": "Yağış toplamı (7–30 gün)", "days_since_rain": "Son yağıştan bu yana gün", "days_since_hw": "Son dalgadan bu yana gün",
    "doy": "Mevsim (yılın günü)", "lat": "Enlem", "lon": "Boylam",
    "pl_z500": "Z500 (il hücresi)", "pl_z850": "Z850 (il hücresi)", "pl_t500": "T500 (il hücresi)", "pl_t850": "T850 (il hücresi)",
    "pl_thick": "500–850 hPa kalınlığı", "pl_z500_gda": "Z500 Güneydoğu Anadolu", "pl_z500_anat": "Z500 Orta Anadolu",
    "pl_z500_emed": "Z500 Doğu Akdeniz", "pl_z500_iran": "Z500 İran", "pl_z500_north": "Z500 Karadeniz",
    "pl_t850_gda": "T850 Güneydoğu Anadolu", "pl_t850_anat": "T850 Orta Anadolu", "pl_t850_emed": "T850 Doğu Akdeniz",
    "pl_t850_iran": "T850 İran", "pl_t850_north": "T850 Karadeniz", "pl_z500_grad_we": "Z500 batı–doğu farkı",
}


def family(c: str) -> str:
    c = re.sub(r"_(lag\d|mean\d+|trend3|change7)$", "", c)
    c = re.sub(r"_an$", "", c)
    c = re.sub(r"^(tp_sum)\d+$", r"\1", c)
    c = re.sub(r"^doy_(sin|cos)$", "doy", c)
    return c


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    d = ROOT / "figures"; d.mkdir(exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load(cfg):
    F = pd.read_csv(ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])
    ycols = [c for c in F.columns if c.startswith("y") and c[1:].isdigit()]
    feats = [c for c in F.columns if c not in META and c not in ycols]
    return F, feats


def split(D, cfg):
    s = cfg["split"]; yr = D.year
    return (D[yr.between(*s["train"])], D[yr.between(*s["val"])], D[yr.between(*s["test"])])


def lr_C(target, fs):
    m = pd.read_csv(ROOT / "results" / "metrics_test.csv")
    r = m[(m.target == target) & (m.model == "logreg") & (m.feature_set == fs)]
    return float(json.loads(r.params.iloc[0]).get("C", 0.1)) if len(r) else 0.1


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def calibration(cfg, F, feats, log, out):
    rows, curves = [], {}
    for t in ["y3", "y5", "y7"]:
        D = F[F[t].notna()]; tr, va, te = split(D, cfg); trva = pd.concat([tr, va])
        y_te = te[t].to_numpy().astype(int); ref = trva[t].mean()
        C = lr_C(t, "full")
        variants = {}
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(trva[feats], trva[t])
        variants["Sınıf ağırlıklı (04)"] = m.predict_proba(te[feats])[:, 1]
        m_tr = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000)).fit(tr[feats], tr[t])
        lv = np.log(np.clip(m_tr.predict_proba(va[feats])[:, 1], 1e-6, 1 - 1e-6)); lv = lv - np.log1p(-np.exp(lv))
        platt = LogisticRegression(C=1e6).fit(lv.reshape(-1, 1), va[t])
        lt = np.log(np.clip(m_tr.predict_proba(te[feats])[:, 1], 1e-6, 1 - 1e-6)); lt = lt - np.log1p(-np.exp(lt))
        variants["Sınıf ağırlıklı + Platt"] = platt.predict_proba(lt.reshape(-1, 1))[:, 1]
        m_u = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(trva[feats], trva[t])
        variants["Ağırlıksız"] = m_u.predict_proba(te[feats])[:, 1]
        bs_ref = brier_score_loss(y_te, np.full(len(y_te), ref))
        for name, p in variants.items():
            bs = brier_score_loss(y_te, p)
            rows.append({"target": t, "variant": name, "pr_auc": average_precision_score(y_te, p), "brier": bs,
                         "bss": 1 - bs / bs_ref, "ece": ece(y_te, p), "mean_pred": p.mean(), "obs_rate": y_te.mean()})
            log.info("%s %-26s PR-AUC %.3f | Brier %.4f | BSS %+.3f | ECE %.3f | ort. tahmin %.3f (gözlenen %.3f)",
                     t, name, rows[-1]["pr_auc"], bs, rows[-1]["bss"], rows[-1]["ece"], p.mean(), y_te.mean())
        curves[t] = (y_te, variants)
    pd.DataFrame(rows).round(4).to_csv(out / "calibration.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    cols = {"Sınıf ağırlıklı (04)": MUTED, "Sınıf ağırlıklı + Platt": ACCENT, "Ağırlıksız": ACCENT2}
    for ax, t in zip(axes, ["y3", "y7"]):
        y, V = curves[t]
        ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1)
        for name, p in V.items():
            q = np.quantile(p, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
            mp = [p[idx == b].mean() for b in range(10)]; ob = [y[idx == b].mean() for b in range(10)]
            ax.plot(mp, ob, marker="o", ms=4, lw=2, color=cols[name], label=name)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); style(ax); ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title({"y3": "3 gün önceden", "y7": "7 gün önceden"}[t], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Tahmin edilen olasılık", color=INK2, fontsize=9)
    axes[0].set_ylabel("Gözlenen sıklık", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Güvenilirlik diyagramı: sınıf ağırlıklı model olasılıkları şişiriyor", x=0.02, y=1.03, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.03, "Lojistik regresyon, tam öznitelik seti, test 2017–2024. Kesikli çizgi: mükemmel kalibrasyon; 10 eşit dolulukta dilim.",
             fontsize=8, color=INK2)
    save(fig, "fig_calibration")


def lin_shap(cfg, F, feats, t, log):
    D = F[F[t].notna()]; tr, va, te = split(D, cfg); trva = pd.concat([tr, va])
    sc = StandardScaler().fit(trva[feats])
    lr = LogisticRegression(C=lr_C(t, "full"), max_iter=3000).fit(sc.transform(trva[feats]), trva[t])
    Z = sc.transform(te[feats])
    S = pd.DataFrame(Z * lr.coef_[0], columns=feats, index=te.index)   # log-odds katkısı
    return S, te


def family_table(S):
    fam = S.T.groupby(family).sum().T                  # aile içi işaretli toplam
    imp = fam.abs().mean().sort_values(ascending=False)
    return fam, imp


def bar_panel(ax, imp, title, n=12):
    top = imp.head(n)[::-1]
    colors = [ACCENT if f.startswith("pl_") else MUTED for f in top.index]
    ax.barh([FAMILY_TR.get(f, f) for f in top.index], top.values, color=colors, height=0.65)
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=INK2)
    style(ax); ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Ortalama |SHAP| (log-odds)", color=INK2, fontsize=9)


def shap_section(cfg, F, feats, log, out):
    res = {}
    for t in ["y3", "y5", "y7"]:
        S, te = lin_shap(cfg, F, feats, t, log)
        fam, imp = family_table(S)
        imp.rename("mean_abs_shap").to_frame().assign(large_scale=lambda x: x.index.str.startswith("pl_")).to_csv(out / f"shap_lr_family_{t}.csv")
        S.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_csv(out / f"shap_lr_feature_{t}.csv")
        share = imp[imp.index.str.startswith("pl_")].sum() / imp.sum()
        log.info("%s LR SHAP: ilk 5 aile %s | büyük ölçekli payı %.0f%%", t, imp.head(5).round(3).to_dict(), 100 * share)
        res[t] = (fam, imp, te, share)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, t in zip(axes, ["y3", "y7"]):
        bar_panel(ax, res[t][1], f"{ {'y3': '3', 'y7': '7'}[t]} gün önceden · büyük ölçekli payı %{100 * res[t][3]:.0f}")
    fig.suptitle("Orta Anadolu Z500 her iki ufukta en güçlü büyük ölçekli öncül (mavi)", x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.03, "Doğrusal SHAP, ağırlıksız lojistik regresyon (tam set), test 2017–2024. Gecikme ve ortalama türevleri aynı aile altında toplandı. Gri: yerel yüzey değişkenleri.\nBirbirine bağlı aileler (ör. Tmax, Tmean, sıcaklık fazlası) etkiyi paylaşır; grup bazında sağlam karşılaştırma için fig_group_importance'a bakın.",
             fontsize=8, color=INK2)
    fig.tight_layout(); save(fig, "fig_shap_lr")

    # Bağımlılık: Orta Anadolu Z500 anomalisi -> SHAP (7 gün)
    fam, imp, te, _ = res["y7"]
    if "pl_z500_anat" in fam:
        x = te["pl_z500_anat_an"].to_numpy(); s = fam["pl_z500_anat"].to_numpy()
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(x, s, s=4, color=MUTED, alpha=0.35, lw=0)
        bins = np.quantile(x, np.linspace(0, 1, 16)); mid = 0.5 * (bins[1:] + bins[:-1])
        mean = [s[(x >= a) & (x < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(mid, mean, color=ACCENT, lw=2.2)
        ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8, ls=":")
        style(ax); ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xlabel("Orta Anadolu Z500 anomalisi, bugün (gpm)", color=INK2, fontsize=9)
        ax.set_ylabel("SHAP katkısı (log-odds)", color=INK2, fontsize=9)
        ax.set_title("Kuzeybatıdaki sırt güçlendikçe 7 günlük risk artıyor", loc="left", fontsize=11, color=INK)
        fig.text(0.02, -0.04, "Aile toplamı (anomali, gecikmeleri, ortalamaları). Mavi çizgi: 15 dilimde ortalama. Test 2017–2024.", fontsize=8, color=INK2)
        save(fig, "fig_shap_z500_dependence")

    try:
        import lightgbm  # noqa: F401
        lgbm_shap(cfg, F, feats, log, out)
    except ImportError:
        log.info("LightGBM kurulu değil: ağaç SHAP'ı hesaplanmadı")
    plot_lgbm(out, log)


def lgbm_shap(cfg, F, feats, log, out):
    from lightgbm import LGBMClassifier
    m = pd.read_csv(ROOT / "results" / "metrics_test.csv")
    for t in ["y3", "y5", "y7"]:
        D = F[F[t].notna()]; tr, va, te = split(D, cfg); trva = pd.concat([tr, va])
        r = m[(m.target == t) & (m.model == "lgbm") & (m.feature_set == "full")]
        prm = json.loads(r.params.iloc[0]) if len(r) else {}
        g = LGBMClassifier(n_estimators=600, learning_rate=prm.get("learning_rate", 0.03), num_leaves=prm.get("num_leaves", 15),
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, is_unbalance=True, random_state=42, verbose=-1)
        g.fit(trva[feats], trva[t])
        S = pd.DataFrame(g.predict(te[feats], pred_contrib=True)[:, :-1], columns=feats, index=te.index)
        fam, imp = family_table(S)
        imp.rename("mean_abs_shap").to_frame().assign(large_scale=lambda x: x.index.str.startswith("pl_")).to_csv(out / f"shap_lgbm_family_{t}.csv")
        S.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_csv(out / f"shap_lgbm_feature_{t}.csv")
        share = imp[imp.index.str.startswith("pl_")].sum() / imp.sum()
        log.info("%s LightGBM SHAP: ilk 5 aile %s | büyük ölçekli payı %.0f%%", t, imp.head(5).round(3).to_dict(), 100 * share)


def plot_lgbm(out, log):
    files = {t: out / f"shap_lgbm_family_{t}.csv" for t in ("y3", "y7")}
    if not HAVE_MPL or not all(f.exists() for f in files.values()):
        if not HAVE_MPL:
            log.info("matplotlib yok: LightGBM SHAP tabloları yazıldı, şekil çizilmedi")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, t in zip(axes, ["y3", "y7"]):
        imp = pd.read_csv(files[t], index_col=0)["mean_abs_shap"]
        share = imp[imp.index.str.startswith("pl_")].sum() / imp.sum()
        bar_panel(ax, imp, f"{ {'y3': '3', 'y7': '7'}[t]} gün önceden · büyük ölçekli payı %{100 * share:.0f}")
    fig.suptitle("LightGBM de doğruluyor: 7 günde en güçlü öncül Orta Anadolu Z500 (mavi: büyük ölçekli)", x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.03, "TreeSHAP (pred_contrib), LightGBM tam set, test 2017–2024. Gecikme ve ortalama türevleri aynı aile altında toplandı.", fontsize=8, color=INK2)
    fig.tight_layout(); save(fig, "fig_shap_lgbm")


GROUPS = [
    ("Sıcaklık durumu (yerel)", {"excess", "exceed", "run_len", "exceed_cnt3", "exceed_cnt7", "tmax", "tmin", "tmean", "dtr", "skt_max", "days_since_hw"}),
    ("Nem ve kuruluk (yerel)", {"tdew_mean", "rh_mean", "rh_min", "vpd_max", "tp_sum", "days_since_rain", "swvl1", "swvl2"}),
    ("Yüzey basıncı, rüzgâr, radyasyon (yerel)", {"sp_mean", "ws_mean", "ws_max", "ssrd_sum"}),
    ("Mevsim ve konum", {"doy", "lat", "lon"}),
    ("Z500 (büyük ölçekli)", None),
    ("T500/T850 ve kalınlık (büyük ölçekli)", None),
]


def group_of(c):
    f = family(c)
    if f.startswith("pl_t"):                       # pl_t500, pl_t850 ve pl_thick (kalınlık = katman ort. sıcaklığı)
        return "T500/T850 ve kalınlık (büyük ölçekli)"
    if f.startswith("pl_"):
        return "Z500 (büyük ölçekli)"
    for g, members in GROUPS:
        if members and f in members:
            return g
    return "Diğer"


def groups_section(cfg, F, feats, log, out, n_rep=10):
    rng = np.random.default_rng(42)
    rows = []
    for t in ["y3", "y5", "y7"]:
        D = F[F[t].notna()]; tr, va, te = split(D, cfg); trva = pd.concat([tr, va])
        m = make_pipeline(StandardScaler(), LogisticRegression(C=lr_C(t, "full"), max_iter=3000)).fit(trva[feats], trva[t])
        X = te[feats].copy(); y = te[t].to_numpy()
        base = average_precision_score(y, m.predict_proba(X)[:, 1])
        gmap = {}
        for c in feats:
            gmap.setdefault(group_of(c), []).append(c)
        for g, cols in gmap.items():
            drops = []
            for _ in range(n_rep):
                Xp = X.copy(); idx = rng.permutation(len(X))
                Xp[cols] = X[cols].to_numpy()[idx]
                drops.append(base - average_precision_score(y, m.predict_proba(Xp)[:, 1]))
            rows.append({"target": t, "group": g, "n_features": len(cols), "pr_auc_base": base,
                         "drop_mean": float(np.mean(drops)), "drop_std": float(np.std(drops))})
            log.info("%s %-42s (%3d öz.) PR-AUC düşüşü %.3f ± %.3f", t, g, len(cols), np.mean(drops), np.std(drops))
    G = pd.DataFrame(rows); G.round(4).to_csv(out / "group_importance.csv", index=False)

    order = G[G.target == "y7"].sort_values("drop_mean", ascending=False).group.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    yy = np.arange(len(order)); h = 0.26
    shades = {"y3": 0.35, "y5": 0.65, "y7": 1.0}
    for k, t in enumerate(["y3", "y5", "y7"]):
        x = G[G.target == t].set_index("group").loc[order]
        ax.barh(yy + (k - 1) * h, x.drop_mean, height=h - 0.03, color=ACCENT, alpha=shades[t],
                xerr=x.drop_std, error_kw={"ecolor": INK2, "lw": 0.8}, label={"y3": "3 gün", "y5": "5 gün", "y7": "7 gün"}[t])
    ax.set_yticks(yy, order); style(ax)
    ax.set_xlabel("PR-AUC düşüşü (grup karıştırıldığında)", color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="Tahmin ufku", title_fontsize=8.5)
    fig.suptitle("Ufuk uzadıkça bugünkü sıcaklığın önemi azalıyor, kuruluğun önemi artıyor", x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.04, f"Ağırlıksız lojistik regresyon (tam set), test 2017–2024; {n_rep} tekrar, hata çubuğu: standart sapma.", fontsize=8, color=INK2)
    fig.tight_layout(); save(fig, "fig_group_importance")


def regional(cfg, log, out):
    P = pd.read_csv(ROOT / "results" / "predictions_test.csv")
    rows = []
    for (t, pid), d in P.groupby(["target", "point_id"]):
        if t not in ("y3", "y5", "y7"):
            continue
        r = {"target": t, "point_id": pid, "il": NAMES.get(pid, pid), "n": len(d), "pos_rate": d.y_true.mean()}
        for c in ("persistence", "logreg_local", "logreg_full", "lgbm_full"):
            if c in d and d[c].notna().all():
                r[c] = average_precision_score(d.y_true, d[c])
        rows.append(r)
    R = pd.DataFrame(rows)
    R["gain_full_vs_local"] = R.logreg_full - R.logreg_local
    R["gain_full_vs_persistence"] = R.logreg_full - R.persistence
    R.round(4).to_csv(out / "regional.csv", index=False)
    for t in ("y3", "y7"):
        x = R[R.target == t].sort_values("logreg_full")
        log.info("%s il bazında LR tam PR-AUC: %s", t, dict(zip(x.il, x.logreg_full.round(3))))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    lon = {p["id"]: p["lon"] for p in cfg["points"]}
    order = sorted(R.point_id.unique(), key=lambda p: lon[p])          # batıdan doğuya
    for ax, t in zip(axes, ["y3", "y7"]):
        x = R[R.target == t].set_index("point_id").loc[order]
        yy = np.arange(len(order))
        ax.hlines(yy, x.logreg_local, x.logreg_full, color=ACCENT, alpha=0.35, lw=2)
        ax.scatter(x.persistence, yy, marker="|", s=120, color=INK2, label="Persistence", zorder=3)
        ax.scatter(x.logreg_local, yy, s=36, color=MUTED, label="LR, yalnız yerel", zorder=3)
        ax.scatter(x.logreg_full, yy, s=40, color=ACCENT, label="LR, yerel + büyük ölçekli", zorder=4)
        ax.set_yticks(yy, [NAMES[p] for p in order]); style(ax)
        ax.set_title({"y3": "3 gün önceden", "y7": "7 gün önceden"}[t], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("PR-AUC (test 2017–2024)", color=INK2, fontsize=9)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0.02, 0.95))
    fig.suptitle("Büyük ölçekli öncüller dokuz ilin hepsinde tahmini iyileştiriyor", x=0.02, y=1.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, -0.03, "İller batıdan doğuya sıralı. Her il yaklaşık 1.070 test günü; il bazında değerler tek tek belirsizdir, genel karşılaştırma için havuzlanmış sonuçlara bakın.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig_regional")
    n_up = {t: int((R[R.target == t].gain_full_vs_local > 0).sum()) for t in ("y3", "y5", "y7")}
    log.info("Büyük ölçekli verinin LR'yi iyileştirdiği il sayısı (9 üzerinden): %s", n_up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["calibration", "groups", "shap", "regional", "lgbm"])
    args = ap.parse_args()
    cfg = load_config(); log = get_logger("06_analysis", cfg)
    out = ROOT / "results" / "analysis"; out.mkdir(parents=True, exist_ok=True)
    if args.only == "lgbm":
        F, feats = load(cfg)
        lgbm_shap(cfg, F, feats, log, out); plot_lgbm(out, log)
        log.info("Bitti"); return
    if not HAVE_MPL:
        log.warning("matplotlib kurulu değil: şekiller çizilmeyecek")
    if args.only in (None, "regional"):
        regional(cfg, log, out)
    if args.only in (None, "calibration", "groups", "shap"):
        F, feats = load(cfg)
        if args.only in (None, "calibration"):
            calibration(cfg, F, feats, log, out)
        if args.only in (None, "groups"):
            groups_section(cfg, F, feats, log, out)
        if args.only in (None, "shap"):
            shap_section(cfg, F, feats, log, out)
    log.info("Bitti: results/analysis ve figures klasörleri")


if __name__ == "__main__":
    main()
