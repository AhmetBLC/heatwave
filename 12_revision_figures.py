"""Aşama 12 - Revizyon şekilleri (11_revision.py çıktılarından).

  fig_group_importance : Grup bazında permütasyon önemi; göreli PR-AUC düşüşü ve test yıllarının
                         blok bootstrap'ı ile %95 güven aralığı (kuruluk grubu üçe ayrılmış).
  fig_robustness       : "Tam − yerel" PR-AUC farkı (LR) ve %95 güven aralığı; ana kurgu ve tüm sağlamlık testleri.

Kullanım:
    python 12_revision_figures.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from src.common import ROOT

INK, INK2, GRID, MUTED, ACCENT, ACCENT2 = "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29b", "#2a78d6", "#eb6834"
R = ROOT / "results" / "revision"
plt.rcParams.update({"font.size": 9})
TR = lambda v, d=2: f"{v:.{d}f}".replace(".", ",")


def style(ax, axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, name):
    d = ROOT / "figures"; d.mkdir(exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def group_importance():
    G = pd.read_csv(R / "group_importance_boot.csv")
    G["group"] = G.group.replace({"Nem içeriği (çiy noktası, özgül nem)": "Nem içeriği (çiy noktası)"})   # ana modelde özgül nem yok
    order = G[G.target == "y3"].sort_values("rel_drop", ascending=False).group.tolist()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    yy = np.arange(len(order)); h = 0.26
    shades = {"y3": 0.35, "y5": 0.65, "y7": 1.0}
    for k, t in enumerate(["y3", "y5", "y7"]):
        x = G[G.target == t].set_index("group").loc[order]
        ax.barh(yy + (k - 1) * h, 100 * x.rel_drop, height=h - 0.03, color=ACCENT, alpha=shades[t],
                xerr=[100 * (x.rel_drop - x.rel_ci_lo), 100 * (x.rel_ci_hi - x.rel_drop)],
                error_kw={"ecolor": INK2, "lw": 0.8, "capsize": 1.5}, label={"y3": "3 gün", "y5": "5 gün", "y7": "7 gün"}[t])
    ax.set_yticks(yy, order); style(ax); ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"%{v:.0f}"))
    ax.set_xlabel("Grup karıştırıldığında PR-AUC'deki göreli düşüş", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="Tahmin ufku", title_fontsize=8.5)
    save(fig, "fig_group_importance")


def robustness():
    sig = pd.read_csv(R / "significance_10k.csv")
    det = pd.read_csv(R / "detrend_pl.csv"); tv = pd.read_csv(R / "trend_variables.csv")
    ew = pd.read_csv(R / "early_warning_subset.csv"); sens = pd.read_csv(R / "sensitivity_10k.csv")
    box = pd.read_csv(R / "nonoverlap_box.csv") if (R / "nonoverlap_box.csv").exists() else None
    rows = []
    for t in ("y3", "y7"):
        m = sig[(sig.target == t) & (sig.model_a == "logreg_full") & (sig.model_b == "logreg_local")].iloc[0]
        rows.append((t, "Ana kurgu", m["diff"], m.ci_lo, m.ci_hi, "ana"))
        for mode, lab in (("train", "Z500/T850 eğilimi çıkarılmış (eğitim dönemi eğilimi)"),
                          ("trailing5", "Z500/T850 eğilimi çıkarılmış (önceki 5 sezon)")):
            r = det[(det["mode"] == mode) & (det.target == t)].iloc[0]
            rows.append((t, lab, r.full_minus_local, r.fl_ci_lo, r.fl_ci_hi, "egilim"))
        r = tv[tv.target == t].iloc[0]
        rows.append((t, "Yerel sete yıl ve geçmiş bölgesel sıcaklık eklenmiş", r.full_minus_local, r.fl_ci_lo, r.fl_ci_hi, "egilim"))
        if box is not None:
            r = box[box.target == t].iloc[0]
            rows.append((t, "Çakışmasız Orta Anadolu kutusu", r.full_minus_local, r.ci_lo, r.ci_hi, "diger"))
        r = ew[ew.target == t].iloc[0]
        rows.append((t, "Yalnız t gününde aşım olmayan örnekler", r["logreg_full-logreg_local"],
                     r["logreg_full-logreg_local_lo"], r["logreg_full-logreg_local_hi"], "diger"))
        for v, lab in (("split_alt", "Alternatif bölünme (test 2011–2016)"), ("p95_L3", "Eşik: 95. yüzdelik"),
                       ("p90_L2", "Süre: en az 2 gün"), ("p90_L4", "Süre: en az 4 gün")):
            r = sens[(sens.variant == v) & (sens.target == t)]
            if len(r):
                r = r.iloc[0]
                rows.append((t, lab, r["logreg_full-logreg_local"], r["logreg_full-logreg_local_lo"],
                             r["logreg_full-logreg_local_hi"], "tanim"))
    D = pd.DataFrame(rows, columns=["t", "label", "d", "lo", "hi", "kind"])
    labels = list(dict.fromkeys(D.label))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(labels) + 1.2), sharey=True)
    col = {"ana": INK, "egilim": ACCENT, "diger": ACCENT, "tanim": INK2}
    for ax, t in zip(axes, ("y3", "y7")):
        d = D[D.t == t].set_index("label").reindex(labels)
        yy = np.arange(len(labels))
        for i, (lab, r) in enumerate(d.iterrows()):
            if np.isnan(r.d):
                continue
            ax.hlines(i, r.lo, r.hi, color=col[r.kind], lw=1.6)
            ax.plot(r.d, i, "o", ms=6 if r.kind == "ana" else 5, color=col[r.kind], mfc=col[r.kind] if r.kind != "tanim" else "white")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvline(d.loc["Ana kurgu", "d"], color=INK, lw=0.6, ls=":")
        ax.set_yticks(yy, labels); style(ax)
        ax.set_title({"y3": "3 gün önceden", "y7": "7 gün önceden"}[t], loc="left", fontsize=10.5, color=INK)
        ax.xaxis.set_major_locator(MultipleLocator(0.02)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: TR(v)))
        ax.set_xlabel("PR-AUC farkı: tam − yerel (LR)", color=INK2, fontsize=9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "fig_robustness")


if __name__ == "__main__":
    group_importance()
    robustness()
    print("bitti")
