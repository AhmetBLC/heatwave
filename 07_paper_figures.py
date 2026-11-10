"""Aşama 7 - Makale için özet şekiller.

  fig_trend   : Yaz aylarında başlayan sıcak hava dalgası sayısı (9 il ortalaması, yıllık) ve on yıllık ortalamalar
  fig_horizon : Tahmin ufkuna göre PR-AUC (persistence, LR yerel, LR tam, LightGBM tam), %95 güven aralıklarıyla

Girdi : data/processed/events.csv, results/metrics_test.csv
Çıktı : figures/fig_trend.*, figures/fig_horizon.*
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import ROOT, load_config

INK, INK2, GRID, MUTED, ACCENT = "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29b", "#2a78d6"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def save(fig, name):
    d = ROOT / "figures"; d.mkdir(exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(d / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def trend(cfg):
    ev = pd.read_csv(ROOT / "data" / "processed" / "events.csv", parse_dates=["start"])
    ev = ev[ev.start.dt.month.isin(cfg["heatwave"]["season_months"])]
    years = np.arange(cfg["period"]["start_year"], cfg["period"]["end_year"] + 1)
    n_pts = len(cfg["points"])
    per_year = ev.groupby(ev.start.dt.year).size().reindex(years, fill_value=0) / n_pts
    slope, icpt = np.polyfit(years, per_year.values, 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(years, per_year.values, color=MUTED, width=0.75)
    ax.plot(years, icpt + slope * years, color=ACCENT, lw=2)
    for d0 in range(1980, 2030, 10):
        m = (years >= max(d0, years[0])) & (years < d0 + 10)
        if m.any():
            v = per_year.values[m].mean()
            ax.hlines(v, years[m][0] - 0.4, years[m][-1] + 0.4, color=INK, lw=1.2)
            ax.text(years[m].mean(), v + 0.25, f"{v:.1f}".replace(".", ","), ha="center", fontsize=8, color=INK)
    style(ax); ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_ylabel("Olay / il / yıl", color=INK2, fontsize=9)
    ax.set_title(f"Yaz sıcak hava dalgaları on yılda {slope * 10:.1f} olay arttı".replace(".", ","), loc="left", fontsize=11, color=INK)
    fig.text(0.01, -0.04, "Gri: yıllık 9 il ortalaması (Mayıs–Eylül'de başlayan olaylar); siyah: on yıllık ortalama (2020–24: 5 yıl); mavi: doğrusal eğilim. ERA5-Land.",
             fontsize=8, color=INK2)
    save(fig, "fig_trend")
    return slope * 10


def horizon():
    m = pd.read_csv(ROOT / "results" / "metrics_test.csv")
    m["h"] = m.target.str[1:].astype(int)
    m = m[m.h.isin([3, 5, 7])]
    series = [("persistence", "-", "Persistence", INK2, "s", "--"),
              ("lgbm", "full", "LightGBM, tam set", MUTED, "^", ":"),
              ("logreg", "local", "LR, yalnız yerel", MUTED, "o", "-"),
              ("logreg", "full", "LR, yerel + büyük ölçekli", ACCENT, "o", "-")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    off = {0: -0.12, 1: -0.04, 2: 0.04, 3: 0.12}
    for i, (mdl, fs, lab, col, mk, ls) in enumerate(series):
        r = m[(m.model == mdl) & (m.feature_set == fs)].sort_values("h")
        x = r.h + off[i]
        ax.errorbar(x, r.pr_auc, yerr=[r.pr_auc - r.pr_auc_ci_lo, r.pr_auc_ci_hi - r.pr_auc], color=col, marker=mk, ms=6,
                    ls=ls, lw=1.8 if mdl == "logreg" and fs == "full" else 1.2, capsize=0, elinewidth=1, label=lab)
        last = r.iloc[-1]
        if mdl == "lgbm":
            continue
        ax.text(7.25, last.pr_auc, f"{last.pr_auc:.2f}".replace(".", ","), va="center", fontsize=8, color=col if col != MUTED else INK2)
    style(ax); ax.set_xticks([3, 5, 7], ["3 gün", "5 gün", "7 gün"]); ax.set_xlim(2.6, 7.6)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.set_ylabel("PR-AUC (test 2017–2024)", color=INK2, fontsize=9)
    ax.set_title("Büyük ölçekli öncüller her ufukta en iyi modeli daha da iyileştiriyor", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.text(0.01, -0.04, "Dikey çizgiler: %95 güven aralığı (test yıllarının blok bootstrap'ı). Ufuk uzadıkça pozitif oranı arttığından PR-AUC değerleri ufuklar arasında doğrudan karşılaştırılmamalıdır.",
             fontsize=7.5, color=INK2)
    save(fig, "fig_horizon")


if __name__ == "__main__":
    cfg = load_config()
    print("eğilim (olay/on yıl):", round(trend(cfg), 2)); horizon(); print("bitti")
