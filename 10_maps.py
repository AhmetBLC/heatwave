"""Aşama 10 - Harita şekilleri (cartopy gerektirmez; kıyı çizgisi ERA5 kara-deniz maskesinden çizilir).

  fig_study_area : Çalışma alanı - yükselti, 9 il hücresi, büyük ölçekli bölge kutuları
  fig_composites : Sıcak hava dalgası başlangıcından 7, 5, 3, 1 gün önce ve başlangıç günü
                   kompozit Z500 anomalisi (renk) ve T850 anomalisi (kontur); p<0,05 noktalı

Girdi : results/analysis/basemap.csv, composites.csv, composites_meta.csv (08 çıktıları)
Çıktı : figures/fig_study_area.*, figures/fig_composites.*
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from src.common import ROOT, load_config

INK, INK2 = "#0b0b0b", "#52514e"
NAMES_SHORT = {"gaziantep": "Gaziantep", "adiyaman": "Adıyaman", "kilis": "Kilis", "sanliurfa": "Şanlıurfa",
               "diyarbakir": "Diyarbakır", "mardin": "Mardin", "batman": "Batman", "siirt": "Siirt", "sirnak": "Şırnak"}
REGION_TR = {"gda": "GDA", "anat": "Orta Anadolu", "emed": "Doğu Akdeniz", "iran": "İran", "north": "Karadeniz–Kafkasya", "anat_n": "Orta Anadolu (çakışmasız)"}


def grid(df, col):
    lats = np.sort(df.lat.unique())[::-1]; lons = np.sort(df.lon.unique())
    g = df.pivot(index="lat", columns="lon", values=col).loc[lats, lons]
    return lats, lons, g.to_numpy()


def coast(ax, base, color=None, lw=0.6):
    la, lo, m = grid(base, "lsm")
    ax.contour(lo, la, m, levels=[0.5], colors=color or INK, linewidths=lw)


def save(fig, name):
    d = ROOT / "figures"; d.mkdir(exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(d / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def study_area(cfg, base):
    la, lo, orog = grid(base, "orog_m"); _, _, lsm = grid(base, "lsm")
    orog = np.where(lsm > 0.5, orog, np.nan)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    im = ax.pcolormesh(lo, la, orog, cmap="Greys", vmin=-200, vmax=3500, shading="nearest")
    ax.set_facecolor("#eef3f7"); coast(ax, base)
    for name, (s, n, w, e) in cfg["pressure_levels"]["regions"].items():
        col = "#2a78d6" if name in ("anat", "anat_n") else "#52514e"
        ls = "-" if name in ("gda", "anat") else (":" if name == "anat_n" else "--")
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=col, lw=1.6 if name == "anat" else 1.1, ls=ls))
        if name == "anat_n":
            ax.text(w + 0.2, n + 0.15, REGION_TR.get(name, name), fontsize=7.5, color=col)
        else:
            ax.text(w + 0.2, n - 0.45, REGION_TR.get(name, name), fontsize=8, color=col, fontweight="bold" if name == "anat" else "normal")
    for p in cfg["points"]:
        ax.plot(p["lon"], p["lat"], "o", ms=5, mfc="#eb6834", mec="white", mew=0.8, zorder=5)
    ax.set_xlim(lo.min(), lo.max()); ax.set_ylim(la.min(), la.max()); ax.set_aspect(1 / np.cos(np.radians(37.5)))
    ax.set_xlabel("Boylam (°D)", fontsize=9, color=INK2); ax.set_ylabel("Enlem (°K)", fontsize=9, color=INK2)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.set_label("Yükselti (m)", fontsize=8)
    # yakın plan
    ins = ax.inset_axes([0.60, 0.04, 0.38, 0.30])
    ins.pcolormesh(lo, la, orog, cmap="Greys", vmin=-200, vmax=3500, shading="nearest"); coast(ins, base)
    for p in cfg["points"]:
        ins.plot(p["lon"], p["lat"], "o", ms=4, mfc="#eb6834", mec="white", mew=0.6)
        off = {"diyarbakir": (-34, 4), "batman": (2, -8), "siirt": (2, 4)}.get(p["id"], (2, 3))
        ins.annotate(NAMES_SHORT[p["id"]], (p["lon"], p["lat"]), xytext=off, textcoords="offset points", fontsize=6, color=INK)
    ins.set_xlim(36.5, 43.2); ins.set_ylim(36.3, 38.6); ins.set_xticks([]); ins.set_yticks([])
    for s_ in ins.spines.values(): s_.set_color(INK2)
    save(fig, "fig_study_area")


def fdr(p, alpha):
    """Benjamini-Hochberg: alan genelinde yanlış keşif oranını alpha'da tutan anlamlılık maskesi."""
    flat = p.ravel(); ok = ~np.isnan(flat)
    ps = np.sort(flat[ok]); n = len(ps)
    below = ps <= alpha * np.arange(1, n + 1) / n
    thr = ps[below].max() if below.any() else -1.0
    return np.where(np.isnan(p), False, p <= thr)


def composites(cfg, base):
    C = pd.read_csv(ROOT / "results" / "analysis" / "composites.csv")
    sfx = "_dt" if "z500_an_dt" in C else ""          # eğilimden arındırılmış sürüm varsa onu kullan
    pcol = "z500_pmc_dt" if "z500_pmc_dt" in C else f"z500_p{sfx}"   # yıl karıştırmalı Monte Carlo p'si varsa onu kullan
    C = C.assign(z500_an=C[f"z500_an{sfx}"], t850_an=C[f"t850_an{sfx}"], z500_p=C[pcol])
    meta = pd.read_csv(ROOT / "results" / "analysis" / "composites_meta.csv").set_index("lag").n_dates
    lags = sorted(C.lag.unique())
    vmax = np.nanpercentile(np.abs(C.z500_an), 99)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.07, wspace=0.12, hspace=0.22)
    flat = axes.ravel()
    for ax, L in zip(flat, lags):
        d = C[C.lag == L]
        la, lo, z = grid(d, "z500_an"); _, _, t = grid(d, "t850_an"); _, _, p = grid(d, "z500_p")
        im = ax.pcolormesh(lo, la, z, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), shading="nearest")
        cs = ax.contour(lo, la, t, levels=[-2, -1, 1, 2, 3, 4], colors="#111111", linewidths=0.9, linestyles="-")
        ax.clabel(cs, fontsize=6.5, fmt=lambda v: f"{v:+.0f}")
        sig = fdr(p, 0.10)                       # Benjamini-Hochberg, alfa_FDR = 0,10 (Wilks 2016)
        LO, LA = np.meshgrid(lo, la)
        thin = (np.arange(len(la))[:, None] % 4 == 0) & (np.arange(len(lo))[None, :] % 4 == 0)   # 1° aralıklı seyrek nokta
        m = sig & thin
        ax.scatter(LO[m], LA[m], s=2.5, c="#1d1d1b", alpha=0.65, lw=0)
        if base is not None:
            coast(ax, base, color="#8a8984", lw=0.5)
        s, n, w, e = cfg["pressure_levels"]["regions"]["gda"]
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec="#eb6834", lw=1.4))
        s, n, w, e = cfg["pressure_levels"]["regions"]["anat"]
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=INK, lw=0.9, ls="--"))
        ax.set_title(("Başlangıç günü (gün 0)" if L == 0 else f"{-L} gün önce") + f", n = {meta[L]}", fontsize=10, color=INK, loc="left")
        ax.set_aspect(1 / np.cos(np.radians(37.5))); ax.tick_params(labelsize=7.5)
    for ax in flat[len(lags):]:
        ax.axis("off")
    cax = flat[-1].inset_axes([0.08, 0.45, 0.84, 0.07])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Z500 anomalisi (gpm)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    save(fig, "fig_composites")


if __name__ == "__main__":
    cfg = load_config()
    bp = ROOT / "results" / "analysis" / "basemap.csv"
    base = pd.read_csv(bp) if bp.exists() else None
    if base is not None:
        study_area(cfg, base)
    if (ROOT / "results" / "analysis" / "composites.csv").exists():
        composites(cfg, base)
    print("bitti")
