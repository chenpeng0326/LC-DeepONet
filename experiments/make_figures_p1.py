"""P1 补图：fig10 Accuracy–Certificate trade-off、fig11 long-horizon RMSE(t)、
fig12 margin vs tracking RMSE（E6-A，审稿意见：理论预测 vs 实际性能散点图）。
用法: python make_figures_p1.py [fig10|fig11|fig12|all]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT
sys.path.insert(0, os.path.join(ROOT, "src"))
import config

FIG = config.FIGURES_DIR
RES = config.RESULTS_DIR
plt.rcParams.update({"font.size": 10, "figure.dpi": 300})


def savefig(fig, name, dpi=300):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


# ---------- Fig.10 Accuracy–Certificate trade-off (E7b) ----------
def fig10():
    df = pd.read_csv(os.path.join(RES, "e7b_ltarget_sensitivity.csv"))
    df = df.sort_values("L_target")
    x, xs = df["L_cert_mean"].to_numpy(), df["L_cert_std"].to_numpy()
    y, ys = df["ID_mean"].to_numpy(), df["ID_std"].to_numpy()
    lt = df["L_target"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(x, y, "-", color="#b8b8b8", lw=1.2, zorder=1)
    ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="o", color="tab:red",
                ecolor="tab:red", elinewidth=1.2, capsize=3, ms=6, zorder=3,
                label="ID RMSE (mean $\\pm$ std, 5 seeds)")
    offsets = {1.5: (2, -16), 2.0: (10, 5), 2.5: (8, -15),
               3.0: (10, 5), 4.0: (6, 10), 5.0: (-2, -17)}
    for xi, yi, l in zip(x, y, lt):
        dx, dy = offsets.get(float(l), (8, 8))
        ax.annotate(f"$L_{{target}}$={l:g}", (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color="#444444",
                    ha="center" if dx == 0 or dx == 2 or dx == -2 else "left")
    ax.set_xlabel(r"certificate $L_{\mathrm{cert}}$")
    ax.set_ylabel("identification RMSE (ID test set)")
    ax.set_title("Accuracy--certificate trade-off over $L_{target}$")
    ax.legend(fontsize=8, loc="upper left")
    savefig(fig, "fig10_tradeoff.png")


# ---------- Fig.11 Long-horizon rollout RMSE(t) (E9) ----------
def fig11():
    df = pd.read_csv(os.path.join(RES, "e9_rmse_t.csv"))
    style = {
        "LC-DeepONet": dict(color="tab:red", lw=2.0, zorder=5),
        "DeepONet":    dict(color="tab:blue", lw=1.5, ls="--", zorder=4),
        "FNO":         dict(color="tab:gray", lw=1.3, ls=":", zorder=3),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=False)
    for ax, t_end in zip(axes, sorted(df["T_test"].unique())):
        sub = df[df["T_test"] == t_end]
        for name, g in sub.groupby("model"):
            g = g.sort_values("t")
            st = style.get(name, {})
            ax.plot(g["t"], g["rmse_t_mean"], label=name, **st)
        ax.axvline(2.0, color="#999999", lw=1.0, ls="--")
        ax.text(2.05, ax.get_ylim()[1] * 0.92, "training horizon",
                fontsize=7.5, color="#666666", rotation=90, va="top")
        ax.set_title(f"$T_{{test}}$ = {t_end:g} s")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("RMSE(t)")
        ax.legend(fontsize=8)
    savefig(fig, "fig11_longhorizon.png")


# ---------- Fig.12 margin vs tracking RMSE (E6-A, delay-free) ----------
def fig12():
    df = pd.read_csv(os.path.join(RES, "e6a_delayfree.csv"))
    m = df["M_margin"].to_numpy()
    e = df["track_err_L2"].to_numpy()
    mult = df["mult"].to_numpy()
    order = np.argsort(-m)  # M>0 -> M<0
    m, e, mult = m[order], e[order], mult[order]

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    xmin = min(m.min() * 1.3, -130)
    ax.axvspan(xmin, 0, color="0.93", zorder=0)
    pos, neg = m > 0, m <= 0
    ax.plot(m[pos], e[pos], "o-", color="tab:blue", lw=1.4, ms=6,
            zorder=3, label="$M>0$: guarantee holds")
    ax.plot(m[neg], e[neg], "s-", color="tab:red", lw=1.4, ms=5.5,
            zorder=3, label="$M<0$: guarantee lost")
    ax.axvline(0.0, color="k", ls="--", lw=1.1, zorder=2)
    ax.text(0.012, 0.965, "$M=0$", transform=ax.get_yaxis_transform(),
            fontsize=8, va="top", color="#333333")
    # 标注关键 sweep 倍率
    for target in (0.25, 1.0, 16.0, 128.0):
        idx = np.where(np.isclose(mult, target))[0]
        if len(idx):
            i = idx[0]
            off = {0.25: (-26, 6), 1.0: (-10, 6), 16.0: (6, 2), 128.0: (6, 4)}.get(
                target, (6, 4))
            ax.annotate(f"$ {target:g}\\times$", (m[i], e[i]),
                        textcoords="offset points", xytext=off,
                        fontsize=7.5, color="#444444")
    ax.set_xscale("symlog", linthresh=0.2, linscale=0.8)
    ax.set_xlim(xmin, 1.6)
    ax.set_xlabel(r"small-gain margin $M = 1-k_c\,(L_{\mathrm{cert}}+\hat\varepsilon_L)$"
                  "  (symlog)")
    ax.set_ylabel("tracking error (RMSE)")
    ax.set_title("Theoretical margin vs. closed-loop performance (E6-A, delay-free)")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "fig12_margin_rmse.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"fig10": fig10, "fig11": fig11, "fig12": fig12}
    if which == "all":
        for fn in jobs.values():
            fn()
    else:
        jobs[which]()
