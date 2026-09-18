"""P0 补图：fig1 方法框架图、fig2 闭环框图、fig6 噪声鲁棒性。
用法: python make_figures_p0.py [fig1|fig2|fig6|all]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import pandas as pd

from common import ROOT
sys.path.insert(0, os.path.join(ROOT, "src"))
import config

FIG = config.FIGURES_DIR
plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

BLUE, GREEN, YELLOW, RED = "#EAF2FB", "#EAF7EE", "#FFF6DC", "#FDECEC"
EDGE, ARROW = "#4A6FA5", "#4A6FA5"


def savefig(fig, name, dpi=300):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def box(ax, x, y, w, h, text, fc=BLUE, ec=EDGE, fs=8.5, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, ls="-", color=ARROW, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 color=color, lw=lw, linestyle=ls,
                                 mutation_scale=13, shrinkA=3, shrinkB=3))


# ---------- Fig.1 方法框架 ----------
def fig1():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0.4, 4.6); ax.axis("off")

    # 上行：branch 路径
    box(ax, 0.1, 3.55, 1.7, 0.8, "input samples\n$[u(t_0),\\dots,u(t_{63})]$")
    box(ax, 2.15, 3.70, 0.95, 0.5, r"$\times\sqrt{\Delta t}$")
    box(ax, 3.45, 3.45, 2.75, 1.0,
        "Branch MLP\nspectral norm $\\sigma(W_i)\\leq s_i$\n+ Lipschitz penalty", fc=BLUE)
    box(ax, 6.55, 3.70, 1.15, 0.5, "$B(u)\\in\\mathbb{R}^p$")

    # 下行：trunk 路径
    box(ax, 0.1, 2.10, 1.7, 0.8, "query time\n$t_j\\in[0,T]$")
    box(ax, 3.45, 2.25, 2.75, 0.55, "Trunk MLP", fc=GREEN)
    box(ax, 6.55, 2.25, 1.15, 0.5, "$T(t_j)\\in\\mathbb{R}^p$")

    # 汇合输出
    box(ax, 8.15, 2.70, 2.6, 0.8,
        "$\\hat G(u)(t_j)=B(u)\\!\\cdot\\!T(t_j)+b_j$", fc=YELLOW)

    arrow(ax, 1.8, 3.95, 2.15, 3.95)
    arrow(ax, 3.1, 3.95, 3.45, 3.95)
    arrow(ax, 6.2, 3.95, 6.55, 3.95)
    arrow(ax, 7.7, 3.95, 8.9, 3.55)
    arrow(ax, 1.8, 2.50, 3.45, 2.50)
    arrow(ax, 6.2, 2.50, 6.55, 2.50)
    arrow(ax, 7.7, 2.50, 8.9, 2.68)

    # 下行：证书与控制链
    box(ax, 3.45, 0.65, 2.75, 0.85,
        "certificate (exact SVD, eval.)\n$L_{\\mathrm{cert}}=\\sigma(W_1)\\cdots\\sigma(W_L)\\,C_T$", fc=RED)
    box(ax, 6.55, 0.65, 2.0, 0.85,
        "Assumption 1\n$L_G\\leq L_{\\mathrm{cert}}+\\varepsilon_L$", fc=RED)
    box(ax, 8.85, 0.55, 1.95, 1.05,
        "small gain\n$L_C(L_{\\mathrm{cert}}+\\varepsilon_L)<1$\n$\\Rightarrow$ gain $k_c$", fc=YELLOW)

    arrow(ax, 4.8, 3.45, 4.8, 1.50, ls="--")   # branch -> certificate
    arrow(ax, 6.2, 1.07, 6.55, 1.07)
    arrow(ax, 8.55, 1.07, 8.85, 1.07)
    arrow(ax, 9.45, 2.70, 9.8, 1.62, ls="--")  # output -> small gain (uses L-hat chain)

    ax.text(4.8, 3.02, r"$\sigma(W_i)$ from weights", ha="center",
            fontsize=7.5, color="#666666")
    savefig(fig, "fig1_framework.png")


# ---------- Fig.2 闭环框图 ----------
def fig2():
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 3.0); ax.axis("off")
    ymid = 1.85

    # 求和点1
    ax.add_patch(Circle((1.55, ymid), 0.17, fc="white", ec=EDGE, lw=1.2))
    ax.text(1.55, ymid, "$-$", ha="center", va="center", fontsize=10)

    box(ax, 2.35, ymid - 0.4, 1.75, 0.8, "controller\n$u=k_c\\,e$")
    box(ax, 4.75, ymid - 0.4, 1.45, 0.8, "computation\ndelay $\\tau$")
    ax.add_patch(Circle((6.85, ymid), 0.17, fc="white", ec=EDGE, lw=1.2))
    ax.text(6.85, ymid, "$+$", ha="center", va="center", fontsize=10)
    box(ax, 7.6, ymid - 0.4, 2.6, 0.8,
        "plant $G$: MSD\n$(c,k,\\alpha)$ uncertain", fc=GREEN)

    arrow(ax, 0.35, ymid, 1.36, ymid)
    ax.text(0.55, ymid + 0.16, "$r(t)$", fontsize=9)
    arrow(ax, 1.72, ymid, 2.35, ymid)
    arrow(ax, 4.1, ymid, 4.75, ymid)
    arrow(ax, 6.2, ymid, 6.66, ymid)
    arrow(ax, 7.02, ymid, 7.6, ymid)
    arrow(ax, 10.2, ymid, 11.15, ymid)
    ax.text(10.45, ymid + 0.16, "$y(t)$", fontsize=9)
    ax.text(7.05, ymid + 0.38, "$d(t)$", fontsize=9)
    arrow(ax, 6.85, ymid + 0.75, 6.85, ymid + 0.19)

    # 反馈回路
    ax.plot([10.9, 10.9], [ymid, 0.55], color=ARROW, lw=1.4)
    ax.plot([10.9, 1.55], [0.55, 0.55], color=ARROW, lw=1.4)
    arrow(ax, 1.55, 0.55, 1.55, ymid - 0.19)
    ax.text(6.2, 0.68, "output feedback $y(t)$", fontsize=8, color="#666666")

    # 误差与裕度标注
    ax.text(1.95, 2.42, "$e=r-y$", fontsize=8, color="#666666")
    ax.text(6.2, 0.24, r"margin $M=1-k_c(L_{\mathrm{cert}}+\varepsilon_L)$", fontsize=8,
            color="#666666", ha="center")
    savefig(fig, "fig2_loop.png")


# ---------- Fig.6 噪声鲁棒性 ----------
def fig6():
    """E4b：6 模型 x 5 seeds，mean±std 误差带。"""
    df = pd.read_csv(
        os.path.join(config.RESULTS_DIR, "e4b_noise_5seed_per_seed.csv"))
    levels = [1, 3, 5, 10]
    x = np.array(levels)
    cols = [f"RMSE_noise_{lv}%" for lv in levels]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    style = {
        "LC-DeepONet": dict(color="tab:red", lw=2.2, marker="o", zorder=5),
        "DeepONet":    dict(color="tab:blue", lw=1.6, marker="s", zorder=4),
        "FNO":         dict(color="tab:gray", lw=1.3, ls="--", marker="^", zorder=3),
        "MLP":         dict(color="tab:green", lw=1.0, marker="v", alpha=0.7),
        "LSTM":        dict(color="tab:purple", lw=1.0, marker="d", alpha=0.7),
        "Transformer": dict(color="tab:orange", lw=1.0, marker="x", alpha=0.7),
    }
    for model, g in df.groupby("model"):
        arr = g[cols].to_numpy()
        m, s = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        st = style.get(model, {})
        ax.plot(x, m, label=model, **st)
        ax.fill_between(x, m - s, m + s, color=st.get("color"),
                        alpha=0.15, lw=0, zorder=st.get("zorder", 1))
    ax.set_xlabel(r"input noise level (% of $\|u\|_\infty$)")
    ax.set_ylabel("RMSE (ID test set)")
    ax.set_xticks(x)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Degradation under test-time input noise "
                 "(mean $\\pm$ std over 5 seeds)")
    savefig(fig, "fig6_noise.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"fig1": fig1, "fig2": fig2, "fig6": fig6}
    if which == "all":
        for fn in jobs.values():
            fn()
    else:
        jobs[which]()
