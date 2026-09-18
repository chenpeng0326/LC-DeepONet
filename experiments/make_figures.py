"""根据 results/*.csv 生成论文核心图（Fig.3/4/5/7/8/9 对应素材）。
用法: python make_figures.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import ROOT
sys.path.insert(0, os.path.join(ROOT, "src"))

import config
import models as M
import train as T
import control as C
from dataset import get_loaders

config.set_seed(config.SEED)
FIG = config.FIGURES_DIR
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def savefig(fig, name, dpi=300):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def load_csv(name):
    return pd.read_csv(os.path.join(config.RESULTS_DIR, name))


# ---------- Fig.3 训练损失与 Lipschitz bound ----------
def fig3():
    tr, te, ood = get_loaders()
    res_lc = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=40, use_lip=True)
    res_do = T.full_pipeline("DeepONet", tr, te, ood, epochs=40, use_lip=False)
    h1, h2 = pd.DataFrame(res_lc["history"]), pd.DataFrame(res_do["history"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(h1["epoch"], h1["loss"], label="LC-DeepONet")
    ax[0].plot(h2["epoch"], h2["loss"], label="DeepONet")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss")
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("(a) Training loss")
    if "L_hat" in h1:
        ax[1].plot(h1["epoch"], h1["L_hat"], color="tab:red")
        ax[1].axhline(config.LIP_TARGET, ls="--", c="gray", label=f"L_target={config.LIP_TARGET}")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel(r"$\tilde L$ (differentiable surrogate)")
        ax[1].legend(); ax[1].set_title("(b) Lipschitz bound during training")
    savefig(fig, "fig3_training_lipschitz.png")


# ---------- Fig.4 辨识结果 ----------
def fig4():
    tr, te, ood = get_loaders()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, name in zip(axes, ["DeepONet", "LC-DeepONet", "LSTM"]):
        res = T.full_pipeline(name, tr, te, ood, epochs=40,
                              use_lip=(name == "LC-DeepONet"))
        u, q, y = T.predict(res["model_obj"], te)
        j = int(torch.argmax((y - q).pow(2).mean(dim=1)))  # 最差样本（诚实展示）
        tj = np.linspace(0, config.T_END, config.N_POINTS)
        ax.plot(tj, q[j], "k-", lw=2, label="true q(t)")
        ax.plot(tj, y[j], "--", color="tab:red", label=f"{name}")
        ax.set_title(f"{name} (worst-case test sample)")
        ax.set_xlabel("t [s]"); ax.set_ylabel("q(t)")
        ax.legend(fontsize=8)
    savefig(fig, "fig4_identification.png")


# ---------- Fig.5 OOD ----------
def fig5():
    df = load_csv("e1_identification.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["ID_RMSE_mean"], 0.4, yerr=df["ID_RMSE_std"], label="ID (alpha 0.5-1.2)")
    ax.bar(x + 0.2, df["OOD_RMSE_mean"], 0.4, yerr=df["OOD_RMSE_std"], label="OOD (alpha 1.2-1.5)")
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=20)
    ax.set_ylabel("RMSE"); ax.legend()
    ax.set_title("Identification vs parameter-OOD performance (mean over seeds)")
    savefig(fig, "fig5_ood.png")


# ---------- Fig.7 敏感性 vs 证书 ----------
def fig7():
    df = load_csv("e3b_sensitivity_5seed.csv")   # 5-seed 聚合
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(df))
    colors = ["tab:red" if m == "LC-DeepONet" else "tab:gray" for m in df["model"]]
    ax.bar(x, df["S_emp_mean"], 0.55, yerr=df["S_emp_std"].fillna(0),
           capsize=3, color=colors, error_kw=dict(lw=1.1),
           label=r"empirical sensitivity $S_{emp}$ (mean$\pm$std, 5 seeds)")
    if "L_cert_mean" in df:
        m = df["L_cert_mean"]
        ax.scatter(x[m.notna()], m.dropna(), marker="D", s=70,
                   color="tab:blue", zorder=3,
                   label=r"certified bound $L_{\mathrm{cert}}$")
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=20)
    ax.set_ylabel(r"Lipschitz gain (L$^2$)")
    ax.set_ylim(0, 5.2)
    ax.set_title(r"Empirical sensitivity vs certified Lipschitz bound")
    ax.legend(loc="upper right")
    savefig(fig, "fig7_sensitivity_vs_certificate.png")


# ---------- Fig.8 小增益裕度扫描（E6-A 无延迟 + E6-B 延迟） ----------
def fig8():
    da = load_csv("e6a_delayfree.csv")
    db = load_csv("e6b_delay.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    # (a) 理论严格对应：无延迟。M(k_c) + 跟踪误差，M=0 竖线
    ax1.plot(da["k_c"], da["M_margin"], "o-", color="tab:blue",
             label=r"margin $M$")
    ax1.axhline(0, color="gray", ls="--", lw=1)
    ax1.set_xlabel(r"controller gain $k_c$")
    ax1.set_ylabel(r"$M = 1-k_c(L_{\mathrm{cert}}+\varepsilon_L)$")
    ax2a = ax1.twinx()
    err_a = da["track_err_L2"].replace(float("inf"), np.nan)
    ax2a.semilogy(da["k_c"], err_a, "s--", color="tab:red",
                  label=r"tracking error (one-step lag)")
    ax2a.set_ylabel(r"$\|q-r\|_{L^2}$ (after transient)")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2a.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8)
    ax1.set_title("(a) No intentional delay (1 ms): E6-A")
    # (b) 实现延迟的放大效应：tau = 0/100/200 ms
    err0 = da.set_index("mult")["track_err_L2"]
    ax2.semilogy(da["mult"], err0.values, "o-", color="tab:red",
                 label=r"one-step lag")
    for tau_ms, color in [(100, "tab:orange"), (200, "tab:purple")]:
        d = db[db["tau_ms"] == tau_ms].sort_values("mult")
        ax2.semilogy(d["mult"], d["track_err_L2"], "s--", color=color,
                     label=rf"$\tau={tau_ms}$ ms")
    ax2.axvline(1.0, color="gray", ls=":", lw=1)
    ax2.text(1.05, 0.30, r"$k_c^{\mathrm{safe}}$", fontsize=8, color="gray")
    ax2.set_xlabel(r"$k_c / k_c^{\mathrm{safe}}$")
    ax2.set_ylabel(r"$\|q-r\|_{L^2}$ (after transient)")
    ax2.set_title("(b) Implementation delay (E6-B)")
    ax2.legend(fontsize=8, loc="upper left")
    savefig(fig, "fig8_smallgain_margin.png")


# ---------- Fig.9 闭环轨迹 ----------
def fig9():
    df = load_csv("e6_closedloop.csv")
    row_safe = df[(df["design"].str.startswith("LC")) & (df["M_margin"] > 0)].iloc[-1]
    k_bad = df[(df["design"].str.startswith("LC")) & (df["M_margin"] < 0)]["k_c"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, (k_c, ttl) in zip(axes, [
            (row_safe["k_c"], f"(c) M>0: k_c={row_safe['k_c']:.2f} (calibrated safe)"),
            (float(k_bad.iloc[-1]) if len(k_bad) else row_safe["k_c"] * 8,
             f"(d) M<0: k_c={float(k_bad.iloc[-1]):.2f} = 128 k_c^safe" if len(k_bad) else "(d) M<0: N/A")]):
        t, q, r, u, div = C.closed_loop_sim(k_c, C.MSD_MID, C.reference_step,
                                            d_amp=0.0, delay_steps=100)
        ax.plot(t, r, "k--", lw=1, label="reference r(t)")
        ax.plot(t, q, color="tab:blue" if not div else "tab:red", lw=1.5, label="closed loop q(t)")
        ax.set_title(ttl + (" [DIVERGED]" if div else ""))
        ax.set_xlabel("t [s]"); ax.set_ylabel("q(t)"); ax.legend(fontsize=8)
    savefig(fig, "fig9_closedloop.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {"fig3": fig3, "fig4": fig4, "fig5": fig5, "fig7": fig7,
            "fig8": fig8, "fig9": fig9}
    if which == "all":
        for k, fn in jobs.items():
            try:
                fn()
            except FileNotFoundError as e:
                print(f"[skip] {k}: 缺少结果文件 {e}")
    else:
        jobs[which]()
