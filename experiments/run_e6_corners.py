"""E6-C：参数不确定性盒内的角点扫描（回应审稿意见 C-M2 / R1-M2 / R2-M3）。

动机：原 E6 的 eps_L 估计、L_G_true 与闭环仿真都固定在盒心 MSD_MID=(0.6,1.0,0.85)，
     因此只能算"标称工况的校准演示"。本脚本把同一个训练好的 LC-DeepONet 放到
     不确定性盒的 8 个角点 + 盒心，重新估计残差敏感度与真实对象增益，
     并给出最坏情形下的闭环表现。

盒：c∈[0.4,0.8], k∈[0.8,1.2], alpha∈[0.5,1.2]（config.MSD）

输出：
- results/e6c_corners.csv          每个参数点的 eps_L_hat / L_G_true / 比值
- results/e6c_closedloop.csv       盒心设计增益 vs 角点设计增益在各参数点的闭环表现
- 控制台摘要（worst-case eps_L、是否覆盖真实增益、M>0 区间是否稳定）

用法: python run_e6_corners.py [--smoke]
"""
import itertools
import sys

import numpy as np
import pandas as pd
import torch

from common import save_csv
import config
import train as T
import control as C
from dataset import get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
n_seeds = 1 if smoke else 5
n_pairs = 64 if smoke else 256

config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
u_pool = next(iter(te))[0]
# 角点扫描用固定子池，保证各参数点的配对完全一致（可比）
g_pool = torch.Generator().manual_seed(7)
pool_idx = torch.randperm(u_pool.shape[0], generator=g_pool)[: 32 if smoke else 96]
u_sub = u_pool[pool_idx]

MID = C.MSD_MID
BOX = {"c": (0.4, 0.8), "k": (0.8, 1.2), "alpha": (0.5, 1.2)}
corner_vals = [BOX["c"], BOX["k"], BOX["alpha"]]
CORNERS = [tuple(v) for v in itertools.product(*corner_vals)]
POINTS = CORNERS + [MID]
LABELS = [f"corner({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})" for p in CORNERS] + ["mid"]


def true_gain(params, n_pairs_gain=512):
    """真实对象增益 L_G_true：数值 G 的 Lipschitz 比值（同一置换配对）。"""
    from control import true_G_batch
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, u_sub.shape[0], (min(n_pairs_gain, u_sub.shape[0]),), generator=g)
    up = u_sub[idx]
    Gy = true_G_batch(up, params=params)
    perm = torch.randperm(len(up), generator=g)
    du = up[perm] - up
    dg = Gy[perm] - Gy
    num = torch.sqrt(config.DT * (dg ** 2).sum(dim=1))
    den = torch.sqrt(config.DT * (du ** 2).sum(dim=1)).clamp_min(1e-8)
    return float((num / den).max())


# ---------- 1. 训练若干 LC-DeepONet，逐个模型在全部参数点重新估计 ----------
rows = []
cert_rows = []
for s_off in range(n_seeds):
    seed = config.SEED + s_off
    res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                          use_lip=True, seed=seed)
    model = res["model_obj"]
    L_cert = float(model.evaluate_certificate())
    cert_rows.append({"seed": seed, "L_cert": L_cert})
    for label, params in zip(LABELS, POINTS):
        eps = C.estimate_eps_L(model, u_sub, n_pairs=n_pairs, params=params)
        Lg = true_gain(params)
        rows.append({"seed": seed, "point": label,
                     "c": params[0], "k": params[1], "alpha": params[2],
                     "L_cert": L_cert, "eps_L_hat": eps,
                     "plant_bound": L_cert + eps,
                     "L_G_true": Lg,
                     "covers_true": bool(L_cert + eps >= Lg)})
        print(f"  seed={seed} {label:22s} L_cert={L_cert:.3f} "
              f"eps_L={eps:.4f} bound={L_cert+eps:.3f} L_G_true={Lg:.3f} "
              f"{'OK' if L_cert + eps >= Lg else 'UNDER!'}", flush=True)

df = pd.DataFrame(rows)
save_csv(df, "e6c_corners.csv")

# ---------- 2. 汇总最坏情形 ----------
summ = (df.groupby("point")
          .agg(eps_L_mean=("eps_L_hat", "mean"),
               eps_L_max=("eps_L_hat", "max"),
               L_G_true_max=("L_G_true", "max"),
               bound_mean=("plant_bound", "mean"),
               covers=("covers_true", "all"))
          .reset_index())
print("\n[E6-C 汇总]")
print(summ.to_string(index=False))

L_cert_bar = float(np.mean([r["L_cert"] for r in cert_rows]))
eps_mid = float(df[df["point"] == "mid"]["eps_L_hat"].mean())
eps_worst = float(df["eps_L_hat"].max())
Lg_worst = float(df["L_G_true"].max())

k_c_mid = 0.9 / (L_cert_bar + eps_mid)          # 盒心校准设计（原 E6 设计）
k_c_worst = 0.9 / (L_cert_bar + eps_worst)      # 箱内最坏校准设计
print(f"\nL_cert(mean over {n_seeds} seeds) = {L_cert_bar:.3f}")
print(f"eps_L: mid={eps_mid:.4f}  worst-case over box={eps_worst:.4f} "
      f"({eps_worst/max(eps_mid,1e-12):.2f}x mid)")
print(f"L_G_true: max over box = {Lg_worst:.3f}")
print(f"k_c: mid-calibrated={k_c_mid:.4f}  box-worst-calibrated={k_c_worst:.4f} "
      f"(ratio {k_c_worst/k_c_mid:.3f})")
print(f"Coverage: mid bound {L_cert_bar+eps_mid:.3f} >= box max true gain "
      f"{Lg_worst:.3f} -> {'YES' if L_cert_bar+eps_mid >= Lg_worst else 'NO'}")

# ---------- 3. 闭环：盒心设计 vs 箱内最坏设计，在各参数点仿真 ----------
cl = []
for label, params in zip(LABELS, POINTS):
    for tag, k_c in (("mid-calibrated", k_c_mid), ("box-worst-calibrated", k_c_worst)):
        sim = C.sweep_gain([k_c], params, r_func=C.reference_step,
                           d_amp=0.05, delay_override=100)[0]
        cl.append({"point": label, "c": params[0], "k": params[1],
                   "alpha": params[2], "design": tag, "k_c": k_c,
                   "track_err_L2": sim["track_err_L2"],
                   "overshoot": sim["overshoot"], "diverged": sim["diverged"]})
df_cl = pd.DataFrame(cl)
df_cl.to_csv(config.RESULTS_DIR.replace("\\", "/") + "/e6c_closedloop.csv", index=False)

piv = df_cl.pivot_table(index="point", columns="design",
                        values="track_err_L2").reset_index()
print("\n[E6-C 闭环跟踪误差 L2（d_amp=0.05, 100ms 延迟）]")
print(piv.to_string(index=False))

print("\n[E6-C 摘要] 盒心设计与箱内最坏设计在各参数点的闭环跟踪误差如上；"
      "发散点数: %d" % int(df_cl["diverged"].sum()))
