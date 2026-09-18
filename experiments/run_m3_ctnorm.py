"""模拟审稿 M3+M4 探针实验：
M3: Theorem 1 的 C_T = sqrt(DT * sum_j ||T(t_j)||^2) 是 trunk 特征矩阵
    Phi_T = [sqrt(DT) T(t_j)] 的 Frobenius 范数；induced-norm 意义下的紧常数
    是谱范数 ||Phi_T||_2 <= ||Phi_T||_F。本脚本训练 LC-DeepONet 5 seeds，
    测量两者差距与 L_cert 的两个版本，量化 trunk 项保守度。
M4: 逐 epoch 记录可微 surrogate L_hat（history 中的 "L_hat"），统计 hinge
    罚项 max(0, L_hat - L_target)^2 的激活情况（哪些 seed / 多少 epoch 激活），
    解释 E7（单 seed 永不激活，λ_lip 无影响）与 E5-B（与 A 有差异）的张力。
输出: results/m3_ctnorm_per_seed.csv
用法: python run_m3_ctnorm.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd
import torch

from common import save_csv
import config
import train as T
import lipschitz as lip
from dataset import get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]


@torch.no_grad()
def trunk_constants(model):
    """返回 (C_T_fro, C_T_spec)。Phi_T 的行是 sqrt(DT)*T(t_j)。"""
    Tt = model.trunk(model.t_grid.unsqueeze(-1))            # [N, p]
    Phi = (config.DT ** 0.5) * Tt
    c_fro = float(torch.linalg.matrix_norm(Phi, ord="fro"))
    c_spec = float(torch.linalg.matrix_norm(Phi, ord=2))
    return c_fro, c_spec


@torch.no_grad()
def branch_prod_sigma(model):
    p = 1.0
    for lin in model.branch_linears:
        p *= lip.exact_sigma(lin.weight)
    return float(p)


config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)

rows = []
for s in seeds:
    # --- LC 完整配置（dyn+lip，与 E1/E3b 一致）---
    res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs, seed=s,
                          use_dyn=True, use_lip=True)
    m = res["model_obj"]
    hist = pd.DataFrame(res["history"])
    surro_max = float(hist["L_hat"].max())
    surro_init = float(hist["L_hat"].iloc[0])
    active_eps = int((hist["L_hat"] > config.LIP_TARGET).sum())

    g = branch_prod_sigma(m)
    cf, cs = trunk_constants(m)
    rows.append({
        "seed": s, "config": "LC(D)",
        "surro_init": surro_init, "surro_max": surro_max,
        "hinge_active_epochs": active_eps,
        "branch_prod": g, "C_T_fro": cf, "C_T_spec": cs,
        "L_cert_fro": g * cf, "L_cert_spec": g * cs,
        "ID_RMSE": res["ID_RMSE"],
    })
    print(f"  LC   seed={s} init_surro={surro_init:.2f} max={surro_max:.2f} "
          f"active_eps={active_eps} L_fro={g*cf:.3f} L_spec={g*cs:.3f} "
          f"spec/fro={cs/cf:.3f}", flush=True)

    # --- B 配置（仅 lip，无 dyn，与 E5-B 一致）---
    resB = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs, seed=s,
                           use_dyn=False, use_lip=True)
    histB = pd.DataFrame(resB["history"])
    rows.append({
        "seed": s, "config": "B(+lip)",
        "surro_init": float(histB["L_hat"].iloc[0]),
        "surro_max": float(histB["L_hat"].max()),
        "hinge_active_epochs": int((histB["L_hat"] > config.LIP_TARGET).sum()),
        "branch_prod": np.nan, "C_T_fro": np.nan, "C_T_spec": np.nan,
        "L_cert_fro": np.nan, "L_cert_spec": np.nan,
        "ID_RMSE": resB["ID_RMSE"],
    })
    print(f"  B    seed={s} init_surro={rows[-1]['surro_init']:.2f} "
          f"max={rows[-1]['surro_max']:.2f} "
          f"active_eps={rows[-1]['hinge_active_eps'] if 'hinge_active_eps' in rows[-1] else rows[-1]['hinge_active_epochs']} "
          f"ID={resB['ID_RMSE']:.4f}", flush=True)

df = pd.DataFrame(rows)
save_csv(df, "m3_ctnorm_per_seed.csv")
lc = df[df["config"] == "LC(D)"]
print("\n=== M3 汇总（LC 5 seeds）===")
print(f"L_cert(Fro)  = {lc['L_cert_fro'].mean():.3f} ± {lc['L_cert_fro'].std():.3f}")
print(f"L_cert(spec) = {lc['L_cert_spec'].mean():.3f} ± {lc['L_cert_spec'].std():.3f}")
print(f"C_T spec/fro = {(lc['C_T_spec']/lc['C_T_fro']).mean():.3f}")
print(f"trunk 项占比: spec 版 C_T²·? — branch_prod={lc['branch_prod'].mean():.3f}")
print("\n=== M4 汇总（hinge 激活）===")
print(df[["seed", "config", "surro_init", "surro_max",
          "hinge_active_epochs", "ID_RMSE"]].to_string(index=False))
