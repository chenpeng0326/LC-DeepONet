"""E6 闭环控制（压轴）：鲁棒裕度 M = 1 - k_c(L_hat + eps_L) 与真实闭环行为。

流程：
1. 训练 DeepONet 与 LC-DeepONet；
2. LC 侧：证书 L_cert + eps_L_emp -> k_c_safe = 0.9/(L_cert+eps_L)（M=0.1，理论安全域）；
   普通侧：S_emp -> k_c_opt = 0.9/S_emp（乐观设计）；
3. 估计真实系统增益 L_G_true（数值 G 的 Lipschitz 比值）；
4. k_c 倍率扫描 {0.25,0.5,1,2,4,8} x k_c_safe，闭环仿真（带一拍延迟 + 阶跃参考），
   观察跟踪误差 / 发散 vs M。
用法: python run_e6.py [--smoke]
"""
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
config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
u_pool = next(iter(te))[0]

# 1. 训练两个模型
res_lc = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs, use_lip=True)
res_do = T.full_pipeline("DeepONet", tr, te, ood, epochs=epochs, use_lip=False)

L_cert = res_lc["model_obj"].evaluate_certificate()
eps_L = C.estimate_eps_L(res_lc["model_obj"], u_pool, n_pairs=256)
S_emp_do = C.estimate_S_emp(res_do["model_obj"], u_pool)

# 2. 真实系统增益（数值 Lipschitz 比值，同一置换配对——修正配对 bug）
from control import true_G_batch
g = torch.Generator().manual_seed(1)
idx = torch.randint(0, u_pool.shape[0], (512,), generator=g)
up = u_pool[idx]
Gy = true_G_batch(up)
perm = torch.randperm(len(up), generator=g)
du = up[perm] - up          # 同一置换：du 与 dg 逐行配对
dg = Gy[perm] - Gy
num = torch.sqrt(config.DT * (dg ** 2).sum(dim=1))
den = torch.sqrt(config.DT * (du ** 2).sum(dim=1)).clamp_min(1e-8)
L_G_true = float((num / den).max())

# 3. 两种设计规则
k_c_safe = 0.9 / (L_cert + eps_L)      # LC 证书设计（保守，理论安全）
k_c_opt = 0.9 / S_emp_do               # 普通 DeepONet 经验设计（可能低估增益）
print(f"\nL_cert={L_cert:.3f} eps_L={eps_L:.3f} L_G_true={L_G_true:.3f}")
print(f"S_emp(DeepONet)={S_emp_do:.3f}")
print(f"k_c_safe(LC)={k_c_safe:.3f}  k_c_opt(DeepONet)={k_c_opt:.3f}")
print(f"覆盖检验: L_cert+eps_L vs L_G_true -> "
      f"{'安全' if L_cert + eps_L >= L_G_true else '不足(增益低估!)'}")

# 4. 倍率扫描（基于 k_c_safe；×64/128 进入性能恶化区，100ms 延迟）
rows = []
for mult in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
    k_c = k_c_safe * mult
    sim = C.sweep_gain([k_c], C.MSD_MID, r_func=C.reference_step, d_amp=0.05,
                       delay_override=100)[0]
    M_margin = 1.0 - k_c * (L_cert + eps_L)
    rows.append({"mult": mult, "k_c": k_c, "M_margin": M_margin,
                 "track_err_L2": sim["track_err_L2"],
                 "overshoot": sim["overshoot"], "diverged": sim["diverged"],
                 "design": "LC-certificate x mult"})
# 普通 DeepONet 设计点也放进对比
sim_opt = C.sweep_gain([k_c_opt], C.MSD_MID, r_func=C.reference_step, d_amp=0.05,
                       delay_override=100)[0]
rows.append({"mult": np.nan, "k_c": k_c_opt,
             "M_margin": 1.0 - k_c_opt * S_emp_do,  # 乐观裕度（用 S_emp 当增益）
             "track_err_L2": sim_opt["track_err_L2"],
             "overshoot": sim_opt["overshoot"], "diverged": sim_opt["diverged"],
             "design": "DeepONet S_emp design"})

df = pd.DataFrame(rows)
save_csv(df, "e6_closedloop.csv")
print(df.to_string(index=False))

# 设计量溯源（审稿意见 C-m5）：把 E6 设计用到的全部标量单独落盘，
# 使正文中的 L_cert / eps_L / L_cert+eps_L / k_c_safe / M 可由一个文件直接复算。
import os
pd.DataFrame([{
    "seed": config.SEED, "epochs": epochs,
    "L_cert": L_cert, "eps_L_hat": eps_L,
    "plant_bound": L_cert + eps_L,
    "k_c_safe": k_c_safe, "M_margin": 1.0 - k_c_safe * (L_cert + eps_L),
    "S_emp_DeepONet": S_emp_do, "k_c_opt_DeepONet": k_c_opt,
    "L_G_true_lower_bound": L_G_true,
    "n_pairs_epsL": 256, "sweep_delay_steps": 100, "sweep_d_amp": 0.05,
}]).to_csv(os.path.join(config.RESULTS_DIR, "e6_design_provenance.csv"),
           index=False)

# 摘要
safe_rows = df[(df["M_margin"] > 0) & df["M_margin"].notna() & (df["design"].str.startswith("LC"))]
unsafe_rows = df[(df["M_margin"] < 0) & df["M_margin"].notna() & (df["design"].str.startswith("LC"))]
print(f"\n[E6 摘要] M>0 区间: {len(safe_rows)} 个点, 发散 {int(safe_rows['diverged'].sum())} 个 | "
      f"M<0 区间: {len(unsafe_rows)} 个点, 发散 {int(unsafe_rows['diverged'].sum())} 个")
print("（预期：M>0 时闭环稳定，M<0 后稳定性失去理论保证，可能出现大振荡/发散——"
      "二者相关性是论文核心图 Fig.8/9 的素材）")
