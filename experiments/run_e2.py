"""E2 参数 OOD（加固版）：5 seeds + 512 个 OOD 测试样本，报告 mean±std。
训练 alpha∈[0.5,1.2]，测试 alpha∈[1.2,1.5]。含 Duffing/Pendulum 迁移验证。
用法: python run_e2.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd

from common import save_csv
import config
import train as T
import systems
from dataset import to_loader, get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
N_SEEDS_E2 = 1 if smoke else 5
N_OOD_BIG = config.SMOKE_N_OOD if smoke else 512   # 加固：扩大 OOD 测试集

rows, rows_seed = [], []
# 主系统 OOD（训练分布内 vs OOD 区间）
tr, te, _ = get_loaders(smoke=smoke)
ood_data = systems.generate_dataset("msd", N_OOD_BIG, seed=902, ood=True)
ood = to_loader(ood_data, 256, False)

for name in ["DeepONet", "LC-DeepONet"]:
    per_seed = []
    for s_off in range(N_SEEDS_E2):
        res = T.full_pipeline(name, tr, te, ood, epochs=epochs,
                              use_lip=(name == "LC-DeepONet"),
                              seed=config.SEED + s_off)
        per_seed.append(res)
        rows_seed.append({"model": name, "seed": config.SEED + s_off,
                          "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"]})
    id_m, id_s = float(np.mean([r["ID_RMSE"] for r in per_seed])), float(np.std([r["ID_RMSE"] for r in per_seed]))
    ood_m, ood_s = float(np.mean([r["OOD_RMSE"] for r in per_seed])), float(np.std([r["OOD_RMSE"] for r in per_seed]))
    lc = per_seed[-1]["model_obj"].evaluate_certificate() if name == "LC-DeepONet" else np.nan
    rows.append({"system": "msd(主)", "model": name, "n_seeds": N_SEEDS_E2,
                 "ID_RMSE_mean": id_m, "ID_RMSE_std": id_s,
                 "OOD_RMSE_mean": ood_m, "OOD_RMSE_std": ood_s,
                 "OOD_degradation_%": 100 * (ood_m - id_m) / id_m,
                 "L_cert": lc})
    print(f"  msd {name:12s} ID={id_m:.4f}±{id_s:.4f} OOD={ood_m:.4f}±{ood_s:.4f}")

# 验证系统：Duffing / Pendulum（仅验证可迁移性，不做全套）
for sys_name in ["duffing", "pendulum"]:
    data_tr = systems.generate_dataset(sys_name, config.SMOKE_N_TRAIN if smoke else config.N_TRAIN, seed=7)
    data_te = systems.generate_dataset(sys_name, config.SMOKE_N_TEST if smoke else config.N_TEST, seed=8)
    tr2 = to_loader(data_tr, config.BATCH, True, seed=7)
    te2 = to_loader(data_te, 256, False)
    for name in ["DeepONet", "LC-DeepONet"]:
        res = T.full_pipeline(name, tr2, te2, te2, epochs=epochs,
                              use_lip=(name == "LC-DeepONet"))
        rows.append({"system": sys_name, "model": name, "n_seeds": 1,
                     "ID_RMSE_mean": res["ID_RMSE"], "ID_RMSE_std": np.nan,
                     "OOD_RMSE_mean": np.nan, "OOD_RMSE_std": np.nan,
                     "OOD_degradation_%": np.nan,
                     "L_cert": res.get("L_cert", np.nan)})
        print(f"  {sys_name} {name:12s} ID={res['ID_RMSE']:.4f}")

df = pd.DataFrame(rows)
save_csv(df, "e2_ood.csv")
pd.DataFrame(rows_seed).to_csv(
    __import__("os").path.join(config.RESULTS_DIR, "e2_ood_per_seed.csv"), index=False)
print(df.to_string(index=False))
