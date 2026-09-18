"""E1 系统辨识性能：6 模型 × (RMSE/MAE/NRMSE/参数量/推理时间)。
正式模式：EPOCHS × N_SEEDS 取 mean±std。用法: python run_e1.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd

from common import save_csv
import config
import models as M
import train as T
from dataset import get_loaders

smoke = "--smoke" in sys.argv
config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]

rows = []
for name in M.MODEL_NAMES:
    per_seed = []
    for s in seeds:
        res = T.full_pipeline(name, tr, te, ood, epochs=epochs, seed=s)
        per_seed.append(res)
        print(f"  {name:12s} seed={s} ID_RMSE={res['ID_RMSE']:.4f} "
              f"OOD_RMSE={res['OOD_RMSE']:.4f}")
    row = {"model": name, "params": per_seed[0]["params"],
           "train_time_s_mean": float(np.mean([r["train_time_s"] for r in per_seed]))}
    for key in ["ID_RMSE", "ID_MAE", "ID_NRMSE", "OOD_RMSE", "OOD_MAE"]:
        vals = [r[key] for r in per_seed]
        row[key + "_mean"] = float(np.mean(vals))
        row[key + "_std"] = float(np.std(vals))
    row["infer_ms"] = per_seed[0]["ID_infer_ms"]
    if "L_cert" in per_seed[-1]:
        row["L_cert"] = per_seed[-1]["L_cert"]
    rows.append(row)

df = pd.DataFrame(rows)
save_csv(df, "e1_identification.csv")
print(df[["model", "ID_RMSE_mean", "OOD_RMSE_mean"]].to_string(index=False))
