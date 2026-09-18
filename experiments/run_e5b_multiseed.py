"""E5b 消融 5-seed 化：{Lipschitz} × {动态损失} 2x2 × 5 seeds。
对应终审问题⑨（核心实验统一 mean±std）。
输出: results/e5b_ablation_5seed.csv（聚合）+ e5b_ablation_per_seed.csv
用法: python run_e5b_multiseed.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd

from common import save_csv
import config
import train as T
import control as C
from dataset import get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]

ABLTIONS = {
    "A (bare DeepONet)": dict(use_lip=False, use_dyn=False),
    "B (+Lip)":          dict(use_lip=True, use_dyn=False),
    "C (+Dyn)":          dict(use_lip=False, use_dyn=True),
    "D (full LC)":       dict(use_lip=True, use_dyn=True),
}

config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
u_pool = next(iter(te))[0]

per_seed = []
for cfg_name, kw in ABLTIONS.items():
    for s in seeds:
        res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                              seed=s, **kw)
        m = res["model_obj"]
        rec = {"config": cfg_name, "seed": s,
               "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"],
               "S_emp": C.estimate_S_emp(m, u_pool)}
        if kw["use_lip"]:
            rec["L_cert"] = m.evaluate_certificate()
        per_seed.append(rec)
        print(f"  {cfg_name:18s} seed={s} ID={res['ID_RMSE']:.4f} "
              f"OOD={res['OOD_RMSE']:.4f}", flush=True)

df = pd.DataFrame(per_seed)
agg = df.groupby("config").agg(
    ID_mean=("ID_RMSE", "mean"), ID_std=("ID_RMSE", "std"),
    OOD_mean=("OOD_RMSE", "mean"), OOD_std=("OOD_RMSE", "std"),
    S_emp_mean=("S_emp", "mean"), S_emp_std=("S_emp", "std"),
    L_cert_mean=("L_cert", "mean"), L_cert_std=("L_cert", "std"),
    n=("seed", "count")).reset_index()
save_csv(df, "e5b_ablation_per_seed.csv")
save_csv(agg, "e5b_ablation_5seed.csv")
print(agg.to_string(index=False))
