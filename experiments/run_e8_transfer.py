"""E8 多系统迁移 benchmark：Duffing + Pendulum × {LC-DeepONet, DeepONet, FNO} × 5 seeds。
对应终审问题⑤：方法不绑定 MSD 方程；pendulum 验证 sin(theta) 非多项式非线性。
OOD 协议镜像 MSD（参数上界外推 25%）：duffing beta_d∈[4.0,5.0]，pendulum g/l∈[10.0,12.5]。
输出: results/e8_transfer.csv（聚合 mean±std）+ e8_transfer_per_seed.csv
用法: python run_e8_transfer.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd

from common import save_csv
import config
import train as T
from dataset import get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]

MODELS = [
    ("LC-DeepONet", dict(use_lip=True, use_dyn=True)),   # D 配置（与 E1 一致）
    ("DeepONet", dict(use_lip=False, use_dyn=False)),    # A 配置
    ("FNO", dict(use_lip=False, use_dyn=False)),
]

per_seed = []
for system in ["duffing", "pendulum"]:
    config.set_seed(config.SEED)
    tr, te, ood = get_loaders(system=system, smoke=smoke)
    for name, kw in MODELS:
        for s in seeds:
            res = T.full_pipeline(name, tr, te, ood, epochs=epochs, seed=s, **kw)
            rec = {"system": system, "model": name, "seed": s,
                   "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"]}
            if "L_cert" in res:
                rec["L_cert"] = res["L_cert"]
            per_seed.append(rec)
            print(f"  {system:8s} {name:12s} seed={s} "
                  f"ID={res['ID_RMSE']:.4f} OOD={res['OOD_RMSE']:.4f}", flush=True)

df = pd.DataFrame(per_seed)
agg = df.groupby(["system", "model"]).agg(
    ID_mean=("ID_RMSE", "mean"), ID_std=("ID_RMSE", "std"),
    OOD_mean=("OOD_RMSE", "mean"), OOD_std=("OOD_RMSE", "std"),
    L_cert_mean=("L_cert", "mean"), L_cert_std=("L_cert", "std"),
    n=("seed", "count")).reset_index()
save_csv(df, "e8_transfer_per_seed.csv")
save_csv(agg, "e8_transfer.csv")
print(agg.to_string(index=False))
