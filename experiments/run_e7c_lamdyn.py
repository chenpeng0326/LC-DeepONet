"""E7c λ_dyn 敏感性：λ_dyn∈{0,0.01,0.1,1} × 5 seeds（LC 完整配置 D）。
对应终审问题⑧：dynamic loss 的必要性/代价。指标：ID、OOD、参数 OOD 外的
T=4s 长时程 RMSE（与 E9 协议一致）。
输出: results/e7c_lamdyn.csv（聚合）+ e7c_lamdyn_per_seed.csv
用法: python run_e7c_lamdyn.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd

from common import save_csv
import config
import train as T
import systems
from dataset import get_loaders, to_loader

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]
LAMS = [0.0, 0.01, 0.1, 1.0]


def long_test_loader(t_end, seed):
    old = config.T_END
    config.T_END = t_end
    data = systems.generate_dataset("msd", 128, seed=seed)
    config.T_END = old
    return to_loader(data, 256, False)


config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
loader4 = long_test_loader(4.0, seed=901)

per_seed = []
old_lam = config.LAMBDA_DYN
try:
    for lam in LAMS:
        config.LAMBDA_DYN = lam
        for s in seeds:
            res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                                  seed=s, use_lip=True, use_dyn=True)
            model = res["model_obj"]
            rec = {"lambda_dyn": lam, "seed": s,
                   "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"],
                   "RMSE_4s": T.evaluate(model, loader4)["RMSE"]}
            if "L_cert" in res:
                rec["L_cert"] = res["L_cert"]
            per_seed.append(rec)
            print(f"  lam={lam:<5} seed={s} ID={rec['ID_RMSE']:.4f} "
                  f"OOD={rec['OOD_RMSE']:.4f} RMSE4s={rec['RMSE_4s']:.4f}", flush=True)
finally:
    config.LAMBDA_DYN = old_lam

df = pd.DataFrame(per_seed)
agg = df.groupby("lambda_dyn").agg(
    ID_mean=("ID_RMSE", "mean"), ID_std=("ID_RMSE", "std"),
    OOD_mean=("OOD_RMSE", "mean"), OOD_std=("OOD_RMSE", "std"),
    RMSE4s_mean=("RMSE_4s", "mean"), RMSE4s_std=("RMSE_4s", "std"),
    L_cert_mean=("L_cert", "mean"), n=("seed", "count")).reset_index()
save_csv(df, "e7c_lamdyn_per_seed.csv")
save_csv(agg, "e7c_lamdyn.csv")
print(agg.to_string(index=False))
