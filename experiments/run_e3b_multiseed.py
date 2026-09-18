"""E3b 敏感性 5-seed 化：6 模型 × 5 seeds，S_emp mean±std。
对应终审问题⑨（sensitivity 属核心实验）；LC 附带 L_cert / eps_L_emp 逐种子分布。
输出: results/e3b_sensitivity_5seed.csv（聚合）+ e3b_sensitivity_per_seed.csv
用法: python run_e3b_multiseed.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd

from common import save_csv
import config
import models as M
import train as T
import control as C
from dataset import get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]

config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
u_pool = next(iter(te))[0]

per_seed = []
for name in M.MODEL_NAMES:
    use_lip = (name == "LC-DeepONet")
    for s in seeds:
        res = T.full_pipeline(name, tr, te, ood, epochs=epochs, seed=s,
                              use_lip=use_lip)
        m = res["model_obj"]
        rec = {"model": name, "seed": s,
               "S_emp": C.estimate_S_emp(m, u_pool),
               "ID_RMSE": res["ID_RMSE"]}
        if use_lip:
            rec["L_cert"] = m.evaluate_certificate()
            rec["eps_L_emp"] = C.estimate_eps_L(m, u_pool, n_pairs=256)
        per_seed.append(rec)
        print(f"  {name:12s} seed={s} S_emp={rec['S_emp']:.3f}", flush=True)

df = pd.DataFrame(per_seed)
agg = df.groupby("model").agg(
    S_emp_mean=("S_emp", "mean"), S_emp_std=("S_emp", "std"),
    ID_mean=("ID_RMSE", "mean"), ID_std=("ID_RMSE", "std"),
    L_cert_mean=("L_cert", "mean"), L_cert_std=("L_cert", "std"),
    eps_L_mean=("eps_L_emp", "mean"), eps_L_std=("eps_L_emp", "std"),
    n=("seed", "count")).reset_index()
save_csv(df, "e3b_sensitivity_per_seed.csv")
save_csv(agg, "e3b_sensitivity_5seed.csv")
print(agg.to_string(index=False))
