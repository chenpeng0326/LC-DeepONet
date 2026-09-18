"""E3 输入扰动敏感性（核心实验）：S_emp vs 证书 L_hat。
对每个模型计算经验敏感度，LC-DeepONet 应满足 S_emp <= L_cert。
用法: python run_e3.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd
import torch

from common import save_csv
import config
import models as M
import train as T
import control as C
from dataset import get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
u_pool = next(iter(te))[0]   # 固定输入池（中值参数真实 G 由 control.true_G_batch 提供）

rows = []
for name in M.MODEL_NAMES:
    use_lip = (name == "LC-DeepONet")
    res = T.full_pipeline(name, tr, te, ood, epochs=epochs, use_lip=use_lip)
    m = res["model_obj"]
    s_emp = C.estimate_S_emp(m, u_pool)
    row = {"model": name, "S_emp": s_emp,
           "ID_RMSE": res["ID_RMSE"]}
    if use_lip:
        L_cert = m.evaluate_certificate()
        eps_L = C.estimate_eps_L(m, u_pool, n_pairs=256)
        row.update({"L_cert": L_cert, "eps_L_emp": eps_L,
                    "S_emp_le_Lcert": bool(s_emp <= L_cert),
                    "plant_bound_Lhat_plus_epsL": L_cert + eps_L})
    rows.append(row)
    print(f"  {name:12s} S_emp={s_emp:.3f} {row.get('L_cert', '')}")

df = pd.DataFrame(rows)
save_csv(df, "e3_sensitivity.csv")
print(df.to_string(index=False))
