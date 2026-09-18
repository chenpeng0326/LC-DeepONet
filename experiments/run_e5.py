"""E5 消融实验：{Lipschitz 约束} × {动态损失} 2x2（A/B/C/D 四配置）。
重点观察 RMSE 与 L_hat/L_cert 的关系。
用法: python run_e5.py [--smoke]
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
config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
u_pool = next(iter(te))[0]

ABLTIONS = {
    "A(基线)": dict(use_lip=False, use_dyn=False),
    "B(+Lip)": dict(use_lip=True, use_dyn=False),
    "C(+Dyn)": dict(use_lip=False, use_dyn=True),
    "D(完整LC)": dict(use_lip=True, use_dyn=True),
}

rows = []
for cfg_name, kw in ABLTIONS.items():
    res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs, **kw)
    m = res["model_obj"]
    s_emp = C.estimate_S_emp(m, u_pool)
    row = {"config": cfg_name, **kw, "ID_RMSE": res["ID_RMSE"],
           "OOD_RMSE": res["OOD_RMSE"], "S_emp": s_emp}
    if kw["use_lip"]:
        row["L_cert"] = m.evaluate_certificate()
    rows.append(row)
    print(f"  {cfg_name:10s} ID={res['ID_RMSE']:.4f} OOD={res['OOD_RMSE']:.4f} "
          f"S_emp={s_emp:.3f} L_cert={row.get('L_cert', float('nan')):.2f}")

df = pd.DataFrame(rows)
save_csv(df, "e5_ablation.csv")
print(df.to_string(index=False))
