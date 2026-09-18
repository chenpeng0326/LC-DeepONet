"""E7: lambda_lip sensitivity —— Accuracy-Certificate Trade-off.

对 LC-DeepONet 以 lambda_lip ∈ {0.01, 0.1, 1, 10} 训练（seed=42，100 epochs），
记录 ID/OOD RMSE 与证书 L_hat，绘制 trade-off 曲线的原始数据。
lambda=0 等效 bare DeepONet（E5 的 A 行），不重复训练。

输出: results/e7_lam_sensitivity.csv
用法: python run_e7_lam.py [--smoke]
"""
import sys
import os

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from dataset import get_loaders  # noqa: E402
import train as T  # noqa: E402

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
LAMS = [0.01, 0.1, 1.0, 10.0] if not smoke else [0.1, 1.0]

results = []
for lam in LAMS:
    config.LAMBDA_LIP = lam
    config.set_seed(config.SEED)
    tr, te, ood = get_loaders(smoke=smoke)
    res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                          use_lip=True, seed=config.SEED)
    row = {"lambda_lip": lam,
           "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"],
           "L_cert": res.get("L_cert", float("nan")),
           "train_time_s": res["train_time_s"]}
    results.append(row)
    print(f"[lam] lambda_lip={lam}: ID={res['ID_RMSE']:.4f} "
          f"OOD={res['OOD_RMSE']:.4f} L_cert={row['L_cert']:.3f}")

out = os.path.join(config.RESULTS_DIR, "e7_lam_sensitivity.csv")
pd.DataFrame(results).to_csv(out, index=False)
print("saved", out)
