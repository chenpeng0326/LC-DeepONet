"""E7b: L_target sweep —— Accuracy-Certificate Trade-off.

lambda_lip 扫描在默认 L_target=5 下罚项不激活（surrogate 全程 < L_target），
所有 lambda 给出同一模型 —— 这本身是对 lambda 的稳健性结论（见论文）。
有信息量的 trade-off 来自压缩 L_target：证书上限越紧，RMSE 代价越大。

L_target ∈ {1.5, 2.0, 2.5, 3.0, 4.0, 5.0}（5.0=默认），lambda_lip=1，seed=42。

输出: results/e7b_ltarget_sensitivity.csv
用法: python run_e7b_ltarget.py [--smoke]
"""
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from dataset import get_loaders  # noqa: E402
import train as T  # noqa: E402

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
TARGETS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0] if not smoke else [2.0, 5.0]
SEEDS = [config.SEED + i for i in range(config.N_SEEDS)] if not smoke else [config.SEED]

config.LAMBDA_LIP = 1.0
results = []
for lt in TARGETS:
    config.LIP_TARGET = lt
    for seed in SEEDS:
        config.set_seed(seed)
        tr, te, ood = get_loaders(smoke=smoke)
        res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                              use_lip=True, seed=seed)
        results.append({"L_target": lt, "seed": seed,
                        "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"],
                        "L_cert": res.get("L_cert", float("nan")),
                        "train_time_s": res["train_time_s"]})
        print(f"[lt] L_target={lt} seed={seed}: ID={res['ID_RMSE']:.4f} "
              f"OOD={res['OOD_RMSE']:.4f} L_cert={results[-1]['L_cert']:.3f}")

df = pd.DataFrame(results)
if len(SEEDS) > 1:
    agg = df.groupby("L_target").agg(
        ID_mean=("ID_RMSE", "mean"), ID_std=("ID_RMSE", "std"),
        OOD_mean=("OOD_RMSE", "mean"), OOD_std=("OOD_RMSE", "std"),
        L_cert_mean=("L_cert", "mean"), L_cert_std=("L_cert", "std"),
        n=("seed", "count")).reset_index()
    print(agg.to_string(index=False))
    out = os.path.join(config.RESULTS_DIR, "e7b_ltarget_sensitivity.csv")
    agg.to_csv(out, index=False)
else:
    out = os.path.join(config.RESULTS_DIR, "e7b_ltarget_sensitivity.csv")
    df.to_csv(out, index=False)
print("saved", out)
