"""E4b: 噪声鲁棒性 5 seeds 版（回应终审清单第 7 项：核心实验统一 5 seeds）。

6 模型 × 5 seeds 训练；噪声评估用固定噪声种子（123）以隔离模型间方差。
输出: results/e4b_noise_5seed.csv（含聚合 mean±std）
用法: python run_e4b_multiseed.py [--smoke]
"""
import sys
import os

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import models as M  # noqa: E402
import train as T  # noqa: E402
from dataset import get_loaders  # noqa: E402

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
SEEDS = [config.SEED + i for i in range(config.N_SEEDS)] if not smoke else [config.SEED]
NOISES = [0.01, 0.03, 0.05, 0.10]


@torch.no_grad()
def eval_noise(model, te, level):
    model.eval()
    se, n = 0.0, 0
    for u, q, _ in te:
        g = torch.Generator().manual_seed(123)  # 固定噪声实现
        un = u + level * u.abs().max() * torch.randn(u.shape, generator=g)
        y = model(un)
        se += float(((y - q) ** 2).sum())
        n += q.numel()
    return (se / n) ** 0.5


rows = []
for seed in SEEDS:
    config.set_seed(seed)
    tr, te, ood = get_loaders(smoke=smoke)
    for name in M.MODEL_NAMES:
        use_lip = (name == "LC-DeepONet")
        res = T.full_pipeline(name, tr, te, ood, epochs=epochs,
                              use_lip=use_lip, seed=seed)
        row = {"model": name, "seed": seed}
        for lv in NOISES:
            row[f"RMSE_noise_{int(lv*100)}%"] = eval_noise(res["model_obj"], te, lv)
        rows.append(row)
        print(f"  seed={seed} {name:12s} " +
              " ".join(f"{row[k]:.4f}" for k in row if k not in ("model", "seed")))

df = pd.DataFrame(rows)
num_cols = [c for c in df.columns if c not in ("model", "seed")]
# 注意：早期版本此处用 .agg([("mean","mean"),("std","std")]) 后再 drop，
# 会误把 seed 列的 std(=1.5811) 当成首个指标列写出，产生列错位的坏 CSV。
# 现在改为显式命名 mean/std 两套列，文件名即含义。
agg = df.groupby("model")[num_cols].agg(["mean", "std"]).round(6)
agg.columns = [f"{c}_{s}" for c, s in agg.columns]
print(agg.to_string())

out = os.path.join(config.RESULTS_DIR, "e4b_noise_5seed.csv")
df.to_csv(out.replace(".csv", "_per_seed.csv"), index=False)
agg.to_csv(out)
print("saved", out)
