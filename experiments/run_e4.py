"""E4 噪声鲁棒性：输入/输出噪声 1%/3%/5%/10%。
用法: python run_e4.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd
import torch

from common import save_csv
import config
import models as M
import train as T
from dataset import get_loaders

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)

NOISES = [0.01, 0.03, 0.05, 0.10]


@torch.no_grad()
def eval_noise(model, te, level):
    model.eval()
    se, n = 0.0, 0
    for u, q, _ in te:
        g = torch.Generator().manual_seed(123)
        un = u + level * u.abs().max() * torch.randn(u.shape, generator=g)
        y = model(un)
        se += float(((y - q) ** 2).sum())
        n += q.numel()
    return (se / n) ** 0.5


rows = []
for name in M.MODEL_NAMES:
    use_lip = (name == "LC-DeepONet")
    res = T.full_pipeline(name, tr, te, ood, epochs=epochs, use_lip=use_lip)
    row = {"model": name}
    for lv in NOISES:
        row[f"RMSE_noise_{int(lv*100)}%"] = eval_noise(res["model_obj"], te, lv)
    rows.append(row)
    print(f"  {name:12s} " + " ".join(f"{row[k]:.4f}" for k in row if k != "model"))

df = pd.DataFrame(rows)
save_csv(df, "e4_noise.csv")
print(df.to_string(index=False))
