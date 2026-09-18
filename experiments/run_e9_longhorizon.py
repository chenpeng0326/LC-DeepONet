"""E9 long-horizon rollout：T_train=2s 训练，T_test∈{4,6}s 全时程评估。
对应终审问题⑥：短时精度 != 长时动态一致性。固定输入预算 N=64（训练约定 √Δt 缩放不变），
报告全时程 RMSE（mean±std over 5 seeds）与逐时刻 RMSE(t) 曲线。
输出: results/e9_longhorizon.csv + e9_longhorizon_per_seed.csv + e9_rmse_t.csv
用法: python run_e9_longhorizon.py [--smoke]
"""
import sys

import numpy as np
import pandas as pd
import torch

from common import save_csv
import config
import train as T
import systems
from dataset import get_loaders, to_loader

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]
T_TESTS = [4.0, 6.0]
N_LONG = config.SMOKE_N_TEST if smoke else 128
MODELS = ["LC-DeepONet", "DeepONet", "FNO"]


def long_test_loader(t_end, seed):
    """生成 T=t_end 的 ID 参数测试集（√Δt 缩放保持训练约定，不重算）。"""
    old = config.T_END
    config.T_END = t_end
    data = systems.generate_dataset("msd", N_LONG, seed=seed)
    config.T_END = old
    return to_loader(data, 256, False)


config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)

rows_agg, rows_curve = [], []
for name in MODELS:
    use_lip = (name == "LC-DeepONet")
    for s in seeds:
        res = T.full_pipeline(name, tr, te, ood, epochs=epochs, seed=s,
                              use_lip=use_lip, use_dyn=True)
        model = res["model_obj"]
        for t_end in T_TESTS:
            loader = long_test_loader(t_end, seed=900 + s)
            _, qs, ys = T.predict(model, loader)
            diff2 = ((ys - qs) ** 2).mean(dim=0)          # [N] 逐时刻 MSE
            rmse_t = torch.sqrt(diff2).cpu().numpy()
            tt = np.linspace(0.0, t_end, len(rmse_t))
            rows_agg.append({"model": name, "seed": s, "T_test": t_end,
                             "RMSE": float(np.sqrt(diff2.mean().item()))})
            for j in range(len(tt)):
                rows_curve.append({"model": name, "T_test": t_end, "t": tt[j],
                                   "seed": s, "rmse_t": float(rmse_t[j])})
        print(f"  {name:12s} seed={s} done", flush=True)

df = pd.DataFrame(rows_agg)
agg = df.groupby(["model", "T_test"]).agg(
    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
    n=("seed", "count")).reset_index()
save_csv(df, "e9_longhorizon_per_seed.csv")
save_csv(agg, "e9_longhorizon.csv")
curve = pd.DataFrame(rows_curve).groupby(["model", "T_test", "t"], as_index=False).agg(
    rmse_t_mean=("rmse_t", "mean"), rmse_t_std=("rmse_t", "std"))
save_csv(curve, "e9_rmse_t.csv")
print(agg.to_string(index=False))
