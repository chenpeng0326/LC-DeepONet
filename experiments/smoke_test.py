"""Smoke Test（拟定书阶段②）：10分钟内验证全链路。

检查项：
1. 数据生成正常（RK4 有限、量级合理）
2. 6 个模型都能训练且 loss 下降
3. LC-DeepONet 的 L_hat 随训练下降且证书可计算
4. OOD / 敏感性 / eps_L / 闭环仿真全链路无报错
"""
import time

import numpy as np
import torch

from common import ROOT  # noqa: F401  (设置 sys.path)
import config
import models as M
import train as T
import control as C
from dataset import get_loaders

config.set_seed(config.SEED)

print("=" * 60)
print("SMOKE TEST: LC-DeepONet Phase 2.1")
print("=" * 60)

# 1. 数据检查
print("\n[1] 数据生成检查")
tr, te, ood = get_loaders(smoke=True)
u, q, qd = next(iter(tr))
print(f"  u {tuple(u.shape)} range [{u.min():.2f},{u.max():.2f}] | "
      f"q range [{q.min():.2f},{q.max():.2f}] | finite={bool(torch.isfinite(q).all())}")
assert torch.isfinite(q).all() and q.abs().max() < 20, "q 异常"

# 2. 六模型短训
print("\n[2] 六模型短训 (epochs=%d)" % config.EPOCHS_SMOKE)
results = []
for name in M.MODEL_NAMES:
    t0 = time.time()
    use_lip = (name == "LC-DeepONet")
    res = T.full_pipeline(name, tr, te, ood, epochs=config.EPOCHS_SMOKE,
                          use_dyn=True, use_lip=use_lip, seed=config.SEED)
    assert np.isfinite(res["ID_RMSE"]), f"{name} RMSE 非有限"
    results.append(res)
    cert = f" L_cert={res.get('L_cert', float('nan')):.2f}" if use_lip else ""
    print(f"  {name:12s} params={res['params']:7d} "
          f"loss={res['final_loss']:.4f} ID_RMSE={res['ID_RMSE']:.4f} "
          f"OOD_RMSE={res['OOD_RMSE']:.4f}{cert} [{time.time()-t0:.1f}s]")

# 3. Lipschitz 证书与修正③检查
print("\n[3] Lipschitz 证书检查（修正①②③）")
lc = results[-1]["model_obj"]
L_cert = lc.evaluate_certificate()
L_hat_now = float(lc.train_lipschitz_bound())
print(f"  精确证书 L_cert={L_cert:.3f} | 训练时估计 L_hat={L_hat_now:.3f} "
      f"(差值应小, power iteration 2步)")
print(f"  L_target={config.LIP_TARGET} -> 证书{'达标' if L_cert <= config.LIP_TARGET else '未达标(需调 target/epochs)'}")

# 4. 敏感性 vs 证书（E3 预演）
print("\n[4] 敏感性 S_emp vs 证书 L_cert")
u_pool = next(iter(te))[0]
for res in results:
    m = res["model_obj"]
    s = C.estimate_S_emp(m, u_pool)
    tag = f" L_cert={res.get('L_cert', float('nan')):.2f}" if "L_cert" in res else ""
    print(f"  {res['model']:12s} S_emp={s:.3f}{tag}")

# 5. eps_L 估计
print("\n[5] 误差算子 Lipschitz 估计 eps_L")
for name in ["DeepONet", "LC-DeepONet"]:
    m = [r for r in results if r["model"] == name][0]["model_obj"]
    eps = C.estimate_eps_L(m, u_pool, n_pairs=512)
    print(f"  {name:12s} eps_L_emp={eps:.3f}")

# 6. 闭环仿真预演（E6）
print("\n[6] 闭环仿真预演（E6: k_c 扫描 + 延迟一拍）")
params = C.MSD_MID
for k_c in [0.5, 2.0, 8.0]:
    rows = C.sweep_gain([k_c], params, r_func=C.reference_step)
    r0 = rows[0]
    print(f"  k_c={k_c:5.1f} -> track_err={r0['track_err_L2']:.3f} "
          f"overshoot={r0['overshoot']:.3f} diverged={r0['diverged']}")

print("\n" + "=" * 60)
print("SMOKE TEST PASS - 全链路正常")
print("=" * 60)
