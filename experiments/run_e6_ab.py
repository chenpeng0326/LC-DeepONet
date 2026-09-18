"""E6-A / E6-B: delay-free (theory-exact) sweep + implementation-delay sweep.

E6-A: u(t) = k_c (r(t) - y(t))，无控制延迟 —— 与小增益分析（margin M）
      严格对应的闭环；理论边界 M=0 用竖线标出。
E6-B: u(t) = k_c (r(t) - y(t - tau))，tau ∈ {50, 100, 200} ms —— 实现延迟
      敏感性扩展实验（定理假设之外）。

k_c 网格：k_c_safe 的倍率（k_c_safe = 0.240978，来自 E6 正式 run 的
LC 证书设计；L_hat + eps_hat = 0.9 / k_c_safe = 3.7352）。

输出: results/e6a_delayfree.csv, results/e6b_delay.csv
用法: python run_e6_ab.py
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import control as C  # noqa: E402
import config  # noqa: E402

K_SAFE = 0.240978                       # E6 正式 run 的 LC 证书设计增益
LHE = 0.9 / K_SAFE                      # 证书设计的 L_hat + eps_hat = 3.7352
MULTS = [0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def one_run(k_c, delay_steps):
    t, q, r, u, div = C.closed_loop_sim(
        k_c, C.MSD_MID, C.reference_step, d_amp=0.0, delay_steps=delay_steps)
    err = C.tracking_error_l2(t, q, r)
    mask = t >= 0.55
    ov = float(np.nanmax(np.abs(q[mask] - r[mask]))) if not div else float("inf")
    return err, ov, div


def main():
    os.makedirs(RESULTS, exist_ok=True)
    k_grid = [m * K_SAFE for m in MULTS]

    # ---------- E6-A: delay-free ----------
    rows_a = []
    for m, k in zip(MULTS, k_grid):
        err, ov, div = one_run(k, delay_steps=1)
        rows_a.append({"mult": m, "k_c": k,
                       "M_margin": 1.0 - k * LHE,
                       "track_err_L2": err, "overshoot": ov, "diverged": div})
        print(f"[A] mult={m:6.2f}  k_c={k:7.4f}  M={1 - k * LHE:7.3f}  "
              f"err={err:8.4f}  ov={ov:7.4f}  div={div}")
    pd.DataFrame(rows_a).to_csv(os.path.join(RESULTS, "e6a_delayfree.csv"), index=False)

    # ---------- E6-B: implementation delay ----------
    rows_b = []
    for tau_ms in (50, 100, 200):
        for m, k in zip(MULTS, k_grid):
            err, ov, div = one_run(k, delay_steps=tau_ms)  # dt_sim=1ms
            rows_b.append({"tau_ms": tau_ms, "mult": m, "k_c": k,
                           "M_margin": 1.0 - k * LHE,
                           "track_err_L2": err, "overshoot": ov, "diverged": div})
            print(f"[B] tau={tau_ms:3d}ms mult={m:6.2f}  err={err:8.4f}  "
                  f"ov={ov:7.4f}  div={div}")
    pd.DataFrame(rows_b).to_csv(os.path.join(RESULTS, "e6b_delay.csv"), index=False)
    print("saved e6a_delayfree.csv / e6b_delay.csv")


if __name__ == "__main__":
    main()
