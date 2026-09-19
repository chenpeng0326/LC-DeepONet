"""E6-S: seed-to-seed reproducibility of the small-gain design rule.

Motivation (reviewer-facing, R2-M1): the E6-A/E6-B closed-loop sweeps and the
E6-C per-corner loops are each a *single* design implementation (one trained
network -> one k_c^safe). A referee can ask whether the reported margin-to-
behaviour picture is an artefact of that one draw. This script re-runs the whole
E6 design pipeline end to end for the five seeds already used in E6-C:

  train LC-DeepONet(s) and DeepONet(s)
    -> L_cert(s), eps_L(s) -> b(s) = L_cert + eps_L -> k_c^safe(s) = 0.9 / b(s)
    -> closed-loop sweep k_c = k_c^safe(s) * mult, mult in MULTS
       at delay_steps in {1 (E6-A), 100 (E6-C protocol)}
    -> DeepONet optimistic design k_c^opt(s) = 0.9 / S_emp(s), same two delays

Two margins are recorded for every point:
  M_design = 1 - k_c * b(s)      <- what the designer believes (seed-independent
                                    in the per-b sweep, since k_c scales with b)
  M_true   = 1 - k_c * L_G_true  <- the real small-gain margin, using the numerical
                                    plant-gain lower bound L_G_true (seed-free scalar)
M_true is the diagnostic that matters when asking whether the optimistic
(uncertified) design actually violates the condition it claims to satisfy.

Outputs (incremental, resumable):
- results/e6s_design_per_seed.csv        per-seed design scalars
- results/e6s_closedloop_per_seed.csv    long-form sweep records
- results/e6s_summary.csv                mean/std over seeds per (design, delay, mult)

Usage: python run_e6_seeds.py [--smoke]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import save_csv  # noqa: E402
import config  # noqa: E402
import train as T  # noqa: E402
import control as C  # noqa: E402
from dataset import get_loaders  # noqa: E402

smoke = "--smoke" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
n_seeds = 1 if smoke else config.N_SEEDS
n_pairs = 64 if smoke else 256
suffix = "_smoke" if smoke else ""

MULTS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
DELAYS = [1, 100]          # 1 step = E6-A (certificate-consistent), 100 = E6-B/C
D_AMP = 0.05

# ---- shared data pool (dataset generation fixed at the E6/E6-C seed) ----
config.set_seed(config.SEED)
tr, te, ood = get_loaders(smoke=smoke)
u_pool = next(iter(te))[0]

# ---- real plant gain at the nominal vector: seed-free reference scalar ----
# The sampling protocol is copied verbatim from run_e6.py (512 draws with
# replacement from the 64-sample pool, generator seed 1, paired permutation),
# so that the M_true column here is directly comparable with the L_G_true value
# recorded in results/e6_design_provenance.csv. A larger draw is evaluated as
# well to show how much the estimate moves with the sample size.
from control import true_G_batch  # noqa: E402


def plant_gain_lower_bound(n_draw, gen_seed=1):
    gg = torch.Generator().manual_seed(gen_seed)
    ii = torch.randint(0, u_pool.shape[0], (n_draw,), generator=gg)
    uu = u_pool[ii]
    Gyy = true_G_batch(uu)
    pp = torch.randperm(len(uu), generator=gg)
    ddu = uu[pp] - uu
    ddg = Gyy[pp] - Gyy
    nn = torch.sqrt(config.DT * (ddg ** 2).sum(dim=1))
    dd = torch.sqrt(config.DT * (ddu ** 2).sum(dim=1)).clamp_min(1e-8)
    return float((nn / dd).max())


L_G_true = plant_gain_lower_bound(512)          # == E6 protocol
L_G_true_big = plant_gain_lower_bound(4096)     # larger draw, robustness check
print(f"[E6-S] L_G_true (nominal, E6 protocol 512 draws) = {L_G_true:.4f} | "
      f"4096 draws = {L_G_true_big:.4f}", flush=True)

design_rows = []
cl_rows = []

for s_off in range(n_seeds):
    seed = config.SEED + s_off

    res_lc = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                             use_lip=True, seed=seed)
    res_do = T.full_pipeline("DeepONet", tr, te, ood, epochs=epochs,
                             use_lip=False, seed=seed)
    model_lc = res_lc["model_obj"]
    model_do = res_do["model_obj"]

    L_cert = float(model_lc.evaluate_certificate())
    eps_L = C.estimate_eps_L(model_lc, u_pool, n_pairs=n_pairs)
    b = L_cert + eps_L
    k_safe = 0.9 / b
    S_emp = C.estimate_S_emp(model_do, u_pool)
    k_opt = 0.9 / S_emp

    design_rows.append({
        "seed": seed, "epochs": epochs,
        "ID_RMSE": res_lc["ID_RMSE"], "OOD_RMSE": res_lc["OOD_RMSE"],
        "L_cert": L_cert, "eps_L_hat": eps_L, "plant_bound": b,
        "k_c_safe": k_safe,
        "M_design_at_k_safe": 1.0 - k_safe * b,
        "M_true_at_k_safe": 1.0 - k_safe * L_G_true,
        "S_emp_DeepONet": S_emp, "k_c_opt_DeepONet": k_opt,
        "M_design_DeepONet_optimistic": 1.0 - k_opt * S_emp,
        "M_true_at_k_opt": 1.0 - k_opt * L_G_true,
        "L_G_true": L_G_true, "L_G_true_4096": L_G_true_big,
        "M_true_big_at_k_safe": 1.0 - k_safe * L_G_true_big,
        "M_true_big_at_k_opt": 1.0 - k_opt * L_G_true_big,
    })
    save_csv(pd.DataFrame(design_rows), f"e6s_design_per_seed{suffix}.csv")
    print(f"  seed={seed}: L_cert={L_cert:.3f} eps_L={eps_L:.4f} b={b:.3f} "
          f"k_safe={k_safe:.4f} (M_true={1 - k_safe * L_G_true:+.3f}) | "
          f"S_emp={S_emp:.4f} k_opt={k_opt:.4f} "
          f"(M_true={1 - k_opt * L_G_true:+.3f})", flush=True)

    # ---- closed-loop sweeps from the LC certificate design ----
    for delay in DELAYS:
        for mult in MULTS:
            k = k_safe * mult
            sim = C.sweep_gain([k], C.MSD_MID, r_func=C.reference_step,
                               d_amp=D_AMP, delay_override=delay)[0]
            cl_rows.append({"seed": seed, "design": "LC-certificate", "delay_steps": delay,
                            "mult": mult, "k_c": k, "M_design": 1.0 - k * b,
                            "M_true": 1.0 - k * L_G_true,
                            "M_true_big": 1.0 - k * L_G_true_big,
                            "track_err_L2": sim["track_err_L2"],
                            "overshoot": sim["overshoot"], "diverged": sim["diverged"]})
        # ---- optimistic (uncertified) DeepONet design point ----
        sim = C.sweep_gain([k_opt], C.MSD_MID, r_func=C.reference_step,
                           d_amp=D_AMP, delay_override=delay)[0]
        cl_rows.append({"seed": seed, "design": "DeepONet-optimistic", "delay_steps": delay,
                        "mult": np.nan, "k_c": k_opt,
                        "M_design": 1.0 - k_opt * S_emp,
                        "M_true": 1.0 - k_opt * L_G_true,
                        "M_true_big": 1.0 - k_opt * L_G_true_big,
                        "track_err_L2": sim["track_err_L2"],
                        "overshoot": sim["overshoot"], "diverged": sim["diverged"]})
    save_csv(pd.DataFrame(cl_rows), f"e6s_closedloop_per_seed{suffix}.csv")

df_cl = pd.DataFrame(cl_rows)
agg = (df_cl.groupby(["design", "delay_steps", "mult"], dropna=False)
       .agg(n=("seed", "count"),
            k_c_mean=("k_c", "mean"), k_c_std=("k_c", "std"),
            M_design_mean=("M_design", "mean"),
            M_true_mean=("M_true", "mean"), M_true_std=("M_true", "std"),
            M_true_big_mean=("M_true_big", "mean"), M_true_big_std=("M_true_big", "std"),
            err_mean=("track_err_L2", "mean"), err_std=("track_err_L2", "std"),
            err_min=("track_err_L2", "min"), err_max=("track_err_L2", "max"),
            diverged=("diverged", "sum"))
       .reset_index())
save_csv(agg, f"e6s_summary{suffix}.csv")

print("\n[E6-S design spread]")
dfd = pd.DataFrame(design_rows)
print(dfd[["L_cert", "plant_bound", "k_c_safe", "k_c_opt_DeepONet",
            "M_true_at_k_safe", "M_true_at_k_opt",
            "M_true_big_at_k_safe", "M_true_big_at_k_opt"]]
      .agg(["mean", "std", "min", "max"]).to_string())
print(f"\n[E6-S] optimistic design violates the small-gain condition at the true "
      f"plant gain in {int((dfd['M_true_at_k_opt'] < 0).sum())}/{len(dfd)} seeds "
      f"(certificate design: {int((dfd['M_true_at_k_safe'] > 0).sum())}/{len(dfd)} satisfy it)")

for delay in DELAYS:
    sub = agg[agg["delay_steps"] == delay]
    piv = sub.pivot_table(index="mult", columns="design", values="err_mean").reset_index()
    print(f"\n[E6-S] mean track_err_L2 over {n_seeds} seeds, delay={delay} step(s)")
    print(piv.to_string(index=False))

print("\n[E6-S] done.", flush=True)
