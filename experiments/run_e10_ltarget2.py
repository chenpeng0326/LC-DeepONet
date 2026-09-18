"""E10: LC-DeepONet at a BINDING target (L_target=2.0) — the constraint actually active.

Motivation (reviewer-facing): at the default L_target=5 the hinge is inactive, so the
main-table LC column differs from DeepONet only by the RNG stream. This experiment
re-runs the LC side at L_target=2.0, where the surrogate is driven to the target and
the certificate is *earned* by the penalty. Baselines (DeepONet / FNO / MLP / ...)
do not depend on L_target and are NOT re-run; comparisons reuse the existing CSVs.

Parts (all with config.LIP_TARGET=2.0, LAMBDA_LIP=1.0):
  A. MSD identification, 5 seeds -> e10_lt2_msd_per_seed.csv
     (extends e7b_ltarget_sensitivity.csv with per-seed records)
  B. E6-style closed-loop design from the binding certificate, using the
     median-L_cert seed's model -> e10_lt2_closedloop.csv + e10_lt2_design_provenance.csv
  C. E8-style transfer (duffing + pendulum), 5 seeds -> e10_lt2_transfer_per_seed.csv
     (+ aggregated e10_lt2_transfer.csv)

Incremental saving: per-seed rows are flushed to disk after every training, so the
run is resumable / partially inspectable. The E10a smoke mode (epochs=5, 1 seed)
validates the plumbing without touching the正式 outputs (suffix _smoke).

Usage: python run_e10_ltarget2.py [--smoke] [--skip-transfer]
"""
import sys
import os

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import save_csv  # noqa: E402
import config  # noqa: E402
import train as T  # noqa: E402
import control as C  # noqa: E402
from dataset import get_loaders  # noqa: E402

smoke = "--smoke" in sys.argv
skip_transfer = "--skip-transfer" in sys.argv
epochs = config.EPOCHS_SMOKE if smoke else config.EPOCHS
seeds = [config.SEED] if smoke else [config.SEED + i for i in range(config.N_SEEDS)]
suffix = "_smoke" if smoke else ""

config.LIP_TARGET = 2.0   # binding target: surrogate is driven to ~2.0 during training
config.LAMBDA_LIP = 1.0   # same weight as E7b sweep

LT2 = config.LIP_TARGET
print(f"[E10] L_target={LT2} (binding), epochs={epochs}, seeds={seeds}", flush=True)

# ----------------------------------------------------------------------
# Part A: MSD identification, 5 seeds
# ----------------------------------------------------------------------
msd_rows = []
design_model = None
design_u_pool = None
design_Lcert = None

for s in seeds:
    config.set_seed(s)
    tr, te, ood = get_loaders(smoke=smoke)
    res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                          use_lip=True, seed=s)
    msd_rows.append({"L_target": LT2, "seed": s,
                     "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"],
                     "L_cert": res.get("L_cert", float("nan")),
                     "train_time_s": res["train_time_s"]})
    print(f"  [A] seed={s}: ID={res['ID_RMSE']:.4f} OOD={res['OOD_RMSE']:.4f} "
          f"L_cert={res.get('L_cert', float('nan')):.3f}", flush=True)
    save_csv(pd.DataFrame(msd_rows), f"e10_lt2_msd_per_seed{suffix}.csv")

    # keep the model whose L_cert is the running median (design model, part B)
    certs = [r["L_cert"] for r in msd_rows]
    med = sorted(certs)[len(certs) // 2]
    if res.get("L_cert") == med and design_model is None:
        design_model = res["model_obj"]
        design_u_pool = next(iter(te))[0]
        design_Lcert = res["L_cert"]

if design_model is None:  # fallback: last model
    design_model = res["model_obj"]
    design_u_pool = next(iter(te))[0]
    design_Lcert = res.get("L_cert", float("nan"))

if not smoke:
    dfm = pd.DataFrame(msd_rows)
    print("[A] summary:",
          dfm[["ID_RMSE", "OOD_RMSE", "L_cert"]].agg(["mean", "std"]).to_string(),
          flush=True)

# ----------------------------------------------------------------------
# Part B: closed-loop design from the BINDING certificate (E6 protocol)
# ----------------------------------------------------------------------
L_cert = design_model.evaluate_certificate()
eps_L = C.estimate_eps_L(design_model, design_u_pool, n_pairs=256)

# real plant gain (numerical Lipschitz ratio, same paired-permutation protocol as E6)
from control import true_G_batch  # noqa: E402
g = torch.Generator().manual_seed(1)
idx = torch.randint(0, design_u_pool.shape[0], (512,), generator=g)
up = design_u_pool[idx]
Gy = true_G_batch(up)
perm = torch.randperm(len(up), generator=g)
du = up[perm] - up
dg = Gy[perm] - Gy
num = torch.sqrt(config.DT * (dg ** 2).sum(dim=1))
den = torch.sqrt(config.DT * (du ** 2).sum(dim=1)).clamp_min(1e-8)
L_G_true = float((num / den).max())

k_c_safe = 0.9 / (L_cert + eps_L)
print(f"\n[B] binding L_cert={L_cert:.3f} eps_L={eps_L:.3f} L_G_true={L_G_true:.3f} "
      f"k_c_safe={k_c_safe:.3f} | coverage: "
      f"{'SAFE' if L_cert + eps_L >= L_G_true else 'INSUFFICIENT'}", flush=True)

rows = []
for mult in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
    k_c = k_c_safe * mult
    sim = C.sweep_gain([k_c], C.MSD_MID, r_func=C.reference_step, d_amp=0.05,
                       delay_override=100)[0]
    M_margin = 1.0 - k_c * (L_cert + eps_L)
    rows.append({"mult": mult, "k_c": k_c, "M_margin": M_margin,
                 "track_err_L2": sim["track_err_L2"],
                 "overshoot": sim["overshoot"], "diverged": sim["diverged"],
                 "design": "LC-binding-certificate x mult"})
dfb = pd.DataFrame(rows)
save_csv(dfb, f"e10_lt2_closedloop{suffix}.csv")

pd.DataFrame([{
    "L_target": LT2, "seed_design": getattr(design_model, "seed", None),
    "epochs": epochs, "L_cert": L_cert, "eps_L_hat": eps_L,
    "plant_bound": L_cert + eps_L, "k_c_safe": k_c_safe,
    "M_margin": 1.0 - k_c_safe * (L_cert + eps_L),
    "L_G_true_lower_bound": L_G_true,
    "covered": bool(L_cert + eps_L >= L_G_true),
    "n_pairs_epsL": 256, "sweep_delay_steps": 100, "sweep_d_amp": 0.05,
}]).to_csv(os.path.join(config.RESULTS_DIR, f"e10_lt2_design_provenance{suffix}.csv"),
           index=False)

safe_rows = dfb[(dfb["M_margin"] > 0)]
unsafe_rows = dfb[(dfb["M_margin"] < 0)]
print(dfb.to_string(index=False))
print(f"\n[B summary] M>0: {len(safe_rows)} pts, diverged {int(safe_rows['diverged'].sum())} | "
      f"M<0: {len(unsafe_rows)} pts, diverged {int(unsafe_rows['diverged'].sum())}", flush=True)

# ----------------------------------------------------------------------
# Part C: transfer benchmark (duffing + pendulum), LC only, 5 seeds
# ----------------------------------------------------------------------
if not skip_transfer:
    tr_rows = []
    for system in ["duffing", "pendulum"]:
        for s in seeds:
            config.set_seed(s)
            tr, te, ood = get_loaders(system=system, smoke=smoke)
            res = T.full_pipeline("LC-DeepONet", tr, te, ood, epochs=epochs,
                                  use_lip=True, seed=s)
            rec = {"L_target": LT2, "system": system, "seed": s,
                   "ID_RMSE": res["ID_RMSE"], "OOD_RMSE": res["OOD_RMSE"]}
            if "L_cert" in res:
                rec["L_cert"] = res["L_cert"]
            tr_rows.append(rec)
            print(f"  [C] {system:8s} seed={s} ID={res['ID_RMSE']:.4f} "
                  f"OOD={res['OOD_RMSE']:.4f} "
                  f"L_cert={res.get('L_cert', float('nan')):.3f}", flush=True)
            save_csv(pd.DataFrame(tr_rows), f"e10_lt2_transfer_per_seed{suffix}.csv")

    dft = pd.DataFrame(tr_rows)
    agg = dft.groupby(["L_target", "system"]).agg(
        ID_mean=("ID_RMSE", "mean"), ID_std=("ID_RMSE", "std"),
        OOD_mean=("OOD_RMSE", "mean"), OOD_std=("OOD_RMSE", "std"),
        L_cert_mean=("L_cert", "mean"), L_cert_std=("L_cert", "std"),
        n=("seed", "count")).reset_index()
    save_csv(agg, f"e10_lt2_transfer{suffix}.csv")
    print("[C] aggregated:\n" + agg.to_string(index=False), flush=True)

print("\n[E10] done.", flush=True)
