# LC-DeepONet: Lipschitz-Constrained DeepONet with a Computable Gain Certificate

Official implementation and experiment code for:

> **Lipschitz-Constrained DeepONet for Nonlinear System Identification and
> Small-Gain Robust Feedback Stabilization**
> Peng Chen (ORCID iD `0009-0004-9021-6039`) — School of Computer Science,
> Xinyang University, Xinyang 464000, Henan, China
> (Draft v0.9, prepared for *Transactions of the Institute of Measurement and Control*;
> submission source `paper/main_sagej.tex` uses the official SAGE `sagej.cls`
> template with Sage Vancouver numbered citations)

Revision status: v0.8.1 implements the combined reviewer report
(`paper/review_v07_combined.md`) plus a pre-submission self-audit of every
claim against the code and result CSVs (audit table:
`submission/SUBMISSION_CHECKLIST.md` §7, response doc §8). Point-by-point
account: `paper/response_to_reviewers_v08.md`. Word audit: abstract 289 +
body 5685 = 5974 words (`paper/_wordcount.py`), inside the TIMC limit of
6000 words / 10 illustrations; `paper/main_v08.pdf` compiles to 26 pages with
0 errors / 0 Overfull / 0 undefined references. Submission package:
`submission/SUBMISSION_CHECKLIST.md`.

Theoretical chain: DeepONet identification → Lipschitz certificate
`L̂ = (∏ᵢ σ(Wᵢ)) · C_T` → residual gain bound `Lip(E) ≤ L̂ + ε_L` →
small-gain condition `L_C (L̂ + ε_L) < 1` → finite-gain robust stability.
The certificate is a **by-product of training** and is computed **exactly**
(SVD) at evaluation time — no extra data or optimization required.

Scope note: `L̂` is a **certified upper bound** on the learned model's gain.
The residual term `ε_L` is estimated empirically as a **sampled lower bound**
(`control.estimate_eps_L`), so the plant bound `L̂ + ε̂_L` and the closed-loop
statements built on it are **empirically calibrated, not formally certified**.
The a priori route (`Proposition 2` in the paper) is the only data-free,
box-covering bound; the a posteriori route and the closed-loop sweeps are
evaluated at the nominal parameter vector, with `run_e6_corners.py` extending
the check to all eight corners of the parameter box.

## Repository layout

```
LC-DeepONet/
├── src/
│   ├── config.py       # Global config (grid / parameter ranges / hyperparams / loss weights)
│   ├── systems.py      # Nonlinear systems + batched RK4 + random input signals
│   ├── dataset.py      # Data loaders (ID / test / parameter-OOD)
│   ├── models.py       # MLP / LSTM / Transformer / FNO / DeepONet / LC-DeepONet
│   ├── lipschitz.py    # Spectral norms (power iteration / SVD) + certificate L̂ (Frobenius & spectral C_T)
│   ├── losses.py       # L_data + λ1·L_dyn + λ2·ReLU(L̂ − L_target)²  (per-epoch surrogate logged)
│   ├── train.py        # Unified training / evaluation pipeline (returns history incl. L_hat)
│   └── control.py      # Closed-loop simulation, margin M, ε_L / S_emp estimation
├── experiments/        # One script per paper experiment (see mapping table below)
├── results/            # CSV results + EXPERIMENT_REPORT.md (every paper number is backed)
├── figures/            # Paper figures (PNG)
├── paper/              # main.tex (LaTeX source) + cover letter
└── submission/         # Compiled submission-side documents (cover letter PDF, highlights)
```

## Paper ↔ experiments ↔ results mapping

| Paper item | Script | Result file(s) |
|---|---|---|
| Table 1 (E1: 6 models × 5 seeds, ID) | `run_e1.py` | `e1_identification.csv` |
| E2: parameter-OOD (Fig. 1b) | `run_e2.py` | `e2_ood.csv`, `e2_ood_per_seed.csv` |
| E3: adversarial sensitivity S_emp (single / 5-seed) | `run_e3.py` / `run_e3b_multiseed.py` | `e3_sensitivity.csv`, `e3b_sensitivity_5seed.csv`, `e3b_sensitivity_per_seed.csv` |
| E4: test-time input noise 1–10% (single / 5-seed) | `run_e4.py` / `run_e4b_multiseed.py` | `e4_noise.csv`, `e4b_noise_5seed.csv` (aggregate, `*_mean`/`*_std` columns), `e4b_noise_5seed_per_seed.csv` (authoritative) |
| E5: ablation {Lip}×{Dyn} + λ_dyn sweep, 5-seed | `run_e5.py` / `run_e5b_multiseed.py` | `e5_ablation.csv`, `e5b_ablation_5seed.csv` |
| E6-A/B: closed-loop margin sweeps (one-step lag / delay) | `run_e6.py` / `run_e6_ab.py` | `e6_closedloop.csv`, `e6a_delayfree.csv`, `e6b_delay.csv`; design scalars in `e6_design_provenance.csv` |
| E6-C: parameter-box corner check (8 corners, 5 models) | `run_e6_corners.py` | `e6c_corners.csv`, `e6c_closedloop.csv` |
| E7: λ_lip invariance / L_target trade-off / λ_dyn check | `run_e7_lam.py` / `run_e7b_ltarget.py` / `run_e7c_lamdyn.py` | `e7_lam_sensitivity.csv`, `e7b_ltarget_sensitivity.csv`, `e7c_lamdyn.csv` |
| E8: transfer to Duffing + pendulum | `run_e8_transfer.py` | `e8_transfer.csv` |
| E9: long-horizon rollout (4 s / 6 s) | `run_e9_longhorizon.py` | `e9_longhorizon.csv`, `e9_rmse_t.csv` |
| Certificate audit: spectral-vs-Frobenius C_T, hinge-activation probe (M3/M4 of the internal review) | `run_m3_ctnorm.py` | `m3_ctnorm_per_seed.csv` (per-seed: the paper quotes sample std over the 5 seeds) |
| Figures (paper) | `make_figures.py` / `make_figures_p0.py` / `make_figures_p1.py` | → `figures/*.png` |

## Quick start

```bash
pip install -r requirements.txt

cd experiments
python smoke_test.py          # 1) end-to-end self-check (~1 min)
python run_all.py             # 2) full suite E1-E9 + M3 (several hours on CPU)
python summarize_results.py   # 3) aggregate all CSVs into one report
python make_figures.py        # 4) regenerate paper figures
python make_figures_p0.py
python make_figures_p1.py
```

All datasets are **generated deterministically** by the code (fixed seeds);
no external data are required. The CSV files in `results/` reproduce every
table and figure in the paper.

Two conventions worth knowing when reproducing numbers:

- **Aggregate vs per-seed files.** Files ending in `_per_seed.csv` hold the
  raw runs and are authoritative; the plain-named file holds the aggregated
  `mean`/`std`. Every `mean ± std` in the paper is the **sample** standard
  deviation (`ddof = 1`) over the stated number of seeds, which is the
  convention emitted by the aggregate files.
- **Where a number lives.** `e6_design_provenance.csv` records every scalar
  behind the closed-loop design (`L̂`, `ε̂_L`, `L̂+ε̂_L`, `k_c^safe`, `M`), so
  the E6 numbers can be recomputed from a single file without retraining.
- **The constraint is a loss penalty, not a normalisation, and it is inactive
  at the default target.** `LCDeepONet` shares the `DeepONetBase` architecture
  — there is no spectral-normalisation layer and no weight projection
  (`renormalize_weights()` exists but is never called). The Lipschitz
  constraint enters only as `λ_lip · ReLU(L̂_surrogate − L_target)²` in the
  loss. At the default `LIP_TARGET = 5` the power-iteration surrogate stays
  below 3.8 at every epoch (see `m3_ctnorm_per_seed.csv`,
  `hinge_active_epochs = 0`), so the hinge is identically zero and the default
  configuration is *unconstrained*; the penalty becomes binding, and the
  certificate shrinks by 62% at no accuracy cost, only when the target is
  tightened (E7, `L_target = 1.5`).

## Key implementation details

1. **√Δt input scaling** — the branch network receives `z = √Δt · u`, so the
   network's Euclidean Lipschitz constant corresponds to the discrete-L²
   operator-norm sense (`models.DeepONetBase.branch_out`).
2. **Value error ≠ Lipschitz error** — ε_L is estimated on the *error
   operator* (`control.estimate_eps_L`), never conflated with function-value
   error.
3. **Differentiable Lipschitz penalty** — training uses a power-iteration
   surrogate (`LCDeepONet.train_lipschitz_bound`); the evaluation certificate
   uses exact SVD (`LCDeepONet.evaluate_certificate`, under `no_grad`).
4. **Conservative by design** — the reported certificates use the
   Δt-weighted Frobenius trunk norm C_T; `run_m3_ctnorm.py` also measures the
   tighter spectral-norm variant (≈15% lower) and confirms the hinge penalty
   stays inactive at L_target = 5 (surrogate ∈ [1.5, 3.8] across all runs).

## Citation

If you find this code useful, please cite the accompanying paper
(bibTeX entry to be added upon publication).

## License

Released under the [MIT License](LICENSE).
