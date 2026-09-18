"""Lipschitz 证书工具（论文 Theorem 1 / 修正①②③）。

- 训练时：power iteration 得到可微谱范数估计 sigma_hat（保留计算图，进 loss）——修正③
- 评估时：精确 SVD 谱范数（no_grad），证书严格成立
- 证书：Lip(B) = prod_i sigma(W_i)（1-Lipschitz 激活 ReLU）
        C_T = sqrt(DT * sum_j ||T(t_j)||^2)   （Trunk 输出的离散 L2 范数）
        L_hat = Lip(B) * C_T
"""
import torch
import torch.nn as nn

import config


def power_iteration_sigma(W: torch.Tensor, iters: int = 2) -> torch.Tensor:
    """可微谱范数估计（训练用，修正③：必须保留计算图）。"""
    with torch.no_grad():
        v = torch.randn(W.shape[1], 1, device=W.device)
    for _ in range(iters):
        u = torch.nn.functional.normalize(W @ v, dim=0)
        v = torch.nn.functional.normalize(W.T @ u, dim=0)
    sigma = (u.T @ W @ v).squeeze()
    return sigma.clamp_min(1e-8)


def exact_sigma(W: torch.Tensor) -> float:
    """精确谱范数（评估/证书用）。"""
    with torch.no_grad():
        return float(torch.linalg.matrix_norm(W.detach(), ord=2))


def _as_grid(t):
    """统一为 [N,1] 查询网格形状。"""
    return t.unsqueeze(-1) if t.dim() == 1 else t


@torch.no_grad()
def trunk_l2_norm(trunk: nn.Module, t_grid: torch.Tensor) -> torch.Tensor:
    """C_T = sqrt(DT * sum_j ||T(t_j)||_2^2)（评估版，no_grad）。t_grid [N]。"""
    Tt = trunk(_as_grid(t_grid))            # [N, p]
    return torch.sqrt(config.DT * (Tt ** 2).sum()).item()


def trunk_l2_norm_diff(trunk: nn.Module, t_grid: torch.Tensor) -> torch.Tensor:
    """C_T 可微版（训练时进 loss，修正③）。"""
    Tt = trunk(_as_grid(t_grid))
    return torch.sqrt(config.DT * (Tt ** 2).sum()).clamp_min(1e-8)


@torch.no_grad()
def certificate(model) -> float:
    """评估 LC-DeepONet 的严格 Lipschitz 证书 L_hat（精确 SVD）。"""
    if not hasattr(model, "branch_linears"):
        raise ValueError("certificate() 仅支持带 branch_linears 的 LC-DeepONet")
    lip_b = 1.0
    for lin in model.branch_linears:
        lip_b *= exact_sigma(lin.weight)
    c_t = trunk_l2_norm(model.trunk, model.t_grid)
    return float(lip_b * c_t)


@torch.no_grad()
def renormalize_weights(model, s_target: float = 1.0):
    """可选后处理：把超过 s_target 的权重谱收缩到 s_target，使证书严格达标。
    返回 renorm 前后的证书。"""
    before = certificate(model)
    with torch.no_grad():
        for lin in model.branch_linears:
            sig = exact_sigma(lin.weight)
            if sig > s_target and sig > 0:
                lin.weight.mul_(s_target / sig)
    after = certificate(model)
    return before, after
