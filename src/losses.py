"""四项损失（论文第四节）：

L = L_data + lambda_dyn * L_dyn + lambda_lip * max(0, L_hat - L_target)^2 + lambda_smooth * L_smooth

关键实现要点：
- L_dyn：网络输出与真实 q_dot 的有限差分一致性（physics/dynamics-aware）
- L_lip：可反传的张量版 L_hat（修正③：绝不能用 no_grad 版本进 loss）
"""
import torch

import config


def loss_data(y_hat, q):
    return torch.mean((y_hat - q) ** 2)


def loss_dyn(y_hat, qd):
    """d/dt y_hat vs q_dot 的有限差分一致性（内部点）。"""
    dy = (y_hat[:, 2:] - y_hat[:, :-2]) / (2 * config.DT)
    return torch.mean((dy - qd[:, 1:-1]) ** 2)


def loss_lip(L_hat_tensor, lip_target):
    """ReLU(L_hat - L_target)^2 —— L_hat_tensor 必须带计算图。"""
    return torch.relu(L_hat_tensor - lip_target) ** 2


def loss_smooth(y_hat):
    d2 = y_hat[:, 2:] - 2 * y_hat[:, 1:-1] + y_hat[:, :-2]
    return torch.mean(d2 ** 2)


def total_loss(model, y_hat, q, qd, use_dyn=True, use_lip=False,
               lip_target=None, use_smooth=False):
    cfg = config
    lip_target = cfg.LIP_TARGET if lip_target is None else lip_target
    l_data = loss_data(y_hat, q)
    total = l_data
    parts = {"data": l_data}
    if use_dyn:
        l_d = loss_dyn(y_hat, qd)
        total = total + cfg.LAMBDA_DYN * l_d
        parts["dyn"] = l_d
    if use_lip:
        L_hat = model.train_lipschitz_bound()   # 可反传（修正③）
        l_l = loss_lip(L_hat, lip_target)
        total = total + cfg.LAMBDA_LIP * l_l
        parts["lip"] = l_l
        parts["L_hat"] = L_hat.detach()
    if use_smooth and cfg.LAMBDA_SMOOTH > 0:
        l_s = loss_smooth(y_hat)
        total = total + cfg.LAMBDA_SMOOTH * l_s
        parts["smooth"] = l_s
    return total, parts
