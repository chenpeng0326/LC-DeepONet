"""统一训练循环 + 评估工具。

消融配置（论文 E5）：
- A: use_lip=False, use_dyn=False  （普通 DeepONet）
- B: use_lip=True,  use_dyn=False  （仅 Lipschitz 约束）
- C: use_lip=False, use_dyn=True   （仅动态损失）
- D: use_lip=True,  use_dyn=True   （完整 LC-DeepONet）
"""
import time

import numpy as np
import torch

import config
import models as M
from losses import total_loss


def train_model(model, train_loader, epochs=None, use_dyn=True, use_lip=False,
                seed=config.SEED, verbose=False):
    epochs = config.EPOCHS if epochs is None else epochs
    config.set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    history = []
    model.train()
    for ep in range(epochs):
        ep_loss, ep_parts, nb = 0.0, {}, 0
        for u, q, qd in train_loader:
            opt.zero_grad()
            y_hat = model(u)
            loss, parts = total_loss(model, y_hat, q, qd,
                                     use_dyn=use_dyn, use_lip=use_lip)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach())
            for k, v in parts.items():
                ep_parts[k] = ep_parts.get(k, 0.0) + float(v.detach())
            nb += 1
        rec = {"epoch": ep, "loss": ep_loss / max(nb, 1)}
        rec.update({k: v / max(nb, 1) for k, v in ep_parts.items()})
        history.append(rec)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            extra = f" L_hat={rec.get('L_hat', float('nan')):.3f}" if "L_hat" in rec else ""
            print(f"  ep{ep:03d} loss={rec['loss']:.5f}{extra}")
    return history


@torch.no_grad()
def evaluate(model, loader):
    """返回 RMSE / MAE / NRMSE / 推理时间(ms/样本)。"""
    model.eval()
    se, ae, sq = 0.0, 0.0, 0.0
    n = 0
    t0 = time.perf_counter()
    for u, q, _ in loader:
        y_hat = model(u)
        se += float(((y_hat - q) ** 2).sum())
        ae += float((y_hat - q).abs().sum())
        sq += float((q ** 2).sum())
        n += q.numel()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / n
    rmse = (se / n) ** 0.5
    mae = ae / n
    nrmse = rmse / ((sq / n) ** 0.5 + 1e-12)
    return {"RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "infer_ms": dt_ms}


@torch.no_grad()
def predict(model, loader):
    """返回全部 (u, q, y_hat) 拼接张量（画图用）。"""
    model.eval()
    us, qs, ys = [], [], []
    for u, q, _ in loader:
        us.append(u)
        qs.append(q)
        ys.append(model(u))
    return torch.cat(us), torch.cat(qs), torch.cat(ys)


def full_pipeline(model_name, train_loader, test_loader, ood_loader,
                  epochs=None, use_dyn=True, use_lip=False, seed=config.SEED,
                  verbose=False):
    """构建模型 -> 训练 -> 评估 ID/OOD -> 计算证书。返回 dict。"""
    model = M.build_model(model_name)
    n_params = M.count_params(model)
    t0 = time.time()
    hist = train_model(model, train_loader, epochs=epochs, use_dyn=use_dyn,
                       use_lip=use_lip, seed=seed, verbose=verbose)
    train_time = time.time() - t0
    res = {"model": model_name, "params": n_params, "train_time_s": train_time,
           "seed": seed, "final_loss": hist[-1]["loss"]}
    res.update({f"ID_{k}": v for k, v in evaluate(model, test_loader).items()})
    res.update({f"OOD_{k}": v for k, v in evaluate(model, ood_loader).items()})
    if use_lip and hasattr(model, "evaluate_certificate"):
        res["L_cert"] = model.evaluate_certificate()
    res["model_obj"] = model
    res["history"] = hist
    return res
