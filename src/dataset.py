"""PyTorch Dataset：把 numpy 数据转为张量，并提供 DataLoader。"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config


def to_loader(data, batch_size, shuffle, seed=0):
    u = torch.tensor(data["u"], dtype=torch.float32)
    q = torch.tensor(data["q"], dtype=torch.float32)
    qd = torch.tensor(data["qd"], dtype=torch.float32)
    ds = TensorDataset(u, q, qd)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def get_loaders(system="msd", smoke=False, seed=config.SEED):
    """返回 train / test_id / test_ood 三个 DataLoader。"""
    import systems
    if smoke:
        n_tr, n_te, n_ood = config.SMOKE_N_TRAIN, config.SMOKE_N_TEST, config.SMOKE_N_OOD
    else:
        n_tr, n_te, n_ood = config.N_TRAIN, config.N_TEST, config.N_OOD
    tr = systems.generate_dataset(system, n_tr, seed=seed)
    te = systems.generate_dataset(system, n_te, seed=seed + 1)
    ood = systems.generate_dataset(system, n_ood, seed=seed + 2, ood=True)
    return (to_loader(tr, config.BATCH, True, seed),
            to_loader(te, 256, False),
            to_loader(ood, 256, False))
