"""六个模型统一接口：forward(u [B,N]) -> y_hat [B,N]。

- MLP / LSTM / Transformer / FNO1d : 基线
- DeepONet : 基线算子（输入同样做 sqrt(DT) 缩放，保证与 LC 公平对比）
- LCDeepONet : 谱范数可微约束 + sqrt(DT) 缩放（修正①③）
"""
import math

import torch
import torch.nn as nn

import config


# ---------------- 基线 1: MLP（flatten 序列） ----------------

class MLPModel(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        n = config.N_POINTS
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n),
        )

    def forward(self, u):
        return self.net(u)


# ---------------- 基线 2: LSTM ----------------

class LSTMModel(nn.Module):
    def __init__(self, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, u):
        x = u.unsqueeze(-1)                    # [B,N,1]
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)      # [B,N]


# ---------------- 基线 3: Transformer ----------------

class TransformerModel(nn.Module):
    def __init__(self, d_model=96, nhead=4, layers=3, ff=256):
        super().__init__()
        n = config.N_POINTS
        self.inp = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(n, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff,
                                         batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, u):
        x = self.inp(u.unsqueeze(-1)) + self.pos.unsqueeze(0)
        return self.head(self.enc(x)).squeeze(-1)


# ---------------- 基线 4: FNO1d ----------------

class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / math.sqrt(in_ch * out_ch)
        self.weight = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, x_ft.shape[1], N // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weight)
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, width=48, modes=12, n_layers=4):
        super().__init__()
        self.lift = nn.Linear(1, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, u):
        x = self.lift(u.unsqueeze(-1)).transpose(1, 2)   # [B,C,N]
        for sc, sk in zip(self.spectral, self.skip):
            x = torch.nn.functional.gelu(sc(x) + sk(x))
        return self.proj(x.transpose(1, 2)).squeeze(-1)


# ---------------- DeepONet 骨干（Branch + Trunk 内积层） ----------------

class DeepONetBase(nn.Module):
    """y_hat(t_j) = sum_k B_k(u) T_k(t_j) + b
    Branch 输入做 sqrt(DT) 缩放：z = sqrt(DT) * u（修正①，
    使欧氏 Lipschitz 常数直接对应 L2 算子范数意义）。"""

    def __init__(self, p=64, width=256):
        super().__init__()
        n = config.N_POINTS
        self.p = p
        self.t_grid = torch.linspace(0.0, 1.0, n)   # 归一化查询时间
        self.branch = nn.Sequential(
            nn.Linear(n, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, p),
        )
        self.bias = nn.Parameter(torch.zeros(n))

    def branch_out(self, u):
        z = config.SQRT_DT * u                      # sqrt(DT) 缩放（修正①）
        return self.branch(z)                       # [B,p]

    def trunk_out(self):
        t = self.t_grid.unsqueeze(-1)               # [N,1]
        return self.trunk(t)                        # [N,p]

    def forward(self, u):
        B = self.branch_out(u)
        Tt = self.trunk_out()                       # [N,p]
        return B @ Tt.T + self.bias                 # [B,N]


# ---------------- LC-DeepONet（本文方法） ----------------

import lipschitz as lip


class LCDeepONet(DeepONetBase):
    """在 DeepONet 基础上：
    1) branch_linears 显式登记（证书计算用）
    2) 提供 train_lipschitz_bound()：可反传的 L_hat（修正③）
    3) 提供 evaluate_certificate()：精确 SVD 严格证书
    """

    def __init__(self, p=64, width=256, pi_iters=4):
        super().__init__(p=p, width=width)
        self.pi_iters = pi_iters
        # 登记第 0/2/4 层 Linear（激活 ReLU 为 1-Lipschitz）
        self.branch_linears = [self.branch[0], self.branch[2], self.branch[4]]

    def train_lipschitz_bound(self) -> torch.Tensor:
        """可微证书 L_hat = prod sigma_hat(W_i) * C_T（保留计算图）。"""
        lip_b = None
        for lin in self.branch_linears:
            s = lip.power_iteration_sigma(lin.weight, self.pi_iters)
            lip_b = s if lip_b is None else lip_b * s
        c_t = lip.trunk_l2_norm_diff(self.trunk, self.t_grid.unsqueeze(-1))
        return lip_b * c_t

    @torch.no_grad()
    def evaluate_certificate(self) -> float:
        return lip.certificate(self)


# ---------------- 工厂 ----------------

MODEL_NAMES = ["MLP", "LSTM", "Transformer", "FNO", "DeepONet", "LC-DeepONet"]


def build_model(name: str):
    if name == "MLP":
        return MLPModel()
    if name == "LSTM":
        return LSTMModel()
    if name == "Transformer":
        return TransformerModel()
    if name == "FNO":
        return FNO1d()
    if name == "DeepONet":
        return DeepONetBase()
    if name == "LC-DeepONet":
        return LCDeepONet()
    raise ValueError(f"unknown model {name}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
