"""全局配置：LC-DeepONet 论文级实验 (Phase 2.1)
对应拟定书第七节 E1-E6 与第六节方法设计。
"""
import os

# ---------- 随机性 ----------
SEED = 42

# ---------- 时间网格 ----------
T_END = 2.0          # 时间区间 [0, T]（论文 Theorem 1 的 [0,T]）
N_POINTS = 64        # 离散点数 N
DT = T_END / N_POINTS            # Δt
SQRT_DT = DT ** 0.5              # √Δt（修正①：Branch 输入缩放）

# ---------- 主系统：非线性质量-弹簧-阻尼 ----------
# q'' + c q' + k q + alpha q^3 = u + d
MSD = {
    "c": (0.4, 0.8),
    "k": (0.8, 1.2),
    "alpha": (0.5, 1.2),
    "alpha_ood": (1.2, 1.5),     # E2 参数 OOD
}

# 验证系统
DUFFING = {"delta": (0.3, 0.6), "alpha_d": (0.8, 1.2), "beta_d": (2.0, 4.0),
           "beta_d_ood": (4.0, 5.0)}          # E8 参数 OOD（上界外推 25%）
PENDULUM = {"b": (0.15, 0.35), "g_over_l": (7.0, 10.0),
            "g_over_l_ood": (10.0, 12.5)}     # E8 参数 OOD（上界外推 25%）

# ---------- 随机输入信号 u(t) = sum a_i sin(w_i t + phi_i) ----------
U_N_HARM = 3
U_AMP = 1.0          # 每个谐波幅值 U(-1,1)
U_FREQ_RANGE = (0.2, 2.0)   # Hz，远低于采样率 32 Hz

# ---------- 数据规模 ----------
N_TRAIN = 512
N_TEST = 128
N_OOD = 128
SMOKE_N_TRAIN = 96
SMOKE_N_TEST = 32
SMOKE_N_OOD = 32

# ---------- 训练 ----------
LR = 1e-3
BATCH = 64
EPOCHS = 100          # 正式实验
EPOCHS_SMOKE = 5      # smoke test
N_SEEDS = 5           # 正式实验重复次数

# ---------- 损失权重（论文第四节）----------
LAMBDA_DYN = 0.1      # 动态一致性
LAMBDA_LIP = 1.0      # Lipschitz 罚项
LIP_TARGET = 5.0      # 证书目标 L_target
LAMBDA_SMOOTH = 0.0   # 平滑正则（默认关闭，保留接口）

# ---------- 路径 ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
