"""非线性系统定义 + RK4 批量积分 + 随机输入信号生成。

主系统（论文实验主对象）：非线性质量-弹簧-阻尼
    q'' + c q' + k q + alpha q^3 = u(t) + d,  c,k,alpha 参数不确定
验证系统：Duffing 振荡器；补充系统：非线性摆。
"""
import numpy as np

import config


# ---------- 系统动力学（状态 x = [q, q_dot]）----------

def msd_rhs(x, u, c, k, alpha):
    """质量-弹簧-阻尼: q'' = u - c q' - k q - alpha q^3"""
    q, qd = x[..., 0], x[..., 1]
    return np.stack([qd, u - c * qd - k * q - alpha * q ** 3], axis=-1)


def duffing_rhs(x, u, delta, alpha_d, beta_d):
    """Duffing: x'' = u - delta x' - alpha_d x - beta_d x^3"""
    q, qd = x[..., 0], x[..., 1]
    return np.stack([qd, u - delta * qd - alpha_d * q - beta_d * q ** 3], axis=-1)


def pendulum_rhs(x, u, b, g_over_l):
    """非线性摆: theta'' = u - b theta' - (g/l) sin(theta)"""
    q, qd = x[..., 0], x[..., 1]
    return np.stack([qd, u - b * qd - g_over_l * np.sin(q)], axis=-1)


SYSTEMS = {
    "msd": (msd_rhs, ("c", "k", "alpha")),
    "duffing": (duffing_rhs, ("delta", "alpha_d", "beta_d")),
    "pendulum": (pendulum_rhs, ("b", "g_over_l")),
}


# ---------- 批量 RK4 ----------

def rk4_batch(rhs, x0, u_func, t_dense, params_cols):
    """向量化 RK4。rhs(x,u,**params)，x [B,2]，u_func(t)->[B]，t_dense [Tn]。
    params_cols: 每个参数一个 [B] 数组（按样本广播）。
    返回 x_traj [B,Tn,2]。"""
    B = x0.shape[0]
    Tn = len(t_dense)
    traj = np.empty((B, Tn, 2), dtype=np.float64)
    x = x0.copy()
    traj[:, 0] = x
    dt = t_dense[1] - t_dense[0]
    for j in range(Tn - 1):
        t = t_dense[j]
        u0 = u_func(t)
        k1 = rhs(x, u0, *params_cols)
        k2 = rhs(x + 0.5 * dt * k1, u_func(t + 0.5 * dt), *params_cols)
        k3 = rhs(x + 0.5 * dt * k2, u_func(t + 0.5 * dt), *params_cols)
        k4 = rhs(x + dt * k3, u_func(t + dt), *params_cols)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[:, j + 1] = x
    return traj


# ---------- 随机输入信号（解析式，任意 t 可求值）----------

def sample_inputs(rng, B):
    """采样 B 个随机正弦和输入的系数。返回 dict of arrays。
    带宽限制在 FREQ_RANGE 内，远低于 32 Hz 采样率的奈奎斯特频率。"""
    a = rng.uniform(-config.U_AMP, config.U_AMP, size=(B, config.U_N_HARM))
    f = rng.uniform(*config.U_FREQ_RANGE, size=(B, config.U_N_HARM))
    phi = rng.uniform(0, 2 * np.pi, size=(B, config.U_N_HARM))
    return {"a": a, "f": f, "phi": phi}


def make_u_func(sig):
    a, f, phi = sig["a"], sig["f"], sig["phi"]

    def u_func(t):
        w = 2 * np.pi * f * t  # [B,H] 广播 t 标量
        return (a * np.sin(w + phi)).sum(axis=1)

    return u_func


# ---------- 数据集生成 ----------

def simulate_system(system_name, params_list, sig_list, rng):
    """对一个样本批次做 RK4 仿真，并在 N 个均匀采样点输出 u, q, q_dot。
    params_list: list of tuple（与 SYSTEMS[system_name] 参数名对应，逐样本）
    返回 u [B,N], q [B,N], qd [B,N]"""
    rhs, _ = SYSTEMS[system_name]
    B = len(params_list)
    dt_sim = 1e-3
    t_dense = np.arange(0.0, config.T_END + dt_sim * 0.5, dt_sim)  # 细网格
    u_func = make_u_func(sig_list)
    x0 = np.zeros((B, 2))
    parr = np.asarray(params_list, dtype=np.float64)          # [B, n_params]
    params_cols = tuple(parr[:, i] for i in range(parr.shape[1]))
    traj = rk4_batch(rhs, x0, u_func, t_dense, params_cols)   # [B,Tn,2]
    # 在 N 个均匀点采样（与细网格对齐，取最近索引）
    idx = np.round(np.linspace(0, len(t_dense) - 1, config.N_POINTS)).astype(int)
    u_samp = np.stack([u_func(float(config.T_END) * j / (config.N_POINTS - 1))
                       for j in range(config.N_POINTS)], axis=1)  # [B,N]
    q = traj[:, idx, 0]
    qd = traj[:, idx, 1]
    return u_samp.astype(np.float32), q.astype(np.float32), qd.astype(np.float32)


def sample_params(rng, B, ranges, ood=False):
    """按给定范围采样参数元组列表。ood=True 时 alpha 用 OOD 区间（仅 msd）。
    忽略 ranges 中以 "_ood" 结尾的辅助键（它们只是替代范围标注）。"""
    keys = [kk for kk in ranges if not kk.endswith("_ood")]
    vals = []
    for kk in keys:
        lo, hi = ranges[kk]
        if ood and (kk + "_ood") in ranges:
            lo, hi = ranges[kk + "_ood"]
        vals.append(rng.uniform(lo, hi, size=B))
    return list(zip(*vals))


def generate_dataset(system_name, n, seed, ood=False):
    """生成 n 个样本 {u, q, qd, params}。"""
    rng = np.random.default_rng(seed)
    ranges = {"msd": config.MSD, "duffing": config.DUFFING,
              "pendulum": config.PENDULUM}[system_name]
    params = sample_params(rng, n, ranges, ood=ood)
    sigs = [sample_inputs(rng, 1) for _ in range(n)]
    # 逐样本解析式信号批量仿真：把每个样本作为批次一员统一积分
    sig_batch = {"a": np.concatenate([s["a"] for s in sigs], 0),
                 "f": np.concatenate([s["f"] for s in sigs], 0),
                 "phi": np.concatenate([s["phi"] for s in sigs], 0)}
    u, q, qd = simulate_system(system_name, params, sig_batch, rng)
    return {"u": u, "q": q, "qd": qd, "params": np.array(params, dtype=np.float32)}
