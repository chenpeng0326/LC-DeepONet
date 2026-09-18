"""E6 闭环控制：小增益裕度 M = 1 - L_C*(L_hat + eps_L) 与真实闭环行为的相关性。

设计规则：
- LC-DeepONet : k_c = 0.9 / (L_hat + eps_L)  ->  M = 0.1 > 0（理论安全域内）
- 普通 DeepONet: k_c = 0.9 / S_emp           （经验敏感度可能低估真实增益）
控制器：静态输出反馈 u = k_c (r - y)，带一拍计算延迟（dt_ctrl 离散更新 + 延迟）。
注意口径：M 是本文自定义的实验指标（拟定书 E6），不作为普适标准。
"""
import numpy as np
import torch

import config
import systems

# E3/E6 固定中值参数（真实对象）
MSD_MID = (0.6, 1.0, 0.85)


def closed_loop_sim(k_c, params, r_func, d_amp=0.0, delay_steps=50,
                    dt_sim=1e-3, t_end=None, div_thresh=50.0):
    """真实 MSD 系统 + 静态输出反馈（一拍延迟）。返回 (t, q, r, u, diverged)。
    params: (c, k, alpha)；参考 r_func(t)；输出扰动 d 在 q 通道注入。"""
    t_end = config.T_END if t_end is None else t_end
    c, k, alpha = params
    n_steps = int(t_end / dt_sim)
    dt_ctrl = delay_steps * dt_sim
    x = np.zeros(2)
    q_log = np.empty(n_steps + 1)
    r_log = np.empty(n_steps + 1)
    u_log = np.empty(n_steps + 1)
    diverged = False
    u = 0.0          # ZOH 控制量
    u_prev = 0.0     # 延迟一拍的控制量
    q_log[0] = r_log[0] = 0.0
    u_log[0] = 0.0
    step_in_ctrl = 0
    for j in range(n_steps):
        t = j * dt_sim
        if step_in_ctrl == 0:
            u_prev = u
            r_now = float(r_func(t))
            y_meas = x[0] + d_amp * np.sin(2 * np.pi * 1.7 * t)
            u = k_c * (r_now - y_meas)
            r_log[j] = r_now
        else:
            r_log[j] = float(r_func(t))
        # RK4 on q'' = u_prev - c q' - k q - alpha q^3  (u_prev: 控制延迟)
        def f(x_):
            return np.array([x_[1], u_prev - c * x_[1] - k * x_[0] - alpha * x_[0] ** 3])
        k1 = f(x)
        k2 = f(x + 0.5 * dt_sim * k1)
        k3 = f(x + 0.5 * dt_sim * k2)
        k4 = f(x + dt_sim * k3)
        x = x + (dt_sim / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q_log[j + 1] = x[0]
        u_log[j + 1] = u_prev
        step_in_ctrl = (step_in_ctrl + 1) % delay_steps
        if not np.isfinite(x[0]) or abs(x[0]) > div_thresh:
            diverged = True
            q_log[j + 1:] = np.nan
            u_log[j + 1:] = np.nan
            r_log[j + 1:] = np.nan
            break
    t = np.arange(n_steps + 1) * dt_sim
    r_log[n_steps] = float(r_func(t[-1]))   # fill the last sample (loop covers 0..n_steps-1)
    return t, q_log, r_log, u_log, diverged


def tracking_error_l2(t, q, r):
    """L2 跟踪误差（截断到 1.0s 之后，避开瞬态），发散时返回 inf。"""
    mask = t >= 1.0
    if np.any(np.isnan(q[mask])):
        return float("inf")
    dt = t[1] - t[0]
    return float(np.sqrt(dt * np.sum((q[mask] - r[mask]) ** 2)))


def reference_step(t):
    return 1.0 * (t >= 0.5)


def reference_sine(t):
    return 0.8 * np.sin(2 * np.pi * 0.5 * t)


def sweep_gain(k_c_list, params, r_func=reference_step, d_amp=0.0, delay_override=None):
    """对一组控制器增益做闭环扫描。返回 list of dict。
    overshoot 取阶跃后 (t>=0.55s) 的 max|q-r|，避免把参考跳变本身计入。
    delay_override: 覆盖默认延迟步数（控制周期 = delay_steps·dt_sim 秒）。"""
    rows = []
    for k_c in k_c_list:
        kw = {} if delay_override is None else {"delay_steps": delay_override}
        t, q, r, u, div = closed_loop_sim(k_c, params, r_func, d_amp=d_amp, **kw)
        err = tracking_error_l2(t, q, r)
        mask = t >= 0.55
        ov = float(np.nanmax(np.abs(q[mask] - r[mask]))) if not div else float("inf")
        rows.append({"k_c": k_c, "track_err_L2": err,
                     "overshoot": ov, "diverged": div})
    return rows


# ---------- eps_L 与 S_emp 的估计（E3 用） ----------

@torch.no_grad()
def estimate_eps_L(model, u_pool, n_pairs=4096, seed=0, params=MSD_MID):
    """误差算子 E = G - G_hat 的 Lipschitz 经验估计（L2, 离散）。
    eps_L_emp = max ||E(u_i)-E(u_j)|| / ||u_i-u_j||。
    u_pool: [M,N] 张量；params: 真实系统参数。"""
    g = torch.Generator().manual_seed(seed)
    M_ = u_pool.shape[0]
    idx = torch.randint(0, M_, (n_pairs, 2), generator=g)
    e = model(u_pool) - true_G_batch(u_pool, params)  # [M,N]
    du = u_pool[idx[:, 0]] - u_pool[idx[:, 1]]        # [P,N]
    de = e[idx[:, 0]] - e[idx[:, 1]]
    num = torch.sqrt(config.DT * (de ** 2).sum(dim=1))
    den = torch.sqrt(config.DT * (du ** 2).sum(dim=1)).clamp_min(1e-8)
    return float((num / den).max())


@torch.no_grad()
def estimate_S_emp(model, u_pool, delta_list=(0.02, 0.05, 0.1, 0.2)):
    """输入扰动敏感度 S_emp = max_delta max_u ||G_hat(u+delta)-G_hat(u)|| / ||delta||。
    L2, 离散意义。"""
    best = 0.0
    for d in delta_list:
        dirs = torch.randn_like(u_pool)
        dirs = dirs / torch.sqrt((dirs ** 2).sum(dim=1, keepdim=True))
        y1 = model(u_pool)
        y2 = model(u_pool + d * dirs)
        num = torch.sqrt(config.DT * ((y2 - y1) ** 2).sum(dim=1))
        den = torch.sqrt(config.DT * ((d * dirs) ** 2).sum(dim=1))
        best = max(best, float((num / den).max()))
    return best


def true_G_batch(u_pool, params=MSD_MID):
    """用数值仿真求真实 G(u)（RK4，与训练数据同流程）。u_pool [M,N] torch。
    params: (c, k, alpha)。返回 q [M,N] torch。"""
    import numpy as np
    from systems import rk4_batch, msd_rhs
    M_, N_ = u_pool.shape
    c, k, alpha = params
    dt_sim = 1e-3
    t_dense = np.arange(0.0, config.T_END + dt_sim * 0.5, dt_sim)
    u_np = u_pool.numpy()
    # 线性插值 u(t)：训练输入为解析正弦和，细网格上插值误差可忽略（带宽 2Hz << 采样 32Hz）
    t_samp = np.linspace(0.0, config.T_END, N_)
    u_funcs = []
    for i in range(M_):
        ui = u_np[i]
        u_funcs.append((lambda arr: (lambda t: float(np.interp(t, t_samp, arr))))(ui))

    def u_func(t):
        return np.array([uf(t) for uf in u_funcs])

    x0 = np.zeros((M_, 2))
    params_cols = tuple(np.full(M_, p, dtype=np.float64) for p in (c, k, alpha))
    traj = rk4_batch(msd_rhs, x0, u_func, t_dense, params_cols)
    idx = np.round(np.linspace(0, len(t_dense) - 1, N_)).astype(int)
    q = traj[:, idx, 0]
    return torch.tensor(q, dtype=torch.float32)
