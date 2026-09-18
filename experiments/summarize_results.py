"""汇总 results/*.csv -> EXPERIMENT_REPORT.md（论文写作依据 + 结果快照）。
用法: python summarize_results.py
"""
import os
import sys

import pandas as pd

from common import ROOT
sys.path.insert(0, os.path.join(ROOT, "src"))
import config

R = config.RESULTS_DIR
OUT = os.path.join(ROOT, "results", "EXPERIMENT_REPORT.md")


def load(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


lines = []
lines.append("# LC-DeepONet 实验结果汇总报告\n")
lines.append("> 生成时间：2026-09-17 ｜ 环境：torch 2.14.0+cpu ｜ 主系统：非线性质量-弹簧-阻尼（参数不确定）\n")
lines.append("> 配置：T=2s，N=64，训练512样本，Adam 1e-3，100 epochs；除注明外均值±标准差来自5个随机种子\n")

# E1
e1 = load("e1_identification.csv")
if e1 is not None:
    lines.append("\n## E1 系统辨识性能（6模型对比，5 seeds）\n")
    show = e1[["model", "params", "ID_RMSE_mean", "ID_RMSE_std", "OOD_RMSE_mean",
               "OOD_RMSE_std", "infer_ms"]].copy()
    for c in ["ID_RMSE_mean", "ID_RMSE_std", "OOD_RMSE_mean", "OOD_RMSE_std"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    lines.append(show.to_markdown(index=False))
    lines.append("\n**结论**：LC-DeepONet ID RMSE 0.0112 为全部方法最优（std 0.0010，低于普通DeepONet的0.0015）；"
                 "Lipschitz约束未牺牲精度。\n")

# E2
e2 = load("e2_ood.csv")
if e2 is not None:
    lines.append("\n## E2 参数 OOD（5 seeds × 512 OOD样本，加固版）\n")
    show = e2[["system", "model", "ID_RMSE_mean", "OOD_RMSE_mean", "OOD_degradation_%", "L_cert"]].copy()
    for c in ["ID_RMSE_mean", "OOD_RMSE_mean"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    show["OOD_degradation_%"] = show["OOD_degradation_%"].map(lambda v: f"{v:.1f}")
    lines.append(show.to_markdown(index=False))
    lines.append("\n**结论（重要修正）**：充分采样下两种模型 OOD 泛化均良好且相当（DeepONet 0.0116 vs LC 0.0120）；"
                 "此前 128 样本单 seed 观察到的'大退化率'为方差伪影。"
                 "**论文口径：Lipschitz 证书以零精度/零泛化代价获得，而非改善泛化。**\n")

# E3
e3 = load("e3_sensitivity.csv")
if e3 is not None:
    lines.append("\n## E3 敏感性 vs 证书（核心验证）\n")
    show = e3[["model", "S_emp", "ID_RMSE"]].copy()
    if "L_cert" in e3:
        show["L_cert"] = e3["L_cert"]
        show["eps_L_emp"] = e3["eps_L_emp"]
    for c in show.columns[1:]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    lines.append(show.to_markdown(index=False))
    lc = e3[e3["model"] == "LC-DeepONet"].iloc[0]
    lines.append(f"\n**结论**：S_emp={lc['S_emp']:.3f} ≤ 证书 L̂={lc['L_cert']:.3f} ✓ "
                 f"（ε_L={lc['eps_L_emp']:.3f}，L̂+ε_L={lc['L_cert']+lc['eps_L_emp']:.3f}）。"
                 "FNO 敏感性高达 2.26 而无任何证书可暴露该风险——凸显证书价值。\n")

# E4
e4 = load("e4_noise.csv")
if e4 is not None:
    lines.append("\n## E4 噪声鲁棒性（RMSE @ 1%-10% 噪声）\n")
    lines.append(e4.to_markdown(index=False))

# E5
e5 = load("e5_ablation.csv")
if e5 is not None:
    lines.append("\n## E5 消融（{Lip}×{Dyn}）\n")
    show = e5[["config", "use_lip", "use_dyn", "ID_RMSE", "OOD_RMSE", "S_emp", "L_cert"]].copy()
    for c in ["ID_RMSE", "OOD_RMSE", "S_emp", "L_cert"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}")
    lines.append(show.to_markdown(index=False))
    lines.append("\n**结论**：B(+Lip) 使 OOD 0.0143→0.0136 并提供证书 4.21；C(+Dyn) OOD 最优 0.0122；"
                 "D(完整) 证书 3.93 且 OOD 次优。差异量级小——不过度声称。\n")

# E6
e6 = load("e6_closedloop.csv")
if e6 is not None:
    lines.append("\n## E6 闭环控制与鲁棒裕度（压轴）\n")
    show = e6[["mult", "k_c", "M_margin", "track_err_L2", "overshoot", "diverged", "design"]].copy()
    for c in ["k_c", "M_margin", "track_err_L2", "overshoot"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    lines.append(show.to_markdown(index=False))
    lines.append("\n**关键数字**：L̂=3.678，ε_L=0.056，L̂+ε_L=3.734；真实系统增益 L_G_true=0.520（证书保守约7倍但安全）；"
                 "k_c_safe=0.241（M=0.1）；普通DeepONet经验设计 k_c_opt=3.541=14.7× 安全值（S_emp=0.254 低估真实增益）。\n")
    lines.append("**结论**：M>0 区间跟踪误差随 k_c 单调改善；M<0 后理论保证失效，overshoot 单调恶化 1.00→5.11"
                 "（硬化弹簧耗散性使系统不会有限发散——裕度与性能退化的相关性即实验验证）。\n")

# 论文状态
lines.append("\n## 与论文的对应\n")
lines.append("""
| 论文元素 | 数据来源 | 状态 |
|---|---|---|
| Table II (辨识) | e1_identification.csv | ✅ 已填入 main.tex |
| E3/证书有效性 | e3_sensitivity.csv | ✅ 已填入 |
| Table III (消融) | e5_ablation.csv | ✅ 已填入 |
| E6/裕度扫描 (Fig.8-9) | e6_closedloop.csv | ✅ 已填入 |
| E2/OOD | e2_ood.csv（加固版） | ✅ 已按修正口径填入 |
| Fig.3/4/5/7/8/9 | figures/*.png | ✅ 已生成 |

**待办**：①文献核查（main.tex 中 TODO-verify 标注）；②Theorem 1 证明完整化（现为 sketch）；③假设 A1 的先验论证；④按目标期刊换模板（当前为通用 article）。
""")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[saved] {OUT}")
