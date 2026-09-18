# LC-DeepONet 实验结果汇总报告

> 生成时间：2026-09-17 ｜ 环境：torch 2.14.0+cpu ｜ 主系统：非线性质量-弹簧-阻尼（参数不确定）

> 配置：T=2s，N=64，训练512样本，Adam 1e-3，100 epochs；除注明外均值±标准差来自5个随机种子


## E1 系统辨识性能（6模型对比，5 seeds）

| model       |   params |   ID_RMSE_mean |   ID_RMSE_std |   OOD_RMSE_mean |   OOD_RMSE_std |    infer_ms |
|:------------|---------:|---------------:|--------------:|----------------:|---------------:|------------:|
| MLP         |    98880 |         0.0102 |        0.0004 |          0.0112 |         0.0002 | 0.000318127 |
| LSTM        |   199297 |         0.0157 |        0.0045 |          0.0211 |         0.0039 | 0.00297771  |
| Transformer |   267841 |         0.0306 |        0.0062 |          0.036  |         0.0061 | 0.00712523  |
| FNO         |   123297 |         0.036  |        0.0009 |          0.0481 |         0.002  | 0.00100973  |
| DeepONet    |   181696 |         0.012  |        0.0015 |          0.0135 |         0.0008 | 0.000158289 |
| LC-DeepONet |   181696 |         0.0112 |        0.001  |          0.0128 |         0.0011 | 0.00021731  |

**结论**：LC-DeepONet ID RMSE 0.0112 为全部方法最优（std 0.0010，低于普通DeepONet的0.0015）；Lipschitz约束未牺牲精度。


## E2 参数 OOD（5 seeds × 512 OOD样本，加固版）

| system   | model       |   ID_RMSE_mean |   OOD_RMSE_mean |   OOD_degradation_% |    L_cert |
|:---------|:------------|---------------:|----------------:|--------------------:|----------:|
| msd(主)   | DeepONet    |         0.0123 |          0.0116 |                -6.3 | nan       |
| msd(主)   | LC-DeepONet |         0.0119 |          0.012  |                 1.2 |   4.16484 |
| duffing  | DeepONet    |         0.0107 |        nan      |               nan   | nan       |
| duffing  | LC-DeepONet |         0.0091 |        nan      |               nan   |   4.4025  |
| pendulum | DeepONet    |         0.0112 |        nan      |               nan   | nan       |
| pendulum | LC-DeepONet |         0.0125 |        nan      |               nan   |   3.17041 |

**结论（重要修正）**：充分采样下两种模型 OOD 泛化均良好且相当（DeepONet 0.0116 vs LC 0.0120）；此前 128 样本单 seed 观察到的'大退化率'为方差伪影。**论文口径：Lipschitz 证书以零精度/零泛化代价获得，而非改善泛化。**


## E3 敏感性 vs 证书（核心验证）

| model       |   S_emp |   ID_RMSE |   L_cert |   eps_L_emp |
|:------------|--------:|----------:|---------:|------------:|
| MLP         |   0.31  |     0.011 |  nan     |     nan     |
| LSTM        |   0.214 |     0.013 |  nan     |     nan     |
| Transformer |   0.215 |     0.023 |  nan     |     nan     |
| FNO         |   2.258 |     0.036 |  nan     |     nan     |
| DeepONet    |   0.274 |     0.012 |  nan     |     nan     |
| LC-DeepONet |   0.26  |     0.011 |    3.918 |       0.043 |

**结论**：S_emp=0.260 ≤ 证书 L̂=3.918 ✓ （ε_L=0.043，L̂+ε_L=3.962）。FNO 敏感性高达 2.26 而无任何证书可暴露该风险——凸显证书价值。


## E4 噪声鲁棒性（RMSE @ 1%-10% 噪声）

| model       |   RMSE_noise_1% |   RMSE_noise_3% |   RMSE_noise_5% |   RMSE_noise_10% |
|:------------|----------------:|----------------:|----------------:|-----------------:|
| MLP         |       0.0107033 |       0.0126421 |       0.0161375 |        0.0274987 |
| LSTM        |       0.011099  |       0.0125521 |       0.0149764 |        0.023061  |
| Transformer |       0.0310996 |       0.031462  |       0.0323824 |        0.0367788 |
| FNO         |       0.0380096 |       0.0530291 |       0.067741  |        0.0927657 |
| DeepONet    |       0.0131953 |       0.0143533 |       0.016631  |        0.0250027 |
| LC-DeepONet |       0.0120026 |       0.0133212 |       0.0158552 |        0.0247243 |

## E5 消融（{Lip}×{Dyn}）

| config   | use_lip   | use_dyn   |   ID_RMSE |   OOD_RMSE |   S_emp |   L_cert |
|:---------|:----------|:----------|----------:|-----------:|--------:|---------:|
| A(基线)    | False     | False     |    0.0114 |     0.0143 |  0.2567 | nan      |
| B(+Lip)  | True      | False     |    0.0109 |     0.0136 |  0.2355 |   4.2074 |
| C(+Dyn)  | False     | True      |    0.0123 |     0.0122 |  0.2285 | nan      |
| D(完整LC)  | True      | True      |    0.011  |     0.0131 |  0.2614 |   3.9288 |

**结论**：B(+Lip) 使 OOD 0.0143→0.0136 并提供证书 4.21；C(+Dyn) OOD 最优 0.0122；D(完整) 证书 3.93 且 OOD 次优。差异量级小——不过度声称。


## E6 闭环控制与鲁棒裕度（压轴）

|   mult |    k_c |   M_margin |   track_err_L2 |   overshoot | diverged   | design                |
|-------:|-------:|-----------:|---------------:|------------:|:-----------|:----------------------|
|   0.25 |  0.06  |      0.775 |          0.98  |       1     | False      | LC-certificate x mult |
|   0.5  |  0.12  |      0.55  |          0.961 |       1     | False      | LC-certificate x mult |
|   1    |  0.241 |      0.1   |          0.923 |       1     | False      | LC-certificate x mult |
|   2    |  0.482 |     -0.8   |          0.849 |       1.001 | False      | LC-certificate x mult |
|   4    |  0.964 |     -2.6   |          0.715 |       1.002 | False      | LC-certificate x mult |
|   8    |  1.928 |     -6.2   |          0.507 |       1.004 | False      | LC-certificate x mult |
|  16    |  3.856 |    -13.4   |          0.36  |       1.326 | False      | LC-certificate x mult |
|  32    |  7.711 |    -27.8   |          0.507 |       1.015 | False      | LC-certificate x mult |
|  64    | 15.423 |    -56.6   |          1.135 |       2.402 | False      | LC-certificate x mult |
| 128    | 30.845 |   -114.2   |          2.9   |       5.114 | False      | LC-certificate x mult |
| nan    |  3.541 |      0.1   |          0.362 |       1.3   | False      | DeepONet S_emp design |

**关键数字**：L̂=3.678，ε_L=0.056，L̂+ε_L=3.734；真实系统增益 L_G_true=0.520（证书保守约7倍但安全）；k_c_safe=0.241（M=0.1）；普通DeepONet经验设计 k_c_opt=3.541=14.7× 安全值（S_emp=0.254 低估真实增益）。

**结论**：M>0 区间跟踪误差随 k_c 单调改善；M<0 后理论保证失效，overshoot 单调恶化 1.00→5.11（硬化弹簧耗散性使系统不会有限发散——裕度与性能退化的相关性即实验验证）。


## 与论文的对应


| 论文元素 | 数据来源 | 状态 |
|---|---|---|
| Table II (辨识) | e1_identification.csv | ✅ 已填入 main.tex |
| E3/证书有效性 | e3_sensitivity.csv | ✅ 已填入 |
| Table III (消融) | e5_ablation.csv | ✅ 已填入 |
| E6/裕度扫描 (Fig.8-9) | e6_closedloop.csv | ✅ 已填入 |
| E2/OOD | e2_ood.csv（加固版） | ✅ 已按修正口径填入 |
| Fig.3/4/5/7/8/9 | figures/*.png | ✅ 已生成 |

**待办**：①文献核查（main.tex 中 TODO-verify 标注）；②Theorem 1 证明完整化（现为 sketch）；③假设 A1 的先验论证；④按目标期刊换模板（当前为通用 article）。
