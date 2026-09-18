"""一键运行全部实验 E1-E9 + M3（审稿意见 C-m11 / R2-m5：此前只覆盖 E1-E6）。

用法: python run_all.py [--smoke]
正式模式约需数小时（多模型 x 100 epochs x 多 seeds），建议夜间运行。
顺序按依赖手工核对：E1-E6 为单跑基线 → E6 的 A/B/C 变体 →
E3b/E4b/E5b 的 5-seed 加固 → E7 系列扫描 → E8/E9 迁移与长时程 → M3 证书紧度审计。
结果全部写入 results/，图由 make_figures*.py 生成。
"""
import subprocess
import sys
import time

scripts = [
    # --- 单跑基线（E1-E6）---
    "run_e1.py", "run_e2.py", "run_e3.py", "run_e4.py", "run_e5.py", "run_e6.py",
    # --- 闭环变体：A/B（延迟扫描）与 C（参数箱角点）---
    "run_e6_ab.py", "run_e6_corners.py",
    # --- 5-seed 加固 ---
    "run_e3b_multiseed.py", "run_e4b_multiseed.py", "run_e5b_multiseed.py",
    # --- E7：证书-精度权衡三组扫描 ---
    "run_e7_lam.py", "run_e7b_ltarget.py", "run_e7c_lamdyn.py",
    # --- E8/E9：跨系统迁移与长时程 ---
    "run_e8_transfer.py", "run_e9_longhorizon.py",
    # --- M3：证书紧度审计（Frobenius vs 谱范数）---
    "run_m3_ctnorm.py",
]
flag = ["--smoke"] if "--smoke" in sys.argv else []
for s in scripts:
    t0 = time.time()
    print(f"\n{'='*60}\n>>> RUN {s}\n{'='*60}", flush=True)
    r = subprocess.run([sys.executable, s] + flag)
    print(f"<<< {s} finished in {time.time()-t0:.1f}s (exit={r.returncode})")
    if r.returncode != 0:
        sys.exit(f"{s} failed")
print("\nALL EXPERIMENTS DONE. 结果见 results/ ，图见 figures/")
