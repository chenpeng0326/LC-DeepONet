"""实验公共工具：路径/导入、CSV 写出、模型重建。"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))


def save_csv(df, name):
    import config
    path = os.path.join(config.RESULTS_DIR, name)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return path
