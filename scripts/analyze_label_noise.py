"""Summarize label-noise robustness CSV: mean F1 table + Wilcoxon on degradations.

Usage: PYTHONPATH=. python scripts/analyze_label_noise.py [raw_csv_path]
"""
import sys
import pathlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RAW = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "artifacts/phase1_analysis/robustness_label_noise_raw.csv")
df = pd.read_csv(RAW)
print(f"# source: {RAW}")
print(f"# rows={len(df)}  datasets={sorted(df.dataset.unique())}  "
      f"methods={sorted(df.method.unique())}  noise={sorted(df.noise_p.unique())}")
print(f"# (seed,fold) cells per dataset: "
      f"{df.groupby('dataset')[['seed','fold']].apply(lambda g: g.drop_duplicates().shape[0]).to_dict()}")

LAB = {"calibrated_xgboost": "Cal-XGB", "catboost": "CatBoost", "hies": "HIES"}
order_ds = ["uci_dropout_success", "uci_student", "xapi"]
order_m = ["calibrated_xgboost", "catboost", "hies"]

print("\n=== Table 8 (regenerated) - mean macro-F1, 3 dp ===")
print(f"{'dataset':22} {'method':9} {'p=0':>6} {'p=5%':>6} {'p=10%':>7} {'p=20%':>7} {'Diff(20-0)':>10}")
summary = (df.groupby(["dataset", "noise_p", "method"], as_index=False)
             .agg(f1_mean=("f1_macro", "mean"), f1_std=("f1_macro", "std")))
for ds in [d for d in order_ds if d in df.dataset.unique()]:
    for m in order_m:
        s = summary[(summary.dataset == ds) & (summary.method == m)].set_index("noise_p")["f1_mean"]
        if s.empty:
            continue
        v0, v5, v10, v20 = (s.get(0.0, np.nan), s.get(0.05, np.nan),
                            s.get(0.10, np.nan), s.get(0.20, np.nan))
        print(f"{ds:22} {LAB[m]:9} {v0:6.3f} {v5:6.3f} {v10:7.3f} {v20:7.3f} {v20 - v0:+10.3f}")

print("\n=== Paired Wilcoxon on per-(seed,fold) degradation  delta = f1(0.20) - f1(0.00) ===")
print("(one-sided: H1 = HIES less negative / degrades less than comparator)")
for ds in [d for d in order_ds if d in df.dataset.unique()]:
    sub = df[df.dataset == ds]
    wide = (sub.pivot_table(index=["seed", "fold"], columns=["method", "noise_p"],
                            values="f1_macro"))
    for comp in ["catboost", "calibrated_xgboost"]:
        try:
            d_h = wide[("hies", 0.20)] - wide[("hies", 0.0)]
            d_c = wide[(comp, 0.20)] - wide[(comp, 0.0)]
            pair = pd.concat([d_h, d_c], axis=1, keys=["hies", comp]).dropna()
            n = len(pair)
            stat, p_two = wilcoxon(pair["hies"], pair[comp])
            # one-sided: HIES degradation less negative (greater) than comparator
            stat1, p_one = wilcoxon(pair["hies"], pair[comp], alternative="greater")
            print(f"{ds:22} HIES vs {LAB[comp]:9} n={n:3d} "
                  f"median dHIES={pair['hies'].median():+.4f} "
                  f"median d{LAB[comp]}={pair[comp].median():+.4f} "
                  f"p_two={p_two:.2e} p_one={p_one:.2e}")
        except (KeyError, ValueError) as e:
            print(f"{ds:22} HIES vs {LAB[comp]:9} — skipped ({e})")
