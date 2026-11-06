"""
三層方法：傳球失誤 → 15 秒內失球
協定和犯規題目相同：依比賽日期 60 / 20 / 20 切分；超參數、特徵組、τ 只用驗證集；測試集只評估一次。
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_1samp, ttest_rel
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CANDIDATES, DANGER, FULL, GRID, RES, SEEDS, best_f1_tau, bootstrap_gap,  # noqa: E402
                    fit_if as _fit_if, load_split, metrics, summarize)
from timing import lap, summary  # noqa: E402

# ---------------------------------------------------------------------------
tr, va, te, n_matches = load_split()
ytr, yva, yte = tr.label.values, va.label.values, te.label.values
normal = tr[tr.label == 0]
log = {"rows": {s: len(d) for s, d in (("train", tr), ("val", va), ("test", te))},
       "positives": {s: int(d.label.sum()) for s, d in (("train", tr), ("val", va), ("test", te))},
       "matches": n_matches}
print(log)
lap("讀取資料與切分")


# ---- 探索（只用訓練集）----
sep = []
for c in FULL:
    if tr[c].nunique() < 2:
        continue
    auc = roc_auc_score(ytr, tr[c])
    sep.append({"feature": c, "auc_dir": max(auc, 1 - auc), "sign": 1 if auc >= 0.5 else -1,
                "ks": ks_2samp(tr.loc[ytr == 1, c], tr.loc[ytr == 0, c]).statistic,
                "mean_pos": tr.loc[ytr == 1, c].mean(), "mean_neg": tr.loc[ytr == 0, c].mean()})
sep = pd.DataFrame(sep).sort_values("auc_dir", ascending=False)
sep.to_csv(RES / "separability_train.csv", index=False)
zone = pd.cut(tr.x, [0, 100 / 3, 200 / 3, 100], labels=["own third", "middle", "opp third"], include_lowest=True)
eda = tr.groupby(zone, observed=True).label.agg(n="size", rate="mean")
eda["share"] = eda.n / eda.n.sum()
eda.to_csv(RES / "eda_zone_train.csv")
print(sep.head(10).round(4).to_string(index=False))
print(eda)
lap("探索分析（訓練集）")

rows, scores = [], {}
g = te.matchId.values

# ---- Tier 1 ----
rows.append(summarize("T1 all-normal", [{**metrics(yte, np.zeros(len(te))), "precision": 0.0, "recall": 0.0,
                                          "f1": 0.0, "alert_rate": 0.0}]))
rr = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    s_va, s_te = rng.random(len(va)), rng.random(len(te))
    rr.append(metrics(yte, s_te, best_f1_tau(yva, s_va)))
rows.append(summarize("T1 random", rr))
scores["random"] = np.random.default_rng(0).random(len(te))
SINGLE, SIGN = sep.iloc[0].feature, int(sep.iloc[0].sign)
s1_va, s1_te = SIGN * va[SINGLE].values, SIGN * te[SINGLE].values
rows.append(summarize(f"T1 single feature ({SINGLE})", [metrics(yte, s1_te, best_f1_tau(yva, s1_va))]))
scores["single"] = s1_te
log["single_feature"] = {"feature": SINGLE, "sign": SIGN, "train_auc": float(sep.iloc[0].auc_dir)}
lap("Tier 1")


# ---- Tier 2：Isolation Forest（只用正常樣本）；特徵組、分數方向（雙邊 / 單邊）、超參數都用驗證集選 ----
def fit_if(cand, p, seed):
    feats, transform = CANDIDATES[cand]
    f, fit_s, _, _ = _fit_if(normal, feats, p, seed, transform)
    return f, fit_s


search, best_if = [], {}
bar = tqdm(total=len(CANDIDATES) * len(GRID), desc="T2 搜尋（驗證集）", unit="組")
for cand in CANDIDATES:
    res = []
    for p in GRID:
        f, _ = fit_if(cand, p, 0)
        res.append({"candidate": cand, **p, **metrics(yva, f(va))})
        bar.update()
    res = pd.DataFrame(res).sort_values("pr_auc", ascending=False)
    search.append(res)
    t = res.iloc[0]
    best_if[cand] = {"params": {"n_estimators": int(t.n_estimators), "max_samples": int(t.max_samples),
                                "max_features": float(t.max_features)}, "val_pr_auc": float(t.pr_auc)}
bar.close()
pd.concat(search).to_csv(RES / "iforest_search_val.csv", index=False)
T2 = max(best_if, key=lambda k: best_if[k]["val_pr_auc"])
log["iforest_best"], log["T2_config"] = best_if, T2
lap(f"Tier 2：驗證集搜尋 {len(CANDIDATES) * len(GRID)} 組")

if_runs = {}
for cand in CANDIDATES:
    runs, sl = [], []
    for seed in tqdm(SEEDS, desc=f"T2 IF {cand}", unit="seed"):
        f, fit_s = fit_if(cand, best_if[cand]["params"], seed)
        s_va = f(va)
        t0 = time.perf_counter()
        s_te = f(te)
        lat = (time.perf_counter() - t0) / len(te) * 1000
        runs.append({**metrics(yte, s_te, best_f1_tau(yva, s_va)), "fit_s": fit_s, "ms_per_sample": lat})
        sl.append(s_te)
    if_runs[cand] = runs
    scores[f"IF {cand}"] = np.mean(sl, axis=0)
    rows.append(summarize(f"T2 Isolation Forest, {cand}" + (" (selected on val, primary)" if cand == T2 else ""), runs))

sc = StandardScaler().fit(normal[DANGER])
cov = EmpiricalCovariance().fit(sc.transform(normal[DANGER]))
s_m = cov.mahalanobis(sc.transform(te[DANGER]))
rows.append(summarize("T2 ref: Mahalanobis, danger (6), two-sided",
                      [metrics(yte, s_m, best_f1_tau(yva, cov.mahalanobis(sc.transform(va[DANGER]))))]))
lap("Tier 2：5 個 seed + Mahalanobis")

# ---- Tier 3：XGBoost ----
xs = []
for depth, mcw in tqdm(list(itertools.product((3, 5), (1, 10))), desc="T3 搜尋（驗證集）", unit="組"):
    mdl = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=depth, min_child_weight=mcw,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", early_stopping_rounds=100,
                        random_state=0, n_jobs=2, tree_method="hist")
    mdl.fit(tr[FULL], ytr, eval_set=[(va[FULL], yva)], verbose=False)
    xs.append({"max_depth": depth, "min_child_weight": mcw, "best_iteration": mdl.best_iteration,
               **metrics(yva, mdl.predict_proba(va[FULL])[:, 1])})
xs = pd.DataFrame(xs).sort_values("pr_auc", ascending=False)
xs.to_csv(RES / "xgb_search_val.csv", index=False)
XB = {k: int(xs.iloc[0][k]) for k in ("max_depth", "min_child_weight", "best_iteration")}
log["xgb_best"] = XB
lap("Tier 3：驗證集搜尋 4 組")
xgb_runs, sl, models = [], [], []
for seed in tqdm(SEEDS, desc="T3 XGBoost", unit="seed"):
    mdl = XGBClassifier(n_estimators=XB["best_iteration"] + 1, learning_rate=0.05, max_depth=XB["max_depth"],
                        min_child_weight=XB["min_child_weight"], subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=2, tree_method="hist")
    t0 = time.perf_counter()
    mdl.fit(tr[FULL], ytr)
    fit_s = time.perf_counter() - t0
    s_va = mdl.predict_proba(va[FULL])[:, 1]
    t0 = time.perf_counter()
    s_te = mdl.predict_proba(te[FULL])[:, 1]
    lat = (time.perf_counter() - t0) / len(te) * 1000
    xgb_runs.append({**metrics(yte, s_te, best_f1_tau(yva, s_va)), "fit_s": fit_s, "ms_per_sample": lat})
    sl.append(s_te)
    models.append(mdl)
rows.append(summarize("T3 XGBoost (supervised)", xgb_runs))
scores["xgb"] = np.mean(sl, axis=0)
lap("Tier 3：5 個 seed")

res = pd.DataFrame(rows)
res.to_csv(RES / "main_results.csv", index=False)
# 測試集上每筆失誤的分數（5 個 seed 平均）：消融與誤差分析（ablation_analysis.py）直接沿用，確保和主表是同一組分數
te[["matchId", "eventId", "label"]].assign(
    single=scores["single"], if_full=scores["IF full, two-sided"], if_danger=scores["IF danger (6), two-sided"],
    if_os=scores["IF danger (6), one-sided"], t2=scores[f"IF {T2}"], xgb=scores["xgb"],
).to_csv(RES / "test_scores.csv", index=False)

# 顯著性。兩種「差距」要分清楚：
#   主表的 PR-AUC 是「5 個 seed 各自算 PR-AUC 再平均」；
#   bootstrap 的點估計與 CI 是「先把 5 個 seed 的分數平均，再算 PR-AUC」（等於 5 棵森林的集成），兩者數字不同。
#   另外附上逐 seed 的檢定：T3 vs T2 用配對 t 檢定；單一特徵是確定性的，用單樣本 t 檢定（T2 的 5 個 seed 對單一常數）。
t2 = scores[f"IF {T2}"]
pr = lambda runs: [r["pr_auc"] for r in runs]  # noqa: E731
single_pr = rows[2]["pr_auc"]
log["significance"] = {
    "T2_minus_random": bootstrap_gap(yte, t2, scores["random"], g, desc="bootstrap T2 − 隨機"),
    "T2_minus_single": bootstrap_gap(yte, t2, scores["single"], g, desc="bootstrap T2 − 單一特徵"),
    "T3_minus_T2": bootstrap_gap(yte, scores["xgb"], t2, g, desc="bootstrap T3 − T2"),
    "T3_minus_single": bootstrap_gap(yte, scores["xgb"], scores["single"], g, desc="bootstrap T3 − 單一特徵"),
    "one_sided_minus_two_sided": bootstrap_gap(yte, scores["IF danger (6), one-sided"],
                                               scores["IF danger (6), two-sided"], g, desc="bootstrap 單邊 − 雙邊"),
    "two_sided_danger_minus_full": bootstrap_gap(yte, scores["IF danger (6), two-sided"], scores["IF full, two-sided"],
                                                 g, desc="bootstrap 危險 − 全部"),
    "per_seed": {
        "T2_pr_auc": pr(if_runs[T2]), "T3_pr_auc": pr(xgb_runs), "single_pr_auc": single_pr,
        "two_sided_pr_auc": pr(if_runs["danger (6), two-sided"]),
        "T3_vs_T2_paired_t_p": float(ttest_rel(pr(xgb_runs), pr(if_runs[T2])).pvalue),
        "T2_vs_single_one_sample_t_p": float(ttest_1samp(pr(if_runs[T2]), single_pr).pvalue),
        "one_vs_two_sided_paired_t_p": float(ttest_rel(pr(if_runs["danger (6), one-sided"]),
                                                       pr(if_runs["danger (6), two-sided"])).pvalue),
    },
}
lap("顯著性（block bootstrap 5 組 × 1000 次）")

# 標記的失誤長什麼樣（T2，審查預算 1%）
k = int(round(len(te) * 0.01))
flag = np.zeros(len(te), bool)
flag[np.argsort(-t2, kind="stable")[:k]] = True
prof_cols = ["x", "dist_own_goal", "end_x", "pass_len", "backward", "opp_duel_3s", "pt_launch", "pt_high_pass",
             "pt_hand_pass", "role_GK", "counter_attack"]
prof = te.assign(flag=flag).groupby(["flag", "label"])[prof_cols].mean()
prof.to_csv(RES / "error_profile_T2.csv")

(RES / "log.json").write_text(json.dumps(log, indent=2, default=float))
pd.set_option("display.width", 250)
cols = ["method", "pr_auc", "pr_auc_std", "roc_auc", "p_at_1", "hits_at_1", "r_at_1", "p_at_10", "hits_at_10",
        "r_at_10", "f1", "alert_rate", "fit_s", "ms_per_sample"]
print(res[[c for c in cols if c in res]].round(4).to_string(index=False))
print(json.dumps(log, indent=1, default=float))
print(prof.round(3).to_string())
lap("誤差樣貌與輸出")
summary("三層實驗", RES / "timing_experiment.csv")
