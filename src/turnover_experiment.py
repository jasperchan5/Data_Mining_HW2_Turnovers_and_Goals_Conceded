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
from scipy.stats import ks_2samp, ttest_rel
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from timing import lap, summary  # noqa: E402
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = (0.01, 0.10)

PASS_TYPES = ["pt_simple_pass", "pt_high_pass", "pt_launch", "pt_cross", "pt_head_pass", "pt_smart_pass", "pt_hand_pass"]
ROLES = ["role_DF", "role_MD", "role_FW", "role_GK"]
RAW = ["x", "lateral", "end_x", "end_lateral", "pass_len", "backward", "minute", "home"] + PASS_TYPES + ROLES
CONTEXT = ["dist_own_goal", "in_own_third", "counter_attack", "score_diff", "team_prev_losses",
           "own_events_10s", "opp_events_10s", "opp_duel_3s", "own_min_x_10s"]
FULL = RAW + CONTEXT
# 事先依足球常識定義的「危險」特徵：在哪裡丟、球丟到哪裡、是不是回傳、是否在對手壓迫下
DANGER = ["dist_own_goal", "lateral", "end_x", "end_lateral", "backward", "opp_duel_3s"]
SETS = {"full": FULL, "danger (6)": DANGER}
GRID = [dict(n_estimators=n, max_samples=s, max_features=f)
        for n in (200, 500) for s in (256, 1024, 4096) for f in (0.5, 1.0)]

# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "processed" / "turnovers.csv", parse_dates=["date"])
for r in ["DF", "MD", "FW", "GK"]:
    df[f"role_{r}"] = (df.role == r).astype(int)
m = df[["matchId", "date"]].drop_duplicates().sort_values(["date", "matchId"]).reset_index(drop=True)
c1, c2 = int(len(m) * 0.6), int(len(m) * 0.8)
split = pd.Series("train", index=m.matchId)
split.iloc[c1:c2] = "val"
split.iloc[c2:] = "test"
df["split"] = df.matchId.map(split)
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
ytr, yva, yte = tr.label.values, va.label.values, te.label.values
normal = tr[tr.label == 0]
log = {"rows": {s: len(d) for s, d in (("train", tr), ("val", va), ("test", te))},
       "positives": {s: int(d.label.sum()) for s, d in (("train", tr), ("val", va), ("test", te))},
       "matches": {"train": c1, "val": c2 - c1, "test": len(m) - c2}}
print(log)
lap("讀取資料與切分")


def metrics(y, s, tau=None):
    y, s = np.asarray(y), np.asarray(s)
    out = {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s)}
    order = np.argsort(-s, kind="stable")
    for b in BUDGETS:
        k = int(round(len(s) * b))
        hit = int(y[order[:k]].sum())
        out[f"p_at_{int(b*100)}"], out[f"r_at_{int(b*100)}"], out[f"hits_at_{int(b*100)}"] = hit / k, hit / max(y.sum(), 1), hit
    if tau is not None:
        pred = s >= tau
        tp = (pred & (y == 1)).sum()
        p, r = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
        out.update(precision=p, recall=r, f1=0 if p + r == 0 else 2 * p * r / (p + r), alert_rate=pred.mean())
    return out


def best_f1_tau(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[int(np.nanargmax(f1[:-1]))])


def bootstrap_gap(y, a, b, groups, n=1000, seed=0, desc="bootstrap"):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    gaps = []
    for _ in tqdm(range(n), desc=desc, unit="次", leave=False):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        if y[idx].sum():
            gaps.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    return [float(average_precision_score(y, a) - average_precision_score(y, b)),
            float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]


def summarize(name, runs):
    row = {"method": name}
    for k in runs[0]:
        v = [r[k] for r in runs]
        row[k], row[f"{k}_std"] = float(np.mean(v)), float(np.std(v))
    return row


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


# ---- Tier 2：Isolation Forest（只用正常樣本），特徵組與超參數都用驗證集選 ----
def fit_if(feats, p, seed):
    sc = StandardScaler().fit(normal[feats])
    t0 = time.perf_counter()
    mdl = IsolationForest(random_state=seed, n_jobs=-1, **p).fit(sc.transform(normal[feats]))
    return (lambda d: -mdl.score_samples(sc.transform(d[feats]))), time.perf_counter() - t0


search, best_if = [], {}
bar = tqdm(total=len(SETS) * len(GRID), desc="T2 搜尋（驗證集）", unit="組")
for name, feats in SETS.items():
    res = []
    for p in GRID:
        f, _ = fit_if(feats, p, 0)
        res.append({"feature_set": name, **p, **metrics(yva, f(va))})
        bar.update()
    res = pd.DataFrame(res).sort_values("pr_auc", ascending=False)
    search.append(res)
    t = res.iloc[0]
    best_if[name] = {"params": {"n_estimators": int(t.n_estimators), "max_samples": int(t.max_samples),
                                "max_features": float(t.max_features)}, "val_pr_auc": float(t.pr_auc)}
bar.close()
pd.concat(search).to_csv(RES / "iforest_search_val.csv", index=False)
T2_SET = max(best_if, key=lambda k: best_if[k]["val_pr_auc"])
log["iforest_best"], log["T2_feature_set"] = best_if, T2_SET
lap(f"Tier 2：驗證集搜尋 {len(SETS) * len(GRID)} 組")

if_runs = {}
for name, feats in SETS.items():
    runs, sl = [], []
    for seed in tqdm(SEEDS, desc=f"T2 IF {name}", unit="seed"):
        f, fit_s = fit_if(feats, best_if[name]["params"], seed)
        s_va = f(va)
        t0 = time.perf_counter()
        s_te = f(te)
        lat = (time.perf_counter() - t0) / len(te) * 1000
        runs.append({**metrics(yte, s_te, best_f1_tau(yva, s_va)), "fit_s": fit_s, "ms_per_sample": lat})
        sl.append(s_te)
    if_runs[name] = runs
    scores[f"IF {name}"] = np.mean(sl, axis=0)
    rows.append(summarize(f"T2 Isolation Forest, {name}" + (" (selected on val)" if name == T2_SET else ""), runs))

sc = StandardScaler().fit(normal[DANGER])
cov = EmpiricalCovariance().fit(sc.transform(normal[DANGER]))
s_m = cov.mahalanobis(sc.transform(te[DANGER]))
rows.append(summarize("T2 ref: Mahalanobis, danger (6)",
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

t2 = scores[f"IF {T2_SET}"]
log["significance"] = {
    "T2_minus_random": bootstrap_gap(yte, t2, scores["random"], g, desc="bootstrap T2 − 隨機"),
    "T2_minus_single": bootstrap_gap(yte, t2, scores["single"], g, desc="bootstrap T2 − 單一特徵"),
    "T3_minus_T2": bootstrap_gap(yte, scores["xgb"], t2, g, desc="bootstrap T3 − T2"),
    "T3_minus_single": bootstrap_gap(yte, scores["xgb"], scores["single"], g, desc="bootstrap T3 − 單一特徵"),
    "IF_danger_minus_IF_full": bootstrap_gap(yte, scores["IF danger (6)"], scores["IF full"], g,
                                             desc="bootstrap IF 危險 − 全部"),
    "T3_vs_T2_paired_t_p": float(ttest_rel([r["pr_auc"] for r in xgb_runs],
                                           [r["pr_auc"] for r in if_runs[T2_SET]]).pvalue),
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
