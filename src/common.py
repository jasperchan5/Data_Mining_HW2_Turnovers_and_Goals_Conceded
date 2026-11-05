"""
三層實驗與消融共用的東西：特徵清單、切分、指標、門檻、bootstrap、Isolation Forest
turnover_experiment.py（主實驗）和 ablation_analysis.py（消融與誤差分析）都從這裡拿，確保兩邊的切分和做法完全一樣。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
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
# 每個危險特徵「往哪個方向是危險」：事先依足球常識決定，沒有看標籤
#   -1 = 數值越小越危險（離自家球門近、靠中路、球落在自家半場）；+1 = 數值越大越危險（回傳、被壓迫）
DANGER_SIGN = {"dist_own_goal": -1, "lateral": -1, "end_x": -1, "end_lateral": -1, "backward": 1, "opp_duel_3s": 1}
SETS = {"full": FULL, "danger (6)": DANGER}
GRID = [dict(n_estimators=n, max_samples=s, max_features=f)
        for n in (200, 500) for s in (256, 1024, 4096) for f in (0.5, 1.0)]


def load_split():
    """讀 turnovers.csv，依比賽日期 60 / 20 / 20 切成 train / val / test（以比賽為單位）"""
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
    return tr, va, te, {"train": c1, "val": c2 - c1, "test": len(m) - c2}


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
    """PR-AUC(a) − PR-AUC(b)：點估計與以比賽為單位的 block bootstrap 95% CI"""
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


def one_sided(z, feats):
    """
    單邊尾端轉換（消融 A2）：把標準化後的特徵轉成「往危險方向偏離多少」，安全方向的值一律壓成 0。
    這樣 Isolation Forest 只會把「往危險方向罕見」的失誤隔離出來，往安全方向罕見的不再算異常。
    方向來自 DANGER_SIGN（事先決定，沒有用標籤）。
    """
    sign = np.array([DANGER_SIGN[f] for f in feats], dtype=float)
    return np.clip(z * sign, 0, None)


def fit_if(train_rows, feats, p, seed, transform=None):
    """
    在 train_rows 上 fit StandardScaler + Isolation Forest；回傳 (打分函數, 訓練秒數, 模型, scaler)。
    分數 = −score_samples，越大越異常。transform：標準化之後再套一層轉換（例如 one_sided）。
    """
    sc = StandardScaler().fit(train_rows[feats])
    tf = (lambda z: z) if transform is None else (lambda z: transform(z, feats))
    t0 = time.perf_counter()
    mdl = IsolationForest(random_state=seed, n_jobs=-1, **p).fit(tf(sc.transform(train_rows[feats])))
    fit_s = time.perf_counter() - t0
    return (lambda d: -mdl.score_samples(tf(sc.transform(d[feats])))), fit_s, mdl, sc
