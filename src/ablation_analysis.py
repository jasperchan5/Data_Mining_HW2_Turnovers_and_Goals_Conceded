"""
消融與誤差分析（Tier 2 為主），沿用主實驗的切分、超參數與測試分數
    先跑 turnover_experiment.py（會寫出 results/log.json、results/test_scores.csv）

消融：一次只改一個因素，其他照主實驗（危險 6 個特徵、驗證集選出的超參數、只用正常樣本、5 個 seed）
    A1 訓練資料污染：混入 0% / 0.1% / 0.2% / 全部（0.29%）致命失誤
    A2 分數函數（異常偵測特有）：雙邊 Isolation Forest（主方法）vs 單邊尾端 vs 其他 one-class 分數
    A3 特徵組：拿掉 / 加入一組特徵
    A4 門檻 τ：驗證集最佳 F1 / 審查預算 1%、10% / 不用標籤（正常訓練分數的 99 百分位）

誤差分析：審查預算 1%（τ = 驗證集分數的 99 百分位）下的 TP / FP / FN，
    具體案例 + 每個特徵的貢獻（Isolation Forest 用遮蔽法，XGBoost 用 TreeSHAP）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.covariance import EmpiricalCovariance  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DANGER, DANGER_SIGN, FULL, RES, ROOT, SEEDS, best_f1_tau, bootstrap_gap,  # noqa: E402
                    fit_if, load_split, metrics, one_sided, summarize)
from timing import lap, summary  # noqa: E402

FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

tr, va, te, _ = load_split()
ytr, yva, yte = tr.label.values, va.label.values, te.label.values
normal = tr[tr.label == 0]
g = te.matchId.values

main_log = json.loads((RES / "log.json").read_text())
T2_SET = main_log["T2_feature_set"]
assert T2_SET == "danger (6)", "消融以危險 6 個特徵為主方法；若驗證集選了別的特徵組，請先檢查"
P = main_log["iforest_best"][T2_SET]["params"]
XB = main_log["xgb_best"]
ts = pd.read_csv(RES / "test_scores.csv")
assert (ts.eventId.values == te.eventId.values).all(), "test_scores.csv 和目前的切分對不上，請先重跑 turnover_experiment.py"
log = {"T2_params": P, "xgb_best": XB}
lap("讀取資料、主實驗設定")


# ---------------------------------------------------------------------------
def run_if(feats=DANGER, train_rows=None, transform=None, tau_rule="best_f1", contam=0.0):
    """5 個 seed；回傳 (每個 seed 的測試指標, 測試分數平均, 驗證分數平均)"""
    runs, s_te_all, s_va_all = [], [], []
    for seed in SEEDS:
        rows = normal if train_rows is None else train_rows
        if contam > 0:
            n_pos = int(round(len(normal) * contam / (1 - contam)))
            pos = tr[tr.label == 1]
            rows = pd.concat([normal, pos.sample(n=min(n_pos, len(pos)), random_state=seed)])
        f, _, _, _ = fit_if(rows, feats, P, seed, transform)
        s_va, s_te = f(va), f(te)
        if tau_rule == "best_f1":
            tau = best_f1_tau(yva, s_va)
        elif tau_rule.startswith("budget"):
            tau = float(np.quantile(s_va, 1 - float(tau_rule.split("_")[1])))
        elif tau_rule == "normal_p99":
            tau = float(np.quantile(f(normal), 0.99))
        else:
            raise ValueError(tau_rule)
        runs.append({**metrics(yte, s_te, tau), "val_pr_auc": average_precision_score(yva, s_va)})
        s_te_all.append(s_te)
        s_va_all.append(s_va)
    return runs, np.mean(s_te_all, axis=0), np.mean(s_va_all, axis=0)


abl, abl_scores = [], {}


def add(group, setting, runs, score=None, note=""):
    r = summarize(setting, runs)
    r = {"ablation": group, "setting": setting, **{k: v for k, v in r.items() if k != "method"}, "note": note}
    abl.append(r)
    if score is not None:
        abl_scores[(group, setting)] = score
    tqdm.write(f"  {group:16s} {setting:45s} PR-AUC {r['pr_auc']:.4f} ± {r['pr_auc_std']:.4f}  "
               f"前 1% {r['hits_at_1']:.1f}  前 10% {r['hits_at_10']:.1f}")


# ---- 主方法（參照列）----
ref_runs, s_t2, s_t2_va = run_if()
assert np.allclose(s_t2, ts.if_danger.values), "重算的 Tier 2 分數和主實驗不同"
lap("主方法重算（5 seed）")

# ---- A1 訓練資料污染 ----
nat = ytr.mean()
for rate, label in [(0.0, "0% (normal only, primary)"), (0.001, "0.1%"), (0.002, "0.2%"),
                    (nat, f"{nat:.2%} (all {int(ytr.sum())}, no filtering)")]:
    runs = ref_runs if rate == 0 else run_if(contam=rate)[0]
    add("A1 contamination", label, runs)
lap("A1 訓練資料污染")

# ---- A2 分數函數（全部只用正常樣本、危險 6 個特徵）----
add("A2 scoring", "IF two-sided (primary)", ref_runs, s_t2)
os_runs, s_os, s_os_va = run_if(transform=one_sided)
add("A2 scoring", "IF one-sided tail", os_runs, s_os, "only deviations toward the dangerous side count")

sc = StandardScaler().fit(normal[DANGER])
Zn, Zva, Zte = (sc.transform(d[DANGER]) for d in (normal, va, te))
# 不用模型：單邊偏離量直接加總（看 Isolation Forest 本身有沒有比「加總」多學到什麼）
s_sum, s_sum_va = one_sided(Zte, DANGER).sum(1), one_sided(Zva, DANGER).sum(1)
add("A2 scoring", "one-sided deviation sum (no model)",
    [{**metrics(yte, s_sum, best_f1_tau(yva, s_sum_va)), "val_pr_auc": average_precision_score(yva, s_sum_va)}], s_sum)
cov = EmpiricalCovariance().fit(Zn)
s_maha = cov.mahalanobis(Zte)
s_maha_va = cov.mahalanobis(Zva)
add("A2 scoring", "Mahalanobis distance (two-sided)",
    [{**metrics(yte, s_maha, best_f1_tau(yva, s_maha_va)), "val_pr_auc": average_precision_score(yva, s_maha_va)}], s_maha)
knn = []
for seed in tqdm(SEEDS, desc="kNN", unit="seed", leave=False):
    idx = np.random.default_rng(seed).choice(len(Zn), 20000, replace=False)
    nn = NearestNeighbors(n_neighbors=10).fit(Zn[idx])
    kv = nn.kneighbors(Zva)[0].mean(1)
    knn.append({**metrics(yte, nn.kneighbors(Zte)[0].mean(1), best_f1_tau(yva, kv)), "val_pr_auc": average_precision_score(yva, kv)})
add("A2 scoring", "kNN mean distance, k=10 (two-sided)", knn, note="20k normal reference points per seed")
gmm = []
for seed in tqdm(SEEDS, desc="GMM", unit="seed", leave=False):
    gm = GaussianMixture(n_components=8, covariance_type="full", reg_covar=1e-3, random_state=seed).fit(Zn)
    gv = -gm.score_samples(Zva)
    gmm.append({**metrics(yte, -gm.score_samples(Zte), best_f1_tau(yva, gv)), "val_pr_auc": average_precision_score(yva, gv)})
add("A2 scoring", "GMM neg. log-likelihood, 8 comp. (two-sided)", gmm)
lap("A2 分數函數")

# ---- A3 特徵組（雙邊 Isolation Forest）----
CTX_EXTRA = ["counter_attack", "score_diff", "own_events_10s", "opp_events_10s", "own_min_x_10s"]
for name, feats in [("danger 6 (primary)", DANGER), ("all 28", FULL),
                    ("− loss location (dist_own_goal, lateral)", [f for f in DANGER if f not in ("dist_own_goal", "lateral")]),
                    ("− landing point (end_x, end_lateral)", [f for f in DANGER if f not in ("end_x", "end_lateral")]),
                    ("− backward", [f for f in DANGER if f != "backward"]),
                    ("− pressure (opp_duel_3s)", [f for f in DANGER if f != "opp_duel_3s"]),
                    ("+ 5 context features", DANGER + CTX_EXTRA)]:
    runs = ref_runs if feats is DANGER else run_if(feats=feats)[0]
    add("A3 features", name, runs)
# 同樣的危險 6 個特徵給 XGBoost：分開「標籤」和「特徵」各自貢獻多少
xr = []
for seed in SEEDS:
    m = xgb.XGBClassifier(n_estimators=XB["best_iteration"] + 1, learning_rate=0.05, max_depth=XB["max_depth"],
                          min_child_weight=XB["min_child_weight"], subsample=0.8, colsample_bytree=0.8,
                          random_state=seed, n_jobs=2, tree_method="hist").fit(tr[DANGER], ytr)
    xv = m.predict_proba(va[DANGER])[:, 1]
    xr.append({**metrics(yte, m.predict_proba(te[DANGER])[:, 1], best_f1_tau(yva, xv)), "val_pr_auc": average_precision_score(yva, xv)})
add("A3 features", "ref: XGBoost on danger 6 (supervised)", xr, note="Tier 3 hyperparameters")
lap("A3 特徵組")

# ---- A4 門檻 ----
for rule, label in [("best_f1", "best F1 on val (primary)"), ("budget_0.01", "review budget 1% (val 99th pct)"),
                    ("budget_0.10", "review budget 10% (val 90th pct)"),
                    ("normal_p99", "label-free: 99th pct of normal train scores")]:
    runs = ref_runs if rule == "best_f1" else run_if(tau_rule=rule)[0]
    add("A4 threshold", label, runs)
lap("A4 門檻")

abl_df = pd.DataFrame(abl)
abl_df.to_csv(RES / "ablations.csv", index=False)

# ---- 關鍵比較的顯著性（5 個 seed 平均後的分數，以比賽為單位的 block bootstrap）----
log["significance"] = {
    "one_sided_minus_two_sided": bootstrap_gap(yte, s_os, s_t2, g, desc="單邊 − 雙邊"),
    "one_sided_minus_single": bootstrap_gap(yte, s_os, ts.single.values, g, desc="單邊 − 單一特徵"),
    "xgb_minus_one_sided": bootstrap_gap(yte, ts.xgb.values, s_os, g, desc="XGBoost − 單邊"),
    "sum_minus_one_sided_if": bootstrap_gap(yte, s_sum, s_os, g, desc="加總 − 單邊 IF"),
}
for k, v in log["significance"].items():
    print(f"  {k:28s} {v[0]:+.4f}  95% CI [{v[1]:+.4f}, {v[2]:+.4f}]")
lap("消融顯著性（bootstrap 4 組 × 1000 次）")

# ===========================================================================
# 誤差分析：審查預算 1%（τ 用驗證集決定，測試集只套用）
# ===========================================================================
tau = float(np.quantile(s_t2_va, 0.99))
tau_os = float(np.quantile(s_os_va, 0.99))
A = te.assign(score=s_t2, flag=s_t2 >= tau, score_os=s_os, flag_os=s_os >= tau_os,
              single=ts.single.values, s_xgb=ts.xgb.values,
              pct=pd.Series(s_t2).rank(pct=True).values)
A["outcome"] = np.select([A.flag & (A.label == 1), A.flag & (A.label == 0), ~A.flag & (A.label == 1)],
                         ["TP", "FP", "FN"], "TN")
log["tau_budget_1pct"] = {"tau": tau, "test_alert_rate": float(A.flag.mean()),
                          "confusion": A.outcome.value_counts().to_dict(),
                          "one_sided": {"tau": tau_os, "test_alert_rate": float(A.flag_os.mean()),
                                        "TP": int((A.flag_os & (A.label == 1)).sum()),
                                        "FP": int((A.flag_os & (A.label == 0)).sum())}}

# 四種結果的平均樣貌（含「之後發生了什麼」：只用於解釋，不是特徵）
PROFILE = ["x", "dist_own_goal", "lateral", "end_x", "end_lateral", "pass_len", "backward", "opp_duel_3s",
           "pt_cross", "pt_launch", "pt_high_pass", "counter_attack", "opp_shot_15s", "sec_to_goal", "events_to_goal"]
prof = A.groupby("outcome")[PROFILE].mean().reindex(["TP", "FP", "FN", "TN"])
prof.insert(0, "n", A.outcome.value_counts())
prof.to_csv(RES / "error_outcome_profile.csv")
print(prof.round(3).T.to_string())

# FP 是不是「危險但沒進」：被標記的非失球中，對手 15 秒內有射門的比例
log["fp_near_miss"] = {"FP_opp_shot_rate": float(A.loc[A.outcome == "FP", "opp_shot_15s"].mean()),
                       "TN_opp_shot_rate": float(A.loc[A.outcome == "TN", "opp_shot_15s"].mean())}

# 依丟球位置：每種方法在各區的召回（前 1%）
k1 = int(round(len(A) * 0.01))
top = {name: np.isin(np.arange(len(A)), np.argsort(-A[col].values, kind="stable")[:k1])
       for name, col in [("single", "single"), ("T2", "score"), ("T2_one_sided", "score_os"), ("T3", "s_xgb")]}
A["zone"] = pd.cut(A.x, [0, 100 / 3, 200 / 3, 100], labels=["own third", "middle", "opp third"], include_lowest=True)
zr = []
for z, d in A.groupby("zone", observed=True):
    row = {"zone": z, "n": len(d), "goals": int(d.label.sum()), "goal_rate": d.label.mean()}
    for name, fl in top.items():
        f = fl[d.index]
        row[f"{name}_flagged"] = int(f.sum())
        row[f"{name}_hits"] = int((f & (d.label.values == 1)).sum())
    zr.append(row)
zone_df = pd.DataFrame(zr)
zone_df.to_csv(RES / "zone_breakdown_top1.csv", index=False)
print(zone_df.round(4).to_string(index=False))

# ---- 每個特徵的貢獻 ----
# Isolation Forest：遮蔽法。把一個特徵換成「正常訓練樣本的中位數」，看分數掉多少（5 個 seed 平均）
#   正值 = 這個特徵讓這筆失誤更像異常
models = [fit_if(normal, DANGER, P, seed) for seed in SEEDS]
med = normal[DANGER].median()


def if_score(d):
    return np.mean([f(d) for f, _, _, _ in models], axis=0)


def occlusion(d):
    base = if_score(d)
    return pd.DataFrame({f: base - if_score(d.assign(**{f: med[f]})) for f in DANGER}, index=d.index)


# XGBoost：TreeSHAP（xgboost 內建 pred_contribs，5 個 seed 平均），單位是 log-odds
xgb_models = [xgb.XGBClassifier(n_estimators=XB["best_iteration"] + 1, learning_rate=0.05, max_depth=XB["max_depth"],
                                min_child_weight=XB["min_child_weight"], subsample=0.8, colsample_bytree=0.8,
                                random_state=seed, n_jobs=2, tree_method="hist").fit(tr[FULL], ytr) for seed in SEEDS]
assert np.allclose(np.mean([m.predict_proba(te[FULL])[:, 1] for m in xgb_models], axis=0), ts.xgb.values, atol=1e-6)


def shap_xgb(d):
    dm = xgb.DMatrix(d[FULL])
    sv = np.mean([m.get_booster().predict(dm, pred_contribs=True) for m in xgb_models], axis=0)
    return pd.DataFrame(sv[:, :-1], columns=FULL, index=d.index)


# 全域：T3 的 mean |SHAP|（全部測試集）；T2 的遮蔽貢獻（被標記的前 1%）
sv_all = shap_xgb(te)
imp = pd.DataFrame({"mean_abs_shap_xgb": sv_all.abs().mean(),     # 方向：特徵值和 SHAP 的相關（正 = 越大越容易失球）
                    "dir_xgb": [np.corrcoef(te[f], sv_all[f])[0, 1] if te[f].std() > 0 and sv_all[f].std() > 0 else 0
                                for f in FULL]})
occ_flag = occlusion(A[A.flag])
imp["if_occlusion_flagged"] = occ_flag.mean().reindex(FULL)
imp.sort_values("mean_abs_shap_xgb", ascending=False).to_csv(RES / "feature_importance.csv")
lap("誤差分析：樣貌、分區、全域貢獻")

# ---- 具體案例 ----
picks = [
    ("TP: highest-scored goal turnover", A[A.label == 1].nlargest(1, "score")),
    ("FP: highest-scored non-goal turnover", A[A.label == 0].nlargest(1, "score")),
    ("FP: flagged, opponent shot but missed", A[(A.outcome == "FP") & (A.opp_shot_15s == 1)].nlargest(1, "score")),
    ("FN: near miss (just below tau)", A[A.outcome == "FN"].nlargest(1, "score")),
    ("FN: lowest-scored goal turnover", A[A.label == 1].nsmallest(1, "score")),
]
cases = pd.concat([d.assign(case=c) for c, d in picks])
names = {}
try:                                    # 有原始資料時補上球隊、球員、比賽名稱，方便對照影片或紀錄
    from wyscout_data import load_all
    db = load_all(ROOT / "data" / "raw")
    for i, r in cases.iterrows():
        mt = db.matches.get(int(r.matchId), {})
        names[i] = {"match": mt.get("label"), "team": db.name("teams", int(r.teamId), "name"),
                    "player": db.name("players", int(r.playerId))}
except Exception as e:                  # noqa: BLE001
    print("（找不到原始資料，案例不附名稱）", e)
cases = cases.join(pd.DataFrame.from_dict(names, orient="index")) if names else cases
occ_c = occlusion(cases).add_prefix("occ_")
shp_c = shap_xgb(cases)
top_shap = shp_c.apply(lambda r: "; ".join(f"{f} {r[f]:+.2f}" for f in r.abs().nlargest(4).index), axis=1)
SHOW = ["case", "match", "team", "player", "competition", "matchId", "eventId", "minute", "x", "y", "end_x",
        "end_lateral", "pass_len", "backward", "opp_duel_3s", "pt_cross", "pt_launch", "counter_attack",
        "pct", "single", "s_xgb", "opp_shot_15s", "sec_to_goal", "events_to_goal", "label"]
case_df = cases[[c for c in SHOW if c in cases]].join(occ_c).assign(xgb_top_shap=top_shap)
case_df.to_csv(RES / "error_cases.csv", index=False)
print(case_df.T.to_string())
lap("誤差分析：具體案例")

# ===========================================================================
# 圖
# ===========================================================================
C = {"single": "#8c8c8c", "T2": "#4C78A8", "T2os": "#54A24B", "T3": "#F58518", "neg": "#9ecae9", "pos": "#e45756"}

# 圖 1：探索（只用訓練集）正常 vs 致命
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, f, bins, xl in [(ax[0], "end_x", np.arange(0, 101, 4), "end_x (0 = own goal line)"),
                       (ax[1], "dist_own_goal", np.arange(0, 121, 5), "distance to own goal (m)")]:
    a.hist(tr.loc[ytr == 0, f], bins=bins, density=True, alpha=.6, color=C["neg"], label="no goal (normal)")
    a.hist(tr.loc[ytr == 1, f], bins=bins, density=True, alpha=.6, color=C["pos"], label="goal conceded ≤15 s")
    a.set_xlabel(xl); a.set_ylabel("density")
ax[0].legend(fontsize=8)
rate = tr.groupby("backward").label.mean() * 100
ax[2].bar(["forward / lateral", "backward"], rate.values, color=[C["neg"], C["pos"]])
ax[2].set_ylabel("goal-conceded rate (%)")
for i, v in enumerate(rate.values):
    ax[2].text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
fig.suptitle("Training split only", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig1_eda_train.png", dpi=160); plt.close(fig)

# 圖 2：PR 曲線（測試集）
fig, ax = plt.subplots(figsize=(5.2, 4))
for lab, s, c in [("T1 single feature (end_x)", ts.single.values, C["single"]), ("T2 IF two-sided (primary)", s_t2, C["T2"]),
                  ("T2 IF one-sided (ablation A2)", s_os, C["T2os"]), ("T3 XGBoost", ts.xgb.values, C["T3"])]:
    p, r, _ = precision_recall_curve(yte, s)
    ax.plot(r, p, c=c, lw=1.4, label=lab)
ax.axhline(yte.mean(), ls="--", c="k", lw=.8, label=f"random ({yte.mean():.4f})")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, .2); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig2_pr_curves.png", dpi=160); plt.close(fig)

# 圖 3：消融（PR-AUC ± std）
fig, ax = plt.subplots(figsize=(8, 7))
d = abl_df[abl_df.ablation != "A4 threshold"].iloc[::-1]
cols = d.ablation.map({"A1 contamination": "#B279A2", "A2 scoring": C["T2os"], "A3 features": C["T2"]})
ax.barh(range(len(d)), d.pr_auc, xerr=d.pr_auc_std, color=cols)
ax.set_yticks(range(len(d)), [f"{a[:2]} {s}" for a, s in zip(d.ablation, d.setting)], fontsize=7)
ax.axvline(summarize("", ref_runs)["pr_auc"], c=C["T2"], ls="--", lw=.8, label="primary (IF two-sided, danger 6)")
ax.axvline(metrics(yte, ts.single.values)["pr_auc"], c=C["single"], ls=":", lw=1, label="T1 single feature")
ax.set_xlabel("test PR-AUC (mean ± std over 5 seeds)"); ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig3_ablations.png", dpi=160); plt.close(fig)

# 圖 4：特徵貢獻
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t = imp.sort_values("mean_abs_shap_xgb").tail(12)
ax[0].barh(t.index, t.mean_abs_shap_xgb, color=np.where(t.dir_xgb > 0, C["pos"], C["neg"]))
ax[0].set_xlabel("mean |SHAP| (log-odds)")
ax[0].set_title("(a) T3 XGBoost, all test turnovers\nred = higher value → more likely goal", fontsize=9)
o = occ_flag.mean().sort_values()
ax[1].barh(o.index, o.values, color=C["T2"])
ax[1].set_xlabel("score drop when replaced by normal median")
ax[1].set_title("(b) T2 Isolation Forest, flagged top 1%\nwhat makes them anomalous", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig4_feature_contributions.png", dpi=160); plt.close(fig)
lap("畫圖")

(RES / "ablation_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(abl_df[["ablation", "setting", "val_pr_auc", "pr_auc", "pr_auc_std", "roc_auc", "hits_at_1", "hits_at_10",
              "precision", "recall", "f1", "alert_rate"]].round(4).to_string(index=False))
print(json.dumps(log, indent=1, ensure_ascii=False, default=float))
summary("消融與誤差分析", RES / "timing_ablation.csv")
