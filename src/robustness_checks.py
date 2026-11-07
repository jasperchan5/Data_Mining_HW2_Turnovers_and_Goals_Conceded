"""
穩健性檢查與修正方案的實測（沿用主實驗的切分、Tier 2 候選與超參數）
    先跑 turnover_experiment.py

R1 落點補值有沒有洩漏未來資訊
    (a) 只用傳球當下的資訊：補值的失誤一律把落點設成傳球起點（end_x = x、end_lateral = lateral、
        pass_len = 0、backward = 0），三層全部重新訓練、評估
    (b) 只評估有記錄落點的失誤：主實驗的分數不變，測試集排除補值的失誤
R2 修正方案一：標籤改成「15 秒內被射門」（誤差分析發現 FP 多半是危險但沒進）
    同樣的方法、超參數，只換標籤；Tier 2 的「正常」改成沒有被射門的失誤
R3 修正方案二 / 檢驗 H3：加入失誤後 5 秒的發展（對手動作數、本隊動作數、球最接近自家球門的位置）
    這些特徵用到失誤之後的資訊，不能部署，只用來檢驗「漏掉的致命失誤是不是取決於之後發生的事」
R4 可部署的修正：傳球當下的情境特徵（XGBoost 危險 6 個 vs 全部 28 個）能不能抓回延遲的失球
R5 leave-one-league-out：只用測試期之前的比賽，鎖定所有設定，每次拿掉一個聯賽當評估集
    （單邊的想法來自看過測試集的觀察，這是唯一完全不受那個觀察影響的檢查）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANDIDATES, FULL, RES, SEEDS, best_f1_tau, fit_if, load_split, metrics, summarize  # noqa: E402
from timing import lap, summary  # noqa: E402

pd.set_option("display.width", 250)
tr0, va0, te0, _ = load_split()
main_log = json.loads((RES / "log.json").read_text())
T2 = main_log["T2_config"]
FEATS, TF = CANDIDATES[T2]
P = main_log["iforest_best"][T2]["params"]
XB = main_log["xgb_best"]
ts = pd.read_csv(RES / "test_scores.csv")
assert (ts.eventId.values == te0.eventId.values).all()
POST = ["post_opp_events_5s", "post_own_events_5s", "post_ball_min_x_5s"]
log, rows = {"T2_config": T2}, []
lap("讀取資料")


def xgb_model(seed):
    return XGBClassifier(n_estimators=XB["best_iteration"] + 1, learning_rate=0.05, max_depth=XB["max_depth"],
                         min_child_weight=XB["min_child_weight"], subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, n_jobs=2, tree_method="hist")


def three_tiers(tr, va, te, label, tag, xgb_feats=FULL, single_feat="end_x"):
    """同一套做法跑 T1 單一特徵、T2 主方法（只用 label = 0 的訓練樣本）、T3 XGBoost；回傳分數（5 個 seed 平均）"""
    ytr, yva, yte = tr[label].values, va[label].values, te[label].values
    auc = roc_auc_score(ytr, tr[single_feat])
    sign = 1 if auc >= 0.5 else -1                      # 方向用訓練集決定
    s1 = sign * te[single_feat].values
    out = {"single": s1}
    rows.append({"check": tag, **summarize(f"T1 single feature ({single_feat})",
                                           [metrics(yte, s1, best_f1_tau(yva, sign * va[single_feat].values))])})
    runs, sl = [], []
    for seed in tqdm(SEEDS, desc=f"{tag}：T2", unit="seed", leave=False):
        f, _, _, _ = fit_if(tr[ytr == 0], FEATS, P, seed, TF)
        sl.append(f(te))
        runs.append(metrics(yte, sl[-1], best_f1_tau(yva, f(va))))
    out["t2"] = np.mean(sl, axis=0)
    rows.append({"check": tag, **summarize(f"T2 IF {T2}", runs)})
    runs, sl = [], []
    for seed in tqdm(SEEDS, desc=f"{tag}：T3", unit="seed", leave=False):
        m = xgb_model(seed).fit(tr[xgb_feats], ytr)
        sl.append(m.predict_proba(te[xgb_feats])[:, 1])
        runs.append(metrics(yte, sl[-1], best_f1_tau(yva, m.predict_proba(va[xgb_feats])[:, 1])))
    out["xgb"] = np.mean(sl, axis=0)
    rows.append({"check": tag, **summarize("T3 XGBoost" + (" + post-turnover 5 s" if xgb_feats != FULL else ""), runs)})
    for r in rows[-3:]:
        tqdm.write(f"  {tag:34s} {r['method']:42s} PR-AUC {r['pr_auc']:.4f} ± {r['pr_auc_std']:.4f}  "
                   f"前 1% {r['hits_at_1']:.1f}  前 10% {r['hits_at_10']:.1f}")
    return out


# ---- 參照：主實驗（標籤 = 15 秒內失球）----
for name, col in [("T1 single feature (end_x)", "single"), (f"T2 IF {T2}", "t2"), ("T3 XGBoost", "xgb")]:
    rows.append({"check": "main (reference)", **summarize(name, [metrics(te0.label.values, ts[col].values)])})

# ---- R1 (a) 只用傳球當下的資訊 ----
def pass_time_only(d):
    d = d.copy()
    imp = d.end_source != "recorded"
    d.loc[imp, "end_x"] = d.loc[imp, "x"]
    d.loc[imp, "end_lateral"] = d.loc[imp, "lateral"]
    d.loc[imp, "pass_len"] = 0.0
    d.loc[imp, "backward"] = 0
    return d


tr1, va1, te1 = (pass_time_only(d) for d in (tr0, va0, te0))
log["imputed"] = {s: {"n": int((d.end_source != "recorded").sum()), "positives": int(d.label[d.end_source != "recorded"].sum())}
                  for s, d in (("train", tr0), ("val", va0), ("test", te0))}
print("補值的失誤（各切分）：", log["imputed"])
three_tiers(tr1, va1, te1, "label", "R1a pass-time-only end points")
lap("R1 (a) 只用傳球當下的資訊")

# ---- R1 (b) 只評估有記錄落點的失誤 ----
rec = (te0.end_source == "recorded").values
for name, col in [("T1 single feature (end_x)", "single"), (f"T2 IF {T2}", "t2"), ("T3 XGBoost", "xgb")]:
    rows.append({"check": "R1b recorded end points only (test subset)",
                 **summarize(name, [metrics(te0.label.values[rec], ts[col].values[rec])])})
lap("R1 (b) 只評估有記錄落點的失誤")

# ---- R2 標籤改成 15 秒內被射門 ----
for d in (tr0, va0, te0):
    d["label_shot"] = d.opp_shot_15s.astype(int)
log["shot_label"] = {s: {"n": len(d), "positives": int(d.label_shot.sum()), "rate": float(d.label_shot.mean())}
                     for s, d in (("train", tr0), ("val", va0), ("test", te0))}
print("射門標籤：", log["shot_label"])
shot = three_tiers(tr0, va0, te0, "label_shot", "R2 label = opponent shot within 15 s")
# 用「失球」訓練的主方法，拿「射門」標籤來評估：FP 裡有多少其實是讓對手射門
ys = te0.label_shot.values
k = int(round(len(te0) * 0.01))
top_goal_model = np.argsort(-ts.t2.values, kind="stable")[:k]
log["goal_model_on_shot_label"] = {
    "pr_auc_shot": float(average_precision_score(ys, ts.t2.values)),
    "precision_at_1pct_goal_label": float(te0.label.values[top_goal_model].mean()),
    "precision_at_1pct_shot_label": float(ys[top_goal_model].mean()),
    "shot_rate_test": float(ys.mean()), "goal_rate_test": float(te0.label.mean()),
}
print(log["goal_model_on_shot_label"])
lap("R2 射門標籤")

# ---- R3 加入失誤後 5 秒的發展（診斷用，不可部署）----
post = three_tiers(tr0, va0, te0, "label", "R3 + post-turnover 5 s (diagnostic)", xgb_feats=FULL + POST)
yte = te0.label.values
delayed = (te0.events_to_goal >= 4).values & (yte == 1)
quick = (te0.events_to_goal <= 3).values & (yte == 1)
r3 = {}
for name, s in [("T3 XGBoost", ts.xgb.values), ("T3 XGBoost + post 5 s", post["xgb"]), (f"T2 IF {T2}", ts.t2.values)]:
    top = np.zeros(len(te0), bool)
    top[np.argsort(-s, kind="stable")[:k]] = True
    top10 = np.zeros(len(te0), bool)
    top10[np.argsort(-s, kind="stable")[:10 * k]] = True
    r3[name] = {"quick_goals": int(quick.sum()), "delayed_goals": int(delayed.sum()),
                "hits1_quick": int((top & quick).sum()), "hits1_delayed": int((top & delayed).sum()),
                "hits10_quick": int((top10 & quick).sum()), "hits10_delayed": int((top10 & delayed).sum())}
log["H3_delayed_vs_quick"] = r3
print(pd.DataFrame(r3).T.to_string())
lap("R3 失誤後 5 秒的發展")

# ---- R4 可部署的修正：傳球當下的情境特徵能不能抓回延遲的失球 ----
# 比較 XGBoost 只用危險 6 個 vs 再加上 22 個傳球當下就知道的特徵（反擊、前 10 秒雙方動作數、本隊最後防線…）
from common import DANGER  # noqa: E402
r5 = {}
for name, feats in [("XGBoost danger 6", DANGER), ("XGBoost all 28 (adds pass-time context)", FULL)]:
    sl = [xgb_model(seed).fit(tr0[feats], tr0.label.values).predict_proba(te0[feats])[:, 1] for seed in SEEDS]
    s = np.mean(sl, axis=0)
    order = np.argsort(-s, kind="stable")
    top, top10 = np.zeros(len(te0), bool), np.zeros(len(te0), bool)
    top[order[:k]] = True
    top10[order[:10 * k]] = True
    r5[name] = {"hits1_quick": int((top & quick).sum()), "hits1_delayed": int((top & delayed).sum()),
                "hits10_quick": int((top10 & quick).sum()), "hits10_delayed": int((top10 & delayed).sum())}
log["R4_context_features_delayed"] = r5
print(pd.DataFrame(r5).T.to_string())
lap("R4 傳球當下的情境特徵")

# ---- R5 換一個聯賽評估（leave-one-league-out），只用測試期之前的比賽 ----
# 單邊的想法來自看過測試集的觀察（§6.2），所以這裡完全不用測試期的比賽：
# 在訓練 + 驗證期間（2018-04-07 以前）的比賽中，每次拿掉一個聯賽當評估集，其他聯賽與國際賽當訓練集
# 方法、超參數、方向完全鎖定（沿用主實驗驗證集選出的設定），不再調整
pre = pd.concat([tr0, va0], ignore_index=True)
P_TWO = main_log["iforest_best"]["danger (6), two-sided"]["params"]
LEAGUES = ["English first division", "French first division", "German first division",
           "Italian first division", "Spanish first division"]
lolo = []
for lg in tqdm(LEAGUES, desc="R5 leave-one-league-out", unit="聯賽"):
    trn, ev = pre[pre.competition != lg], pre[pre.competition == lg]
    y = ev.label.values
    row = {"league": lg, "n": len(ev), "positives": int(y.sum()),
           "single_end_x": average_precision_score(y, -ev.end_x.values)}
    nrm = trn[trn.label == 0]
    for name, feats, tf, prm in [("IF two-sided", FEATS, None, P_TWO), ("IF one-sided", FEATS, "one_sided", P)]:
        prs, h1 = [], []
        for seed in SEEDS:
            f, _, _, _ = fit_if(nrm, feats, prm, seed, tf)
            m = metrics(y, f(ev))
            prs.append(m["pr_auc"]); h1.append(m["hits_at_1"])
        row[name], row[f"{name} std"], row[f"{name} hits@1%"] = np.mean(prs), np.std(prs), np.mean(h1)
    row["single hits@1%"] = metrics(y, -ev.end_x.values)["hits_at_1"]
    xs = [xgb_model(seed).fit(trn[FULL], trn.label.values).predict_proba(ev[FULL])[:, 1] for seed in SEEDS]
    row["XGBoost"] = np.mean([average_precision_score(y, x) for x in xs])
    lolo.append(row)
lolo = pd.DataFrame(lolo)
lolo.to_csv(RES / "leave_one_league_out.csv", index=False)
log["R5_leave_one_league_out"] = {
    "one_sided_beats_two_sided": int((lolo["IF one-sided"] > lolo["IF two-sided"]).sum()),
    "one_sided_beats_single": int((lolo["IF one-sided"] > lolo["single_end_x"]).sum()),
    "mean": lolo[["single_end_x", "IF two-sided", "IF one-sided", "XGBoost"]].mean().to_dict()}
print(lolo.round(4).to_string(index=False))
print(log["R5_leave_one_league_out"])
lap("R5 leave-one-league-out")

out = pd.DataFrame(rows)
out.to_csv(RES / "robustness_checks.csv", index=False)
(RES / "robustness_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(out[["check", "method", "pr_auc", "pr_auc_std", "roc_auc", "hits_at_1", "hits_at_10", "p_at_1"]].round(4).to_string(index=False))
summary("穩健性檢查", RES / "timing_robustness.csv")
