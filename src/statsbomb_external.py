"""
外部測試：StatsBomb Open Data Premier League 2015/16（380 場；和 Wyscout 2017/18 完全不同的賽季，整個研究過程都沒看過）

流程完全鎖定，不重新訓練、不調參：
    - 特徵：用和 build_turnovers.py 相同的定義，從 StatsBomb 事件算出同樣的 28 個特徵
    - 模型：在 Wyscout 訓練集上 fit 的 scaler + Isolation Forest（單邊主方法、雙邊）、XGBoost，超參數照 log.json
    - τ：Wyscout 驗證集分數的 99 百分位（審查預算 1%）
    - 單一特徵規則：−end_x（方向沿用 Wyscout 訓練集）

兩家資料的定義差異（無法完全一致，報告中說明）
    - 失誤：StatsBomb 的 Pass 有 outcome（Incomplete / Out / Pass Offside / Unknown）、且不是定位球
      （Wyscout 的定位球是 Free Kick 事件，不算 Pass）
    - 標籤：對手 15 秒內射門進球（shot.outcome = Goal），或我方 Own Goal Against
    - 座標：StatsBomb 120 × 80（每個事件都是該隊視角、往 x = 120 進攻）→ 換成 0–100
    - 傳球種類、球員位置、前 10 秒事件數等情境特徵只能近似（見 PASS_TYPE_RULES、ROLE_RULES、COUNTED）
      危險 6 個特徵中只有「對手 3 秒內有對抗」需要近似：Wyscout 的對抗兩隊各記一筆，所以等於
      「3 秒內有任何對抗」；StatsBomb 對應 Duel、50/50、Dribble、Dribbled Past 事件

用法
    python src/download_statsbomb.py      # 先下載（約 1 GB，放在 data/external/statsbomb/，不進 git）
    python src/statsbomb_external.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANDIDATES, DANGER, FULL, RES, ROOT, SEEDS, bootstrap_gap, fit_if, load_split, metrics  # noqa: E402
from timing import lap, step, summary  # noqa: E402

SB = ROOT / "data" / "external" / "statsbomb"
OUT = ROOT / "data" / "processed" / "turnovers_statsbomb_epl_2015_16.csv"
MATCH_FILES = ["2_27.json"]                     # Premier League 2015/16（和 download_statsbomb.py 的 COMPS 對應）
WINDOW, CONTEXT = 15.0, 10.0
PERIOD = {1: ("1H", 0), 2: ("2H", 45), 3: ("E1", 90), 4: ("E2", 105)}
PITCH_L, PITCH_W = 105.0, 68.0
SET_PIECES = {"Corner", "Free Kick", "Goal Kick", "Throw-in", "Kick Off"}
FAILED = {"Incomplete", "Out", "Pass Offside", "Unknown"}
DUELS = {"Duel", "50/50", "Dribble", "Dribbled Past"}
# 前 10 秒的「事件數」：Wyscout 沒有 Carry、Ball Receipt、Pressure 這類事件，算的時候排除
NOT_COUNTED = {"Ball Receipt*", "Carry", "Pressure", "Starting XI", "Half Start", "Half End", "Substitution",
               "Tactical Shift", "Player On", "Player Off", "Injury Stoppage", "Camera On", "Camera off",
               "Referee Ball-Drop"}


def secs(e):
    h, m, s = e["timestamp"].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def xy(loc):
    """StatsBomb 120 × 80 → Wyscout 0–100（兩家都是該隊視角、往 x 大的方向進攻）"""
    return loc[0] / 1.2, loc[1] / 0.8


def pass_type(p):
    """Wyscout 的 7 種傳球（近似）：依序判斷，第一個符合的就是"""
    body = (p.get("body_part") or {}).get("name", "")
    if body == "Keeper Arm":
        return "hand_pass"
    if body == "Head":
        return "head_pass"
    if p.get("cross"):
        return "cross"
    if p.get("through_ball") or (p.get("technique") or {}).get("name") == "Through Ball":
        return "smart_pass"
    if (p.get("height") or {}).get("name") == "High Pass":
        return "launch" if p.get("length", 0) >= 40 else "high_pass"  # 長距離的高球 ≈ Wyscout 的 Launch
    return "simple_pass"


def role(pos):
    if not pos:
        return "NA"
    if pos == "Goalkeeper":
        return "GK"
    if "Back" in pos:
        return "DF"
    if "Forward" in pos or pos.endswith("Wing"):
        return "FW"
    return "MD"


def match_rows(es, match):
    home = match["home_team"]["home_team_id"]
    es = [e for e in es if e["period"] in PERIOD]
    es.sort(key=lambda e: (e["period"], secs(e), e["index"]))
    teams = {match["home_team"]["home_team_id"], match["away_team"]["away_team_id"]}
    other = lambda t: next(x for x in teams if x != t)
    goals = []
    for e in es:
        if e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal":
            goals.append((e["period"], secs(e), e["team"]["id"]))
        if e["type"]["name"] == "Own Goal Against":
            goals.append((e["period"], secs(e), other(e["team"]["id"])))
    score, losses, rows = defaultdict(int), defaultdict(int), []
    for i, e in enumerate(es):
        tid, per, sec = e["team"]["id"], e["period"], secs(e)
        if e["type"]["name"] == "Pass" and "location" in e:
            p = e["pass"]
            if (p.get("outcome") or {}).get("name") in FAILED and (p.get("type") or {}).get("name") not in SET_PIECES:
                opp = other(tid)
                x0, y0 = xy(e["location"])
                x1, y1 = xy(p["end_location"])
                ctx = [c for c in es[max(0, i - 120):i] if c["period"] == per and sec - secs(c) <= CONTEXT
                       and c["type"]["name"] not in NOT_COUNTED]
                own = [c for c in ctx if c["team"]["id"] == tid]
                oppc = [c for c in ctx if c["team"]["id"] == opp]
                duel3 = any(c["type"]["name"] in DUELS and sec - secs(c) <= 3 for c in ctx)
                gs = [s for pp, s, tm in goals if pp == per and 0 <= s - sec <= WINDOW and tm == opp]
                pt = pass_type(p)
                rows.append({
                    "matchId": match["match_id"], "eventId": e["id"], "date": match["match_date"],
                    "competition": match["competition"]["competition_name"], "teamId": tid,
                    "x": x0, "y": y0, "lateral": abs(y0 - 50),
                    "dist_own_goal": float(np.hypot(x0 / 100 * PITCH_L, (y0 - 50) / 100 * PITCH_W)),
                    "in_own_third": int(x0 < 100 / 3),
                    "end_x": x1, "end_lateral": abs(y1 - 50),
                    "pass_len": float(np.hypot((x1 - x0) / 100 * PITCH_L, (y1 - y0) / 100 * PITCH_W)),
                    "backward": int(x1 < x0),
                    **{f"pt_{k}": int(pt == k) for k in ["simple_pass", "high_pass", "launch", "cross", "head_pass",
                                                       "smart_pass", "hand_pass"]},
                    "counter_attack": int(e["play_pattern"]["name"] == "From Counter"),
                    "minute": PERIOD[per][1] + sec / 60,
                    "score_diff": score[tid] - score[opp],
                    "home": int(tid == home),
                    "team_prev_losses": losses[tid],
                    "own_events_10s": len(own), "opp_events_10s": len(oppc),
                    "opp_duel_3s": int(duel3),
                    "own_min_x_10s": min((xy(c["location"])[0] for c in own if "location" in c), default=x0),
                    "role": role((e.get("position") or {}).get("name")),
                    "sb_under_pressure": int(bool(e.get("under_pressure"))),   # 只用於比較，不是特徵
                    "sec_to_goal": min(gs) - sec if gs else np.nan,
                    "label": int(bool(gs)),
                })
                losses[tid] += 1
        if e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal":
            score[tid] += 1
        if e["type"]["name"] == "Own Goal Against":
            score[other(tid)] += 1
    return rows


def build():
    matches = [m for f in MATCH_FILES for m in json.loads((SB / "matches" / f).read_text(encoding="utf-8"))]
    missing = [m["match_id"] for m in matches if not (SB / "events" / f"{m['match_id']}.json").exists()]
    if missing:
        sys.exit(f"缺少 {len(missing)} 場的事件檔，請先執行 python src/download_statsbomb.py")
    rows = []
    for m in tqdm(matches, desc="StatsBomb 比賽", unit="場"):
        es = json.loads((SB / "events" / f"{m['match_id']}.json").read_text(encoding="utf-8"))
        rows.extend(match_rows(es, m))
    df = pd.DataFrame(rows)
    for r in ["DF", "MD", "FW", "GK"]:
        df[f"role_{r}"] = (df.role == r).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


# ---------------------------------------------------------------------------
if OUT.exists() and "--rebuild" not in sys.argv:
    sb = pd.read_csv(OUT)
    lap("讀取 StatsBomb 失誤表")
else:
    with step("建立 StatsBomb 失誤表"):
        sb = build()
log = {"n": len(sb), "matches": int(sb.matchId.nunique()), "positives": int(sb.label.sum()),
       "rate": float(sb.label.mean()), "by_competition": sb.groupby("competition").label.agg(["size", "sum"]).to_dict()}
print(log)

tr, va, te, _ = load_split()
main_log = json.loads((RES / "log.json").read_text())
normal = tr[tr.label == 0]
ys, g = sb.label.values, sb.matchId.values
# 特徵分布對照（危險 6 個）：兩家資料是否可比
cmp = pd.DataFrame({"wyscout_train_mean": tr[DANGER + ["pass_len"]].mean(),
                    "statsbomb_mean": sb[DANGER + ["pass_len"]].mean()})
cmp.to_csv(RES / "external_feature_comparison.csv")
print(cmp.round(3).to_string())

rows, scores = [], {}
k = int(round(len(sb) * 0.01))


def evaluate(name, s, tau=None):
    m = metrics(ys, s, tau)
    rows.append({"method": name, **m})
    scores[name] = s
    tqdm.write(f"  {name:38s} PR-AUC {m['pr_auc']:.4f}  ROC {m['roc_auc']:.3f}  前 1% {m['hits_at_1']}  前 10% {m['hits_at_10']}")


evaluate("random (base rate)", np.random.default_rng(0).random(len(sb)))
evaluate("T1 single feature (-end_x)", -sb.end_x.values)
for cand in ["danger (6), two-sided", "danger (6), one-sided"]:
    feats, tf = CANDIDATES[cand]
    P = main_log["iforest_best"][cand]["params"]
    s_sb, s_va = [], []
    for seed in tqdm(SEEDS, desc=f"IF {cand}", unit="seed", leave=False):
        f, _, _, _ = fit_if(normal, feats, P, seed, tf)
        s_sb.append(f(sb))
        s_va.append(f(va))
    tau = float(np.quantile(np.mean(s_va, axis=0), 0.99))      # Wyscout 驗證集的 99 百分位，鎖定
    s = np.mean(s_sb, axis=0)
    evaluate(f"T2 IF {cand}" + (" (primary)" if cand == main_log["T2_config"] else ""), s, tau)
    log[f"tau_{cand}"] = tau
XB = main_log["xgb_best"]
xs = [XGBClassifier(n_estimators=XB["best_iteration"] + 1, learning_rate=0.05, max_depth=XB["max_depth"],
                    min_child_weight=XB["min_child_weight"], subsample=0.8, colsample_bytree=0.8, random_state=seed,
                    n_jobs=2, tree_method="hist").fit(tr[FULL], tr.label.values).predict_proba(sb[FULL])[:, 1]
      for seed in tqdm(SEEDS, desc="XGBoost", unit="seed", leave=False)]
evaluate("T3 XGBoost (trained on Wyscout)", np.mean(xs, axis=0))
lap("鎖定的模型套到 StatsBomb")

PRIM = [n for n in scores if "(primary)" in n][0]
TWO = "T2 IF danger (6), two-sided"
log["significance"] = {
    "one_sided_minus_two_sided": bootstrap_gap(ys, scores[PRIM], scores[TWO], g, desc="單邊 − 雙邊"),
    "one_sided_minus_single": bootstrap_gap(ys, scores[PRIM], scores["T1 single feature (-end_x)"], g, desc="單邊 − 單一特徵"),
    "two_sided_minus_single": bootstrap_gap(ys, scores[TWO], scores["T1 single feature (-end_x)"], g, desc="雙邊 − 單一特徵"),
    "xgb_minus_one_sided": bootstrap_gap(ys, scores["T3 XGBoost (trained on Wyscout)"], scores[PRIM], g, desc="XGB − 單邊"),
}
for kk, v in log["significance"].items():
    print(f"  {kk:28s} {v[0]:+.4f}  95% CI [{v[1]:+.4f}, {v[2]:+.4f}]")
lap("bootstrap（4 組 × 1000 次）")

# 各聯賽
per = []
for comp, d in sb.groupby("competition"):
    idx = d.index.values
    r = {"competition": comp, "n": len(d), "positives": int(d.label.sum())}
    for n, s in scores.items():
        r[n] = average_precision_score(d.label, s[idx]) if d.label.sum() else np.nan
    per.append(r)
per = pd.DataFrame(per)
per.to_csv(RES / "external_by_league.csv", index=False)
print(per.round(4).to_string(index=False))

# 對方三分之一的警示（H2 的機制在新資料上是否一樣）
zone = pd.cut(sb.x, [0, 100 / 3, 200 / 3, 100], labels=["own", "mid", "opp"], include_lowest=True)
log["top1_by_zone"] = {}
for n in [PRIM, TWO, "T1 single feature (-end_x)"]:
    top = np.argsort(-scores[n], kind="stable")[:k]
    log["top1_by_zone"][n] = {z: {"alerts": int((zone.iloc[top] == z).sum()),
                                  "hits": int(((zone.iloc[top] == z) & (ys[top] == 1)).sum())} for z in ["own", "mid", "opp"]}
print(json.dumps(log["top1_by_zone"], indent=1))

out = pd.DataFrame(rows)
out.to_csv(RES / "external_statsbomb.csv", index=False)
(RES / "external_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(out[["method", "pr_auc", "roc_auc", "hits_at_1", "hits_at_10", "p_at_1"]].round(4).to_string(index=False))
summary("外部測試（StatsBomb Premier League 2015/16）", RES / "timing_external.csv")
