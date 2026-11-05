"""
建立「一次傳球失誤一列」的資料表

異常定義（看結果之前就固定）
    一筆資料 = 一次不準確的傳球（eventName = Pass，tag 1802 not accurate），也就是把球權傳丟
    標籤 = 傳丟之後 15 秒內、同一個半場，對手進球（射門或自由球帶 tag 101，或我方烏龍球 tag 102）

特徵只用傳球當下（含之前）的資訊；不用：
    - 傳球之後的任何事件（對手怎麼推進、有沒有射門）
    - tag 2001 dangerous_ball_lost：Wyscout 標註員可能是看完後續才標的，有洩漏風險
對抗（Duel）失敗不算丟球權：Wyscout 的對抗兩邊各記一筆，防守方輸了對抗並不是丟球權
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wyscout_data import iter_events, load_all  # noqa: E402
from timing import fmt, record, step, summary  # noqa: E402

WINDOW = 15.0
CONTEXT = 10.0
PERIOD_ORDER = {"1H": 0, "2H": 1, "E1": 2, "E2": 3, "P": 4}
PERIOD_OFFSET = {"1H": 0, "2H": 45, "E1": 90, "E2": 105}
PITCH_L, PITCH_W = 105.0, 68.0
PASS_TYPES = ["Simple pass", "High pass", "Launch", "Cross", "Head pass", "Smart pass", "Hand pass"]


def tags(e):
    return {t["id"] for t in e["tags"]}


end_counts = defaultdict(int)


def is_placeholder(q):
    """四個角落 (0|100, 0|100) 是 Wyscout 的佔位座標（守門員出擊、撲救、比賽中斷常見），不是真的位置"""
    return q["x"] in (0, 100) and q["y"] in (0, 100)


def fix_end(p0, p1, e, es, i, max_gap=5.0):
    """
    落點 (0, 0) 是 Wyscout 沒記錄落點時的佔位值（多半是被解圍或被守門員接走的傳中），不是真的傳到自家底線。
    用「之後第一個有真實座標的事件」推回落點：
        - 往後找同一個半場、相隔不超過 max_gap 秒的事件，跳過：沒有座標的、比賽中斷（Interruption）、
          座標是四個角落佔位值的（守門員出擊 / 撲救事件常記成 (100, 100)，換成我方視角會變成 (0, 0) 附近，
          等於又把傳中放回自家底線）
        - 找到的事件是對手的：座標換回我方視角 (100 - x, 100 - y)；是本隊的：直接用
        - 找不到：退回用傳球起點
    之後的事件只用來補回「這次傳球本身的落點」，不用它的任何其他資訊。
    """
    if not (p1["x"] == 0 and p1["y"] == 0):
        return p1, "recorded"
    for nx in es[i + 1:i + 10]:
        if nx["matchPeriod"] != e["matchPeriod"] or nx["eventSec"] - e["eventSec"] > max_gap:
            break
        if not nx["positions"] or nx["eventName"] == "Interruption" or is_placeholder(nx["positions"][0]):
            continue
        q = nx["positions"][0]
        if nx["teamId"] != e["teamId"]:
            return {"x": 100 - q["x"], "y": 100 - q["y"]}, "imputed_next_opp"
        return {"x": q["x"], "y": q["y"]}, "imputed_next_own"
    return p0, "imputed_start"


def match_rows(es, match, roles):
    es = sorted(es, key=lambda e: (PERIOD_ORDER.get(e["matchPeriod"], 9), e["eventSec"]))
    teams = list(match["teamsData"])
    sides = {t: match["teamsData"][t]["side"] for t in teams}
    other = lambda t: next((x for x in teams if x != t), None)

    goals = []                                  # (period, sec, scoring team)
    shots = []                                  # (period, sec, shooting team)：只用於誤差分析
    for e in es:
        if e["matchPeriod"] not in PERIOD_OFFSET:
            continue
        t = tags(e)
        if e["eventName"] == "Shot" or (e["eventName"] == "Free Kick" and e["subEventName"] in ("Free kick shot", "Penalty")):
            shots.append((e["matchPeriod"], e["eventSec"], e["teamId"]))
        if 101 in t and e["eventName"] in ("Shot", "Free Kick"):
            goals.append((e["matchPeriod"], e["eventSec"], e["teamId"]))
        if 102 in t:
            goals.append((e["matchPeriod"], e["eventSec"], other(e["teamId"])))

    score = defaultdict(int)
    losses_so_far = defaultdict(int)
    rows = []
    for i, e in enumerate(es):
        period, tid = e["matchPeriod"], e["teamId"]
        if period not in PERIOD_OFFSET:
            continue
        t = tags(e)
        sec = e["eventSec"]
        if e["eventName"] == "Pass" and 1802 in t and e["positions"]:
            opp = other(tid)
            p0 = e["positions"][0]
            p1 = e["positions"][1] if len(e["positions"]) > 1 else p0
            p1, end_source = fix_end(p0, p1, e, es, i)
            end_counts[end_source] += 1
            ctx = [c for c in es[max(0, i - 40):i] if c["matchPeriod"] == period and sec - c["eventSec"] <= CONTEXT]
            own = [c for c in ctx if c["teamId"] == tid]
            oppc = [c for c in ctx if c["teamId"] == opp]
            in_window = lambda lst: [s for p, s, tm in lst if p == period and 0 <= s - sec <= WINDOW and tm == opp]
            goal_secs = in_window(goals)
            label = int(bool(goal_secs))
            # 以下三欄是「失誤之後」發生的事，只用於誤差分析（解釋誤判），絕不當特徵
            sec_to_goal = min(goal_secs) - sec if goal_secs else np.nan
            after = {
                "sec_to_goal": sec_to_goal,
                "events_to_goal": (sum(1 for c in es[i + 1:i + 80] if c["matchPeriod"] == period
                                       and sec < c["eventSec"] <= sec + sec_to_goal) if goal_secs else np.nan),
                "opp_shot_15s": int(bool(in_window(shots))),
            }
            row = {
                "matchId": match["wyId"], "eventId": e["id"], "date": match["dateutc"],
                "competitionId": match["competitionId"], "teamId": tid, "playerId": e["playerId"],
                "period": period, "eventSec": sec,
                # 丟球的位置（我方視角，x = 0 是自家球門）
                "x": p0["x"], "y": p0["y"], "lateral": abs(p0["y"] - 50),
                "dist_own_goal": float(np.hypot(p0["x"] / 100 * PITCH_L, (p0["y"] - 50) / 100 * PITCH_W)),
                "in_own_third": int(p0["x"] < 100 / 3),
                # 傳球本身
                "end_x": p1["x"], "end_lateral": abs(p1["y"] - 50),
                "pass_len": float(np.hypot((p1["x"] - p0["x"]) / 100 * PITCH_L, (p1["y"] - p0["y"]) / 100 * PITCH_W)),
                "backward": int(p1["x"] < p0["x"]),
                **{f"pt_{pt.replace(' ', '_').lower()}": int(e["subEventName"] == pt) for pt in PASS_TYPES},
                "counter_attack": int(1901 in t),
                # 比賽狀態
                "minute": PERIOD_OFFSET[period] + sec / 60,
                "score_diff": score[tid] - score[opp],
                "home": int(sides.get(tid) == "home"),
                "team_prev_losses": losses_so_far[tid],
                # 前 10 秒的情境
                "own_events_10s": len(own), "opp_events_10s": len(oppc),
                "opp_duel_3s": int(any(c["eventName"] == "Duel" and sec - c["eventSec"] <= 3 for c in oppc)),
                "own_min_x_10s": min((c["positions"][0]["x"] for c in own if c["positions"]), default=p0["x"]),
                "role": roles.get(e["playerId"], "NA"),
                "end_source": end_source,       # 只用於檢查，不是特徵
                **after,                        # 只用於誤差分析，不是特徵
                "label": label,
            }
            rows.append(row)
            losses_so_far[tid] += 1
        # 比分在記錄後才更新
        if 101 in t and e["eventName"] in ("Shot", "Free Kick"):
            score[tid] += 1
        if 102 in t:
            score[other(tid)] += 1
    return rows


def main():
    with step("讀取比賽、球員、球隊資料"):
        db = load_all(ROOT / "data" / "raw")
    roles = {p: (v.get("role") or {}).get("code2") or "NA" for p, v in db.players.items()}
    comp_name = {c: v["name"] for c, v in db.competitions.items()}
    rows = []

    # 一次讀一個賽事的事件檔（每個 150~190 MB），讀完逐場找出傳球失誤
    n_comps = len(set(db.match_file.values()))
    bar = tqdm(iter_events(db, verbose=False), total=n_comps, desc="賽事", unit="個")
    t_load = time.perf_counter()
    for comp, events in bar:
        load_s = time.perf_counter() - t_load                       # 讀這個賽事的 JSON 花的時間
        bar.set_postfix_str(comp)
        t0 = time.perf_counter()
        by = defaultdict(list)
        for e in events:
            by[e["matchId"]].append(e)
        del events
        for mid, es in tqdm(by.items(), desc=f"  {comp}", unit="場", leave=False):
            rows.extend(match_rows(es, db.matches[mid], roles))
        proc_s = time.perf_counter() - t0
        record(f"{comp}：讀檔", load_s)
        record(f"{comp}：處理 {len(by)} 場", proc_s)
        tqdm.write(f"  {comp}：讀檔 {fmt(load_s)}，處理 {len(by)} 場 {fmt(proc_s)}，累計 {len(rows):,} 次失誤")
        t_load = time.perf_counter()

    with step("寫出 turnovers.csv"):
        df = pd.DataFrame(rows)
        df["competition"] = df.competitionId.map(comp_name)
        out = ROOT / "data" / "processed" / "turnovers.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    print(f"失誤 {len(df):,}，15 秒內失球 {df.label.sum():,}（{df.label.mean():.3%}）")
    print("落點來源：", dict(end_counts))
    chk = df[df.end_source != "recorded"]
    print("落點補值的傳球種類：", chk.filter(like="pt_").sum().sort_values(ascending=False).to_dict())
    print("落點補值，依賽事：", chk.groupby("competition").size().to_dict())
    print("落點補值的失誤，失球率：", round(chk.label.mean(), 5), "；補值後落點 x 中位數：", chk.end_x.median())
    summary("建立資料", ROOT / "results" / "timing_build.csv")


if __name__ == "__main__":
    main()
