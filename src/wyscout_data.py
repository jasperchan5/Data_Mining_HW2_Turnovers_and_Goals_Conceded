"""
Wyscout 公開資料集（Pappalardo et al., 2019）的整合讀取

把 players / teams / competitions / coaches / referees / matches / events
用同一套清理規則讀進來，並依 wyId 互相連起來。

用法（notebook）：
    from wyscout_data import load_all, iter_events, foul_table

    db = load_all("data/")                    # 資料夾內放 json，或 matches.zip、events.zip
    db.players[32777]["shortName"]
    db.matches[2058017]["label"]
    db.teams[4418]["name"]

    # 犯規 → 是否吃牌 的資料表（一次只讀一個聯賽的 events，避免記憶體爆掉）
    df = foul_table(db, competitions=["World_Cup"])

檔案間的關聯
    events.playerId          -> players.wyId
    events.teamId            -> teams.wyId
    events.matchId           -> matches.wyId
    matches.competitionId    -> competitions.wyId
    matches.referees[].refereeId -> referees.wyId
    matches.teamsData[teamId].coachId -> coaches.wyId
    players.currentTeamId / currentNationalTeamId -> teams.wyId（是「資料蒐集時」的球隊，不一定是比賽當時）
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# 共用清理：各檔案的問題都一樣，所以寫一次
#   - id 有的是 int、有的是字串、有的是 "null"
#   - 名字裡有字面上的 é（多跳脫一次），要還原
#   - 比賽陣容裡的 goals / yellowCards 等是字串 "0"、"null"
# ---------------------------------------------------------------------------

def to_int(v):
    if v is None or v == "null" or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def fix_text(s):
    if isinstance(s, str) and "\\u" in s:
        try:
            return json.loads(f'"{s}"')
        except json.JSONDecodeError:
            return s
    return s


def _clean(obj, key=None):
    """遞迴清理：key 以 Id / id 結尾的轉 int，字串還原跳脫字元"""
    if isinstance(obj, dict):
        return {k: _clean(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if key and (key == "id" or key.endswith("Id") or key == "winner"):
        return to_int(obj)
    if key in ("goals", "ownGoals", "assists", "redCards", "yellowCards"):
        return to_int(obj) or 0
    return fix_text(obj)


# ---------------------------------------------------------------------------
# 讀檔：資料夾裡的 .json，或 zip 裡的 .json 都可以
# ---------------------------------------------------------------------------

def _find(root: Path, stem: str) -> list[tuple[str, callable]]:
    """
    回傳 (名稱, 開檔函式)；含子資料夾（例如 data/events/events_England.json）
    先找解壓後的 json，找不到才讀 zip；同名檔只取第一個（events.zip、events_1.zip 重複也沒關係）
    """
    out, seen = [], set()
    for p in sorted(root.rglob(f"{stem}*.json")):
        if p.stem not in seen:
            seen.add(p.stem)
            out.append((p.stem, lambda p=p: open(p, encoding="utf-8")))
    if not out:
        for z in sorted(root.rglob(f"{stem}*.zip")):
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist()
                         if n.endswith(".json") and not Path(n).name.startswith("._")]
            for n in names:
                if Path(n).stem not in seen:
                    seen.add(Path(n).stem)
                    out.append((Path(n).stem,
                                lambda z=z, n=n: zipfile.ZipFile(z).open(n)))
    return out


def _norm(s: str) -> str:
    """World_Cup / world cup / World-Cup 視為同一個"""
    return s.lower().replace(" ", "_").replace("-", "_")


def _load_json(opener, name=""):
    """一般情況直接 json.load；檔案被截斷時，救回最後一筆完整紀錄之前的內容"""
    with opener() as f:
        text = f.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    dec, rows, i = json.JSONDecoder(), [], text.index("[") + 1
    while True:
        while i < len(text) and text[i] in " ,\n\r\t":
            i += 1
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        rows.append(obj)
    print(f"[警告] {name} 不完整（可能被截斷），只讀到 {len(rows)} 筆")
    return rows


def _by_id(rows):
    return {r["wyId"]: r for r in rows}


@dataclass
class WyscoutDB:
    root: Path
    players: dict = field(default_factory=dict)
    teams: dict = field(default_factory=dict)
    competitions: dict = field(default_factory=dict)
    coaches: dict = field(default_factory=dict)
    referees: dict = field(default_factory=dict)
    matches: dict = field(default_factory=dict)
    match_file: dict = field(default_factory=dict)   # matchId -> 檔名後綴（England、World_Cup…）

    # ---- 常用查詢 ----
    def name(self, table: str, wy_id, key="shortName"):
        row = getattr(self, table).get(wy_id)
        if row is None:
            return None
        return row.get(key) or row.get("name")

    def match_referee(self, match_id, role="referee"):
        m = self.matches.get(match_id)
        if not m:
            return None
        return next((r["refereeId"] for r in m["referees"] if r["role"] == role), None)

    def match_team(self, match_id, team_id):
        """該場比賽某隊的 teamsData（side、coachId、score、formation…）"""
        m = self.matches.get(match_id)
        return m["teamsData"].get(team_id) if m else None

    def lineup_role(self, match_id, team_id, player_id):
        """球員在這場是先發(lineup)、替補(bench) 還是沒在名單"""
        td = self.match_team(match_id, team_id)
        if not td or not td.get("formation"):
            return None
        for part in ("lineup", "bench"):
            if any(p["playerId"] == player_id for p in td["formation"].get(part) or []):
                return part
        return None


def load_all(root: str | Path) -> WyscoutDB:
    root = Path(root).expanduser()
    db = WyscoutDB(root=root)

    for table in ("players", "teams", "competitions", "coaches", "referees"):
        files = _find(root, table)
        if not files:
            print(f"[略過] 找不到 {table}")
            continue
        rows = _clean(_load_json(files[0][1], table))
        setattr(db, table, _by_id(rows))

    for name, opener in _find(root, "matches"):
        comp = name.removeprefix("matches_")
        for m in _clean(_load_json(opener, name)):
            # teamsData 的 key 原本是字串 "4418"，改成 int 方便和 events.teamId 對
            m["teamsData"] = {int(k): v for k, v in m["teamsData"].items()}
            db.matches[m["wyId"]] = m
            db.match_file[m["wyId"]] = comp

    if not db.players and not db.matches:
        raise FileNotFoundError(f"{root.resolve()} 底下找不到任何 Wyscout 檔案，請確認路徑"
                                f"（notebook 的工作目錄是 {Path.cwd()}）")
    print(f"資料夾：{root.resolve()}")
    print(f"players {len(db.players)}, teams {len(db.teams)}, competitions {len(db.competitions)}, "
          f"coaches {len(db.coaches)}, referees {len(db.referees)}, matches {len(db.matches)}")
    return db


def iter_events(db: WyscoutDB, competitions: list[str] | None = None, verbose: bool = True) -> Iterator[tuple[str, list]]:
    """
    一次讀一個 events_*.json（每個 150~190 MB，全讀會吃掉十幾 GB 記憶體）
    competitions：例如 ["World_Cup", "England"]；None 表示全部
    """
    files = _find(db.root, "events")
    if not files:
        raise FileNotFoundError(
            f"{db.root.resolve()} 底下（含子資料夾）找不到 events_*.json 或 events*.zip")
    available = {_norm(n.removeprefix("events_")): n for n, _ in files}
    if competitions:
        missing = [c for c in competitions if _norm(c) not in available]
        if missing:
            raise ValueError(f"找不到 {missing}，可用的有："
                             f"{[n.removeprefix('events_') for n, _ in files]}")
    wanted = {_norm(c) for c in competitions} if competitions else None

    for name, opener in files:
        comp = name.removeprefix("events_")
        if wanted and _norm(comp) not in wanted:
            continue
        if verbose:
            print(f"讀取 {name} …")
        yield comp, _load_json(opener, name)


# ---------------------------------------------------------------------------
# 作業用：犯規 → 吃牌
# ---------------------------------------------------------------------------

TAG_RED, TAG_YELLOW, TAG_SECOND_YELLOW = 1701, 1702, 1703
FOUL_SUBEVENTS = {"Foul", "Hand foul"}          # 先前討論決定只看這兩種


def foul_table(db: WyscoutDB, competitions: list[str] | None = None):
    """每一筆犯規一列，附上球員、球隊、比賽、裁判的資訊與是否吃牌"""
    import pandas as pd

    rows = []
    for comp, events in iter_events(db, competitions):
        for e in events:
            if e["eventName"] != "Foul" or e["subEventName"] not in FOUL_SUBEVENTS:
                continue
            tags = {t["id"] for t in e["tags"]}
            mid, tid, pid = e["matchId"], e["teamId"], e["playerId"]
            m = db.matches.get(mid, {})
            td = db.match_team(mid, tid) or {}
            player = db.players.get(pid, {})
            pos = e["positions"][0] if e["positions"] else {"x": None, "y": None}
            rows.append({
                "competition": comp,
                "matchId": mid,
                "date": m.get("dateutc"),
                "matchPeriod": e["matchPeriod"],
                "eventSec": e["eventSec"],
                "subEventName": e["subEventName"],
                "x": pos["x"], "y": pos["y"],
                "playerId": pid,
                "player": player.get("shortName"),
                "role": (player.get("role") or {}).get("code2"),
                "teamId": tid,
                "team": db.name("teams", tid, "name"),
                "side": td.get("side"),                       # home / away
                "refereeId": db.match_referee(mid),
                "yellow": TAG_YELLOW in tags,
                "second_yellow": TAG_SECOND_YELLOW in tags,
                "red": TAG_RED in tags,
            })
        del events                                            # 讀下一個聯賽前釋放記憶體

    df = pd.DataFrame(rows)
    if len(df):
        df["card"] = df["yellow"] | df["second_yellow"] | df["red"]
    return df
