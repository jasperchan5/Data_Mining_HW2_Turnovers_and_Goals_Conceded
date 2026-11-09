"""
下載外部測試資料：StatsBomb Open Data（hudl/open-data）Premier League 2015/16 的比賽與事件檔（380 場）
    python src/download_statsbomb.py
放在 data/external/statsbomb/（約 1 GB，已在 .gitignore 中排除）

資料授權：StatsBomb Open Data 可免費用於研究；發表時須註明資料來源為 StatsBomb
（https://github.com/hudl/open-data）
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import urllib.request
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "statsbomb"
BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data"
# (competition_id, season_id)：Premier League 2015/16
COMPS = [(2, 27)]


def fetch(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 1000:
        return True
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def main():
    (OUT / "matches").mkdir(parents=True, exist_ok=True)
    (OUT / "events").mkdir(parents=True, exist_ok=True)
    ids = []
    for c, s in COMPS:
        p = OUT / "matches" / f"{c}_{s}.json"
        if not fetch(f"{BASE}/matches/{c}/{s}.json", p):
            raise SystemExit(f"下載失敗：matches/{c}/{s}.json")
        ids += [m["match_id"] for m in json.loads(p.read_text(encoding="utf-8"))]
    with cf.ThreadPoolExecutor(8) as ex:
        ok = list(tqdm(ex.map(lambda i: fetch(f"{BASE}/events/{i}.json", OUT / "events" / f"{i}.json"), ids),
                       total=len(ids), desc="事件檔", unit="場"))
    bad = [i for i, o in zip(ids, ok) if not o]
    print(f"{len(ids)} 場，失敗 {len(bad)} 場" + (f"：{bad[:10]}…（重新執行即可續傳）" if bad else ""))


if __name__ == "__main__":
    main()
