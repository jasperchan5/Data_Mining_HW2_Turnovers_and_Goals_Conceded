"""
計時小工具
    with step("名稱"):   … 開始、結束各印一行，並記下花了多久
    record(名稱, 秒數)   手動記一筆（例如迴圈裡每個賽事各自的時間）
    summary(標題, 檔案)  最後印一張總表，也寫進 CSV（results/ 底下）
"""

from __future__ import annotations

import csv
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

_records: list[tuple[str, float]] = []
_start = time.perf_counter()


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    return f"{int(seconds // 60)} 分 {seconds % 60:04.1f} 秒"


def _width(text: str) -> int:
    """終端機上的顯示寬度：中文等全形字佔 2 格（表格才對得齊）"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _width(text))


def record(name: str, seconds: float) -> None:
    _records.append((name, seconds))


@contextmanager
def step(name: str):
    print(f"\n[{name}] 開始", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        record(name, dt)
        print(f"[{name}] 完成，{fmt(dt)}", flush=True)


def summary(title: str, csv_path: Path | None = None) -> None:
    total = time.perf_counter() - _start
    width = max([_width(n) for n, _ in _records] + [4])
    print(f"\n===== {title}：各步驟花費時間 =====")
    for name, s in _records:
        print(f"  {_pad(name, width)}  {fmt(s)}")
    print(f"  {_pad('總計', width)}  {fmt(total)}")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "seconds"])
            w.writerows([(n, round(s, 2)) for n, s in _records] + [("total", round(total, 2))])


_last = _start


def lap(name: str) -> None:
    """記下「從上一次 lap（或程式開始）到現在」花的時間，適合不想把整段程式縮排進 with 的地方。"""
    global _last
    now = time.perf_counter()
    record(name, now - _last)
    print(f"[{name}] 完成，{fmt(now - _last)}", flush=True)
    _last = now
