#!/usr/bin/env bash
# 從原始資料重跑（約 10 分鐘，2 核 CPU）
# Windows 的 Git Bash 也能跑：
#   - 自動找可以用的 Python（Windows 上 python3 常常是微軟商店的空殼，執行了不會有任何反應）
#   - python -u：不緩衝輸出，進度會即時印出來（Git Bash 的視窗不是真正的終端機，不加 -u 要等跑完才一次印出）
#   - PYTHONUTF8=1：用 UTF-8 印中文，避免 Windows 主控台的編碼錯誤
set -euo pipefail
cd "$(dirname "$0")"

# 優先使用 setup_env.sh 建立的虛擬環境 DM_HW2_G14；也可以自己指定：PY=/path/to/python bash run_all.sh
PY="${PY:-}"
VENV_PY=""
[ -x DM_HW2_G14/Scripts/python.exe ] && VENV_PY=DM_HW2_G14/Scripts/python.exe   # Windows
[ -x DM_HW2_G14/bin/python ] && VENV_PY=DM_HW2_G14/bin/python                   # Linux / macOS
for cand in ${PY:+"$PY"} ${VENV_PY:+"$VENV_PY"} python3 python py; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys; assert sys.version_info >= (3, 12)" >/dev/null 2>&1; then
        PY="$cand"
        FOUND=1
        break
    fi
done
if [ -z "${FOUND:-}" ]; then
    echo "找不到 Python 3.12 以上的版本（試過 python3 / python / py）" >&2
    exit 1
fi
echo "使用 $($PY --version)（$PY）"

export PYTHONUTF8=1
START=$SECONDS
"$PY" -u src/build_turnovers.py       # data/raw → data/processed/turnovers.csv（各步驟時間 → results/timing_build.csv）
"$PY" -u src/turnover_experiment.py   # 探索、三層方法、顯著性、誤差樣貌（各步驟時間 → results/timing_experiment.csv）
T=$((SECONDS - START))
echo
echo "全部完成，總共 $((T / 60)) 分 $((T % 60)) 秒"
