#!/usr/bin/env bash
# 建立虛擬環境 DM_HW2_G14 並安裝 requirements.txt（Windows 的 Git Bash、Linux、macOS 都能用）
#   bash setup_env.sh
# 之後直接 bash run_all.sh，它會自動使用這個虛擬環境。
set -euo pipefail
cd "$(dirname "$0")"
ENV=DM_HW2_G14

# 1. 找 64 位元的 Python 3.12～3.14：先試 Windows 的啟動器 py（優先 3.12，和產生結果時的版本一致；沒有再用 3.13），再試 python3、python
BASE=()
for cand in "py -3.12" "py -3.13" "python3" "python"; do
    read -r -a cmd <<< "$cand"
    if command -v "${cmd[0]}" >/dev/null 2>&1 \
       && "${cmd[@]}" -c "import sys, struct; assert (3, 12) <= sys.version_info[:2] <= (3, 14) and struct.calcsize('P') == 8" >/dev/null 2>&1; then
        BASE=("${cmd[@]}")
        break
    fi
done
if [ ${#BASE[@]} -eq 0 ]; then
    echo "找不到 64 位元的 Python 3.12～3.14（試過 py -3.12 / py -3.13 / python3 / python）。" >&2
    echo "請安裝 Python 3.12（py install 3.12，或到 python.org 下載並勾選「Add python.exe to PATH」），然後重開 Git Bash。" >&2
    exit 1
fi
echo "用 $("${BASE[@]}" --version) 建立虛擬環境 $ENV"

# 檢查一個 python 能不能用：3.12～3.14、64 位元（requirements.txt 的套件只有這些版本的預先編譯檔）
OK_CHECK="import sys, struct; assert (3, 12) <= sys.version_info[:2] <= (3, 14) and struct.calcsize('P') == 8"

# 2. 虛擬環境裡的 python：Windows 在 Scripts/，Linux / macOS 在 bin/
venv_python() {
    if [ -x "$ENV/Scripts/python.exe" ]; then echo "$ENV/Scripts/python.exe"; else echo "$ENV/bin/python"; fi
}

# 3. 已經有 DM_HW2_G14 但版本不對（例如之前用別的 Python 建的）→ 刪掉重建
if [ -d "$ENV" ] && ! "$(venv_python)" -c "$OK_CHECK" >/dev/null 2>&1; then
    echo "現有的 $ENV 不是 64 位元的 Python 3.12～3.14，刪掉重建"
    rm -rf "$ENV"
fi
if [ ! -d "$ENV" ]; then
    "${BASE[@]}" -m venv "$ENV"
fi
VPY="$(venv_python)"
echo "虛擬環境：$("$VPY" -c 'import sys, platform; print(sys.version.split()[0], platform.architecture()[0])')"

# 4. 安裝套件（用 python -m pip，不需要 pip 在 PATH 上）
#    --only-binary=:all:：只裝預先編譯好的檔案，找不到就直接報錯，不會在 Windows 上嘗試從原始碼編譯
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install --only-binary=:all: -r requirements.txt

echo
echo "完成。之後直接執行：bash run_all.sh（會自動使用 $ENV）"
echo "想在 Git Bash 裡手動使用這個環境：source $ENV/Scripts/activate   （Linux / macOS：source $ENV/bin/activate）"