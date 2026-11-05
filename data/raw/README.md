# data/raw

Wyscout 原始資料**沒有放進 repo**：全部約 900 MB，其中 5 個事件檔超過 GitHub 單一檔案 100 MB 的上限（見 `.gitignore`）。
處理好的資料 `data/processed/turnovers.csv` 已在 repo 中，只跑實驗（`src/turnover_experiment.py`）不需要這些原始檔。

## 下載

Pappalardo et al. (2019) 的 Wyscout 公開資料集，授權 CC BY 4.0：
https://figshare.com/collections/Soccer_match_event_dataset/4415000

把下面這些檔案放在這個資料夾（`events` / `matches` 可以放 zip，不用解壓縮；也可以放在 `events/`、`matches/` 子資料夾）：

```
data/raw/
├── events.zip          或 events/events_England.json …（7 個賽事）
├── matches.zip         或 matches/matches_England.json …
├── players.json
├── teams.json
├── competitions.json
├── coaches.json
└── referees.json
```

## 已知問題

官方的 `referees.json` 本身就被截斷（192.1 kB，斷在第 627 筆 Michael Oliver 的資料中間），讀取程式會略過壞掉的尾巴，只讀到 626 筆。
本研究沒有使用裁判資訊，不受影響。