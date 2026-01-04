# 🏥 A病院 待ち人数・待ち時間 統合予測（Streamlit Cloud）

## 変更点（重要）
- `jpholiday` 依存を廃止し、`data/syukujitsu.csv`（祝日CSV）で祝日判定します。
- Streamlit Cloudで `ModuleNotFoundError: jpholiday` が出ない構成です。

## リポジトリ構成
```
.
├── app.py
├── requirements.txt
├── packages.txt
├── data/
│   └── syukujitsu.csv
├── .streamlit/
│   └── config.toml
└── models/
    ├── README.md
    ├── model_A_timeseries.json
    ├── columns_A_timeseries.json
    ├── model_A_waittime_30min_FULL.json
    ├── model_A_queue_30min_FULL.json
    └── columns_A_multi_30min_FULL.json
```

## Streamlit Cloud へデプロイ
1. GitHubへPush（Private推奨）
2. Streamlit Cloud → New app → main file `app.py` を指定してDeploy
3. 変更が反映されない場合：Manage app → **Clear cache** → **Reboot**

## 必須ファイル
- models/ に5ファイル
- data/syukujitsu.csv（同梱済み）
