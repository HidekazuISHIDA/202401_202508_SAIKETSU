## models/ に置くファイル（必須）

必須（合計5つ）:
- model_A_timeseries.json
- columns_A_timeseries.json
- model_A_waittime_30min_FULL.json
- model_A_queue_30min_FULL.json
- columns_A_multi_30min_FULL.json

※ Streamlit Cloudで動かすには、models/ の中身が実行環境から参照できる必要があります。
大きい場合は Privateリポジトリ + Git LFS を推奨。
