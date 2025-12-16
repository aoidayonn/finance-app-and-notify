# 📈 AI Stock Prediction & Notification System

AI（LightGBM）を活用して日本株の「買いシグナル」を予測するWebアプリと、LINE通知ボットの統合システムです。
過去のバックテストで勝率70%〜100%を記録した「Sランク銘柄」のみを対象に、セクターごとの厳格な閾値を用いて判定を行います。

## 📂 ファイル構成

```text
.
├── app.py                # Webアプリ本体 (Streamlit)
├── notify_bot.py         # LINE通知ボット (GitHub Actionsで自動実行)
├── strict_eval.py        # 厳格検証用スクリプト (ロジックの正当性確認用)
├── requirements.txt      # 依存ライブラリ一覧
└── .github
    └── workflows
        └── run_bot.yml   # 自動実行のスケジュール設定

🚀 機能概要Webアプリケーション (app.py)

ブラウザ上で銘柄ごとのAI判定結果、チャート、確信度を確認可能。セクターごとの特性に合わせた「動的閾値」を採用。PC/スマホ対応。

LINE通知ボット (notify_bot.py)平日の市場終了後（16:00頃）に自動分析を実行。「買い推奨」が出た銘柄がある場合のみ、スマホに即時通知。完全無料・サーバーレス（GitHub Actions活用）。検証モード (strict_eval.py)未来のデータを参照しない「厳格なバックテスト」を実行可能。モデルの実力を客観的に測定するために使用。🛠 インストール & 実行方法 (ローカル環境)

1. 環境構築Python 3.9以上推奨。Bash# 必要なライブラリをインストール
pip install -r requirements.txt
2. Webアプリの起動以下のコマンドでブラウザが立ち上がります。Bashstreamlit run app.py
3. 検証スクリプトの実行バックテストを行い、勝率を確認したい場合に実行します。Bashpython strict_eval.py

🤖 LINE通知ボットのセットアップ (GitHub Actions)サーバーを用意せず、GitHub上で自動運用するための手順です。

1. LINE Developers設定LINE Developers でMessaging APIチャネルを作成。チャネルアクセストークン（長期） を発行してコピー。あなたのユーザーID (User ID) を確認してコピー。
2. GitHubリポジトリ設定リポジトリの Settings > Secrets and variables > Actions に以下を登録します。NameValueLINE_CHANNEL_ACCESS_TOKEN(コピーしたアクセストークン)LINE_USER_ID(コピーしたユーザーID)
3. スケジュール設定.github/workflows/run_bot.yml の cron を変更することで実行時間を調整可能です。※ GitHub ActionsはUTC（世界標準時）のため、日本時間から -9時間 した値を設定してください。
例：日本時間 16:00 に実行する場合YAMLon:
  schedule:
    - cron: '0 7 * * 1-5'  # UTC 07:00 (平日のみ)


⚠️ 免責事項本システムは過去のデータに基づき予測を行いますが、将来の利益を保証するものではありません。投資判断はご自身の責任において行ってください。