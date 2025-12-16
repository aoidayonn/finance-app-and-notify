# 📈 AI Stock Prediction & Notification System

AI（**LightGBM**）を活用して、日本株の**「買いシグナル」**を予測するWebアプリと、**LINE通知ボット**を統合したシステムです。

* 過去のバックテストで **勝率70%〜100%** を記録した **Sランク銘柄のみ** を対象
* **セクターごとの厳格な動的閾値** を用いて判定
* Webでの可視化 + LINEでの自動通知を実現

---

## 📂 ファイル構成

```text
.
├── app.py                # Webアプリ本体 (Streamlit)
├── notify_bot.py         # LINE通知ボット (GitHub Actionsで自動実行)
├── strict_eval.py        # 厳格検証用スクリプト（ロジック検証用）
├── requirements.txt      # 依存ライブラリ一覧
└── .github
    └── workflows
        └── run_bot.yml   # 自動実行のスケジュール設定
```

---

## 🚀 機能概要

### 🖥 Webアプリケーション（`app.py`）

* ブラウザ上で以下を確認可能

  * 銘柄ごとの **AI判定結果**
  * **株価チャート**
  * **予測の確信度**
* セクター特性に応じた **動的閾値** を採用
* **PC / スマホ両対応**

---

### 📩 LINE通知ボット（`notify_bot.py`）

* **平日の市場終了後（16:00頃）** に自動分析
* 「**買い推奨**」が出た銘柄のみをLINEで通知
* **完全無料・サーバーレス**

  * GitHub Actions を活用

---

### 🔍 検証モード（`strict_eval.py`）

* 未来のデータを参照しない **厳格なバックテスト** を実行
* モデル性能を **客観的に評価** するための検証用スクリプト

---

## 🛠 インストール & 実行方法（ローカル環境）

### 1️⃣ 環境構築

* **Python 3.9以上 推奨**

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Webアプリの起動

以下のコマンドでブラウザが立ち上がります。

```bash
streamlit run app.py
```

---

### 3️⃣ 検証スクリプトの実行

バックテストを行い、勝率を確認したい場合に実行します。

```bash
python strict_eval.py
```

---

## 🤖 LINE通知ボットのセットアップ（GitHub Actions）

サーバーを用意せず、GitHub上で自動運用するための手順です。

---

### 1️⃣ LINE Developers 設定

1. **LINE Developers** で Messaging API チャネルを作成
2. 以下を取得して控えておきます

   * チャネルアクセストークン（**長期**）
   * あなたの **User ID**

---

### 2️⃣ GitHub リポジトリ設定

リポジトリの以下にシークレットを登録します。

**Settings → Secrets and variables → Actions**

| Name                        | Value        |
| --------------------------- | ------------ |
| `LINE_CHANNEL_ACCESS_TOKEN` | チャネルアクセストークン |
| `LINE_USER_ID`              | あなたのUser ID  |

---

### 3️⃣ スケジュール設定

`.github/workflows/run_bot.yml` の `cron` を変更することで、実行時間を調整できます。

⚠️ **GitHub Actions は UTC（世界標準時）** で動作します。
日本時間に合わせる場合は **-9時間** してください。

**例：日本時間 16:00 に実行する場合**

```yaml
on:
  schedule:
    - cron: '0 7 * * 1-5'  # UTC 07:00（平日のみ）
```

---

## ⚠️ 免責事項

* 本システムは **過去データに基づく予測** を行うものであり、
  **将来の利益を保証するものではありません**。
* 投資判断は **ご自身の責任** において行ってください。
