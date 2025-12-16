# notify_bot.py
import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import lightgbm as lgb
import warnings

# 警告非表示
warnings.filterwarnings('ignore')

# --- 設定: LINE API ---
LINE_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.environ.get("LINE_USER_ID")

# --- 設定: 判定ロジック (Webアプリと同一) ---
SECTOR_SETTINGS = {
    "投資・グロース (注目)": {"threshold": 0.55, "label": "注目"},
    "銀行・金融 (鉄板)": {"threshold": 0.55, "label": "鉄板"},
    "商社・市況 (高勝率)": {"threshold": 0.55, "label": "高勝率"},
    "半導体・ハイテク": {"threshold": 0.55, "label": "高ボラ"},
    "自動車・機械": {"threshold": 0.58, "label": "標準"},
    "通信・医薬・生活": {"threshold": 0.60, "label": "内需"}
}

# 監視対象銘柄リスト
TARGET_TICKERS = {
    "投資・グロース (注目)": {
        "9984": "ソフトバンクG", "9983": "ファストリ",
        "7974": "任天堂", "6098": "リクルート", "6920": "レーザーテック"
    },
    "銀行・金融 (鉄板)": {
        "8306": "三菱UFJ", "8316": "三井住友FG",
        "8411": "みずほFG", "8766": "東京海上"
    },
    "商社・市況 (高勝率)": {
        "8031": "三井物産", "8058": "三菱商事", "8001": "伊藤忠",
        "5401": "日本製鉄", "9101": "日本郵船"
    },
    "自動車・機械": {
        "7203": "トヨタ", "7267": "ホンダ", "6902": "デンソー",
        "6501": "日立", "6367": "ダイキン", "6954": "ファナック"
    },
    "半導体・ハイテク": {
        "8035": "東エレク", "6857": "アドバン", "6758": "ソニーG",
        "6861": "キーエンス", "4063": "信越化", "6981": "村田製", "7741": "HOYA"
    },
    "通信・医薬・生活": {
        "9432": "NTT", "9433": "KDDI", "2914": "JT",
        "4502": "武田", "4568": "第一三共", "3382": "セブンi", "4452": "花王"
    }
}


# --- 関数群 ---
def get_macro_data():
    tickers = {"^N225": "Nikkei", "JPY=X": "USDJPY", "^GSPC": "SP500"}
    macro_df = pd.DataFrame()
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df.columns = df.columns.get_level_values(0)
                except IndexError:
                    pass
            df[f'{name}_Change'] = df['Close'].pct_change()
            sma5 = ta.trend.sma_indicator(df['Close'], window=5)
            df[f'{name}_SMA5_Ratio'] = (df['Close'] - sma5) / sma5
            if macro_df.empty:
                macro_df = df[[f'{name}_Change', f'{name}_SMA5_Ratio']]
            else:
                macro_df = macro_df.join(df[[f'{name}_Change', f'{name}_SMA5_Ratio']], how='outer')
        except:
            pass
    macro_df['SP500_Change'] = macro_df['SP500_Change'].shift(1)
    macro_df['SP500_SMA5_Ratio'] = macro_df['SP500_SMA5_Ratio'].shift(1)
    return macro_df.ffill()


def get_data_with_macro(ticker_code, macro_df):
    symbol = f"{ticker_code}.T"
    try:
        df = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
    except:
        return None

    if df.empty or len(df) < 200: return None
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except IndexError:
            pass

    try:
        df['SMA5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA25'] = ta.trend.sma_indicator(df['Close'], window=25)
        df['SMA75'] = ta.trend.sma_indicator(df['Close'], window=75)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Position'] = (df['Close'] - indicator_bb.bollinger_lband()) / (
                indicator_bb.bollinger_hband() - indicator_bb.bollinger_lband())
        df['Vol_SMA5'] = ta.trend.sma_indicator(df['Volume'], window=5)
    except:
        return None

    df['SMA5_Ratio'] = (df['Close'] - df['SMA5']) / df['SMA5']
    df['SMA25_Ratio'] = (df['Close'] - df['SMA25']) / df['SMA25']
    df['SMA75_Ratio'] = (df['Close'] - df['SMA75']) / df['SMA75']
    df['Vol_Ratio'] = (df['Volume'] - df['Vol_SMA5']) / df['Vol_SMA5']

    df = df.join(macro_df, how='left').dropna()
    return df


def add_binary_labels(df):
    df['Future_Close'] = df['Close'].shift(-5)
    df['Change_Rate'] = (df['Future_Close'] - df['Close']) / df['Close']
    df['Target'] = df['Change_Rate'].apply(lambda x: 1 if x >= 0.02 else 0)
    return df.dropna()


def train_and_predict():
    print("Market Data Download...")
    macro_df = get_macro_data()

    teacher_tickers = [
        "6758", "6861", "8035", "6501", "6902", "6981", "6954", "7741", "6920",
        "7203", "7267", "8306", "8316", "8411", "8766", "8031", "8058", "8001",
        "9984", "9432", "9433", "6098", "7974", "4502", "4568", "9983", "3382",
        "6367", "4063", "2914"
    ]

    train_dfs = []
    for code in teacher_tickers:
        df = get_data_with_macro(code, macro_df)
        if df is not None:
            df = add_binary_labels(df)
            train_dfs.append(df)

    full_train_df = pd.concat(train_dfs)

    feature_cols = [
        'SMA5_Ratio', 'SMA25_Ratio', 'SMA75_Ratio', 'BB_Position', 'Vol_Ratio', 'RSI',
        'Nikkei_Change', 'Nikkei_SMA5_Ratio',
        'USDJPY_Change', 'USDJPY_SMA5_Ratio',
        'SP500_Change', 'SP500_SMA5_Ratio'
    ]

    print("Training Model...")
    model = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss', n_estimators=100,
        learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
    )
    model.fit(full_train_df[feature_cols], full_train_df['Target'])

    print("Predicting Targets...")
    results = []

    for category, tickers in TARGET_TICKERS.items():
        settings = SECTOR_SETTINGS[category]
        threshold = settings["threshold"]

        for code, name in tickers.items():
            df = get_data_with_macro(code, macro_df)
            if df is None: continue

            latest_data = df.iloc[[-1]][feature_cols]
            current_price = df['Close'].iloc[-1]
            prob = model.predict_proba(latest_data)[0][1]

            if prob >= threshold:
                results.append({
                    "name": name,
                    "code": code,
                    "price": current_price,
                    "prob": prob,
                    "threshold": threshold,
                    "category": category
                })

    return results


def send_line_message(messages):
    # LINE API Endpoint
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }

    # メッセージ本文の作成（分岐ロジック）
    if not messages:
        # シグナルなしの場合
        text_content = (
            "【📊 本日のAI分析結果】\n\n"
            "現在、Sランク基準（勝率70%超期待）を満たす買いシグナルはありません。\n\n"
            "無理なエントリーは控え、次のチャンスを待ちましょう。☕\n"
            "(明日の16:00に再度分析します)"
        )
    else:
        # シグナルありの場合
        text_content = "【🎯 AI買いシグナル検知】\n以下の銘柄がチャンスです！\n"
        for item in messages:
            text_content += f"\n💎 {item['name']} ({item['code']})"
            text_content += f"\n   株価: {item['price']:,.0f}円"
            text_content += f"\n   AI確信度: {item['prob']:.1%} (閾値 {item['threshold']:.2f})"
            text_content += f"\n   セクター: {item['category']}\n"

        text_content += "\n⚠️ 投資は自己責任で行ってください。"

    data = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": text_content}]
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("LINE notification sent successfully.")
    else:
        print(f"Failed to send LINE: {response.status_code} {response.text}")


if __name__ == "__main__":
    if not LINE_ACCESS_TOKEN or not LINE_USER_ID:
        print("Error: LINE API Token or User ID is missing.")
    else:
        # 予測を実行
        signals = train_and_predict()
        # シグナルがあってもなくても通知関数を呼ぶ
        send_line_message(signals)