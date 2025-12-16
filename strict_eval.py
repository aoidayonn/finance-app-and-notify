import pandas as pd
import numpy as np
import yfinance as yf
import ta
import lightgbm as lgb
from sklearn.metrics import precision_score
import warnings

# 不要な警告を非表示
warnings.filterwarnings('ignore')


# --- 1. データ取得・加工（共通ロジック） ---
def get_macro_data(start_date="2000-01-01"):
    tickers = {"^N225": "Nikkei", "JPY=X": "USDJPY", "^GSPC": "SP500"}
    macro_df = pd.DataFrame()
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
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
        df = yf.download(symbol, period="max", auto_adjust=True, progress=False)
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


# --- メイン処理 ---
if __name__ == "__main__":
    print("=== 超・厳格検証モード: 未来情報の完全遮断版 ===")

    macro_df = get_macro_data()

    # 検証ターゲット
    test_targets = {
        "【投資・グロース (注目)】": ["9984", "9983", "7974", "6098", "6920"],
        "【銀行・金融 (鉄板)】": ["8306", "8316", "8411", "8766"],
        "【商社・市況 (高勝率)】": ["8031", "8058", "8001", "5401", "9101"],
        "【自動車・機械】": ["7203", "7267", "6902", "6501", "6367", "6954"],
        "【半導体・ハイテク】": ["8035", "6857", "6758", "6861", "4063", "6981", "7741"],
        "【通信・医薬・生活】": ["9432", "9433", "2914", "4502", "4568", "3382", "4452"]
    }

    # 先生役
    teacher_tickers = []
    for codes in test_targets.values():
        teacher_tickers.extend(codes)

    print(f"先生役の銘柄数: {len(teacher_tickers)} (Webアプリ全銘柄)")

    # データロード
    print("\n[準備] 全銘柄のデータをダウンロード中...")
    all_codes = set(teacher_tickers)
    for cat_codes in test_targets.values():
        all_codes.update(cat_codes)

    data_cache = {}
    for code in all_codes:
        print(f"Reading: {code}...", end="\r")
        df = get_data_with_macro(code, macro_df)
        if df is not None:
            df = add_binary_labels(df)
            data_cache[code] = df
    print(f"\nデータロード完了: {len(data_cache)} 銘柄")

    # --- ここで期間を厳密に定義 ---
    TRAIN_END_DATE = "2020-12-31"  # 学習はここまでのデータしか見てはいけない
    TEST_START_DATE = "2021-01-01"  # 検証はここからのデータで行う

    THRESHOLD = 0.55
    feature_cols = [
        'SMA5_Ratio', 'SMA25_Ratio', 'SMA75_Ratio', 'BB_Position', 'Vol_Ratio', 'RSI',
        'Nikkei_Change', 'Nikkei_SMA5_Ratio',
        'USDJPY_Change', 'USDJPY_SMA5_Ratio',
        'SP500_Change', 'SP500_SMA5_Ratio'
    ]

    print(f"\n[検証フェーズ] 閾値 {THRESHOLD:.0%} (期間完全分離: {TEST_START_DATE}以降を予測)")

    for category, codes in test_targets.items():
        print(f"\n{category}")
        print(f"{'Code':<6} | {'取引回数':<8} | {'勝率':<10} | {'判定'}")
        print("-" * 50)

        cat_precisions = []

        for target_code in codes:
            if target_code not in data_cache: continue

            # --- 修正ポイント: 学習データを「日付」でフィルタリング ---
            current_train_dfs = []
            for teacher in teacher_tickers:
                if teacher == target_code:
                    continue

                if teacher in data_cache:
                    df_teacher = data_cache[teacher]
                    # ★重要: 先生データも 2019年末まで に制限する
                    # これにより、2020年以降の相場環境をカンニングできなくなる
                    df_train_period = df_teacher[df_teacher.index <= TRAIN_END_DATE]
                    current_train_dfs.append(df_train_period)

            if not current_train_dfs: continue

            full_train_df = pd.concat(current_train_dfs)

            # データ不足チェック
            if len(full_train_df) < 100: continue

            model = lgb.LGBMClassifier(
                objective='binary', metric='binary_logloss', n_estimators=100,
                learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
            )
            model.fit(full_train_df[feature_cols], full_train_df['Target'])

            # テスト実行
            test_df = data_cache[target_code]
            test_df = test_df[test_df.index >= TEST_START_DATE]

            if len(test_df) == 0: continue

            probs = model.predict_proba(test_df[feature_cols])[:, 1]
            y_pred = (probs >= THRESHOLD).astype(int)
            trade_count = np.sum(y_pred)

            if trade_count > 0:
                precision = precision_score(test_df['Target'], y_pred, zero_division=0)
                cat_precisions.append(precision)

                judge = "★★" if precision >= 0.6 else ("★" if precision >= 0.55 else "×")
                print(f"{target_code:<6} | {trade_count:>4}回     | {precision:.1%}      | {judge}")
            else:
                print(f"{target_code:<6} |    0回     | -         | スルー")

        if cat_precisions:
            avg = np.mean(cat_precisions)
            print(f"  >>> {category} 平均勝率: {avg:.1%}")