import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

def fetch_binance_ohlc(symbol, interval, start_time, end_time):
    """
    从 Binance API 获取 K 线数据
    :param symbol: 交易对 (ETHUSDT 或 BTCUSDT)
    :param interval: K 线时间间隔 (如 '1h')
    :param start_time: 数据开始时间 (datetime 对象)
    :param end_time: 数据结束时间 (datetime 对象)
    :return: K 线数据的列表
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
        "limit": 1000  # Binance API 限制最多 1000 条数据
    }

    all_data = []
    while True:
        response = requests.get(BINANCE_API_URL, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        params["startTime"] = data[-1][0] + 1  # 继续从最后一个时间点获取数据
        if len(data) < 1000:
            break  # 如果返回的数据小于 1000 条，说明已获取全部数据
    return all_data

def process_ohlc_data(data):
    """
    处理 Binance K 线数据，将其转换为 DataFrame
    :param data: Binance API 返回的 K 线数据
    :return: 格式化后的 DataFrame
    """
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)
    return df

def create_lag_features(df, prefix, lags=10):
    """
    生成滞后特征
    :param df: 原始 DataFrame
    :param prefix: 列名前缀 (ETHUSDT 或 BTCUSDT)
    :param lags: 需要创建的滞后步数
    :return: 带有滞后特征的 DataFrame
    """
    for lag in range(1, lags + 1):
        df[f'{prefix}_open_lag{lag}'] = df['open'].shift(lag)
        df[f'{prefix}_high_lag{lag}'] = df['high'].shift(lag)
        df[f'{prefix}_low_lag{lag}'] = df['low'].shift(lag)
        df[f'{prefix}_close_lag{lag}'] = df['close'].shift(lag)
    return df

def download_binance_current_day_data():
    """
    下载 ETHUSDT 和 BTCUSDT 的 1 小时 K 线数据，并生成 81 个特征
    :return: 处理后的 DataFrame
    """
    # 设定开始和结束时间（过去 24 小时的数据）
    end_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=1)  # 过去 24 小时数据
    symbols = ['ETHUSDT', 'BTCUSDT']
    interval = '1h'
    lags = 10
    all_features = pd.DataFrame()

    for symbol in symbols:
        print(f"Fetching {symbol} data...")
        raw_data = fetch_binance_ohlc(symbol, interval, start_time, end_time)
        df = process_ohlc_data(raw_data)
        df = create_lag_features(df, symbol)

        if all_features.empty:
            all_features = df
        else:
            all_features = all_features.join(df, how='outer')

    # 添加 hour_of_day 特征
    all_features['hour_of_day'] = all_features.index.hour

    # 计算目标变量 target_ETHUSDT（未来 1 小时 ETHUSDT 的收盘价变化率）
    all_features['target_ETHUSDT'] = all_features['ETHUSDT_close'].shift(-1) - all_features['ETHUSDT_close']

    # 移除 NaN 值（因滞后特征导致的缺失数据）
    all_features.dropna(inplace=True)

    # 选择最终的 81 个特征
    selected_columns = [
        f"open_ETHUSDT_lag{i}" for i in range(1, 11)] + \
        [f"high_ETHUSDT_lag{i}" for i in range(1, 11)] + \
        [f"low_ETHUSDT_lag{i}" for i in range(1, 11)] + \
        [f"close_ETHUSDT_lag{i}" for i in range(1, 11)] + \
        [f"open_BTCUSDT_lag{i}" for i in range(1, 11)] + \
        [f"high_BTCUSDT_lag{i}" for i in range(1, 11)] + \
        [f"low_BTCUSDT_lag{i}" for i in range(1, 11)] + \
        [f"close_BTCUSDT_lag{i}" for i in range(1, 11)] + \
        ["hour_of_day", "target_ETHUSDT"]

    return all_features[selected_columns]

if __name__ == "__main__":
    # 运行数据下载
    df = download_binance_current_day_data()

    # 保存到 CSV 文件
    df.to_csv("binance_data.csv", index=True)

    print("数据已下载并保存至 binance_data.csv")
