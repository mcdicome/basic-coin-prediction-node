import os
from datetime import date, timedelta, datetime
import pathlib
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json

# 定义重试策略
retry_strategy = Retry(
    total=4,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)

# 创建 HTTP 适配器
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount('http://', adapter)
session.mount('https://', adapter)

files = []

def download_url(url, download_path, name=None):
    try:
        global files
        file_name = os.path.join(download_path, name) if name else os.path.join(download_path, os.path.basename(url))
        pathlib.Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(file_name):
            return
        response = session.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            files.append(file_name)
        elif response.status_code == 404:
            print(f"File does not exist: {url}")
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(str(e))

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def download_binance_daily_data(pair, training_days, region, download_path):
    """
    下载过去 `training_days` 天的 Binance `1m` K 线数据，并生成 81 维滞后特征。
    :param pair: 交易对，如 'ETHUSDT'
    :param training_days: 需要下载的历史天数
    :param region: Binance 服务器区域（如 'com'）
    :param download_path: 存储路径
    :return: 处理后的 Pandas DataFrame（包含 81 个特征）
    """
    training_days = int(training_days)  # 确保是整数
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    end_date = date.today()
    start_date = end_date - timedelta(days=int(training_days))
    os.makedirs(download_path, exist_ok=True)

    # 生成下载链接并下载数据
    def download_and_extract(single_date):
        url = f"{base_url}/{pair}/1m/{pair}-1m-{single_date}.zip"
        file_path = os.path.join(download_path, f"{pair}-1m-{single_date}.zip")
        extracted_path = os.path.join(download_path, f"{pair}-1m-{single_date}.csv")

        # 跳过已下载文件
        if os.path.exists(extracted_path):
            print(f"Already downloaded: {extracted_path}")
            return extracted_path

        # 下载 `.zip` 文件
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {file_path}")
        else:
            print(f"File not found: {url}")
            return None

        # 解压 `.zip`
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(file_path)  # 删除 `.zip` 文件
        return extracted_path

    # 使用多线程并发下载
    file_paths = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(download_and_extract, [start_date + timedelta(n) for n in range(training_days)])
        file_paths.extend(filter(None, results))

    # 读取所有 CSV 文件并合并数据
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None, names=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "n_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close"]].astype(float)
        dfs.append(df)

    if not dfs:
        print("No data downloaded.")
        return None

    # 合并所有天的数据
    df_all = pd.concat(dfs).sort_index()

    # **生成 81 维滞后特征**
    def create_lag_features(df, col_prefix, lags=10):
        for lag in range(1, lags + 1):
            df[f"{col_prefix}_open_lag{lag}"] = df["open"].shift(lag)
            df[f"{col_prefix}_high_lag{lag}"] = df["high"].shift(lag)
            df[f"{col_prefix}_low_lag{lag}"] = df["low"].shift(lag)
            df[f"{col_prefix}_close_lag{lag}"] = df["close"].shift(lag)
        return df

    df_all = create_lag_features(df_all, pair)

    # **添加 hour_of_day**
    df_all["hour_of_day"] = df_all.index.hour

    # **计算 target_ETHUSDT（未来 1 小时价格变化）**
    df_all["target_ETHUSDT"] = df_all["close"].shift(-1) - df_all["close"]

    # **选择最终 81 个特征**
    selected_columns = [
        f"{pair}_open_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_high_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_low_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_close_lag{i}" for i in range(1, 11)] + \
        ["hour_of_day", "target_ETHUSDT"]

    df_final = df_all[selected_columns].dropna()
    return df_final

def download_binance_current_day_data(pair, region):
    limit = 1000
    base_url = f'https://api.binance.{region}/api/v3/klines?symbol={pair}&interval=1m&limit={limit}'
    response = session.get(base_url)
    response.raise_for_status()
    columns = ['start_time','open','high','low','close','volume','end_time','volume_usd','n_trades','taker_volume','taker_volume_usd','ignore']
    df = pd.DataFrame(json.loads(response.text), columns=columns)
    df['date'] = pd.to_datetime(df['end_time'] + 1, unit='ms')
    df[["open", "high", "low", "close", "volume", "taker_volume"]] = df[["open", "high", "low", "close", "volume", "taker_volume"]].apply(pd.to_numeric)
    df.set_index("date", inplace=True)

    # **生成 81 个滞后特征**
    def create_lag_features(df, col_prefix, lags=10):
        for lag in range(1, lags + 1):
            df[f'{col_prefix}_open_lag{lag}'] = df['open'].shift(lag)
            df[f'{col_prefix}_high_lag{lag}'] = df['high'].shift(lag)
            df[f'{col_prefix}_low_lag{lag}'] = df['low'].shift(lag)
            df[f'{col_prefix}_close_lag{lag}'] = df['close'].shift(lag)
        return df

    df = create_lag_features(df, pair)

    # **添加 hour_of_day**
    df['hour_of_day'] = df.index.hour

    # **计算目标变量 target_ETHUSDT**
    df['target_ETHUSDT'] = df['close'].shift(-1) - df['close']

    # **选择最终的 81 个特征**
    selected_columns = [
        f"{pair}_open_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_high_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_low_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_close_lag{i}" for i in range(1, 11)] + \
        ["hour_of_day", "target_ETHUSDT"]

    return df[selected_columns].dropna()

def get_coingecko_coin_id(token):
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum',
    }
    return token_map.get(token.upper(), None)

def download_coingecko_data(token, training_days, download_path, CG_API_KEY):
    days = min([d for d in [7, 14, 30, 90, 180, 365, float('inf')] if d >= training_days])
    coin_id = get_coingecko_coin_id(token)
    if not coin_id:
        raise ValueError("Unsupported token")
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}&api_key={CG_API_KEY}'
    global files
    files = []
    with ThreadPoolExecutor() as executor:
        print(f"Downloading data for {coin_id}")
        name = f"{coin_id}_ohlc_{days}.json"
        executor.submit(download_url, url, download_path, name)
    return files

def download_coingecko_current_day_data(token, CG_API_KEY):
    coin_id = get_coingecko_coin_id(token)
    if not coin_id:
        raise ValueError("Unsupported token")
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1&api_key={CG_API_KEY}'
    response = session.get(url)
    response.raise_for_status()
    columns = ['timestamp','open','high','low','close']
    df = pd.DataFrame(json.loads(response.text), columns=columns)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index("date", inplace=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    return df.sort_index()

# **示例调用**
if __name__ == "__main__":
    df_eth = download_binance_current_day_data("ETHUSDT", "com")
    df_btc = download_binance_current_day_data("BTCUSDT", "com")

    # **合并 ETHUSDT 和 BTCUSDT 的数据**
    df = df_eth.merge(df_btc, on="hour_of_day", suffixes=("_ETH", "_BTC"))

    # **保存 CSV**
    df.to_csv("binance_lagged_features.csv", index=True)
    print("数据已下载并保存至 binance_lagged_features.csv")
