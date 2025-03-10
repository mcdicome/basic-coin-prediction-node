import os
import pandas as pd
import requests
from datetime import date, timedelta
from zipfile import ZipFile
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

# 下载 Binance 历史数据，并处理 81 个特征
def download_binance_daily_data(pair, training_days, region, download_path):
    training_days = int(training_days)  # 确保是整数
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    end_date = date.today()
    start_date = end_date - timedelta(days=training_days)
    os.makedirs(download_path, exist_ok=True)

    # 下载并解压数据
    def download_and_extract(single_date):
        url = f"{base_url}/{pair}/1m/{pair}-1m-{single_date}.zip"
        file_path = os.path.join(download_path, f"{pair}-1m-{single_date}.zip")
        extracted_path = os.path.join(download_path, f"{pair}-1m-{single_date}.csv")

        if os.path.exists(extracted_path):
            print(f"[SKIP] Already downloaded: {extracted_path}")
            return extracted_path

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"[SUCCESS] Downloaded: {file_path}")
        else:
            print(f"[ERROR] File not found: {url}")
            return None

        # 解压
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(file_path)  # 删除 .zip 文件
        return extracted_path

    # 并行下载数据
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
        
        # **确保 timestamp 正确**
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')

        # 过滤无效时间戳（1970 - 3000年）
        df = df[(df["timestamp"] > 1000000000000) & (df["timestamp"] < 32503680000000)]

        # 转换为 datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

        # 去除无效值
        df = df.dropna(subset=["timestamp"])
        
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close"]].astype(float)
        dfs.append(df)

    if not dfs:
        print("No data downloaded.")
        return None

    # 合并所有数据
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
    print(f"[DEBUG] Final DataFrame shape (should be 81 columns): {df_final.shape}")
    
    return df_final

# 下载 Binance 当日数据并生成 81 维特征
def download_binance_current_day_data(pair, region):
    limit = 1000
    url = f'https://api.binance.{region}/api/v3/klines?symbol={pair}&interval=1m&limit={limit}'

    response = requests.get(url)
    response.raise_for_status()

    columns = ["timestamp", "open", "high", "low", "close", "volume",
               "close_time", "quote_asset_volume", "n_trades",
               "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"]

    df = pd.DataFrame(response.json(), columns=columns)

    # **确保 timestamp 正确**
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')

    # 过滤无效数据
    df = df[(df["timestamp"] > 1000000000000) & (df["timestamp"] < 32503680000000)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close"]].astype(float)

    # **生成 81 维滞后特征**
    df = create_lag_features(df, pair)

    # **添加 hour_of_day**
    df["hour_of_day"] = df.index.hour

    # **计算 target_ETHUSDT**
    df["target_ETHUSDT"] = df["close"].shift(-1) - df["close"]

    # **选择最终 81 个特征**
    selected_columns = [
        f"{pair}_open_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_high_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_low_lag{i}" for i in range(1, 11)] + \
        [f"{pair}_close_lag{i}" for i in range(1, 11)] + \
        ["hour_of_day", "target_ETHUSDT"]

    df_final = df[selected_columns].dropna()
    print(f"[DEBUG] Current day DataFrame shape (should be 81 columns): {df_final.shape}")

    return df_final
