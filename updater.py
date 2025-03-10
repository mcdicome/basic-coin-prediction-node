import os
import pandas as pd
import requests
from datetime import date, timedelta
from zipfile import ZipFile
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

def create_lag_features(df, col_prefix, lags=10):
    """创建滞后特征"""
    for lag in range(1, lags + 1):
        df[f"{col_prefix}_open_lag{lag}"] = df["open"].shift(lag)
        df[f"{col_prefix}_high_lag{lag}"] = df["high"].shift(lag)
        df[f"{col_prefix}_low_lag{lag}"] = df["low"].shift(lag)
        df[f"{col_prefix}_close_lag{lag}"] = df["close"].shift(lag)
    return df

def download_binance_daily_data(pair, training_days, region, download_path):
    """下载 Binance 历史数据，并处理 81 维特征"""
    training_days = int(training_days)
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    end_date = date.today()
    start_date = end_date - timedelta(days=training_days)
    os.makedirs(download_path, exist_ok=True)

    def download_and_extract(single_date, pair):
        url = f"{base_url}/{pair}/1m/{pair}-1m-{single_date}.zip"
        file_path = os.path.join(download_path, f"{pair}-1m-{single_date}.zip")
        extracted_path = os.path.join(download_path, f"{pair}-1m-{single_date}.csv")

        if os.path.exists(extracted_path):
            return extracted_path

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"[ERROR] File not found: {url}")
            return None

        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(file_path)
        return extracted_path

    file_paths = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(download_and_extract, [start_date + timedelta(n) for n in range(training_days)], ["ETHUSDT"] * training_days)
        file_paths.extend(filter(None, results))

    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, header=None, names=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "n_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')
        df = df[(df["timestamp"] > 1000000000000) & (df["timestamp"] < 32503680000000)]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close"]].astype(float)
        dfs.append(df)

    if not dfs:
        print("No ETHUSDT data downloaded.")
        return None

    df_all = pd.concat(dfs).sort_index()
    df_all = create_lag_features(df_all, "ETHUSDT")
    df_all["hour_of_day"] = df_all.index.hour
    df_all["target_ETHUSDT"] = df_all["ETHUSDT_close_lag1"].shift(-1) - df_all["ETHUSDT_close_lag1"]

    selected_columns = [
        f"ETHUSDT_open_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_high_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_low_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_close_lag{i}" for i in range(1, 11)] + \
        ["hour_of_day", "target_ETHUSDT"]

    df_final = df_all[selected_columns].dropna()
    print(f"[DEBUG] Final DataFrame shape (should be 81 columns): {df_final.shape}")
    
    return df_final

def download_binance_current_day_data(pair, region):
    """获取最新 1m K 线数据，并生成 81 维特征"""
    limit = 1000
    url = f'https://api.binance.{region}/api/v3/klines?symbol={pair}&interval=1m&limit={limit}'

    response = requests.get(url)
    response.raise_for_status()

    columns = ["timestamp", "open", "high", "low", "close", "volume",
               "close_time", "quote_asset_volume", "n_trades",
               "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"]

    df = pd.DataFrame(response.json(), columns=columns)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[(df["timestamp"] > 1000000000000) & (df["timestamp"] < 32503680000000)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close"]].astype(float)

    df = create_lag_features(df, "ETHUSDT")
    df["hour_of_day"] = df.index.hour
    df["target_ETHUSDT"] = df["close"].shift(-1) - df["close"]

    selected_columns = [
        f"ETHUSDT_open_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_high_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_low_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_close_lag{i}" for i in range(1, 11)] + \
        ["hour_of_day", "target_ETHUSDT"]

    df_final = df[selected_columns].dropna()
    print(f"[DEBUG] Current day DataFrame shape (should be 81 columns): {df_final.shape}")

    return df_final

# 确保 `model.py` 能正确导入这些函数
__all__ = ["download_binance_daily_data", "download_binance_current_day_data", "create_lag_features"]
