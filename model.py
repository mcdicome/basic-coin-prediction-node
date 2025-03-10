import os
import pickle
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from updater import download_binance_daily_data, download_binance_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL

binance_data_path = os.path.join(data_base_path, "binance")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def download_data_binance(token, training_days, region):
    """下载 Binance 历史数据（返回 81 维特征的 DataFrame）"""
    training_days = int(training_days)
    df = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {df.shape[0]} rows with {df.shape[1]} features.")
    return df

def download_data(token, training_days, region, data_provider):
    """根据数据提供方（Binance）下载数据"""
    if data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(df, data_provider):
    """格式化 81 维特征的数据"""
    if df is None or df.empty:
        print("No data available.")
        return
    
    df.to_csv(training_price_data_path, index=True)
    print(f"Formatted data saved to {training_price_data_path}")

def load_frame(frame, timeframe):
    """加载 81 维特征的 DataFrame"""
    print(f"[DEBUG] Loading frame with shape: {frame.shape}")

    df = frame.dropna()
    df = df.fillna(0)  # 确保所有 NaN 变成 0
    df['date'] = pd.to_datetime(df.index)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    print(f"[DEBUG] After preprocessing, frame shape: {df.shape}")

    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def train_model(timeframe):
    """训练机器学习模型"""
    df = pd.read_csv(training_price_data_path, index_col=0, parse_dates=True)
    df = load_frame(df, timeframe)

    print(f"[DEBUG] Training Data Shape (should be 81 features): {df.shape}")

    df = df.dropna()
    y_train = df['target_ETHUSDT'].values
    X_train = df.drop(columns=['target_ETHUSDT'])

    print(f"[DEBUG] X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
    print(f"[DEBUG] X_train columns: {X_train.columns.tolist()}")

    # **正确检测 `NaN`**
    if X_train.isna().any().any() or np.isnan(y_train).any():
        print("[ERROR] 数据仍然包含 NaN，检查数据预处理流程！")
        return

    # **定义模型**
    if MODEL == "LinearRegression":
        model = LinearRegression()
    elif MODEL == "SVR":
        model = SVR()
    elif MODEL == "KernelRidge":
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
    elif MODEL == "kNN":
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError("Unsupported model")
    
    # **训练模型**
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # **保存训练好的模型**
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    """加载训练好的模型并进行预测"""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # 获取最新 81 维特征数据
    if data_provider == "binance":
        X_new = download_binance_current_day_data(f"{TOKEN}USDT", region)
    else:
        raise ValueError("Unsupported data provider")

    X_new = load_frame(X_new, timeframe)

    print(f"[DEBUG] X_new Shape: {X_new.shape}")
    print(X_new.tail())

    # **去掉列名，防止 KNeighborsRegressor 报错**
    X_new = X_new.values
    print(f"[DEBUG] X_new Shape after removing column names: {X_new.shape}")

    # **进行预测**
    current_price_pred = loaded_model.predict(X_new)

    return current_price_pred[0]
