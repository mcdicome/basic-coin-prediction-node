import os
import pickle
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from updater import download_binance_daily_data, download_binance_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, REGION

binance_data_path = os.path.join(data_base_path, "binance")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

def download_data(token, training_days, region, data_provider=None):
    """下载 Binance 历史数据（支持 `data_provider` 参数）"""
    training_days = int(training_days)
    
    if data_provider == "binance" or data_provider is None:
        df = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    else:
        raise ValueError(f"[ERROR] Unsupported data provider: {data_provider}")

    if df is None or df.empty:
        print("[ERROR] 下载数据为空，检查 `download_binance_daily_data`")
        return None

    print(f"[DEBUG] Downloaded {df.shape[0]} rows with {df.shape[1]} features.")
    return df

def format_data(df, data_provider=None):
    """格式化 81 维特征的数据"""
    if df is None or df.empty:
        print("[ERROR] No data available.")
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

    if X_train.isna().any().any() or np.isnan(y_train).any():
        print("[ERROR] 数据仍然包含 NaN，检查数据预处理流程！")
        return

    # 选择机器学习模型
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
    
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[SUCCESS] Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider=None):
    """加载模型并进行预测"""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    X_new = download_binance_current_day_data(region)

    if X_new is None or X_new.empty:
        return {"error": "No valid data available for inference"}

    print(f"[DEBUG] Current day DataFrame shape (should be 81 columns): {X_new.shape}")
    print(X_new.head())

    selected_columns = [
        f"ETHUSDT_open_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_high_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_low_lag{i}" for i in range(1, 11)] + \
        [f"ETHUSDT_close_lag{i}" for i in range(1, 11)] + \
        [f"BTCUSDT_open_lag{i}" for i in range(1, 11)] + \
        [f"BTCUSDT_high_lag{i}" for i in range(1, 11)] + \
        [f"BTCUSDT_low_lag{i}" for i in range(1, 11)] + \
        [f"BTCUSDT_close_lag{i}" for i in range(1, 11)] + \
        ["hour_of_day"]

    X_new = X_new[selected_columns]
    print(f"[DEBUG] Filtered X_new shape (should be (1, 81)): {X_new.shape}")

    X_new = X_new.iloc[-1:].to_numpy()
    print(f"[DEBUG] X_new Shape for prediction: {X_new.shape}")

    try:
        X_new = pd.DataFrame(X_new, columns=selected_columns)
        prediction = loaded_model.predict(X_new)
        return prediction[0]
    except Exception as e:
        return {"error": str(e)}
