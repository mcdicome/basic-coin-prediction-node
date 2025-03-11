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

def train_model(timeframe):
    """训练机器学习模型"""
    df = pd.read_csv(training_price_data_path, index_col=0, parse_dates=True)
    
    df = df.dropna()
    y_train = df['target_ETHUSDT'].values
    X_train = df.drop(columns=['target_ETHUSDT'])

    print(f"[DEBUG] X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")

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
