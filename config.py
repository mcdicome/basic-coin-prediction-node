import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
model_file_path = os.path.join(data_base_path, "model.pkl")

TOKEN = os.getenv("TOKEN", "").upper()
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", "30"))  # 默认值 30 天
TIMEFRAME = os.getenv("TIMEFRAME")

# **✅ 增加默认模型，并确保支持 kNN**
MODEL = os.getenv("MODEL", "LinearRegression")  # 默认使用 LinearRegression
SUPPORTED_MODELS = ["LinearRegression", "SVR", "KernelRidge", "BayesianRidge", "kNN"]

if MODEL not in SUPPORTED_MODELS:
    print(f"⚠️  Unsupported model '{MODEL}', falling back to LinearRegression")
    MODEL = "LinearRegression"

REGION = os.getenv("REGION", "com").lower()
if REGION in ["us", "com", "usa"]:
    REGION = "us"
else:
    REGION = "com"

DATA_PROVIDER = os.getenv("DATA_PROVIDER", "binance").lower()
CG_API_KEY = os.getenv("CG_API_KEY", default=None)
