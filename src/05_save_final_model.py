from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Xác định đường dẫn thư mục
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "tech_jobs_for_model.csv"
MODELS_DIR = ROOT_DIR / "models"

# Tạo thư mục models nếu chưa tồn tại
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Kiểm tra file dữ liệu
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Đã tải dữ liệu thành công với {len(df)} dòng để huấn luyện mô hình chính thức.")

# 2. Định nghĩa Features (Đầu vào) và Target (Đầu ra)
features = ["experience_years", "job_title", "location"]
target = "salary_avg_million"

df_model = df.dropna(subset=features + [target]).copy()

X = df_model[features]
y = df_model[target]

# 3. Xây dựng Pipeline tiền xử lý và mô hình Random Forest
categorical_features = ["job_title", "location"]
numeric_features = ["experience_years"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features
        ),
    ]
)

# Khởi tạo mô hình Random Forest (mô hình tối ưu đã chọn)
model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", model)]
)

print("\n--- ĐANG HUẤN LUYỆN MÔ HÌNH TRÊN TOÀN BỘ DỮ LIỆU SẠCH ---")
pipeline.fit(X, y)
print("Huấn luyện hoàn tất!")

# 4. Lưu mô hình ra file joblib chuẩn
model_output_path = MODELS_DIR / "salary_random_forest.joblib"
joblib.dump(pipeline, model_output_path)
print(f"Đã lưu file mô hình chính thức tại: {model_output_path}")