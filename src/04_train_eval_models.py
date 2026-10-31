from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Xác định đường dẫn thư mục
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "tech_jobs_for_model.csv"
REPORTS_DIR = ROOT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Kiểm tra file dữ liệu
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Đã tải dữ liệu thành công với {len(df)} dòng để huấn luyện.")

# 2. Định nghĩa Features (Đầu vào) và Target (Đầu ra)
features = ["experience_years", "job_title", "location"]
target = "salary_avg_million"

# Lọc bỏ các dòng bị thiếu dữ liệu ở các cột cốt lõi
df_model = df.dropna(subset=features + [target]).copy()

X = df_model[features]
y = df_model[target]

# Chia tập dữ liệu Train / Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Xây dựng bộ tiền xử lý dữ liệu trong Pipeline (Preproccessor)
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

# 4. Khởi tạo 2 mô hình cần so sánh
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

evaluation_results = []

print("\n--- ĐANG TIẾN HÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ---")
for name, model in models.items():
    # Tạo Pipeline hoàn chỉnh: Tiền xử lý -> Mô hình học máy
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model)]
    )

    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)

    # Dự đoán trên tập Test
    y_pred = pipeline.predict(X_test)

    # Tính toán các chỉ số đánh giá
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nMô hình: {name}")
    print(f" - MAE  : {mae:.4f}")
    print(f" - RMSE : {rmse:.4f}")
    print(f" - R2   : {r2:.4f}")

    evaluation_results.append({
        "Model": name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4)
    })

# 5. Lưu kết quả so sánh ra file CSV phục vụ báo cáo
eval_df = pd.DataFrame(evaluation_results)
output_csv = REPORTS_DIR / "model_comparison.csv"
eval_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\nĐã lưu bảng so sánh mô hình tại: {output_csv}")