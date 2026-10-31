from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


# 1. Đọc dữ liệu
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "data" / "tech_jobs_for_model.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# 2. Chuẩn bị dữ liệu
df["technical_skills"] = (
    df["technical_skills"].fillna("").astype(str)
)

df["job_title"] = df["job_title"].fillna("Không rõ")
df["location"] = df["location"].fillna("Không rõ")

X = df[
    [
        "job_title",
        "location",
        "experience_years",
        "technical_skills",
    ]
]

y = df["salary_avg_million"]

# 3. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 4. Xử lý các loại dữ liệu khác nhau (Có tích hợp Imputer cho kinh nghiệm)
experience_pipeline = make_pipeline(
    SimpleImputer(strategy="median"), 
    StandardScaler()
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "title_location",
            OneHotEncoder(handle_unknown="ignore"),
            ["job_title", "location"],
        ),
        (
            "experience",
            experience_pipeline,
            ["experience_years"],
        ),
        (
            "skills",
            TfidfVectorizer(max_features=100),
            "technical_skills",
        ),
    ]
)

# 5. Khởi tạo hai mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8,
        min_samples_leaf=2,
    ),
}

results = []

# 6. Huấn luyện và đánh giá
for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)  # Bổ sung R2 Score

    results.append(
        {
            "Model": name,
            "MAE (triệu đồng)": round(mae, 2),
            "RMSE (triệu đồng)": round(rmse, 2),
            "R2 Score": round(r2, 3),  # Đưa vào bảng kết quả
        }
    )

    print(f"\n===== {name} =====")
    print(f"MAE:  {mae:.2f} triệu đồng")
    print(f"RMSE: {rmse:.2f} triệu đồng")
    print(f"R2 Score: {r2:.3f}")

# 7. So sánh kết quả
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE (triệu đồng)")

print("\n===== BẢNG SO SÁNH MÔ HÌNH =====")
print(results_df.to_string(index=False))

OUTPUT_FILE = ROOT_DIR / "reports" / "model_comparison.csv"
results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nĐã lưu kết quả tại:", OUTPUT_FILE)