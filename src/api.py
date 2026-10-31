from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="IT Salary Prediction API",
    description="API dự đoán mức lương lập trình viên dựa trên Machine Learning",
    version="1.0.0"
)

# 2. Xác định đường dẫn mô hình
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "salary_random_forest.joblib"

# Tải mô hình khi khởi động API
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy mô hình tại: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# 3. Định nghĩa cấu trúc dữ liệu đầu vào bằng Pydantic (để kiểm tra lỗi tự động)
class JobInput(BaseModel):
    experience_years: float = Field(..., example=2.0, description="Số năm kinh nghiệm")
    job_title: str = Field(..., example="Backend Developer", description="Vị trí công việc")
    location: str = Field(..., example="Hồ Chí Minh", description="Địa điểm làm việc")

# 4. Định nghĩa Endpoint gốc kiểm tra server
@app.get("/", tags=["System Status"])
def read_root():
    return {"message": "Chào mừng bạn đến với IT Salary Prediction API!"}

# 5. Định nghĩa Endpoint dự đoán lương
@app.post("/predict", tags=["Salary Prediction"])
def predict_salary(data: JobInput):
    try:
        # Chuyển dữ liệu nhận được thành DataFrame đúng định dạng mô hình yêu cầu
        input_df = pd.DataFrame([{
            "experience_years": data.experience_years,
            "job_title": data.job_title,
            "location": data.location
        }])

        # Dự đoán bằng mô hình Random Forest Pipeline đã lưu
        prediction = float(model.predict(input_df)[0])

        return {
            "status": "success",
            "input_data": data.dict(),
            "predicted_salary_million_vnd": round(prediction, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))