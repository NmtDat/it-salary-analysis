# 💻 IT SALARY ANALYSIS & PREDICTION SYSTEM
> **Đồ án môn học / Xây dựng hệ thống dự đoán và phân tích thị trường việc làm CNTT toàn diện (Full-stack Data Science)**

---

## 📋 1. Giới thiệu tổng quan đồ án
Đồ án xây dựng một hệ thống hoàn chỉnh từ khâu thu thập, làm sạch dữ liệu tuyển dụng ngành CNTT tại Việt Nam, huấn luyện mô hình Machine Learning dự đoán mức lương, cho đến việc triển khai dưới dạng **Web App tương tác (Streamlit)** tích hợp kiến trúc **REST API (FastAPI)** chuẩn doanh nghiệp.

---

## 🗂️ 2. Cấu trúc thư mục dự án
Toàn bộ mã nguồn và tài liệu của dự án được tổ chức theo chuẩn Pipeline kỹ thuật từ `01` đến `06`:

```text
IT_Salary_Analysis/
│
├── data/
│   ├── raw/                       # Dữ liệu tuyển dụng thô gốc (VietJobs.csv)
│   └── tech_jobs_for_model.csv    # Dữ liệu sạch đã qua xử lý để huấn luyện mô hình
│
├── models/
│   └── salary_random_forest.joblib # Mô hình Machine Learning đã lưu (Pipeline chuẩn)
│
├── reports/
│   ├── figures/                   # Các biểu đồ phân tích EDA và so sánh mô hình
│   ├── model_comparison.csv       # Bảng so sánh giữa các mô hình (Linear Regression vs Random Forest)
│   └── skill_salary_insights.csv  # Thống kê kỹ năng và mức lương trung vị
│
├── src/
│   ├── 01_data_cleaning.py        # Bước 1: Làm sạch dữ liệu, chuẩn hóa location & kỹ năng
│   ├── 02_eda_visualization.py    # Bước 2: Phân tích khám phá dữ liệu (EDA) & vẽ biểu đồ
│   ├── 03_skill_analysis.py       # Bước 3: Phân tích xu hướng kỹ năng & mức lương
│   ├── 04_train_eval_models.py    # Bước 4: Huấn luyện và đánh giá/so sánh các mô hình ML
│   ├── 05_save_final_model.py     # Bước 5: Lưu mô hình tối ưu chính thức (Random Forest)
│   ├── 06_plot_model_metrics.py   # Bước 6: Vẽ biểu đồ đánh giá hiệu suất mô hình
│   ├── api.py                     # Backend REST API xây dựng bằng FastAPI
│   └── streamlit_app.py           # Frontend Web App tương tác gọi qua FastAPI / Model
│
├── requirements.txt               # Danh sách thư viện Python cần thiết
└── README.md                      # Tài liệu tổng quan hướng dẫn dự án
```

---

## 👥 3. Phân công công việc các thành viên trong nhóm

| Thành viên | Phân công công việc cụ thể | File mã nguồn / Nhiệm vụ phụ trách |
| :--- | :--- | :--- |
| **Nguyễn Bá long** <br>*(Xử lý Dữ liệu)* | Tìm kiếm dataset, kiểm tra nguồn gốc dữ liệu, viết code làm sạch, chuẩn hóa mức lương, địa điểm và bóc tách kỹ năng. | • `01_data_cleaning.py` |
| **Nguyễn Bá long** <br>*(Phân tích Dữ liệu)* | Sử dụng Pandas, Seaborn, Matplotlib để phân tích thống kê khám phá (EDA) và tìm ra xu hướng tuyển dụng thị trường. | • `02_eda_visualization.py`<br>• `03_skill_analysis.py` |
| **Nguyễn Minh Thành Đạt** <br>*(Machine Learning)* | Xây dựng pipeline, huấn luyện, đánh giá và so sánh hiệu suất các mô hình dự đoán (Linear Regression, Random Forest). | • `04_train_eval_models.py`<br>• `05_save_final_model.py`<br>• `06_plot_model_metrics.py` |
| **Võ Thành Long** <br>*(Ứng dụng / Deployment)* | Phát triển giao diện web tương tác bằng Streamlit, tích hợp gọi REST API qua FastAPI để xử lý dự đoán thời gian thực. | • `src/api.py`<br>• `src/streamlit_app.py` |
| **Bùi Yến Nhi** <br>*(Báo cáo & Kiểm thử)* | Tổng hợp tài liệu báo cáo, quản lý phiên bản Git/GitHub, kiểm thử toàn bộ hệ thống (Integration Testing) và chuẩn bị slide thuyết trình. | • Quản lý chung dự án,<br>• Tài liệu `README.md` |

---

## ⚙️ 4. Hướng dẫn cài đặt và chạy hệ thống

### Bước 1: Clone dự án và kích hoạt môi trường ảo
```bash
# Mở terminal tại thư mục dự án
python -m venv .venv
# Kích hoạt môi trường ảo (Windows PowerShell)
.venv\Scripts\Activate
```

### Bước 2: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

---

### Bước 3: Chạy hệ thống (Full-stack Mode)

Để hệ thống hoạt động theo mô hình **Client - Server** chuẩn thực tế (Giao diện Streamlit gọi ngầm qua FastAPI Backend), bạn cần mở **2 cửa sổ Terminal** và chạy đồng thời:

#### 🔹 Terminal 1: Khởi chạy Backend FastAPI
```bash
uvicorn src.api:app --reload
```
* Sau khi chạy thành công, bạn có thể kiểm tra tài liệu API tự động (Swagger UI) tại trình duyệt: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### 🔹 Terminal 2: Khởi chạy Frontend Streamlit
```bash
streamlit run src/streamlit_app.py
```
* Giao diện web trực quan sẽ tự động mở lên tại trình duyệt: [http://localhost:8501](http://localhost:8501)

---

## 📊 5. Kết quả đạt được của mô hình
* **Thuật toán được chọn**: Random Forest Regressor (đạt độ chính xác và ổn định cao hơn so với Linear Regression sau quá trình đánh giá trên tập Test).
* **Tính năng đầu vào**: Số năm kinh nghiệm (`experience_years`), Vị trí công việc (`job_title`), Địa điểm làm việc (`location`).
* **Đầu ra**: Ước tính mức lương trung bình (triệu VNĐ/tháng) kèm theo các gợi ý kỹ năng giá trị cần học theo từng vị trí.

---
*Đồ án hoàn thành phục vụ cho việc báo cáo trước hội đồng chuyên môn.*