from pathlib import Path
import ast

import joblib
import pandas as pd
import streamlit as st
import requests


# =========================
# 1. Cấu hình ứng dụng
# =========================
st.set_page_config(
    page_title="IT Salary Analysis",
    page_icon="💻",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "salary_random_forest.joblib"
SKILL_PATH = PROJECT_DIR / "reports" / "skill_salary_insights.csv"
DATA_PATH = PROJECT_DIR / "data" / "tech_jobs_for_model.csv"

st.title("💻 PHÂN TÍCH THỊ TRƯỜNG VIỆC LÀM CNTT")
st.subheader("Ứng dụng dự đoán mức lương lập trình viên")
st.write(
    "Nhập thông tin nghề nghiệp và kỹ năng để ước tính mức lương "
    "và tham khảo các kỹ năng nên học."
)


# =========================
# 2. Kiểm tra dữ liệu & Tài nguyên
# =========================
if not MODEL_PATH.exists():
    st.error(f"Không tìm thấy mô hình: {MODEL_PATH}")
    st.stop()

if not SKILL_PATH.exists():
    st.error(f"Không tìm thấy dữ liệu kỹ năng: {SKILL_PATH}")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_skills(path):
    return pd.read_csv(path)

@st.cache_data
def load_job_data(path):
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()

try:
    model = load_model(str(MODEL_PATH))
    skill_df = load_skills(str(SKILL_PATH))
    jobs_df = load_job_data(DATA_PATH)
except Exception as error:
    st.error(f"Không thể tải mô hình hoặc dữ liệu: {error}")
    st.stop()


# =========================
# 3. Đồng bộ danh mục từ dữ liệu thực tế
# =========================
if not jobs_df.empty and "job_title" in jobs_df.columns:
    role_options = jobs_df["job_title"].dropna().astype(str).value_counts().head(20).index.tolist()
else:
    role_options = [
        "Backend Developer",
        "Full-stack Developer",
        "Frontend Developer",
        "Software Engineer",
        "Mobile Developer",
        "QA Engineer",
        "AI Engineer",
        "Data Engineer"
    ]

if not jobs_df.empty and "location" in jobs_df.columns:
    raw_cities = jobs_df["location"].dropna().astype(str).tolist()
    unique_cities = set()
    
    for item in raw_cities:
        # Bỏ qua các chuỗi chứa dấu phẩy bị gộp lỗi
        if "," in item:
            continue
            
        clean_city = item.strip().title()
        if clean_city and clean_city.lower() not in ["not specified", "nan", "none", "khác"]:
            unique_cities.add(clean_city)
    
    # Ưu tiên đưa các thành phố lớn lên đầu danh sách
    priority_cities = ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng"]
    sorted_other_cities = sorted([c for c in unique_cities if c not in priority_cities])
    
    city_options = [c for c in priority_cities if c in unique_cities] + sorted_other_cities
else:
    city_options = ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Khác"]
    
st.markdown("### 1. Thông tin công việc")

col1, col2, col3 = st.columns(3)

with col1:
    job_title = st.selectbox(
        "Vị trí công việc",
        role_options
    )

with col2:
    location = st.selectbox(
        "Địa điểm",
        city_options
    )

with col3:
    experience_years = st.number_input(
        "Số năm kinh nghiệm",
        min_value=0.0,
        max_value=40.0,
        value=2.0,
        step=0.5
    )

st.markdown("### 2. Kỹ năng hiện có")

if "technical_skills" in skill_df.columns:
    skill_column = "technical_skills"
elif "skill" in skill_df.columns:
    skill_column = "skill"
else:
    skill_column = None

skill_options = []
if skill_column:
    raw_skills = skill_df[skill_column].dropna().astype(str).tolist()
    skill_options_set = set()

    for item in raw_skills:
        try:
            parsed = ast.literal_eval(item)
            if isinstance(parsed, (list, tuple, set)):
                items_to_add = [str(s).strip() for s in parsed if str(s).strip()]
            else:
                items_to_add = [s.strip() for s in item.split(",") if s.strip()]
        except (ValueError, SyntaxError):
            items_to_add = [s.strip() for s in item.split(",") if s.strip()]

        for s in items_to_add:
            clean_s = s.replace("'", "").replace('"', "").strip()
            if not clean_s:
                continue
            if clean_s.endswith(")") and clean_s.count("(") == 0:
                continue
            if clean_s.count("(") > clean_s.count(")"):
                clean_s += ")"
                
            skill_options_set.add(clean_s)

    skill_options = sorted(list(skill_options_set))

selected_skills = st.multiselect(
    "Chọn kỹ năng bạn đã biết",
    options=skill_options,
    placeholder="Ví dụ: Python, SQL, Docker"
)


# =========================
# 4. Dự đoán lương (Gọi qua FastAPI)
# =========================
st.markdown("### 3. Kết quả dự đoán")

if st.button("🔍 Dự đoán mức lương", type="primary"):
    # Địa chỉ Endpoint của FastAPI (đang chạy ở port 8000)
    api_url = "http://127.0.0.1:8000/predict"
    
    # Đóng gói dữ liệu đầu vào theo đúng schema Pydantic bên FastAPI
    payload = {
        "experience_years": float(experience_years),
        "job_title": str(job_title),
        "location": str(location)
    }

    try:
        # Gửi request phương thức POST tới FastAPI
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            prediction = result["predicted_salary_million_vnd"]

            st.success("Đã hoàn thành dự đoán qua API thành công!")
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Mức lương dự đoán",
                    f"{prediction:.2f} triệu đồng/tháng"
                )

            with result_col2:
                st.metric("Kinh nghiệm", f"{experience_years} năm")

            st.caption(
                "Đây là kết quả ước tính từ mô hình thông qua REST API, "
                "không phải mức lương cam kết."
            )
        else:
            st.error(f"Lỗi từ server API: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối tới API server! Hãy chắc chắn bạn đã chạy lệnh khởi động FastAPI (uvicorn).")
    except Exception as error:
        st.error(f"Đã xảy ra lỗi: {error}")
        
        
# =========================
# 5. Gợi ý kỹ năng
# =========================
st.divider()
st.markdown("### 4. Kỹ năng tham khảo phù hợp theo vị trí")

if {"job_count", "median_salary"}.issubset(skill_df.columns):
    display_df = skill_df.copy()

    if "skill" not in display_df.columns:
        possible_names = [
            column for column in display_df.columns
            if column.lower() in {"technology", "technical_skill", "name"}
        ]
        if possible_names:
            display_df = display_df.rename(
                columns={possible_names[0]: "skill"}
            )

    if "skill" in display_df.columns:
        role_skills_map = {
            "backend": ["Java", "Python", "SQL", "PostgreSQL", "MySQL", "Docker", "Git", "Spring Boot", "AWS", "Redis"],
            "frontend": ["JavaScript", "TypeScript", "HTML", "CSS", "ReactJS", "Git"],
            "data": ["Python", "SQL", "Excel", "Power BI", "Tableau", "PostgreSQL", "Spark"],
            "devops": ["Linux", "Docker", "Kubernetes", "AWS", "Azure", "Git", "Grafana"],
        }
        
        role_lower = job_title.casefold()
        matched_group = next((key for key in role_skills_map if key in role_lower), None)

        if matched_group:
            allowed = {s.casefold() for s in role_skills_map[matched_group]}
            mask = display_df["skill"].astype(str).str.casefold().isin(allowed)
            if mask.sum() > 0:
                display_df = display_df[mask]

        display_df = display_df[
            ~display_df["skill"].astype(str).isin(selected_skills)
        ].sort_values(
            by=["job_count", "median_salary"],
            ascending=[False, False]
        )

        st.write(
            f"Gợi ý các kỹ năng quan trọng nhất cho vị trí **{job_title}** "
            "dựa trên dữ liệu tuyển dụng và mức lương trung vị:"
        )

        st.dataframe(display_df.head(10), width="stretch", hide_index=True)
        
    else:
        st.info("Chưa xác định được tên cột kỹ năng trong file CSV.")
else:
    st.info("File dữ liệu kỹ năng chưa có các cột job_count và median_salary.")