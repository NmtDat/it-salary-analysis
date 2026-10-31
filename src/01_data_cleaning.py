from pathlib import Path
import re
import pandas as pd

# 1. Xác định đường dẫn thư mục
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "raw" / "VietJobs.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "tech_jobs_for_model.csv"

# Kiểm tra file dữ liệu thô
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc tại: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Số tin tuyển dụng ban đầu: {len(df)}")

# ==========================================
# PHẦN A: LỌC NGÀNH CNTT & LÀM SẠCH LƯƠNG, KINH NGHIỆM
# ==========================================

# 1. Chỉ giữ nhóm ngành CNTT
it_df = df[
    df["category"].str.strip().str.lower()
    == "công_nghệ_thông_tin_kỹ_thuật_số"
].copy()

print(f"Số tin thuộc nhóm CNTT: {len(it_df)}")

# 2. Chuyển cột lương sang dạng số
for col in ["salary_min", "salary_max"]:
    it_df[col] = pd.to_numeric(it_df[col], errors="coerce")

# 3. Loại bỏ lương không hợp lệ (thiếu, <= 0, min > max)
valid_salary = (
    it_df["salary_min"].notna()
    & it_df["salary_max"].notna()
    & (it_df["salary_min"] > 0)
    & (it_df["salary_max"] > 0)
    & (it_df["salary_min"] <= it_df["salary_max"])
)

it_df = it_df.loc[valid_salary].copy()

# 4. Tính mức lương trung bình (triệu đồng)
it_df["salary_avg_million"] = (
    it_df["salary_min"] + it_df["salary_max"]
) / 2


# 5. Hàm chuẩn hóa kinh nghiệm sang số năm
def parse_experience(value):
    if pd.isna(value):
        return float("nan")

    text = str(value).strip().lower()

    if "không yêu cầu" in text:
        return 0.0

    # Bóc tách số tháng
    month_match = re.search(r"(\d+(?:[.,]\d+)?)\s*tháng", text)
    if month_match:
        months = float(month_match.group(1).replace(",", "."))
        return months / 12

    # Bóc tách số năm
    year_match = re.search(r"(\d+(?:[.,]\d+)?)\s*năm", text)
    if year_match:
        return float(year_match.group(1).replace(",", "."))

    return float("nan")


it_df["experience_years"] = (
    it_df["experience_required"].apply(parse_experience)
)

# Chuẩn hóa các trường văn bản cơ bản
for col in ["job_title", "technical_skills"]:
    it_df[col] = it_df[col].fillna("").astype(str).str.strip()


# ==========================================
# PHẦN B: LỌC CHUYÊN SÂU CHỨC DANH KỸ THUẬT & CHUẨN HÓA ĐỊA ĐIỂM
# ==========================================

# Danh sách từ khóa nhận diện lập trình viên, kỹ sư, tester, data, devops...
keywords = [
    "developer",
    "software engineer",
    "software developer",
    "lập trình",
    "backend",
    "back-end",
    "frontend",
    "front-end",
    "full-stack",
    "fullstack",
    "web developer",
    "mobile developer",
    "programmer",
    "kỹ sư phần mềm",
    "tester",
    "qa engineer",
    "quality assurance",
    "data engineer",
    "data scientist",
    "data analyst",
    "machine learning",
    "ai engineer",
    "artificial intelligence",
    "devops",
    "cloud engineer",
    "database administrator",
    "system administrator",
    "quản trị hệ thống",
    "an ninh mạng",
    "cybersecurity",
    "network engineer",
    "kỹ thuật viên máy tính",
    "it support",
    "helpdesk",
]

title = it_df["job_title"].str.lower()
pattern = "|".join(
    pd.Series(keywords).str.replace(
        r"([.*+?^${}()|\[\]\\])", r"\\\1", regex=True
    )
)

mask = title.str.contains(pattern, regex=True, na=False)
tech_df = it_df.loc[mask].copy()


# Hàm chuẩn hóa tên thành phố / địa điểm chuyên sâu
def clean_location(loc_value):
    if pd.isna(loc_value) or not isinstance(loc_value, str):
        return "Khác"
    
    loc_str = loc_value.strip()
    if not loc_str or loc_str.lower() in ["not specified", "nan", "none", "khác"]:
        return "Khác"
    
    loc_lower = loc_str.lower()
    
    # 1. Gom nhóm các quận/huyện hoặc biến thể về Hà Nội
    hanoi_keywords = [
        "hà nội", "ha noi", "hoàng mai", "cầu giấy", "đống đa", "thanh xuân", 
        "hai bà trưng", "ba đình", "hoàn kiếm", "tây hồ", "long biên", "nam từ liêm", 
        "bắc từ liêm", "hà đông", "gia lâm", "đông anh", "sóc sơn", "thanh trì"
    ]
    if any(kw in loc_lower for kw in hanoi_keywords):
        return "Hà Nội"
        
    # 2. Gom nhóm các quận/huyện hoặc biến thể về Hồ Chí Minh
    hcm_keywords = [
        "hồ chí minh", "ho chi minh", "hcm", "sài gòn", "saigon", 
        "quận 1", "quận 2", "quận 3", "quận 4", "quận 5", "quận 6", "quận 7", 
        "quận 8", "quận 9", "quận 10", "quận 11", "quận 12", "tân bình", 
        "tân phú", "bình tân", "phú nhuận", "gò vấp", "bình thạnh", "thủ đức", "bình chánh"
    ]
    if any(kw in loc_lower for kw in hcm_keywords):
        return "Hồ Chí Minh"
        
    # 3. Gom nhóm Đà Nẵng
    danang_keywords = ["đà nẵng", "da nang", "hải châu", "thanh khê", "sơn trà"]
    if any(kw in loc_lower for kw in danang_keywords):
        return "Đà Nẵng"
        
    # 4. Các tỉnh thành khác chuẩn hóa viết hoa chữ cái đầu
    first_part = loc_str.split(",")[0].strip()
    return first_part.title()

# Áp dụng vào cột location
tech_df["location"] = tech_df["location"].apply(clean_location)


# ==========================================
# PHẦN C: XUẤT KẾT QUẢ
# ==========================================
print("\n===== KẾT QUẢ SAU KHI LÀM SẠCH & LỌC =====")
print(f"Số tin cuối cùng đưa vào mô hình: {len(tech_df)}")
print(f"Phân bố lương trung bình (triệu VNĐ):\n{tech_df['salary_avg_million'].describe()}")

# Lưu ra file hoàn chỉnh để dùng cho các bước huấn luyện phía sau
tech_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\nĐã lưu file thành công tại: {OUTPUT_FILE}")