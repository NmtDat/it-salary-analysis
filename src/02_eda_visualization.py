from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Cấu hình hiển thị biểu đồ tiếng Việt & giao diện
plt.rcParams["font.sans-serif"] = "DejaVu Sans"  # Hoặc font hỗ trợ tiếng Việt nếu có
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# 2. Xác định đường dẫn thư mục
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "tech_jobs_for_model.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Tạo thư mục chứa biểu đồ nếu chưa có
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Kiểm tra file dữ liệu
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Đã tải dữ liệu thành công với {len(df)} dòng.")

# ==========================================
# BIỂU ĐỒ 1: Tương quan giữa Số năm kinh nghiệm và Mức lương trung bình
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="experience_years",
    y="salary_avg_million",
    alpha=0.7,
    color="teal"
)
sns.regplot(
    data=df,
    x="experience_years",
    y="salary_avg_million",
    scatter=False,
    color="darkorange"
)
plt.title("Mối quan hệ giữa Số năm kinh nghiệm và Mức lương", fontsize=14, fontweight="bold")
plt.xlabel("Số năm kinh nghiệm", fontsize=12)
plt.ylabel("Mức lương trung bình (Triệu VNĐ/tháng)", fontsize=12)
plt.tight_layout()

path_fig1 = FIGURES_DIR / "experience_vs_salary.png"
plt.savefig(path_fig1, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ 1 tại: {path_fig1}")

# ==========================================
# BIỂU ĐỒ 2: Top các vị trí công việc có số lượng tin tuyển dụng nhiều nhất
# ==========================================
plt.figure(figsize=(12, 6))
top_roles = df["job_title"].value_counts().head(10)
sns.barplot(
    x=top_roles.values,
    y=top_roles.index,
    palette="viridis",
    hue=top_roles.index,
    legend=False
)
plt.title("Top 10 Vị trí công việc phổ biến nhất", fontsize=14, fontweight="bold")
plt.xlabel("Số lượng tin tuyển dụng", fontsize=12)
plt.ylabel("Vị trí công việc", fontsize=12)
plt.tight_layout()

path_fig2 = FIGURES_DIR / "top_10_roles.png"
plt.savefig(path_fig2, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ 2 tại: {path_fig2}")

# ==========================================
# BIỂU ĐỒ 3: Phân bố mức lương trung bình theo Thành phố lớn
# ==========================================
plt.figure(figsize=(12, 6))
top_cities = df["location"].value_counts().head(5).index
filtered_city_df = df[df["location"].isin(top_cities)]

sns.boxplot(
    data=filtered_city_df,
    x="location",
    y="salary_avg_million",
    palette="Set2",
    hue="location",
    legend=False
)
plt.title("Phân bố mức lương theo các thành phố lớn", fontsize=14, fontweight="bold")
plt.xlabel("Địa điểm", fontsize=12)
plt.ylabel("Mức lương (Triệu VNĐ/tháng)", fontsize=12)
plt.tight_layout()

path_fig3 = FIGURES_DIR / "salary_by_location.png"
plt.savefig(path_fig3, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ 3 tại: {path_fig3}")

print("\nHoàn thành việc vẽ và lưu tất cả biểu đồ EDA vào thư mục reports/figures/!")