from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Đường dẫn và đọc dữ liệu
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "tech_jobs_for_model.csv"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, low_memory=False)

# 2. Biểu đồ Scatterplot/Regplot: Tương quan Kinh nghiệm vs Lương
plt.figure(figsize=(9, 5))
sns.regplot(
    data=df,
    x="experience_years",
    y="salary_avg_million",
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"},
)
plt.title("Tương quan giữa Kinh nghiệm và Mức lương")
plt.xlabel("Kinh nghiệm (năm)")
plt.ylabel("Lương trung bình (triệu VNĐ)")
plt.savefig(FIGURE_DIR / "experience_vs_salary.png", dpi=150)
plt.close()

# 3. Biểu đồ Barplot: So sánh Lương theo Vai trò & Khu vực (Chỉ lấy Top 10 vị trí phổ biến nhất)
top_roles = df["job_title"].value_counts().head(10).index
top_cities = df["location"].value_counts().head(3).index

df_filtered = df[
    df["job_title"].isin(top_roles) & df["location"].isin(top_cities)
]

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_filtered,
    x="job_title",
    y="salary_avg_million",
    hue="location",
    estimator="median",
    errorbar=None,
)
plt.title("Mức lương trung vị theo Top 10 Vị trí và Khu vực")
plt.xlabel("Vị trí công việc")
plt.ylabel("Lương trung vị (triệu VNĐ)")

# Xoay nhãn trục x 45 độ và căn lề phải để không bị đè chữ
plt.xticks(rotation=45, ha="right")
plt.legend(title="Khu vực")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "salary_role_location.png", dpi=150)
plt.close()