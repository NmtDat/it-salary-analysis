from pathlib import Path
import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Cấu hình hiển thị biểu đồ
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# 2. Xác định đường dẫn thư mục
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "data" / "tech_jobs_for_model.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Đã tải dữ liệu thành công với {len(df)} dòng.")

# ==========================================
# 3. Xử lý và bóc tách kỹ năng chuẩn xác tuyệt đối
# ==========================================
if "technical_skills" in df.columns:
    skill_rows = []
    for _, row in df.iterrows():
        skills_raw = str(row["technical_skills"])
        if skills_raw and skills_raw.lower() != "nan":
            skill_list = []
            try:
                # Thử parse dạng list Python ['Java', 'Python']
                parsed = ast.literal_eval(skills_raw)
                if isinstance(parsed, (list, tuple, set)):
                    skill_list = [str(s).strip() for s in parsed if str(s).strip()]
                else:
                    skill_list = [s.strip() for s in str(skills_raw).split(",") if s.strip()]
            except (ValueError, SyntaxError):
                # Nếu là chuỗi phân tách thông thường
                skills_clean = skills_raw.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                skill_list = [s.strip() for s in skills_clean.split(",") if s.strip()]

            for skill in skill_list:
                clean_skill = skill.replace("'", "").replace('"', "").strip()
                
                # Bỏ qua các kỹ năng bị lỗi đóng/mở ngoặc hoặc cú pháp cắt cụt không hợp lệ
                if not clean_skill:
                    continue
                if clean_skill.endswith(")") and clean_skill.count("(") == 0:
                    continue
                if clean_skill.count("(") > clean_skill.count(")"):
                    # Tự động đóng ngoặc nếu thiếu dấu đóng ngoặc đơn
                    clean_skill += ")"
                
                skill_rows.append({
                    "skill": clean_skill,
                    "salary_avg_million": row["salary_avg_million"]
                })

    skill_df = pd.DataFrame(skill_rows)

    if not skill_df.empty:
        # Gom nhóm theo tên kỹ năng (phân biệt hoa thường nhẹ nhàng hoặc giữ nguyên form chuẩn)
        skill_stats = (
            skill_df.groupby("skill")
            .agg(
                job_count=("salary_avg_million", "count"),
                median_salary=("salary_avg_million", "median"),
                mean_salary=("salary_avg_million", "mean")
            )
            .reset_index()
        )

        # Làm tròn giá trị lương
        skill_stats["median_salary"] = skill_stats["median_salary"].round(2)
        skill_stats["mean_salary"] = skill_stats["mean_salary"].round(4)

        # Sắp xếp theo số lượng tin tuyển dụng giảm dần
        skill_stats = skill_stats.sort_values(by="job_count", ascending=False)

        # Lưu bảng thống kê kỹ năng sạch ra file báo cáo
        output_csv = REPORTS_DIR / "skill_salary_insights.csv"
        skill_stats.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"Đã chuẩn hóa và lưu bảng thống kê kỹ năng tại: {output_csv}")

        # ==========================================
        # 4. Vẽ biểu đồ Top 10 Kỹ năng phổ biến nhất
        # ==========================================
        plt.figure(figsize=(12, 6))
        top_skills = skill_stats.head(10)
        
        sns.barplot(
            data=top_skills,
            x="job_count",
            y="skill",
            palette="crest",
            hue="skill",
            legend=False
        )
        plt.title("Top 10 Kỹ năng công nghệ phổ biến nhất", fontsize=14, fontweight="bold")
        plt.xlabel("Số lượng tin tuyển dụng", fontsize=12)
        plt.ylabel("Kỹ năng", fontsize=12)
        plt.tight_layout()

        path_fig = FIGURES_DIR / "top_10_skills.png"
        plt.savefig(path_fig, dpi=300)
        plt.close()
        print(f"Đã lưu biểu đồ Top 10 Kỹ năng tại: {path_fig}")
    else:
        print("Không tìm thấy dữ liệu kỹ năng hợp lệ để bóc tách.")
else:
    print("Cột 'technical_skills' không tồn tại trong dữ liệu.")