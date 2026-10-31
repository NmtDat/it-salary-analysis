from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Cấu hình hiển thị
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

# 2. Xác định đường dẫn thư mục
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
CSV_FILE = REPORTS_DIR / "model_comparison.csv"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Kiểm tra file kết quả so sánh
if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file kết quả so sánh tại: {CSV_FILE}. "
        "Hãy chạy file 04_train_eval_models.py trước!"
    )

eval_df = pd.read_csv(CSV_FILE)
print("Đã tải bảng kết quả so sánh mô hình thành công.")

# 3. Chuẩn hóa dữ liệu để vẽ biểu đồ dạng cột nhóm (melt dataframe)
eval_melted = eval_df.melt(
    id_vars="Model",
    value_vars=["MAE", "RMSE", "R2_Score"],
    var_name="Metric",
    value_name="Score"
)

# 4. Vẽ biểu đồ so sánh các chỉ số
plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=eval_melted,
    x="Metric",
    y="Score",
    hue="Model",
    palette="muted"
)

plt.title("So sánh hiệu suất giữa Linear Regression và Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Chỉ số đánh giá (Metrics)", fontsize=12)
plt.ylabel("Giá trị", fontsize=12)
plt.legend(title="Mô hình", loc="upper right")

# Gắn giá trị cụ thể lên đầu mỗi cột biểu đồ để dễ nhìn
for p in barplot.patches:
    height = p.get_height()
    if height > 0:  # Chỉ hiển thị nếu cột có giá trị dương
        barplot.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 3),
            textcoords="offset points"
        )

plt.tight_layout()

# Lưu biểu đồ vào thư mục reports/figures/
output_fig = FIGURES_DIR / "model_performance_comparison.png"
plt.savefig(output_fig, dpi=300)
plt.close()

print(f"Đã lưu biểu đồ so sánh mô hình tại: {output_fig}")