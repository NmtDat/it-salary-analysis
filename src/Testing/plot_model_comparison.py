
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "reports" / "model_comparison.csv"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Chuẩn hóa tên cột để dễ sử dụng
df.columns = df.columns.str.strip()

# Kiểm tra dữ liệu trước khi vẽ
print("Dữ liệu so sánh mô hình:")
print(df.to_string(index=False))

# Biểu đồ MAE
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["MAE (triệu đồng)"])
plt.title("So sánh MAE của các mô hình")
plt.xlabel("Mô hình")
plt.ylabel("MAE (triệu đồng)")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "model_comparison_mae.png",
    dpi=150
)
plt.close()

# Biểu đồ RMSE
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["RMSE (triệu đồng)"])
plt.title("So sánh RMSE của các mô hình")
plt.xlabel("Mô hình")
plt.ylabel("RMSE (triệu đồng)")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "model_comparison_rmse.png",
    dpi=150
)
plt.close()

print("\nĐã lưu hai biểu đồ vào reports/figures/")
