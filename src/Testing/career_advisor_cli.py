from pathlib import Path
import joblib
import pandas as pd

# 1. Xác định đường dẫn thư mục và file mô hình
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT_DIR / "models" / "salary_random_forest.joblib"

def main():
    print("==================================================")
    print("    CÔNG CỤ TƯ VẤN LƯƠNG & NGHỀ NGHIỆP IT (CLI)    ")
    print("==================================================")
    
    if not MODEL_FILE.exists():
        print(f"\n[LỖI] Không tìm thấy file mô hình tại: {MODEL_FILE}")
        print("Vui lòng chạy script '05_save_final_model.py' trước để huấn luyện và lưu mô hình!")
        return

    # Tải mô hình
    print("Đang tải mô hình học máy từ thư mục models...")
    try:
        model = joblib.load(MODEL_FILE)
        print("Tải mô hình thành công!\n")
    except Exception as e:
        print(f"[LỖI] Không thể tải mô hình: {e}")
        return

    while True:
        print("-" * 50)
        print("Nhập thông tin tra cứu (hoặc gõ 'q' để thoát):")
        
        # Nhập số năm kinh nghiệm
        exp_input = input(" - Số năm kinh nghiệm (VD: 2.5): ").strip()
        if exp_input.lower() == 'q':
            print("Cảm ơn bạn đã sử dụng công cụ tư vấn lương IT!")
            break
        
        try:
            experience_years = float(exp_input)
        except ValueError:
            print("[CẢNH BÁO] Số năm kinh nghiệm phải là một số hợp lệ. Vui lòng nhập lại!\n")
            continue

        # Nhập vị trí công việc
        job_title = input(" - Vị trí công việc (VD: Software Engineer, Data Scientist): ").strip()
        
        # Nhập địa điểm làm việc
        location = input(" - Địa điểm làm việc (VD: Ho Chi Minh, Hanoi): ").strip()

        # Tạo DataFrame đầu vào cho mô hình
        input_data = pd.DataFrame({
            "experience_years": [experience_years],
            "job_title": [job_title],
            "location": [location]
        })

        try:
            # Thực hiện dự đoán mức lương
            predicted_salary = model.predict(input_data)[0]

            print("\n" + "=" * 45)
            print(" 💡 KẾT QUẢ DỰ ĐOÁN MỨC LƯƠNG:")
            print(f" • Vị trí công việc : {job_title}")
            print(f" • Kinh nghiệm      : {experience_years} năm")
            print(f" • Địa điểm         : {location}")
            print(f" -> Mức lương dự kiến: {predicted_salary:.2f} Triệu VNĐ/tháng")
            print("=" * 45 + "\n")
            
        except Exception as e:
            print(f"[LỖI] Đã xảy ra lỗi trong quá trình dự đoán: {e}\n")

if __name__ == "__main__":
    main()