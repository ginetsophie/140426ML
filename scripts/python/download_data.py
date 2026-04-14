import kagglehub
import os
import shutil
from pathlib import Path

# 1. Tải dataset
path_telco = kagglehub.dataset_download("blastchar/telco-customer-churn")
path_bank = kagglehub.dataset_download("tranhuunhan/vietnam-bank-churn-dataset-2025")

# 2. Tạo thư mục đích nếu chưa có
destination_dir = Path("./data")
destination_dir.mkdir(parents=True, exist_ok=True)

# 3. Hàm hỗ trợ copy file CSV
def move_csv_files(src_dir, dest_name):
    for file in os.listdir(src_dir):
        if file.endswith(".csv"):
            src_file = os.path.join(src_dir, file)
            dest_file = destination_dir / dest_name
            shutil.copy(src_file, dest_file)
            print(f"Đã chép: {file} -> {dest_file}")
            break # Lấy file đầu tiên tìm thấy

# 4. Thực hiện copy
move_csv_files(path_telco, "data_telco.csv")
move_csv_files(path_bank, "data_bank.csv")