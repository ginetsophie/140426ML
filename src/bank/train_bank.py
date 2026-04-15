import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import joblib

# Tắt cảnh báo để terminal sạch sẽ
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

# =================================================================================================
# GIẢI PHẪU CHI TIẾT CÁC HÀM TỪ THƯ VIỆN SCIKIT-LEARN (DÀNH CHO NEWBIE)
# =================================================================================================

# 1. train_test_split
# - Lý thuyết: Kỹ thuật Hold-out Validation (Chia để trị). Trong Machine Learning, ta không được phép 
#   kiểm tra model trên chính dữ liệu nó đã học (sẽ bị học vẹt - Overfitting).
# - Chức năng: Cắt bộ dữ liệu gốc thành 2 phần: Tập huấn luyện (Train) và Tập kiểm thử (Test).
# - Input: Dữ liệu biến độc lập (X), dữ liệu biến mục tiêu (y).
# - Xử lý: Xáo trộn ngẫu nhiên dữ liệu, sau đó cắt theo tỷ lệ (vd: 80% học, 20% thi). Có tính năng 'stratify' 
#   giúp giữ nguyên tỷ lệ phân bổ của các nhóm (ví dụ: nhóm rời bỏ chiếm 20% thì ở cả Train và Test đều là 20%).
# - Output: 4 tập dữ liệu: X_train, X_test, y_train, y_test.
from sklearn.model_selection import train_test_split

# 2. StandardScaler
# - Lý thuyết: Data Standardization (Chuẩn hóa dữ liệu / Đưa về phân phối chuẩn). Thuật toán có thể bị 
#   lệch lạc nếu các cột có thang đo quá khác nhau (Ví dụ: Tuổi 20-60, nhưng Lương lại 10.000.000 - 50.000.000).
# - Chức năng: Ép các biến số học về cùng một hệ quy chiếu: Trung bình (Mean) = 0 và Độ lệch chuẩn (Std) = 1.
# - Input: Các cột dữ liệu dạng số (int, float).
# - Xử lý: Áp dụng công thức Z-score = (Giá trị - Giá trị trung bình) / Độ lệch chuẩn.
# - Output: Dữ liệu dạng số thập phân có giá trị dao động chủ yếu quanh dải [-3, 3].
from sklearn.preprocessing import StandardScaler

# 3. OneHotEncoder
# - Lý thuyết: Categorical Encoding (Mã hóa phân loại). Máy tính không hiểu được chữ (vd: "Nam", "Nữ"), 
#   nên cần biến chữ thành số.
# - Chức năng: Bẻ 1 cột chứa N chữ khác nhau thành N cột mới chứa số nhị phân (0 và 1).
# - Input: Các cột dữ liệu dạng phân loại (object, string, category).
# - Xử lý: Nếu cột Giới tính có "Nam" và "Nữ", nó tạo 2 cột mới là "GioiTinh_Nam" và "GioiTinh_Nu". 
#   Nếu là Nam, cột "GioiTinh_Nam" đánh số 1, cột "GioiTinh_Nu" đánh số 0.
# - Output: Ma trận gồm toàn số 0 và 1.
from sklearn.preprocessing import OneHotEncoder

# 4. ColumnTransformer & Pipeline
# - Lý thuyết: Data Workflow / ML Architecture (Kiến trúc luồng dữ liệu).
# - Chức năng ColumnTransformer: Trạm phân luồng. Cho phép áp dụng StandardScaler cho cột số và 
#   OneHotEncoder cho cột chữ CÙNG LÚC trên cùng 1 bảng dữ liệu.
# - Chức năng Pipeline: Dây chuyền sản xuất khép kín. Gắn cục tiền xử lý và thuật toán AI lại với nhau.
# - Xử lý: Dữ liệu thô đi vào đầu Pipeline -> Tự động qua trạm phân luồng xử lý -> Đi thẳng vào thuật toán.
#   Đảm bảo không bị rò rỉ dữ liệu (data leakage) giữa tập Train và Test.
# - Output: Một Object Model hoàn chỉnh, chỉ cần gọi .fit() để học và .predict() để dự đoán.
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 5. RandomForestClassifier
# - Lý thuyết: Ensemble Learning - Bagging (Học tập tập hợp). Thay vì tạo 1 cây quyết định duy nhất (rất dễ bị ảo tưởng), 
#   thuật toán tạo ra hàng trăm, hàng ngàn cây khác nhau (tạo thành Rừng - Forest).
# - Chức năng: Phân loại xem khách hàng thuộc nhóm 0 (Ở lại) hay 1 (Rời bỏ).
# - Input: Tập dữ liệu huấn luyện đã được làm sạch và chuyển hóa hoàn toàn thành số.
# - Xử lý: Mỗi cây con học một phần nhỏ ngẫu nhiên của dữ liệu. Khi dự đoán, hàng trăm cây sẽ cùng 
#   bỏ phiếu (Voting). Lựa chọn có nhiều phiếu nhất sẽ là quyết định cuối cùng.
# - Output: Lớp dự đoán (0 hoặc 1) hoặc xác suất dự đoán (0.0 đến 1.0).
from sklearn.ensemble import RandomForestClassifier

# 6. Các hàm đánh giá (Metrics)
# - classification_report: Bảng điểm chi tiết. Chứa 3 chỉ số vàng: Precision (Đoán Rời bỏ thì trúng mấy người?), 
#   Recall (Trong đám Rời bỏ thật, máy bắt được mấy người?), và F1-Score (Trung bình hài hòa của 2 thằng kia).
# - confusion_matrix: Ma trận nhầm lẫn (Bảng 2x2). Cho biết cụ thể có bao nhiêu ca đoán đúng, bao nhiêu ca đoán sai 
#   (Ví dụ: Khách ở lại nhưng máy vu oan là rời bỏ - False Positive).
# - accuracy_score: Tỷ lệ đoán đúng tổng thể = Tổng số ca đoán đúng / Tổng số hồ sơ kiểm tra.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# KHỐI 1: CÁC THÔNG SỐ CẤU HÌNH (PARAMETERS)
# ==========================================
DATA_PATH = './data/data_bank.csv'
MODEL_DIR = './model'
REPORT_DIR = './reports/bank'

TEST_SIZE = 0.2          
RANDOM_SEED = 42         

N_ESTIMATORS = 200       
MAX_DEPTH = 15           
# ==========================================

pathlib.Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def train_and_visualize():
    print("="*50)
    print("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN")
    print("="*50)

    # ==========================================
    # KHỐI 2: ĐỌC VÀ LÀM SẠCH DỮ LIỆU
    # ==========================================
    print("[1/5] Tải & làm sạch dữ liệu...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {DATA_PATH}.")
        return
    
    cols_to_drop = ['id', 'full_name', 'address', 'created_date', 'last_active_date', 
                    'last_transaction_month', 'cluster_group', 'risk_segment']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    if 'exit' in df.columns:
        df['exit'] = df['exit'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)
        X = df.drop('exit', axis=1)
        y = df['exit']
    else:
        raise ValueError("Lỗi: Thiếu cột mục tiêu 'exit'!")

    # ==========================================
    # KHỐI 3: TIỀN XỬ LÝ VÀ CHIA DỮ LIỆU
    # ==========================================
    print("[2/5] Khởi tạo Pipeline & chia tập train/test...")
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'string', 'bool']).columns.tolist()
    X[cat_cols] = X[cat_cols].astype(str)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # ==========================================
    # KHỐI 4: HUẤN LUYỆN MÔ HÌNH
    # ==========================================
    print("[3/5] Đang huấn luyện Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        max_depth=MAX_DEPTH, 
        class_weight='balanced', 
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    model_pipeline.fit(X_train, y_train)

    # ==========================================
    # KHỐI 5: ĐÁNH GIÁ VÀ XUẤT BIỂU ĐỒ
    # ==========================================
    print("[4/5] Kết xuất biểu đồ báo cáo...")
    y_pred = model_pipeline.predict(X_test)
    raw_feature_names = preprocessor.get_feature_names_out()
    all_feature_names = [name.split('__')[-1] for name in raw_feature_names]

    # 1. Biểu đồ tròn
    plt.figure(figsize=(6, 6))
    y.value_counts().plot.pie(autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=(0, 0.1), shadow=True)
    plt.title('Tỷ lệ: Ở lại vs Rời bỏ', fontsize=14, pad=15)
    plt.ylabel(''); plt.tight_layout(); plt.savefig(f'{REPORT_DIR}/01_class_distribution.png', dpi=300)

    # 2. Heatmap
    plt.figure(figsize=(10, 8))
    corr_df = df[num_cols].copy(); corr_df['exit'] = y
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Ma trận tương quan biến số', fontsize=14, pad=15)
    plt.tight_layout(); plt.savefig(f'{REPORT_DIR}/02_correlation_heatmap.png', dpi=300)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Ma Trận Nhầm Lẫn', fontsize=14, pad=15)
    plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
    plt.tight_layout(); plt.savefig(f'{REPORT_DIR}/03_confusion_matrix.png', dpi=300)

    # 4. Feature Importance
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    feat_imp.sort_values(ascending=True).plot(kind='barh', color='#3498db')
    plt.title('Top 10 Đặc trưng Quan trọng nhất', fontsize=14, pad=15)
    plt.tight_layout(); plt.savefig(f'{REPORT_DIR}/04_feature_importance.png', dpi=300, bbox_inches='tight')

    # ==========================================
    # KHỐI 6: LƯU MÔ HÌNH VÀ BÁO CÁO
    # ==========================================
    print("[5/5] Lưu mô hình & in kết quả...")
    joblib.dump(model_pipeline, f'{MODEL_DIR}/churn_model_bank.pkl')
    
    print("\n" + "-"*40)
    print("BÁO CÁO PHÂN LOẠI (TEST SET):")
    print("-" * 40)
    print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=['Ở lại (0)', 'Rời bỏ (1)']))
    print("="*50)
    print("Hoàn tất! Báo cáo đã lưu vào 'reports/bank'.")

if __name__ == "__main__":
    train_and_visualize()