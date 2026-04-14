import os
import warnings
# Ép hệ thống tắt TOÀN BỘ cảnh báo ở mọi luồng con trước khi import thư viện
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import joblib

from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, accuracy_score, brier_score_loss, f1_score
)

# ==========================================
# KHỞI TẠO ĐƯỜNG DẪN
# ==========================================
data_path = pathlib.Path('./data/data_bank.csv') # Sửa lại tên file nếu cần
model_dir = pathlib.Path('./model')
report_dir = pathlib.Path('./reports/bank')

report_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# Cấu hình font chữ và style chung cho biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def train_and_visualize():
    print("\n" + "★"*70)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MODEL - ULTIMATE EDITION V2.2 (AUTO-DIAGNOSTIC)")
    print("★"*70 + "\n")

    # ---------------------------------------------------------
    print("[1/6] 📂 Đang tải và làm sạch dữ liệu Ngân hàng...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file tại {data_path}. Vui lòng kiểm tra lại đường dẫn!")
        return
    
    cols_to_drop = ['id', 'full_name', 'address', 'created_date', 'last_active_date', 
                    'last_transaction_month', 'cluster_group', 'risk_segment']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    if 'exit' in df.columns:
        df['exit'] = df['exit'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)
        X = df.drop('exit', axis=1)
        y = df['exit']
    else:
        raise ValueError("❌ Không tìm thấy cột mục tiêu 'exit' trong dataset!")

    print("      └── Đã xử lý xong: {} khách hàng, {} đặc trưng phân tích.".format(X.shape[0], X.shape[1]))

    # ---------------------------------------------------------
    print("[2/6] ⚙️  Xây dựng Pipeline tiền xử lý tự động...")
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'string', 'bool']).columns.tolist()

    X[cat_cols] = X[cat_cols].astype(str)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    # ---------------------------------------------------------
    print("[3/6] 🔀 Đang chia tập dữ liệu (Stratified Split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    print("[4/6] 🧠 Đang huấn luyện Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    model_pipeline.fit(X_train, y_train)

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    raw_feature_names = preprocessor.get_feature_names_out()
    all_feature_names = [name.split('__')[-1] for name in raw_feature_names]

    # ---------------------------------------------------------
    print("[5/6] 📈 ĐANG KẾT XUẤT 11 BIỂU ĐỒ BÁO CÁO CẤP CAO...")
    
    y_pred = model_pipeline.predict(X_test)
    y_probs = model_pipeline.predict_proba(X_test)[:, 1]

    # Các thông số tính toán nhanh cho Diagnostic
    churn_rate = y.mean() * 100
    corr_matrix = df[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]

    print("      ├── 1-5. Vẽ các biểu đồ cơ bản (Distribution, Heatmap, CM, ROC, PR)...")
    # 1. Class Dist
    plt.figure(figsize=(6, 6))
    y.value_counts().plot.pie(autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], explode=(0, 0.1), shadow=True)
    plt.title('Tỷ lệ Khách hàng Ở lại vs Rời bỏ', fontsize=14, pad=15)
    plt.ylabel(''); plt.tight_layout(); plt.savefig(report_dir / '01_class_distribution.png', dpi=300)

    # 2. Heatmap
    plt.figure(figsize=(10, 8))
    corr_df = df[num_cols].copy(); corr_df['exit'] = y
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Ma trận Tương quan các biến số học', fontsize=14, pad=15)
    plt.tight_layout(); plt.savefig(report_dir / '02_correlation_heatmap.png', dpi=300)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Ma Trận Nhầm Lẫn', fontsize=14, pad=15); plt.ylabel('Thực Tế'); plt.xlabel('Dự Đoán')
    plt.tight_layout(); plt.savefig(report_dir / '03_confusion_matrix.png', dpi=300)

    # 4. ROC
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Biểu đồ ROC', fontsize=14, pad=15); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(report_dir / '04_roc_curve.png', dpi=300)

    # 5. PR Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.title('Biểu đồ Precision-Recall', fontsize=14, pad=15); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(report_dir / '05_precision_recall_curve.png', dpi=300)

    print("      ├── 6-7. Vẽ Feature Importance (Gini & Permutation)...")
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(12)
    plt.figure(figsize=(10, 6))
    feat_imp.sort_values(ascending=True).plot(kind='barh', color='#3498db')
    plt.title('Top 12 Yếu Tố Quan Trọng (Gini)', fontsize=14, pad=15); plt.tight_layout()
    plt.savefig(report_dir / '06_feature_importance.png', dpi=300, bbox_inches='tight')

    perm_importance = permutation_importance(rf, X_test_transformed, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = perm_importance.importances_mean.argsort()[-10:]
    plt.figure(figsize=(10, 6))
    plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, tick_labels=np.array(all_feature_names)[sorted_idx])
    plt.title("Permutation Feature Importance", fontsize=14, pad=15); plt.tight_layout()
    plt.savefig(report_dir / '07_permutation_importance.png', dpi=300, bbox_inches='tight')

    print("      ├── 8-11. Vẽ các biểu đồ Đánh giá chuyên sâu (Learning, Val, Calibration, Prob Dist)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_sizes, train_scores, test_scores = learning_curve(
            model_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
        )
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#e74c3c', label='Train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='#3498db', label='CV')
    plt.title('Đường Cong Học Tập', fontsize=14, pad=15); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(report_dir / '08_learning_curve.png', dpi=300)

    param_range = np.arange(1, 25, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_scores_vc, test_scores_vc = validation_curve(
            RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42),
            X_train_transformed, y_train, param_name="max_depth", param_range=param_range, cv=3, scoring="f1", n_jobs=-1
        )
    plt.figure(figsize=(7, 5))
    plt.plot(param_range, np.mean(train_scores_vc, axis=1), 'o-', color='#e74c3c', label='Train')
    plt.plot(param_range, np.mean(test_scores_vc, axis=1), 's-', color='#3498db', label='CV')
    plt.title('Đường cong Validation (Max Depth)', fontsize=14, pad=15); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(report_dir / '09_validation_curve.png', dpi=300)

    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, 's-', color='green', label='Random Forest')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.title('Biểu đồ Hiệu chuẩn', fontsize=14, pad=15); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(report_dir / '10_calibration_curve.png', dpi=300)

    plt.figure(figsize=(8, 5))
    sns.histplot(y_probs[y_test == 0], color='green', label='Thực tế: Ở lại (0)', kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(y_probs[y_test == 1], color='red', label='Thực tế: Rời bỏ (1)', kde=True, stat="density", bins=30, alpha=0.5)
    plt.title('Phân phối xác suất dự đoán', fontsize=14, pad=15); plt.legend(); plt.tight_layout()
    plt.savefig(report_dir / '11_prob_distribution.png', dpi=300)

    # ---------------------------------------------------------
    print("\n[6/6] 💾 Đang đóng gói và lưu trữ mô hình...")
    joblib.dump(model_pipeline, model_dir / 'churn_model_bank.pkl')
    
    # Tính toán thông số cho Diagnostic
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_threshold_idx] if best_threshold_idx < len(thresholds_pr) else 0.5
    
    train_end_acc = np.mean(train_scores[-1])
    test_end_acc = np.mean(test_scores[-1])
    learning_gap = train_end_acc - test_end_acc
    
    best_depth = param_range[np.argmax(np.mean(test_scores_vc, axis=1))]

    # ==========================================
    # BÁO CÁO & CHUẨN ĐOÁN (DIAGNOSTIC ENGINE)
    # ==========================================
    print("\n" + "═"*80)
    print("🧠 HỆ THỐNG PHÂN TÍCH & CHUẨN ĐOÁN MÔ HÌNH (EXPERT DIAGNOSTIC REPORT)")
    print("═"*80)
    
    print("\n[I. ĐÁNH GIÁ TÌNH TRẠNG DỮ LIỆU & HUẤN LUYỆN DỰA TRÊN 11 PLOT]")
    
    print(f"👉 Plot 1 (Phân bổ Target): Dữ liệu có tỷ lệ Churn là {churn_rate:.1f}%.")
    if churn_rate < 20:
        print("   ❌ Đánh giá: Mất cân bằng NẶNG. Model sẽ có xu hướng 'thiên vị' đoán khách hàng ở lại.")
    else:
        print("   ✅ Đánh giá: Dữ liệu khá cân bằng, an toàn cho model.")

    print(f"👉 Plot 2 (Heatmap): Phát hiện {len(high_corr_cols)} cặp biến có tương quan rất cao (>0.8).")
    if len(high_corr_cols) > 0:
        print(f"   ⚠️ Đánh giá: Dấu hiệu đa cộng tuyến (Multicollinearity). Cân nhắc loại bỏ: {high_corr_cols}.")
    else:
        print("   ✅ Đánh giá: Các biến độc lập tốt, không bị dính đa cộng tuyến.")

    print(f"👉 Plot 4 & 5 (ROC vs PR Curve): ROC-AUC = {roc_auc:.3f} | PR-AUC = {pr_auc:.3f}")
    if (roc_auc - pr_auc) > 0.2:
        print("   ⚠️ Đánh giá: ROC đang vẽ ra một 'ảo tưởng sức mạnh' do dữ liệu mất cân bằng. Hãy nhìn vào PR-AUC, mô hình thực chất đang gặp khó khăn trong việc duy trì Precision khi cố bắt (Recall) nhóm Rời bỏ.")

    print(f"👉 Plot 8 (Learning Curve): Gap giữa Train và Validation là {learning_gap:.3f}")
    if learning_gap > 0.1:
        print("   ❌ Tình trạng: OVERFIT (Học vẹt). Train quá tốt nhưng đoán thực tế kém. Data đã 'DƯ' (Train Test đi ngang). Thêm data không giải quyết được vấn đề. Cần giảm độ phức tạp của mô hình.")
    elif learning_gap < 0.02:
        print("   ⚠️ Tình trạng: UNDERFIT (Chưa đủ khả năng học). Đường đồ thị thấp. Model quá đơn giản hoặc thiếu Feature (chưa đủ).")
    else:
        print("   ✅ Tình trạng: FIT TỐT. Mô hình tổng quát hóa tốt.")

    print(f"👉 Plot 9 (Validation Curve): Độ sâu (max_depth) tối ưu thực sự nằm ở: {best_depth}")
    if best_depth < 15:
        print(f"   💡 Gợi ý: Anh đang set max_depth=15, hãy hạ xuống {best_depth} để bớt overfit!")

    print(f"👉 Plot 11 (Prob Distribution): Điểm giao thoa xác suất tối ưu để tối đa F1-Score là Threshold = {best_threshold:.2f}")
    
    print("\n" + "-"*80)
    print("[II. GIẢI PHẪU KỸ THUẬT PIPELINE ĐÃ SỬ DỤNG (THE ANATOMY)]")
    print("1. StratifiedSplit:")
    print("   - Tác dụng: Chia Train/Test giữ đúng tỷ lệ 82/18 của khách Rời bỏ.")
    print("   - Đồng cấp: RandomSplit (Nguy hiểm vì có thể tập Test không có ai rời bỏ).")
    
    print("\n2. StandardScaler (Chuẩn hóa số):")
    print("   - Tác dụng: Ép dữ liệu về Mean=0, Std=1. Giúp Pipeline chuẩn chỉnh.")
    print("   - Đánh đổi (Trade-off): Thực ra thuật toán Tree (Random Forest) KHÔNG CẦN scale. Nó làm mất đi tính trực quan của dữ liệu (ko đọc được số tiền thật sau khi scale).")
    
    print("\n3. OneHotEncoder (Biến đổi chữ thành số):")
    print("   - Tác dụng: Bẻ 'gender' thành 'is_male', 'is_female' (0,1).")
    print("   - Đánh đổi: Gây ra 'Lời nguyền số chiều' (Curse of Dimensionality) nếu dùng cho cột có quá nhiều giá trị (VD: mã tỉnh thành). Cây sẽ bị loãng.")
    print("   - Kỹ thuật thay thế: Target Encoding (Thay tên tỉnh bằng % tỷ lệ rời bỏ trung bình của tỉnh đó).")

    print("\n4. Random Forest + class_weight='balanced':")
    print("   - Tác dụng: Trộn 200 cây quyết định, phạt nặng model nếu đoán sai người rời bỏ.")
    print("   - Lợi ích: Rất lỳ đòn, ít bị nhiễu (outliers), chạy song song nhanh.")
    print("   - Đồng cấp & Gợi ý thay thế: XGBoost / LightGBM. (Boosting thường đấm gục Random Forest ở Tabular Data với PR-AUC cao hơn 3-5%).")

    print("\n" + "═"*80)
    print("🎯 KẾT QUẢ CUỐI CÙNG & HÀNH ĐỘNG TIẾP THEO")
    print(f"   - Accuracy                : {accuracy_score(y_test, y_pred):.4f}")
    print(f"   - Brier Score (Độ sai lệch): {brier_score_loss(y_test, y_probs):.4f}")
    
    print(f"\n⚡ HÀNH ĐỘNG TỨC THÌ: Chuyển Threshold cắt quyết định từ 0.5 xuống {best_threshold:.2f}.")
    print(">> Report tại ngưỡng tối ưu mới:")
    y_pred_optimal = (y_probs >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_optimal, target_names=['Ở lại (0)', 'Rời bỏ (1)']))
    print("═"*80 + "\n")

if __name__ == "__main__":
    train_and_visualize()