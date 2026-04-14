import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import joblib
import warnings

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, accuracy_score
)

# Tắt các cảnh báo lặt vặt để terminal sạch sẽ tuyệt đối
warnings.filterwarnings('ignore')

# ==========================================
# KHỞI TẠO ĐƯỜNG DẪN
# ==========================================
data_path = pathlib.Path('./data/data_telco.csv')
model_dir = pathlib.Path('./model')
report_dir = pathlib.Path('./reports/telco')

report_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

def train_and_visualize():
    print("\n" + "★"*60)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH RANDOM FOREST (ULTIMATE)")
    print("★"*60 + "\n")

    # ---------------------------------------------------------
    print("[1/6] 📂 Đang tải và làm sạch dữ liệu...")
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'customerID' in df.columns: 
        df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print("      └── Đã xử lý xong: {} dòng, {} đặc trưng.".format(X.shape[0], X.shape[1]))

    # ---------------------------------------------------------
    print("[2/6] ⚙️  Xây dựng Pipeline tiền xử lý chuẩn công nghiệp...")
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
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
        n_jobs=-1 # Tận dụng đa luồng CPU cho nhanh
    )
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    model_pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    print("[5/6] 📈 Đang kết xuất các biểu đồ phân tích chuyên sâu...")
    
    # A. Confusion Matrix
    y_pred = model_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)', fontsize=14, pad=15)
    plt.ylabel('Thực Tế', fontsize=12)
    plt.xlabel('Dự Đoán', fontsize=12)
    plt.tight_layout()
    plt.savefig(report_dir / 'confusion_matrix.png', dpi=300)

    # B. ROC Curve
    y_probs = model_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Biểu đồ ROC (Receiver Operating Characteristic)', fontsize=14, pad=15)
    plt.xlabel('Tỷ lệ Dương tính giả (FPR)')
    plt.ylabel('Tỷ lệ Dương tính thật (TPR)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(report_dir / 'roc_curve.png', dpi=300)

    # C. Feature Importance
    ohe_cols = model_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_cols)
    all_cols = num_cols + list(ohe_cols)
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    
    feat_imp = pd.Series(importances, index=all_cols).sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 6))
    feat_imp.sort_values(ascending=True).plot(kind='barh', color='#2ecc71')
    plt.title('Top 10 Yếu Tố Quyết Định Rời Bỏ Dịch Vụ', fontsize=14, pad=15)
    plt.xlabel('Mức độ quan trọng (Gini Importance)')
    plt.tight_layout()
    plt.savefig(report_dir / 'feature_importance.png', dpi=300)

    # D. Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model_pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#e74c3c', label='Điểm Train')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 's-', color='#3498db', label='Điểm Cross-Validation')
    plt.title('Đường Cong Học Tập (Learning Curve)', fontsize=14, pad=15)
    plt.xlabel('Số lượng mẫu huấn luyện')
    plt.ylabel('Độ chính xác (Accuracy)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(report_dir / 'learning_curve.png', dpi=300)
    
    print("      └── Đã lưu 4 biểu đồ sắc nét vào thư mục './reports/telco'")

    # ---------------------------------------------------------
    print("[6/6] 💾 Đang đóng gói và lưu trữ mô hình...")
    joblib.dump(model_pipeline, model_dir / 'churn_model_telco.pkl')
    
    # ==========================================
    # IN BÁO CÁO RA TERMINAL
    # ==========================================
    print("\n" + "═"*60)
    print("📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH (ULTIMATE METRICS)")
    print("═"*60)
    
    # In Classification Report tinh chỉnh
    report = classification_report(y_test, y_pred, target_names=['Giữ lại (0)', 'Rời bỏ (1)'])
    print(report)
    
    print("-" * 60)
    print(f"🎯 Độ chính xác tổng thể (Accuracy) : {accuracy_score(y_test, y_pred):.4f}")
    print(f"🏆 Khả năng phân loại diện rộng (AUC): {roc_auc:.4f}")
    print("═"*60 + "\n")
    
    print("🎉 HOÀN TẤT! Đồ án VIP Pro đã sẵn sàng để demo!")
    print("★"*60 + "\n")

if __name__ == "__main__":
    train_and_visualize()