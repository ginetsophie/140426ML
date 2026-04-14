import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Churn Predictor Telco",
    page_icon="📊",
    layout="wide"
)

# Custom CSS để giao diện trông "doanh nghiệp" hơn
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# HÀM LOAD MODEL
# ==========================================
@st.cache_resource
def load_assets():
    model_path = pathlib.Path('./model/churn_model_telco.pkl')
    if not model_path.exists():
        st.error("❌ Không tìm thấy file model! Anh hãy chạy file train.py trước nhé.")
        return None
    return joblib.load(model_path)

model_pipeline = load_assets()

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🚀 Churn Predictor Telco")
st.markdown("---")

if model_pipeline:
    # Chia layout: Cột trái nhập liệu - Cột phải Dashboard kết quả
    col_input, col_result = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("👤 Thông tin khách hàng")
        with st.expander("Dịch vụ & Hợp đồng", expanded=True):
            tenure = st.slider("Thời gian gắn bó (tháng):", 1, 72, 12)
            contract = st.selectbox("Loại hợp đồng:", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.number_input("Cước phí hàng tháng ($):", 10.0, 150.0, 70.0)
            total_charges = st.number_input("Tổng cước tích lũy ($):", 10.0, 9000.0, tenure * monthly_charges)

        with st.expander("Chi tiết dịch vụ Internet"):
            is_internet = st.selectbox("Dịch vụ Internet:", ["DSL", "Fiber optic", "No"])
            tech_support = st.selectbox("Hỗ trợ kỹ thuật:", ["Yes", "No", "No internet service"])
            online_sec = st.selectbox("Bảo mật online:", ["Yes", "No", "No internet service"])

        with st.expander("Thông tin cá nhân"):
            gender = st.selectbox("Giới tính:", ["Male", "Female"])
            senior = st.radio("Khách hàng cao tuổi?", ["Yes", "No"])
            partner = st.radio("Có người thân?", ["Yes", "No"])

    with col_result:
        st.subheader("🤖 Phân tích AI & Dự báo")
        
        # Nút bấm dự đoán
        if st.button("CHẠY PHÂN TÍCH NGAY 🚀", use_container_width=True):
            # 1. Tạo DataFrame từ input (phải khớp chính xác tên cột gốc)
            # Lưu ý: Các cột không có trong form nhập liệu sẽ được gán giá trị mặc định 'No'
            raw_input = {
                'gender': gender, 'SeniorCitizen': 1 if senior == 'Yes' else 0,
                'Partner': partner, 'Dependents': 'No', 'tenure': tenure,
                'PhoneService': 'Yes', 'MultipleLines': 'No',
                'InternetService': is_internet, 'OnlineSecurity': online_sec,
                'OnlineBackup': 'No', 'DeviceProtection': 'No',
                'TechSupport': tech_support, 'StreamingTV': 'No',
                'StreamingMovies': 'No', 'Contract': contract,
                'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
                'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
            }
            input_df = pd.DataFrame([raw_input])

            # 2. Dự báo bằng Pipeline (Không cần xử lý tay, Pipeline tự lo hết)
            prediction = model_pipeline.predict(input_df)[0]
            prob = model_pipeline.predict_proba(input_df)[0][1]

            # 3. Hiển thị kết quả bằng Metrics & Alerts
            st.write("---")
            m1, m2 = st.columns(2)
            
            with m1:
                status = "🚨 NGUY CƠ RỜI BỎ" if prediction == 1 else "✅ AN TOÀN"
                st.metric("Trạng thái", status)
            with m2:
                st.metric("Xác suất rời bỏ (Probability)", f"{prob*100:.2f}%")

            if prediction == 1:
                st.error(f"Khách hàng này có **{prob*100:.1f}%** khả năng sẽ hủy dịch vụ. Cần có chính sách ưu đãi ngay!")
            else:
                st.success(f"Khách hàng đang hài lòng. Khả năng ở lại là **{(1-prob)*100:.1f}%**.")

            # 4. Giải thích yếu tố ảnh hưởng (Feature Importance cho Random Forest)
            st.markdown("### 🔍 Các yếu tố ảnh hưởng nhất")
            rf_model = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']
            
            # Lấy tên đặc trưng sau khi mã hóa
            cat_cols = input_df.select_dtypes(include=['object']).columns.tolist()
            ohe_features = preprocessor.transformers_[1][1].get_feature_names_out(cat_cols)
            all_feature_names = input_df.select_dtypes(include=['number']).columns.tolist() + list(ohe_features)
            
            importances = pd.Series(rf_model.feature_importances_, index=all_feature_names).sort_values(ascending=False).head(8)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
            ax.set_title("Tại sao khách hàng này lại rời đi/ở lại?")
            st.pyplot(fig)
else:
    st.info("💡 Hệ thống đang chờ model được huấn luyện...")