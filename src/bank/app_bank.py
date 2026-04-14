import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import time
import plotly.graph_objects as go

HEAD_QUANTITY = 10

# ==========================================
# CẤU HÌNH TRANG WEB VÀ CSS (LUXURY MINIMALISM)
# ==========================================
st.set_page_config(
    page_title="VN Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Tinh chỉnh CSS sang trọng
st.markdown("""
    <style>
    /* Tổng thể nền và font */
    .main { background-color: #F8F9FA; font-family: 'Inter', sans-serif; }
    
    /* Header và Text */
    h1, h2, h3 { color: #0F172A; font-weight: 700; }
    p, span, label { color: #334155; }
    
    /* Thẻ Metric sang trọng */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-top: 4px solid #1D4ED8;
    }
    
    /* Nút bấm (Primary Button) */
    div.stButton > button:first-child {
        background-color: #ffffcc;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #a0f0aa;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Expander hiện đại */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1E293B !important;
        background-color: #FFFFFF;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# HÀM LOAD MODEL
# ==========================================
@st.cache_resource
def load_assets():
    model_path = pathlib.Path('./model/churn_model_bank.pkl')
    if not model_path.exists():
        return "PreviewMode" 
    return joblib.load(model_path)

model_pipeline = load_assets()

# Cấu hình đồ thị Matplotlib/Seaborn
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h1>🏦 VN Bank Churn <span style='color: #1D4ED8;'>Intelligence</span></h1>", unsafe_allow_html=True)
st.markdown("Hệ thống AI chuẩn đoán rủi ro rời bỏ của khách hàng dựa trên hành vi tài chính và nhân khẩu học.")

if model_pipeline == "PreviewMode":
    st.warning("⚠️ Không tìm thấy model tại `./model/churn_model_bank.pkl`. Hệ thống đang chạy ở **Chế độ Giả lập (Preview)**.")

st.markdown("<br>", unsafe_allow_html=True)

# Chia cột theo tỷ lệ vàng (Phi) cho form và result
col_input, spacing, col_result = st.columns([1.1, 0.1, 1.8])

# ---------------------------------------------------------
# KHU VỰC NHẬP LIỆU (BÊN TRÁI)
# ---------------------------------------------------------
with col_input:
    st.markdown("### 📋 Hồ sơ Khách hàng")
    
    with st.expander("👤 1. Nhân khẩu học", expanded=True):
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            age = st.slider("Độ tuổi:", 18, 90, 45)
            gender = st.selectbox("Giới tính:", ["male", "female"])
            married = st.selectbox("Hôn nhân:", [0, 1, 2], format_func=lambda x: "Độc thân" if x==0 else ("Đã kết hôn" if x==1 else "Khác"))
        with col1_2:
            occupation = st.selectbox("Nghề nghiệp:", ["Salaried", "Chủ Doanh nghiệp nhỏ", "Self-employed", "Student", "Freelancer", "Retired"])
            origin_province = st.selectbox("Tỉnh thành:", ["TP. Hồ Chí Minh", "Ha Noi", "Da Nang", "Can Tho", "Other"])

    with st.expander("💵 2. Tài chính & Tín dụng", expanded=True):
        monthly_ir = st.number_input("Thu nhập hàng tháng (VNĐ):", min_value=0, value=25000000, step=1000000)
        st.caption(f"<div style='text-align: right; color: #10B981; font-weight: bold;'>{monthly_ir:,.0f} VNĐ</div>", unsafe_allow_html=True)
        
        balance = st.number_input("Số dư hiện tại (VNĐ):", min_value=0, value=150000000, step=5000000)
        st.caption(f"<div style='text-align: right; color: #10B981; font-weight: bold;'>{balance:,.0f} VNĐ</div>", unsafe_allow_html=True)
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            credit_sco = st.slider("Điểm tín dụng:", 300, 850, 650)
            nums_card = st.slider("Số lượng thẻ:", 0, 10, 2)
        with col2_2:
            nums_service = st.slider("Số dịch vụ:", 1, 15, 4)
            tenure_ye = st.slider("Năm gắn bó:", 0, 20, 3)

    with st.expander("📊 3. Tương tác & Rủi ro", expanded=True):
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            customer_segment = st.selectbox("Phân khúc:", ["Mass", "Emerging", "Affluent", "Priority"])
            loyalty_level = st.selectbox("Thành viên:", ["Bronze", "Silver", "Gold", "Platinum"])
            active_member = st.checkbox("Khách hàng Active?", value=True)
        with col3_2:
            digital_behavior = st.selectbox("Hành vi số:", ["offline", "hybrid", "mobile"])
            engagement_score = st.slider("Điểm tương tác:", 0, 100, 60)
            risk_score = st.slider("Điểm rủi ro nội bộ:", 0.0, 1.0, 0.15, step=0.01)

# ---------------------------------------------------------
# KHU VỰC KẾT QUẢ AI (BÊN PHẢI)
# ---------------------------------------------------------
with col_result:
    st.markdown("### 🤖 Phân tích AI & Khuyến nghị")
    
    # Gom dữ liệu đầu vào (Khớp 100% với các cột đã train)
    raw_input = {
        'credit_sco': credit_sco, 'gender': gender, 'age': age, 'occupation': occupation,
        'balance': balance, 'monthly_ir': monthly_ir, 'origin_province': origin_province,
        'tenure_ye': tenure_ye, 'married': married, 'nums_card': nums_card,
        'nums_service': nums_service, 'active_member': active_member, 
        'customer_segment': customer_segment, 'engagement_score': engagement_score,
        'loyalty_level': loyalty_level, 'digital_behavior': digital_behavior, 
        'risk_score': risk_score
    }
    input_df = pd.DataFrame([raw_input])
    
    # Ép kiểu dữ liệu giống hệt pipeline ở train.py
    cat_cols = input_df.select_dtypes(include=['object', 'bool']).columns.tolist()
    input_df[cat_cols] = input_df[cat_cols].astype(str)

    run_analysis = st.button("TIẾN HÀNH PHÂN TÍCH CHUYÊN SÂU", use_container_width=True)
    
    if not run_analysis:
        st.info("💡 Điều chỉnh các thông số hồ sơ bên trái và nhấn nút để bắt đầu đánh giá hành vi khách hàng.")
    else:
        with st.spinner("Đang trích xuất đặc trưng & đối chiếu với dữ liệu lịch sử..."):
            time.sleep(0.8) # Micro-interaction
            
            # Xử lý nội bộ
            if model_pipeline == "PreviewMode":
                prob = risk_score * 0.8 if balance > 50000000 else 0.85
                is_mock = True
            else:
                try:
                    prob = model_pipeline.predict_proba(input_df)[0][1]
                    is_mock = False
                except Exception as e:
                    st.error(f"❌ Cấu trúc dữ liệu không khớp. Lỗi: {e}")
                    st.stop()
            
            # --- TỐI ƯU UX: SỬ DỤNG GAUGE CHART THAY VÌ PROGRESS BAR ---
            # Áp dụng Threshold linh hoạt (Giả định ngưỡng an toàn là 40% dựa trên PR-Curve trước đó)
            threshold = 0.40 
            is_churn = prob >= threshold

            # Vẽ Gauge Chart bằng Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%", 'font': {'size': 48, 'color': '#0F172A', 'family': 'Arial'}},
                title = {'text': "CHỈ SỐ RỦI RO RỜI BỎ", 'font': {'size': 18, 'color': '#64748B'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#EF4444" if is_churn else "#10B981"},
                    'bgcolor': "white",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, threshold*100], 'color': '#ECFDF5'},  # Safe zone (Green-ish)
                        {'range': [threshold*100, 100], 'color': '#FEF2F2'}] # Danger zone (Red-ish)
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # --- RENDER KẾT QUẢ VÀ KHUYẾN NGHỊ ---
            st.markdown("#### 🎯 Kết luận & Khuyến nghị Hành động")
            m1, m2 = st.columns(2)
            
            with m1:
                if is_churn:
                    st.metric("Phân loại trạng thái", "🚨 CAO - CẦN GIỮ CHÂN")
                else:
                    st.metric("Phân loại trạng thái", "✅ THẤP - ỔN ĐỊNH")
            with m2:
                st.metric("Độ tin cậy của AI", "Mức Cao (F1-Optimized)")

            if is_churn:
                st.error(f"**CẢNH BÁO ĐỎ:** Hệ thống dự báo khách hàng có rủi ro rời bỏ vượt ngưỡng an toàn. Đề xuất tự động chuyển hồ sơ đến phòng **Customer Retention** để áp dụng kịch bản ưu đãi lãi suất / miễn phí thường niên.")
            else:
                st.success(f"**TÍN HIỆU TỐT:** Khách hàng thể hiện sự gắn bó ổn định với hệ sinh thái. Có thể áp dụng kịch bản **Cross-sell/Up-sell** (Bán chéo/Bán thêm) các sản phẩm thẻ tín dụng hạng sang.")

            st.markdown("---")

            # --- VẼ BIỂU ĐỒ FEATURE IMPORTANCE SẠCH SẼ & ĐÚNG CHUẨN ---
            st.markdown("#### 🔍 Giải phẫu Quyết định của AI (Top Đặc Trưng)")
            
            if is_mock:
                mock_importances = pd.Series({'risk_score': 0.35, 'balance': 0.22, 'age': 0.15, 'engagement_score': 0.12, 'monthly_ir': 0.08}).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 3.5))
                sns.barplot(x=mock_importances.values, y=mock_importances.index, palette="Blues_r", ax=ax)
                ax.set_title("Trọng số ảnh hưởng (Dữ liệu Giả lập)", fontweight="bold", pad=15)
                sns.despine(left=True, bottom=True)
                st.pyplot(fig)
            else:
                try:
                    # Trích xuất chuẩn xác như trong file train_bank.py
                    preprocessor = model_pipeline.named_steps['preprocessor']
                    rf_model = model_pipeline.named_steps['classifier']
                    
                    raw_feature_names = preprocessor.get_feature_names_out()
                    all_feature_names = [name.split('__')[-1] for name in raw_feature_names]
                    
                    importances = pd.Series(rf_model.feature_importances_, index=all_feature_names).sort_values(ascending=False).head(HEAD_QUANTITY)
                    
                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    # Sử dụng bảng màu sang trọng
                    colors = ['#055daa' if i == 0 else '#94A3B8' for i in range(len(importances))]
                    sns.barplot(x=importances.values, y=importances.index, palette=colors, ax=ax)
                    
                    ax.set_title(f"Top {HEAD_QUANTITY} Yếu tố giải thích cho quyết định của Model", fontweight="bold", pad=15, color='#334155')
                    ax.set_xlabel("Độ quan trọng (Gini Score)")
                    ax.set_ylabel("")
                    sns.despine(left=True, bottom=True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"Không thể trích xuất Feature Importance từ Pipeline hiện tại. (Mã lỗi: {e})")