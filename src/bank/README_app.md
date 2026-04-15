```python
import os
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import time
import plotly.graph_objects as go

# Tắt cảnh báo để server chạy mượt, không in rác ra terminal
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

# =================================================================================================
# GIẢI PHẪU CHI TIẾT THƯ VIỆN STREAMLIT (DÀNH CHO NEWBIE TẠO WEB BẰNG PYTHON)
# =================================================================================================

# ĐẶC ĐIỂM CỐT LÕI CỦA STREAMLIT:
# Khác với các web framework phức tạp (Django, React), Streamlit chạy code từ trên xuống dưới.
# Mỗi khi người dùng bấm nút hoặc kéo slider, nó sẽ CHẠY LẠI TỪ ĐẦU file code. 
import streamlit as st

# 1. st.set_page_config()
# - Chức năng: Cấu hình khung sườn của trang web. Phải được gọi đầu tiên.
# - Input: Tiêu đề tab (page_title), chế độ dàn trang (layout="wide" để web tràn viền).
# - Output: Giao diện cơ bản của trình duyệt.

# 2. st.markdown()
# - Chức năng: Cho phép viết text có định dạng, hoặc nhúng thẳng mã HTML/CSS vào web.
# - Input: Chuỗi string chứa HTML/CSS, kèm tham số unsafe_allow_html=True để ép hệ thống hiểu đây là code giao diện.

# 3. @st.cache_resource
# - Lý thuyết: Vì Streamlit chạy lại toàn bộ code mỗi khi thao tác, nếu lần nào cũng bắt nó đọc 
#   file model AI (rất nặng) thì web sẽ bị đơ.
# - Chức năng: Đây là một "Decorator" (Đánh dấu). Nó bảo hệ thống: "Cái hàm load model này chỉ chạy 1 lần 
#   duy nhất lúc bật web thôi, sau đó lưu thẳng kết quả vào RAM (Cache) nhé".

# 4. st.columns()
# - Chức năng: Cắt màn hình web thành nhiều cột dọc để chia bố cục.
# - Input: Một mảng tỷ lệ, ví dụ [1.1, 0.1, 1.8] nghĩa là chia màn hình làm 3 cột, cột 1 vừa vừa, 
#   cột 2 xíu xiu làm khoảng trắng, cột 3 to nhất.
# - Output: Các đối tượng cột. Ta dùng từ khóa "with" để nhét nội dung vào từng cột.

# 5. st.expander()
# - Chức năng: Tạo một cái hộp có thể bấm thu gọn/mở rộng. Giúp giao diện bớt rối mắt khi có quá nhiều ô nhập liệu.

# 6. Các hàm nhập liệu (st.slider, st.selectbox, st.number_input, st.checkbox)
# - Chức năng: Tạo các widget để người dùng nhập thông tin.
# - Input: Tên nhãn, giá trị nhỏ nhất, lớn nhất, và giá trị mặc định.
# - Output: Nó sẽ trả về đúng cái giá trị mà người dùng đang chọn trên web để truyền vào biến.

# 7. st.button()
# - Chức năng: Nút bấm thực thi lệnh.
# - Output: Trả về giá trị False (nếu chưa bấm) và True (nếu đã bấm).

# 8. st.metric()
# - Chức năng: Hiển thị các con số quan trọng (KPI) dưới dạng thẻ thông tin đẹp mắt, sang trọng.

# 9. st.plotly_chart() và st.pyplot()
# - Chức năng: Đưa biểu đồ từ các thư viện vẽ đồ thị (Plotly - biểu đồ động, Matplotlib - biểu đồ tĩnh) lên web.
# =================================================================================================

HEAD_QUANTITY = 10

# ==========================================
# CẤU HÌNH TRANG WEB VÀ CSS
# ==========================================
st.set_page_config(
    page_title="VN Bank Churn System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Tinh chỉnh CSS theo phong cách dashboard nội bộ, chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #F8F9FA; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #0F172A; font-weight: 700; }
    p, span, label { color: #334155; }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF; padding: 20px; border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); border-left: 4px solid #1D4ED8;
    }
    div.stButton > button:first-child {
        background-color: #f3aaaa; color: white; border-radius: 6px; padding: 10px 24px; font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background-color: #aa9090; color: white;
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

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h1>HỆ THỐNG PHÂN TÍCH RỦI RO RỜI BỎ <span style='color: #1D4ED8;'>(CHURN PREDICTION)</span></h1>", unsafe_allow_html=True)
st.markdown("Dashboard nội bộ đánh giá xác suất ngừng sử dụng dịch vụ của khách hàng dựa trên mô hình Machine Learning.")

if model_pipeline == "PreviewMode":
    st.warning("Cảnh báo hệ thống: Không tìm thấy file model. Đang khởi chạy chế độ mô phỏng (Mock Data).")

st.markdown("<br>", unsafe_allow_html=True)

col_input, spacing, col_result = st.columns([1.1, 0.1, 1.8])

# ---------------------------------------------------------
# KHU VỰC NHẬP LIỆU
# ---------------------------------------------------------
with col_input:
    st.markdown("### Thông tin Khách hàng")
    
    with st.expander("1. Nhân khẩu học", expanded=True):
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            age = st.slider("Độ tuổi:", 18, 90, 45)
            gender = st.selectbox("Giới tính:", ["male", "female"])
            married = st.selectbox("Hôn nhân:", [0, 1, 2], format_func=lambda x: "Độc thân" if x==0 else ("Đã kết hôn" if x==1 else "Khác"))
        with col1_2:
            occupation = st.selectbox("Nghề nghiệp:", ["Salaried", "Chủ Doanh nghiệp nhỏ", "Self-employed", "Student", "Freelancer", "Retired"])
            origin_province = st.selectbox("Tỉnh thành:", ["TP. Hồ Chí Minh", "Ha Noi", "Da Nang", "Can Tho", "Other"])

    with st.expander("2. Tài chính & Tín dụng", expanded=True):
        monthly_ir = st.number_input("Thu nhập hằng tháng (VNĐ):", min_value=0, value=25000000, step=1000000)
        balance = st.number_input("Số dư hiện tại (VNĐ):", min_value=0, value=150000000, step=5000000)
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            credit_sco = st.slider("Điểm tín dụng:", 300, 850, 650)
            nums_card = st.slider("Số lượng thẻ:", 0, 10, 2)
        with col2_2:
            nums_service = st.slider("Số dịch vụ:", 1, 15, 4)
            tenure_ye = st.slider("Năm gắn bó:", 0, 20, 3)

    with st.expander("3. Tương tác & Đánh giá rủi ro", expanded=True):
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            customer_segment = st.selectbox("Phân khúc:", ["Mass", "Emerging", "Affluent", "Priority"])
            loyalty_level = st.selectbox("Hạng thành viên:", ["Bronze", "Silver", "Gold", "Platinum"])
            active_member = st.checkbox("Đang hoạt động (Active)?", value=True)
        with col3_2:
            digital_behavior = st.selectbox("Hành vi số:", ["offline", "hybrid", "mobile"])
            engagement_score = st.slider("Điểm tương tác:", 0, 100, 60)
            risk_score = st.slider("Điểm rủi ro nội bộ:", 0.0, 1.0, 0.15, step=0.01)

# ---------------------------------------------------------
# KHU VỰC KẾT QUẢ PHÂN TÍCH
# ---------------------------------------------------------
with col_result:
    st.markdown("### Kết quả Phân tích từ Hệ thống")
    
    # Đóng gói dữ liệu đầu vào thành DataFrame
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
    
    cat_cols = input_df.select_dtypes(include=['object', 'bool']).columns.tolist()
    input_df[cat_cols] = input_df[cat_cols].astype(str)

    run_analysis = st.button("XỬ LÝ DỮ LIỆU & DỰ BÁO", use_container_width=True)
    
    if not run_analysis:
        st.info("Hệ thống đang chờ lệnh. Vui lòng kiểm tra thông tin bên trái và nhấn nút xử lý.")
    else:
        with st.spinner("Đang chạy luồng dữ liệu qua hệ thống Pipeline ML..."):
            time.sleep(0.5) 
            
            if model_pipeline == "PreviewMode":
                prob = risk_score * 0.8 if balance > 50000000 else 0.85
                is_mock = True
            else:
                try:
                    prob = model_pipeline.predict_proba(input_df)[0][1]
                    is_mock = False
                except Exception as e:
                    st.error(f"Lỗi cấu trúc dữ liệu đầu vào. Chi tiết: {e}")
                    st.stop()
            
            # Khởi tạo ngưỡng quyết định 
            threshold = 0.40 
            is_churn = prob >= threshold

            # Vẽ Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%", 'font': {'size': 40, 'color': '#0F172A'}},
                title = {'text': "XÁC SUẤT RỜI BỎ (CHURN PROBABILITY)", 'font': {'size': 14, 'color': '#64748B'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#EF4444" if is_churn else "#10B981"},
                    'bgcolor': "white",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, threshold*100], 'color': '#ECFDF5'},
                        {'range': [threshold*100, 100], 'color': '#FEF2F2'}]
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Xuất kết luận
            st.markdown("#### Trạng thái & Khuyến nghị")
            m1, m2 = st.columns(2)
            
            with m1:
                if is_churn:
                    st.metric("Phân loại trạng thái", "CẢNH BÁO RỦI RO")
                else:
                    st.metric("Phân loại trạng thái", "ỔN ĐỊNH")
            with m2:
                st.metric("Độ tin cậy hệ thống", "Mức Cao")

            if is_churn:
                st.error("Yêu cầu chú ý: Khách hàng có xu hướng ngừng sử dụng dịch vụ. Hệ thống đề xuất chuyển hồ sơ sang bộ phận Chăm sóc Khách hàng (Retention Team) để áp dụng chính sách ưu đãi.")
            else:
                st.success("Thông tin: Hồ sơ khách hàng ổn định. Đề xuất giữ nguyên kịch bản chăm sóc hiện tại hoặc áp dụng chiến lược bán chéo (Cross-sell) dịch vụ.")

            st.markdown("---")

            # Vẽ Feature Importance
            st.markdown("#### Trọng số ảnh hưởng tới quyết định (Feature Importance)")
            
            if is_mock:
                mock_importances = pd.Series({'risk_score': 0.35, 'balance': 0.22, 'age': 0.15, 'engagement_score': 0.12, 'monthly_ir': 0.08}).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.barplot(x=mock_importances.values, y=mock_importances.index, palette="Blues_r", ax=ax)
                sns.despine(left=True, bottom=True)
                st.pyplot(fig)
            else:
                try:
                    preprocessor = model_pipeline.named_steps['preprocessor']
                    rf_model = model_pipeline.named_steps['classifier']
                    
                    raw_feature_names = preprocessor.get_feature_names_out()
                    all_feature_names = [name.split('__')[-1] for name in raw_feature_names]
                    
                    importances = pd.Series(rf_model.feature_importances_, index=all_feature_names).sort_values(ascending=False).head(HEAD_QUANTITY)
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    colors = ['#1D4ED8' if i == 0 else '#94A3B8' for i in range(len(importances))]
                    sns.barplot(x=importances.values, y=importances.index, palette=colors, ax=ax)
                    ax.set_xlabel("Mức độ quan trọng")
                    sns.despine(left=True, bottom=True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.warning(f"Không thể kết xuất dữ liệu phân tích thành phần. Mã lỗi nội bộ: {e}")

```