"""
Streamlit Web Application for Kinetic Modeling Analysis
Универсальная программная платформа для анализа кинетического моделирования
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

# OCR specific imports (Already in your imports)
import easyocr
from PIL import Image
import cv2

# Import custom modules (Assuming they exist based on your GitHub image)
try:
    from data_processor import validate_data_structure, preprocess_data, get_data_summary, read_csv_file
    from kinetic_models import (find_stable_points, fit_zo_model, fit_pfo_model, fit_pso_model, create_results_summary)
    from visualization import create_matplotlib_plots
except ImportError as e:
    st.error(f"❌ Ошибка загрузки кастомных модулей: {e}")
    st.stop()

# =============================================================================
# РАЗДЕЛ: Технологии OCR (Интеграция المباشرة والفعالة)
# =============================================================================
@st.cache_resource
def load_ocr_reader():
    """Загружает EasyOCR Reader и كاش 모델 لمنع إعادة التحميل، يدعم ru/en."""
    with st.spinner("⏳ Загрузка OCR моделей (en/ru)..."):
        # en للأرقام والنقاط، ru للعناوين المحتملة بالروسية
        return easyocr.Reader(['en', 'ru'], gpu=False) 

def process_image_and_extract_data(uploaded_image, col_names):
    """
    تحليل الصورة، استخراج الأرقام، تصفيتها، ترتيبها في جدول Pandas مكون من عمودين.
    """
    reader = load_ocr_reader()
    
    # تحويل بايتات Streamlit إلى تنسيق مفهوم لـ OpenCV
    try:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Не удалось декодировать изображение.")
    except Exception as e:
        st.error(f"❌ Ошибка обработки изображения: {e}")
        return None

    extracted_numbers = []
    
    with st.spinner("🧠 Идет распознавание текста على الصورة..."):
        # تنفيذ OCR
        result = reader.readtext(image)

        # استخراج النصوص وتحويلها لأرقام
        for (bbox, text, prob) in result:
            # معالجة النص: إزالة الفراغات، تحويل الكوما لنقطة
            cleaned_text = text.replace(' ', '').replace(',', '.').strip()
            
            # محاولة التحويل لرقم طفو (Float)
            try:
                # التحقق من أن النص يحتوي على أرقام فقط (أو نقطة/إشارة سالب)
                if cleaned_text and any(char.isdigit() for char in cleaned_text):
                     extracted_numbers.append(float(cleaned_text))
            except ValueError:
                pass # تخطي النصوص غير الرقمية

    if len(extracted_numbers) < 4:
        st.error("❌ OCR Ошибка: На изображении найдено недостаточно دیجیتال даних للمحاكاة (требуется минимум 2 пары x, y). Убедитесь، أن جدول واضح.")
        return None

    # ضمان عدد زوجي للأزواج (X, Y)
    if len(extracted_numbers) % 2 != 0:
        extracted_numbers = extracted_numbers[:-1]

    # إعادة تشكيل المصفوفة لعمودين
    try:
        data_array = np.array(extracted_numbers).reshape(-1, 2)
        # إنشاء جدول بأسماء الأعمدة المحددة لهذا التفاعل
        df = pd.DataFrame(data_array, columns=col_names)
        return df
    except Exception as e:
        st.error(f"❌ Ошибка форматирования данных OCR في جدول: {e}")
        return None

# =============================================================================
# РАЗДЕЛ: Вспомогательные функции (إضافة الدوال العامة المفقودة)
# =============================================================================
def apply_axis_style(ax):
    """Applies a consistent grid and tick style to matplotlib axes."""
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis='x', rotation=15, labelsize=8.5)
    ax.tick_params(axis='y', labelsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.6)

def metric_box(css_class, label, value):
    """Returns an HTML snippet for a stylized metric metric_box."""
    return f'<div class="metric-box {css_class}"><h4>{label}</h4><h2>{value}</h2></div>'

def section_header(css_class, icon, text):
    """Returns an HTML snippet for a stylized section header."""
    return f'<div class="section-header {css_class}"><h2>{icon} {text}</h2></div>'

# ✅ تم إضافة الدالة المفقودة هنا حلّاً للخطأ الأخير
def sidebar_params(inputs: list, outputs: list, file_types: list):
    """Displays application parameters in the sidebar."""
    st.sidebar.markdown('<div class="sidebar-params-title">ПАРАМЕТРЫ</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**📥 Входные данные**")
    for item in inputs:
        st.sidebar.markdown(f"- {item}")
    st.sidebar.markdown("**📤 Выходные данные:**")
    for item in outputs:
        st.sidebar.markdown(f"- {item}")
    st.sidebar.markdown("**📁 Поддерживаемые файлы:**")
    st.sidebar.markdown(", ".join(file_types))

def handle_file_upload(uploaded_file, key_prefix: str):
    """Generic handler for CSV/Excel file uploads with sheet selection."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        return read_csv_file(uploaded_file)
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names
    if len(sheet_names) > 1:
        st.sidebar.markdown("**📄 Лист Excel:**")
        selected_sheet = st.sidebar.selectbox(
            "Выберите лист", sheet_names, index=0, key=f"sheet_{key_prefix}", label_visibility="collapsed"
        )
    else:
        selected_sheet = sheet_names[0]
    return pd.read_excel(uploaded_file, sheet_name=selected_sheet)

def input_method_choice(key_prefix: str) -> str:
    """Returns a radio selector for data input method."""
    return st.radio(
        "Способ ввода данных:",
        ["📁 Загрузить файл", "✏️ Ввести данные вручную", "📷 Изображение (OCR)"],
        index=0, horizontal=True, key=f"input_method_{key_prefix}"
    )

def convert_df_to_excel(df):
    """Converts a DataFrame to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Кинетический_Расчет')
    return output.getvalue()

# CSS -- Тформирование интерфейса и стилей (בقي كما هو في مشروعك الجاهز)
st.markdown("""
<style>
/* (الـ CSS الخاص بك يبدأ من هنا كما في مشروعك الجاهز...) */
:root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stApp {
    --background-color: #f5f7fa !important;
    --secondary-background-color: #ffffff !important;
    --text-color: #1e293b !important;
    --primary-color: #2563eb !important;
    background-color: #f5f7fa !important;
    color: #1e293b !important;
}
/* (...بقية الـ CSS الخاص بك كما هو) */
</style>
""", unsafe_allow_html=True)

# =============================================================================
# РАЗДЕЛ 1: Фотокаталитические реакции (Render Photocatalysis، تم تحديث OCR)
# =============================================================================
def render_photocatalysis():
    # استدعاء الدالة (المضافة الآن)
    sidebar_params(
        inputs=["т, мин (время)", "А (оптическая плотность)", "А0 (опционально)"],
        outputs=["k₀, k₁, k₂ — константы скорости", "R², MAPE — метрики качества", "3 графика (ZO, PFO, PSO)"],
        file_types=["Excel (.xlsx)", "CSV (.csv)"]
    )

    st.markdown(section_header("sh-data", "📊", "Ввод данных"), unsafe_allow_html=True)
    method = input_method_choice("photo")

    df = None
    
    if method == "📁 Загрузить файл":
        uploaded_file = st.file_uploader("Выберите файл Excel/CSV", type=['xlsx', 'csv'], key="photo_upload")
        if uploaded_file is not None:
            try:
                df = handle_file_upload(uploaded_file, "photo")
            except Exception as e:
                st.error(f"❌ Ошибка загрузки файла: {str(e)}")
                
    elif method == "✏️ Ввести данные вручную":
        blank_data = pd.DataFrame({'т, мин': [0.0]*6, 'А': [0.0]*6})
        st.markdown("**Заполните экспериментальные данные вручную:**")
        df = st.data_editor(blank_data, use_container_width=True, num_rows="dynamic", key="photo_manual_ed")
        
    elif method == "📷 Изображение (OCR)":
        # رسالة النجاح الخضراء المخصصة
        st.markdown(f'<div class="highlight-success">✅ Оптическая плотность (А) теперь распознается и отображается. Пожалуйста, проверьте и отредактируйте данные ниже при необходимости.</div>', unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader("Выберите изображение таблицы (т vs А)", type=['png', 'jpg', 'jpeg'], key="ocr_upload_photo")
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Загруженное изображение", use_container_width=True)
            
            # استخراج البيانات فعلياً باستخدام الدالة الجديدة
            extracted_df = process_image_and_extract_data(uploaded_image, ['т, мин', 'А'])
            
            # عرض محرر بيانات للتحقق والتحرير قبل التمرير للمعالجة
            if extracted_df is not None:
                # منطق إيجاد A0 تلقائياً في البيانات المستخرجة
                # قصر البيانات على القيم الصالحة قبل حساب A0
                temp_df = extracted_df[pd.to_numeric(extracted_df['А'], errors='coerce') > 0].copy()
                if not temp_df.empty:
                    extracted_df['А0'] = temp_df['А'].iloc[0]
                    st.markdown(f'<div class="highlight-success">✅ Автоопределение: А0 = {extracted_df["А0"].iloc[0]:.5f}</div>', unsafe_allow_html=True)

                if 'А/А0' not in extracted_df.columns and 'А' in extracted_df.columns and 'А0' in extracted_df.columns:
                    try:
                        extracted_df['А/А0'] = pd.to_numeric(extracted_df['А'], errors='coerce') / pd.to_numeric(extracted_df['А0'], errors='coerce')
                    except:
                        pass
                
                with st.expander("👁️ Просмотр и редактирование распознанных данных OCR"):
                    # تمرير الجدول المستخرج إلى محرر البيانات للتأكيد النهائي من المستخدم
                    df = st.data_editor(extracted_df, use_container_width=True, num_rows="dynamic", key=f"photo_final_ed_ocr")

    if df is None: return
    
    # تنظيف وتجهيز الأعمدة لضمان عمل الحسابات (استدعاء دالتك المدمجة preprocess_data)
    # التأكد من تمرير الأعمدة المستهدفة فقط
    final_cols = [c for c in ['т, мин', 'А', 'А0', 'А/А0'] if c in df.columns]
    if len(final_cols) < 2: 
        st.warning("⚠️ Недостаточно столбцов البيانات для расчета.")
        return
        
    df_clean = df[final_cols].copy()
    
    # استدعاء دالتك الأصليةpreprocess_data المستوردة منdata_processor.py
    processed_df = preprocess_data(df_clean) 
    
    if processed_df.empty:
        st.warning("⚠️ Нет допустимых данных للمحاكاة. Убедитесь، أن الأرقام تم استخراجها بشكل صحيح.")
        return
        
    # --- استمرار الكود الأصلي للحسابات والرسوم كما هو ---
    summary = get_data_summary(processed_df)

    st.markdown(section_header("sh-selected", "📊", "Сводка данных"), unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_box("mb-amber", "Действительные точки", summary["total_points"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_box("mb-amber", "Диапазон времени", f'{summary["time_range"][0]:.1f}-{summary["time_range"][1]:.1f} мин'), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_box("mb-amber", "Начальная концентрация", f'{summary["a0_value"]:.3f}'), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_box("mb-amber", "Диапазон А/А0", f'{summary["a_a0_range"][0]:.3f}-{summary["a_a0_range"][1]:.3f}'), unsafe_allow_html=True)

    try:
        # استدعاء دوال الملاءمة المستوردة من kinetic_models.py
        k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(processed_df)
        k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(processed_df)
        k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(processed_df)

        st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
        results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
        st.dataframe(results_summary, use_container_width=True)

        st.markdown(section_header("sh-compare", "⚖️", "Сравнение моделей"), unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="model-card mc-zo"><h3>Модель ZO</h3><p><strong>k₀:</strong> {abs(k0):.4f}</p><p><strong>R²:</strong> {r2_zo:.4f}</p><p><strong>MAPE:</strong> {mape_zo:.2f}%</p></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="model-card mc-pfo"><h3>Модель PFO</h3><p><strong>k₁:</strong> {abs(k1):.5f} мин⁻¹</p><p><strong>R²:</strong> {r2_pfo:.4f}</p><p><strong>MAPE:</strong> {mape_pfo:.2f}%</p></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="model-card mc-pso"><h3>Модель PSO</h3><p><strong>k₂:</strong> {k2:.5f} л/(мг·мин)</p><p><strong>R²:</strong> {r2_pso:.4f}</p><p><strong>MAPE:</strong> {mape_pso:.2f}%</p></div>', unsafe_allow_html=True)

        st.markdown(section_header("sh-viz", "📊", "Графика"), unsafe_allow_html=True)
        # استدعاء دالة الرسم المستوردة من visualization.py
        fig_main = create_matplotlib_plots(processed_df, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
        
        for ax in fig_main.get_axes():
            apply_axis_style(ax)

        col_side1, col_chart_photo, col_side2 = st.columns([1, 2, 1])
        with col_chart_photo:
            st.pyplot(fig_main)

        st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(results_summary), file_name="kinetic_analysis_results.xlsx", use_container_width=True)
        with d_col2:
            png_b = BytesIO()
            fig_main.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
            png_b.seek(0)
            st.download_button("🖼️ Скачать графики (PNG)", data=png_b, file_name="kinetic_plots.png", mime="image/png", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Ошибка моделирования: {str(e)}")

# =============================================================================
# MAIN ENTRYPOINT (بقي كما هو في مشروعك الجاهز)
# =============================================================================
def main():
    st.markdown("""
    <div class="main-header-title">
         <h1>Анализ кинетического моделирования</h1>
    </div>
    <div class="main-header-authors">
         <p>АВТОР: Алсади К. &nbsp;|&nbsp; РУКОВОДИТЕЛЬ: Киреева А.В</p>
    </div>
    """, unsafe_allow_html=True)

    reaction_type = st.sidebar.selectbox(
        "🛠️ Тип процесса / реакции",
        options=["Фотокаталитические реакции"], # قدمت هذا الخيار كمثال للدمج الكامل
        index=0, key="reaction_type_choice"
    )

    if reaction_type == "Фотокаталитические реакции":
        render_photocatalysis()

if __name__ == "__main__":
    main()
