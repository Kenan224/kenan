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

# Import custom modules
from data_processor import validate_data_structure, preprocess_data, get_data_summary, read_csv_file
from kinetic_models import (
    find_stable_points, fit_zo_model, fit_pfo_model, fit_pso_model,
    create_results_summary
)
from visualization import create_matplotlib_plots

# Configure page
st.set_page_config(
    page_title="Анализ кинетического моделирования",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS -- التحديث الشامل للأحجام وتكبير العنوان وتنسيق الألوان
# =============================================================================
st.markdown("""
<style>
/* 1) إجبار الجذور والمتغيرات الأساسية لـ Streamlit على الألوان الفاتحة */
:root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stApp {
    --background-color: #f5f7fa !important;
    --secondary-background-color: #ffffff !important;
    --text-color: #1e293b !important;
    --primary-color: #2563eb !important;
    background-color: #f5f7fa !important;
    color: #1e293b !important;
}

/* 2) استهدف القوائم وحقول الإدخال والكبسولات */
div[data-baseweb="select"], 
div[data-baseweb="popover"], 
div[role="listbox"], 
ul[data-baseweb="menu"],
div[data-testid="stSelectbox"],
.stSelectbox, 
input, 
select, 
textarea,
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}

/* كبسولة الملف المرفوع */
[data-testid="stUploadedFile"] {
    background-color: #e2e8f0 !important;
    background: #e2e8f0 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stUploadedFile"] * {
    background-color: transparent !important;
    background: transparent !important;
    color: #1e293b !important;
    fill: #1e293b !important;
}

/* سهم القائمة المنسدلة */
div[data-baseweb="select"] button, 
div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] button,
div[data-testid="stSelectbox"] div {
    background-color: transparent !important;
    background: transparent !important;
    color: #1e293b !important;
}
div[data-baseweb="select"] svg,
div[data-testid="stSelectbox"] svg {
    fill: #1e293b !important;
    color: #1e293b !important;
}

div[data-baseweb="select"] *, 
div[data-baseweb="popover"] *, 
div[role="listbox"] *,
[data-testid="stSelectbox"] * {
    color: #1e293b !important;
}

li[role="option"], [data-baseweb="menu"] li, div[data-baseweb="popover"] div {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1e293b !important;
}
li[role="option"]:hover, [data-baseweb="menu"] li:hover {
    background-color: #eff6ff !important;
    color: #1e3a8a !important;
}

/* إجبار نصوص التطبيق العادية على حجم أكبر شحطتين وتوحيده */
html, body, p, span, label, th, td, .stMarkdown, .stRadio label, input, select, button {
    color: #1e293b !important;
    font-size: 1.1rem !important;
}

/* 3) الهيدر الرئيسي - تم تكبيره هنا ليكون ضخماً وواضحاً */
.main-header-title {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.2);
    text-align: center;
}
.main-header-title h1 {
    margin: 0;
    font-weight: 700;
    font-size: 5.2rem !important; /* حجم كبير جداً للعنوان */
    color: #ffffff !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

/* مستطيل منفرد ومنفصل تماماً للأسماء والمعلومات */
.main-header-authors {
    background: #eff6ff;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.06);
    border: 1px solid #bfdbfe;
    text-align: center;
}
.main-header-authors p { 
    color: #1e40af !important; 
    margin: 0; 
    font-weight: 600;
    font-size: 1.1rem !important;
}

/* كروت المعلومات */
.info-card {
    background: #ffffff;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e2e8f0;
}

/* صناديق المقاييس */
.metric-box {
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    border: 1px solid;
}
.metric-box h4 { margin: 0; font-size: 0.95rem !important; font-weight: 600; }
.metric-box h2 { margin: 0.3rem 0 0 0; font-size: 1.5rem !important; font-weight: 700; }

.mb-amber { background: #fffbeb !important; border-color: #fde68a !important; }
.mb-amber h4, .mb-amber h2 { color: #92400e !important; }

.mb-blue { background: #eff6ff !important; border-color: #bfdbfe !important; }
.mb-blue h4, .mb-blue h2 { color: #1e40af !important; }

.mb-green { background: #ecfdf5 !important; border-color: #a7f3d0 !important; }
.mb-green h4, .mb-green h2 { color: #065f46 !important; }

.mb-rose { background: #fff1f2 !important; border-color: #fecdd3 !important; }
.mb-rose h4, .mb-rose h2 { color: #9f1239 !important; }

.performance-metric {
    background: #f0fdf4 !important;
    border: 1px solid #10b981 !important;
    padding: 0.7rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    color: #065f46 !important;
    font-weight: 600;
    text-align: center;
    font-size: 1.15rem !important;
}

/* عناوين الأقسام */
.section-header {
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    border-left: 5px solid;
    margin-bottom: 1rem;
}
.section-header h2 { margin: 0; font-size: 1.35rem !important; }

.sh-data { background: #eff6ff !important; border-left-color: #2563eb !important; }
.sh-data h2 { color: #1e3a8a !important; }

.sh-results { background: #ecfdf5 !important; border-left-color: #10b981 !important; }
.sh-results h2 { color: #065f46 !important; }

.sh-selected { background: #fff1f2 !important; border-left-color: #f43f5e !important; }
.sh-selected h2 { color: #9f1239 !important; }

.sh-compare { background: #f0fdfa !important; border-left-color: #0d9488 !important; }
.sh-compare h2 { color: #115e59 !important; }

.sh-viz { background: #faf5ff !important; border-left-color: #a855f7 !important; }
.sh-viz h2 { color: #6b21a8 !important; }

.sh-download { background: #fef2f2 !important; border-left-color: #ef4444 !important; margin-top: 1.2rem; }
.sh-download h2 { color: #991b1b !important; }

.highlight-success {
    background: #dcfce7 !important;
    border: 1px solid #22c55e !important;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    color: #14532d !important;
}

/* كروت الموديلات */
.model-card {
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid;
}
.model-card h3 { margin: 0 0 0.5rem 0; font-size: 1.25rem !important; }
.model-card p { margin: 0.25rem 0; font-size: 1.05rem !important; }

.mc-zo   { background: #f8fafc !important; border-left-color: #94a3b8 !important; }
.mc-zo   h3, .mc-zo   p { color: #334155 !important; }

.mc-pfo  { background: #eff6ff !important; border-left-color: #2563eb !important; }
.mc-pfo  h3, .mc-pfo  p { color: #1e3a8a !important; }

.mc-pso  { background: #f0fdf4 !important; border-left-color: #16a34a !important; }
.mc-pso  h3, .mc-pso  p { color: #14532d !important; }

/* الشريط الجانبي */
section[data-testid="stSidebar"] { background-color: #ffffff !important; }
.sidebar-params-title {
    font-size: 1.2rem !important;
    font-weight: 700;
    color: #1e3a8a !important;
    margin: 0.6rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #2563eb;
}

.stDownloadButton button, .stButton button {
    background-color: #ffffff !important;
    color: #1e3a8a !important;
    border: 1.5px solid #2563eb !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}
.stDownloadButton button:hover, .stButton button:hover {
    background-color: #2563eb !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# دوال مساعدة عامة وتنسيق المخططات البيانية
# =============================================================================
def apply_axis_style(ax):
    """دالة موحدة لمنع تداخل أرقام المحاور وإعطائها مظهراً متناسقاً متباعداً"""
    ax.xaxis.set_major_locator(MaxNLocator(5))  # حد أقصى 5 قيم على المحور الأفقي لمنع الالتصاق
    ax.yaxis.set_major_locator(MaxNLocator(5))  # حد أقصى 5 قيم على المحور الرأسي
    ax.tick_params(axis='x', rotation=15, labelsize=8.5) # تدوير بسيط لمنع التداخل تماماً
    ax.tick_params(axis='y', labelsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.6)

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
    return r2, mape

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Кинетический_Расчет')
    return output.getvalue()

def clean_homogeneous_data(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    if any('Unnamed' in str(col) for col in df.columns):
        for i in range(min(5, len(df))):
            row_values = [str(x).strip().lower() for x in df.iloc[i].dropna()]
            has_t = any('t' == x or 'время' in x or 'time' in x for x in row_values)
            has_ca = any('ca' in x or 'концентрация' in x or 'c_a' in x for x in row_values)
            if has_t or has_ca:
                new_headers = df.iloc[i].tolist()
                df = df.iloc[i + 1:].copy()
                df.columns = new_headers
                break

    if len(df.columns) == 1:
        first_col = str(df.columns[0])
        for sep in [';', '\t']:
            if sep in first_col:
                header_parts = first_col.split(sep)
                rows = [str(row.iloc[0]).split(sep) for _, row in df.iterrows()]
                df = pd.DataFrame(rows, columns=header_parts)
                break

    new_columns = []
    for col in df.columns:
        c = str(col).strip().lower()
        c_clean = c.replace(' ', '').replace('_', '').replace('-', '').replace(',', '').replace('.', '')
        c_clean = c_clean.replace('с', 'c').replace('а', 'a').replace('в', 'b').replace('т', 't').replace('р', 'r')

        if 'ca' in c_clean or 'концентрацияa' in c_clean or c_clean == 'а' or c_clean == 'a':
            new_columns.append('CA')
        elif 'cb' in c_clean or 'концентрацияb' in c_clean or c_clean == 'в' or c_clean == 'b':
            new_columns.append('CB')
        elif 'cc' in c_clean or 'концентрацияc' in c_clean or c_clean == 'с' or c_clean == 'c':
            new_columns.append('CC')
        elif any(x in c_clean for x in ['rate', 'скорость', 'скоростьреакции']) or c_clean in ['r', 'w', 'v']:
            new_columns.append('r')
        elif 'k' in c_clean and c_clean != 'tk':
            new_columns.append('k')
        elif 'temp' in c_clean or 'темп' in c_clean or c_clean == 'tk' or str(col).strip() in ['T', 'Т']:
            new_columns.append('T')
        elif 'time' in c_clean or 'время' in c_clean or c_clean in ['t', 'т']:
            new_columns.append('t')
        else:
            new_columns.append(col)

    cleaned_dict = {}
    for i in range(len(df.columns)):
        col_name = new_columns[i]
        if 'Unnamed' in str(col_name):
            continue
        series = df.iloc[:, i].astype(str)
        series = series.str.replace(r'\s+', '', regex=True)
        series = series.str.replace(',', '.', regex=False)
        numeric_series = pd.to_numeric(series, errors='coerce')

        if col_name in cleaned_dict:
            cleaned_dict[f"{col_name}_dup_{i}"] = numeric_series
        else:
            cleaned_dict[col_name] = numeric_series

    final_df = pd.DataFrame(cleaned_dict)
    return final_df.dropna(how='all')

def metric_box(css_class, label, value):
    return f'<div class="metric-box {css_class}"><h4>{label}</h4><h2>{value}</h2></div>'

def section_header(css_class, icon, text):
    return f'<div class="section-header {css_class}"><h2>{icon} {text}</h2></div>'

def sidebar_params(inputs: list, outputs: list, file_types: list):
    st.sidebar.markdown('<div class="sidebar-params-title">ПАРАМЕТРЫ</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**📥 Входные данные:**")
    for item in inputs:
        st.sidebar.markdown(f"- {item}")
    st.sidebar.markdown("**📤 Выходные данные:**")
    for item in outputs:
        st.sidebar.markdown(f"- {item}")
    st.sidebar.markdown("**📁 Поддерживаемые файлы:**")
    st.sidebar.markdown(", ".join(file_types))

def handle_file_upload(uploaded_file, key_prefix: str):
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
    return st.radio(
        "Способ ввода данных:",
        ["📁 Загрузить файл", "✏️ Ввести данные вручную"],
        index=0, horizontal=True, key=f"input_method_{key_prefix}"
    )

# =============================================================================
# قسم 1: الفوتوكاتاليز
# =============================================================================
def render_photocatalysis():
    sidebar_params(
        inputs=["т, мин (время)", "А (оптическая плотность)", "А0 (опционально)"],
        outputs=["k₀, k₁, k₂ — константы скорости", "R², MAPE — метрики качества", "3 графика (ZO, PFO, PSO)"],
        file_types=["Excel (.xlsx)", "CSV (.csv)"]
    )

    st.markdown(section_header("sh-data", "📊", "Ввод данных"), unsafe_allow_html=True)
    method = input_method_choice("photo")

    df = None
    if method == "📁 Загрузить файл":
        uploaded_file = st.file_uploader("Выберите файл", type=['xlsx', 'csv'], key="photo_upload")
        if uploaded_file is not None:
            try:
                df = handle_file_upload(uploaded_file, "photo")
                df.columns = df.columns.str.strip()
                is_valid, error_message = validate_data_structure(df)
                if not is_valid:
                    st.error(f"❌ {error_message}")
                    return

                if 'А0' not in df.columns and 'А' in df.columns and len(df) > 0:
                    df['А'] = pd.to_numeric(df['А'], errors='coerce')
                    valid_a_mask = (df['А'] > 0) & (~df['А'].isna())
                    if valid_a_mask.any():
                        df['А0'] = df.loc[valid_a_mask, 'А'].iloc[0]
                        st.markdown(f'<div class="highlight-success">✅ Автоопределение: А0 = {df["А0"].iloc[0]:.5f}</div>', unsafe_allow_html=True)

                if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                    df['А/А0'] = df['А'] / df['А0']

                with st.expander("👁️ Предварительный просмотр данных"):
                    st.dataframe(df, use_container_width=True)

                edited_df = st.data_editor(df[['т, мин', 'А']].copy(), use_container_width=True, num_rows="dynamic", key="photo_upload_ed")
                if 'А' in edited_df.columns and len(edited_df) > 0 and edited_df.iloc[0]['А'] > 0:
                    edited_df['А0'] = edited_df.iloc[0]['А']
                    edited_df = edited_df[(edited_df['А'] > 0) & (edited_df['т, мин'] >= 0)]
                    if not edited_df.empty:
                        edited_df['А/А0'] = edited_df['А'] / edited_df['А0']
                        df = edited_df.copy()
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
    else:
        default_data = pd.DataFrame({'т, мин': [0.0], 'А': [0.0]})
        edited_data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="photo_manual_ed")
        if not edited_data.empty and 'А' in edited_data.columns and len(edited_data) > 0 and edited_data.iloc[0]['А'] > 0:
            edited_data['А0'] = edited_data.iloc[0]['А']
            valid_data = edited_data[(edited_data['А'] > 0) & (edited_data['т, мин'] >= 0)].copy()
            if not valid_data.empty:
                valid_data['А/А0'] = valid_data['А'] / valid_data['А0']
                df = valid_data.copy()
                st.dataframe(df[['т, мин', 'А', 'А/А0']], use_container_width=True)

    if df is None or df.empty:
        return

    processed_df = preprocess_data(df)
    if processed_df.empty:
        st.warning("⚠️ Нет допустимых данных для анализа.")
        return
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

    stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
    selected_data = processed_df.iloc[stable_indices].copy()

    st.markdown(section_header("sh-selected", "📌", "Выбранные точки"), unsafe_allow_html=True)
    col_pts1, col_pts2 = st.columns(2)
    with col_pts1:
        st.markdown(metric_box("mb-blue", "Выбранные точки данных", f'{len(selected_data)} из {len(processed_df)}'), unsafe_allow_html=True)
    with col_pts2:
        if not selected_data.empty:
            st.markdown(metric_box("mb-blue", "Временной диапазон", f'{selected_data["т, мин"].min():.1f} - {selected_data["т, мин"].max():.1f} мин'), unsafe_allow_html=True)
        else:
            st.markdown(metric_box("mb-blue", "Временной диапазон", "0.0 мин"), unsafe_allow_html=True)

    try:
        k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(selected_data)
        k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
        k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

        st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
        results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
        st.dataframe(results_summary, use_container_width=True)

        st.markdown(section_header("sh-compare", "⚖️", "Сравнение моделей"), unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="model-card mc-zo"><h3>Модель ZO</h3><p><strong>k₀:</strong> {abs(k0):.5f}</p><p><strong>R²:</strong> {r2_zo:.4f}</p><p><strong>MAPE:</strong> {mape_zo:.2f}%</p></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="model-card mc-pfo"><h3>Модель PFO</h3><p><strong>k₁:</strong> {abs(k1):.5f} мин⁻¹</p><p><strong>R²:</strong> {r2_pfo:.4f}</p><p><strong>MAPE:</strong> {mape_pfo:.2f}%</p></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="model-card mc-pso"><h3>Модель PSO</h3><p><strong>k₂:</strong> {k2:.5f} л/(мг·мин)</p><p><strong>R²:</strong> {r2_pso:.4f}</p><p><strong>MAPE:</strong> {mape_pso:.2f}%</p></div>', unsafe_allow_html=True)

        st.markdown(section_header("sh-viz", "📊", "Графика"), unsafe_allow_html=True)
        fig_main = create_matplotlib_plots(processed_df, selected_data, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
        
        # لتعديل محاور هذا الجرافيك منعاً للتداخل
        for ax in fig_main.get_axes():
            apply_axis_style(ax)

        # توسيط جرافيك الفوتوكاتاليز في منتصف الصفحة تماماً
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
# قسم 2: التحفيز المتجانس
# =============================================================================
HOMO_MODEL_INFO = {
    "Power-law (степенной закон)": {
        "inputs": ["t (время)", "CA (концентрация А)", "CB (концентрация B)", "r (скорость)"],
        "outputs": ["k, α, β — параметры модели", "R², MAPE", "график линейной зависимости"],
    },
    "Arrhenius": {
        "inputs": ["T (температура, K)", "k (константа скорости)"],
        "outputs": ["A, Ea — параметры Аррениуса", "R², MAPE", "график Аррениуса"],
    },
    "Последовательные реакции": {
        "inputs": ["t (время)", "CA, CB, CC (концентрации веществ)"],
        "outputs": ["k1, k2 — константы скорости", "R², RMSE (%), Max C_B — метрики", "профиль концентраций"],
    },
}

def render_homogeneous():
    st.markdown("### 📈 Выберите кинетическую модель:")
    homo_model = st.radio("Кинетическая модель:", list(HOMO_MODEL_INFO.keys()), index=0, horizontal=True, key="homo_model_choice")
    info = HOMO_MODEL_INFO[homo_model]
    sidebar_params(inputs=info["inputs"], outputs=info["outputs"], file_types=["Excel (.xlsx)", "CSV (.csv)"])

    st.markdown(section_header("sh-data", "📊", f"Ввод данных ({homo_model})"), unsafe_allow_html=True)
    method = input_method_choice(f"homo_{homo_model}")

    h_df = None
    if method == "📁 Загрузить файл":
        uploaded_h_file = st.file_uploader("Выберите файл Excel/CSV", type=['xlsx', 'csv'], key=f"file_{homo_model}")
        if uploaded_h_file is not None:
            try:
                h_df = handle_file_upload(uploaded_h_file, f"homo_{homo_model}")
            except Exception as e:
                st.error(f"❌ Ошибка загрузки файла: {str(e)}")
    else:
        if homo_model == "Power-law (степенной закон)":
            empty_df = pd.DataFrame(columns=['t', 'CA', 'CB', 'r'], data=[[0.0, 1.0, 1.5, 0.05], [5.0, 0.8, 1.3, 0.035]])
        elif homo_model == "Arrhenius":
            empty_df = pd.DataFrame(columns=['T', 'k'], data=[[298.0, 0.01], [308.0, 0.02]])
        else:
            empty_df = pd.DataFrame(columns=['t', 'CA', 'CB', 'CC'], data=[[0.0, 1.0, 0.0, 0.0], [10.0, 0.5, 0.4, 0.1]])
        st.markdown("**Заполните таблицу данными:**")
        h_df = st.data_editor(empty_df, use_container_width=True, num_rows="dynamic", key=f"editor_{homo_model}")

    if h_df is None or len(h_df) == 0:
        return

    h_df = clean_homogeneous_data(h_df)

    # 1) مـوديل الـ Power-law
    if homo_model == "Power-law (степенной закон)":
        if not all(c in h_df.columns for c in ['t', 'CA', 'CB', 'r']):
            st.error("❌ **Ошибка структуры!**")
            return
        try:
            clean_df = h_df[(h_df['CA'] > 0) & (h_df['CB'] > 0) & (h_df['r'] > 0)]
            if clean_df.empty: return
            log_CA, log_CB, log_r = np.log(clean_df['CA'].values), np.log(clean_df['CB'].values), np.log(clean_df['r'].values)
            X = np.column_stack((np.ones_like(log_CA), log_CA, log_CB))
            beta_matrix, _, _, _ = np.linalg.lstsq(X, log_r, rcond=None)
            k_val, alpha_val, beta_val = np.exp(beta_matrix[0]), beta_matrix[1], beta_matrix[2]
            r_pred = k_val * (clean_df['CA'].values ** alpha_val) * (clean_df['CB'].values ** beta_val)
            r2, mape = calculate_metrics(clean_df['r'].values, r_pred)

            st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f'<div class="performance-metric">⚡ k = {k_val:.4f}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">🔸 α = {alpha_val:.2f}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">🔹 β = {beta_val:.2f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">📊 R² = {r2:.4f}</div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="performance-metric">📈 MAPE = {mape:.2f}%</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4.8, 2.8))
            x_linear = (clean_df['CA'].values ** alpha_val) * (clean_df['CB'].values ** beta_val)
            ax.scatter(x_linear, clean_df['r'].values, color='#ef4444', label='Эксперимент', s=20)
            ax.plot(x_linear, r_pred, color='#1e40af', label=f'Модель', linewidth=1.5)
            ax.set_xlabel('Фактор концентраций', fontsize=8.5)
            ax.set_ylabel('Скорость (r)', fontsize=8.5)
            
            apply_axis_style(ax) # تطبيق التنسيق والتباعد لمنع تداخل الأرقام
            ax.legend(fontsize=7.5)
            
            # توسيط الجرافيك تماماً في منتصف الشاشة
            col_side1, col_chart_pl, col_side2 = st.columns([1, 2, 1])
            with col_chart_pl:
                st.pyplot(fig)

            results_summary = pd.DataFrame({
                'Параметр / Метрика': ['Константа скорости (k)', 'Порядок по веществу A (alpha)', 'Порядок по веществу B (beta)', 'Коэффициент детерминации (R²)', 'Ошибка (MAPE, %)'],
                'Значение': [float(k_val), float(alpha_val), float(beta_val), float(r2), float(mape)]
            })
            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(results_summary), file_name="power_law_results.xlsx", use_container_width=True, key="dl_excel_pl")
            with d_col2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
                png_b.seek(0)
                st.download_button("🖼️ Скачать график (PNG)", data=png_b, file_name="power_law_plot.png", mime="image/png", use_container_width=True, key="dl_png_pl")

        except Exception as e: st.error(f"❌ Ошибка: {str(e)}")

    # 2) مـوديل Arrhenius
    elif homo_model == "Arrhenius":
        if not all(c in h_df.columns for c in ['T', 'k']): return
        try:
            clean_df = h_df[(h_df['T'] > 0) & (h_df['k'] > 0)]
            if clean_df.empty: return
            R = 8.314
            inv_T, log_k = 1.0 / clean_df['T'].values, np.log(clean_df['k'].values)
            slope, intercept = np.polyfit(inv_T, log_k, 1)
            Ea_val, A_val = -slope * R / 1000.0, np.exp(intercept)
            k_pred = A_val * np.exp(-(Ea_val * 1000.0) / (R * clean_df['T'].values))
            r2, mape = calculate_metrics(clean_df['k'].values, k_pred)

            st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="performance-metric">🧪 A = {A_val:.2e}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">🔥 Ea = {Ea_val:.2f} кДж/моль</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">📊 R² = {r2:.4f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">📈 MAPE = {mape:.2f}%</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4.8, 2.8))
            ax.scatter(inv_T, log_k, color='#ef4444', label='Эксперимент', s=20)
            ax.plot(inv_T, slope * inv_T + intercept, color='#10b981', linewidth=1.5, label='Линейный тренд')
            ax.set_xlabel('1/T (1/K)', fontsize=8.5)
            ax.set_ylabel('ln(k)', fontsize=8.5)
            
            apply_axis_style(ax) # تطبيق التنسيق والتدوير الفريد لمنع تداخل أرقام الكسور الطويلة تماماً
            ax.legend(fontsize=7.5)
            
            # توسيط جرافيك معادلة أرينيوس بالمنتصف تماماً
            col_side1, col_chart_arr, col_side2 = st.columns([1, 2, 1])
            with col_chart_arr:
                st.pyplot(fig)

            results_summary = pd.DataFrame({
                'Параметр / Метрика': ['Предэкспоненциальный множитель (A)', 'Энергия активации (Ea, кДж/моль)', 'Коэффициент детерминации (R²)', 'Ошибка (MAPE, %)'],
                'Значение': [float(A_val), float(Ea_val), float(r2), float(mape)]
            })
            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(results_summary), file_name="arrhenius_results.xlsx", use_container_width=True, key="dl_excel_arr")
            with d_col2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
                png_b.seek(0)
                st.download_button("🖼️ Скачать график (PNG)", data=png_b, file_name="arrhenius_plot.png", mime="image/png", use_container_width=True, key="dl_png_arr")

        except Exception as e: st.error(f"❌ Ошибка: {str(e)}")

    # 3) مـوديل التفاعلات المتتالية (Последовательные реакции)
    elif homo_model == "Последовательные реакции":
        if not all(c in h_df.columns for c in ['t', 'CA', 'CB', 'CC']): return
        try:
            t_data, CA_data = h_df['t'].values, h_df['CA'].values
            popt1, _ = curve_fit(lambda t, k1: CA_data[0] * np.exp(-k1 * t), t_data, CA_data, p0=[0.05])
            k1_fit = popt1[0]

            def fit_B(t, k2):
                return (k1_fit * CA_data[0] / (k2 - k1_fit)) * (np.exp(-k1_fit * t) - np.exp(-k2 * t)) if k2 != k1_fit else (k1_fit * CA_data[0] * t * np.exp(-k1_fit * t))

            popt2, _ = curve_fit(fit_B, t_data, h_df['CB'].values, p0=[0.02])
            k2_fit = popt2[0]

            CA_pred, CB_pred = CA_data[0] * np.exp(-k1_fit * t_data), fit_B(t_data, k2_fit)
            CC_pred = CA_data[0] - CA_pred - CB_pred
            
            r2_A, _ = calculate_metrics(CA_data, CA_pred)
            r2_B, _ = calculate_metrics(h_df['CB'].values, CB_pred)
            r2_C, _ = calculate_metrics(h_df['CC'].values, CC_pred)
            r2_final = (r2_A + r2_B + r2_C) / 3.0
            
            rmse_val = np.sqrt(np.mean(list((CA_data - CA_pred)**2) + list((h_df['CB'].values - CB_pred)**2) + list((h_df['CC'].values - CC_pred)**2)))

            t_fine_mesh = np.linspace(0, max(t_data) * 1.5, 2000)
            CB_fine_mesh = fit_B(t_fine_mesh, k2_fit)
            max_idx = np.argmax(CB_fine_mesh)
            CB_max_val = CB_fine_mesh[max_idx]
            t_max_val = t_fine_mesh[max_idx]

            st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f'<div class="performance-metric">¼ k₁ = {k1_fit:.4f}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">½ k₂ = {k2_fit:.4f}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">📊 R² = {r2_final:.4f}</div>', unsafe_allow_html=True)
            # تم إلحاق النسبة المئوية % هنا لهذ الموديل بناءً على طلبك ليتوحد مع البقية
            c4.markdown(f'<div class="performance-metric">📈 RMSE = {rmse_val:.2f}%</div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="performance-metric">🔝 CB,max={CB_max_val:.3f} ({t_max_val:.1f} мин)</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4.8, 2.8))
            ax.plot(t_data, CA_data, 'o', color='#ef4444', markersize=4)
            ax.plot(t_data, CA_pred, '-', color='#ef4444', label='A', linewidth=1.5)
            ax.plot(t_data, h_df['CB'].values, 'o', color='#16a34a', markersize=4)
            ax.plot(t_data, CB_pred, '-', color='#16a34a', label='B (Промежуточный)', linewidth=1.5)
            ax.plot(t_data, h_df['CC'].values, 'o', color='#2563eb', markersize=4)
            ax.plot(t_data, CC_pred, '-', color='#2563eb', label='C', linewidth=1.5)
            
            ax.set_xlabel('Время (t)', fontsize=8.5)
            ax.set_ylabel('Концентрация (C)', fontsize=8.5)
            
            apply_axis_style(ax) # تطبيق التنسيق والتباعد لمنع تداخل الأرقام
            ax.legend(fontsize=7.5)
            
            # توسيط جرافيك التفاعلات المتتالية بالمنتصف تماماً
            col_side1, col_chart_cons, col_side2 = st.columns([1, 2, 1])
            with col_chart_cons:
                st.pyplot(fig)

            # إضافة علامة % في جدول البيانات المخصص للتحميل أيضاً لتوحيد المنظر تماماً
            results_summary = pd.DataFrame({
                'Параметр / Метрика': ['Константа скорости k1', 'Константа скорости k2', 'Коэффициент детерминации (R²)', 'Ошибка (RMSE, %)', 'Макс. концентрация B (CB,max)', 'Время достижения макс. конц. (t_max)'],
                'Значение': [float(k1_fit), float(k2_fit), float(r2_final), float(rmse_val), float(CB_max_val), float(t_max_val)]
            })
            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(results_summary), file_name="consecutive_reactions_results.xlsx", use_container_width=True, key="dl_excel_cons")
            with d_col2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
                png_b.seek(0)
                st.download_button("🖼️ Скачать график (PNG)", data=png_b, file_name="consecutive_reactions_plot.png", mime="image/png", use_container_width=True, key="dl_png_cons")

        except Exception as e: st.error(f"❌ Ошибка: {str(e)}")

def render_placeholder(section_name: str):
    sidebar_params(inputs=["— в разработке —"], outputs=["— в разработке —"], file_types=["Excel (.xlsx)", "CSV (.csv)"])
    st.markdown(section_header("sh-data", "🚧", f"{section_name}"), unsafe_allow_html=True)
    st.info(f"Раздел «{section_name}» находится в разработке.")

# =============================================================================
# MAIN ENTRYPOINT
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
        options=["Фотокаталитические реакции", "Гомогенный катализ", "Гетерогенный катализ", "Ферментативные реакции"],
        index=0, key="reaction_type_choice"
    )

    if reaction_type == "Фотокаталитические реакции":
        render_photocatalysis()
    elif reaction_type == "Гомогенный катализ":
        render_homogeneous()
    elif reaction_type == "Гетерогенный катализ":
        render_placeholder("Гетерогенный катализ")
    elif reaction_type == "Ферментативные реакции":
        render_placeholder("Ферментативные реакции")

if __name__ == "__main__":
    main()
