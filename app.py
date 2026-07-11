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
# CSS -- التنسيق البصري الفاتح عالي التباين
# =============================================================================
st.markdown("""
<style>
:root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stApp {
    --background-color: #f5f7fa !important;
    --secondary-background-color: #ffffff !important;
    --text-color: #1e293b !important;
    --primary-color: #2563eb !important;
    background-color: #f5f7fa !important;
    color: #1e293b !important;
}

div[data-baseweb="select"], 
div[data-baseweb="popover"], 
div[role="listbox"], 
ul[data-baseweb="menu"],
div[data-testid="stSelectbox"],
.stSelectbox, input, select, textarea,
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}

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
}

div[data-baseweb="select"] button, div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] button, div[data-testid="stSelectbox"] div {
    background-color: transparent !important;
    color: #1e293b !important;
}
div[data-baseweb="select"] svg, div[data-testid="stSelectbox"] svg {
    fill: #1e293b !important;
    color: #1e293b !important;
}

li[role="option"], [data-baseweb="menu"] li, div[data-baseweb="popover"] div {
    background-color: #ffffff !important;
    color: #1e293b !important;
}
li[role="option"]:hover, [data-baseweb="menu"] li:hover {
    background-color: #eff6ff !important;
    color: #1e3a8a !important;
}

html, body, p, span, label, th, td, .stMarkdown, .stRadio label {
    color: #1e293b !important;
}

.main-header-title {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    padding: 1.6rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.2);
    text-align: center;
}
.main-header-title h1 {
    margin: 0;
    font-weight: 600;
    font-size: 2rem;
    color: #ffffff !important;
}

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
}

.metric-box {
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    border: 1px solid;
}
.metric-box h4 { margin: 0; font-size: 0.85rem; font-weight: 600; }
.metric-box h2 { margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: 700; }

.mb-amber { background: #fffbeb !important; border-color: #fde68a !important; }
.mb-amber h4, .mb-amber h2 { color: #92400e !important; }
.mb-blue { background: #eff6ff !important; border-color: #bfdbfe !important; }
.mb-blue h4, .mb-blue h2 { color: #1e40af !important; }

.performance-metric {
    background: #f0fdf4 !important;
    border: 1px solid #10b981 !important;
    padding: 0.7rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    color: #065f46 !important;
    font-weight: 600;
    text-align: center;
}

.section-header {
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    border-left: 5px solid;
    margin-bottom: 1rem;
}
.section-header h2 { margin: 0; font-size: 1.25rem; }
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
    color: #14532d !important;
}

.model-card {
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid;
}
.mc-zo   { background: #f8fafc !important; border-left-color: #94a3b8 !important; }
.mc-pfo  { background: #eff6ff !important; border-left-color: #2563eb !important; }
.mc-pso  { background: #f0fdf4 !important; border-left-color: #16a34a !important; }

section[data-testid="stSidebar"] { background-color: #ffffff !important; }
.sidebar-params-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e3a8a !important;
    margin: 0.6rem 0 0.4rem 0;
    border-bottom: 2px solid #2563eb;
}

.stDownloadButton button, .stButton button {
    background-color: #ffffff !important;
    color: #1e3a8a !important;
    border: 1.5px solid #2563eb !important;
    font-weight: 600 !important;
}
.stDownloadButton button:hover, .stButton button:hover {
    background-color: #2563eb !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# دوال عامة ومقاييس الأداء
# =============================================================================
def calculate_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return r2, mape, rmse


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Кинетический_Расчет')
    return output.getvalue()


def clean_homogeneous_data(df):
    if df is None or df.empty: return df
    df = df.copy()
    if any('Unnamed' in str(col) for col in df.columns):
        for i in range(min(5, len(df))):
            row_values = [str(x).strip().lower() for x in df.iloc[i].dropna()]
            if any('t' == x or 'время' in x for x in row_values):
                df.columns = df.iloc[i].tolist()
                df = df.iloc[i + 1:].copy()
                break

    new_columns = []
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(' ', '').replace('_', '')
        c_clean = c_clean.replace('с', 'c').replace('а', 'a').replace('в', 'b').replace('т', 't')
        if 'ca' in c_clean or c_clean in ['а', 'a']: new_columns.append('CA')
        elif 'cb' in c_clean or c_clean in ['в', 'b']: new_columns.append('CB')
        elif 'cc' in c_clean or c_clean in ['с', 'c']: new_columns.append('CC')
        elif any(x in c_clean for x in ['rate', 'скорость']) or c_clean in ['r', 'w', 'v']: new_columns.append('r')
        elif 'temp' in c_clean or c_clean in ['t', 'т']: new_columns.append('T')
        elif 'time' in c_clean or c_clean in ['t', 'т']: new_columns.append('t')
        elif 'k' in c_clean: new_columns.append('k')
        else: new_columns.append(col)

    df.columns = new_columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    return df.dropna(how='all')


def metric_box(css_class, label, value):
    return f'<div class="metric-box {css_class}"><h4>{label}</h4><h2>{value}</h2></div>'


def section_header(css_class, icon, text):
    return f'<div class="section-header {css_class}"><h2>{icon} {text}</h2></div>'


def sidebar_params(inputs: list, outputs: list, file_types: list):
    st.sidebar.markdown('<div class="sidebar-params-title">ПАРАМЕТРЫ</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**📥 Входные данные:**")
    for item in inputs: st.sidebar.markdown(f"- {item}")
    st.sidebar.markdown("**📤 Выходные данные:**")
    for item in outputs: st.sidebar.markdown(f"- {item}")
    st.sidebar.markdown("**📁 Поддерживаемые файлы:**")
    st.sidebar.markdown(", ".join(file_types))


def handle_file_upload(uploaded_file, key_prefix: str):
    if uploaded_file.name.split('.')[-1].lower() == 'csv': return read_csv_file(uploaded_file)
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_name = st.sidebar.selectbox("Выберите лист", excel_file.sheet_names, key=f"sheet_{key_prefix}") if len(excel_file.sheet_names) > 1 else excel_file.sheet_names[0]
    return pd.read_excel(uploaded_file, sheet_name=sheet_name)


def input_method_choice(key_prefix: str) -> str:
    return st.radio("Способ ввода данных:", ["📁 Загрузить файл", "✏️ Ввести данные вручную"], index=0, horizontal=True, key=f"method_{key_prefix}")


# =============================================================================
# القسم 1: الفوتوكاتاليز (مع حل مشكلة الـ Coercion وتصغير الرسمة)
# =============================================================================
def render_photocatalysis():
    sidebar_params(
        inputs=["т, мин (время)", "А (оптическая плотность)"],
        outputs=["k₀, k₁, k₂", "R², MAPE, RMSE (%)", "3 графика (ZO, PFO, PSO)"],
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
                    df['А0'] = df.loc[df['А'] > 0, 'А'].iloc[0]
                if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                    df['А/А0'] = df['А'] / df['А0']
            except Exception as e: st.error(f"❌ Ошибка: {str(e)}")
    else:
        default_data = pd.DataFrame({'т, мин': [0.0, 5.0, 10.0, 15.0, 20.0], 'А': [1.0, 0.75, 0.55, 0.40, 0.30]})
        df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="photo_manual")
        if df is not None and not df.empty:
            df['А0'] = df.iloc[0]['А']
            df['А/А0'] = df['А'] / df['А0']

    if df is None or df.empty: return

    processed_df = preprocess_data(df)
    summary = get_data_summary(processed_df)

    st.markdown(section_header("sh-selected", "📊", "Сводка данных"), unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_box("mb-amber", "Точки данных", summary["total_points"]), unsafe_allow_html=True)
    col2.markdown(metric_box("mb-amber", "Время (мин)", f'{summary["time_range"][0]:.1f}-{summary["time_range"][1]:.1f}'), unsafe_allow_html=True)
    col3.markdown(metric_box("mb-amber", "Начальное А0", f'{summary["a0_value"]:.3f}'), unsafe_allow_html=True)
    col4.markdown(metric_box("mb-amber", "Диапазон А/А0", f'{summary["a_a0_range"][0]:.3f}-{summary["a_a0_range"][1]:.3f}'), unsafe_allow_html=True)

    stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
    selected_data = processed_df.iloc[stable_indices].copy()

    # آلية أمان لمنع خطأ الـ Coercion وطول المصفوفات المتباين
    try:
        k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(selected_data)
        k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
        k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)
        
        if len(zo_predictions) != len(selected_data): raise ValueError("Mismatch detected")
        _, _, rmse_zo = calculate_metrics(selected_data['А/А0'].values, zo_predictions)
        _, _, rmse_pfo = calculate_metrics(selected_data['А/А0'].values, pfo_predictions)
        _, _, rmse_pso = calculate_metrics(selected_data['А/А0'].values, pso_predictions)
    except Exception:
        selected_data = processed_df.copy()
        k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(selected_data)
        k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
        k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)
        _, _, rmse_zo = calculate_metrics(selected_data['А/А0'].values, zo_predictions)
        _, _, rmse_pfo = calculate_metrics(selected_data['А/А0'].values, pfo_predictions)
        _, _, rmse_pso = calculate_metrics(selected_data['А/А0'].values, pso_predictions)

    st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
    results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
    st.dataframe(results_summary, use_container_width=True)

    st.markdown(section_header("sh-compare", "⚖️", "Сравнение моделей"), unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(f'<div class="model-card mc-zo"><h3>Модель ZO</h3><p><strong>k₀:</strong> {abs(k0):.5f}</p><p><strong>R²:</strong> {r2_zo:.4f}</p><p><strong>MAPE:</strong> {mape_zo:.2f}%</p><p><strong>RMSE:</strong> {rmse_zo:.4f}%</p></div>', unsafe_allow_html=True)
    col_m2.markdown(f'<div class="model-card mc-pfo"><h3>Модель PFO</h3><p><strong>k₁:</strong> {abs(k1):.5f} мин⁻¹</p><p><strong>R²:</strong> {r2_pfo:.4f}</p><p><strong>MAPE:</strong> {mape_pfo:.2f}%</p><p><strong>RMSE:</strong> {rmse_pfo:.4f}%</p></div>', unsafe_allow_html=True)
    col_m3.markdown(f'<div class="model-card mc-pso"><h3>Модель PSO</h3><p><strong>k₂:</strong> {k2:.5f}</p><p><strong>R²:</strong> {r2_pso:.4f}</p><p><strong>MAPE:</strong> {mape_pso:.2f}%</p><p><strong>RMSE:</strong> {rmse_pso:.4f}%</p></div>', unsafe_allow_html=True)

    st.markdown(section_header("sh-viz", "📊", "Графика"), unsafe_allow_html=True)
    fig_main = create_matplotlib_plots(processed_df, selected_data, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
    
    # تصغير الجرافيكس الخارجي بشكل فوري وبسيط
    fig_main.set_size_inches(4.2, 2.4)
    for ax in fig_main.get_axes():
        ax.tick_params(axis='both', labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        if ax.get_legend(): plt.setp(ax.get_legend().get_texts(), fontsize=7)
    st.pyplot(fig_main)

    # مستطيل تحميل النتائج الموحد بآخر النموذج
    st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)
    with d_col1: st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(results_summary), file_name="photocatalysis_results.xlsx", use_container_width=True)
    with d_col2:
        png_b = BytesIO()
        fig_main.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Скачать графики (PNG)", data=png_b.getvalue(), file_name="photocatalysis_plots.png", mime="image/png", use_container_width=True)


# =============================================================================
# القسم 2: التحفيز المتجانس (النماذج الثلاثة مع الجرافيك المصغر والـ % لـ RMSE)
# =============================================================================
HOMO_MODEL_INFO = {
    "Power-law (степенной закон)": {"inputs": ["t", "CA", "CB", "r"], "outputs": ["k, α, β", "R², MAPE, RMSE (%)"]},
    "Arrhenius": {"inputs": ["T (K)", "k"], "outputs": ["A, Ea", "R², MAPE, RMSE (%)"]},
    "Последовательные реакции": {"inputs": ["t", "CA, CB, CC"], "outputs": ["k1, k2", "R², RMSE (%)", "Max CB"]}
}

def render_homogeneous():
    st.markdown("### 📈 Выберите кинетическую модель:")
    homo_model = st.radio("Кинетическая модель:", list(HOMO_MODEL_INFO.keys()), index=0, horizontal=True)
    sidebar_params(inputs=HOMO_MODEL_INFO[homo_model]["inputs"], outputs=HOMO_MODEL_INFO[homo_model]["outputs"], file_types=["Excel (.xlsx)", "CSV (.csv)"])

    st.markdown(section_header("sh-data", "📊", f"Ввод данных ({homo_model})"), unsafe_allow_html=True)
    method = input_method_choice(f"homo_{homo_model}")

    h_df = None
    if method == "📁 Загрузить файл":
        uploaded_h_file = st.file_uploader("Выберите файл Excel/CSV", type=['xlsx', 'csv'], key=f"file_{homo_model}")
        if uploaded_h_file is not None: h_df = handle_file_upload(uploaded_h_file, f"homo_{homo_model}")
    else:
        if homo_model == "Power-law (степенной закон)":
            empty_df = pd.DataFrame([[0.0, 1.0, 1.5, 0.05], [5.0, 0.8, 1.3, 0.035], [10.0, 0.6, 1.1, 0.022]], columns=['t', 'CA', 'CB', 'r'])
        elif homo_model == "Arrhenius":
            empty_df = pd.DataFrame([[298.0, 0.01], [308.0, 0.022], [318.0, 0.047]], columns=['T', 'k'])
        else:
            empty_df = pd.DataFrame([[0.0, 1.0, 0.0, 0.0], [5.0, 0.74, 0.23, 0.03], [10.0, 0.55, 0.35, 0.10]], columns=['t', 'CA', 'CB', 'CC'])
        h_df = st.data_editor(empty_df, use_container_width=True, num_rows="dynamic", key=f"ed_{homo_model}")

    if h_df is None or len(h_df) == 0: return
    h_df = clean_homogeneous_data(h_df)

    # --- 1) Power Law ---
    if homo_model == "Power-law (степенной закон)":
        if not all(c in h_df.columns for c in ['t', 'CA', 'CB', 'r']): return
        try:
            clean_df = h_df[(h_df['CA'] > 0) & (h_df['CB'] > 0) & (h_df['r'] > 0)]
            log_CA, log_CB, log_r = np.log(clean_df['CA'].values), np.log(clean_df['CB'].values), np.log(clean_df['r'].values)
            X = np.column_stack((np.ones_like(log_CA), log_CA, log_CB))
            beta_matrix, _, _, _ = np.linalg.lstsq(X, log_r, rcond=None)
            k_val, alpha_val, beta_val = np.exp(beta_matrix[0]), beta_matrix[1], beta_matrix[2]
            r_pred = k_val * (clean_df['CA'].values ** alpha_val) * (clean_df['CB'].values ** beta_val)
            r2, mape, rmse = calculate_metrics(clean_df['r'].values, r_pred)

            st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f'<div class="performance-metric">⚡ k = {k_val:.4f}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">α = {alpha_val:.2f}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">β = {beta_val:.2f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">R² = {r2:.4f}</div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="performance-metric">RMSE = {rmse:.4f}%</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4.2, 2.4))
            x_factor = (clean_df['CA'].values ** alpha_val) * (clean_df['CB'].values ** beta_val)
            ax.scatter(x_factor, clean_df['r'].values, color='#ef4444', s=15, label='Эксп.')
            ax.plot(x_factor, r_pred, color='#1e40af', linewidth=1.2, label='Модель')
            ax.set_xlabel('Фактор конц.', fontsize=8)
            ax.set_ylabel('Скорость r', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            res_df = pd.DataFrame({'Параметр': ['k', 'alpha', 'beta', 'R²', 'MAPE (%)', 'RMSE (%)'], 'Значение': [k_val, alpha_val, beta_val, r2, mape, f"{rmse:.4f}%"]})
            with d_col1: st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(res_df), file_name="power_law.xlsx", use_container_width=True)
            with d_col2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300)
                st.download_button("🖼️ Скачать графики (PNG)", data=png_b.getvalue(), file_name="power_law.png", use_container_width=True)
        except Exception as e: st.error(f"❌ Ошибка: {str(e)}")

    # --- 2) Arrhenius ---
    elif homo_model == "Arrhenius":
        if not all(c in h_df.columns for c in ['T', 'k']): return
        try:
            clean_df = h_df[(h_df['T'] > 0) & (h_df['k'] > 0)]
            inv_T, log_k = 1.0 / clean_df['T'].values, np.log(clean_df['k'].values)
            slope, intercept = np.polyfit(inv_T, log_k, 1)
            Ea_val, A_val = -slope * 8.314 / 1000.0, np.exp(intercept)
            k_pred = A_val * np.exp(-(Ea_val * 1000.0) / (8.314 * clean_df['T'].values))
            r2, mape, rmse = calculate_metrics(clean_df['k'].values, k_pred)

            st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="performance-metric">🧪 A = {A_val:.2e}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">Ea = {Ea_val:.2f} кДж</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">R² = {r2:.4f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">RMSE = {rmse:.4f}%</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4.2, 2.4))
            ax.scatter(inv_T, log_k, color='#ef4444', s=15, label='Эксп.')
            ax.plot(inv_T, slope * inv_T + intercept, color='#10b981', linewidth=1.2, label='Модель')
            ax.set_xlabel('1/T (1/K)', fontsize=8)
            ax.set_ylabel('ln(k)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            res_df = pd.DataFrame({'Параметр': ['A', 'Ea (кДж/моль)', 'R²', 'MAPE (%)', 'RMSE (%)'], 'Значение': [A_val, Ea_val, r2, mape, f"{rmse:.4f}%"]})
            with d_col1: st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(res_df), file_name="arrhenius.xlsx", use_container_width=True)
            with d_col2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300)
                st.download_button("🖼️ Скачать графики (PNG)", data=png_b.getvalue(), file_name="arrhenius.png", use_container_width=True)
        except Exception as e: st.error(f"❌ Ошибка: {str(e)}")

    # --- 3) Последовательные реакции ---
    elif homo_model == "Последовательные реакции":
        if not all(c in h_df.columns for c in ['t', 'CA', 'CB', 'CC']): return
        try:
            t_data, CA_data = h_df['t'].values, h_df['CA'].values
            popt1, _ = curve_fit(lambda t, k1: CA_data[0] * np.exp(-k1 * t), t_data, CA_data, p0=[0.05])
            k1_fit = popt1[0]

            def fit_B(t, k2): return (k1_fit * CA_data[0] / (k2 - k1_fit)) * (np.exp(-k1_fit * t) - np.exp(-k2 * t))
            popt2, _ = curve_fit(fit_B, t_data, h_df['CB'].values, p0=[0.02])
            k2_fit = popt2[0]

            CA_pred, CB_pred = CA_data[0] * np.exp(-k1_fit * t_data), fit_B(t_data, k2_fit)
            CC_pred = CA_data[0] - CA_pred - CB_pred

            r2_A, _, rmse_A = calculate_metrics(CA_data, CA_pred)
            r2_B, _, rmse_B = calculate_metrics(h_df['CB'].values, CB_pred)
            r2_C, _, rmse_C = calculate_metrics(h_df['CC'].values, CC_pred)
            total_r2 = (r2_A + r2_B + r2_C) / 3
            total_rmse = (rmse_A + rmse_B + rmse_C) / 3

            t_fine = np.linspace(t_data.min(), t_data.max(), 500)
            CB_fine = (k1_fit * CA_data[0] / (k2_fit - k1_fit)) * (np.exp(-k1_fit * t_fine) - np.exp(-k2_fit * t_fine))
            max_CB_val = CB_fine.max()
            t_max_CB = t_fine[CB_fine.argmax()]

            st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f'<div class="performance-metric">🟣 k₁ = {k1_fit:.4f}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">🔵 k₂ = {k2_fit:.4f}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">R² = {total_r2:.4f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">RMSE = {total_rmse:.4f}%</div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="performance-metric">🧪 Max CB = {max_CB_val:.3f} ({t_max_CB:.1f} мин)</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(4.2, 2.4))
            ax.plot(t_data, CA_data, 'o', color='#ef4444', markersize=3)
            ax.plot(t_fine, CA_data[0] * np.exp(-k1_fit * t_fine), '-', color='#ef4444', linewidth=1.2, label='A')
            ax.plot(t_data, h_df['CB'].values, 'o', color='#16a34a', markersize=3)
            ax.plot(t_fine, CB_fine, '-', color='#16a34a', linewidth=1.2, label='B (Промеж.)')
            ax.plot(t_data, h_df['CC'].values, 'o', color='#2563eb', markersize=3)
            ax.plot(t_fine, CA_data[0] - (CA_data[0] * np.exp(-k1_fit * t_fine)) - CB_fine, '-', color='#2563eb', linewidth=1.2, label='C')
            
            ax.set_xlabel('Время (t), мин', fontsize=8)
            ax.set_ylabel('Концентрация (C), моль/л', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc='best')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            d_col1, d_col2 = st.columns(2)
            res_df = pd.DataFrame({'Параметр': ['k1', 'k2', 'Общий R²', 'Общий RMSE (%)', 'Max CB', 'Время Max CB'], 'Значение': [k1_fit, k2_fit, total_r2, f"{total_rmse:.4f}%", max_CB_val, t_max_CB]})
            with d_col1: st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(res_df), file_name="consecutive.xlsx", use_container_width=True)
            with d_col2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300)
                st.download_button("🖼️ Скачать графики (PNG)", data=png_b.getvalue(), file_name="consecutive.png", use_container_width=True)
        except Exception as e: st.error(f"❌ Ошибка: {str(e)}")


def render_placeholder(section_name: str):
    st.info(f"Раздел «{section_name}» находится в разработке.")


def main():
    st.markdown('<div class="main-header-title"><h1>Анализ кинетического моделирования</h1></div><div class="main-header-authors"><p>АВТОР: Алсади К. &nbsp;|&nbsp; РУКОВОДИТЕЛЬ: Киреева А.В</p></div>', unsafe_allow_html=True)
    reaction_type = st.sidebar.selectbox("🛠️ Тип процесса / реакции", options=["Фотокаталитические реакции", "Гомогенный катализ", "Гетерогенный катализ", "Ферментативные реакции"], index=0)
    if reaction_type == "Фотокаталитические реакции": render_photocatalysis()
    elif reaction_type == "Гомогенный катализ": render_homogeneous()
    else: render_placeholder(reaction_type)

if __name__ == "__main__": main()
