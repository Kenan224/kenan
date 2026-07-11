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
# CSS -- تصميم موحد بتباين ثابت (لا نص أبيض على خلفية بيضا، ولا نص غامق على خلفية غامقة)
# =============================================================================
st.markdown("""
<style>
/* خلفية عامة فاتحة موحدة */
.stApp {
    background-color: #f5f7fa;
}

/* إجبار كل نص افتراضي (عناوين، فقرات، labels) على لون غامق مقروء دائماً */
html, body, [class*="css"], p, span, label, .stMarkdown, .stRadio label, .stSelectbox label,
.stTextInput label, .stNumberInput label, .stDataFrame, .stFileUploader label {
    color: #1e293b;
}

/* الهيدر الرئيسي -- خلفية غامقة + نص أبيض (تباين صحيح) */
.main-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    padding: 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.2);
    text-align: center;
}
.main-header h1 {
    margin: 0;
    font-weight: 600;
    font-size: 2rem;
    color: #ffffff !important;
}
.main-header p { color: #dbeafe !important; margin: 0.3rem 0 0 0; }

/* كرت معلومات -- خلفية بيضاء + نص غامق دائماً */
.info-card {
    background: #ffffff;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e2e8f0;
    color: #1e293b !important;
}
.info-card * { color: #1e293b !important; }

/* صناديق المقاييس -- خلفية فاتحة + نص غامق من نفس العائلة اللونية (تباين مضمون) */
.metric-box {
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    border: 1px solid;
}
.metric-box h4 { margin: 0; font-size: 0.85rem; font-weight: 600; }
.metric-box h2 { margin: 0.3rem 0 0 0; font-size: 1.4rem; font-weight: 700; }

.mb-amber { background: #fffbeb; border-color: #fde68a; }
.mb-amber h4, .mb-amber h2 { color: #92400e !important; }

.mb-blue { background: #eff6ff; border-color: #bfdbfe; }
.mb-blue h4, .mb-blue h2 { color: #1e40af !important; }

.mb-green { background: #ecfdf5; border-color: #a7f3d0; }
.mb-green h4, .mb-green h2 { color: #065f46 !important; }

.mb-rose { background: #fff1f2; border-color: #fecdd3; }
.mb-rose h4, .mb-rose h2 { color: #9f1239 !important; }

/* شارة نتيجة صغيرة (المعاملات، R2، MAPE) */
.performance-metric {
    background: #f0fdf4;
    border: 1px solid #10b981;
    padding: 0.7rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    color: #065f46 !important;
    font-weight: 600;
    text-align: center;
}

/* عناوين الأقسام -- خلفية فاتحة ونص غامق دائماً */
.section-header {
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    border-left: 5px solid;
    margin-bottom: 1rem;
}
.section-header h2 { margin: 0; font-size: 1.25rem; }

.sh-data { background: #eff6ff; border-left-color: #2563eb; }
.sh-data h2 { color: #1e3a8a !important; }

.sh-results { background: #ecfdf5; border-left-color: #10b981; }
.sh-results h2 { color: #065f46 !important; }

.sh-selected { background: #fff1f2; border-left-color: #f43f5e; }
.sh-selected h2 { color: #9f1239 !important; }

.sh-compare { background: #f0fdfa; border-left-color: #0d9488; }
.sh-compare h2 { color: #115e59 !important; }

.sh-viz { background: #faf5ff; border-left-color: #a855f7; }
.sh-viz h2 { color: #6b21a8 !important; }

.sh-download { background: #fef2f2; border-left-color: #ef4444; margin-top: 1.2rem; }
.sh-download h2 { color: #991b1b !important; }

.highlight-success {
    background: #dcfce7;
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    color: #14532d !important;
}

/* كرت موديل صغير (مقارنة الموديلات) -- كل كرت بلون واضح ونص غامق مطابق */
.model-card {
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid;
}
.model-card h3 { margin: 0 0 0.5rem 0; }
.model-card p { margin: 0.25rem 0; }

.mc-zo   { background: #f8fafc; border-left-color: #94a3b8; }
.mc-zo   h3, .mc-zo   p { color: #334155 !important; }

.mc-pfo  { background: #eff6ff; border-left-color: #2563eb; }
.mc-pfo  h3, .mc-pfo  p { color: #1e3a8a !important; }

.mc-pso  { background: #f0fdf4; border-left-color: #16a34a; }
.mc-pso  h3, .mc-pso  p { color: #14532d !important; }

/* الشريط الجانبي */
section[data-testid="stSidebar"] { background-color: #ffffff; }
section[data-testid="stSidebar"] * { color: #1e293b !important; }
.sidebar-params-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e3a8a !important;
    margin: 0.6rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #2563eb;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# دوال مساعدة عامة
# =============================================================================
def calculate_metrics(y_true, y_pred):
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
    """
    يعرض بالهامش (بعد اختيار نوع التفاعل): عنوان ПАРАМЕТРЫ ثم المدخلات ثم النواتج ثم أنواع الملفات.
    """
    st.sidebar.markdown('<div class="sidebar-params-title">ПАРАМЕТРЫ</div>', unsafe_allow_html=True)

    st.sidebar.markdown("**📥 Входные данные (столбцы):**")
    for item in inputs:
        st.sidebar.markdown(f"- {item}")

    st.sidebar.markdown("**📤 Выходные данные:**")
    for item in outputs:
        st.sidebar.markdown(f"- {item}")

    st.sidebar.markdown("**📁 Поддерживаемые файлы:**")
    st.sidebar.markdown(", ".join(file_types))


def handle_file_upload(uploaded_file, key_prefix: str):
    """
    يقرأ ملف (Excel أو CSV). إذا كان Excel وفيه أكثر من صفحة، يظهر اختيار الصفحة
    بالهامش (فقط بهاي الحالة). يرجع DataFrame أو None.
    """
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
    """اختيار إجباري: تحميل ملف أو إدخال يدوي -- إلزامي بكل الأقسام."""
    return st.radio(
        "طريقة إدخال البيانات / Способ ввода данных:",
        ["📁 Загрузить файл", "✏️ Ввести данные вручную"],
        index=0, horizontal=True, key=f"input_method_{key_prefix}"
    )


# =============================================================================
# قسم 1: الفوتوكاتاليز -- بدون اختيار موديل (ZO + PFO + PSO مع بعض)
# =============================================================================
def render_photocatalysis():
    sidebar_params(
        inputs=["т, мин (время)", "А (оптическая плотность)", "А0 (опционально)", "А/А0 (опционально)"],
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
                        st.markdown(
                            f'<div class="highlight-success">✅ Автоопределение: А0 = {df["А0"].iloc[0]:.5f}</div>',
                            unsafe_allow_html=True)

                if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                    df['А/А0'] = df['А'] / df['А0']

                with st.expander("👁️ Предварительный просмотр данных"):
                    st.dataframe(df, use_container_width=True)

                edited_df = st.data_editor(df[['т, мин', 'А']].copy(), use_container_width=True,
                                            num_rows="dynamic", key="photo_upload_ed")
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
        st.warning("⚠️ لا توجد بيانات صالحة للتحليل بعد المعالجة.")
        return
    summary = get_data_summary(processed_df)

    st.markdown(section_header("sh-selected", "📊", "Сводка данных"), unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_box("mb-amber", "Действительные точки", summary["total_points"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_box("mb-amber", "Диапазон времени",
                                f'{summary["time_range"][0]:.1f}-{summary["time_range"][1]:.1f} мин'),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(metric_box("mb-amber", "Начальная концентрация", f'{summary["a0_value"]:.3f}'),
                    unsafe_allow_html=True)
    with col4:
        st.markdown(metric_box("mb-amber", "Диапазон А/А0",
                                f'{summary["a_a0_range"][0]:.3f}-{summary["a_a0_range"][1]:.3f}'),
                    unsafe_allow_html=True)

    stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
    selected_data = processed_df.iloc[stable_indices].copy()

    st.markdown(section_header("sh-selected", "📌", "Выбранные точки"), unsafe_allow_html=True)
    col_pts1, col_pts2 = st.columns(2)
    with col_pts1:
        st.markdown(metric_box("mb-blue", "Выбранные точки данных", f'{len(selected_data)} из {len(processed_df)}'),
                    unsafe_allow_html=True)
    with col_pts2:
        if not selected_data.empty:
            st.markdown(metric_box("mb-blue", "Временной диапазон",
                                    f'{selected_data["т, мин"].min():.1f} - {selected_data["т, мин"].max():.1f} мин'),
                        unsafe_allow_html=True)
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
            st.markdown(f"""
            <div class="model-card mc-zo">
                <h3>Модель ZO</h3>
                <p><strong>k₀:</strong> {abs(k0):.5f}</p>
                <p><strong>R²:</strong> {r2_zo:.4f}</p>
                <p><strong>MAPE:</strong> {mape_zo:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="model-card mc-pfo">
                <h3>Модель PFO</h3>
                <p><strong>k₁:</strong> {abs(k1):.5f} мин⁻¹</p>
                <p><strong>R²:</strong> {r2_pfo:.4f}</p>
                <p><strong>MAPE:</strong> {mape_pfo:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="model-card mc-pso">
                <h3>Модель PSO</h3>
                <p><strong>k₂:</strong> {k2:.5f} л/(мг·мин)</p>
                <p><strong>R²:</strong> {r2_pso:.4f}</p>
                <p><strong>MAPE:</strong> {mape_pso:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(section_header("sh-viz", "📊", "Графика"), unsafe_allow_html=True)
        fig_main = create_matplotlib_plots(processed_df, selected_data, zo_predictions, pfo_predictions,
                                           pso_predictions, k0, k1, k2)
        st.pyplot(fig_main)

        st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(results_summary),
                                file_name="kinetic_analysis_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)
        with d_col2:
            png_b = BytesIO()
            fig_main.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
            png_b.seek(0)
            st.download_button("🖼️ Скачать графики (PNG)", data=png_b, file_name="kinetic_plots.png",
                                mime="image/png", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Ошибка моделирования: {str(e)}")


# =============================================================================
# قسم 2: التحفيز المتجانس -- لازم اختيار موديل
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
        "outputs": ["k1, k2 — константы скорости", "R², MAPE", "профиль концентраций"],
    },
}


def render_homogeneous():
    st.markdown("### 📈 Выберите кинетическую модель:")
    homo_model = st.radio(
        "Кинетическая модель:",
        list(HOMO_MODEL_INFO.keys()),
        index=0, horizontal=True, key="homo_model_choice"
    )

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
            cols_pl = ['t', 'CA', 'CB', 'r']
            data_pl = [[0.0, 1.0, 1.5, 0.05], [5.0, 0.8, 1.3, 0.035]]
            empty_df = pd.DataFrame(columns=cols_pl, data=data_pl)
        elif homo_model == "Arrhenius":
            cols_arr = ['T', 'k']
            data_arr = [[298.0, 0.01], [308.0, 0.02]]
            empty_df = pd.DataFrame(columns=cols_arr, data=data_arr)
        else:
            cols_seq = ['t', 'CA', 'CB', 'CC']
            data_seq = [[0.0, 1.0, 0.0, 0.0], [10.0, 0.5, 0.4, 0.1]]
            empty_df = pd.DataFrame(columns=cols_seq, data=data_seq)

        st.markdown("**Заполните таблицу данными:**")
        h_df = st.data_editor(empty_df, use_container_width=True, num_rows="dynamic", key=f"editor_{homo_model}")

    if h_df is None or len(h_df) == 0:
        return

    h_df = clean_homogeneous_data(h_df)

    # ---------------- 1. Power-law ----------------
    if homo_model == "Power-law (степенной закон)":
        required_cols = ['t', 'CA', 'CB', 'r']
        missing_cols = [c for c in required_cols if c not in h_df.columns]
        if missing_cols:
            st.error(f"❌ **Ошибка структуры!** Не найдены столбцы: `{missing_cols}`.")
            return
        try:
            valid_mask = (h_df['CA'] > 0) & (h_df['CB'] > 0) & (h_df['r'] > 0)
            clean_df = h_df[valid_mask]
            if clean_df.empty:
                st.warning("⚠️ لا توجد بيانات صالحة.")
                return

            log_CA = np.log(clean_df['CA'].values)
            log_CB = np.log(clean_df['CB'].values)
            log_r = np.log(clean_df['r'].values)

            X = np.column_stack((np.ones_like(log_CA), log_CA, log_CB))
            beta_matrix, _, _, _ = np.linalg.lstsq(X, log_r, rcond=None)

            k_val = np.exp(beta_matrix[0])
            alpha_val = beta_matrix[1]
            beta_val = beta_matrix[2]

            r_pred = k_val * (clean_df['CA'].values ** alpha_val) * (clean_df['CB'].values ** beta_val)
            r2, mape = calculate_metrics(clean_df['r'].values, r_pred)

            st.markdown(section_header("sh-results", "📋", "Сводка результатов (Power-law)"), unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f'<div class="performance-metric">⚡ k = {k_val:.4f}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">🔸 α = {alpha_val:.2f}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">🔹 β = {beta_val:.2f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">📊 R² = {r2:.4f}</div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="performance-metric">📈 MAPE = {mape:.2f}%</div>', unsafe_allow_html=True)

            st.markdown(section_header("sh-viz", "📊", "График линейной зависимости скорости"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            x_linear = (clean_df['CA'].values ** alpha_val) * (clean_df['CB'].values ** beta_val)
            ax.scatter(x_linear, clean_df['r'].values, color='#ef4444', s=45, zorder=3, label='Эксперимент')
            ax.plot(x_linear, r_pred, color='#1e40af', linestyle='-', linewidth=2, label=f'Модель (k = {k_val:.4f})')
            ax.set_xlabel(r'Фактор концентраций ($C_A^\alpha \cdot C_B^\beta$)')
            ax.set_ylabel('Скорость реакции (r)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                res_df = pd.DataFrame({'Параметр': ['k', 'alpha', 'beta', 'R2', 'MAPE (%)'],
                                        'Значение': [k_val, alpha_val, beta_val, r2, mape]})
                st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(res_df),
                                    file_name="power_law_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)
            with b2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
                png_b.seek(0)
                st.download_button("🖼️ Скачать график (PNG)", data=png_b, file_name="power_law_plot.png",
                                    mime="image/png", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Ошибка расчета: {str(e)}")

    # ---------------- 2. Arrhenius ----------------
    elif homo_model == "Arrhenius":
        required_cols = ['T', 'k']
        missing_cols = [c for c in required_cols if c not in h_df.columns]
        if missing_cols:
            st.error(f"❌ **Ошибка в структуре данных!** Отсутствуют столбцы: `{missing_cols}`.")
            return
        try:
            valid_mask = (h_df['T'] > 0) & (h_df['k'] > 0)
            clean_df = h_df[valid_mask]
            if clean_df.empty:
                st.warning("⚠️ لا توجد بيانات صالحة.")
                return

            R = 8.314
            inv_T = 1.0 / clean_df['T'].values
            log_k = np.log(clean_df['k'].values)

            slope, intercept = np.polyfit(inv_T, log_k, 1)
            Ea_val = -slope * R / 1000.0
            A_val = np.exp(intercept)

            k_pred = A_val * np.exp(-(Ea_val * 1000.0) / (R * clean_df['T'].values))
            r2, mape = calculate_metrics(clean_df['k'].values, k_pred)

            st.markdown(section_header("sh-results", "📋", "Сводка результатов (Arrhenius)"), unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="performance-metric">🧪 A = {A_val:.2e}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">🔥 Ea = {Ea_val:.2f} кДж/моль</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">📊 R² = {r2:.4f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">📈 MAPE = {mape:.2f}%</div>', unsafe_allow_html=True)

            st.markdown(section_header("sh-viz", "📊", "График Аррениуса"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.scatter(inv_T, log_k, color='#ef4444', s=45, zorder=3, label='Эксперимент')
            ax.plot(inv_T, slope * inv_T + intercept, color='#10b981', linewidth=2, label='Линейная аппроксимация')
            ax.set_xlabel('1/T (1/K)')
            ax.set_ylabel('ln(k)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                res_df = pd.DataFrame({'Параметр': ['A', 'Ea (kJ/mol)', 'R2', 'MAPE (%)'],
                                        'Значение': [A_val, Ea_val, r2, mape]})
                st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(res_df),
                                    file_name="arrhenius_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)
            with b2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
                png_b.seek(0)
                st.download_button("🖼️ Скачать график (PNG)", data=png_b, file_name="arrhenius_plot.png",
                                    mime="image/png", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Ошибка расчета: {str(e)}")

    # ---------------- 3. Последовательные реакции ----------------
    elif homo_model == "Последовательные реакции":
        required_cols = ['t', 'CA', 'CB', 'CC']
        missing_cols = [c for c in required_cols if c not in h_df.columns]
        if missing_cols:
            st.error(f"❌ **Ошибка в структуре данных!** Отсутствуют столбцы: `{missing_cols}`")
            return
        try:
            t_data = h_df['t'].values
            CA_data = h_df['CA'].values

            def fit_A(t, k1):
                return CA_data[0] * np.exp(-k1 * t)

            popt1, _ = curve_fit(fit_A, t_data, CA_data, p0=[0.05])
            k1_fit = popt1[0]

            def fit_B(t, k2):
                term1 = k1_fit * CA_data[0] / (k2 - k1_fit)
                term2 = np.exp(-k1_fit * t) - np.exp(-k2 * t)
                return term1 * term2

            popt2, _ = curve_fit(fit_B, t_data, h_df['CB'].values, p0=[0.02])
            k2_fit = popt2[0]

            CA_pred = fit_A(t_data, k1_fit)
            CB_pred = fit_B(t_data, k2_fit)
            CC_pred = CA_data[0] - CA_pred - CB_pred

            r2_A, mape_A = calculate_metrics(CA_data, CA_pred)
            r2_B, mape_B = calculate_metrics(h_df['CB'].values, CB_pred)
            total_r2 = (r2_A + r2_B) / 2.0
            total_mape = (mape_A + mape_B) / 2.0

            st.markdown(section_header("sh-results", "📋", "Сводка результатов (Последовательные реакции)"),
                        unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="performance-metric">🟣 k₁ = {k1_fit:.4f}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="performance-metric">🔵 k₂ = {k2_fit:.4f}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="performance-metric">📊 Средний R² = {total_r2:.4f}</div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="performance-metric">📈 Средний MAPE = {total_mape:.2f}%</div>', unsafe_allow_html=True)

            st.markdown(section_header("sh-viz", "📊", "Кинетический профиль концентраций"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.plot(t_data, CA_data, 'o', color='#ef4444', label='A (Эксп)')
            ax.plot(t_data, CA_pred, '-', color='#ef4444', linewidth=2, label='A (Модель)')
            ax.plot(t_data, h_df['CB'].values, 'o', color='#16a34a', label='B (Эксп)')
            ax.plot(t_data, CB_pred, '-', color='#16a34a', linewidth=2, label='B (Модель)')
            ax.plot(t_data, h_df['CC'].values, 'o', color='#2563eb', label='C (Эксп)')
            ax.plot(t_data, CC_pred, '-', color='#2563eb', linewidth=2, label='C (Модель)')
            ax.set_xlabel('Время (t)')
            ax.set_ylabel('Концентрация (C)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

            st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                res_df = pd.DataFrame({'Параметр': ['k1', 'k2', 'Average R2', 'Average MAPE (%)'],
                                        'Значение': [k1_fit, k2_fit, total_r2, total_mape]})
                st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(res_df),
                                    file_name="consecutive_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)
            with b2:
                png_b = BytesIO()
                fig.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
                png_b.seek(0)
                st.download_button("🖼️ Скачать график (PNG)", data=png_b, file_name="consecutive_plot.png",
                                    mime="image/png", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Ошибка расчета: {str(e)}")


# =============================================================================
# قسمين لسا ما اشتغلنا عليهم (محفوظين متل ما كانوا)
# =============================================================================
def render_placeholder(section_name: str):
    sidebar_params(
        inputs=["— قيد التطوير —"],
        outputs=["— قيد التطوير —"],
        file_types=["Excel (.xlsx)", "CSV (.csv)"]
    )
    st.markdown(section_header("sh-data", "🚧", f"{section_name}"), unsafe_allow_html=True)
    st.info(f"قسم «{section_name}» لسا قيد التطوير. رح نشتغل عليه لاحقاً.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>Кинетическое моделирование каталитических процессов</h1>
        <p>СТУДЕНТ: Алсади К. &nbsp;|&nbsp; РУКОВОДИТЕЛЬ: Киреева А.В</p>
    </div>
    """, unsafe_allow_html=True)

    # 1) اختيار نوع التفاعل -- بالهامش، أول عنصر
    reaction_type = st.sidebar.selectbox(
        "🛠️ Тип процесса / реакции",
        options=[
            "Фотокаталитические реакции",
            "Гомогенный катализ",
            "Гетерогенный катализ",
            "Ферментативные реакции",
        ],
        index=0,
        key="reaction_type_choice"
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
