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
    create_results_summary, create_detailed_results
)
from visualization import create_matplotlib_plots

# Configure page
st.set_page_config(
    page_title="Анализ кинетического моделирования",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling for scientific/industrial application
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&family=JetBrains+Mono:wght=400;500&display=swap');

/* Global background and fonts */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main header styling */
.main-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(30, 64, 175, 0.15);
}

.main-header h1 {
    margin: 0;
    font-weight: 600;
    font-size: 2.2rem;
    color: white !important;
}

/* Info card styling */
.info-card {
    background: white;
    padding: 1.2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
}

/* Data summary statistics */
.summary-stat {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #f59e0b;
    padding: 1.2rem;
    border-radius: 10px;
    margin: 0.8rem 0;
    color: #92400e !important;
}

.summary-stat h3 {
    margin: 0.3rem 0 0 0;
    color: #78350f !important;
}

/* Model performance metric cards */
.performance-metric {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #10b981;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: #047857 !important;
    font-weight: 600;
}

/* Distinct section headers styling */
.section-header-data { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); color: #1e40af; padding: 1rem; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 1rem; }
.section-header-results { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); color: #047857; padding: 1rem; border-radius: 10px; border-left: 5px solid #10b981; margin-bottom: 1rem; }
.section-header-analysis { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color: #92400e; padding: 1rem; border-radius: 10px; border-left: 5px solid #f59e0b; margin-bottom: 1rem; }
.section-header-visualization { background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); color: #7c3aed; padding: 1rem; border-radius: 10px; border-left: 5px solid #a855f7; margin-bottom: 1rem; }
.section-header-download { background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%); color: #7f1d1d; padding: 1rem; border-radius: 10px; border-left: 5px solid #ef4444; margin-bottom: 1rem; }

.section-header-data h2, .section-header-results h2, .section-header-analysis h2, .section-header-visualization h2, .section-header-download h2 {
    color: inherit !important;
    margin: 0;
}

.highlight-success { background: #dcfce7; border: 1px solid #22c55e; border-radius: 12px; padding: 1rem; margin: 1rem 0; color: #14532d; }

.model-title {
    color: #1e293b !important;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Target specific streamlit widgets safely to prevent white-on-white text in MAIN page */
div[data-testid="stRadio"] label p,
div[data-testid="stSelectbox"] label p,
div[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"],
div[data-testid="stMarkdownContainer"] h3 {
    color: #1e293b !important;
    font-weight: 500 !important;
}

/* Sidebar styling for Dark Backgrounds */
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] li, 
section[data-testid="stSidebar"] span, 
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2 {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3 {
    color: #fde68a !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


def calculate_metrics(y_true, y_pred):
    """Helper function to calculate R2 and RMSE"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return r2, rmse


def main():
    st.markdown("""
    <div class="main-header">
        <h1>Кинетическое моделирование каталитических процессов</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <p style="margin:0; font-size:14px; color:#475569;">
            <strong>СТУДЕНТ:</strong> Алсади К. &nbsp;&nbsp;|&nbsp;&nbsp; 
            <strong>РУКОВОДИТЕЛЬ:</strong> Киреева А.В
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # Выбор типа процесса / реакции
    # =========================================================================
    st.markdown("### 🛠️ Выберите тип процесса / реакции:")
    reaction_type = st.selectbox(
        label="Тип химического процесса для анализа:",
        options=[
            "Фотокаталитические реакции",
            "Гомогенный катализ",
            "Гетерогенный катализ",
            "Ферментативные реакции"
        ],
        index=0
    )
    
    st.markdown("---")

    # =========================================================================
    # Раздел 1: Фотокаталитические реакции
    # =========================================================================
    if reaction_type == "Фотокаталитические реакции":
        with st.sidebar:
            st.header("⚙️ Параметры процесса")
            st.info("Тип: Фотокатализ")
            st.markdown("---")
            st.markdown("### 📋 Требования к файлу")
            st.markdown("""
            **Обязательные столбцы:**
            - `т, мин` (Время в минутах)
            - `А` (Оптическая плотность)

            **Опциональные столбцы:**
            - `А0` (Начальная плотность)
            - `А/А0`
            - `ln А/А0`
            - `1/А`
            """)

        st.markdown("""
        <div class="section-header-data">
            <h2>📊 Ввод данных (Фотокатализ)</h2>
        </div>
        """, unsafe_allow_html=True)

        input_method = st.radio(
            "Выберите способ ввода данных:",
            ["Загрузить файл", "Ввести данные вручную"],
            index=0, horizontal=True
        )

        df = None
        if input_method == "Загрузить файл":
            uploaded_file = st.file_uploader("Выберите файл", type=['xlsx', 'csv'], key="photo_upload")
            if uploaded_file is not None:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'csv':
                        df = read_csv_file(uploaded_file)
                    else:
                        excel_file = pd.ExcelFile(uploaded_file)
                        sheet_names = excel_file.sheet_names
                        selected_sheet = sheet_names[0]
                        if len(sheet_names) > 1:
                            selected_sheet = st.sidebar.selectbox("Выберите лист Excel", sheet_names, index=0)
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

                    is_valid, error_message = validate_data_structure(df)
                    if not is_valid:
                        st.error(f"{error_message}")
                        return

                    if 'А0' not in df.columns and 'А' in df.columns and len(df) > 0:
                        df['А'] = pd.to_numeric(df['А'], errors='coerce')
                        valid_a_mask = (df['А'] > 0) & (~df['А'].isna())
                        if valid_a_mask.any():
                            df['А0'] = df.loc[valid_a_mask, 'А'].iloc[0]
                            st.markdown(f'<div class="highlight-success">✅ Автоопределение: А0 = {df["А0"].iloc[0]:.5f}</div>', unsafe_allow_html=True)

                    if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                        df['А/А0'] = df['А'] / df['А0']

                    with st.expander("Предварительный просмотр данных"):
                        st.dataframe(df, use_container_width=True)

                    edited_df = st.data_editor(df[['т, мин', 'А']].copy(), use_container_width=True, num_rows="dynamic", key="upload_ed")
                    if 'А' in edited_df.columns and len(edited_df) > 0 and edited_df.iloc[0]['А'] > 0:
                        edited_df['А0'] = edited_df.iloc[0]['А']
                        edited_df = edited_df[(edited_df['А'] > 0) & (edited_df['т, мин'] >= 0)]
                        if not edited_df.empty:
                            edited_df['А/А0'] = edited_df['А'] / edited_df['А0']
                            df = edited_df.copy()
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
        else:
            default_data = pd.DataFrame({'т, мин': [0.0], 'А': [0.0]})
            edited_data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, key="manual_ed")
            if not edited_data.empty and 'А' in edited_data.columns and len(edited_data) > 0 and edited_data.iloc[0]['А'] > 0:
                edited_data['А0'] = edited_data.iloc[0]['А']
                valid_data = edited_data[(edited_data['А'] > 0) & (edited_data['т, мин'] >= 0)].copy()
                if not valid_data.empty:
                    valid_data['А/А0'] = valid_data['А'] / valid_data['А0']
                    df = valid_data.copy()
                    st.dataframe(df[['т, мин', 'А', 'А/А0']], use_container_width=True)

        if df is not None and not df.empty:
            processed_df = preprocess_data(df)
            summary = get_data_summary(processed_df)
            
            st.markdown('<div class="section-header-analysis"><h2>📈 Сводка данных</h2></div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f'<div class="summary-stat">📊 Точки<br><h3>{summary["total_points"]}</h3></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="summary-stat">⏱️ Время<br><h3>{summary["time_range"][0]:.1f}-{summary["time_range"][1]:.1f} мин</h3></div>', unsafe_allow_html=True)
            col3.markdown(f'<div class="summary-stat">🧪 А0<br><h3>{summary["a0_value"]:.3f}</h3></div>', unsafe_allow_html=True)
            col4.markdown(f'<div class="summary-stat">📊 Диапазон A/A0<br><h3>{summary["a_a0_range"][0]:.3f}-{summary["a_a0_range"][1]:.3f}</h3></div>', unsafe_allow_html=True)

            stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
            selected_data = processed_df.iloc[stable_indices].copy()

            try:
                k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(selected_data)
                k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
                k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

                st.markdown('<div class="section-header-results"><h2>📋 Сводка результатов</h2></div>', unsafe_allow_html=True)
                results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
                st.dataframe(results_summary, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<p class="model-title">🟣 Модель ZO</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric">⚡ k₀ = {abs(k0):.5f}</div><div class="performance-metric">📊 R² = {r2_zo:.4f}</div><div class="performance-metric">📈 MAPE = {mape_zo:.2f}%</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<p class="model-title">🔵 Модель PFO</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric">⚡ k₁ = {abs(k1):.5f}</div><div class="performance-metric">📊 R² = {r2_pfo:.4f}</div><div class="performance-metric">📈 MAPE = {mape_pfo:.2f}%</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<p class="model-title">🟢 Модель PSO</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric">⚡ k₂ = {k2:.5f}</div><div class="performance-metric">📊 R² = {r2_pso:.4f}</div><div class="performance-metric">📈 MAPE = {mape_pso:.2f}%</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header-visualization"><h2>📊 Графики</h2></div>', unsafe_allow_html=True)
                fig_main = create_matplotlib_plots(processed_df, selected_data, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
                st.pyplot(fig_main)

                png_buffer = BytesIO()
                fig_main.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
                png_buffer.seek(0)
                st.download_button(label="📥 Скачать графики (PNG)", data=png_buffer, file_name="kinetic_plots.png", mime="image/png")

            except Exception as e:
                st.error(f"Ошибка моделирования: {str(e)}")

    # =========================================================================
    # Раздел 2: ГОМОГЕННЫЙ КАТАЛИЗ (Activated & Enhanced)
    # =========================================================================
    elif reaction_type == "Гомогенный катализ":
        # اختيار الموديل الفرعي للتحفيز المتجانس
        st.markdown("### 📈 Выберите кинетическую модель гомогенного катализа:")
        homo_model = st.radio(
            "Кинетическая модель:",
            ["Power-law (степенной закон)", "Arrhenius", "Последовательные реакции"],
            index=0, horizontal=True
        )

        # تحضير وتحديث الـ Sidebar ديناميكياً بحسب الموديل المختار
        with st.sidebar:
            st.header("⚙️ Параметры процесса")
            st.info("Тип: Гомогенный катализ")
            st.markdown("---")
            st.markdown("### 📋 Требования к файлу")
            
            if homo_model == "Power-law (степенной закон)":
                st.markdown("""
                **Входные данные:**
                - `t` — время реакции
                - `CA` — концентрация реагента A
                - `CB` — концентрация реагента B
                - `r` — скорость реакции
                """)
            elif homo_model == "Arrhenius":
                st.markdown("""
                **Входные данные:**
                - `T` — температура (K)
                - `k` — константа скорости
                """)
            elif homo_model == "Последовательные реакции":
                st.markdown("""
                **Входные данные:**
                - `t` — время реакции
                - `CA` — концентрация вещества A
                - `CB` — концентрация вещества B
                - `CC` — концентрация вещества C
                """)

        st.markdown(f"""
        <div class="section-header-data">
            <h2>📊 Ввод данных и расчет ({homo_model})</h2>
        </div>
        """, unsafe_allow_html=True)

        # 1. СТЕПЕННОЙ ЗАКОН (Power-law)
        if homo_model == "Power-law (степенной закон)":
            default_homo = pd.DataFrame({
                't': [0, 5, 10, 15, 20],
                'CA': [1.0, 0.8, 0.65, 0.50, 0.40],
                'CB': [1.5, 1.3, 1.15, 1.00, 0.90],
                'r': [0.050, 0.035, 0.024, 0.015, 0.010]
            })
            st.markdown("**Редактировать входные данные таблицы:**")
            h_df = st.data_editor(default_homo, use_container_width=True, num_rows="dynamic", key="pl_ed")
            
            if st.button("🚀 Выполнить кинетический расчет (Power-law)"):
                try:
                    # Fit: ln(r) = ln(k) + alpha*ln(CA) + beta*ln(CB)
                    log_CA = np.log(h_df['CA'].values)
                    log_CB = np.log(h_df['CB'].values)
                    log_r = np.log(h_df['r'].values)
                    
                    X = np.column_stack((np.ones_like(log_CA), log_CA, log_CB))
                    beta_matrix, _, _, _ = np.linalg.lstsq(X, log_r, rcond=None)
                    
                    k_val = np.exp(beta_matrix[0])
                    alpha_val = beta_matrix[1]
                    beta_val = beta_matrix[2]
                    
                    r_pred = k_val * (h_df['CA'].values**alpha_val) * (h_df['CB'].values**beta_val)
                    r2, rmse = calculate_metrics(h_df['r'].values, r_pred)
                    
                    # عرض النتائج والمعاملات الخارجية بدقة
                    st.markdown('<div class="section-header-results"><h2>📋 Определяемые параметры (Выходные данные)</h2></div>', unsafe_allow_html=True)
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.markdown(f'<div class="performance-metric">⚡ k = {k_val:.4f}</div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="performance-metric">🔸 α (по A) = {alpha_val:.2f}</div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="performance-metric">🔹 β (по B) = {beta_val:.2f}</div>', unsafe_allow_html=True)
                    c4.markdown(f'<div class="performance-metric">📊 R² = {r2:.4f}</div>', unsafe_allow_html=True)
                    c5.markdown(f'<div class="performance-metric">📈 RMSE = {rmse:.5f}</div>', unsafe_allow_html=True)
                    
                    # رسم البياني الخاص بـ Power-law
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(h_df['t'], h_df['r'], 'ro-', label='Эксперимент (r)')
                    ax.plot(h_df['t'], r_pred, 'b--', label='Модель (Power-law)')
                    ax.set_xlabel('Время (t)')
                    ax.set_ylabel('Скорость реакции (r)')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Ошибка расчета: {str(e)}")

        # 2. ARRHENIUS MODEL
        elif homo_model == "Arrhenius":
            default_arr = pd.DataFrame({
                'T': [298, 308, 318, 328, 338],
                'k': [0.012, 0.034, 0.089, 0.215, 0.485]
            })
            st.markdown("**Редактировать входные данные таблицы:**")
            h_df = st.data_editor(default_arr, use_container_width=True, num_rows="dynamic", key="arr_ed")
            
            if st.button("🚀 Выполнить кинетический расчет (Arrhenius)"):
                try:
                    # Fit: ln(k) = ln(A) - Ea / (R * T)
                    R = 8.314
                    inv_T = 1.0 / h_df['T'].values
                    log_k = np.log(h_df['k'].values)
                    
                    slope, intercept = np.polyfit(inv_T, log_k, 1)
                    Ea_val = -slope * R / 1000.0  # kJ/mol
                    A_val = np.exp(intercept)
                    
                    k_pred = A_val * np.exp(- (Ea_val * 1000.0) / (R * h_df['T'].values))
                    r2, rmse = calculate_metrics(h_df['k'].values, k_pred)
                    
                    st.markdown('<div class="section-header-results"><h2>📋 Определяемые параметры (Выходные данные)</h2></div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.markdown(f'<div class="performance-metric">🧪 A = {A_val:.2e}</div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="performance-metric">🔥 Ea = {Ea_val:.2f} кДж/моль</div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="performance-metric">📊 R² = {r2:.4f}</div>', unsafe_allow_html=True)
                    c4.markdown(f'<div class="performance-metric">📈 RMSE = {rmse:.5f}</div>', unsafe_allow_html=True)
                    
                    # رسم البياني الخاص بـ Arrhenius
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.scatter(inv_T, log_k, color='red', label='Эксперимент')
                    ax.plot(inv_T, slope*inv_T + intercept, 'g-', label='Линейная аппроксимация')
                    ax.set_xlabel('1/T (1/K)')
                    ax.set_ylabel('ln(k)')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Ошибка расчета: {str(e)}")

        # 3. ПОСЛЕДОВАТЕЛЬНЫЕ РЕАКЦИИ (Consecutive Reactions)
        elif homo_model == "Последовательные реакции":
            default_cons = pd.DataFrame({
                't': [0, 10, 20, 30, 40, 50],
                'CA': [1.00, 0.67, 0.45, 0.30, 0.20, 0.13],
                'CB': [0.00, 0.28, 0.41, 0.45, 0.44, 0.41],
                'CC': [0.00, 0.05, 0.14, 0.25, 0.36, 0.46]
            })
            st.markdown("**Редактировать входные данные таблицы:**")
            h_df = st.data_editor(default_cons, use_container_width=True, num_rows="dynamic", key="cons_ed")
            
            if st.button("🚀 Выполнить кинетический расчет (Последовательные реакции)"):
                try:
                    t_data = h_df['t'].values
                    CA_data = h_df['CA'].values
                    
                    # Fit CA to find k1: CA = CA0 * exp(-k1 * t)
                    def fit_A(t, k1):
                        return CA_data[0] * np.exp(-k1 * t)
                    
                    popt1, _ = curve_fit(fit_A, t_data, CA_data, p0=[0.05])
                    k1_fit = popt1[0]
                    
                    # Fit CB to find k2
                    def fit_B(t, k2):
                        return (k1_fit * CA_data[0] / (k2 - k1_fit)) * (np.exp(-k1_fit * t) - np.exp(-k2 * t))
                    
                    popt2, _ = curve_fit(fit_B, t_data, h_df['CB'].values, p0=[0.02])
                    k2_fit = popt2[0]
                    
                    CA_pred = fit_A(t_data, k1_fit)
                    CB_pred = fit_B(t_data, k2_fit)
                    CC_pred = CA_data[0] - CA_pred - CB_pred
                    
                    r2_A, _ = calculate_metrics(CA_data, CA_pred)
                    r2_B, _ = calculate_metrics(h_df['CB'].values, CB_pred)
                    total_r2 = (r2_A + r2_B) / 2.0
                    
                    st.markdown('<div class="section-header-results"><h2>📋 Определяемые параметры (Выходные данные)</h2></div>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f'<div class="performance-metric">🟣 k₁ (Стадия 1) = {k1_fit:.4f}</div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="performance-metric">🔵 k₂ (Стадия 2) = {k2_fit:.4f}</div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="performance-metric">📊 Средний R² = {total_r2:.4f}</div>', unsafe_allow_html=True)
                    
                    # رسم مخطط Concentrations vs Time للمركبات الثلاثة
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(t_data, CA_data, 'ro', label='A (Эксп)')
                    ax.plot(t_data, CA_pred, 'r-', label='A (Модель)')
                    ax.plot(t_data, h_df['CB'].values, 'go', label='B (Эксп)')
                    ax.plot(t_data, CB_pred, 'g-', label='B (Модель)')
                    ax.plot(t_data, h_df['CC'].values, 'bo', label='C (Эксп)')
                    ax.plot(t_data, CC_pred, 'b-', label='C (Модель)')
                    ax.set_xlabel('Время (t)')
                    ax.set_ylabel('Концентрация (C)')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Ошибка расчета: {str(e)}")

    # =========================================================================
    # Остальные разделы
    # =========================================================================
    elif reaction_type == "Гетерогенный катализ":
        st.info("Раздел 'Гетерогенный катализ' (Модели Ленгмюра-Хиншельвуда и др.)")
        
    elif reaction_type == "Ферментативные реакции":
        st.info("Раздел 'Ферментативные реакции' (Модели Михаэлиса-Ментен, Коэффициент Хилла)")


if __name__ == "__main__":
    main()
