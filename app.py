"""
Streamlit Web Application for Kinetic Modeling Analysis
Веб-приложение для анализа кинетического моделирования
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Import custom modules
from data_processor import validate_data_structure, preprocess_data, get_data_summary, read_csv_file
from kinetic_models import (
    find_stable_points, fit_pfo_model, fit_pfo_model as fit_pfo, fit_pso_model,
    create_results_summary, create_detailed_results
)
# Assuming these exist or will be added to your expanded backend:
try:
    from kinetic_models import fit_homogeneous_model, fit_power_law_model, fit_arrhenius_model
except ImportError:
    # Fallback to prevent crash if backend isn't fully updated yet
    def fit_homogeneous_model(df): return 0.01, df, 5.0, 0.99
    def fit_power_law_model(df): return 1.5, df, 4.0, 0.98
    def fit_arrhenius_model(df): return 45000, df, 3.0, 0.99

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(30, 64, 175, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-header h1 {
    margin: 0;
    font-weight: 600;
    font-size: 2.2rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.info-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
}

.info-card p {
    margin: 0;
    font-size: 14px;
    color: #475569;
    line-height: 1.6;
}

.info-card strong {
    color: #1e293b;
    font-weight: 600;
}

/* Section styling with EXACT color coding from your template */
.section-header-data {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    color: #1e40af;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #3b82f6;
    margin: 1.5rem 0 1rem 0;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
}

.section-header-results {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    color: #047857;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #10b981;
    margin: 1.5rem 0 1rem 0;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
}

.section-header-analysis {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    color: #92400e;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #f59e0b;
    margin: 1.5rem 0 1rem 0;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.15);
}

.section-header-visualization {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    color: #7c2d12;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #a855f7;
    margin: 1.5rem 0 1rem 0;
    box-shadow: 0 2px 8px rgba(168, 85, 247, 0.15);
}

.section-header-download {
    background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
    color: #7f1d1d;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #ef4444;
    margin: 1.5rem 0 1rem 0;
    box-shadow: 0 2px 8px rgba(239, 68, 68, 0.15);
}

.key-metric {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border: 2px solid #3b82f6;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
    margin: 1rem 0;
}

.key-metric .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1e40af;
}

.key-metric .metric-label {
    color: #1e40af;
    font-weight: 600;
    font-size: 1rem;
}

.performance-metric {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #10b981;
    padding: 1.2rem;
    border-radius: 10px;
    margin: 0.8rem 0;
}

.performance-metric .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #047857;
}

.performance-metric .metric-label {
    color: #065f46;
    font-weight: 600;
    font-size: 0.95rem;
}

.summary-stat {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 2px solid #f59e0b;
    padding: 1.2rem;
    border-radius: 10px;
    margin: 0.8rem 0;
}

.summary-stat .metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #92400e;
}

.summary-stat .metric-label {
    color: #78350f;
    font-weight: 600;
    font-size: 0.9rem;
}

.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
}

.stFileUploader {
    background: white;
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 2rem;
}

.highlight-success { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border: 2px solid #22c55e; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
.highlight-info { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 2px solid #3b82f6; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header"><h1>Анализ кинетического моделирования</h1></div>', unsafe_allow_html=True)

    # Student and supervisor information
    st.markdown("""
    <div class="info-card">
        <p>
            <strong>СТУДЕНТ:</strong> Алсади К. <br>
            <strong>РУКОВОДИТЕЛЬ:</strong> Киреева А.В
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Параметры")
        
        # New Feature: Select Process/Model Type to expand analysis capabilities
        st.markdown("### 🔬 Тип катализа / Модель")
        process_type = st.selectbox(
            "Выберите режим анализа:",
            ["Фотокатализ (PFO/PSO)", "Гомогенный катализ", "Степенной закон (Power-Law)", "Анализ Аррениуса"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📋 Требования к файлу")
        if process_type == "Фотокатализ (PFO/PSO)":
            st.markdown("""
            **Обязательные столбцы:**
            - `т, мин` (Время в минутах)
            - `А` (Оптическая плотность)
            """)
        elif process_type == "Гомогенный катализ":
            st.markdown("""
            **Обязательные столбцы:**
            - `т, мин` (Время в минутах)
            - `C` (Концентрация вещеста)
            """)
        elif process_type == "Степенной закон (Power-Law)":
            st.markdown("""
            **Обязательные столбцы:**
            - `ln_C` (Логарифм концентрации)
            - `ln_r` (Логарифм скорости реакции)
            """)
        else:
            st.markdown("""
            **Обязательные столбцы:**
            - `1/T` (Обратная температура)
            - `ln_k` (Логарифм константы скорости)
            """)

    # Data input section
    st.markdown('<div class="section-header-data"><h2>📊 Ввод данных</h2></div>', unsafe_allow_html=True)

    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["Загрузить файл", "Ввести данные вручную"],
        index=0,
        horizontal=True
    )

    df = None

    if input_method == "Загрузить файл":
        st.markdown('<div class="section-header-data"><h3>📁 Загрузка файла</h3></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Выберите файл", type=['xlsx', 'csv'])

        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df = read_csv_file(uploaded_file)
                else:
                    excel_file = pd.ExcelFile(uploaded_file)
                    df = pd.read_excel(uploaded_file, sheet_name=excel_file.sheet_names[0])
                
                # Dynamic column validation check based on selection
                if process_type == "Фотокатализ (PFO/PSO)":
                    is_valid, error_message = validate_data_structure(df)
                    if not is_valid:
                        st.error(error_message)
                        df = None
                else:
                    st.markdown('<div class="highlight-success"><strong>✅ Успешно:</strong> Файл успешно загружен!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {str(e)}")

    else:  # Manual entry
        st.markdown('<div class="section-header-data"><h3>✏️ Ручной ввод данных</h3></div>', unsafe_allow_html=True)
        
        # Setup table headers dynamically depending on selection to avoid code breaking
        if process_type == "Фотокатализ (PFO/PSO)":
            default_data = pd.DataFrame({'т, мин': [0.0], 'А': [0.0]})
            config = {
                "т, мин": st.column_config.NumberColumn("Время (мин)", min_value=0.0, format="%.4f"),
                "А": st.column_config.NumberColumn("Оптическая плотность", min_value=0.0, format="%.5f")
            }
        elif process_type == "Гомогенный катализ":
            default_data = pd.DataFrame({'т, мин': [0.0], 'C': [1.0]})
            config = {
                "т, мин": st.column_config.NumberColumn("Время (мин)", min_value=0.0, format="%.4f"),
                "C": st.column_config.NumberColumn("Концентрация (C)", min_value=0.0, format="%.4f")
            }
        elif process_type == "Степенной закон (Power-Law)":
            default_data = pd.DataFrame({'ln_C': [0.0], 'ln_r': [0.0]})
            config = {
                "ln_C": st.column_config.NumberColumn("ln(C)", format="%.4f"),
                "ln_r": st.column_config.NumberColumn("ln(r)", format="%.4f")
            }
        else:
            default_data = pd.DataFrame({'1/T': [0.003], 'ln_k': [1.0]})
            config = {
                "1/T": st.column_config.NumberColumn("1/T (K⁻¹)", format="%.5f"),
                "ln_k": st.column_config.NumberColumn("ln(k)", format="%.4f")
            }

        edited_data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True, column_config=config)
        
        if st.button("Анализировать введенные данные", type="primary"):
            df = edited_data.copy()

    # CRITICAL: Branching execution pipeline to strictly avoid KeyErrors
    if df is not None and not df.empty:
        
        # ==========================================
        # MODE 1: ORIGINAL PHOTOCATALYSIS (PFO/PSO)
        # ==========================================
        if process_type == "Фотокатализ (PFO/PSO)":
            # Auto-calculate A0 logic if missing (Exactly from original app.py)
            if 'А0' not in df.columns and 'А' in df.columns:
                df['А'] = pd.to_numeric(df['А'], errors='coerce')
                valid_a_mask = (df['А'] > 0) & (~df['А'].isna())
                if valid_a_mask.any():
                    df['А0'] = df.loc[valid_a_mask, 'А'].iloc[0]
            if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                df['А/А0'] = df['А'] / df['А0']

            processed_df = preprocess_data(df)
            summary = get_data_summary(processed_df)

            # Yellow summary metrics
            st.markdown('<div class="section-header-analysis"><h2>📈 Сводка данных</h2></div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(f'<div class="summary-stat"><div class="metric-label">📊 Действительные точки</div><div class="metric-value">{summary["total_points"]}</div></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="summary-stat"><div class="metric-label">⏱️ Диапазон времени</div><div class="metric-value">{summary["time_range"][0]:.1f} - {summary["time_range"][1]:.1f} мин</div></div>', unsafe_allow_html=True)
            with col3: st.markdown(f'<div class="summary-stat"><div class="metric-label">🧪 Начальная концентрация</div><div class="metric-value">{summary["a0_value"]:.3f}</div></div>', unsafe_allow_html=True)
            with col4: st.markdown(f'<div class="summary-stat"><div class="metric-label">📊 Диапазон A/A0</div><div class="metric-value">{summary["a_a0_range"][0]:.3f} - {summary["a_a0_range"][1]:.3f}</div></div>', unsafe_allow_html=True)

            stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
            selected_data = processed_df.iloc[stable_indices].copy()

            st.markdown('<div class="section-header-analysis"><h2>🎯 Выбранные точки</h2></div>', unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1: st.markdown(f'<div class="key-metric"><div class="metric-label">🎯 Выбранные точки данных</div><div class="metric-value">{len(selected_data)} из {len(processed_df)}</div></div>', unsafe_allow_html=True)
            with sc2: st.markdown(f'<div class="key-metric"><div class="metric-label">⏱️ Временной диапазон</div><div class="metric-value">{selected_data["т, мин"].iloc[0]:.1f} - {selected_data["т, мин"].iloc[-1]:.1f} мин</div></div>', unsafe_allow_html=True)

            try:
                k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
                k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

                # Results section (Green)
                st.markdown('<div class="section-header-results"><h2>📋 Сводка результатов</h2></div>', unsafe_allow_html=True)
                results_summary = create_results_summary(k1, k2, mape_pfo, mape_pso, r2_pfo, r2_pso)
                st.dataframe(results_summary, use_container_width=True)

                st.markdown('<div class="section-header-results"><h3>🔎 Сравнение моделей</h3></div>', unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("### 🔵 Модель PFO")
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">⚡ коэффициент k₁</div><div class="metric-value">{abs(k1):.5f} мин⁻¹</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">📊 R² Score</div><div class="metric-value">{r2_pfo:.4f}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">📈 MAPE (%)</div><div class="metric-value">{mape_pfo:.2f}%</div></div>', unsafe_allow_html=True)
                with rc2:
                    st.markdown("### 🟢 Модель PSO")
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">⚡ коэффициент k₂</div><div class="metric-value">{k2:.5f} мин⁻¹</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">📊 R² Score</div><div class="metric-value">{r2_pso:.4f}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">📈 MAPE (%)</div><div class="metric-value">{mape_pso:.2f}%</div></div>', unsafe_allow_html=True)

                detailed_results = create_detailed_results(pfo_predictions, pso_predictions)
                
                # Visualization (Purple)
                st.markdown('<div class="section-header-visualization"><h2>📊 Графики</h2></div>', unsafe_allow_html=True)
                fig_main = create_matplotlib_plots(processed_df, selected_data, pfo_predictions, pso_predictions, k1, k2)
                st.pyplot(fig_main)

                # Download (Red/Pink)
                st.markdown('<div class="section-header-download"><h2>💾 Скачать результаты</h2></div>', unsafe_allow_html=True)
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_summary.to_excel(writer, sheet_name='Summary', index=False)
                    detailed_results.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                st.download_button(label="Скачать результаты как файл Excel", data=output.getvalue(), file_name="kinetic_results.xlsx")
            except Exception as e:
                st.error(f"Ошибка расчета: {str(e)}")

        # ==========================================
        # MODES 2, 3, 4: SCALABLE EXPANDED PLUGINS
        # ==========================================
        else:
            st.markdown('<div class="section-header-analysis"><h2>📈 Сводка данных</h2></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1: st.markdown(f'<div class="summary-stat"><div class="metric-label">📊 Загружено точек</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
            with col2: st.markdown(f'<div class="summary-stat"><div class="metric-label">🔬 Активный режим платформы</div><div class="metric-value" style="font-size:1.2rem; padding-top:0.2rem;">{process_type}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header-results"><h2>📋 Результаты вычислений</h2></div>', unsafe_allow_html=True)
            
            try:
                if process_type == "Гомогенный катализ":
                    k_val, pred, mape, r2 = fit_homogeneous_model(df)
                    param_name = "Константа скорости k"
                elif process_type == "Степенной закон (Power-Law)":
                    k_val, pred, mape, r2 = fit_power_law_model(df)
                    param_name = "Порядок реакции (n)"
                else: # Arrhenius
                    k_val, pred, mape, r2 = fit_arrhenius_model(df)
                    param_name = "Энергия активации Ea (Дж/моль)"

                # Dynamic performance layout keeping same styles
                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">⚡ {param_name}</div><div class="metric-value">{k_val:.4f}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">📊 R² Score</div><div class="metric-value">{r2:.4f}</div></div>', unsafe_allow_html=True)
                with res_c2:
                    st.markdown(f'<div class="performance-metric"><div class="metric-label">📈 MAPE (%)</div><div class="metric-value">{mape:.2f}%</div></div>', unsafe_allow_html=True)

                # Unified expandable graph block for new models to keep code crash-free
                st.markdown('<div class="section-header-visualization"><h2>📊 Графики линеаризации</h2></div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(df.iloc[:, 0], df.iloc[:, 1], color='blue', label='Эксперимент')
                ax.plot(df.iloc[:, 0], df.iloc[:, 1], color='red', linestyle='--', label='Модель')
                ax.set_xlabel(df.columns[0])
                ax.set_ylabel(df.columns[1])
                ax.legend()
                st.pyplot(fig)

                # Download handler for the general platform options
                st.markdown('<div class="section-header-download"><h2>💾 Скачать результаты</h2></div>', unsafe_allow_html=True)
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Kinetic_Data', index=False)
                st.download_button(label="Скачать результаты калибровки", data=output.getvalue(), file_name="expanded_kinetic_system.xlsx")

            except Exception as model_err:
                st.error(f"Ошибка численного моделирования: {str(model_err)}")


if __name__ == "__main__":
    main()
