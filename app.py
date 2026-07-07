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
/* Import professional fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&family=JetBrains+Mono:wght=400;500&display=swap');

/* Global styling */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Header styling */
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

/* Info card styling */
.info-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.info-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
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

/* Section styling */
.section-container {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.section-container:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 2px solid #e2e8f0;
}

/* Metric cards styling */
.metric-container {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

/* Enhanced metric styling */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    border-color: #3b82f6;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1e40af;
}

/* Special metric styling for key results */
.key-metric {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border: 2px solid #3b82f6;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
    margin: 1rem 0;
    transition: all 0.3s ease;
}

/* Performance metric styling */
.performance-metric {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #10b981;
    padding: 1.2rem;
    border-radius: 10px;
    margin: 0.8rem 0;
    box-shadow: 0 3px 12px rgba(16, 185, 129, 0.15);
    transition: all 0.3s ease;
}

.performance-metric:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(16, 185, 129, 0.25);
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

/* Summary statistics styling */
.summary-stat {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 2px solid #f59e0b;
    padding: 1.2rem;
    border-radius: 10px;
    margin: 0.8rem 0;
    box-shadow: 0 3px 12px rgba(245, 158, 11, 0.15);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
}

/* Download button special styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
}

/* Enhanced section headers with color coding */
.section-header-data {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    color: #1e40af;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #3b82f6;
    margin: 1.5rem 0 1rem 0;
}

.section-header-results {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    color: #047857;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #10b981;
    margin: 1.5rem 0 1rem 0;
}

.section-header-analysis {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    color: #92400e;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #f59e0b;
    margin: 1.5rem 0 1rem 0;
}

.section-header-visualization {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
    color: #7c2d12;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #a855f7;
    margin: 1.5rem 0 1rem 0;
}

.section-header-download {
    background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
    color: #7f1d1d;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #ef4444;
    margin: 1.5rem 0 1rem 0;
}

.highlight-success {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.highlight-info {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

code {
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    background: #f1f5f9;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
    <div class="main-header">
        <h1>Анализ кинетического моделирования</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <p>
            <strong>СТУДЕНТ:</strong> Алсади К. <br>
            <strong>РУКОВОДИТЕЛЬ:</strong> Киреева А.В
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Параметры")
        st.markdown("---")
        st.markdown("### 📋 Требования к файлу")
        st.markdown("""
        **Обязательные столбцы:**
        - `т, мин` (Время в минутах)
        - `А` (Оптическая плотность)

        **Опциональные столбцы:**
        - `А0` (Начальная Оптическая плотность)
        - `А/А0` (Отношение A/A0)
        """)

    st.markdown("""
    <div class="section-header-data">
        <h2>📊 Ввод данных</h2>
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["Загрузить файл", "Ввести данные вручную"],
        index=0,
        horizontal=True
    )

    df = None
    selected_sheet = None

    if input_method == "Загрузить файл":
        st.markdown("""
        <div class="section-header-data">
            <h3>📁 Загрузка файла</h3>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=['xlsx', 'csv']
        )

        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    df = read_csv_file(uploaded_file)
                    if df.empty:
                        st.error("Файл CSV пуст")
                        return
                else:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names

                    if len(sheet_names) > 1:
                        with st.sidebar:
                            st.markdown("---")
                            selected_sheet = st.selectbox(
                                "Выберите лист Excel",
                                sheet_names,
                                index=0
                            )
                    else:
                        selected_sheet = sheet_names[0]

                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    if df.empty:
                        st.error(f"Лист '{selected_sheet}' пуст")
                        return

                is_valid, error_message = validate_data_structure(df)
                if not is_valid:
                    st.error(f"{error_message}")
                    return

                auto_a0_calculated = False
                if 'А0' not in df.columns:
                    if 'А' in df.columns and len(df) > 0:
                        df['А'] = pd.to_numeric(df['А'], errors='coerce')
                        valid_a_mask = (df['А'] > 0) & (~df['А'].isna())
                        if valid_a_mask.any():
                            first_a_value = df.loc[valid_a_mask, 'А'].iloc[0]
                            df['А0'] = first_a_value
                            auto_a0_calculated = True
                            st.markdown(f"""
                            <div class="highlight-success">
                                <strong>✅ Автоопределение:</strong> А0 = {first_a_value:.5f} (из первого значения А)
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Не удалось определить А0")
                            return

                if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                    df['А/А0'] = df['А'] / df['А0']

                st.markdown("""
                <div class="highlight-success">
                    <strong>✅ Успешно:</strong> Файл успешно загружен!
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Предварительный просмотр данных"):
                    st.dataframe(df, use_container_width=True)

                st.subheader("Редактирование данных")
                
                if 'uploaded_first_a_value' not in st.session_state:
                    st.session_state.uploaded_first_a_value = None

                edit_df_display = df[['т, мин', 'А']].copy()
                edited_df = st.data_editor(
                    edit_df_display,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="uploaded_data_editor"
                )

                if 'А' in edited_df.columns and len(edited_df) > 0 and edited_df.iloc[0]['А'] > 0:
                    first_a_value = edited_df.iloc[0]['А']
                    edited_df['А0'] = first_a_value

                    valid_mask = (edited_df['А'] > 0) & (edited_df['т, мин'] >= 0)
                    edited_df = edited_df[valid_mask]

                    if not edited_df.empty:
                        edited_df['А/А0'] = edited_df['А'] / edited_df['А0']
                        df = edited_df.copy()

            except Exception as e:
                st.error(f"Ошибка чтения файла: {str(e)}")
    else:
        st.markdown("""
        <div class="section-header-data">
            <h3>✏️ Ручной ввод данных</h3>
        </div>
        """, unsafe_allow_html=True)

        if 'first_a_value' not in st.session_state:
            st.session_state.first_a_value = None

        default_data = pd.DataFrame({'т, мин': [0.0], 'А': [0.0]})
        edited_data = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_data_editor"
        )

        if not edited_data.empty and 'А' in edited_data.columns and len(edited_data) > 0 and edited_data.iloc[0]['А'] > 0:
            first_a_value = edited_data.iloc[0]['А']
            edited_data['А0'] = first_a_value
            valid_data = edited_data[(edited_data['А'] > 0) & (edited_data['т, мин'] >= 0)].copy()
            
            if not valid_data.empty:
                valid_data['А/А0'] = valid_data['А'] / valid_data['А0']
                st.subheader("Предварительный просмотр с рассчитанными А/А0")
                st.dataframe(valid_data[['т, мин', 'А', 'А/А0']], use_container_width=True)

        if st.button("Анализировать введенные данные", type="primary"):
            if not edited_data.empty and 'А' in edited_data.columns and edited_data.iloc[0]['А'] > 0:
                first_a_value = edited_data.iloc[0]['А']
                edited_data['А0'] = first_a_value
                valid_data = edited_data[(edited_data['А'] > 0) & (edited_data['т, мин'] >= 0)].copy()
                if not valid_data.empty:
                    valid_data['А/А0'] = valid_data['А'] / valid_data['А0']
                    df = valid_data.copy()
                    st.success("Данные успешно введены!")

    if df is not None and not df.empty:
        processed_df = preprocess_data(df)
        if len(processed_df) == 0:
            st.error("Нет действительных данных после обработки")
            return

        summary = get_data_summary(processed_df)
        st.markdown("""
        <div class="section-header-analysis">
            <h2>📈 Сводка данных</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="summary-stat"><div class="metric-label">📊 Точки</div><div class="metric-value">{summary["total_points"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="summary-stat"><div class="metric-label">⏱️ Время</div><div class="metric-value">{summary["time_range"][0]:.1f}-{summary["time_range"][1]:.1f} мин</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="summary-stat"><div class="metric-label">🧪 А0</div><div class="metric-value">{summary["a0_value"]:.3f}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="summary-stat"><div class="metric-label">📊 Диапазон A/A0</div><div class="metric-value">{summary["a_a0_range"][0]:.3f}-{summary["a_a0_range"][1]:.3f}</div></div>', unsafe_allow_html=True)

        stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
        selected_data = processed_df.iloc[stable_indices].copy()

        try:
            k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(selected_data)
            k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
            k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

            st.markdown("""
            <div class="section-header-results">
                <h2>📋 Сводка результатов</h2>
            </div>
            """, unsafe_allow_html=True)

            results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
            st.dataframe(results_summary, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 🟣 Модель ZO")
                st.markdown(f'<div class="performance-metric"><div class="metric-label">⚡ k₀</div><div class="metric-value">{abs(k0):.5f}</div></div><div class="performance-metric"><div class="metric-label">📊 R²</div><div class="metric-value">{r2_zo:.4f}</div></div><div class="performance-metric"><div class="metric-label">📈 MAPE</div><div class="metric-value">{mape_zo:.2f}%</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown("### 🔵 Модель PFO")
                st.markdown(f'<div class="performance-metric"><div class="metric-label">⚡ k₁</div><div class="metric-value">{abs(k1):.5f}</div></div><div class="performance-metric"><div class="metric-label">📊 R²</div><div class="metric-value">{r2_pfo:.4f}</div></div><div class="performance-metric"><div class="metric-label">📈 MAPE</div><div class="metric-value">{mape_pfo:.2f}%</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown("### 🟢 Модель PSO")
                st.markdown(f'<div class="performance-metric"><div class="metric-label">⚡ k₂</div><div class="metric-value">{k2:.5f}</div></div><div class="performance-metric"><div class="metric-label">📊 R²</div><div class="metric-value">{r2_pso:.4f}</div></div><div class="performance-metric"><div class="metric-label">📈 MAPE</div><div class="metric-value">{mape_pso:.2f}%</div></div>', unsafe_allow_html=True)

            with st.expander("Подробные результаты"):
                detailed_results = create_detailed_results(zo_predictions, pfo_predictions, pso_predictions)
                st.dataframe(detailed_results, use_container_width=True)

            st.markdown("""
            <div class="section-header-visualization">
                <h2>📊 Графики</h2>
            </div>
            """, unsafe_allow_html=True)

            fig_main = create_matplotlib_plots(processed_df, selected_data, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
            st.pyplot(fig_main)
       # ========================================================
            # أولاً: زر تحميل الجرافيك   
            png_buffer = BytesIO()
            fig_main.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
            png_buffer.seek(0)
            
            st.download_button(
                label="📥 Скачать графики в формате PNG",
                data=png_buffer,
                file_name="kinetic_plots_300dpi.png",
                mime="image/png"
            )
            # ========================================================

            # ثانياً: الكود الخاص بك لتحميل ملف الـ Excel
            st.markdown("""
            <div class="section-header-download">
                <h2>💾 Скачать результаты</h2>
            </div>
            """, unsafe_allow_html=True)

            download_data = {
                'Summary': results_summary,
                'Detailed_Results': detailed_results,
                'Selected_Data': selected_data,
                'ZO_Predictions': zo_predictions,
                'PFO_Predictions': pfo_predictions,
                'PSO_Predictions': pso_predictions
            }

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, data in download_data.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)

            st.download_button(
                label="Скачать результаты как файл Excel",
                data=output.getvalue(),
                file_name="kinetic_modeling_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Ошибка моделирования: {str(e)}")


if __name__ == "__main__":
    main()
