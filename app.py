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
    find_stable_points, fit_pfo_model, fit_pso_model,
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global styling */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
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

.css-1d391kg .css-1v0mbdj {
    border-radius: 8px;
    margin-bottom: 1rem;
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

.metric-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
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

[data-testid="metric-container"] [data-testid="metric-label"] {
    color: #64748b;
    font-weight: 500;
    font-size: 0.9rem;
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

.key-metric:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(59, 130, 246, 0.3);
}

.key-metric .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1e40af;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.key-metric .metric-label {
    color: #1e40af;
    font-weight: 600;
    font-size: 1rem;
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
    transition: all 0.3s ease;
}

.summary-stat:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(245, 158, 11, 0.25);
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

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Download button special styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3);
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(5, 150, 105, 0.4);
    background: linear-gradient(135deg, #047857 0%, #065f46 100%);
}

/* File uploader styling */
.stFileUploader {
    background: white;
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 2rem;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #3b82f6;
    background: #f8fafc;
}

.stFileUploader [data-testid="stFileUploaderDropzone"] {
    border-radius: 8px;
}

/* Data editor styling */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
}

/* Radio button styling */
.stRadio > div {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Selectbox styling */
.stSelectbox > div > div {
    background: white;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-radius: 8px;
    border: 1px solid #cbd5e1;
    font-weight: 600;
    color: #1e293b;
}

.streamlit-expanderContent {
    background: white;
    border: 1px solid #e2e8f0;
    border-top: none;
    border-radius: 0 0 8px 8px;
}

/* Alert styling */
.stAlert {
    border-radius: 8px;
    border: none;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* Success alert */
.stAlert[data-baseweb="notification"] {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border-left: 4px solid #22c55e;
}

/* Info alert */
.stInfo {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border-left: 4px solid #3b82f6;
}

/* Error alert */
.stError {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-left: 4px solid #ef4444;
}

/* Warning alert */
.stWarning {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
}

/* Divider styling */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
    margin: 2rem 0;
}

/* Typography improvements */
h1, h2, h3, h4, h5, h6 {
    color: #1e293b;
    font-weight: 600;
    line-height: 1.4;
}

h1 {
    font-size: 2.2rem;
    margin-bottom: 1rem;
}

h2 {
    font-size: 1.8rem;
    margin-bottom: 1rem;
    color: #334155;
}

h3 {
    font-size: 1.4rem;
    margin-bottom: 0.8rem;
    color: #475569;
}

/* Enhanced section headers with color coding */
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

/* Highlighted info boxes */
.highlight-info {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border: 2px solid #3b82f6;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
}

.highlight-success {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2);
}

.highlight-warning {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 2px solid #f59e0b;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.2);
}

/* Code and monospace styling */
code {
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    background: #f1f5f9;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-size: 0.9rem;
}

/* Responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .main-header {
        padding: 1.5rem;
    }

    .main-header h1 {
        font-size: 1.8rem;
    }

    .section-container {
        padding: 1.5rem;
    }
}

/* Animation for loading states */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.stApp > div {
    animation: fadeIn 0.5s ease-out;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Professional header with title
    st.markdown("""
    <div class="main-header">
        <h1>Анализ кинетического моделирования</h1>
    </div>
    """, unsafe_allow_html=True)

    # Student and supervisor information with enhanced styling
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

        st.markdown("---")
        st.markdown("### 📋 Требования к файлу")
        st.markdown("""
        **Обязательные столбцы:**
        - `т, мин` (Время в минутах)
        - `А` (Оптическая плотность)

        **Опциональные столбцы:**
        - `А0` (Начальная Оптическая плотность) - если отсутствует, используется первое значение А
        - `А/А0` (Отношение A/A0) - рассчитывается автоматически если отсутствует

        **Поддерживаемые форматы:**
        - Excel (.xlsx,)
        - CSV (.csv) с автоопределением разделителя
        """)

    # Data input method selection with enhanced styling and emoji
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
        # File upload section with enhanced styling and emoji
        st.markdown("""
        <div class="section-header-data">
            <h3>📁 Загрузка файла</h3>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Выберите файл",
            type=['xlsx', 'csv'],
            help="Загрузите файл Excel или CSV с кинетическими данными"
        )

        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == 'csv':
                    # Handle CSV files
                    df = read_csv_file(uploaded_file)

                    # Check if file is empty
                    if df.empty:
                        st.error("Файл CSV пуст")
                        return

                else:
                    # Handle Excel files
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names

                    # Add sheet selector to sidebar if multiple sheets exist
                    selected_sheet = None
                    if len(sheet_names) > 1:
                        with st.sidebar:
                            st.markdown("---")
                            selected_sheet = st.selectbox(
                                "Выберите лист Excel",
                                sheet_names,
                                index=0,
                                help="Выберите лист для анализа из доступных в файле"
                            )
                    else:
                        selected_sheet = sheet_names[0]

                    # Read the selected sheet
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

                        # Check if sheet is empty
                        if df.empty:
                            st.error(f"Лист '{selected_sheet}' пуст")
                            if len(sheet_names) > 1:
                                st.error("Попробуйте выбрать другой лист.")
                            return

                    except Exception as sheet_error:
                        st.error(f"Ошибка чтения листа '{selected_sheet}': {str(sheet_error)}")
                        if len(sheet_names) > 1:
                            st.error("Попробуйте выбрать другой лист.")
                        return

                # Validate structure (common for both CSV and Excel)
                is_valid, error_message = validate_data_structure(df)

                if not is_valid:
                    st.error(f"{error_message}")
                    if file_extension != 'csv' and len(sheet_names) > 1:
                        st.error(f"Лист '{selected_sheet}' не содержит требуемых данных. Попробуйте выбрать другой лист.")
                    return

                # Check if А0 column is missing and auto-calculate if needed
                auto_a0_calculated = False
                if 'А0' not in df.columns:
                    if 'А' in df.columns and len(df) > 0:
                        # Convert А column to numeric first
                        df['А'] = pd.to_numeric(df['А'], errors='coerce')

                        # Find the first valid А value
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
                            st.markdown("""
                            <div class="highlight-info">
                                <strong>ℹ️ Информация:</strong> Столбец А0 отсутствовал в файле и был автоматически рассчитан из первого значения концентрации А
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Не удалось определить А0: нет действительных значений в столбце А")
                            return
                    else:
                        st.error("Не удалось определить А0: столбец А отсутствует или файл пуст")
                        return

                # Auto-calculate А/А0 if missing
                if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                    df['А/А0'] = df['А'] / df['А0']
                    if auto_a0_calculated:
                        st.markdown("""
                        <div class="highlight-info">
                            <strong>ℹ️ Автоматический расчет:</strong> Столбец А/А0 был автоматически рассчитан
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("""
                <div class="highlight-success">
                    <strong>✅ Успешно:</strong> Файл успешно загружен!
                </div>
                """, unsafe_allow_html=True)
                if file_extension != 'csv' and len(sheet_names) > 1:
                    st.info(f"Анализируется лист: **{selected_sheet}**")

                # Show raw data preview
                with st.expander("Предварительный просмотр данных"):
                    st.dataframe(df, use_container_width=True)
                    st.info(f"Общее количество строк: {len(df)}")

                    # Debug information for CSV files
                    if file_extension == 'csv':
                        st.markdown("**Информация о типах данных:**")
                        for col in ['т, мин', 'А', 'А0', 'А/А0']:
                            if col in df.columns:
                                sample_values = df[col].head(3).tolist()
                                data_type = df[col].dtype
                                st.text(f"{col}: {data_type} | Примеры: {sample_values}")

                # Data editing section for uploaded files
                st.subheader("Редактирование данных")

                # Show different messages based on whether А0 was auto-calculated
                if auto_a0_calculated:
                    st.info("Вы можете отредактировать загруженные данные перед анализом. А0 был автоматически определен из первого значения А и будет обновляться при изменении первой строки. Столбец А/А0 будет пересчитан автоматически.")
                else:
                    st.info("Вы можете отредактировать загруженные данные перед анализом. Первое значение в столбце А будет автоматически использовано как начальная концентрация А0 для всех расчетов. Столбец А/А0 будет пересчитан автоматически.")

                # Initialize session state for uploaded file editing
                if 'uploaded_first_a_value' not in st.session_state:
                    st.session_state.uploaded_first_a_value = None

                # Prepare data for editing (remove А0 column from display)
                edit_df = df.copy()
                display_columns = ['т, мин', 'А']

                # Create display dataframe with only the columns we want to show
                edit_df_display = edit_df[display_columns].copy()

                # Configure column settings for the data editor
                column_config = {
                    "т, мин": st.column_config.NumberColumn(
                        "Время (мин)",
                        help="Время в минутах",
                        min_value=0.0,
                        step=0.0001,
                        format="%.4f"
                    ),
                    "А": st.column_config.NumberColumn(
                        "Концентрация А",
                        help="Концентрация А (первое значение используется как А0)",
                        min_value=0.0,
                        step=0.0001,
                        format="%.5f"
                    )
                }

                # Create editable data table (only show т, мин and А columns)
                edited_df = st.data_editor(
                    edit_df_display,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config=column_config,
                    key="uploaded_data_editor"
                )

                # Auto-calculate А/А0 if А column exists
                if 'А' in edited_df.columns:
                    # Create a copy to avoid modifying the original
                    processed_edited_df = edited_df.copy()

                    # Apply auto-population logic: if first A value exists, populate all A0 values
                    if len(processed_edited_df) > 0 and processed_edited_df.iloc[0]['А'] > 0:
                        first_a_value = processed_edited_df.iloc[0]['А']
                        # Auto-populate all A0 values with the first A value
                        processed_edited_df['А0'] = first_a_value

                        # Show auto-population notification
                        if first_a_value != st.session_state.uploaded_first_a_value:
                            st.session_state.uploaded_first_a_value = first_a_value
                            st.success(f"Автоопределение: А0 = {first_a_value:.5f} (из первого значения А)")

                        # Remove rows with invalid data
                        valid_mask = (processed_edited_df['А'] > 0) & (processed_edited_df['А0'] > 0) & (processed_edited_df['т, мин'] >= 0)
                        processed_edited_df = processed_edited_df[valid_mask]

                        if not processed_edited_df.empty:
                            # Recalculate А/А0
                            processed_edited_df['А/А0'] = processed_edited_df['А'] / processed_edited_df['А0']
                            processed_edited_df['А/А0'] = processed_edited_df['А/А0'].round(4)

                            # Update df to use the edited and processed data
                            df = processed_edited_df.copy()

                            # Show updated preview
                            with st.expander("Предварительный просмотр отредактированных данных"):
                                # Show only visible columns plus А/А0
                                display_data = df[['т, мин', 'А', 'А/А0']].copy()
                                st.dataframe(display_data, use_container_width=True)
                                st.info(f"Действительных строк после редактирования: {len(df)}")
                                if len(processed_edited_df) > 0 and processed_edited_df.iloc[0]['А'] > 0:
                                    st.info(f"Автоопределение активно: А0 = {processed_edited_df.iloc[0]['А']:.5f} для всех расчетов")
                        else:
                            st.warning("Нет действительных данных после редактирования. Убедитесь, что все значения положительные.")
                            df = None

            except Exception as e:
                st.error(f"Ошибка чтения файла: {str(e)}")

        else:
            # Instructions when no file is uploaded
            st.info("Пожалуйста, загрузите файл для начала анализа")

    else:  # Manual data entry
        st.markdown("""
        <div class="section-header-data">
            <h3>✏️ Ручной ввод данных</h3>
        </div>
        """, unsafe_allow_html=True)

        # Initialize session state for tracking first A value
        if 'first_a_value' not in st.session_state:
            st.session_state.first_a_value = None
        if 'manual_data_initialized' not in st.session_state:
            st.session_state.manual_data_initialized = False

        # Create empty template data
        default_data = pd.DataFrame({
            'т, мин': [0.0],
            'А': [0.0]
        })

        st.info("Введите данные в таблицу ниже. Первое значение в столбце А будет автоматически использовано как начальная концентрация А0 для всех расчетов. Столбец А/А0 будет рассчитан автоматически.")

        # Data editor
        edited_data = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "т, мин": st.column_config.NumberColumn(
                    "Время (мин)",
                    help="Время в минутах",
                    min_value=0.0,
                    step=0.0001,
                    format="%.4f"
                ),
                "А": st.column_config.NumberColumn(
                    "Концентрация А",
                    help="Концентрация А (первое значение используется как А0)",
                    min_value=0.0,
                    step=0.0001,
                    format="%.5f"
                )
            },
            key="manual_data_editor"
        )

        # Auto-calculate A/A0 ratios and show preview
        if not edited_data.empty and 'А' in edited_data.columns:
            # Apply auto-population logic: use first A value as A0 for all calculations
            processed_data = edited_data.copy()

            # Get the first valid A value (non-zero) from the first row
            first_a_value = None
            if len(processed_data) > 0 and processed_data.iloc[0]['А'] > 0:
                first_a_value = processed_data.iloc[0]['А']
                # Add А0 column internally for calculations
                processed_data['А0'] = first_a_value

                # Show auto-determination info
                if first_a_value != st.session_state.first_a_value:
                    st.session_state.first_a_value = first_a_value
                    st.success(f"Автоопределение: А0 = {first_a_value:.5f} (из первого значения А)")

                # Remove rows with zero or negative values
                valid_data = processed_data[(processed_data['А'] > 0) & (processed_data['А0'] > 0) & (processed_data['т, мин'] >= 0)]

                if not valid_data.empty:
                    # Calculate A/A0 ratios
                    valid_data = valid_data.copy()
                    valid_data['А/А0'] = valid_data['А'] / valid_data['А0']
                    valid_data['А/А0'] = valid_data['А/А0'].round(4)

                    # Show preview with calculated ratios (only visible columns plus А/А0)
                    st.subheader("Предварительный просмотр с рассчитанными А/А0")
                    display_data = valid_data[['т, мин', 'А', 'А/А0']].copy()
                    st.dataframe(display_data, use_container_width=True)
                    st.info(f"Автоопределение активно: А0 = {first_a_value:.5f} для всех расчетов")

        # Validate manual data
        if st.button("Анализировать введенные данные", type="primary"):
            if edited_data.empty:
                st.error("Пожалуйста, введите данные для анализа")
            else:
                # Use the processed data with auto-population and calculated A/A0
                if not edited_data.empty and 'А' in edited_data.columns:
                    # Apply auto-population logic first
                    processed_data = edited_data.copy()

                    # Get the first valid A value (non-zero) from the first row
                    if len(processed_data) > 0 and processed_data.iloc[0]['А'] > 0:
                        first_a_value = processed_data.iloc[0]['А']
                        # Auto-populate all A0 values with the first A value
                        processed_data['А0'] = first_a_value
                    else:
                        st.error("Первое значение в столбце А должно быть положительным числом.")
                        return

                    # Remove rows with zero or negative values
                    valid_data = processed_data[(processed_data['А'] > 0) & (processed_data['А0'] > 0) & (processed_data['т, мин'] >= 0)]

                    if valid_data.empty:
                        st.error("Нет действительных данных. Убедитесь, что все значения положительные.")
                    else:
                        # Calculate A/A0 ratios
                        valid_data = valid_data.copy()
                        valid_data['А/А0'] = valid_data['А'] / valid_data['А0']

                        # Validate the processed data
                        is_valid, error_message = validate_data_structure(valid_data)

                        if not is_valid:
                            st.error(f"{error_message}")
                        else:
                            # Check for additional validation
                            if valid_data['т, мин'].duplicated().any():
                                st.error("Значения времени не должны повторяться")
                            elif not valid_data['т, мин'].is_monotonic_increasing:
                                st.error("Значения времени должны быть в возрастающем порядке")
                            else:
                                # Data is valid, use it for analysis
                                df = valid_data.copy()
                                st.success("Данные успешно введены!")
                                st.info(f"Введено {len(df)} точек данных")
                                st.info(f"Автоопределение: А0 = {first_a_value:.5f} для всех расчетов")

    # Process data if we have valid data (from either source)
    if df is not None and not df.empty:
        pass  # Debug information removed as requested

        # Process data
        processed_df = preprocess_data(df)

        if len(processed_df) == 0:
            st.error("Нет действительных данных после обработки")
            st.error("Возможные причины:")
            st.error("- Все значения равны нулю или отрицательные")
            st.error("- Проблемы с форматом чисел (например, запятая вместо точки)")
            st.error("- Отсутствуют обязательные столбцы")
            return

        # Data summary with enhanced styling and emojis
        summary = get_data_summary(processed_df)

        st.markdown("""
        <div class="section-header-analysis">
            <h2>📈 Сводка данных</h2>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced metrics with custom styling
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="summary-stat">
                <div class="metric-label">📊 Действительные точки</div>
                <div class="metric-value">{summary['total_points']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="summary-stat">
                <div class="metric-label">⏱️ Диапазон времени</div>
                <div class="metric-value">{summary['time_range'][0]:.1f} - {summary['time_range'][1]:.1f} мин</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="summary-stat">
                <div class="metric-label">🧪 Начальная концентрация</div>
                <div class="metric-value">{summary['a0_value']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="summary-stat">
                <div class="metric-label">📊 Диапазон A/A0</div>
                <div class="metric-value">{summary['a_a0_range'][0]:.3f} - {summary['a_a0_range'][1]:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Find stable points (using fixed threshold of 0.1)
        stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
        selected_data = processed_df.iloc[stable_indices].copy()

        st.markdown("""
        <div class="section-header-analysis">
            <h2>🎯 Выбранные точки</h2>
        </div>
        """, unsafe_allow_html=True)

        # Highlighted key information about selected points
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="key-metric">
                <div class="metric-label">🎯 Выбранные точки данных</div>
                <div class="metric-value">{len(selected_data)} из {len(processed_df)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="key-metric">
                <div class="metric-label">⏱️ Временной диапазон</div>
                <div class="metric-value">{selected_data['т, мин'].iloc[0]:.1f} - {selected_data['т, мин'].iloc[-1]:.1f} мин</div>
            </div>
            """, unsafe_allow_html=True)

        # Fit models
        try:
            k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
            k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

            # Results summary with enhanced styling and emojis
            st.markdown("""
            <div class="section-header-results">
                <h2>📋 Сводка результатов</h2>
            </div>
            """, unsafe_allow_html=True)

            results_summary = create_results_summary(k1, k2, mape_pfo, mape_pso, r2_pfo, r2_pso)

            # Display summary table
            #st.subheader("📋 Сводка результатов")
            st.dataframe(results_summary, use_container_width=True)

            # Model comparison metrics with enhanced performance styling
            st.markdown("""
            <div class="section-header-results">
                <h3>🔎 Сравнение моделей</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔵 Модель PFO")

                # Enhanced performance metrics for PFO
                st.markdown(f"""
                <div class="performance-metric">
                    <div class="metric-label">⚡ коэффициент k₁</div>
                    <div class="metric-value">{abs(k1):.5f} мин⁻¹</div>
                </div>
                
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="performance-metric">
                    <div class="metric-label">📊 R² Score</div>
                    <div class="metric-value">{r2_pfo:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="performance-metric">
                    <div class="metric-label">📈 MAPE (%)</div>
                    <div class="metric-value">{mape_pfo:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                

            with col2:
                st.markdown("### 🟢 Модель PSO")

                # Enhanced performance metrics for PSO
                st.markdown(f"""
                <div class="performance-metric">
                    <div class="metric-label">⚡ коэффициент k₂</div>
                    <div class="metric-value">{k2:.5f} л/(мг·мин)</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="performance-metric">
                    <div class="metric-label">📊 R² Score</div>
                    <div class="metric-value">{r2_pso:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="performance-metric">
                    <div class="metric-label">📈 MAPE (%)</div>
                    <div class="metric-value">{mape_pso:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed results
            with st.expander("Подробные результаты"):
                detailed_results = create_detailed_results(pfo_predictions, pso_predictions)
                st.dataframe(detailed_results, use_container_width=True)

            # Plots with enhanced styling and emoji
            st.markdown("""
            <div class="section-header-visualization">
                <h2>📊 Графики</h2>
            </div>
            """, unsafe_allow_html=True)

            # Generate Matplotlib plots
            fig_main = create_matplotlib_plots(processed_df, selected_data, pfo_predictions, pso_predictions, k1, k2)
            st.pyplot(fig_main)

            # Download results with enhanced styling and emoji
            st.markdown("""
            <div class="section-header-download">
                <h2>💾 Скачать результаты</h2>
            </div>
            """, unsafe_allow_html=True)

            # Prepare download data
            download_data = {
                'Summary': results_summary,
                'Detailed_Results': detailed_results,
                'Selected_Data': selected_data,
                'PFO_Predictions': pfo_predictions,
                'PSO_Predictions': pso_predictions
            }

            # Create Excel file for download
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

    else:
        # Show instructions when no data is available
        if input_method == "Загрузить файл":
            pass
        else:
            st.info("Пожалуйста, введите данные и нажмите кнопку анализа")


if __name__ == "__main__":
    main()
