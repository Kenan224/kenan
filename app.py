"""
Streamlit Web Application for Kinetic Modeling Analysis
Универсальная программная платформа для анализа кинетического моделирования
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
st.markdown


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
    # الخطوة الأقوى: أداة اختيار نوع التفاعل الكيميائي الرئيسي في أعلى التطبيق
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
    # القسم الأول: Фотокаталитические реакции (نظامك الحالي بالكامل)
    # =========================================================================
    if reaction_type == "Фотокаталитические реакции":
        
        # إعدادات الشريط الجانبي الخاصة بالفوتوكاتاليز ديناميكياً
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
            uploaded_file = st.file_uploader("Выберите файл", type=['xlsx', 'csv'])
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

                # زر تحميل الصورة عالي الجودة المضاف حديثاً بالروسية
                png_buffer = BytesIO()
                fig_main.savefig(png_buffer, format='png', dpi=300, bbox_inches='tight')
                png_buffer.seek(0)
                st.download_button(
                    label="📥 Скачать графики в высоком разрешении (PNG)",
                    data=png_buffer,
                    file_name="kinetic_plots_300dpi.png",
                    mime="image/png"
                )

                st.markdown('<div class="section-header-download"><h2>💾 Скачать результаты</h2></div>', unsafe_allow_html=True)
                detailed_results = create_detailed_results(zo_predictions, pfo_predictions, pso_predictions)
                download_data = {'Summary': results_summary, 'Detailed_Results': detailed_results}
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for s_name, data in download_data.items():
                        data.to_excel(writer, sheet_name=s_name, index=False)
                st.download_button(label="Скачать результаты как файл Excel", data=output.getvalue(), file_name="kinetic_results.xlsx")

            except Exception as e:
                st.error(f"Ошибка моделирования: {str(e)}")

    # =========================================================================
    # القسم الثاني: Гомогенный катализ (القسم الجديد للتحفيز المتجانس)
    # =========================================================================
    elif reaction_type == "Гомогенный катализ":
        with st.sidebar:
            st.header("⚙️ Параметры процесса")
            st.info("Тип: Гомогенный катализ")
            st.markdown("---")
            st.markdown("### 📋 Требования к файлу")
            # تغيير متطلبات الإدخال بناءً على طلبك
            st.markdown("""
            **Обязательные столбцы:**
            - `т, мин` (Время реакции)
            - `C, моль/л` (Концентрация вещества)
            - `T, K` (Температура для Arrhenius)
            """)

        st.markdown("""
        <div class="section-header-data" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color: #92400e; border-left-color: #f59e0b;">
            <h2>📊 Ввод данных (Гомогенный катализ)</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("⚠️ Раздел находится в разработке. Здесь будут рассчитываться модели: Power-law, Arrhenius, Последовательные реакции.")
        
        # مثال مستقبلي لشكل أعمدة الإدخال للتحفيز المتجانس
        homo_data = pd.DataFrame({
            'т, мин': [0, 10, 20, 30],
            'C, моль/л': [1.0, 0.8, 0.64, 0.51],
            'T, K': [298, 298, 298, 298]
        })
        st.subheader("Пример структуры таблицы:")
        st.data_editor(homo_data, use_container_width=True, key="homo_ed")

        # هنا في الخطوات القادمة سنضيف توابع الفيتينج الخاصة بـ:
        # 1. fit_power_law()
        # 2. fit_arrhenius()
        # 3. fit_consecutive()

    # =========================================================================
    # القسم الثالث والرابع (قوالب جاهزة للتعبئة مستقبلاً)
    # =========================================================================
    elif reaction_type == "Гетерогенный катализ":
        st.info("Раздел 'Гетерогенный катализ' (Модели Ленгмюра-Хиншельвуда и др.)")
        
    elif reaction_type == "Ферментативные реакции":
        st.info("Раздел 'Ферментативные реакции' (Модели Михаэлиса-Ментен, Коэффициент Хилла)")


if __name__ == "__main__":
    main()
