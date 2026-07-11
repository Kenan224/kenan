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

# استيراد الموديولات الخارجية الخاصة بك
from data_processor import validate_data_structure, preprocess_data, get_data_summary, read_csv_file
from kinetic_models import (
    find_stable_points, fit_zo_model, fit_pfo_model, fit_pso_model,
    create_results_summary, create_detailed_results
)
from visualization import create_matplotlib_plots

# تهيئة الصفحة ومطابقة الهوية البصرية للألوان الفاتحة
st.set_page_config(
    page_title="Анализ кинетического моделирования",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تعديل الـ CSS بالكامل ليتطابق مع الصورة المرسلة بنسبة 100% ومنع النصوص البيضاء
st.markdown("""
<style>
/* تصفير الخلفية لتكون بيضاء بالكامل مثل الصورة */
.stApp {
    background-color: #ffffff;
}

/* تصميم القائمة الجانبية الفاتحة */
[data-testid="stSidebar"] {
    background-color: #f4f6f9;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* الهيدر الأزرق الرئيسي العريض في الأعلى */
.main-header {
    background-color: #1d63ed;
    padding: 1.5rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.main-header h1 {
    margin: 0;
    font-weight: 500;
    font-size: 2.2rem;
    color: #ffffff !important; /* نص أبيض لأن الخلفية زرقاء داكنة */
}

/* بطاقة معلومات الطالب والمشرف خلفية رمادية فاتحة ونص داكن */
.student-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}
.student-card p {
    margin: 0.4rem 0;
    color: #334155 !important;
    font-size: 0.95rem;
}

/* العناوين الفرعية: مربع أزرق فاتح مع خط أزرق عريض جهة اليسار */
.custom-section-header {
    background-color: #cfe2ff;
    border-left: 6px solid #1d63ed;
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.custom-section-header h2 {
    color: #084298 !important; /* نص أزرق داكن واضح جداً */
    margin: 0;
    font-size: 1.6rem;
    font-weight: 500;
}

/* شريط التنبيهات والمعلومات الأزرق الفاتح تحت الجدول */
.info-banner {
    background-color: #e2eefd;
    color: #0a58ca !important;
    padding: 0.8rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    border: 1px solid #b6d4fe;
}

/* إجبار جميع النصوص الافتراضية والعناصر على اللون الداكن منعا للاختفاء */
p, span, label, .stRadio {
    color: #0f172a !important;
}

/* تصميم الزر الأزرق الكبير في الأسفل */
div.stButton > button {
    background-color: #1d63ed !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    font-weight: 500 !important;
    border-radius: 4px !important;
}
div.stButton > button:hover {
    background-color: #154ec2 !important;
    color: #ffffff !important;
}

/* مربعات النتائج الإحصائية الفاتحة */
.metric-box-yellow {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-box-yellow h4 { margin: 0; color: #b45309 !important; font-size: 0.9rem; }
.metric-box-yellow h2 { margin: 0.3rem 0 0 0; color: #78350f !important; font-size: 1.5rem; }

.metric-box-blue {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-box-blue h4 { margin: 0; color: #1d4ed8 !important; font-size: 0.9rem; }
.metric-box-blue h2 { margin: 0.3rem 0 0 0; color: #1e40af !important; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


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
                df = df.iloc[i+1:].copy()
                df.columns = new_headers
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
        series = df.iloc[:, i].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
        cleaned_dict[col_name] = pd.to_numeric(series, errors='coerce')
        
    return pd.DataFrame(cleaned_dict).dropna(how='all')


def main():
    # الهيدر العلوي المتطابق مع لقطة الشاشة
    st.markdown("""
    <div class="main-header">
        <h1>Анализ кинетического моделирования</h1>
    </div>
    """, unsafe_allow_html=True)

    # كرت الطالب والمشرف المتطابق مع لقطة الشاشة
    st.markdown("""
    <div class="student-card">
        <p><strong>СТУДЕНТ:</strong> Алсади К.</p>
        <p><strong>РУКОВОДИТЕЛЬ:</strong> Киреева А.В</p>
    </div>
    """, unsafe_allow_html=True)

    # القائمة الجانبية الجاهزة
    with st.sidebar:
        st.markdown("### ⚙️ Параметры")
        reaction_type = st.selectbox(
            label="Тип химического процесса:",
            options=[
                "Фотокаталитические реакции",
                "Гомогенный катализ",
                "Гетерогенный катализ",
                "Ферментативные реакции"
            ],
            index=0
        )
        st.markdown("---")
        st.markdown("### 📋 Требования к файлу")
        st.markdown("""
        **Обязательные столбцы:**
        * <span style="color:#10b981; font-weight:bold;">т, мин</span> (Время в минутах)
        * <span style="color:#10b981; font-weight:bold;">А</span> (Оптическая плотность)
        
        **Опциональные столбцы:**
        * <span style="color:#10b981; font-weight:bold;">А0</span> (Начальная Оптическая плотность) - если отсутствует, используется первое значение А
        * <span style="color:#10b981; font-weight:bold;">А/А0</span> (Отношение А/А0) - рассчитывается автоматически если отсутствует
        
        **Поддерживаемые форматы:**
        * Excel (.xlsx)
        * CSV (.csv) с автоопределением разделителя
        """, unsafe_allow_html=True)

    # =========================================================================
    # 1. قسم الفتح والتحليل الكينتيكي الضوئي
    # =========================================================================
    if reaction_type == "Фотокаталитические реакции":
        st.markdown('<div class="custom-section-header"><h2>📊 Ввод данных</h2></div>', unsafe_allow_html=True)
        
        input_method = st.radio(
            "Выберите способ ввода данных:",
            ["Загрузить файл", "Ввести данные вручную"],
            index=1,
            horizontal=True
        )

        df = None
        if input_method == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Загрузите файл",
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
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                        # Check if sheet is empty
                        if df.empty:
                            st.error(f"Лист '{selected_sheet}' пуст")
                            return

                    # Validate structure (common for both CSV and Excel)
                    is_valid, error_message = validate_data_structure(df)

                    if not is_valid:
                        st.error(f"{error_message}")
                        if file_extension != 'csv' and len(sheet_names) > 1:
                            st.error(f"Лист '{selected_sheet}' не содержит требуемых данных. Попробуйте выбрать другой лист.")
                        return

                    # التعديل هنا: هذا الجزء أصبح خارج شرط الخطأ ويعمل عندما تكون البيانات صالحة فقط
                    df.columns = df.columns.str.strip()
                    if 'А0' not in df.columns and 'А' in df.columns and len(df) > 0:
                        df['А'] = pd.to_numeric(df['А'], errors='coerce')
                        df['А0'] = df['А'].iloc[0]
                    if 'А/А0' not in df.columns and 'А' in df.columns and 'А0' in df.columns:
                        df['А/А0'] = df['А'] / df['А0']
                    
                    df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
                    
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")

        else:
            st.markdown('<div class="custom-section-header"><h2>📝 Ручной ввод данных</h2></div>', unsafe_allow_html=True)
            st.markdown('<div class="info-banner">Введите данные в таблицу ниже. Первое значение в столбце А будет автоматически использовано как начальная Оптическая плотность А0 для всех расчетов. Столбец А/А0 будет рассчитан автоматически.</div>', unsafe_allow_html=True)
            
            init_cols = ['Время (мин)', 'Оптическая плотность А']
            init_rows = [[0.0, 0.00000]]
            default_data = pd.DataFrame(columns=init_cols, data=init_rows)
            
            edited_data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
            
            if not edited_data.empty and len(edited_data) > 0:
                raw_t = pd.to_numeric(edited_data.iloc[:, 0], errors='coerce')
                raw_a = pd.to_numeric(edited_data.iloc[:, 1], errors='coerce')
                
                valid_mask = (raw_a >= 0) & (raw_t >= 0)
                if valid_mask.any():
                    df = pd.DataFrame({
                        'т, мин': raw_t[valid_mask],
                        'А': raw_a[valid_mask]
                    })
                    if len(df) > 0 and df.iloc[0]['А'] > 0:
                        df['А0'] = df.iloc[0]['А']
                        df['А/А0'] = df['А'] / df['А0']

        # زر بدء الحسابات أسفل الجدول مباشرة متطابق مع الصورة
        if df is not None and not df.empty:
            if st.button("Анализировать введенные данные"):
                processed_df = preprocess_data(df)
                summary = get_data_summary(processed_df)
                
                st.markdown('### 📊 Сводка данных')
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown(f'<div class="metric-box-yellow"><h4>Точки</h4><h2>{summary["total_points"]}</h2></div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-box-yellow"><h4>Время</h4><h2>{summary["time_range"][1]:.1f} мин</h2></div>', unsafe_allow_html=True)
                col3.markdown(f'<div class="metric-box-yellow"><h4>Начальная А0</h4><h2>{summary["a0_value"]:.3f}</h2></div>', unsafe_allow_html=True)
                col4.markdown(f'<div class="metric-box-yellow"><h4>Мин А/А0</h4><h2>{summary["a_a0_range"][1]:.3f}</h2></div>', unsafe_allow_html=True)

                stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
                selected_data = processed_df.iloc[stable_indices].copy()

                try:
                    k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(selected_data)
                    k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
                    k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

                    st.markdown('<div class="custom-section-header"><h2>📋 Результаты кинетического моделирования</h2></div>', unsafe_allow_html=True)
                    results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
                    st.dataframe(results_summary, use_container_width=True)

                    st.markdown('### 📊 Графический анализ')
                    fig_main = create_matplotlib_plots(processed_df, selected_data, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
                    st.pyplot(fig_main)
                except Exception as e:
                    st.error(f"Ошибка расчетов: {str(e)}")

    # =========================================================================
    # 2. قسم الكاتاليز المتجانس (سليم ومنظم برمجياً وبصرياً)
    # =========================================================================
    elif reaction_type == "Гомогенный катализ":
        st.markdown('### Кинетическая модель гомогенного катализа')
        homo_model = st.radio(
            "Выберите модель:",
            ["Power-law (степенной закон)", "Arrhenius", "Последовательные реакции"],
            index=0, horizontal=True
        )

        st.markdown(f'<div class="custom-section-header"><h2>📊 Ввод данных для {homo_model}</h2></div>', unsafe_allow_html=True)
        
        # بناء الجداول الافتراضية بأسطر منفصلة تماما لمنع أي مشاكل في الأقواس
        if homo_model == "Power-law (степенной закон)":
            cols = ['t', 'CA', 'CB', 'r']
            rows = [[0.0, 1.0, 1.5, 0.05], [5.0, 0.8, 1.3, 0.035]]
        elif homo_model == "Arrhenius":
            cols = ['T', 'k']
            rows = [[298.0, 0.01], [308.0, 0.02]]
        else:
            cols = ['t', 'CA', 'CB', 'CC']
            rows = [[0.0, 1.0, 0.0, 0.0], [10.0, 0.5, 0.4, 0.1]]

        empty_df = pd.DataFrame(columns=cols, data=rows)
        h_df = st.data_editor(empty_df, use_container_width=True, num_rows="dynamic")

        if h_df is not None and len(h_df) > 0:
            if st.button("Рассчитать параметры гомогенного катализа"):
                h_df = clean_homogeneous_data(h_df)
                
                if homo_model == "Power-law (степенной закон)" and all(c in h_df.columns for c in ['CA', 'CB', 'r']):
                    try:
                        valid = h_df[(h_df['CA'] > 0) & (h_df['CB'] > 0) & (h_df['r'] > 0)]
                        if not valid.empty:
                            log_CA, log_CB, log_r = np.log(valid['CA'].values), np.log(valid['CB'].values), np.log(valid['r'].values)
                            X = np.column_stack((np.ones_like(log_CA), log_CA, log_CB))
                            beta, _, _, _ = np.linalg.lstsq(X, log_r, rcond=None)
                            k_val, a_val, b_val = np.exp(beta[0]), beta[1], beta[2]
                            
                            st.success(f"Успешно: k = {k_val:.4f}, α = {a_val:.2f}, β = {b_val:.2f}")
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
                        
                elif homo_model == "Arrhenius" and all(c in h_df.columns for c in ['T', 'k']):
                    try:
                        valid = h_df[(h_df['T'] > 0) & (h_df['k'] > 0)]
                        if not valid.empty:
                            inv_T = 1.0 / valid['T'].values
                            log_k = np.log(valid['k'].values)
                            slope, intercept = np.polyfit(inv_T, log_k, 1)
                            Ea = -slope * 8.314 / 1000.0
                            A = np.exp(intercept)
                            st.success(f"Успешно: A = {A:.2e}, Ea = {Ea:.2f} кДж/моль")
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")

    elif reaction_type == "Гетерогенный катализ":
        st.info("Раздел 'Гетерогенный катализ' успешно сохранен.")
    elif reaction_type == "Ферментативные реакции":
        st.info("Раздел 'Ферментативные реакции' успешно сохранен.")


if __name__ == "__main__":
    main()
