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

# 🎨 ضربة قاضية لمشكلة الألوان: إعادة تعيين متغيرات جافا سكريبت و ستايل الـ سبريم ليت بالكامل للون الأسود
st.markdown("""
<style>
/* 1. إعادة تعيين المتغيرات الأساسية للمتصفح لمنع التبديل التلقائي للثيم الداكن */
:root {
    --primary-color: #1d63ed !important;
    --background-color: #ffffff !important;
    --secondary-background-color: #f4f6f9 !important;
    --text-color: #000000 !important;
}

/* 2. إجبار خلفية التطبيق بالكامل والقائمة الجانبية على اللون الفاتح */
.stApp {
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] {
    background-color: #f4f6f9 !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* 3. تعيين لون خط أسود صريح لجميع المكونات النصية بدون استثناء لمنع النص الأبيض */
html, body, .stApp, [data-testid="stSidebar"], p, span, label, h1, h2, h3, h4, h5, h6, input, button, select, textarea {
    color: #000000 !important;
}

/* 4. إجبار صناديق الاختيار، القوائم المنسدلة، وجداول البيانات، وصناديق رفع الملفات على خلفية رمادية فاتحة جداً ونص أسود */
div[data-baseweb="select"], 
div[data-baseweb="select"] *,
div[data-testid="stFileUploaderDropzone"], 
div[data-testid="stFileUploaderDropzone"] *,
.stSelectbox div, 
div[role="radiogroup"] *,
.stDataFrame div,
div[data-testid="stNotification"] * {
    background-color: #f1f5f9 !important;
    color: #000000 !important;
}

/* 5. استثناء وحيد: النصوص داخل الهيدر الأزرق الرئيسي والأزرار الزرقاء يجب أن تكون بيضاء لتظهر بشكل صحيح */
.main-header h1 {
    color: #ffffff !important;
}
div.stButton > button {
    background-color: #1d63ed !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    font-weight: bold !important;
    border-radius: 4px !important;
}
div.stButton > button * {
    color: #ffffff !important;
}
div.stButton > button:hover {
    background-color: #154ec2 !important;
}

/* 6. تنسيقات مخصصة للهيدرات والبطاقات */
.main-header {
    background-color: #1d63ed;
    padding: 1.5rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.student-card {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}
.custom-section-header {
    background-color: #cfe2ff;
    border-left: 6px solid #1d63ed;
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.info-banner {
    background-color: #e2eefd;
    padding: 0.8rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 1.2rem;
    border: 1px solid #b6d4fe;
}
.metric-box-yellow {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Кинетический_Расчет')
    return output.getvalue()


def clean_homogeneous_data(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    
    # محاولة تنظيف وتوحيد أسماء الأعمدة لتتناسب مع الكاتاليز المتجانس
    new_columns = []
    for col in df.columns:
        c = str(col).strip().lower().replace(' ', '').replace('_', '')
        if 'ca' in c or c == 'а' or c == 'a':
            new_columns.append('CA')
        elif 'cb' in c or c == 'в' or c == 'b':
            new_columns.append('CB')
        elif 'cc' in c or c == 'с' or c == 'c':
            new_columns.append('CC')
        elif 'rate' in c or 'скорость' in c or c in ['r', 'w', 'v']:
            new_columns.append('r')
        elif 'temp' in c or 'темп' in c or c in ['t', 'т']:
            new_columns.append('T')
        elif 'k' in c:
            new_columns.append('k')
        elif 'time' in c or 'время' in c or c in ['t', 'т']:
            new_columns.append('t')
        else:
            new_columns.append(col)
            
    df.columns = new_columns
    
    # تحويل البيانات إلى أرقام
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
    return df.dropna(how='all')


def main():
    # الهيدر العلوي ومعلومات الطالب
    st.markdown('<div class="main-header"><h1>Универсальная платформа кинетического анализа</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="student-card"><p><strong>СТУДЕНТ:</strong> Алсади К.</p><p><strong>РУКОВОДИТЕЛЬ:</strong> Киреева А.В</p></div>', unsafe_allow_html=True)

    # =========================================================================
    # 1. بناء الهامش الجانبي (Sidebar) بالترتيب الصارم والمطلوب
    # =========================================================================
    with st.sidebar:
        # زر اختيار نوع التفاعل في أعلى الهامش تماماً
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
        
        # كلمة ПАРАМЕТРЫ تحت الاختيار مباشرة
        st.markdown("### ⚙️ ПАРАМЕТРЫ")
        st.markdown("---")
        
        # متطلبات الملف والمدخلات والنواتج بناءً على نوع التفاعل المختار
        st.markdown("### 📋 Требования к файлу")
        if reaction_type == "Фотокаталитические реакции":
            st.markdown("""
            **Входные данные (Субстраты):**
            * `т, мин` (Время процесса)
            * `А` (Оптическая плотность)
            
            **Выходные данные (Продукты):**
            * `А0` (Начальная плотность)
            * `А/А0` (Относительная плотность)
            """)
        else:
            st.markdown("""
            **Входные данные (Компоненты):**
            * Для Степенного закона: `CA`, `CB`, `r`
            * Для Аррениуса: `T`, `k`
            """)
            
        st.markdown("""
        **Поддерживаемые форматы:**
        * Excel (.xlsx) / CSV (.csv)
        """)
        
        # مكان محجوز ديناميكياً لـ قائمة صفحات الإكسل (تظهر في الأسفل عند تحقق الشرط فقط)
        sheet_selector_placeholder = st.empty()

    # =========================================================================
    # 2. الجزء الرئيسي للموقع: اختيار طريقة إدخال البيانات (يدوي أو رفع ملف)
    # =========================================================================
    st.markdown('<div class="custom-section-header"><h2>📊 Ввод и управление данными</h2></div>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Выберите способ ввода экспериментальных данных:", 
        ["Загрузить файл", "Ввести данные вручную"], 
        index=1, 
        horizontal=True
    )

    df_raw = None

    if input_method == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите файл для загрузки в систему", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'csv':
                    df_raw = read_csv_file(uploaded_file)
                else:
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    selected_sheet = sheet_names[0]
                    
                    # الشرط الحتمي: لا تظهر قائمة الصفحات إلا إذا كان الملف يحتوي على أكثر من صفحة واحدة
                    if len(sheet_names) > 1:
                        with sheet_selector_placeholder.container():
                            st.markdown("---")
                            selected_sheet = st.selectbox("Выберите лист Excel для анализа:", sheet_names, index=0)
                    
                    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {str(e)}")
    else:
        # إنشاء جداول افتراضية نظيفة بناءً على نوع التفاعل لتسهيل الإدخال اليدوي
        if reaction_type == "Фотокаталитические реакции":
            init_cols = ['Время т (мин)', 'Оптическая плотность А']
            init_rows = [[0.0, 1.000], [10.0, 0.750], [20.0, 0.510], [30.0, 0.330]]
        elif reaction_type == "Гомогенный катализ":
            init_cols = ['CA', 'CB', 'r']
            init_rows = [[1.0, 1.5, 0.050], [0.8, 1.3, 0.035], [0.6, 1.1, 0.021]]
        else:
            init_cols = ['Данные']
            init_rows = [[0.0]]
            
        st.markdown('<div class="info-banner">Введите или отредактируйте данные непосредственно в таблице ниже:</div>', unsafe_allow_html=True)
        default_data = pd.DataFrame(columns=init_cols, data=init_rows)
        df_raw = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

    # =========================================================================
    # 3. معالجة وتدريب النماذج حسب منطق كل نوع تفاعل
    # =========================================================================
    if df_raw is not None and not df_raw.empty:

        # ---------------------------------------------------------------------
        # أ. الفوتوكاتاليز (Фотокаталитические реакции) -> حساب الـ 3 موديلات معاً فوراً
        # ---------------------------------------------------------------------
        if reaction_type == "Фотокаталитические реакции":
            # تحضير الأعمدة وتعديل المسميات لتناسب دوال المعالجة الخاصة بك
            df_working = df_raw.copy()
            if input_method == "Ввести данные вручную":
                df_working.columns = ['т, мин', 'А']
            else:
                df_working.columns = df_working.columns.str.strip()
                
            if 'А' in df_working.columns and len(df_working) > 0:
                df_working['А'] = pd.to_numeric(df_working['А'], errors='coerce')
                if 'А0' not in df_working.columns:
                    df_working['А0'] = df_working['А'].iloc[0]
                if 'А/А0' not in df_working.columns:
                    df_working['А/А0'] = df_working['А'] / df_working['А0']

            if st.button("Запустить кинетический анализ фотокатализа"):
                try:
                    processed_df = preprocess_data(df_working)
                    summary = get_data_summary(processed_df)
                    
                    st.markdown('### 📊 Сводка экспериментальных данных')
                    col1, col2, col3, col4 = st.columns(4)
                    col1.markdown(f'<div class="metric-box-yellow"><h4>Точки</h4><h2>{summary["total_points"]}</h2></div>', unsafe_allow_html=True)
                    col2.markdown(f'<div class="metric-box-yellow"><h4>Время</h4><h2>{summary["time_range"][1]:.1f} мин</h2></div>', unsafe_allow_html=True)
                    col3.markdown(f'<div class="metric-box-yellow"><h4>Начальная А0</h4><h2>{summary["a0_value"]:.3f}</h2></div>', unsafe_allow_html=True)
                    col4.markdown(f'<div class="metric-box-yellow"><h4>Мин А/А0</h4><h2>{summary["a_a0_range"][1]:.3f}</h2></div>', unsafe_allow_html=True)

                    stable_indices = find_stable_points(processed_df['ln_A_A0'], processed_df['т, мин'], 0.1)
                    selected_data = processed_df.iloc[stable_indices].copy()

                    # حساب وتدريب النماذج الثلاثة في نفس الوقت
                    k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(selected_data)
                    k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(selected_data)
                    k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(selected_data)

                    st.markdown('<div class="custom-section-header"><h2>📋 Сравнительные результаты моделей (ZO, PFO, PSO)</h2></div>', unsafe_allow_html=True)
                    results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
                    st.dataframe(results_summary, use_container_width=True)

                    st.markdown('### 📊 Графический анализ аппроксимации')
                    fig_main = create_matplotlib_plots(processed_df, selected_data, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
                    st.pyplot(fig_main)
                    
                    # 4. قسم تحميل النتائج في زرين منفصلين بجانب بعضهما تماماً
                    st.markdown('### 💾 Скачать результаты анализа')
                    dl_col1, dl_col2 = st.columns(2)
                    
                    excel_data = convert_df_to_excel(results_summary)
                    dl_col1.download_button(
                        label="📊 Скачать результаты (Excel)",
                        data=excel_data,
                        file_name="Photocatalytic_Results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    img_buf = BytesIO()
                    fig_main.savefig(img_buf, format="png", bbox_inches="tight")
                    img_buf.seek(0)
                    dl_col2.download_button(
                        label="🖼️ Скачать графики (PNG)",
                        data=img_buf,
                        file_name="Photocatalytic_Plots.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Ошибка при расчете фотокатализа: {str(e)}")

        # ---------------------------------------------------------------------
        # ب. الكاتاليز المتجانس (Гомогенный катализ) -> يجب اختيار النموذج أولاً لإتمام الحساب
        # ---------------------------------------------------------------------
        elif reaction_type == "Гомогенный катализ":
            st.markdown('### Кинетические модели гомогенного катализа')
            
            # إجبار المستخدم على اختيار نموذج معين قبل معالجة البيانات وإظهار النتائج كما طلبت
            homo_model = st.radio(
                "Выберите конкретную модель для вычисления параметров:",
                ["Power-law (степенной закон)", "Arrhenius"],
                index=0, 
                horizontal=True
            )

            if st.button("Рассчитать параметры гомогенного катализа"):
                cleaned_df = clean_homogeneous_data(df_raw)
                
                if homo_model == "Power-law (степенной закон)":
                    if all(c in cleaned_df.columns for c in ['CA', 'CB', 'r']):
                        try:
                            valid = cleaned_df[(cleaned_df['CA'] > 0) & (cleaned_df['CB'] > 0) & (cleaned_df['r'] > 0)]
                            if not valid.empty:
                                log_CA, log_CB, log_r = np.log(valid['CA'].values), np.log(valid['CB'].values), np.log(valid['r'].values)
                                X = np.column_stack((np.ones_like(log_CA), log_CA, log_CB))
                                beta, _, _, _ = np.linalg.lstsq(X, log_r, rcond=None)
                                k_val, a_val, b_val = np.exp(beta[0]), beta[1], beta[2]
                                
                                st.markdown('<div class="custom-section-header"><h2>📋 Результаты расчета Степенного закона</h2></div>', unsafe_allow_html=True)
                                res_df = pd.DataFrame({
                                    "Параметр кинетики": ["Константа скорости реакций (k)", "Порядок реакции по компоненту A (α)", "Порядок реакции по компоненту B (β)"],
                                    "Рассчитанное значение": [f"{k_val:.4f}", f"{a_val:.2f}", f"{b_val:.2f}"]
                                })
                                st.dataframe(res_df, use_container_width=True)
                                
                                # إتاحة التحميل المباشر لنتائج الكاتاليز المتجانس كملف إكسل
                                excel_data_homo = convert_df_to_excel(res_df)
                                st.download_button(
                                    label="📊 Скачать результаты гомогенного катализа (Excel)",
                                    data=excel_data_homo,
                                    file_name="Homogeneous_Kinetic_Results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            else:
                                st.error("Нет валидных данных для логарифмирования (значения должны быть больше 0)")
                        except Exception as e:
                            st.error(f"Ошибка вычисления Степенного закона: {str(e)}")
                    else:
                        st.error("Ошибка: Для этой модели в таблице обязаны присутствовать столбцы: CA, CB, r")
                        
                elif homo_model == "Arrhenius":
                    # محاولة توفير أعمدة بديلة لتجنب توقف الكود إذا كانت المسميات مختلفة قليلاً
                    if 'T' not in cleaned_df.columns and 'Время т (мин)' in cleaned_df.columns:
                        cleaned_df.rename(columns={'Время т (мин)': 'T'}, inplace=True)
                    if 'k' not in cleaned_df.columns and 'Оптическая плотность А' in cleaned_df.columns:
                        cleaned_df.rename(columns={'Оптическая плотность А': 'k'}, inplace=True)
                        
                    if all(c in cleaned_df.columns for c in ['T', 'k']):
                        try:
                            valid = cleaned_df[(cleaned_df['T'] > 0) & (cleaned_df['k'] > 0)]
                            if not valid.empty:
                                inv_T = 1.0 / valid['T'].values
                                log_k = np.log(valid['k'].values)
                                slope, intercept = np.polyfit(inv_T, log_k, 1)
                                Ea = -slope * 8.314 / 1000.0
                                A_arr = np.exp(intercept)
                                
                                st.markdown('<div class="custom-section-header"><h2>📋 Результаты расчета уравнения Аррениуса</h2></div>', unsafe_allow_html=True)
                                res_df = pd.DataFrame({
                                    "Параметр кинетики": ["Предэкспоненциальный множитель (A)", "Энергия активации процесса (Ea, кДж/моль)"],
                                    "Рассчитанное значение": [f"{A_arr:.2e}", f"{Ea:.2f}"]
                                })
                                st.dataframe(res_df, use_container_width=True)
                            else:
                                st.error("Данные должны быть положительными числами для уравнения Аррениуса.")
                        except Exception as e:
                            st.error(f"Ошибка вычисления уравнения Аррениуса: {str(e)}")
                    else:
                        st.error("Ошибка: Для расчета Аррениуса требуются столбцы 'T' (Температура) и 'k' (Константа). Проверьте или введите их вручную.")

        # ---------------------------------------------------------------------
        # ج. الأجزاء الباقية المستقرة لحفظ الهيكل
        # ---------------------------------------------------------------------
        elif reaction_type == "Гетерогенный катализ":
            st.info("Раздел 'Гетерогенный катализ' подготовлен и находится в режиме ожидания дополнительных алгоритмов.")
        elif reaction_type == "Ферментативные реакции":
            st.info("Раздел 'Ферментативные реакции' подготовлен и находится в режиме ожидания дополнительных алгоритмов.")


if __name__ == "__main__":
    main()
