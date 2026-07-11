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

# 🎨 سي اس اس صارم لإجبار جميع النصوص على اللون الأسود ومنع تداخل الألوان تماماً
st.markdown("""
<style>
/* تعيين خلفية التطبيق لتكون بيضاء */
.stApp {
    background-color: #ffffff !important;
}

/* تصميم الهامش الجانبي الفاتح */
[data-testid="stSidebar"] {
    background-color: #f4f6f9 !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* إجبار كافة نصوص الهامش الجانبي على اللون الداكن */
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
    color: #1e293b !important;
}

/* إجبار كافة النصوص العادية في الصفحة الرئيسية على اللون الأسود */
.main p, .main span, .main label, .main h3, .stRadio label {
    color: #000000 !important;
}

/* حل مشكلة اختفاء النصوص: إجبار صناديق الاختيار، القوائم المنسدلة، وحقول الرفع على خلفية فاتحة ونص أسود */
div[data-baseweb="select"], 
div[data-testid="stFileUploaderDropzone"], 
.stSelectbox div, 
input, 
div[role="radiogroup"] {
    background-color: #f1f5f9 !important;
    color: #000000 !important;
    border: 1px solid #cbd5e1 !important;
}

/* التأكيد على اللون الأسود لجميع العناصر الداخلية للصناديق */
div[data-baseweb="select"] *, 
div[data-testid="stFileUploaderDropzone"] * {
    color: #000000 !important;
}

/* الهيدر الأزرق الرئيسي العريض */
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
    color: #ffffff !important;
}

/* بطاقة معلومات الطالب والمشرف */
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

/* العناوين الفرعية العريضة للsections */
.custom-section-header {
    background-color: #cfe2ff;
    border-left: 6px solid #1d63ed;
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.custom-section-header h2 {
    color: #084298 !important;
    margin: 0;
    font-size: 1.6rem;
    font-weight: 500;
}

/* شريط التنبيهات والمعلومات الأزرق */
.info-banner {
    background-color: #e2eefd;
    color: #0a58ca !important;
    padding: 0.8rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 1.2rem;
    font-size: 0.95rem;
    border: 1px solid #b6d4fe;
}

/* تصميم الأزرار الزرقاء الكبيرة لضمان ظهور النص داخلها باللون الأبيض */
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
    # الهيدر العلوي ومعلومات الطالب
    st.markdown('<div class="main-header"><h1>Анализ кинетического моделирования</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="student-card"><p><strong>СТУДЕНТ:</strong> Алсади К.</p><p><strong>РУКОВОДИТЕЛЬ:</strong> Киреева А.В</p></div>', unsafe_allow_html=True)

    # 1. بناء الهامش الجانبي (Sidebar) حسب الترتيب المنطقي المطلوب
    with st.sidebar:
        # نوع التفاعل أولاً في الأعلى تماماً
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
        
        # متطلبات الملف والمدخلات والنواتج والصيغ المدعومة
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
            * Зависят от выбранной кинетической модели (Концентрации CA, CB или Температура T)
            """)
            
        st.markdown("""
        **Поддерживаемые форматы:**
        * Excel (.xlsx)
        * CSV (.csv) с автоопределением
        """)
        
        # حاوية فارغة محجوزة ديناميكياً لـ Excel Sheet Selector (تظهر فقط عند تحقق الشرط الحتمي)
        sheet_selector_placeholder = st.empty()

    # =========================================================================
    # القسم الأول: التحليل الكينتيكي الضوئي (Фотокаталитические реакции)
    # =========================================================================
    if reaction_type == "Фотокаталитические реакции":
        st.markdown('<div class="custom-section-header"><h2>📊 Ввод данных для фотокатализа</h2></div>', unsafe_allow_html=True)
        
        input_method = st.radio("Выберите способ ввода данных:", ["Загрузить файл", "Ввести данные вручную"], index=1, horizontal=True)
        df_photo = None

        if input_method == "Загрузить файл":
            uploaded_file = st.file_uploader("Загрузите файл для анализа фотокатализа", type=['xlsx', 'csv'])
            if uploaded_file is not None:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'csv':
                        df_photo = read_csv_file(uploaded_file)
                    else:
                        excel_file = pd.ExcelFile(uploaded_file)
                        sheet_names = excel_file.sheet_names
                        selected_sheet = sheet_names[0]
                        
                        # تظهر القائمة فقط وحصرياً إذا تضمن ملف الاكسل أكثر من صفحة
                        if len(sheet_names) > 1:
                            with sheet_selector_placeholder.container():
                                st.markdown("---")
                                selected_sheet = st.selectbox("Выберите лист Excel", sheet_names, index=0)
                        
                        df_photo = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

                    # التحقق من هيكلية أعمدة الفوتوكاتاليز
                    is_valid, error_message = validate_data_structure(df_photo)
                    if not is_valid:
                        st.error(f"{error_message}")
                        return

                    df_photo.columns = df_photo.columns.str.strip()
                    if 'А0' not in df_photo.columns and 'А' in df_photo.columns and len(df_photo) > 0:
                        df_photo['А'] = pd.to_numeric(df_photo['А'], errors='coerce')
                        df_photo['А0'] = df_photo['А'].iloc[0]
                    if 'А/А0' not in df_photo.columns and 'А' in df_photo.columns and 'А0' in df_photo.columns:
                        df_photo['А/А0'] = df_photo['А'] / df_photo['А0']

                    df_photo = st.data_editor(df_photo, use_container_width=True, num_rows="dynamic")
                except Exception as e:
                    st.error(f"Ошибка чтения файла: {str(e)}")
        else:
            st.markdown('<div class="info-banner">Введите данные процесса в таблицу नीचे. Столбец А/А0 рассчитается автоматически.</div>', unsafe_allow_html=True)
            init_cols = ['Время (мин)', 'Оптическая плотность А']
            default_data = pd.DataFrame(columns=init_cols, data=[[0.0, 0.0000]])
            edited_data = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
            
            if not edited_data.empty and len(edited_data) > 0:
                raw_t = pd.to_numeric(edited_data.iloc[:, 0], errors='coerce')
                raw_a = pd.to_numeric(edited_data.iloc[:, 1], errors='coerce')
                valid_mask = (raw_a >= 0) & (raw_t >= 0)
                if valid_mask.any():
                    df_photo = pd.DataFrame({'т, мин': raw_t[valid_mask], 'А': raw_a[valid_mask]})
                    if len(df_photo) > 0 and df_photo.iloc[0]['А'] > 0:
                        df_photo['A0'] = df_photo.iloc[0]['А']
                        df_photo['А/А0'] = df_photo['А'] / df_photo['A0']

        # معالجة النتائج تلقائياً للـ 3 نماذج معاً للفوتوكاتاليز عند ضغط الزر
        if df_photo is not None and not df_photo.empty:
            if st.button("Анализировать введенные данные"):
                processed_df = preprocess_data(df_photo)
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
                    
                    # قسم تصدير وتحميل النتائج (أزرار التحميل المزدوجة بملف إكسل وصورة)
                    st.markdown('### 💾 Экспорт и сохранение результатов')
                    dl_col1, dl_col2 = st.columns(2)
                    
                    excel_data = convert_df_to_excel(results_summary)
                    dl_col1.download_button(
                        label="📊 Скачать результаты в Excel",
                        data=excel_data,
                        file_name="Photocatalytic_Kinetic_Results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    img_buf = BytesIO()
                    fig_main.savefig(img_buf, format="png", bbox_inches="tight")
                    img_buf.seek(0)
                    dl_col2.download_button(
                        label="🖼️ Скачать графики в PNG",
                        data=img_buf,
                        file_name="Photocatalytic_Plots.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Ошибка вычислений: {str(e)}")

    # =========================================================================
    # القسم الثاني: الكاتاليز المتجانس (Гомогенный катализ) - يعمل بكفاءة مستقلة تماماً
    # =========================================================================
    elif reaction_type == "Гомогенный катализ":
        st.markdown('<div class="custom-section-header"><h2>📊 Кинетика гомогенного катализа</h2></div>', unsafe_allow_html=True)
        
        # اختيار النموذج الحركي المعتمد أولاً قبل معالجة وإدخال الجدول لضمان توافق الأعمدة
        homo_model = st.radio(
            "Выберите модель для проведения вычислений:",
            ["Power-law (степенной закон)", "Arrhenius", "Последовательные реакции"],
            index=0, horizontal=True
        )

        input_method = st.radio("Выберите способ ввода данных для гомогенного катализа:", ["Загрузить файл", "Ввести данные вручную"], index=1, horizontal=True)
        df_homo = None

        # تحديد الأعمدة الافتراضية المناسبة لكل نموذج حركي منعاً للمشاكل البرمجية
        if homo_model == "Power-law (степенной закон)":
            cols = ['t', 'CA', 'CB', 'r']
            rows = [[0.0, 1.0, 1.5, 0.05], [5.0, 0.8, 1.3, 0.035]]
        elif homo_model == "Arrhenius":
            cols = ['T', 'k']
            rows = [[298.0, 0.01], [308.0, 0.02]]
        else:
            cols = ['t', 'CA', 'CB', 'CC']
            rows = [[0.0, 1.0, 0.0, 0.0], [10.0, 0.5, 0.4, 0.1]]

        if input_method == "Загрузить файл":
            uploaded_file = st.file_uploader("Загрузите файл для гомогенного катализа", type=['xlsx', 'csv'])
            if uploaded_file is not None:
                try:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension == 'csv':
                        df_homo = read_csv_file(uploaded_file)
                    else:
                        excel_file = pd.ExcelFile(uploaded_file)
                        sheet_names = excel_file.sheet_names
                        selected_sheet = sheet_names[0]
                        
                        if len(sheet_names) > 1:
                            with sheet_selector_placeholder.container():
                                st.markdown("---")
                                selected_sheet = st.selectbox("Выберите лист Excel", sheet_names, index=0)
                                
                        df_homo = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    
                    df_homo = clean_homogeneous_data(df_homo)
                    df_homo = st.data_editor(df_homo, use_container_width=True, num_rows="dynamic")
                except Exception as e:
                    st.error(f"Ошибка загрузки: {str(e)}")
        else:
            st.markdown(f'<div class="info-banner">Введите экспериментальные данные для модели <b>{homo_model}</b> в таблицу نیچے:</div>', unsafe_allow_html=True)
            empty_df = pd.DataFrame(columns=cols, data=rows)
            df_homo = st.data_editor(empty_df, use_container_width=True, num_rows="dynamic")

        # تشغيل وإتمام عملية الحساب الرياضي للقسم الثاني بشكل مستقل ومضمون
        if df_homo is not None and len(df_homo) > 0:
            if st.button("Рассчитать параметры гомогенного катализа"):
                cleaned_df = clean_homogeneous_data(df_homo)
                
                if homo_model == "Power-law (степенной закон)" and all(c in cleaned_df.columns for c in ['CA', 'CB', 'r']):
                    try:
                        valid = cleaned_df[(cleaned_df['CA'] > 0) & (cleaned_df['CB'] > 0) & (cleaned_df['r'] > 0)]
                        if not valid.empty:
                            log_CA, log_CB, log_r = np.log(valid['CA'].values), np.log(valid['CB'].values), np.log(valid['r'].values)
                            X = np.column_stack((np.ones_like(log_CA), log_CA, log_CB))
                            beta, _, _, _ = np.linalg.lstsq(X, log_r, rcond=None)
                            k_val, a_val, b_val = np.exp(beta[0]), beta[1], beta[2]
                            
                            st.markdown('<div class="custom-section-header"><h2>📋 Результаты расчета Степенного закона</h2></div>', unsafe_allow_html=True)
                            res_df = pd.DataFrame({
                                "Параметр": ["Константа скорости (k)", "Порядок по компоненту A (α)", "Порядок по компоненту B (β)"],
                                "Значение": [f"{k_val:.4f}", f"{a_val:.2f}", f"{b_val:.2f}"]
                            })
                            st.dataframe(res_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Ошибка математического анализа: {str(e)}")
                        
                elif homo_model == "Arrhenius" and all(c in cleaned_df.columns for c in ['T', 'k']):
                    try:
                        valid = cleaned_df[(cleaned_df['T'] > 0) & (cleaned_df['k'] > 0)]
                        if not valid.empty:
                            inv_T = 1.0 / valid['T'].values
                            log_k = np.log(valid['k'].values)
                            slope, intercept = np.polyfit(inv_T, log_k, 1)
                            Ea = -slope * 8.314 / 1000.0
                            A = np.exp(intercept)
                            
                            st.markdown('<div class="custom-section-header"><h2>📋 Результаты расчета уравнения Аррениуса</h2></div>', unsafe_allow_html=True)
                            res_df = pd.DataFrame({
                                "Параметр": ["Предэкспоненциальный множитель (A)", "Энергия активации (Ea, кДж/моль)"],
                                "Значение": [f"{A:.2e}", f"{Ea:.2f}"]
                            })
                            st.dataframe(res_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Ошибка математического анализа: {str(e)}")
                else:
                    st.warning("Убедитесь, что введенные столбцы соответствуют требованиям выбранной модели.")

    # =========================================================================
    # بقية أقسام التفاعلات (المحافظة عليها بشكل سليم ومستقر من الكود الأساسي)
    # =========================================================================
    elif reaction_type == "Гетерогенный катализ":
        st.info("Раздел 'Гетерогенный катализ' сохранен в рабочем состоянии и ожидает расширения ваших алгоритмов.")
    elif reaction_type == "Ферментативные реакции":
        st.info("Раздел 'Ферментативные реакции' сохранен в рабочем состоянии и ожидает расширения ваших алгоритмов.")


if __name__ == "__main__":
    main()
