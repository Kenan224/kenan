import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO

# =================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И СТИЛИЗАЦИЯ UI
# =================================================================
st.set_page_config(page_title="Кинетический Анализ Каталитических Процессов", layout="wide")

# Стили для красивого отображения карточек результатов
st.markdown("""
<style>
    .model-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid;
    }
    .mc-pfo { background-color: #f0f4f8; border-left-color: #2563eb; color: #1e293b; }
    .mc-pso { background-color: #f0fdf4; border-left-color: #16a34a; color: #1e293b; }
    .mc-highlight { background-color: #fefbec; border-left-color: #d97706; color: #1e293b; }
    .section-header {
        padding: 8px 12px;
        background-color: #f8fafc;
        border-radius: 6px;
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def section_header(css_class, icon, title):
    return f'<div class="section-header">{icon} {title}</div>'

def calculate_metrics(y_true, y_pred):
    """Расчет статистических метрик качества аппроксимации"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return r2, rmse

def apply_axis_style(ax):
    """Применение единого стиля для графиков"""
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def convert_df_to_excel(df):
    """Конвертация DataFrame в байты Excel для скачивания"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

# =================================================================
# 2. НЕИЗМЕНЯЕМЫЕ ЧАСТИ (ТФ 1 И ТФ 2) - ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ
# =================================================================
def render_interaction_1():
    st.info("ℹ️ Здесь находится ваш готовый код для Тка взаимодействия 1 (Определение порядка реакции).")
    # Вставьте ваш рабочий код Первого взаимодействия сюда

def render_interaction_2():
    st.info("ℹ️ Здесь находится ваш готовый код для Тка взаимодействия 2 (Уравнение Аррениуса).")
    # Вставьте ваш рабочий код Второго взаимодействия сюда

# =================================================================
# 3. МОДЕРНИЗИРОВАННОЕ ВЗАИМОДЕЙСТВИЕ 3: ГЕТЕРОГЕННЫЙ КАТАЛИЗ
# =================================================================
def render_heterogeneous():
    st.markdown("### 📊 Гетерогенный катализ: Кинетика адсорбционных процессов")
    
    # Селектор выбора модели или режима сравнения в одну строку
    het_model = st.radio(
        "Выберите режим работы:", 
        ["Langmuir-Hinshelwood", "Eley-Rideal", "📊 Сравнение моделей (Комплексный анализ)"], 
        index=0, horizontal=True, key="het_model_choice"
    )
    
    st.markdown(section_header("sh-data", "📥", "Ввод экспериментальных данных"), unsafe_allow_html=True)
    method = st.radio("Метод ввода данных:", ["✍️ Ввести вручную с нуля", "📁 Загрузить файл (Excel/CSV)"], horizontal=True, key="het_method")

    het_df = None
    if method == "📁 Загрузить файл (Excel/CSV)":
        uploaded_file = st.file_uploader("Выберите файл", type=['xlsx', 'csv'], key="het_file")
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    het_df = pd.read_csv(uploaded_file)
                else:
                    het_df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
    else:
        # Абсолютно пустая матрица/таблица из 6 строк для ввода с нуля без предустановленных цифр
        blank_data = pd.DataFrame({'C (Концентрация)': [None]*6, 'r (Скорость реакции)': [None]*6})
        st.write("Заполните столбцы концентрации (C) и скорости (r):")
        het_df = st.data_editor(blank_data, use_container_width=True, num_rows="dynamic", key="het_editor")

    if het_df is None:
        return

    # Очистка и извлечение данных
    het_df = het_df.dropna().copy()
    if het_df.empty:
        st.warning("⚠️ Таблица пуста. Пожалуйста, введите численные экспериментальные данные для начала расчета.")
        return

    try:
        C_data = pd.to_numeric(het_df.iloc[:, 0], errors='coerce').values
        r_data = pd.to_numeric(het_df.iloc[:, 1], errors='coerce').values
        
        # Фильтрация корректных физических значений (больше нуля)
        valid_idx = (C_data > 0) & (r_data > 0) & (~np.isnan(C_data)) & (~np.isnan(r_data))
        C_data, r_data = C_data[valid_idx], r_data[valid_idx]

        if len(C_data) < 2:
            st.error("❌ Недостаточно корректных точек данных (требуется минимум 2 точки с положительными значениями).")
            return

        # Уравнения моделей
        def lh_model(C, k, K): return (k * K * C) / (1.0 + K * C)
        def er_model(C, k, theta): return k * theta * C

        C_fine = np.linspace(max(0.001, min(C_data)*0.9), max(C_data) * 1.1, 300)
        
        # Переменные для результатов расчета
        fit_lh_success, fit_er_success = False, False
        
        # 1. Расчет Langmuir-Hinshelwood
        try:
            popt_lh, _ = curve_fit(lh_model, C_data, r_data, p0=[max(r_data), 1.0], bounds=(0, np.inf), maxfev=5000)
            k_lh, K_lh = popt_lh[0], popt_lh[1]
            r_pred_lh = lh_model(C_data, k_lh, K_lh)
            r2_lh, rmse_lh = calculate_metrics(r_data, r_pred_lh)
            fit_lh_success = True
        except: pass

        # 2. Расчет Eley-Rideal
        try:
            popt_er, _ = curve_fit(er_model, C_data, r_data, p0=[max(r_data)/max(C_data), 0.5], bounds=(0, [np.inf, 1.0]), maxfev=5000)
            k_er, theta_er = popt_er[0], popt_er[1]
            r_pred_er = er_model(C_data, k_er, theta_er)
            r2_er, rmse_er = calculate_metrics(r_data, r_pred_er)
            fit_er_success = True
        except: pass

        # --- РЕЖИМ 1: ОДИНОЧНЫЙ ВЫВОД LANGMUIR-HINSHELWOOD ---
        if het_model == "Langmuir-Hinshelwood":
            if not fit_lh_success:
                st.error("❌ Не удалось аппроксимировать данные моделью Langmuir-Hinshelwood.")
                return
            st.markdown(section_header("sh-res", "📋", "Результаты: Модель Langmuir-Hinshelwood"), unsafe_allow_html=True)
            st.markdown(f'<div class="model-card mc-pfo"><h4>Параметры Langmuir-Hinshelwood:</h4><p><b>k (Константа скорости):</b> {k_lh:.4f}</p><p><b>K (Константа адсорбции):</b> {K_lh:.4f}</p><p><b>R² (Коэффициент детерминации):</b> {r2_lh:.4f}</p><p><b>RMSE (Стандартная ошибка):</b> {rmse_lh:.5f}</p></div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(C_data, r_data, color='#ef4444', label='Эксперимент', zorder=3)
            ax.plot(C_fine, lh_model(C_fine, k_lh, K_lh), color='#2563eb', linewidth=2, label='L-H Аппроксимация')
            ax.set_xlabel('Концентрация (C)'); ax.set_ylabel('Скорость реакции (r)'); apply_axis_style(ax); ax.legend()
            st.pyplot(fig)

        # --- РЕЖИМ 2: ОДИНОЧНЫЙ ВЫВОД ELEY-RIDEAL ---
        elif het_model == "Eley-Rideal":
            if not fit_er_success:
                st.error("❌ Не удалось аппроксимировать данные моделью Eley-Rideal.")
                return
            st.markdown(section_header("sh-res", "📋", "Результаты: Модель Eley-Rideal"), unsafe_allow_html=True)
            st.markdown(f'<div class="model-card mc-pso"><h4>Параметры Eley-Rideal:</h4><p><b>k (Константа скорости):</b> {k_er:.4f}</p><p><b>θ (Степень заполнения поверхности):</b> {theta_er:.4f}</p><p><b>R² (Коэффициент детерминации):</b> {r2_er:.4f}</p><p><b>RMSE (Стандартная ошибка):</b> {rmse_er:.5f}</p></div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(C_data, r_data, color='#ef4444', label='Эксперимент', zorder=3)
            ax.plot(C_fine, er_model(C_fine, k_er, theta_er), color='#16a34a', linestyle='--', linewidth=2, label='E-R Аппроксимация')
            ax.set_xlabel('Концентрация (C)'); ax.set_ylabel('Скорость реакции (r)'); apply_axis_style(ax); ax.legend()
            st.pyplot(fig)

        # --- РЕЖИМ 3: СРАВНЕНИЕ МОДЕЛЕЙ (НА ОДНОЙ СТРАНИЦЕ) ---
        else:
            if not (fit_lh_success and fit_er_success):
                st.error("❌ Для проведения сравнения обе модели должны успешно сойтись на ваших данных.")
                return
            
            st.markdown(section_header("sh-comp", "🔄", "Сравнительный анализ кинетических моделей"), unsafe_allow_html=True)
            
            # Таблица сравнения метрик и значений side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="model-card mc-pfo"><h4>1. Langmuir-Hinshelwood</h4><p><b>k:</b> {k_lh:.4f}</p><p><b>K:</b> {K_lh:.4f}</p><hr><p><b>R²:</b> {r2_lh:.4f}</p><p><b>RMSE:</b> {rmse_lh:.5f}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="model-card mc-pso"><h4>2. Eley-Rideal</h4><p><b>k:</b> {k_er:.4f}</p><p><b>θ (Степень заполнения):</b> {theta_er:.4f}</p><hr><p><b>R²:</b> {r2_er:.4f}</p><p><b>RMSE:</b> {rmse_er:.5f}</p></div>', unsafe_allow_html=True)
            
            # Совмещенный график
            st.markdown(section_header("sh-graph", "📈", "Совмещенный график аппроксимации"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.scatter(C_data, r_data, color='#ef4444', fontdict={'weight': 'bold'}, label='Экспериментальные данные', s=45, zorder=5)
            ax.plot(C_fine, lh_model(C_fine, k_lh, K_lh), color='#2563eb', linewidth=2.5, label='Langmuir-Hinshelwood')
            ax.plot(C_fine, er_model(C_fine, k_er, theta_er), color='#16a34a', linestyle='--', linewidth=2.5, label='Eley-Rideal')
            ax.set_xlabel('Концентрация реагента (C)')
            ax.set_ylabel('Скорость каталитического процесса (r)')
            apply_axis_style(ax)
            ax.legend(loc='best')
            st.pyplot(fig)

            # Экспертное заключение автоматического аналитика
            st.markdown(section_header("sh-conclusion", "🧠", "Аналитическое заключение системы"), unsafe_allow_html=True)
            best_model = "Langmuir-Hinshelwood" if r2_lh > r2_er else "Eley-Rideal"
            r2_diff = abs(r2_lh - r2_er)
            
            conclusion_text = f"""
            На основании проведенного математического анализа экспериментальных точек установлено, что модель **{best_model}** более адекватно описывает кинетику данного процесса.
            *   Модель **{best_model}** демонстрирует более высокий коэффициент детерминации ($R^2 = {max(r2_lh, r2_er):.4f}$) и наименьшую стандартную ошибку ($RMSE$).
            *   Разница по критерию $R^2$ составляет **{r2_diff:.4f}**. 
            
            *Рекомендация:* Использовать параметры кинетического уравнения **{best_model}** для дальнейшего масштабирования химического реактора и симуляции процесса.
            """
            st.markdown(f'<div class="model-card mc-highlight">{conclusion_text}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Критическая ошибка вычислений: {str(e)}")

# =================================================================
# 4. МОДЕРНИЗИРОВАННОЕ ВЗАИМОДЕЙСТВИЕ 4: ФЕРМЕНТАТИВНЫЕ РЕАКЦИИ
# =================================================================
def render_enzymatic():
    st.markdown("### 🧬 Биокатализ: Ферментативная кинетика")
    
    enz_model = st.radio(
        "Выберите режим работы:", 
        ["Michaelis-Menten", "Hill Model", "📊 Сравнение моделей (Комплексный анализ)"], 
        index=0, horizontal=True, key="enz_model_choice"
    )
    
    st.markdown(section_header("sh-data-enz", "📥", "Ввод экспериментальных данных фермента"), unsafe_allow_html=True)
    method = st.radio("Метод ввода данных:", ["✍️ Ввести вручную с нуля", "📁 Загрузить файл (Excel/CSV)"], horizontal=True, key="enz_method")

    enz_df = None
    if method == "📁 Загрузить файл (Excel/CSV)":
        uploaded_file = st.file_uploader("Выберите файл", type=['xlsx', 'csv'], key="enz_file")
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    enz_df = pd.read_csv(uploaded_file)
                else:
                    enz_df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
    else:
        # Абсолютно пустая таблица для ввода биокаталитических данных с нуля
        blank_data = pd.DataFrame({'[S] (Концентрация субстрата)': [None]*6, 'v (Начальная скорость)': [None]*6})
        st.write("Заполните столбцы концентрации субстрата ([S]) и начальной скорости реакции (v):")
        enz_df = st.data_editor(blank_data, use_container_width=True, num_rows="dynamic", key="enz_editor")

    if enz_df is None:
        return

    enz_df = enz_df.dropna().copy()
    if enz_df.empty:
        st.warning("⚠️ Таблица пуста. Пожалуйста, введите экспериментальные данные кинетики ферментов для начала расчета.")
        return

    try:
        S_data = pd.to_numeric(enz_df.iloc[:, 0], errors='coerce').values
        v_data = pd.to_numeric(enz_df.iloc[:, 1], errors='coerce').values
        
        valid_idx = (S_data > 0) & (v_data > 0) & (~np.isnan(S_data)) & (~np.isnan(v_data))
        S_data, v_data = S_data[valid_idx], v_data[valid_idx]

        if len(S_data) < 3:
            st.error("❌ Недостаточно корректных данных (Для модели Хилла требуется минимум 3 точки с положительными значениями).")
            return

        # Уравнения ферментативной кинетики
        def mm_model(S, Vmax, Km): return (Vmax * S) / (Km + S)
        def hill_model(S, Vmax, K05, n): return (Vmax * (S ** n)) / ((K05 ** n) + (S ** n))

        S_fine = np.linspace(max(0.001, min(S_data)*0.9), max(S_data) * 1.15, 300)
        fit_mm_success, fit_hill_success = False, False

        # 1. Расчет Михаэлиса-Ментен
        try:
            popt_mm, _ = curve_fit(mm_model, S_data, v_data, p0=[max(v_data), np.median(S_data)], bounds=(0, np.inf), maxfev=5000)
            Vmax_mm, Km_mm = popt_mm[0], popt_mm[1]
            v_pred_mm = mm_model(S_data, Vmax_mm, Km_mm)
            r2_mm, rmse_mm = calculate_metrics(v_data, v_pred_mm)
            fit_mm_success = True
        except: pass

        # 2. Расчет модели Хилла
        try:
            popt_hill, _ = curve_fit(hill_model, S_data, v_data, p0=[max(v_data), np.median(S_data), 1.0], bounds=(0, [np.inf, np.inf, 10.0]), maxfev=5000)
            Vmax_hl, K05_hl, n_hl = popt_hill[0], popt_hill[1], popt_hill[2]
            v_pred_hl = hill_model(S_data, Vmax_hl, K05_hl, n_hl)
            r2_hl, rmse_hl = calculate_metrics(v_data, v_pred_hl)
            fit_hill_success = True
        except: pass

        # --- РЕЖИМ 1: ОДИНОЧНЫЙ ВЫВОД MICHAELIS-MENTEN ---
        if enz_model == "Michaelis-Menten":
            if not fit_mm_success:
                st.error("❌ Не удалось аппроксимировать данные моделью Михаэлиса-Ментен.")
                return
            st.markdown(section_header("sh-res", "📋", "Результаты: Модель Michaelis-Menten"), unsafe_allow_html=True)
            st.markdown(f'<div class="model-card mc-pfo"><h4>Параметры Михаэлиса-Ментен:</h4><p><b>Vmax (Максимальная скорость):</b> {Vmax_mm:.4f}</p><p><b>Km (Константа Михаэлиса):</b> {Km_mm:.4f}</p><p><b>R² (Коэффициент детерминации):</b> {r2_mm:.4f}</p><p><b>RMSE:</b> {rmse_mm:.5f}</p></div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(S_data, v_data, color='#ef4444', label='Эксперимент', zorder=3)
            ax.plot(S_fine, mm_model(S_fine, Vmax_mm, Km_mm), color='#2563eb', linewidth=2, label='M-M Аппроксимация')
            ax.set_xlabel('Концентрация субстрата [S]'); ax.set_ylabel('Скорость реакции (v)'); apply_axis_style(ax); ax.legend()
            st.pyplot(fig)

        # --- РЕЖИМ 2: ОДИНОЧНЫЙ ВЫВОД HILL MODEL ---
        elif enz_model == "Hill Model":
            if not fit_hill_success:
                st.error("❌ Не удалось аппроксимировать данные моделью Хилла.")
                return
            st.markdown(section_header("sh-res", "📋", "Результаты: Модель Хилла (Hill Model)"), unsafe_allow_html=True)
            
            coop_status = "Положительная аллостерическая кооперативность (n > 1)" if n_hl > 1.05 else ("Отрицательная кооперативность (n < 1)" if n_hl < 0.95 else "Некооперативное поведение (n ≈ 1)")
            st.markdown(f'<div style="background-color: #f3e8ff; color: #6b21a8; padding: 10px; border-radius: 5px; font-weight: bold; margin-bottom: 10px;">🧬 Характер взаимодействия: {coop_status}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="model-card mc-pso"><h4>Параметры уравнения Хилла:</h4><p><b>Vmax (Максимальная скорость):</b> {Vmax_hl:.4f}</p><p><b>K₀.₅ (Константа полунасыщения):</b> {K05_hl:.4f}</p><p><b>n (Коэффициент Хилла - степень кооперативности):</b> {n_hl:.4f}</p><p><b>R²:</b> {r2_hl:.4f}</p><p><b>RMSE:</b> {rmse_hl:.5f}</p></div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(S_data, v_data, color='#ef4444', label='Эксперимент', zorder=3)
            ax.plot(S_fine, hill_model(S_fine, Vmax_hl, K05_hl, n_hl), color='#a855f7', linewidth=2, label='Hill Аппроксимация')
            ax.set_xlabel('Концентрация субстрата [S]'); ax.set_ylabel('Скорость реакции (v)'); apply_axis_style(ax); ax.legend()
            st.pyplot(fig)

        # --- РЕЖИМ 3: СРАВНЕНИЕ ФЕРМЕНТАТИВНЫХ МОДЕЛЕЙ ---
        else:
            if not (fit_mm_success and fit_hill_success):
                st.error("❌ Для сравнения необходимо успешное выполнение расчетов по обоим алгоритмам.")
                return
            
            st.markdown(section_header("sh-comp-enz", "🔄", "Сравнительный анализ биокаталитических моделей"), unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="model-card mc-pfo"><h4>1. Michaelis-Menten</h4><p><b>Vmax:</b> {Vmax_mm:.4f}</p><p><b>Km:</b> {Km_mm:.4f}</p><p><b>n (фиксирован):</b> 1.0000</p><hr><p><b>R²:</b> {r2_mm:.4f}</p><p><b>RMSE:</b> {rmse_mm:.5f}</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="model-card mc-pso"><h4>2. Hill Model</h4><p><b>Vmax:</b> {Vmax_hl:.4f}</p><p><b>K₀.₅:</b> {K05_hl:.4f}</p><p><b>n (Коэф. Хилла):</b> {n_hl:.4f}</p><hr><p><b>R²:</b> {r2_hl:.4f}</p><p><b>RMSE:</b> {rmse_hl:.5f}</p></div>', unsafe_allow_html=True)
            
            # Совмещенный кинетический график
            st.markdown(section_header("sh-graph-enz", "📈", "Кривые насыщения фермента"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.scatter(S_data, v_data, color='#ef4444', label='Эксперимент (Точки)', s=45, zorder=5)
            ax.plot(S_fine, mm_model(S_fine, Vmax_mm, Km_mm), color='#2563eb', linewidth=2.5, label='Михаэлис-Ментен (Классика)')
            ax.plot(S_fine, hill_model(S_fine, Vmax_hl, K05_hl, n_hl), color='#a855f7', linestyle='-.', linewidth=2.5, label='Модель Хилла (Аллостерическая)')
            ax.set_xlabel('Концентрация растворенного субстрата [S]')
            ax.set_ylabel('Начальная кинетическая скорость (v)')
            apply_axis_style(ax)
            ax.legend(loc='best')
            st.pyplot(fig)

            # Экспертный разбор кинетики фермента
            st.markdown(section_header("sh-conclusion-enz", "🧠", "Кинетический вывод аналитической системы"), unsafe_allow_html=True)
            best_enz = "Michaelis-Menten" if r2_mm > r2_hl else "Hill Model"
            
            analysis_text = f"""
            Анализ кинетического механизма ферментативного процесса показал следующие результаты:
            1.  Математически наиболее точной признана **{best_enz}** ($R^2 = {max(r2_mm, r2_hl):.4f}$).
            2.  Рассчитанный коэффициент Хилла составляет **n = {n_hl:.3f}**. 
            """
            if n_hl > 1.1:
                analysis_text += f"\n*   **Интерпретация кооперативности:** Значение $n > 1$ явно указывает на наличие **положительной кооперативности** в биосистеме. Фермент имеет несколько активных центров, демонстрирующих синергетический эффект при связывании субстрата. Классическое уравнение Михаэлиса-Ментен здесь менее применимо."
            elif n_hl < 0.9:
                analysis_text += f"\n*   **Интерпретация кооперативности:** Значение $n < 1$ свидетельствует об **отрицательной кооперативности** (негативный аллостерический эффект), что замедляет прирост скорости при высоких дозировках субстрата."
            else:
                analysis_text += f"\n*   **Интерпретация кооперативности:** Так как $n \\approx 1$, кинетика близка к классической гиперболической форме, и использование простой модели Михаэлиса-Ментен является физически обоснованным."

            st.markdown(f'<div class="model-card mc-highlight">{analysis_text}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Ошибка обработки биокаталитических вычислений: {str(e)}")

# =================================================================
# 5. ГЛАВНАЯ АРХИТЕКТУРА И МАРШРУТИЗАЦИЯ ПРИЛОЖЕНИЯ
# =================================================================
def main():
    st.sidebar.title("⚙️ Панель управления")
    st.sidebar.markdown("### Программный комплекс определения кинетических параметров реакций")
    
    # Главное навигационное меню
    choice = st.sidebar.radio(
        "Выберите блок анализа:",
        [
            "1. Порядок реакции (Вручную/Файл)",
            "2. Уравнение Аррениуса (Энергия активации)",
            "3. Гетерогенный катализ (L-H против E-R)",
            "4. Ферментативная кинетика (M-M против Hill)"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Универсальный расчетный модуль для автоматизированного анализа химических и биокаталитических процессов.")

    # Вызов соответствующих разделов
    if choice == "1. Порядок реакции (Вручную/Файл)":
        render_interaction_1()
    elif choice == "2. Уравнение Аррениуса (Энергия активации)":
        render_interaction_2()
    elif choice == "3. Гетерогенный катализ (L-H против E-R)":
        render_heterogeneous()
    elif choice == "4. Ферментативная кинетика (M-M против Hill)":
        render_enzymatic()

if __name__ == "__main__":
    main()
