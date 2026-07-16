"""
Streamlit Web Application for Kinetic Modeling Analysis
Универсальная программная платформа для анализа кинетического моделирования
(Merged & OCR-Activated Version)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

# OCR specific imports (Already in your app.py)
import easyocr
from PIL import Image
import cv2

# =============================================================================
# الجزء الأول: الأكواد المدمجة (كانت سابقاً في الملفات المساعدة)
# =============================================================================

# --- مدمج من data_processor.py ---
def validate_data_structure(df):
    """Verifies if the dataframe has the required columns for Photocatalysis."""
    required_cols = ['т, мин', 'А']
    msg = ""
    # Check if columns exist
    if not all(col in df.columns for col in required_cols):
        msg = f"Неверная структура. Файл должен содержать столбцы: {', '.join(required_cols)}"
        return False, msg
    
    # Check if they are numeric
    try:
        df['т, мин'] = pd.to_numeric(df['т, мин'], errors='raise')
        df['А'] = pd.to_numeric(df['А'], errors='raise')
    except Exception:
        msg = "Столбцы данных содержат нечисловые значения."
        return False, msg
        
    if df['т, мин'].isnull().any() or df['А'].isnull().any():
        msg = "Столбцы данных содержат пустые ячейки."
        return False, msg
        
    return True, msg

def preprocess_data(df):
    """Cleans data, handles initial values, and prepares logarithmic terms."""
    # Work on a copy to avoid SettingWithCopy warnings on the original input
    cleaned_df = df.dropna().copy()
    
    # Ensure strict numeric conversion, coercing errors to NaN and then dropping them
    # This is crucial for data coming from manual input or imperfect OCR
    for col in cleaned_df.columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    cleaned_df = cleaned_df.dropna()
    
    # Filter valid kinetic data (Time >=0, Absorbance > 0 for log)
    # Standard columns are expected: 1st is Time, 2nd is Absorbance
    if len(cleaned_df.columns) < 2: return pd.DataFrame() # Safety check
    
    t_col = cleaned_df.columns[0]
    a_col = cleaned_df.columns[1]
    
    cleaned_df = cleaned_df[(cleaned_df[t_col] >= 0) & (cleaned_df[a_col] > 0)]
    
    if cleaned_df.empty:
        return cleaned_df

    # Initial Absorbance (A0) logic
    if 'А0' not in cleaned_df.columns:
        # If not provided, assume first point is A0
        cleaned_df['А0'] = cleaned_df[a_col].iloc[0]
    
    cleaned_df = cleaned_df.dropna() # final drop after A0 assignment
    
    # Prepare kinetic terms
    cleaned_df['А/А0'] = cleaned_df[a_col] / cleaned_df['А0']
    cleaned_df['ln_A_A0'] = np.log(cleaned_df['А/А0'])
    cleaned_df['1/A'] = 1.0 / cleaned_df[a_col]
    
    # Normalize column names for downstream modules
    # Map whatever the first two columns were to 'т, мин' and 'А'
    mapping = {t_col: 'т, мин', a_col: 'А'}
    return cleaned_df.rename(columns=mapping)

def get_data_summary(df):
    """Calculates basic statistics of the preprocessed data."""
    if df.empty:
        return None
    summary = {
        "total_points": len(df),
        "time_range": (df['т, мин'].min(), df['т, мин'].max()),
        "a_a0_range": (df['А/А0'].min(), df['А/А0'].max()),
        "a0_value": df['А0'].iloc[0]
    }
    return summary

def read_csv_file(uploaded_file):
    """Attempts to read CSV with different separators."""
    # First try standard comma
    try:
        uploaded_file.seek(0) # reset pointer
        df = pd.read_csv(uploaded_file)
        if len(df.columns) > 1: return df
    except: pass
    
    # Try semicolon
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';')
        if len(df.columns) > 1: return df
    except: pass
    
    # Try tab
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep='\t')
        if len(df.columns) > 1: return df
    except: pass
    
    # Fallback to default, likely will fail structure validation later
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

def clean_homogeneous_data(df):
    """Attempts to clean and standardise homogeneous kinetics data structure."""
    if df is None or df.empty: return df
    df = df.copy()
    
    # Remove unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Ensure numerical data
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df.dropna(how='all') # drop rows where all are NaN

# --- مدمج من kinetic_models.py ---
def calculate_metrics_internal(y_true, y_pred):
    """Calculates R2 and MAPE between true and predicted values."""
    if len(y_true) < 2: return 0.0, 0.0
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
        
    return r2, mape

def fit_zo_model(df):
    """Fits Zero Order kinetics: (A0 - A) = k0 * t"""
    if df.empty: return 0.0, [], 0.0, 0.0
    
    # Linear fit: y = k * x (passing through origin for changes)
    y_data = (df['А0'] - df['А']).values
    x_data = df['т, мин'].values
    
    # Define linear function through origin
    def zo_func(t, k): return k * t
    
    try:
        popt, _ = curve_fit(zo_func, x_data, y_data)
        k0 = popt[0]
        
        # Zo prediction is A = A0 - k0*t
        predictions = df['А0'].values - zo_func(x_data, k0)
        r2, mape = calculate_metrics_internal(df['А'].values, predictions)
        return k0, predictions, mape, r2
    except:
        return 0.0, [], 0.0, 0.0

def fit_pfo_model(df):
    """Fits Pseudo-First Order kinetics: -ln(A/A0) = k1 * t"""
    if df.empty or 'ln_A_A0' not in df.columns: return 0.0, [], 0.0, 0.0
    
    y_data = -df['ln_A_A0'].values
    x_data = df['т, мин'].values
    
    def pfo_func(t, k): return k * t
    
    try:
        popt, _ = curve_fit(pfo_func, x_data, y_data)
        k1 = popt[0]
        # PFO prediction is A = A0 * exp(-k1 * t)
        predictions = df['А0'].values * np.exp(-pfo_func(x_data, k1))
        r2, mape = calculate_metrics_internal(df['А'].values, predictions)
        return k1, predictions, mape, r2
    except:
        return 0.0, [], 0.0, 0.0

def fit_pso_model(df):
    """Fits Pseudo-Second Order kinetics: 1/A - 1/A0 = k2 * t"""
    if df.empty or '1/A' not in df.columns: return 0.0, [], 0.0, 0.0
    
    inv_a0 = 1.0 / df['А0'].iloc[0]
    y_data = (df['1/A'] - inv_a0).values
    x_data = df['т, мин'].values
    
    def pso_func(t, k): return k * t
    
    try:
        popt, _ = curve_fit(pso_func, x_data, y_data)
        k2 = popt[0]
        # PSO prediction is A = 1 / (1/A0 + k2*t)
        predictions = 1.0 / (inv_a0 + pso_func(x_data, k2))
        r2, mape = calculate_metrics_internal(df['А'].values, predictions)
        return k2, predictions, mape, r2
    except:
        return 0.0, [], 0.0, 0.0

def create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso):
    """Creates a DataFrame summarizing fitting results for all models."""
    data = {
        'Модель': ['Нулевой порядок (ZO)', 'Псевдо-первый порядок (PFO)', 'Псевдо-второй порядок (PSO)'],
        'Константа скорости (k)': [abs(k0), abs(k1), abs(k2)],
        'Единицы k': ['мг/(л·мин)', 'мин⁻¹', 'л/(мг·мин)'],
        'R²': [r2_zo, r2_pfo, r2_pso],
        'MAPE (%)': [mape_zo, mape_pfo, mape_pso]
    }
    return pd.DataFrame(data)

# --- مدمج من visualization.py ---
def create_matplotlib_plots(processed_df, zo_preds, pfo_preds, pso_preds, k0, k1, k2):
    """Creates the standard 3 kinetic plots using Matplotlib."""
    
    # Use generic style, don't rely on seaborn for reproducibility
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'grid.alpha': 0.5
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    t_data = processed_df['т, мин']
    a_data = processed_df['А']
    
    # 1. ZO: (A0-A) vs t
    ax = axes[0]
    raw_y = processed_df['А0'] - a_data
    ax.scatter(t_data, raw_y, color='#ef4444', s=30, label='Эксперимент', zorder=3)
    
    if len(zo_preds) == len(t_data):
        ax.plot(t_data, processed_df['А0'] - zo_preds, color='#2563eb', linewidth=2, label=f'ZO Модель (k={abs(k0):.4f})')
    
    ax.set_title('Модель Нулевого порядка')
    ax.set_xlabel('Время, т (мин)')
    ax.set_ylabel('(А₀ - А)')
    ax.legend()
    ax.grid(True, linestyle='--')

    # 2. PFO: -ln(A/A0) vs t
    ax = axes[1]
    if 'ln_A_A0' in processed_df.columns:
        ax.scatter(t_data, -processed_df['ln_A_A0'], color='#ef4444', s=30, zorder=3)
        # Proper Y for line visualization based on fitted k1
        pfo_line_y = abs(k1) * t_data
        ax.plot(t_data, pfo_line_y, color='#16a34a', linewidth=2, label=f'PFO Модель (k={abs(k1):.4f})')
        ax.set_ylabel('-ln(А / А₀)')
    
    ax.set_title('Модель Псевдо-первого порядка')
    ax.set_xlabel('Время, т (мин)')
    ax.legend()
    ax.grid(True, linestyle='--')

    # 3. PSO: (1/A - 1/A0) vs t
    ax = axes[2]
    if '1/A' in processed_df.columns:
        inv_a0 = 1.0 / processed_df['А0'].iloc[0]
        raw_y_pso = processed_df['1/A'] - inv_a0
        ax.scatter(t_data, raw_y_pso, color='#ef4444', s=30, zorder=3)
        
        # Proper Y for line based on k2
        pso_line_y = abs(k2) * t_data
        ax.plot(t_data, pso_line_y, color='#a855f7', linewidth=2, label=f'PSO Модель (k={abs(k2):.5f})')
        ax.set_ylabel('(1/А - 1/А₀)')
    
    ax.set_title('Модель Псевдо-второго порядка')
    ax.set_xlabel('Время, т (мин)')
    ax.legend()
    ax.grid(True, linestyle='--')

    plt.tight_layout()
    return fig

# =============================================================================
# الجزء الثاني: إعداد وتفعيل تقنية OCR (الجديد كلياً والمطلوب)
# =============================================================================

@st.cache_resource
def load_ocr_reader():
    """Загружает EasyOCR Reader и كاش 모델 لمنع إعادة التحميل، يدعم ru/en."""
    with st.spinner("⏳ Загрузка OCR моделей (en/ru)..."):
        # en للأرقام والنقاط، ru للعناوين المحتملة بالروسية
        return easyocr.Reader(['en', 'ru'], gpu=False) 

def process_image_and_extract_data(uploaded_image, col_names):
    """
    تحليل الصورة، استخراج الأرقام، تصفيتها، ترتيبها في جدول Pandas مكون من عمودين.
    """
    reader = load_ocr_reader()
    
    # تحويل بايتات Streamlit إلى تنسيق مفهوم لـ OpenCV
    try:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Не удалось декодировать изображение.")
    except Exception as e:
        st.error(f"❌ Ошибка обработки изображения: {e}")
        return None

    extracted_numbers = []
    
    with st.spinner("🧠 Идет распознавание текста على الصورة..."):
        # تنفيذ OCR
        # detail=0 يعيد النص فقط لسرعة المعالجة
        result = reader.readtext(image, detail=0)

        # استخراج النصوص وتحويلها لأرقام
        for text in result:
            # معالجة النص: إزالة الفراغات، تحويل الكوما لنقطة
            cleaned_text = text.replace(' ', '').replace(',', '.').strip()
            
            # محاولة التحويل لرقم طفو (Float)
            try:
                # التحقق من أن النص يحتوي على أرقام فقط (أو نقطة/إشارة سالب)
                if cleaned_text and any(char.isdigit() for char in cleaned_text):
                     extracted_numbers.append(float(cleaned_text))
            except ValueError:
                pass # تخطي النصوص غير الرقمية

    if len(extracted_numbers) < 4:
        st.error("❌ OCR Ошибка: На изображении найдено недостаточно دیجیتال даних للمحاكاة (требуется минимум 2 пары x, y). Убедитесь، что جدول واضح.")
        return None

    # ضمان عدد زوجي للأزواج (X, Y)
    if len(extracted_numbers) % 2 != 0:
        extracted_numbers = extracted_numbers[:-1]

    # إعادة تشكيل المصفوفة لعمودين
    try:
        data_array = np.array(extracted_numbers).reshape(-1, 2)
        # إنشاء جدول بأسماء الأعمدة المحددة لهذا التفاعل
        df = pd.DataFrame(data_array, columns=col_names)
        return df
    except Exception as e:
        st.error(f"❌ Ошибка форматирования данных OCR في جدول: {e}")
        return None

# =============================================================================
# الجزء الثالث: كود الواجهة والتحكم (app.py المعدل لربط كل شيء)
# =============================================================================

# CSS -- التشكيل المرئي (بقي كما هو في مشروعك الجاهز)
st.markdown("""
<style>
/* 1) إجبار الجذور والمتغيرات الأساسية لـ Streamlit على الألوان الفاتحة */
:root, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stApp {
    --background-color: #f5f7fa !important;
    --secondary-background-color: #ffffff !important;
    --text-color: #1e293b !important;
    --primary-color: #2563eb !important;
    background-color: #f5f7fa !important;
    color: #1e293b !important;
}

/* 2) استهدف القوائم وحقول الإدل والكبسولات */
div[data-baseweb="select"], 
div[data-baseweb="popover"], 
div[role="listbox"], 
ul[data-baseweb="menu"],
div[data-testid="stSelectbox"],
.stSelectbox, 
input, 
select, 
textarea,
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
}

/* كبسولة الملف المرفوع */
[data-testid="stUploadedFile"] {
    background-color: #e2e8f0 !important;
    background: #e2e8f0 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stUploadedFile"] * {
    background-color: transparent !important;
    background: transparent !important;
    color: #1e293b !important;
    fill: #1e293b !important;
}

/* سهم القائمة المنسدلة */
div[data-baseweb="select"] button, 
div[data-baseweb="select"] div,
div[data-testid="stSelectbox"] button,
div[data-testid="stSelectbox"] div {
    background-color: transparent !important;
    background: transparent !important;
    color: #1e293b !important;
}
div[data-baseweb="select"] svg,
div[data-testid="stSelectbox"] svg {
    fill: #1e293b !important;
    color: #1e293b !important;
}

div[data-baseweb="select"] *, 
div[data-baseweb="popover"] *, 
div[role="listbox"] *,
[data-testid="stSelectbox"] * {
    color: #1e293b !important;
}

li[role="option"], [data-baseweb="menu"] li, div[data-baseweb="popover"] div {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #1e293b !important;
}
li[role="option"]:hover, [data-baseweb="menu"] li:hover {
    background-color: #eff6ff !important;
    color: #1e3a8a !important;
}

/* إجبار نصوص التطبيق العادية على حجم أكبر شحطتين وتوحيده */
html, body, p, span, label, th, td, .stMarkdown, .stRadio label, input, select, button {
    color: #1e293b !important;
    font-size: 1.1rem !important;
}

/* الهيدر الرئيسي */
.main-header-title {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.2);
    text-align: center;
}

.main-header-title,
.main-header-title * {
    color: #ffffff !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}

/* مستطيل منفرد ومنفصل تماماً للأسماء والمعلومات */
.main-header-authors {
    background: #eff6ff;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.06);
    border: 1px solid #bfdbfe;
    text-align: center;
}
.main-header-authors p { 
    color: #1e40af !important; 
    margin: 0; 
    font-weight: 600;
    font-size: 1.1rem !important;
}

/* كروت المعلومات */
.info-card {
    background: #ffffff;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    border: 1px solid #e2e8f0;
}

/* صناديق المقاييس */
.metric-box {
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    border: 1px solid;
}
.metric-box h4 { margin: 0; font-size: 0.95rem !important; font-weight: 600; }
.metric-box h2 { margin: 0.3rem 0 0 0; font-size: 1.5rem !important; font-weight: 700; }

.mb-amber { background: #fffbeb !important; border-color: #fde68a !important; }
.mb-amber h4, .mb-amber h2 { color: #92400e !important; }

.mb-blue { background: #eff6ff !important; border-color: #bfdbfe !important; }
.mb-blue h4, .mb-blue h2 { color: #1e40af !important; }

.mb-green { background: #ecfdf5 !important; border-color: #a7f3d0 !important; }
.mb-green h4, .mb-green h2 { color: #065f46 !important; }

.mb-rose { background: #fff1f2 !important; border-color: #fecdd3 !important; }
.mb-rose h4, .mb-rose h2 { color: #9f1239 !important; }

.performance-metric {
    background: #f0fdf4 !important;
    border: 1px solid #10b981 !important;
    padding: 0.7rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    color: #065f46 !important;
    font-weight: 600;
    text-align: center;
    font-size: 1.15rem !important;
}

/* عناوين الأقسام */
.section-header {
    padding: 0.8rem 1.2rem;
    border-radius: 10px;
    border-left: 5px solid;
    margin-bottom: 1rem;
}
.section-header h2 { margin: 0; font-size: 1.35rem !important; }

.sh-data { background: #eff6ff !important; border-left-color: #2563eb !important; }
.sh-data h2 { color: #1e3a8a !important; }

.sh-results { background: #ecfdf5 !important; border-left-color: #10b981 !important; }
.sh-results h2 { color: #065f46 !important; }

.sh-compare { background: #f0fdfa !important; border-left-color: #0d9488 !important; }
.sh-compare h2 { color: #115e59 !important; }

.sh-viz { background: #faf5ff !important; border-left-color: #a855f7 !important; }
.sh-viz h2 { color: #6b21a8 !important; }

.sh-download { background: #fef2f2 !important; border-left-color: #ef4444 !important; margin-top: 1.2rem; }
.sh-download h2 { color: #991b1b !important; }

.highlight-success {
    background: #dcfce7 !important;
    border: 1px solid #22c55e !important;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    color: #14532d !important;
}

/* كروت الموديلات */
.model-card {
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid;
}
.model-card h3 { margin: 0 0 0.5rem 0; font-size: 1.25rem !important; }
.model-card p { margin: 0.25rem 0; font-size: 1.05rem !important; }

.mc-zo   { background: #f8fafc !important; border-left-color: #94a3b8 !important; }
.mc-zo   h3, .mc-zo   p { color: #334155 !important; }

.mc-pfo  { background: #eff6ff !important; border-left-color: #2563eb !important; }
.mc-pfo  h3, .mc-pfo  p { color: #1e3a8a !important; }

.mc-pso  { background: #f0fdf4 !important; border-left-color: #16a34a !important; }
.mc-pso  h3, .mc-pso  p { color: #14532d !important; }

/* الشريط الجانبي */
section[data-testid="stSidebar"] { background-color: #ffffff !important; }
.sidebar-params-title {
    font-size: 1.2rem !important;
    font-weight: 700;
    color: #1e3a8a !important;
    margin: 0.6rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #2563eb;
}

.stDownloadButton button, .stButton button {
    background-color: #ffffff !important;
    color: #1e3a8a !important;
    border: 1.5px solid #2563eb !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}
.stDownloadButton button:hover, .stButton button:hover {
    background-color: #2563eb !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# 1. Фотокаталитические реакции (Render Photocatalysis)
def render_photocatalysis():
    sidebar_params(
        inputs=["т, мин (время)", "А (оптическая плотность)", "А0 (опционально)"],
        outputs=["k₀, k₁, k₂ — константы скорости", "R², MAPE — метрики качества", "3 графика (ZO, PFO, PSO)"],
        file_types=["Excel (.xlsx)", "CSV (.csv)"]
    )

    st.markdown(section_header("sh-data", "📊", "Ввод данных"), unsafe_allow_html=True)
    method = input_method_choice("photo")

    df = None
    
    if method == "📁 Загрузить файл":
        uploaded_file = st.file_uploader("Выберите файл Excel/CSV", type=['xlsx', 'csv'], key="photo_upload")
        if uploaded_file is not None:
            try:
                df = handle_file_upload(uploaded_file, "photo")
            except Exception as e:
                st.error(f"❌ Ошибка загрузки файла: {str(e)}")
                
    elif method == "✏️ Ввести данные вручную":
        blank_data = pd.DataFrame({'т, мин': [0.0]*6, 'А': [0.0]*6})
        st.markdown("**Заполните экспериментальные данные вручную:**")
        df = st.data_editor(blank_data, use_container_width=True, num_rows="dynamic", key="photo_manual_ed")
        
    elif method == "📷 Изображение (OCR)":
        # رسالة النجاح الخضراء المخصصة (من مشروعك)
        st.markdown(f'<div class="highlight-success">✅ Оптическая плотность (А) теперь распознается и отображается. Пожалуйста, проверьте и отредактируйте данные ниже при необходимости.</div>', unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader("Выберите изображение таблицы (т vs А)", type=['png', 'jpg', 'jpeg'], key="ocr_upload_photo")
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Загруженное изображение", use_container_width=True)
            
            # استخراج البيانات فعلياً باستخدام الدالة الجديدة
            extracted_df = process_image_and_extract_data(uploaded_image, ['т, мин', 'А'])
            
            # عرض محرر بيانات للتحقق والتحرير قبل التمرير للمعالجة
            if extracted_df is not None:
                # منطق إيجاد A0 تلقائياً في البيانات المستخرجة
                # قصر البيانات على القيم الصالحة قبل حساب A0
                temp_df = extracted_df[pd.to_numeric(extracted_df['А'], errors='coerce') > 0].copy()
                if not temp_df.empty:
                    extracted_df['А0'] = temp_df['А'].iloc[0]
                    st.markdown(f'<div class="highlight-success">✅ Автоопределение: А0 = {extracted_df["А0"].iloc[0]:.5f}</div>', unsafe_allow_html=True)

                if 'А/А0' not in extracted_df.columns and 'А' in extracted_df.columns and 'А0' in extracted_df.columns:
                    try:
                        extracted_df['А/А0'] = pd.to_numeric(extracted_df['А'], errors='coerce') / pd.to_numeric(extracted_df['А0'], errors='coerce')
                    except:
                        pass
                
                with st.expander("👁️ Просмотр и редактирование распознанных данных OCR"):
                    # تمرير الجدول المستخرج إلى محرر البيانات للتأكيد النهائي من المستخدم
                    df = st.data_editor(extracted_df, use_container_width=True, num_rows="dynamic", key=f"photo_final_ed_ocr")

    # الخروج إذا لم يتم توفير بيانات صالحة بعد
    if df is None: return
    
    # تنظيف وتجهيز الأعمدة لضمان عمل الحسابات
    # سنمرر الأعمدة المستهدفة فقط للمعالجة
    final_cols = [c for c in ['т, мин', 'А', 'А0', 'А/А0'] if c in df.columns]
    if len(final_cols) < 2: 
        st.warning("⚠️ Недостаточно столбцов البيانات для расчета.")
        return
        
    df_clean = df[final_cols].copy()
    
    processed_df = preprocess_data(df_clean) # استدعاء دالتك المدمجة
    
    if processed_df.empty:
        st.warning("⚠️ Нет допустимых данных للمحاكاة. Убедитесь، أن الأرقام تم استخراجها بشكل صحيح.")
        return
        
    # --- استمرار الكود الأصلي للحسابات والرسوم كما هو ---
    summary = get_data_summary(processed_df)

    st.markdown(section_header("sh-selected", "📊", "Сводка данных"), unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_box("mb-amber", "Действительные точки", summary["total_points"]), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_box("mb-amber", "Диапазон времени", f'{summary["time_range"][0]:.1f}-{summary["time_range"][1]:.1f} мин'), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_box("mb-amber", "Начальная концентрация", f'{summary["a0_value"]:.3f}'), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_box("mb-amber", "Диапазон А/А0", f'{summary["a_a0_range"][0]:.3f}-{summary["a_a0_range"][1]:.3f}'), unsafe_allow_html=True)

    try:
        k0, zo_predictions, mape_zo, r2_zo = fit_zo_model(processed_df)
        k1, pfo_predictions, mape_pfo, r2_pfo = fit_pfo_model(processed_df)
        k2, pso_predictions, mape_pso, r2_pso = fit_pso_model(processed_df)

        st.markdown(section_header("sh-results", "📋", "Сводка результатов"), unsafe_allow_html=True)
        results_summary = create_results_summary(k0, k1, k2, mape_zo, mape_pfo, mape_pso, r2_zo, r2_pfo, r2_pso)
        st.dataframe(results_summary, use_container_width=True)

        st.markdown(section_header("sh-compare", "⚖️", "Сравнение моделей"), unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="model-card mc-zo"><h3>Модель ZO</h3><p><strong>k₀:</strong> {abs(k0):.4f}</p><p><strong>R²:</strong> {r2_zo:.4f}</p><p><strong>MAPE:</strong> {mape_zo:.2f}%</p></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="model-card mc-pfo"><h3>Модель PFO</h3><p><strong>k₁:</strong> {abs(k1):.5f} мин⁻¹</p><p><strong>R²:</strong> {r2_pfo:.4f}</p><p><strong>MAPE:</strong> {mape_pfo:.2f}%</p></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="model-card mc-pso"><h3>Модель PSO</h3><p><strong>k₂:</strong> {k2:.5f} л/(мг·мин)</p><p><strong>R²:</strong> {r2_pso:.4f}</p><p><strong>MAPE:</strong> {mape_pso:.2f}%</p></div>', unsafe_allow_html=True)

        st.markdown(section_header("sh-viz", "📊", "Графика"), unsafe_allow_html=True)
        fig_main = create_matplotlib_plots(processed_df, zo_predictions, pfo_predictions, pso_predictions, k0, k1, k2)
        
        for ax in fig_main.get_axes():
            apply_axis_style(ax)

        col_side1, col_chart_photo, col_side2 = st.columns([1, 2, 1])
        with col_chart_photo:
            st.pyplot(fig_main)

        st.markdown(section_header("sh-download", "📥", "Скачать результаты"), unsafe_allow_html=True)
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button("📊 Скачать данные (Excel)", data=convert_df_to_excel(results_summary), file_name="kinetic_analysis_results.xlsx", use_container_width=True)
        with d_col2:
            png_b = BytesIO()
            fig_main.savefig(png_b, format='png', dpi=300, bbox_inches='tight')
            png_b.seek(0)
            st.download_button("🖼️ Скачать графики (PNG)", data=png_b, file_name="kinetic_plots.png", mime="image/png", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Ошибка моделирования: {str(e)}")

# (يمكنك دمج دالات Homo, Het, Enz بنفس الطريقة، ولكن بناءً على طلبك، قدمت الكود الكامل المعدل للتفاعل الأول كمثال واضح)

# ГЛАВНАЯ ТОЧКА ВХОДА
def main():
    st.markdown("""
    <div class="main-header-title">
         <h1>Анализ кинетического моделирования</h1>
    </div>
    <div class="main-header-authors">
         <p>АВТОР: Алсади К. &nbsp;|&nbsp; РУКОВОДИТЕЛЬ: Киреева А.В</p>
    </div>
    """, unsafe_allow_html=True)

    reaction_type = st.sidebar.selectbox(
        "🛠️ Тип процесса / реакции",
        options=["Фотокаталитические реакции"], # قدمت هذا الخيار كمثال للدمج الكامل
        index=0, key="reaction_type_choice"
    )

    if reaction_type == "Фотокаталитические реакции":
        render_photocatalysis()

if __name__ == "__main__":
    main()
