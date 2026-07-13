import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.predict import load_model, predict_single, predict_batch
from src.config import MODEL_PATH


def get_base64_image(path):
    """Read a local image file and return a base64 string (or None if missing)."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None


IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
BOT_IMG_B64 = get_base64_image(os.path.join(IMAGES_DIR, 'bot.png'))
CLICK_IMG_B64 = get_base64_image(os.path.join(IMAGES_DIR, 'click_pic.png'))

st.set_page_config(page_title='Adbot Click Predictor', layout='wide')

# ──────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Google-inspired bright theme
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', Roboto, sans-serif;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #F0F2F5 !important;
}
[data-testid="stHeader"] {
    background: rgba(255,255,255,0) !important;
}

/* Hide default Streamlit chrome that clashes with the custom look */
#MainMenu, footer {visibility: hidden;}

/* Force readable dark text everywhere by default (guards against dark-mode leakage) */
p, span, label, div, .gb-form-title, .gb-title, .gb-subtitle {
    color: #202124;
}
.gb-subtitle, .gb-footer { color: #5f6368 !important; }

/* ---------- HEADER BANNER ---------- */
/* ---------- HEADER BANNER ---------- */
.gb-header {
    width: 100%;
    position: relative;
    overflow: hidden;
    border-radius: 20px;
    padding: 45px 50px;
    margin-bottom: 18px;
    border: 4px solid transparent;

    background:
        linear-gradient(#FFFFFF, #FFFFFF) padding-box,
        linear-gradient(
            270deg,
            #4285F4,
            #EA4335,
            #FBBC05,
            #34A853,
            #4285F4
        ) border-box;

    background-size: 400% 400%;

    animation: gradientMove 8s ease infinite;

    box-shadow:
        0 14px 36px rgba(0,0,0,0.16),
        0 6px 18px rgba(0,0,0,0.10);
}

/* ---------- animation ---------- */
@keyframes gradientMove {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

            
            
.gb-header-inner {
    text-align: center;
}

/* ---------- HEADER IMAGES ---------- */

/* Left image (Click image) */
.gb-header-img-left {
    position: absolute;
    top: 24%;
    left: 2.8%;
    height: clamp(250px, 16vw, 180px);
    width: auto;
    z-index: 2;
    transition: all 0.3s ease;
}

/* Right image (Bot) */
.gb-header-img-right {
    position: absolute;
    bottom: -48%;
    right: 2.5%;
    height: clamp(150px, 13vw, 150px);
    width: auto;
    z-index: 5;
    transition: all 0.3s ease;
}

/* ---------- Mobile ---------- */
@media (max-width: 768px) {

    .gb-header-img-left {
        top: 26%;
        left: 2%;
        height: clamp(65px, 20vw, 120px);
    }

    .gb-header-img-right {
        bottom: -7%;
        right: 2%;
        height: clamp(70px, 18vw, 115px);
    }
}

/* ---------- Very Small Phones ---------- */
@media (max-width: 480px) {

    .gb-header-img-left {
        top: 28%;
        left: 2%;
        height: clamp(55px, 22vw, 95px);
    }

    .gb-header-img-right {
        bottom: -5%;
        right: 2%;
        height: clamp(60px, 20vw, 90px);
    }
}


.gb-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 40px !important;
    font-weight: 700 !important;
    letter-spacing: -3px !important;
    line-height: 1 !important;

    color: #202124 !important;

    margin: 0 !important;
    padding: 0 !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 100% !important;
}
            
.system-blue {
    color: #4285F4 !important;
}


.gb-subtitle {
    font-size: 19px !important;
    font-weight: 500 !important;

    margin-top: 20px !important;

    color: #5f6368 !important;
}

/* ---------- TABS ---------- */
div[data-baseweb="tab-list"] {
    gap: 28px;
    border-bottom: 1px solid #E8EAED;
}
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 500;
    color: #5f6368;
    padding-bottom: 10px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1A73E8;
    font-weight: 700;
    border-bottom: 3px solid #1A73E8 !important;
}

/* ---------- FORM SECTION HEADING ---------- */
.gb-form-title {
    font-size: 22px;
    font-weight: 700;
    color: #202124;
    margin: 26px 0 18px 0;
}

/* ---------- FORM CARD ---------- */
.gb-form-card {
    position: relative;
    background: #FFFFFF;
    border-radius: 20px;
    padding: 26px 34px 40px 34px;
    border: 1px solid #F1F3F4;
    box-shadow:  0 12px 32px rgba(0,0,0,0.18),
    0 4px 12px rgba(0,0,0,0.08);
    overflow: hidden;
}
            
.gb-form-card:hover {
    transform: translateY(-4px);
    box-shadow:
        0 24px 48px rgba(0,0,0,0.16),
        0 12px 24px rgba(0,0,0,0.10);
}



.gb-corner-dots {
    position: relative;
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 20px;
    margin-right: 20px;
    z-index: 1;
}

.gb-corner-dots span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}
/* Streamlit input styling — forced explicitly so dark-mode vars can't leak through */
div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #202124 !important;
    -webkit-text-fill-color: #202124 !important;
    border: 1px solid #DADCE0 !important;
    border-radius: 10px !important;
    padding: 8px 10px !important;
}
div[data-testid="stNumberInput"] button {
    border-radius: 8px !important;
    border: 1px solid #DADCE0 !important;
    background: #F8F9FA !important;
}
div[data-testid="stNumberInput"] button svg {
    fill: #5f6368 !important;
}

/* Selectbox (Ad Type / Currency) */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #DADCE0 !important;
}
div[data-baseweb="select"] * {
    color: #202124 !important;
    -webkit-text-fill-color: #202124 !important;
    fill: #5f6368 !important;
}
div[data-baseweb="popover"] div[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}
ul[role="listbox"] {
    background-color: #FFFFFF !important;
}
ul[role="listbox"] li {
    background-color: #FFFFFF !important;
    color: #202124 !important;
}
ul[role="listbox"] li:hover {
    background-color: #E8F0FE !important;
}

/* Date input */
div[data-testid="stDateInput"] input {
    background-color: #FFFFFF !important;
    color: #202124 !important;
    -webkit-text-fill-color: #202124 !important;
    border-radius: 10px !important;
    border: 1px solid #DADCE0 !important;
}
div[data-baseweb="calendar"], div[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
    color: #202124 !important;
}

/* Sliders — Google blue */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #1A73E8 !important;
    border: 3px solid #FFFFFF !important;
    box-shadow: 0 0 0 1px #1A73E8 !important;
}
/* Unfilled portion (right of the thumb) */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:last-child {
    background: #DADCE0 !important;
}
div[data-testid="stTickBar"] {display: none;}

/* Field labels */
label {
    font-weight: 500 !important;
    color: #3c4043 !important;
}

/* ---------- ACTION BUTTON AREA ---------- */
.gb-button-zone {
    padding: 20px 0 6px 0;
}
div.stButton > button {
    background: #4285F4 !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 12px 34px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    box-shadow: 0 6px 16px rgba(26,115,232,0.35) !important;
    transition: all 0.15s ease !important;
}

div.stButton > button p {
    color: white !important;
    font-weight: 700 !important;
}

div.stButton > button:hover {
    background: #3367D6 !important;
    transform: scale(1.04);
    box-shadow: 0 8px 22px rgba(26,115,232,0.45) !important;
}

div.stButton > button:hover p {
    color: white !important;
}

/* ---------- FILE UPLOADER (Batch tab) ---------- */
div[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
    background-color: #F8F9FA !important;
    border: 1px solid #DADCE0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] div,
div[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #5f6368 !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] svg {
    display: none;
}
div[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child {
    display: none;
}
div[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #5f6368 !important;
    font-size: 13px !important;
}
button[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF !important;
    color: #202124 !important;
    border: 1px solid #DADCE0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ---------- METRIC RESULT ---------- */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 18px;
    border: 1px solid #E8EAED;
    box-shadow:
        0 6px 18px rgba(0,0,0,0.08);
    padding: 18px 22px;
}
div[data-testid="stMetricValue"] {
    color: #1A73E8 !important;
}

/* ---------- FOOTER ---------- */
.gb-footer {
    text-align: center;
    color: #9AA0A6;
    font-size: 13px;
    margin-top: 40px;
    padding-bottom: 8px;
}
.gb-bottom-bar {
    height: 4px;
    border-radius: 4px;
    margin-top: 4px;
    background: linear-gradient(90deg, #4285F4 0%, #EA4335 33%, #FBBC05 66%, #34A853 100%);
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# HEADER BANNER
# ──────────────────────────────────────────────────────────────────────────
left_img_tag = (
    f'<img src="data:image/png;base64,{CLICK_IMG_B64}" class="gb-header-img-left" />'
    if CLICK_IMG_B64 else ''
)
right_img_tag = (
    f'<img src="data:image/png;base64,{BOT_IMG_B64}" class="gb-header-img-right" />'
    if BOT_IMG_B64 else ''
)

st.markdown(f"""
<div class="gb-header">
    {left_img_tag}
    {right_img_tag}
    <div class="gb-header-inner">
        <p class="gb-title">Ads Click Prediction <span class="system-blue">System</span>
            </p>
        <p class="gb-subtitle">Forecast Google Ad clicks for South African small businesses — before they spend a single Rand.</p>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error('Model not found. Run python src/train.py first.')
        st.stop()
    return load_model()

model = get_model()

tab1, tab2 = st.tabs(['Single Ad Prediction', 'Batch Prediction'])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE AD PREDICTION
# ──────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="gb-form-title">Predict Clicks for a Single Ad</p>', unsafe_allow_html=True)



    col1, col2 = st.columns(2)
    with col1:
        impressions        = st.number_input('Impressions', 0.0, 100000.0, 500.0)
        cost               = st.number_input('Ad Cost (ZAR)', 0.0, 1000000.0, 200.0)
        conversions        = st.number_input('Conversions', 0.0, 5000.0, 5.0)
        impression_share   = st.slider('Impression Share', 0.0, 500.0, 50.0)
        conversions_calls  = st.number_input('Call Conversions', 0.0, 100.0, 2.0)
    with col2:
        ad_type            = st.selectbox('Ad Type', ['EXPANDED_TEXT_AD', 'RESPONSIVE_SEARCH_AD', 'EXPANDED_DYNAMIC_SEARCH_AD'])
        currency           = st.selectbox('Currency', ['ZAR', 'USD'])
        headline1_len      = st.slider('Headline 1 Length (words)', 1, 15, 5)
        headline2_len      = st.slider('Headline 2 Length (words)', 1, 15, 4)
        ad_description_len = st.slider('Description Length (words)', 1, 30, 10)
        ad_date            = st.date_input(
            'Ad Date',
            value=pd.Timestamp('2023-06-15'),
            min_value=pd.Timestamp('2020-01-01'),
            max_value=pd.Timestamp('2029-02-13')
        )

        day_of_week = ad_date.weekday()   # Monday=0 ... Sunday=6
        month       = ad_date.month
        year        = ad_date.year

    st.markdown('<div class="gb-button-zone">', unsafe_allow_html=True)
    predict_clicked = st.button('Predict Clicks', type='primary')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="gb-corner-dots">
        <span style="background:#4285F4"></span>
        <span style="background:#EA4335"></span>
        <span style="background:#FBBC05"></span>
        <span style="background:#34A853"></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close gb-form-card

    if predict_clicked:
        input_data = {
            'impressions': impressions, 'cost': cost,
            'conversions': conversions, 'impression_share': impression_share,
            'conversions_calls': conversions_calls, 'ad_type': ad_type,
            'currency': currency, 'ID': 'CLIENT_001',
            'headline1_len': headline1_len, 'headline2_len': headline2_len,
            'ad_description_len': ad_description_len, 'duration': 0.0,
            'day_of_week': day_of_week, 'month': month, 'year': year
        }
        result = predict_single(model, input_data)
        st.divider()
        st.metric('Predicted Clicks', result['predicted_clicks'])

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="gb-form-title">Batch Prediction from CSV</p>', unsafe_allow_html=True)


    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button('Run Predictions', type='primary'):
            results = predict_batch(model, df)
            st.dataframe(results[['Predicted_Clicks']].head(20))
            c1, c2, c3 = st.columns(3)
            c1.metric('Total Ads', len(results))
            c2.metric('Mean Predicted Clicks', f"{results['Predicted_Clicks'].mean():.0f}")
            c3.metric('Max Predicted Clicks', results['Predicted_Clicks'].max())
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(results['Predicted_Clicks'], bins=30, color='#1A73E8', edgecolor='white', lw=0.5)
            ax.set_title('Predicted Clicks Distribution')
            ax.set_xlabel('Predicted Clicks')
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.download_button('Download Predictions', results.to_csv(index=False), 'predictions.csv')

    st.markdown("""
    <div class="gb-corner-dots">
        <span style="background:#4285F4"></span>
        <span style="background:#EA4335"></span>
        <span style="background:#FBBC05"></span>
        <span style="background:#34A853"></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close gb-form-card

# ──────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<div class="gb-bottom-bar"></div>', unsafe_allow_html=True)
st.markdown('<p class="gb-footer">Adbot Ad Click Predictor</p>', unsafe_allow_html=True)