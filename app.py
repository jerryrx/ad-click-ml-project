import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.predict import load_model, predict_single, predict_batch
from src.config import MODEL_PATH

st.set_page_config(page_title='Adbot Click Predictor', layout='wide')
st.title('Ad Click Predictor — Adbot South Africa')
st.markdown('Predict how many clicks a digital advertisement will receive based on its characteristics.')

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error('Model not found. Run python src/train.py first.')
        st.stop()
    return load_model()

model = get_model()
tab1, tab2 = st.tabs(['Single Ad Prediction', 'Batch Prediction'])

with tab1:
    st.header('Predict Clicks for a Single Ad')
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
            max_value=pd.Timestamp('2024-02-13')
        )

        day_of_week = ad_date.weekday()   # Monday=0 ... Sunday=6
        month       = ad_date.month
        year        = ad_date.year
        
        
        
       # day_of_week        = st.selectbox('Day of Week', [0,1,2,3,4,5,6],
                                          # format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
       # month              = st.selectbox('Month', list(range(1, 13)))
        #year               = st.selectbox('Year', [2020, 2021, 2022, 2023, 2024])

    if st.button('Predict Clicks', type='primary'):
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

with tab2:
    st.header('Batch Prediction from CSV')
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
            ax.hist(results['Predicted_Clicks'], bins=30, color='steelblue', edgecolor='black', lw=0.5)
            ax.set_title('Predicted Clicks Distribution')
            ax.set_xlabel('Predicted Clicks')
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.download_button('Download Predictions', results.to_csv(index=False), 'predictions.csv')
