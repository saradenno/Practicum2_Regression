import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Model laden
model = joblib.load('model.joblib')

# Pagina configuratie
st.set_page_config(
    page_title='Student Score Voorspeller',
    page_icon='🎓',
    layout='centered'
)

st.title('🎓 Student Score Voorspeller')
st.subheader('Open University Learning Analytics Dataset')
st.markdown('Vul de gegevens van een student in om de gemiddelde toetsscore te voorspellen.')
st.divider()

# ── Input formulier ──
st.header('📋 Studentgegevens')

col1, col2 = st.columns(2)

with col1:
    code_module = st.selectbox('Module', ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG'])
    code_presentation = st.selectbox('Semester', ['2013B', '2013J', '2014B', '2014J'])
    gender = st.selectbox('Geslacht', ['M', 'F'])
    region = st.selectbox('Regio', [
        'East Anglian Region', 'East Midlands Region', 'Ireland',
        'London Region', 'North Region', 'North Western Region',
        'Scotland', 'South East Region', 'South Region',
        'South West Region', 'Wales', 'West Midlands Region',
        'Yorkshire Region'
    ])
    highest_education = st.selectbox('Hoogste opleiding', [
        'A Level or Equivalent', 'HE Qualification',
        'Lower Than A Level', 'No Formal quals',
        'Post Graduate Qualification'
    ])
    imd_band = st.selectbox('Sociaaleconomische status (IMD)', [
        '0-10%', '10-20', '20-30%', '30-40%', '40-50%',
        '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'
    ])

with col2:
    age_band = st.selectbox('Leeftijdscategorie', ['0-35', '35-55', '55<='])
    disability = st.selectbox('Beperking', ['N', 'Y'])
    num_of_prev_attempts = st.slider('Aantal eerdere pogingen', 0, 6, 0)
    studied_credits = st.slider('Studiepunten', 30, 150, 60)
    num_assessments = st.slider('Aantal opdrachten ingeleverd', 1, 28, 7)
    min_score = st.slider('Laagste score', 0, 100, 50)
    max_score = st.slider('Hoogste score', 0, 100, 90)
    avg_date_submitted = st.slider('Gemiddeld inleverdatum (dagen)', 0, 160, 100)
    total_banked = st.slider('Gebankeerde opdrachten', 0, 12, 0)

st.divider()

# ── Voorspelling ──
if st.button('🔮 Voorspel Score', use_container_width=True):
    input_data = pd.DataFrame([{
        'code_module': code_module,
        'code_presentation': code_presentation,
        'gender': gender,
        'region': region,
        'highest_education': highest_education,
        'imd_band': imd_band,
        'age_band': age_band,
        'num_of_prev_attempts': num_of_prev_attempts,
        'studied_credits': studied_credits,
        'disability': disability,
        'num_assessments': num_assessments,
        'min_score': min_score,
        'max_score': max_score,
        'avg_date_submitted': avg_date_submitted,
        'total_banked': total_banked
    }])

    prediction = model.predict(input_data)[0]
    prediction = np.clip(prediction, 0, 100)

    st.success(f'### Voorspelde gemiddelde score: **{prediction:.1f} / 100**')

    # Kleur indicator
    if prediction >= 75:
        st.success('✅ Goede prestatie verwacht!')
    elif prediction >= 55:
        st.warning('⚠️ Voldoende prestatie verwacht.')
    else:
        st.error('❌ Onvoldoende prestatie verwacht.')