import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Database setup 
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            code_module TEXT,
            code_presentation TEXT,
            gender TEXT,
            region TEXT,
            highest_education TEXT,
            imd_band TEXT,
            age_band TEXT,
            disability TEXT,
            num_of_prev_attempts INTEGER,
            studied_credits INTEGER,
            num_assessments INTEGER,
            min_score REAL,
            max_score REAL,
            avg_date_submitted REAL,
            total_banked INTEGER,
            predicted_score REAL,
            data_source TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(input_data, prediction, source='manual'):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            timestamp, code_module, code_presentation, gender, region,
            highest_education, imd_band, age_band, disability,
            num_of_prev_attempts, studied_credits, num_assessments,
            min_score, max_score, avg_date_submitted, total_banked,
            predicted_score, data_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        input_data['code_module'], input_data['code_presentation'],
        input_data['gender'], input_data['region'],
        input_data['highest_education'], input_data['imd_band'],
        input_data['age_band'], input_data['disability'],
        input_data['num_of_prev_attempts'], input_data['studied_credits'],
        input_data['num_assessments'], input_data['min_score'],
        input_data['max_score'], input_data['avg_date_submitted'],
        input_data['total_banked'],
        round(float(prediction), 2),
        source
    ))
    conn.commit()
    conn.close()

def load_predictions():
    conn = sqlite3.connect('predictions.db')
    try:
        df = pd.read_sql_query(
            'SELECT * FROM predictions ORDER BY timestamp ASC', conn
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

# Model laden
model = joblib.load('model.joblib')
init_db()

# Pagina configuratie
st.set_page_config(
    page_title='Student Score Voorspeller',
    page_icon='🎓',
    layout='centered'
)

st.title('🎓 Student Score Voorspeller')
st.caption('Open University Learning Analytics Dataset')
st.divider()

# ── Twee tabbladen ──
tab1, tab2 = st.tabs(['📋 Voorspelling', '🔬 Synthetische Data'])


# TAB 1 — Handmatige voorspelling

with tab1:
    st.header('📋 Studentgegevens invoeren')
    st.markdown('Vul de gegevens van een student in om de gemiddelde toetsscore te voorspellen.')

    col1, col2 = st.columns(2)

    with col1:
        code_module = st.selectbox('Module', ['AAA','BBB','CCC','DDD','EEE','FFF','GGG'])
        code_presentation = st.selectbox('Semester', ['2013B','2013J','2014B','2014J'])
        gender = st.selectbox('Geslacht', ['M','F'])
        region = st.selectbox('Regio', [
            'East Anglian Region','East Midlands Region','Ireland',
            'London Region','North Region','North Western Region',
            'Scotland','South East Region','South Region',
            'South West Region','Wales','West Midlands Region','Yorkshire Region'
        ])
        highest_education = st.selectbox('Hoogste opleiding', [
            'A Level or Equivalent','HE Qualification',
            'Lower Than A Level','No Formal quals','Post Graduate Qualification'
        ])
        imd_band = st.selectbox('Sociaaleconomische status (IMD)', [
            '0-10%','10-20','20-30%','30-40%','40-50%',
            '50-60%','60-70%','70-80%','80-90%','90-100%'
        ])

    with col2:
        age_band = st.selectbox('Leeftijdscategorie', ['0-35','35-55','55<='])
        disability = st.selectbox('Beperking', ['N','Y'])
        num_of_prev_attempts = st.slider('Aantal eerdere pogingen', 0, 6, 0)
        studied_credits = st.slider('Studiepunten', 30, 150, 60)
        num_assessments = st.slider('Aantal opdrachten ingeleverd', 1, 28, 7)
        min_score = st.slider('Laagste score', 0, 100, 50)
        max_score = st.slider('Hoogste score', 0, 100, 90)
        avg_date_submitted = st.slider('Gemiddeld inleverdatum (dagen)', 0, 160, 100)
        total_banked = st.slider('Gebankeerde opdrachten', 0, 12, 0)

    st.divider()

    if st.button('🔮 Voorspel Score', use_container_width=True):
        input_data = {
            'code_module': code_module,
            'code_presentation': code_presentation,
            'gender': gender,
            'region': region,
            'highest_education': highest_education,
            'imd_band': imd_band,
            'age_band': age_band,
            'disability': disability,
            'num_of_prev_attempts': num_of_prev_attempts,
            'studied_credits': studied_credits,
            'num_assessments': num_assessments,
            'min_score': min_score,
            'max_score': max_score,
            'avg_date_submitted': avg_date_submitted,
            'total_banked': total_banked
        }

        input_df = pd.DataFrame([input_data])
        prediction = np.clip(model.predict(input_df)[0], 0, 100)

        save_prediction(input_data, prediction, source='manual')

        st.success(f'### Voorspelde gemiddelde score: **{prediction:.1f} / 100**')
        st.caption(' Voorspelling opgeslagen in database.')

        if prediction >= 75:
            st.success(' Goede prestatie verwacht!')
        elif prediction >= 55:
            st.warning('⚠️ Voldoende prestatie verwacht.')
        else:
            st.error('x Onvoldoende prestatie verwacht.')


# TAB 2 — Synthetische data

with tab2:
    st.header('🔬 Synthetische Data Simulatie')
    st.markdown('''
    Simuleer app-gebruik over een zelfgekozen periode in 2026.
    De 500 synthetische studenten worden gelijkmatig verspreid 
    over de gekozen periode.
    ''')

    try:
        synthetic_df = pd.read_csv('synthetic_data.csv')
        st.success(f' Synthetische dataset geladen: {len(synthetic_df)} rijen')

        st.subheader('Preview synthetische data')
        st.dataframe(synthetic_df.head(10), use_container_width=True)

        st.divider()

        # Datumkiezer — alleen 2026
        st.subheader('📅 Kies simulatieperiode')
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                'Startdatum',
                value=date(2026, 1, 1),
                min_value=date(2026, 1, 1),
                max_value=date(2026, 12, 30)
            )
        with col2:
            end_date = st.date_input(
                'Einddatum',
                value=date(2026, 12, 31),
                min_value=date(2026, 1, 2),
                max_value=date(2026, 12, 31)
            )

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        num_days = (end_ts - start_ts).days

        if num_days <= 0:
            st.error(' Einddatum moet na de startdatum liggen.')
        else:
            interval_minutes = (num_days * 24 * 60) / len(synthetic_df)
            st.caption(
                f' **{start_date.strftime("%d %b %Y")}** → '
                f'**{end_date.strftime("%d %b %Y")}** '
                f'({num_days} dagen) — '
                f'interval: ~{interval_minutes:.0f} minuten per voorspelling'
            )

            if st.button('🚀 Simuleer 500 synthetische studenten',
                         use_container_width=True):
                predictions = np.clip(model.predict(synthetic_df), 0, 100)

                # Verwijder eerst oude synthetische voorspellingen
                conn = sqlite3.connect('predictions.db')
                c = conn.cursor()
                c.execute("DELETE FROM predictions WHERE data_source = 'synthetic'")
                conn.commit()

                # Sla op met gespreide timestamps over gekozen periode
                for i, (_, row) in enumerate(synthetic_df.iterrows()):
                    timestamp = start_ts + pd.Timedelta(minutes=interval_minutes * i)
                    c.execute('''
                        INSERT INTO predictions (
                            timestamp, code_module, code_presentation, gender, region,
                            highest_education, imd_band, age_band, disability,
                            num_of_prev_attempts, studied_credits, num_assessments,
                            min_score, max_score, avg_date_submitted, total_banked,
                            predicted_score, data_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        row['code_module'], row['code_presentation'],
                        row['gender'], row['region'], row['highest_education'],
                        row['imd_band'], row['age_band'], row['disability'],
                        int(row['num_of_prev_attempts']), int(row['studied_credits']),
                        int(row['num_assessments']), float(row['min_score']),
                        float(row['max_score']), float(row['avg_date_submitted']),
                        int(row['total_banked']),
                        round(float(predictions[i]), 2),
                        'synthetic'
                    ))
                conn.commit()
                conn.close()

                st.success(
                    f' 500 voorspellingen gesimuleerd van '
                    f'{start_date.strftime("%d %b")} → '
                    f'{end_date.strftime("%d %b %Y")}!'
                )

                col1, col2, col3 = st.columns(3)
                col1.metric('Gemiddelde score', f"{predictions.mean():.1f}")
                col2.metric('Min score', f"{predictions.min():.1f}")
                col3.metric('Max score', f"{predictions.max():.1f}")

    except FileNotFoundError:
        st.error('synthetic_data.csv niet gevonden. Genereer eerst de synthetische data.')


# GECOMBINEERD OVERZICHT — onder beide tabs

st.divider()
st.header('📊 Voorspellingenhistorie')

df_all = load_predictions()

if df_all.empty:
    st.info('Nog geen voorspellingen opgeslagen.')
else:
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])

    # ── Gecombineerde tabel ──
    st.subheader('Alle voorspellingen')

    display_cols = ['timestamp', 'code_module', 'gender',
                    'highest_education', 'predicted_score', 'data_source']
    df_display = df_all.sort_values('timestamp', ascending=False)[display_cols].copy()

    def color_rows(row):
        if row['data_source'] == 'manual':
            return ['background-color: #1a3a5c; color: white'] * len(row)
        else:
            return ['background-color: #3d1a1a; color: #ffcccc'] * len(row)

    styled = df_display.style.apply(color_rows, axis=1)
    st.dataframe(styled, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown('🔵 **Handmatig** (blauw)')
    col2.markdown('🔴 **Synthetisch** (rood)')
    col3.metric('Totaal voorspellingen', len(df_all))

    st.divider()

    # ── Tijdlijn grafiek ──
    st.subheader('📈 Voorspellingen over Tijd')

    df_manual_all = df_all[df_all['data_source'] == 'manual'].sort_values('timestamp')
    df_syn_all = df_all[df_all['data_source'] == 'synthetic'].sort_values('timestamp')

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    # Synthetische lijn
    if not df_syn_all.empty:
        ax.plot(df_syn_all['timestamp'], df_syn_all['predicted_score'],
                color='#ff6b6b', linewidth=0.8, alpha=0.4, label='Synthetisch')
        rolling = df_syn_all['predicted_score'].rolling(window=20).mean()
        ax.plot(df_syn_all['timestamp'], rolling,
                color='#ff4444', linewidth=2.5,
                label='Synthetisch voortschrijdend gem. (n=20)')

    # Handmatige punten
    if not df_manual_all.empty:
        ax.scatter(df_manual_all['timestamp'], df_manual_all['predicted_score'],
                   color='#4fc3f7', s=80, zorder=5,
                   label='Handmatig', marker='D')

    # Totaal gemiddelde
    ax.axhline(df_all['predicted_score'].mean(), color='#69db7c',
               linestyle='--', linewidth=1.5,
               label=f'Totaal gemiddelde ({df_all["predicted_score"].mean():.1f})')

    ax.set_title('Voorspelde Scores over Tijd', fontsize=13,
                 fontweight='bold', color='white')
    ax.set_xlabel('Datum', color='white')
    ax.set_ylabel('Voorspelde Score', color='white')
    ax.set_ylim(0, 100)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric('Totaal', len(df_all))
    col2.metric('Handmatig', len(df_manual_all))
    col3.metric('Synthetisch', len(df_syn_all))