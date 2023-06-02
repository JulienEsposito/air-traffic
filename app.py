import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from datetime import timedelta

traffic_df = pd.read_parquet(r'C:\Users\julie\Documents\Fac\traffic_10lines.parquet')

st.title('Traffic Forecaster')

with st.sidebar:
    home_airports = traffic_df['home_airport'].unique()
    paired_airports = traffic_df['paired_airport'].unique()
    home_airport = st.selectbox('Home Airport', home_airports)
    paired_airport = st.selectbox('Paired Airport', paired_airports)
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider('Days of forecast', 7, 150, 1)
    run_forecast = st.button('Forecast')

# Function to extract data for the specific route
def generate_route_df(traffic_df: pd.DataFrame, homeAirport: str, pairedAirport: str) -> pd.DataFrame:
    try:
        _df = traffic_df[(traffic_df['home_airport'] == homeAirport) & (traffic_df['paired_airport'] == pairedAirport)]
        _df = _df.groupby(['home_airport', 'paired_airport', 'date']).agg({'pax': 'sum'}).reset_index()
        return _df
    except ValueError:
        st.error('This route does not exist.')
        return pd.DataFrame()  # Return an empty DataFrame if the route does not exist

# Extraction des données de la route spécifique
route_df = generate_route_df(traffic_df, home_airport, paired_airport)

if not route_df.empty:
    # Renommage des colonnes pour être compatibles avec Prophet
    route_df = route_df.rename(columns={'date': 'ds', 'pax': 'y'})

    # Conversion de la colonne 'ds' en format datetime
    route_df['ds'] = pd.to_datetime(route_df['ds'])

    # Optimisation du modèle
    baseline_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )

    # Entraînement du modèle sur les données de la route spécifique
    baseline_model.fit(route_df)

    # Création d'un dataframe pour les prévisions futures
    future_df = baseline_model.make_future_dataframe(periods=nb_days, freq='D')

    # Génération des prévisions pour la période future
    forecast = baseline_model.predict(future_df)

    # Évaluation du modèle
    cv_results = cross_validation(
        baseline_model,
        initial='500 days',
        period='180 days',
        horizon='365 days'
    )

    # Performance Metrics
    performance = performance_metrics(cv_results)

    # Création du graphique des prévisions avec Plotly
    fig = go.Figure()

    # Ajout des données historiques au graphique
    fig.add_trace(go.Scatter(x=route_df['ds'], y=route_df['y'], name='Données historiques'))

    # Ajout des prévisions au graphique
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prévisions'))

    # Mise en forme du graphique
    fig.update_layout(
        title=f'Prévisions de trafic aérien entre {home_airport} et {paired_airport}',
        xaxis_title='Date',
        yaxis_title='Nombre de passagers',
        legend=dict(
            x=0,
            y=1,
            traceorder='normal',
            font=dict(size=10),
        ),
        autosize=True,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    # Utilisation de st.plotly_chart() pour afficher le graphique dans Streamlit
    st.plotly_chart(fig)

    st.write('Home Airport selected:', home_airport)
    st.write('Paired Airport selected:', paired_airport)
    st.write('Days of forecast:', nb_days)
    st.write('Date selected:', forecast_date)

    if run_forecast:
        # Prévoir la période sélectionnée
        forecast_start = pd.to_datetime(forecast_date)
        forecast_end = forecast_start + timedelta(days=nb_days)
        forecast_period = forecast[(forecast['ds'] >= forecast_start) & (forecast['ds'] <= forecast_end)]

      