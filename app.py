import streamlit as st
import joblib
import numpy as np

modelo_vuelo_economica = joblib.load('models/modelo_vuelo_economica.pkl')
modelo_vuelo_ejecutiva = joblib.load('models/modelo_vuelo_ejecutiva.pkl')
modelo_vuelo_primera = joblib.load('models/modelo_vuelo_primera.pkl')
modelo_hospedaje = joblib.load('models/modelo_hospedaje_general.pkl')

tipo_hospedaje_dict = {
    'Hostal': 0,
    'Hotel 2 estrellas': 1,
    'Hotel 3 estrellas': 2,
    'Hotel 4 estrellas': 3,
    'Hotel 5 estrellas': 4,
    'Airbnb': 0.5
}

clase_dict = {
    'Económica': modelo_vuelo_economica,
    'Ejecutiva': modelo_vuelo_ejecutiva,
    'Primera': modelo_vuelo_primera
}

st.title("Predicción de precios de viaje ✈️🏨")

anio = st.number_input("Año del viaje", min_value=2000, max_value=2100, value=2025)
distancia = st.number_input("Distancia del vuelo (km)", min_value=1)
clase = st.selectbox("Clase de vuelo", list(clase_dict.keys()))
tipo_hospedaje = st.selectbox("Tipo de hospedaje", list(tipo_hospedaje_dict.keys()))
dias = st.number_input("Duración de hospedaje (días/noches)", min_value=1, value=5)

if st.button("Calcular precio estimado"):

    X_vuelo = np.array([[anio, distancia]])
    modelo_vuelo = clase_dict[clase]
    precio_vuelo = modelo_vuelo.predict(X_vuelo)[0]

    tipo_num = tipo_hospedaje_dict[tipo_hospedaje]
    X_hospedaje = np.array([[anio, tipo_num]])
    precio_noche = modelo_hospedaje.predict(X_hospedaje)[0]
    precio_hospedaje_total = precio_noche * dias

    st.success(f"Precio estimado del vuelo ({clase}): ${precio_vuelo:,.2f} USD")
    st.success(f"Hospedaje total por {dias} noches en {tipo_hospedaje}: ${precio_hospedaje_total:,.2f} USD")
    st.info(f"Precio total aproximado: ${precio_vuelo + precio_hospedaje_total:,.2f} USD")

#streamlit run app.py
