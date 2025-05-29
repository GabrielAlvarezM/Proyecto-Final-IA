import streamlit as st
from streamlit_image_select import image_select
import joblib
import numpy as np

img_clases = {
    "Economía": "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
    "Business": "https://images.unsplash.com/photo-1519125323398-675f0ddb6308",
    "Primera": "https://images.unsplash.com/photo-1526778548025-fa2f459cd5c1"
}

img_hospedaje = {
    "Hostal": "https://cdn.pixabay.com/photo/2016/11/29/02/04/airplane-1866611_1280.jpg",
    "Hotel 2 estrellas": "https://cdn.pixabay.com/photo/2017/03/28/12/10/hotel-2189659_1280.jpg",
    "Hotel 3 estrellas": "https://cdn.pixabay.com/photo/2015/09/18/19/03/africa-943132_1280.jpg",
    "Hotel 4 estrellas": "https://cdn.pixabay.com/photo/2016/03/27/22/16/hotel-1281996_1280.jpg",
    "Hotel 5 estrellas": "https://cdn.pixabay.com/photo/2016/11/18/16/19/hotel-1831072_1280.jpg",
    "Airbnb": "https://cdn.pixabay.com/photo/2017/02/23/13/05/airbnb-2091671_1280.jpg"
}
tipo_hospedaje_dict = {
    'Hostal': 0,
    'Hotel 2 estrellas': 1,
    'Hotel 3 estrellas': 2,
    'Hotel 4 estrellas': 3,
    'Hotel 5 estrellas': 4,
    'Airbnb': 0.5
}

ciudades = ["Ciudad de México", "Madrid", "Paris", "New York", "Buenos Aires"]

@st.cache_resource
def cargar_modelos():
    return {
        "Economía": joblib.load('models/modelo_vuelo_economica.pkl'),
        "Business": joblib.load('models/modelo_vuelo_ejecutiva.pkl'),
        "Primera": joblib.load('models/modelo_vuelo_primera.pkl'),
        "Hospedaje": joblib.load('models/modelo_hospedaje_general.pkl')
    }
modelos = cargar_modelos()

if "paso" not in st.session_state:
    st.session_state.paso = 1

st.title("Calculadora de Viajes")

if st.session_state.paso == 1:
    st.header("1. Elige la clase de vuelo")
    clase = image_select(
        label="Selecciona la clase de vuelo",
        images=list(img_clases.values()),
        captions=list(img_clases.keys()),
        return_value="index"
    )
    if st.button("Siguiente", key="sig1"):
        if clase is not None:
            st.session_state.clase = list(img_clases.keys())[clase]
            st.session_state.paso = 2
            st.rerun()

elif st.session_state.paso == 2:
    st.header("2. Elige tu origen y destino")
    col1, col2 = st.columns(2)
    with col1:
        origen_select = st.selectbox("Origen", ciudades, key="origen_select")
    with col2:
        destino_select = st.selectbox("Destino", ciudades, key="destino_select")
    error = ""
    if st.button("Siguiente", key="sig2"):
        if origen_select == destino_select:
            error = "El origen y destino deben ser diferentes."
        else:
            st.session_state.origen = origen_select
            st.session_state.destino = destino_select
            st.session_state.paso = 3
            st.rerun()
    if st.button("Regresar", key="reg1"):
        st.session_state.paso = 1
        st.rerun()
    if error:
        st.error(error)

elif st.session_state.paso == 3:
    st.header("3. Elige el tipo de hospedaje")
    tipo = image_select(
        label="Selecciona el tipo de hospedaje",
        images=list(img_hospedaje.values()),
        captions=list(img_hospedaje.keys()),
        return_value="index"
    )
    if st.button("Siguiente", key="sig3"):
        if tipo is not None:
            st.session_state.tipo_hospedaje = list(img_hospedaje.keys())[tipo]
            st.session_state.paso = 4
            st.rerun()
    if st.button("Regresar", key="reg2"):
        st.session_state.paso = 2
        st.rerun()

elif st.session_state.paso == 4:
    st.header("4. Detalles finales del viaje")
    anio = st.number_input("Año del viaje", min_value=2023, max_value=2030, value=2025, key="anio")
    distancia = st.number_input("Distancia estimada (km)", min_value=1, value=8000, key="distancia")
    dias = st.number_input("Duración del hospedaje (noches)", min_value=1, value=5, key="dias")
    if st.button("Calcular precio estimado"):
        clase = st.session_state.clase
        tipo = st.session_state.tipo_hospedaje
        tipo_num = tipo_hospedaje_dict[tipo]
        X_vuelo = np.array([[anio, distancia]])
        modelo_vuelo = modelos[clase]
        precio_vuelo = modelo_vuelo.predict(X_vuelo)[0]
        X_hospedaje = np.array([[anio, tipo_num]])
        precio_noche = modelos["Hospedaje"].predict(X_hospedaje)[0]
        precio_hospedaje_total = precio_noche * dias
        st.session_state.resultado = {
            "vuelo": precio_vuelo,
            "hospedaje": precio_hospedaje_total,
            "dias": dias,
            "precio_noche": precio_noche
        }
        st.session_state.paso = 5
        st.rerun()
    if st.button("Regresar", key="reg3"):
        st.session_state.paso = 3
        st.rerun()

elif st.session_state.paso == 5:
    st.header("Resultado de tu viaje")
    r = st.session_state.resultado
    st.success(f"Precio estimado del vuelo ({st.session_state.clase}): ${r['vuelo']:,.2f} USD")
    st.success(f"Hospedaje total por {r['dias']} noches en {st.session_state.tipo_hospedaje}: ${r['hospedaje']:,.2f} USD")
    st.info(f"Precio total aproximado: ${r['vuelo']+r['hospedaje']:,.2f} USD")
    st.markdown("---")
    if st.button("Nuevo cálculo"):
        for key in ["paso", "clase", "origen", "destino", "tipo_hospedaje", "resultado"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.paso = 1
        st.rerun()


#streamlit run app.py