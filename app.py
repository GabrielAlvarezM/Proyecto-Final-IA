import streamlit as st
from streamlit_image_select import image_select
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# -- Diccionarios de imágenes y hospedaje --
img_clases = {
    "Economica": "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
    "Ejecutiva": "https://images.unsplash.com/photo-1519125323398-675f0ddb6308",
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

modelo_path = {
    "Economica": "models/modelo_vuelo_economica.pkl",
    "Ejecutiva": "models/modelo_vuelo_ejecutiva.pkl",
    "Primera": "models/modelo_vuelo_primera.pkl"
}



# -- Inicialización --
if "paso" not in st.session_state:
    st.session_state.paso = 1

# -- Cargar CSV con hospedajes para obtener ciudades --
@st.cache_data
def cargar_ciudades_desde_csv():
    df = pd.read_csv('.\\Datasets\\precios_hospedaje_mundial.csv')
    ciudades_unicas = sorted(df['Ciudad'].dropna().unique())
    return ciudades_unicas
ciudades = cargar_ciudades_desde_csv()

# -- Geocoder --
geolocator = Nominatim(user_agent="mi_app_de_viajes")
coordenadas_cache = {}

def obtener_coordenadas(ciudad):
    if ciudad in coordenadas_cache:
        return coordenadas_cache[ciudad]
    try:
        location = geolocator.geocode(ciudad)
        if location:
            coords = (location.latitude, location.longitude)
            coordenadas_cache[ciudad] = coords
            return coords
        else:
            return None
    except Exception:
        return None

# -- App --
st.title("Calculadora de Viajes")

if st.session_state.paso == 1:
    st.header("1. Elige la clase de vuelo")
    clase_idx = image_select(
        label="Selecciona la clase de vuelo",
        images=list(img_clases.values()),
        captions=list(img_clases.keys()),
        return_value="index"
    )
    if st.button("Siguiente", key="sig1"):
        if clase_idx is not None:
            st.session_state.clase = list(img_clases.keys())[clase_idx]
            st.session_state.paso = 2
            st.rerun()

elif st.session_state.paso == 2:
    st.header("2. Elige tu origen y destino")

    origen_select = st.selectbox("Origen", ciudades, key="origen_select")
    destino_select = st.selectbox("Destino", ciudades, key="destino_select")

    if origen_select != destino_select:
        coord_origen = obtener_coordenadas(origen_select)
        coord_destino = obtener_coordenadas(destino_select)

        if coord_origen and coord_destino:
            distancia_km = geodesic(coord_origen, coord_destino).kilometers
            st.success(f"✈️ Distancia estimada entre {origen_select} y {destino_select}: **{distancia_km:.2f} km**")

            st.session_state.distancia_real = distancia_km

            # Mapa
            mapa = folium.Map(
                location=[(coord_origen[0] + coord_destino[0]) / 2,
                          (coord_origen[1] + coord_destino[1]) / 2],
                zoom_start=3
            )
            folium.Marker(coord_origen, tooltip=origen_select, icon=folium.Icon(color='blue')).add_to(mapa)
            folium.Marker(coord_destino, tooltip=destino_select, icon=folium.Icon(color='red')).add_to(mapa)
            folium.PolyLine([coord_origen, coord_destino], color='green', weight=3).add_to(mapa)
            st_folium(mapa, width=700, height=450)
        else:
            st.error("No se pudieron obtener coordenadas para alguna de las ciudades.")
    else:
        st.warning("El origen y destino no pueden ser iguales.")

    if st.button("Siguiente", key="sig2"):
        if origen_select == destino_select:
            st.error("El origen y destino deben ser diferentes.")
        elif "distancia_real" not in st.session_state:
            st.error("Debes seleccionar origen y destino válidos antes de continuar.")
        else:
            st.session_state.origen = origen_select
            st.session_state.destino = destino_select
            st.session_state.paso = 3
            st.rerun()

    if st.button("Regresar", key="reg1"):
        st.session_state.paso = 1
        st.rerun()

elif st.session_state.paso == 3:
    st.header("3. Elige el tipo de hospedaje y fecha de viaje")

    df_hospedaje = pd.read_csv('.\\Datasets\\precios_hospedaje_mundial.csv')
    df_filtrado = df_hospedaje[df_hospedaje['Ciudad'] == st.session_state.destino]

    tipos_disponibles = sorted(df_filtrado['Tipo_Hospedaje'].unique())

    tipo = st.selectbox("Selecciona el tipo de hospedaje", tipos_disponibles, key="tipo_hospedaje_select")

    min_fecha = datetime.date.today()
    fecha_viaje = st.date_input("Selecciona la fecha de inicio del viaje", value=min_fecha, min_value=min_fecha, key="fecha_viaje")

    if st.button("Siguiente", key="sig3"):
        if tipo and fecha_viaje:
            st.session_state.tipo_hospedaje = tipo
            # No reasignar st.session_state.fecha_viaje, ya lo guarda date_input
            st.session_state.paso = 4
            st.rerun()

    if st.button("Regresar", key="reg2"):
        st.session_state.paso = 2
        st.rerun()

elif st.session_state.paso == 4:
    st.header("4. Detalles finales del viaje")

    distancia = st.session_state.get("distancia_real", 8000)
    st.info(f"Distancia calculada automáticamente: **{distancia:.2f} km**")

    dias = st.number_input("Duración del hospedaje (noches)", min_value=1, value=5, key="dias")

    if st.button("Calcular precio estimado"):

        df_hospedaje = pd.read_csv('.\\Datasets\\precios_hospedaje_mundial.csv')
        df_filtrado = df_hospedaje[df_hospedaje['Ciudad'] == st.session_state.destino].copy()

        df_filtrado['Fecha_Ordinal'] = pd.to_datetime(df_filtrado['Fecha']).apply(lambda x: x.toordinal())
        df_filtrado['Tipo_Num'] = df_filtrado['Tipo_Hospedaje'].map(tipo_hospedaje_dict)

        df_filtrado = df_filtrado.dropna(subset=['Fecha_Ordinal', 'Tipo_Num', 'Precio_Noche_USD'])

        X = df_filtrado[['Fecha_Ordinal', 'Tipo_Num']]
        y = df_filtrado['Precio_Noche_USD']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        modelo = LinearRegression()
        modelo.fit(X_scaled, y)

        fecha_viaje = st.session_state.get("fecha_viaje", datetime.date.today())
        fecha_ordinal = fecha_viaje.toordinal()
        tipo_num = tipo_hospedaje_dict.get(st.session_state.tipo_hospedaje, 0)

        X_pred = np.array([[fecha_ordinal, tipo_num]])
        X_pred_scaled = scaler.transform(X_pred)

        precio_noche = max(0, modelo.predict(X_pred_scaled)[0])
        precio_total = precio_noche * dias

        # Cargar modelo según clase
        modelo_path = {
            "Economica": "./models/modelo_vuelo_economica.pkl",
            "Ejecutiva": "./models/modelo_vuelo_ejecutiva.pkl",
            "Primera": "./models/modelo_vuelo_primera.pkl"
        }
                
            
    modelo_clase = st.session_state.get("clase", "Economica")
    ruta_modelo = modelo_path.get(modelo_clase)

    try:
        # Cargar el modelo y el scaler
        modelo_vuelo, scaler_vuelo = joblib.load(ruta_modelo)

        fecha_viaje = st.session_state.get("fecha_viaje", datetime.date.today())
        fecha_ordinal = fecha_viaje.toordinal()

        # Crear el input con las dos características: fecha ordinal y distancia
        X_pred_vuelo = np.array([[fecha_ordinal, distancia]])
        X_pred_vuelo_scaled = scaler_vuelo.transform(X_pred_vuelo)

        # Predecir el precio del vuelo
        precio_vuelo = max(0, modelo_vuelo.predict(X_pred_vuelo_scaled)[0])

    except Exception as e:
        st.error(f"No se pudo cargar el modelo para la clase {modelo_clase.lower()}: {e}")
        precio_vuelo = distancia * 0.1  # Fallback simple


    st.session_state.resultado = {
        "vuelo": precio_vuelo,
        "hospedaje": precio_total,
        "dias": dias,
        "precio_noche": precio_noche
    }
    st.session_state.paso = 5
    st.rerun()

    if st.button("Regresar", key="reg3"):
        st.session_state.paso = 3
        st.rerun()


elif st.session_state.paso == 5:
    res = st.session_state.get("resultado", None)
    if res:
        st.success(f"Precio estimado de vuelo: ${res['vuelo']:.2f} USD")
        st.success(f"Precio estimado de hospedaje ({res['dias']} noches): ${res['hospedaje']:.2f} USD")
        st.info(f"Precio promedio por noche: ${res['precio_noche']:.2f} USD")

    if st.button("Nuevo cálculo"):
        for key in ["clase", "origen", "destino", "distancia_real", "tipo_hospedaje", "fecha_viaje", "resultado"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.paso = 1
        st.rerun()
