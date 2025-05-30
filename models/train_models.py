import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib

# Función para filtrar por ciudad
def filtrar_por_ciudad(diccionario, ciudad):
    filtrado = {col: [] for col in diccionario}
    for i, nombre_ciudad in enumerate(diccionario['Ciudad']):
        if nombre_ciudad == ciudad:
            for col in diccionario:
                filtrado[col].append(diccionario[col][i])
    return filtrado

# Cargar datos
datos_vuelos = pd.read_csv('.\\Datasets\\vuelos_historicos_mundo.csv')
datos_hospedaje = pd.read_csv('.\\Datasets\\precios_hospedaje_mundial.csv')

# Mapear tipos de hospedaje
tipo_a_num = {
    'Hostal': 0,
    'Hotel 2 estrellas': 1,
    'Hotel 3 estrellas': 2,
    'Hotel 4 estrellas': 3,
    'Hotel 5 estrellas': 4,
    'Airbnb': 0.5
}
datos_hospedaje['Tipo_Hospedaje_Num'] = datos_hospedaje['Tipo_Hospedaje'].map(tipo_a_num)

# Convertir DataFrames a diccionarios
diccionarioVuelos = datos_vuelos.to_dict(orient='list')
diccionarioHospedaje = datos_hospedaje.to_dict(orient='list')

# Añadir columna fecha ordinal (vuelos y hospedaje)
diccionarioVuelos['Fecha_Ordinal'] = pd.to_datetime(diccionarioVuelos['Fecha']).map(datetime.toordinal).tolist()
diccionarioHospedaje['Fecha_Ordinal'] = pd.to_datetime(diccionarioHospedaje['Fecha']).map(datetime.toordinal).tolist()

# Separar vuelos por clase
def clasificar_por_clase(diccionario):
    clases = {0: {}, 1: {}, 2: {}}
    for clase in clases:
        clases[clase] = {col: [] for col in diccionario}

    for i, valor in enumerate(diccionario['Clase']):
        for col in diccionario:
            clases[valor][col].append(diccionario[col][i])
    return clases[0], clases[1], clases[2]

diccionarioVuelosEconomica, diccionarioVuelosEjecutiva, diccionarioVuelosPrimClase = clasificar_por_clase(diccionarioVuelos)

# Función para entrenar modelo de regresión lineal
def entrenar_modelo(X_raw, Y, descripcion, ruta_modelo):
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    modelo = LinearRegression().fit(X, Y)
    score = modelo.score(X, Y)
    print(f"Puntuación regresión múltiple - {descripcion}: {score}")
    joblib.dump((modelo, scaler), ruta_modelo)
    return modelo, scaler

# Regresiones de vuelos 
# Económica
fecha_ord = np.array(diccionarioVuelosEconomica['Fecha_Ordinal']).reshape(-1, 1)
dist = np.array(diccionarioVuelosEconomica['Distancia_km']).reshape(-1, 1)
precio = np.array(diccionarioVuelosEconomica['Precio_USD'])
X_combinada = np.column_stack((fecha_ord, dist))
entrenar_modelo(X_combinada, precio, 'Vuelos Económica (fecha ordinal y distancia)', 'models/modelo_vuelo_economica.pkl')

# Ejecutiva
fecha_ord = np.array(diccionarioVuelosEjecutiva['Fecha_Ordinal']).reshape(-1, 1)
dist = np.array(diccionarioVuelosEjecutiva['Distancia_km']).reshape(-1, 1)
precio = np.array(diccionarioVuelosEjecutiva['Precio_USD'])
X_combinada = np.column_stack((fecha_ord, dist))
entrenar_modelo(X_combinada, precio, 'Vuelos Ejecutiva (fecha ordinal y distancia)', 'models/modelo_vuelo_ejecutiva.pkl')

# Primera clase
fecha_ord = np.array(diccionarioVuelosPrimClase['Fecha_Ordinal']).reshape(-1, 1)
dist = np.array(diccionarioVuelosPrimClase['Distancia_km']).reshape(-1, 1)
precio = np.array(diccionarioVuelosPrimClase['Precio_USD'])
X_combinada = np.column_stack((fecha_ord, dist))
entrenar_modelo(X_combinada, precio, 'Vuelos Primera Clase (fecha ordinal y distancia)', 'models/modelo_vuelo_primera.pkl')

#  Regresión de hospedaje 
ciudad = 'Moscú'
diccionarioHospedajeEspecifico = filtrar_por_ciudad(diccionarioHospedaje, ciudad)

fecha_ord = np.array(diccionarioHospedajeEspecifico['Fecha_Ordinal']).reshape(-1, 1)
tipo = np.array(diccionarioHospedajeEspecifico['Tipo_Hospedaje_Num']).reshape(-1, 1)
precio = np.array(diccionarioHospedajeEspecifico['Precio_Noche_USD'])
X_combinada = np.column_stack((fecha_ord, tipo))
entrenar_modelo(X_combinada, precio, f'Hospedaje en {ciudad} (fecha ordinal y tipo)', 'models/modelo_hospedaje_general.pkl')
