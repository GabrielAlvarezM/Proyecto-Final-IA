import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import joblib


def filtrar_por_ciudad(diccionario, ciudad):
    filtrado = {col: [] for col in diccionario if col != 'Tipo_Hospedaje_Num'}
    if 'Tipo_Hospedaje_Num' in diccionario:
        filtrado['Tipo_Hospedaje_Num'] = []

    for i, nombre_ciudad in enumerate(diccionario['Ciudad']):
        if nombre_ciudad == ciudad:
            for col in diccionario:
                filtrado[col].append(diccionario[col][i])
    return filtrado

datos_vuelos = pd.read_csv('.\\Datasets\\vuelos_historicos_mundo.csv')
datos_hospedaje = pd.read_csv('.\\Datasets\\precios_hospedaje_mundial.csv')

tipo_a_num = {
    'Hostal': 0,
    'Hotel 2 estrellas': 1,
    'Hotel 3 estrellas': 2,
    'Hotel 4 estrellas': 3,
    'Hotel 5 estrellas': 4,
    'Airbnb': 0.5
}

datos_hospedaje['Tipo_Hospedaje_Num'] = datos_hospedaje['Tipo_Hospedaje'].map(tipo_a_num)


diccionarioVuelos = {}
diccionarioHospedaje = {}

# Convertir DataFrames a diccionarios
for columna in datos_vuelos.columns:
    diccionarioVuelos[columna] = datos_vuelos[columna].values.tolist()

for columna in datos_hospedaje.columns:
    diccionarioHospedaje[columna] = datos_hospedaje[columna].values.tolist()

# Extraer fechas en listas separadas
diccionarioVuelos['anio'] = []
diccionarioVuelos['mes'] = []
diccionarioVuelos['dia'] = []

for fecha in diccionarioVuelos['Fecha']:
    diccionarioVuelos['anio'].append(fecha[0:4])
    diccionarioVuelos['mes'].append(fecha[5:7])
    diccionarioVuelos['dia'].append(fecha[8:10])

diccionarioHospedaje['anio'] = []
diccionarioHospedaje['mes'] = []
diccionarioHospedaje['dia'] = []

for fecha in diccionarioHospedaje['Fecha']:
    diccionarioHospedaje['anio'].append(fecha[0:4])
    diccionarioHospedaje['mes'].append(fecha[5:7])
    diccionarioHospedaje['dia'].append(fecha[8:10])

# Clasificar vuelos por clase
diccionarioVuelosEconomica = {col: [] for col in diccionarioVuelos}
diccionarioVuelosEjecutiva = {col: [] for col in diccionarioVuelos}
diccionarioVuelosPrimClase = {col: [] for col in diccionarioVuelos}

for i, clase in enumerate(diccionarioVuelos['Clase']):
    if clase == 0:
        for col in diccionarioVuelos:
            diccionarioVuelosEconomica[col].append(diccionarioVuelos[col][i])
    elif clase == 1:
        for col in diccionarioVuelos:
            diccionarioVuelosEjecutiva[col].append(diccionarioVuelos[col][i])
    elif clase == 2:
        for col in diccionarioVuelos:
            diccionarioVuelosPrimClase[col].append(diccionarioVuelos[col][i])

# Regresión Vuelos economicos
Z = np.array(list(map(int, diccionarioVuelosEconomica['anio']))).reshape(-1, 1)
X = np.array(diccionarioVuelosEconomica['Distancia_km']).reshape(-1, 1)
Y = np.array(diccionarioVuelosEconomica['Precio_USD'])

X_Combinada = np.column_stack((Z, X))
modelo_vuelo_economica = LinearRegression().fit(X_Combinada, Y)
print(f"Puntuacion regresion multiple (Año y Distancia) economicos: {modelo_vuelo_economica.score(X_Combinada, Y)}")
joblib.dump(modelo_vuelo_economica, 'models/modelo_vuelo_economica.pkl')

# Regresión Vuelos ejecutivos
Z = np.array(list(map(int, diccionarioVuelosEjecutiva['anio']))).reshape(-1, 1)
X = np.array(diccionarioVuelosEjecutiva['Distancia_km']).reshape(-1, 1)
Y = np.array(diccionarioVuelosEjecutiva['Precio_USD'])

X_Combinada = np.column_stack((Z, X))
modelo_vuelo_ejecutiva = LinearRegression().fit(X_Combinada, Y)
print(f"Puntuacion regresion multiple (Año y Distancia) ejecutivos: {modelo_vuelo_ejecutiva.score(X_Combinada, Y)}")
joblib.dump(modelo_vuelo_ejecutiva, 'models/modelo_vuelo_ejecutiva.pkl')

# Regresión Vuelos primera
Z = np.array(list(map(int, diccionarioVuelosPrimClase['anio']))).reshape(-1, 1)
X = np.array(diccionarioVuelosPrimClase['Distancia_km']).reshape(-1, 1)
Y = np.array(diccionarioVuelosPrimClase['Precio_USD'])

X_Combinada = np.column_stack((Z, X))
modelo_vuelo_primera = LinearRegression().fit(X_Combinada, Y)
print(f"Puntuacion regresion multiple (Año y Distancia) primera clase: {modelo_vuelo_primera.score(X_Combinada, Y)}")
joblib.dump(modelo_vuelo_primera, 'models/modelo_vuelo_primera.pkl')

print( len(next(iter(diccionarioVuelosEconomica.values()))) )
print( len(next(iter(diccionarioVuelosEjecutiva.values()))) )
print( len(next(iter(diccionarioVuelosPrimClase.values()))) )

# Regresión hospedaje

ciudad = 'Moscú'
diccionarioHospedajeEspecifico = filtrar_por_ciudad(diccionarioHospedaje, ciudad)

Z = np.array(diccionarioHospedajeEspecifico['anio'], dtype=int).reshape(-1, 1)
X = np.array(diccionarioHospedajeEspecifico['Tipo_Hospedaje_Num'], dtype=float).reshape(-1, 1)
Y = np.array(diccionarioHospedajeEspecifico['Precio_Noche_USD'], dtype=float)

X_Combinada = np.column_stack((Z, X))
modelo_hospedaje = LinearRegression().fit(X_Combinada, Y)
print(f"Puntuacion regresion multiple (Año y Tipo hospedaje) general: {modelo_hospedaje.score(X_Combinada, Y)}")
joblib.dump(modelo_hospedaje, 'models/modelo_hospedaje_general.pkl')