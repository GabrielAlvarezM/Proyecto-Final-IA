import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

datos_vuelos = pd.read_csv('.\\Datasets\\vuelos_historicos_mundo.csv')
datos_hospedaje = pd.read_csv('.\\Datasets\\precios_hospedaje_mundial.csv')

diccionarioVuelos = {}
diccionarioHospedaje = {}

for columna in datos_vuelos.columns:
    diccionarioVuelos[columna] = datos_vuelos[columna].values

for columna in datos_hospedaje.columns:
    diccionarioHospedaje[columna] = datos_hospedaje[columna].values

for i in range( len( diccionarioVuelos['Fecha'] ) ):
    diccionarioVuelos['anio'] = diccionarioVuelos['Fecha'][i][0:3]
    diccionarioVuelos['mes'] = diccionarioVuelos['Fecha'][i][5:6]
    diccionarioVuelos['dia'] = diccionarioVuelos['Fecha'][i][8:9]

for i in range( len( diccionarioHospedaje['Fecha'] ) ):
    diccionarioHospedaje['anio'] = diccionarioHospedaje['Fecha'][i][0:3]
    diccionarioHospedaje['mes'] = diccionarioHospedaje['Fecha'][i][5:6]
    diccionarioHospedaje['dia'] = diccionarioHospedaje['Fecha'][i][8:9]

X_SinFiltro = diccionarioVuelos['Distancia_km'].reshape(-1,1)
Y_SinFiltro = diccionarioVuelos['Precio_USD']

modelo_lineal = LinearRegression()
modelo_lineal.fit(X_SinFiltro, Y_SinFiltro)

coeficiente = modelo_lineal.coef_[0]
intercepto = modelo_lineal.intercept_

print(f"Coeficiente: {coeficiente} e intercepto: {intercepto}")

precio_prediccion = modelo_lineal.predict(X_SinFiltro)
plt.scatter(X_SinFiltro, Y_SinFiltro, color='blue', label='Datos reales')
plt.plot(X_SinFiltro, precio_prediccion, color='red', label='Línea de regresión')
plt.scatter(X_SinFiltro, precio_prediccion, color='green', s=10, label='Predicción')


plt.xlabel('Kilometraje')
plt.ylabel('Precio (miles de USD)')
plt.title('Regresión lineal')
plt.legend()
plt.show()