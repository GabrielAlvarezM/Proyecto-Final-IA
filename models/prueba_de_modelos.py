import joblib
import numpy as np
from datetime import datetime

def probar_prediccion(modelo_path, fecha, distancia_o_tipo):
    """
    Prueba una predicción con el modelo guardado.
    
    Args:
        modelo_path (str): Ruta al archivo .pkl con (modelo, scaler).
        fecha (datetime.date o str): Fecha para la predicción.
        distancia_o_tipo (float): Distancia (km) para vuelos o tipo numérico para hospedaje.
        
    Returns:
        float: Precio predicho.
    """
    # Cargar modelo y scaler
    modelo, scaler = joblib.load(modelo_path)
    
    # Convertir fecha a ordinal
    if isinstance(fecha, str):
        fecha = datetime.strptime(fecha, '%Y-%m-%d').date()
    fecha_ord = np.array([[fecha.toordinal()]])
    
    # Crear matriz de entrada
    X = np.array([[fecha.toordinal(), distancia_o_tipo]])
    
    # Escalar y predecir
    X_scaled = scaler.transform(X)
    prediccion = modelo.predict(X_scaled)[0]
    
    # Evitar precio negativo
    prediccion = max(0, prediccion)
    return prediccion

print(probar_prediccion("models\modelo_vuelo_economica.pkl", '2025-06-01', 9000))
print(probar_prediccion("models\modelo_vuelo_ejecutiva.pkl", '2025-06-01', 9000))
print(probar_prediccion("models\modelo_vuelo_primera.pkl", '2025-06-01', 9000))
print(probar_prediccion("models\modelo_hospedaje_general.pkl", '2025-06-01', 5))

