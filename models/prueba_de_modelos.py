import joblib
import numpy as np
from datetime import datetime

def probar_prediccion(modelo_path, fecha, distancia_o_tipo):
    
    modelo, scaler = joblib.load(modelo_path)
    
    if isinstance(fecha, str):
        fecha = datetime.strptime(fecha, '%Y-%m-%d').date()
    fecha_ord = np.array([[fecha.toordinal()]])
    
    X = np.array([[fecha.toordinal(), distancia_o_tipo]])
    
    X_scaled = scaler.transform(X)
    prediccion = modelo.predict(X_scaled)[0]
    
    return prediccion

print(probar_prediccion("models\modelo_vuelo_economica.pkl", '2025-06-01', 9000))
print(probar_prediccion("models\modelo_vuelo_ejecutiva.pkl", '2025-06-01', 9000))
print(probar_prediccion("models\modelo_vuelo_primera.pkl", '2025-06-01', 9000))
print(probar_prediccion("models\modelo_hospedaje_general.pkl", '2025-06-01', 5))

