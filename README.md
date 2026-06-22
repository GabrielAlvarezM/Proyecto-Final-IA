# Expepibe

Expepibe is a Streamlit portfolio project that estimates the cost of a trip by combining two components:

- Flight price prediction based on travel date, distance, and cabin class.
- Lodging price estimation based on destination city, lodging type, and trip date.

The app guides the user through a short multi-step flow: choose a flight class, select origin and destination, review the route on a map, pick lodging preferences, and get an estimated travel budget in USD.

## Why This Project

This project was built to showcase practical machine learning in an interactive product instead of only in notebooks. It combines:

- Data preprocessing with `pandas` and `numpy`
- Regression modeling with `scikit-learn`
- Saved ML artifacts with `joblib`
- Geolocation and distance calculation with `geopy`
- Interactive visualization with `folium`
- A full user-facing interface with `Streamlit`

## Main Features

- Multi-step Streamlit interface with session state
- Image-based flight class selection
- City lookup from the lodging dataset
- Route visualization between origin and destination
- Automatic geodesic distance calculation
- Flight price prediction for economy, business, and first class
- Lodging price estimation by city, date, and accommodation type
- Final cost breakdown for flight and stay

## Machine Learning Approach

### Flights

The project uses separate linear regression models for each flight class:

- `Economica`
- `Ejecutiva`
- `Primera`

Each model is trained with:

- `Fecha_Ordinal`
- `Distancia_km`

The serialized models are stored in [`models/`](./models).

### Lodging

Lodging pricing is estimated from the dataset during app execution using:

- `Fecha_Ordinal`
- Encoded lodging type

The app fits a linear regression model on the filtered destination city and predicts the nightly price for the selected trip date.

## Tech Stack

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib
- geopy
- folium
- streamlit-folium
- streamlit-image-select

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- Datasets/
|   |-- vuelos_historicos_mundo.csv
|   `-- precios_hospedaje_mundial.csv
|-- models/
|   |-- train_models.py
|   |-- prueba_de_modelos.py
|   |-- modelo_vuelo_economica.pkl
|   |-- modelo_vuelo_ejecutiva.pkl
|   `-- modelo_vuelo_primera.pkl
|-- src/
`-- imagen logo/
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL shown by Streamlit in your browser.

## Demo Flow

1. Select a flight class.
2. Choose origin and destination cities.
3. Review the route and calculated distance on the map.
4. Select lodging type and travel date.
5. Enter the number of nights.
6. Generate the estimated flight and lodging costs.

## Files of Interest

- [`app.py`](./app.py): main Streamlit application
- [`models/train_models.py`](./models/train_models.py): training script for flight and lodging experiments
- [`models/prueba_de_modelos.py`](./models/prueba_de_modelos.py): quick prediction test script
- [`Datasets/vuelos_historicos_mundo.csv`](./Datasets/vuelos_historicos_mundo.csv): historical flight dataset
- [`Datasets/precios_hospedaje_mundial.csv`](./Datasets/precios_hospedaje_mundial.csv): lodging pricing dataset

## Portfolio Value

This project demonstrates:

- End-to-end ML application development
- Model packaging for inference
- Data-driven UI design
- Integration of prediction, mapping, and interaction in one product
- Translation of technical work into a usable travel-planning experience

## Future Improvements

- Improve UI styling and branding
- Add model evaluation metrics to the interface
- Replace geocoding calls with cached structured location data
- Add validation, error recovery, and loading states
- Introduce more robust models beyond linear regression

