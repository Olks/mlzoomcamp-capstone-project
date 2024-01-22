import numpy as np
import pandas as pd
import json
from pickle import load

from io import StringIO, BytesIO

import keras

from flask import Flask
from flask import request
from flask import jsonify


MODEL_FILE_PATH = 'model_solar_energy_production.keras'
SCALER_FILE_PATH = 'scaler.pkl'

model = keras.models.load_model(MODEL_FILE_PATH)
scaler = load(open('scaler.pkl', 'rb'))

app = Flask('solar_energy_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    solar_data = request.get_json()
    df = pd.Series(solar_data).to_frame().T
    print(solar_data)
    print(df)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['hour'] = df['datetime'].dt.hour

    df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 366)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 366)
    df['sin_hour'] = round(np.sin(2 * np.pi * df['hour'] / 24), 2)
    df['cos_hour'] = round(np.cos(2 * np.pi * df['hour'] / 24), 2)

    features = [
        # solar panels characteristics
        'eic_count',
        'installed_capacity',
        
        # weather forecast
        'temperature',
        'dewpoint',
        'cloudcover_high',
        'cloudcover_low',
        'cloudcover_mid',
        'cloudcover_total',
        '10_metre_u_wind_component',
        '10_metre_v_wind_component',
        'direct_solar_radiation',
        'surface_solar_radiation_downwards',
        'snowfall',
        'total_precipitation',
        
        # historical weather
        'temperature_hist_1h',
        'dewpoint_hist_1h',
        'rain_hist_1h',
        'snowfall_hist_1h',
        'surface_pressure_hist_1h',
        'cloudcover_total_hist_1h',
        'cloudcover_low_hist_1h',
        'cloudcover_mid_hist_1h',
        'cloudcover_high_hist_1h',
        'windspeed_10m_hist_1h',
        'winddirection_10m_hist_1h',
        'shortwave_radiation_hist_1h',
        'direct_solar_radiation_hist_1h',
        'diffuse_radiation_hist_1h',
        
        # date and time
        'year',
        'month',
        'weekday',
        'sin_dayofyear',
        'cos_dayofyear',
        'sin_hour',
        'cos_hour',
        
        # lags of energy production
        'target_1d_lag',
        'target_2d_lag',
        'target_3d_lag',
        'target_4d_lag',
        'target_5d_lag',
        'target_6d_lag',
        'target_7d_lag'
    ]
    
    X = df[features]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    print(predictions[0][0])
    y_pred  = float(predictions[0][0])
    result = {"prediction": y_pred}
    result_json = json.dumps(result) 
    return result_json

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)