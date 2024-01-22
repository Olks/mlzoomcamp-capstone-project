#!/usr/bin/env python
# coding: utf-8

# # Modeling the energy production and consumption

import os.path
import numpy as np
import pandas as pd
from pickle import dump

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout

import tensorflow as tf
tf.keras.utils.set_random_seed(321)


MODEL_FILE_PATH = 'model_solar_energy_production.keras'
SCALER_FILE_PATH = 'scaler.pkl'

# parameters
SIZE = 256 
LEARNING_RATE = 0.01
DROPRATE = 0.2
n_splits = 5


# read data file
DATA_FILE_PATH = 'solar_energy_data_clean.csv'
df = pd.read_csv(DATA_FILE_PATH)


# define target and features columns
target = 'target'

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

# split into full train and test set
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=111)


# method to create nn model
def get_model(n_inputs, n_outputs, learning_rate=0.01, size=32, droprate=0.5):
    model = Sequential()
    model.add(Dense(64, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(size, input_dim=64, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=droprate))
    model.add(Dense(n_outputs, activation='relu'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error', 
        metrics=['mse'])
    return model
    

def train(df_train, y_train, df_val, y_val, learning_rate=0.01, size=128, droprate=0.5, do_checkpoint=False):

    model = get_model(df_train.shape[1], 1, learning_rate=learning_rate, size=size, droprate=droprate)

    if do_checkpoint:
        print('Training the NN model with checkpoints')
        
        # define checkpoint
        chechpoint = keras.callbacks.ModelCheckpoint(
            MODEL_FILE_PATH,
            save_best_only=True,
            monitor='val_mse',
            mode='min'
        )
    
        history = model.fit(
            df_train,
            y_train,
            batch_size=1024,
            epochs=100,
            validation_data=(df_val, y_val),
            callbacks=[chechpoint]
            # verbose=0
        )
    else:
        print('Training the NN model without checkpoints')
        history = model.fit(
            df_train,
            y_train,
            batch_size=1024,
            epochs=100,
            validation_data=(df_val, y_val),
            verbose=0
        )
    return history.history
    

# validation
print(f'doing validation with LEARNING_RATE={LEARNING_RATE} and inner layer size={SIZE}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.target.values
    y_val = df_val.target.values

    X_train = df_train[features]
    X_val = df_val[features]
    
    # scale
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f'training a model for the fold {fold} ...')
    history = train(X_train_scaled, y_train, X_val_scaled, y_val, learning_rate=LEARNING_RATE, size=SIZE, droprate=DROPRATE)
    
    mse = history['val_mse'][-1]
    scores.append(mse)

    print(f'MSE on fold {fold} is {mse}')
    fold = fold + 1

print('validation results:')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# training the final model
print('training the final model')

y_train = df_full_train.target.values
y_test = df_test.target.values

X_train = df_full_train[features]
X_test = df_test[features]

# scale
scaler = StandardScaler()
scaler.fit(X_train)
dump(scaler, open(SCALER_FILE_PATH, 'wb'))
print(f'Scaler saved to \'{SCALER_FILE_PATH}\'')

X_train_scaled = scaler.transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

history = train(X_train_scaled, y_train, X_test_scaled, y_test, learning_rate=LEARNING_RATE, size=SIZE, droprate=DROPRATE, do_checkpoint=True)

mse = history['val_mse'][-1]
print(f'final model MSE: {mse}')

# check if the model is saved
path = f'./{MODEL_FILE_PATH}'
check_file = os.path.isfile(path)

if check_file:
    print(f'the model is saved to \'{MODEL_FILE_PATH}\'')
else:
    print('model is not yet saved')
