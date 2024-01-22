#!/usr/bin/env python
# coding: utf-8

import json
import requests
import numpy as np
import pandas as pd


url = 'http://localhost:9696/predict'

test_data_file = 'test_data.json'

 
# Opening JSON file
with open(test_data_file) as json_file:
    data = json.load(json_file)
response = requests.post(url, json=data).json()
prediction = response["prediction"]
print(f"Predicted production of solar energy in Läänemaa district in Estonia, for the next hour starting {data['datetime']}: {round(prediction,3)} MWh")
