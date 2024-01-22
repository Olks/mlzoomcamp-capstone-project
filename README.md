# Prediction of Solar Panels Energy Production
This is a capstone project of [**ML Engineering Zoomcamp by DataTalks**](https://github.com/DataTalksClub/machine-learning-zoomcamp) - Cohort 2023
<br>
<br>

<img src="solar-2666770_1920-696x464.jpg"/>

**Estonian Dairy Farm with solar panels installed on the roof [(Image source)](https://ceenergynews.com/innovation/eesti-energia-to-install-smart-solar-park-for-the-estonia-dairy-farm/)**

## Energy Production by Solar Panels Prosumers in Estonia
### Who are prosumers? 
Those kinds of customers of electricity providers that both consume and produce energy are called **prosumers**.<br> 
The term is a portmanteau of the words producer and consumer. Here a single prosumer is an entity that produces energy specifically by solar panels.<br>


### Target
We predict **the amount of energy** that will be **produced** by the group of all solar panels prosumers 
- in the Läänemaa district in Estonia,
- **in the nex hour**.


### Why is the accurate prediction of energy consumption and production important?
The number of prosumers in Estonia (and many other countries) has recently increased causing problems of **energy imbalance**. <br>
This could lead to higher operational costs, potential grid instability, and inefficient use of energy resources. And simply energy can be wasted in the area where in a given time period many prosumers produced more energy than expected.<br> 
Current predictions are very inefficient. Improving them would significantly:
- **reduce the imbalance costs**,
- **improve the reliability of the grid** and
- **make the integration of prosumers into the energy system more efficient and sustainable**.<br>

Accurate predictions of energy production and use is needed if we want to promote renewable energy and **incentivize more customers to become prosumers** showing them that their energy will be managed well.


### Weather 
Weather greatly influences the production of energy by solar panels so a weather data is our main source of features for the model.


### Data
Data is based on data from Kaggle competition: [Enefit - Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers).<br>
To decrease the size of the dataset we limit the data to only **one district in Estonia**.
The chosen district is mapped to **number 6** which is **Läänemaa**, located in western Estonia and borders the Baltic Sea to the north and west.<br>
    

#### Data fields:
##### target (predicted value)
- **target** - The energy production amount for the hour [MWh].

##### solar panels/prosumers characteristics
- **eic_count** - The aggregated number of consumption points (EICs - European Identifier Code).
- **installed_capacity** - Installed photovoltaic solar panel capacity in kilowatts.

##### datetime
- **datetime** - It's the start of the hour period for which we measure/predict the energy production. It's also the beginning of the hour for which the forecast is done and the end of the hour of historical weather data.

##### weather forecast
- temperature - The air temperature at 2 meters above ground in degrees Celsius. Estimated for the end of the 1-hour period.
- dewpoint - The dew point temperature at 2 meters above ground in degrees Celsius. Estimated for the end of the 1-hour period.
- cloudcover_[low/mid/high/total] - The percentage of the sky covered by clouds in the following altitude bands: 0-2 km, 2-6, 6+, and total. Estimated for the end of the 1-hour period.
- 10_metre_[u/v]_wind_component - The [eastward/northward] component of wind speed measured 10 meters above surface in meters per second. Estimated for the end of the 1-hour period.
- direct_solar_radiation - The direct solar radiation reaching the surface on a plane perpendicular to the direction of the Sun accumulated during the hour, in watt-hours per square meter.
- surface_solar_radiation_downwards - The solar radiation, both direct and diffuse, that reaches a horizontal plane at the surface of the Earth, accumulated during the hour, in watt-hours per square meter.
- snowfall - Snowfall over hour in units of meters of water equivalent.
- total_precipitation - The accumulated liquid, comprising rain and snow that falls on Earth's surface over the described hour, in units of meters.


##### historical weather
- temperature_hist_1h - Measured at the end of the 1-hour period.
- dewpoint_hist_1h - Measured at the end of the 1-hour period.
- rain_hist_1h - Different from the forecast conventions. The rain from large scale weather systems of the hour in millimeters.
- snowfall_hist_1h - Different from the forecast conventions. Snowfall over the hour in centimeters.
- surface_pressure_hist_1h - The air pressure at surface in hectopascals.
- cloudcover_[low/mid/high/total]_hist_1h - Different from the forecast conventions. Cloud cover at 0-3 km, 3-8, 8+, and total.
- windspeed_10m_hist_1h - Different from the forecast conventions. The wind speed at 10 meters above ground in meters per second.
- winddirection_10m_hist_1h - Different from the forecast conventions. The wind direction at 10 meters above ground in degrees.
- shortwave_radiation_hist_1h - Different from the forecast conventions. The global horizontal irradiation in watt-hours per square meter.
- direct_solar_radiation_hist_1h
- diffuse_radiation_hist_1h - Different from the forecast conventions. The diffuse solar irradiation in watt-hours per square meter.


<img src="tensorflowkeras.png"/>

### Model

#### Neural Network with Keras
We train a neural network using Keras from Tensorflow library.

#### Steps in fitting the model:
##### Adjusting Learning Rate
##### Adjusting the size of Inner Layer of Neural Network

<br>
<br>
<br>

## Instructions how to run the project locally 
- You need to have a Docker installed on your computer.
1. Clone the repository, run:
	- `git clone https://github.com/Olks/mlzoomcamp-capstone-project.git`
2. Go to project directory and build docker image from Dockerfile, run:   (it may take a few minutes)
	- `docker build -t solar-energy-prediction .`
3. Start a container from the sleep-detection image with:
	- `docker run -it --rm -p 9696:9696 solar-energy-prediction`  
4. To run test prediction (reading sample accelerometer data from data/test_data.json file) open another terminal and run:
	- `python predict-test.py` 
		- need to have Python, requests, pandas and numpy installed locally; check predict-test-requirements.txt and .python_version files
		- returns -> table of onset and wakeup events 
	
	or<br> 
	- `curl -X POST -H "Content-Type: application/json" -d @data/test_data.json http://localhost:9696/predict` 
		- should run without additional installations 
		- returns -> json with onset and wakeup events
5. To run notebooks you need to install all the dependencies:
	- in Terminal go to the project directory and run `pipenv install` and then `pipenv shell` (or first `pip install pipenv` if you don't have pipenv installed) 
		- note that Pipenv file contain Linux and Mac specific library -> <b>gunicorn</b>. If you use Windows please remove it from Pipfile.
	- now you can open noetbooks with Jupyter run e.g. `jupyter lab`  
	
<br>
<br>

## Deployment in AWS Elastic Beanstalk
1. Create an instance of EB in AWS Cloud -> "sleep-detection-env"
- `$ eb init -p docker -r eu-west-1 solar-energy-prediction-env`
- `$ eb create solar-energy-prediction-env`
- `$ eb logs solar-energy-prediction-env`
- `$ eb terminate solar-energy-prediction-env`
2. Copy host address and paste in `predict-test-aws-eb.py`.
3. Run `python predict-test-aws-eb.py`

