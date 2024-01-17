## mlzoomcamp-capstone-project
This is a capstone project of [ML Engineering Zoomcamp by DataTalks](https://github.com/DataTalksClub/machine-learning-zoomcamp) - Cohort 2023

# Energy Behavior of Solar Panels Prosumers in Estonia
### Who are prosumers? 
Those kinds of customers of electricity providers that both consume and produce energy are called **prosumers**.<br> 
The term is a portmanteau of the words producer and consumer. Here a single prosumer is an entity that produces energy specifically by solar panels.<br>


### Unit of the analysis
In this analysis we model and predict the amount of electricity produced and consumed by Estonian energy customers who have installed solar panels.<br> 
The unit of the analysis is **the group of prosumers that share the same**:
- **location**: are located in the same district- feature "county",
- **legal status**: are businesses or not - feature "is_business",
- **contract type with the energy provider**: are of the same product type - feature "product_type".


For the group\* we predict separately:
- consumption of energy,
- production of energy.


- **We predict the amount of energy per hour.**

\*To limit the size of data we will include only one group in the analysis. Details below in the "Data" section.

### Why is the accurate prediction of energy consumption and production important?
The number of prosumers in Estonia (and many other countries) has recently increased causing problems of **energy imbalance**. <br>
This could lead to higher operational costs, potential grid instability, and inefficient use of energy resources. And simply energy can be wasted in the area where in a given time period many prosumers produced more energy than expected.<br> 
Current predictions are very inefficient. Improving them would significantly:
- **reduce the imbalance costs**,
- **improve the reliability of the grid** and
- **make the integration of prosumers into the energy system more efficient and sustainable**.<br>

Accurate predictions of energy production and use is needed if we want to promote renewable energy and **incentivize more customers to become prosumers** showing them that their energy will be managed well.


### Data
Data is based on data from Kaggle competition: [Enefit - Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers).<br>
To decrease the size of the dataset we limit the data to only **one district in Estonia**.
The chosen district is mapped to **number 6** which is **Läänemaa**, located in western Estonia and borders the Baltic Sea to the north and west.<br>
This district has only one group of prosumers:
- county==6 (Läänemaa district),
- is_business==1 (business),
- product_type==3 (possible values and mappings: 0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot").
    

Fields in data.csv:
- **county** - An ID code for the county.
- **is_business** - Boolean for whether or not the prosumer is a business.
- **product_type** - ID code with the following mapping of codes to contract types: {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}.
- **target** - The consumption or production amount for the relevant segment for the hour. The segments are defined by the county, is_business, and product_type.
- **is_consumption** - Boolean for whether or not this row's target is consumption or production.

- **temperature** - The air temperature at 2 meters above ground in degrees Celsius.
- **dewpoint** - The dew point temperature at 2 meters above ground in degrees Celsius.
- **cloudcover_[low/mid/high/total]** - The percentage of the sky covered by clouds in the following altitude bands: 0-2 km, 2-6, 6+, and total.
- **10_metre_[u/v]_wind_component** - The [eastward/northward] component of wind speed measured 10 meters above surface in meters per second.
- **direct_solar_radiation** - The direct solar radiation reaching the surface on a plane perpendicular to the direction of the Sun accumulated during the preceding hour, in watt-hours per square meter.
- **surface_solar_radiation_downwards** - The solar radiation, both direct and diffuse, that reaches a horizontal plane at the surface of the Earth, in watt-hours per square meter.
- **snowfall** - Snowfall over the previous hour in units of meters of water equivalent.
- **total_precipitation** - The accumulated liquid, comprising rain and snow that falls on Earth's surface over the preceding hour, in units of meters.
