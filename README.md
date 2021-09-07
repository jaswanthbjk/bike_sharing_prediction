# Bike Sharing Demand prediction

Project to learn basic docker and streamlit skills

## Data

The data for the following example is originally from the [London bike sharing dataset](https://tfl.gov.uk/info-for/open-data-users/our-open-data#on-this-page-5) and is [available on Kaggle.](https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset) Contains OS data © Crown copyright and database rights 2016' and Geomni UK Map data © and database rights [2019] 'Powered by TfL Open Data' freemeteo.com - weather data.

## Getting Started
Run the following command to setup a supported environment and access the web application in your localhost.
```bash
pip install -r requirements.txt

streamlit run app.py
```

## Docker deployment
To build the Docker container and access the application at `localhost:8051` on your browser.
```bash
docker build --tag bike_demand_app:1.0 .
docker run --publish 8051:8051 -it bike_demand_app:1.0
```