import requests
import pandas as pd
from datetime import datetime, timedelta

def get_lat_lon_from_city(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=10&language=en&format=json"
    response = requests.get(url)
    data = response.json()
    print(f"Geocoding response for {city_name}: {data}")

    if data['results']:
        city_info = data['results'][0]
        lat = city_info['latitude']
        lon = city_info['longitude']
        print(f"Latitude: {lat}, Longitude: {lon}")
        return lat, lon
    else:
        raise ValueError(f"City {city_name} not found")

def get_historical_weather_data(lat, lon, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,daylight_duration"
    response = requests.get(url)
    data = response.json()
    print(f"Historical weather data: {data}")
    
    weather_data = []
    for day in data['daily']['time']:
        idx = data['daily']['time'].index(day)
        weather_data.append({
            'date': day,
            'temperature_2m_max': data['daily']['temperature_2m_max'][idx],
            'temperature_2m_min': data['daily']['temperature_2m_min'][idx],
            'daylight_duration': data['daily']['daylight_duration'][idx]
        })
    
    weather_df = pd.DataFrame(weather_data)
    print(f"Historical weather DataFrame: \n{weather_df}")
    return weather_df

def get_forecast_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,daylight_duration"
    response = requests.get(url)
    data = response.json()
    print(f"Forecast weather data: {data}")

    weather_data = []
    for i in range(5):  # Ensure we only get 5 days of data
        weather_data.append({
            'date': data['daily']['time'][i],
            'temperature_2m_max': data['daily']['temperature_2m_max'][i],
            'temperature_2m_min': data['daily']['temperature_2m_min'][i],
            'daylight_duration': data['daily']['daylight_duration'][i]
        })
    
    weather_df = pd.DataFrame(weather_data)
    print(f"Forecast weather DataFrame: \n{weather_df}")
    return weather_df
