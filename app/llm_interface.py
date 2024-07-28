import requests
import json
from flask_socketio import emit
from .weather_data import get_lat_lon_from_city, get_forecast_weather_data

OLLAMA_URL = "http://localhost:11434/api/generate"

def confirm_product_items(columns):
    prompt = f"The uploaded dataset contains the following columns: {', '.join(columns)}. Please confirm the product items for prediction."
    payload = {
        "model": "llama3.1",
        "prompt": prompt
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        
        response_data = response.json()
        confirmed_items = response_data.get('response', '').split(',')  # Expecting a comma-separated response
        return confirmed_items if confirmed_items else None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error occurred: {json_err}")
        print("Response text:", response.text)
    return None

def get_llm_analysis(context_info):
    location = context_info['location']
    lat, lon = get_lat_lon_from_city(location)
    weather_df = get_forecast_weather_data(lat, lon)
    print(f"Weather forecast data for analysis: \n{weather_df}")

    prompt = (
        f"The business is a {context_info['business']} located in {context_info['location']}. "
        "The upcoming weather for the next five days is as follows:\n"
    )
    
    for weather in weather_df.to_dict(orient='records'):
        prompt += (f"Date: {weather['date']}, Max Temperature: {weather['temperature_2m_max']}, "
                   f"Min Temperature: {weather['temperature_2m_min']}, "
                   f"Daylight Duration: {weather['daylight_duration']} minutes\n")
    
    prompt += "Based on this information and the following sales predictions, provide a concise analysis:\n"
    
    for prediction in context_info['predictions']:
        prompt += f"Date: {prediction['date']}, "
        for key, value in prediction.items():
            if key != 'date':
                prompt += f"{key}: {value}, "
        prompt = prompt.rstrip(', ') + '\n'
    
    payload = {
        "model": "llama3:70b",
        "prompt": prompt
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        
        # Streamed responses
        analysis = ""
        for line in response.iter_lines():
            if line:
                response_data = json.loads(line.decode('utf-8'))
                word = response_data.get('response', '')
                analysis += word
                emit('llm_response', {'word': word})
                print(word, end=' ')  # Print each word as it's received with a space to separate them
        
        print("\nFull analysis received from LLM:")  # Print the full accumulated analysis
        print(analysis)  # Print the full accumulated analysis to the terminal
        return #analysis.strip()
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return f"Request error occurred: {req_err}"
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error occurred: {json_err}")
        print("Response text:", response.text)
        return f"JSON decode error occurred: {json_err}"