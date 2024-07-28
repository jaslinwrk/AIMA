from flask import Blueprint, request, render_template, flash, redirect, url_for, send_file, session
from flask_wtf import FlaskForm, CSRFProtect
import wtforms
from wtforms import StringField, SelectMultipleField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed, FileRequired
from werkzeug.utils import secure_filename
import os
import pandas as pd
import torch
import datetime
import requests
import json
from .model_training import InventoryPredictor, load_data_from_df, train_model
from .utils import clean_sales_data, merge_data, encode_categorical_columns, get_column_names
from .llm_interface import get_llm_analysis
from .weather_data import get_historical_weather_data, get_forecast_weather_data, get_lat_lon_from_city
from app import db, socketio
from flask_socketio import emit


csrf = CSRFProtect()

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class UploadForm(FlaskForm):
    file = wtforms.FileField('CSV File', validators=[FileRequired(), FileAllowed(ALLOWED_EXTENSIONS, 'CSV files only!')])
    business = StringField('Business Type', validators=[DataRequired()])
    location = StringField('Location', validators=[DataRequired()])

class ConfirmForm(FlaskForm):
    product_columns = SelectMultipleField('Product Columns', choices=[])

@main.route('/main')
def index():
    form = UploadForm()
    return render_template('index.html', form=form)

@main.route('/')
def welcome():
    form = UploadForm()
    return render_template('welcome.html', form=form)

@main.route('/upload', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        business = form.business.data
        location = form.location.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Process the sales data
            sales_df = clean_sales_data(filepath)

            # Extract column names and store them in the session
            column_names = get_column_names(sales_df)
            session['column_names'] = column_names
            session['filepath'] = filepath
            session['business'] = business
            session['location'] = location
            
            return redirect(url_for('main.confirm_products'))

    return render_template('upload.html', form=form)

@main.route('/confirm_products', methods=['GET', 'POST'])
def confirm_products():
    form = ConfirmForm()
    column_names = session.get('column_names', [])
    form.product_columns.choices = [(col, col) for col in column_names]
    
    if form.validate_on_submit():
        selected_columns = form.product_columns.data
        if selected_columns:
            session['product_columns'] = selected_columns
            return redirect(url_for('main.upload_file_continue'))

    return render_template('confirm_products.html', form=form, column_names=column_names)

@main.route('/upload_file_continue', methods=['GET'])
def upload_file_continue():
    filepath = session.get('filepath')
    business = session.get('business')
    location = session.get('location')
    product_columns = session.get('product_columns')

    if not filepath or not product_columns:
        flash('Filepath or product columns not found in session.')
        return redirect(url_for('main.index'))

    # Process the sales data
    sales_df = clean_sales_data(filepath)
    print(f"Sales DataFrame: \n{sales_df.head()}")

    # Filter sales data to include only the selected product columns
    sales_df = sales_df[['date'] + product_columns]

    # Get the date range from sales data
    start_date = sales_df['date'].min().strftime('%Y-%m-%d')
    end_date = sales_df['date'].max().strftime('%Y-%m-%d')

    # Fetch historical weather data
    weather_df = fetch_weather_data(location, start_date, end_date)
    print(f"Weather DataFrame: \n{weather_df.head()}")

    # Remove holiday data fetching and merging

    final_df = merge_data(sales_df, weather_df)
    print(f"Merged DataFrame: \n{final_df.head()}")
    final_df.to_csv(os.path.join(UPLOAD_FOLDER, 'processed_data.csv'), index=False)

    # Train the model after uploading and processing data
    model_path = train_model_and_save(final_df)

    # Make predictions for the next five days
    predictions = make_predictions_for_next_five_days(model_path, location)
    print(f"Predictions DataFrame: \n{predictions}")

    # Save predictions in session for display
    session['predictions'] = predictions.to_dict(orient='records')

    # Redirect to index to display predictions
    return redirect(url_for('main.index'))

def train_model_and_save(final_df):
    final_df = encode_categorical_columns(final_df)
    X_train, X_test, y_train, y_test = load_data_from_df(final_df)
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]  # Number of products
    model = train_model(X_train, y_train, input_size, output_size)
    model_path = os.path.join(UPLOAD_FOLDER, 'inventory_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model trained and saved to {model_path}")
    return model_path

def fetch_weather_data(location, start_date=None, end_date=None):
    lat, lon = get_lat_lon_from_city(location)
    if start_date and end_date:
        weather_df = get_historical_weather_data(lat, lon, start_date, end_date)
    else:
        weather_df = get_forecast_weather_data(lat, lon)
    return weather_df


def make_predictions_for_next_five_days(model_path, location):
    lat, lon = get_lat_lon_from_city(location)
    future_dates = pd.date_range(start=datetime.date.today(), periods=5)

    # Fetch forecast weather data for the next five days
    weather_df = get_forecast_weather_data(lat, lon)
    print(f"Forecast weather data for prediction: \n{weather_df}")

    # Ensure we have the same weather columns as used in training
    weather_columns = ['temperature_2m_max', 'temperature_2m_min', 'daylight_duration']
    X = weather_df[weather_columns].values

    # Load the trained model
    model = InventoryPredictor(len(weather_columns), len(fetch_sales_columns()))
    model.load_state_dict(torch.load(model_path))

    # Make predictions
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    # Ensure predictions match the number of future dates
    predictions = predictions[:len(future_dates)]

    # Prepare the result
    predictions_df = pd.DataFrame(future_dates, columns=['date'])
    product_columns = fetch_sales_columns()
    for i, column in enumerate(product_columns):
        predictions_df[f'predicted_{column}'] = predictions[:, i]

    print(f"Predictions DataFrame: \n{predictions_df}")
    return predictions_df

def fetch_sales_columns():
    product_columns = session.get('product_columns', [])
    return [col for col in product_columns if col != 'date']

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    form = UploadForm()
    if request.method == 'GET':
        return render_template('predict.html', form=form)
    if form.validate_on_submit():
        file = form.file.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            print(f"Prediction file saved to {filepath}")

            # Load the processed data and the trained model
            data = pd.read_csv(filepath)
            model_path = os.path.join(UPLOAD_FOLDER, 'inventory_model.pth')
            data = encode_categorical_columns(data)
            input_size = data.shape[1]  # Adjust this based on your data

            model = InventoryPredictor(input_size, data.shape[1])
            model.load_state_dict(torch.load(model_path))

            X = data.values
            X_tensor = torch.tensor(X, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor).numpy()

            for i, column in enumerate(data.columns):
                data[f'predicted_{column}'] = predictions[:, i]
            result_path = os.path.join(UPLOAD_FOLDER, 'predicted_sales.csv')
            data.to_csv(result_path, index=False)
            print(f"Predictions saved to {result_path}")

            return send_file(result_path, as_attachment=True)
    return redirect(request.url)

@socketio.on('user_message')
def handle_user_message(message):
    print('User message:', message)
    # Handle the user's message and generate a response using Ollama
    response = generate_response(message)
    emit('bot_message', response)


def generate_response(message):
    OLLAMA_URL = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": message
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()

        full_response = ''
        for chunk in response.iter_lines():
            if chunk:
                chunk_data = json.loads(chunk.decode('utf-8'))
                sentence = chunk_data.get('response', '')
                words = sentence.split()
                for word in words:
                    full_response += word + ' '
                    emit('bot_message', word + ' ', broadcast=True)
                if chunk_data.get('done', False):
                    break
        
        emit('message_done')
        return #full_response.strip()

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the LLM service: {e}")
        emit('bot_message', 'Error communicating with the LLM service. ')
        emit('message_done')
        return None

