# AI-Powered Inventory Management Assistant

## Project Description

The AI-Powered Inventory Management Assistant is an innovative solution designed to help small businesses, particularly in the restaurant industry, optimize their inventory management processes. Leveraging artificial intelligence and machine learning techniques, this project aims to predict future inventory needs based on historical sales data and external factors such as weather conditions.

## Approach

The system uses a combination of neural networks and natural language processing to analyze sales data, predict future inventory needs, and provide actionable insights to business owners. Here's a high-level overview of the approach:

* **Data Collection**: The system collects historical sales data from the business and combines it with weather data from external APIs.
* **Data Processing**: The collected data is cleaned, processed, and prepared for model training.
* **Machine Learning**: A neural network model is trained on the processed data to predict future sales and inventory needs.
* **Natural Language Processing**: An LLM (Language Model) is used to interpret the predictions and generate human-readable insights and recommendations.
* **User Interface**: A web-based interface allows users to upload their data, view predictions, and interact with the AI assistant.

## Bill of Materials (BOM)

### Hardware
* AMD GPUs (for model training, inference, and hosting LLM)
* Server or PC with sufficient CPU and RAM

### Software
* AMD ROCm™ Software
* Python 3.8 or higher
* Flask
* PyTorch
* pandas
* scikit-learn
* requests
* Flask-SocketIO
* React.js
* Bootstrap 5
* Ollama (for running the LLM)

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/jaslinwrk/AIMA.git
   cd AMD-Project-Jasmine
   ```

2. Set up a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate 
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   export SECRET_KEY='your-secret-key'
   export DATABASE_URL='sqlite:///site.db'  # Or your preferred database URL
   ```

5. Initialize the database:
   ```
   flask db init
   flask db migrate
   flask db upgrade
   ```

6. Install and set up Ollama:
* Follow the installation instructions for Ollama from their Github (https://github.com/ollama/ollama)
* Once installed, pull the required model and start the server:
    ```
    ollama pull llama3.1
    ollama serve
    ```

7. Run the Flask application:
   ```
   python run.py
   ```

8. The application should now be running on `http://localhost:5000`.

## Usage

1. Navigate to `http://localhost:5000` in your web browser.
2. Click on "Upload Data" and select your CSV file containing historical sales data. The example dataset can be found in the example folder. Any other dataset should follow the format of this example dataset.
3. Fill in the required information about your business type and location.
4. Confirm the product columns you want to use for prediction.
5. View the sales predictions and AI-generated insights on the main page.
6. Interact with the AI assistant by asking questions or requesting further analysis.

## Acknowledgements

I would like to extend my heartfelt gratitude to AMD for their exceptional support. This project would not have been possible without the AMD AI Hackathon 2023, where I was awarded an AMD Radeon Pro W7900 GPU. The powerful GPU, along with AMD ROCm™ Software, significantly enhanced the development and performance of the AI models used in this project. ROCm's robust and efficient software stack for GPU computing crucial in handling the complex computations required, making the development process smoother and more efficient.
