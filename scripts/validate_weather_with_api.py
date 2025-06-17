import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    filename='validation_with_api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# API configuration (Meteostat via RapidAPI)
API_URL = "https://meteostat.p.rapidapi.com/point/hourly"
LOCATION = {"latitude": 12.9719, "longitude": 77.5937}
API_KEY = "14d0664ca1msh48d8c8f99ee38f7p129d05jsnf0765bf511c5"  # Your RapidAPI key
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "meteostat.p.rapidapi.com"
}
PARAMS = {
    "lat": 12.9719,
    "lon": 77.5937,
    "start": "2025-04-22",  # Matches your predicted data
    "end": "2025-04-22",
    "tz": "Asia/Kolkata"
}

def fetch_weather_data():
    if not API_KEY:
        logger.error("API key is missing. Please provide a valid RapidAPI key.")
        return None
    try:
        response = requests.get(API_URL, headers=HEADERS, params=PARAMS)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched data from Meteostat API (via RapidAPI) for {LOCATION['latitude']}, {LOCATION['longitude']}")
        return data
    except Exception as e:
        logger.error(f"Failed to fetch API data: {str(e)}")
        return None

def process_api_data(api_data):
    if not api_data or 'data' not in api_data:
        logger.error("API data is empty or missing data field.")
        return pd.DataFrame()

    hourly = api_data['data']
    records = []
    for entry in hourly:
        dt = pd.to_datetime(entry['time']).tz_localize('Asia/Kolkata')
        hour = dt.hour
        if hour in [5, 10, 15, 20]:  # Match your time slots in IST
            # Map Meteostat variables to your variable names
            cloud_cover = entry['coco'] / 10 if entry['coco'] else 0  # Convert cloud cover code to fraction (approx)
            record = {
                'valid_time': dt,
                'predicted_t2m': entry['temp'] if entry['temp'] is not None else 0,
                'predicted_d2m': entry['dwpt'] if entry['dwpt'] is not None else 0,
                'predicted_tp': entry['prcp'] if entry['prcp'] is not None else 0,
                'predicted_hcc': cloud_cover,
                'predicted_tcc': cloud_cover
            }
            records.append(record)
            logger.debug(f"Added record for {dt} at hour {hour}")

    if not records:
        logger.error("No records found for the specified hours (5, 10, 15, 20) in IST.")
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates(subset=['valid_time'])
    logger.info(f"Raw API DataFrame:\n{df.head().to_string()}")
    return df

def validate_weather_predictions():
    try:
        logger.info("Starting validation with Meteostat API (via RapidAPI)...")

        # Load predicted data
        pred_df = pd.read_csv(
            r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\weather_predictions.csv")
        pred_df['valid_time'] = pd.to_datetime(pred_df['valid_time']).dt.tz_localize('Asia/Kolkata')
        # No need to convert predicted_sp since it's removed
        logger.info(f"Loaded predicted data. Shape: {pred_df.shape}")
        logger.info(f"Predicted data columns: {pred_df.columns.tolist()}")
        logger.info(f"Sample predicted data:\n{pred_df.head().to_string()}")

        # Fetch and process actual data from API
        api_data = fetch_weather_data()
        if api_data is None:
            return
        actual_df = process_api_data(api_data)
        if actual_df.empty:
            logger.error("No valid API data processed.")
            return
        logger.info(f"Processed API data. Shape: {actual_df.shape}")
        logger.info(f"Actual data columns: {actual_df.columns.tolist()}")
        logger.info(f"Sample actual data:\n{actual_df.head().to_string()}")

        # Filter predicted data for April 22, 2025
        pred_df = pred_df[(pred_df['valid_time'].dt.year == 2025) &
                          (pred_df['valid_time'].dt.month == 4) &
                          (pred_df['valid_time'].dt.day == 22)]

        # Align time slots
        actual_df = actual_df[actual_df['valid_time'].dt.hour.isin([5, 10, 15, 20])]
        pred_df = pred_df[pred_df['valid_time'].dt.hour.isin([5, 10, 15, 20])]

        # Filter for common timestamps
        common_times = actual_df['valid_time'].isin(pred_df['valid_time'])
        actual_df = actual_df[common_times].sort_values('valid_time').reset_index(drop=True)

        common_times = pred_df['valid_time'].isin(actual_df['valid_time'])
        pred_df = pred_df[common_times].sort_values('valid_time').reset_index(drop=True)

        if actual_df.empty or pred_df.empty:
            logger.error("No overlapping timestamps found between predicted and actual data.")
            return

        # Features to validate (removed sp and ssrd)
        features = ['predicted_t2m', 'predicted_d2m', 'predicted_tp', 'predicted_hcc', 'predicted_tcc']
        actual_df[features] = actual_df[features].fillna(0)
        pred_df[features] = pred_df[features].fillna(0)

        # Compute metrics
        metrics = {}
        for feature in features:
            if feature in actual_df.columns and feature in pred_df.columns:
                mae = mean_absolute_error(actual_df[feature], pred_df[feature])
                rmse = np.sqrt(mean_squared_error(actual_df[feature], pred_df[feature]))
                bias = np.mean(pred_df[feature] - actual_df[feature])
                metrics[feature] = {'MAE': mae, 'RMSE': rmse, 'Bias': bias}
                logger.info(f"{feature} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Bias: {bias:.2f}")
            else:
                logger.warning(f"Feature {feature} not found in one of the datasets.")

        # Log unavailable features
        unavailable_features = ['predicted_sp', 'predicted_ssrd', 'predicted_lcc', 'predicted_mcc', 'predicted_tcrw', 'predicted_stl1', 'predicted_stl2', 'predicted_strd']
        logger.info("The following features could not be validated due to unavailability in Meteostat API or poor validation performance:")
        for feature in unavailable_features:
            logger.info(f"- {feature}")

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv(
            r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\validation_metrics.csv",
            index=True
        )
        logger.info(f"Validation metrics saved to validation_metrics.csv")
        logger.info(f"Validation metrics:\n{metrics_df.to_string()}")

    except Exception as e:
        logger.error(f"Error in validation: {str(e)}")
        raise

if __name__ == "__main__":
    validate_weather_predictions()