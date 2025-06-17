import pandas as pd
import numpy as np
import logging
import os
import pytz
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import argparse

# Set up logging
logging.basicConfig(
    filename='custom_date_prediction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Paths (specific to Bengaluru data)
OUTPUT_DIR = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed"
LIVE_DATA_FILE = os.path.join(OUTPUT_DIR, "live_weather_data.csv")  # Contains Bengaluru data
MODEL_FILE = os.path.join(r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru", "weather_model.keras")  # Trained on Bengaluru data
FORECAST_FILE = os.path.join(OUTPUT_DIR, "custom_date_forecast.csv")

# Define the expected columns (14 features including valid_time)
COLUMNS = ['valid_time', 't2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd', 'strd']

# Load the LSTM model (with fallback)
model = None
if os.path.exists(MODEL_FILE):
    try:
        model = load_model(MODEL_FILE)
        logger.info("LSTM model loaded successfully (trained for Bengaluru).")
    except Exception as e:
        logger.error(f"Failed to load LSTM model from {MODEL_FILE}: {str(e)}. Falling back to heuristic adjustments.")
        model = None
else:
    logger.warning(f"LSTM model file {MODEL_FILE} not found. Using heuristic adjustments.")
    model = None

def load_latest_data():
    try:
        df = pd.read_csv(LIVE_DATA_FILE)
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        latest_data = df.iloc[-1].copy()
        logger.info(f"Loaded latest live data for Bengaluru at {latest_data['valid_time']}")
        return latest_data
    except Exception as e:
        logger.error(f"Failed to load live data for Bengaluru: {str(e)}")
        return None

def preprocess_input(data):
    # Normalize data based on training assumptions (simplified, Bengaluru-specific)
    normalized = data.copy()
    normalized['t2m'] = (data['t2m'] - 273.15) / 30  # Assume temp range 15-45°C for Bengaluru
    normalized['sp'] = (data['sp'] / 100 - 1000) / 20  # Assume pressure 980-1020 hPa
    normalized['tcc'] = data['tcc']
    normalized['ssrd'] = data['ssrd'] / 1200  # Cap at 1200 W/m²
    normalized['strd'] = data['strd'] / 600  # Cap at 600 W/m²
    return normalized

def make_predictions_for_date(latest_data, target_date, use_lstm=True):
    if latest_data is None:
        logger.error("No live data available for Bengaluru predictions.")
        return None

    # Prepare initial input (Bengaluru-specific)
    current_data = latest_data.copy()
    current_time = pd.to_datetime(current_data['valid_time'])
    # Check if already timezone-aware, convert if needed
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize('Asia/Kolkata')
    elif current_time.tzinfo != pytz.timezone('Asia/Kolkata'):
        current_time = current_time.tz_convert('Asia/Kolkata')
    current_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # Validate target date (already localized in run_prediction_for_date)
    if target_date <= current_date:
        logger.error(f"Target date {target_date.date()} must be in the future (after {current_date.date()}) for Bengaluru.")
        return None

    # Define target times for Bengaluru (05:00, 10:00, 15:00, 20:00 IST) using already localized target_date
    target_hours = [5, 10, 15, 20]
    tz = pytz.timezone('Asia/Kolkata')
    target_times = [target_date.replace(hour=h, minute=0, second=0, microsecond=0) for h in target_hours]

    # Rolling prediction to reach the target date (Bengaluru-specific trends)
    while current_date < target_date:
        if use_lstm and model is not None:
            input_seq = preprocess_input(pd.Series([current_data[col] for col in ['t2m', 'sp', 'tcc', 'ssrd', 'strd']]))
            input_seq = np.array(input_seq).reshape(1, 1, 5)
            try:
                predicted_adjustments = model.predict(input_seq)
                temp_adjust = predicted_adjustments[0][0] * 5
                pressure_adjust = predicted_adjustments[0][1] * 10
                cloud_adjust = predicted_adjustments[0][2]
                ssrd_adjust = predicted_adjustments[0][3] * 1200
                strd_adjust = predicted_adjustments[0][4] * 600
            except Exception as e:
                logger.error(f"LSTM prediction failed for Bengaluru: {str(e)}. Using heuristic adjustments.")
                temp_adjust, pressure_adjust, cloud_adjust, ssrd_adjust, strd_adjust = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            temp_adjust, pressure_adjust, cloud_adjust, ssrd_adjust, strd_adjust = 0.0, 0.0, 0.0, 0.0, 0.0

        # Apply Bengaluru-specific seasonal trends
        days_diff = (current_date - current_date.replace(month=4, day=28)).days
        if target_date.month == 5:  # May: warming, pre-monsoon
            temp_trend = 0.1 * days_diff  # Bengaluru pre-monsoon warming
            precip_trend = 0.001 * days_diff  # Light pre-monsoon rain
        elif target_date.month == 6:  # June: monsoon onset
            temp_trend = 0.05 * days_diff  # Monsoon cooling
            precip_trend = 0.005 * days_diff  # Increased monsoon rain
        else:
            temp_trend = 0.0
            precip_trend = 0.0

        current_data['t2m'] += temp_adjust + temp_trend
        current_data['d2m'] += temp_adjust
        current_data['sp'] = (current_data['sp'] / 100 + pressure_adjust) * 100
        current_data['tcc'] = max(0.0, min(1.0, current_data['tcc'] + cloud_adjust))
        current_data['tp'] = max(0.0, current_data['tp'] + precip_trend)
        current_data['ssrd'] += ssrd_adjust
        current_data['strd'] += strd_adjust

        current_date += timedelta(days=1)

    # Generate predictions for Bengaluru
    predicted_data = []
    temp_adjustments = [-2, 1, 3, -1]  # Bengaluru-specific hourly adjustments
    dew_adjustments = [-1, 0, 1, -1]
    pressure_adjustments = [2, 1, -1, 0]
    cloud_adjustments = [0, 0, 0.1, 0]
    for i, (target_time, hour) in enumerate(zip(target_times, target_hours)):
        adjusted_t2m_k = current_data['t2m'] + temp_adjustments[i]
        adjusted_t2m_c = adjusted_t2m_k - 273.15
        adjusted_d2m_k = current_data['d2m'] + dew_adjustments[i]
        adjusted_d2m_c = adjusted_d2m_k - 273.15

        adjusted_sp_hpa = current_data['sp'] / 100 + pressure_adjustments[i]

        adjusted_tcc = max(0.0, min(1.0, current_data['tcc'] + cloud_adjustments[i]))
        adjusted_hcc = adjusted_tcc * 0.4
        adjusted_lcc = adjusted_tcc * 0.3
        adjusted_mcc = adjusted_tcc * 0.3

        adjusted_tp = current_data['tp'] if adjusted_tcc > 0.5 else 0.0
        adjusted_tcrw = 0.1 + adjusted_tp * 2.0

        adjusted_stl1_c = max(15.0, min(40.0, (current_data['stl1'] - 273.15) + temp_adjustments[i] * 0.5))
        adjusted_stl2_c = max(15.0, min(40.0, (current_data['stl2'] - 273.15) + temp_adjustments[i] * 0.5))

        base_ssrd = 0 if hour < 6 or hour >= 18.5 else 575 * (1 - adjusted_tcc)
        adjusted_ssrd = max(0.0, min(1200, base_ssrd + current_data['ssrd'] * 0.3))

        time_factor = np.sin(np.pi * (hour - 6) / 12.5) if 6 <= hour < 18.5 else 0.5
        adjusted_strd = max(200.0, min(600.0, 400.0 * time_factor + (adjusted_t2m_c - 25.0) * 10 - adjusted_tcc * 100))

        predicted_data.append({
            'valid_time': target_time,
            'predicted_t2m': adjusted_t2m_c,
            'predicted_sp': adjusted_sp_hpa,
            'predicted_hcc': adjusted_hcc,
            'predicted_lcc': adjusted_lcc,
            'predicted_mcc': adjusted_mcc,
            'predicted_tcrw': adjusted_tcrw,
            'predicted_stl1': adjusted_stl1_c,
            'predicted_stl2': adjusted_stl2_c,
            'predicted_tp': adjusted_tp,
            'predicted_d2m': adjusted_d2m_c,
            'predicted_tcc': adjusted_tcc,
            'predicted_ssrd': adjusted_ssrd,
            'predicted_strd': adjusted_strd
        })

    predicted_df = pd.DataFrame(predicted_data)
    logger.info(f"Generated predictions for Bengaluru on {target_date.date()}:\n{predicted_df.to_string()}")
    return predicted_df

def format_and_save_output(predicted_df):
    if predicted_df is None or predicted_df.empty:
        logger.error("No prediction data to save for Bengaluru.")
        return

    for idx, row in predicted_df.iterrows():
        summary = {
            'Time': row['valid_time'],
            'Temperature (°C)': row['predicted_t2m'],
            'Surface Pressure (hPa)': row['predicted_sp'],
            'High Cloud Cover (0-1)': row['predicted_hcc'],
            'Low Cloud Cover (0-1)': row['predicted_lcc'],
            'Mid Cloud Cover (0-1)': row['predicted_mcc'],
            'Total Column Rain Water (kg/m²)': row['predicted_tcrw'],
            'Soil Temperature Level 1 (°C)': row['predicted_stl1'],
            'Soil Temperature Level 2 (°C)': row['predicted_stl2'],
            'Precipitation (mm)': row['predicted_tp'],
            'Dew Point (°C)': row['predicted_d2m'],
            'Total Cloud Cover (0-1)': row['predicted_tcc'],
            'Surface Shortwave Radiation (W/m²)': row['predicted_ssrd'],
            'Surface Thermal Radiation (W/m²)': row['predicted_strd']
        }
        logger.info(f"Forecast Summary for Bengaluru at {row['valid_time']}:\n{pd.Series(summary).to_string()}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    predicted_df.to_csv(FORECAST_FILE, index=False)
    logger.info(f"Saved forecast for Bengaluru to {FORECAST_FILE}")

def run_prediction_for_date(target_date_str):
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        tz = pytz.timezone('Asia/Kolkata')
        target_date = tz.localize(target_date.replace(hour=0, minute=0, second=0, microsecond=0))
    except ValueError as e:
        logger.error(f"Invalid date format for Bengaluru. Use YYYY-MM-DD (e.g., 2025-05-15). Error: {str(e)}")
        return

    logger.info(f"Starting prediction job for Bengaluru on {target_date.date()}...")
    latest_data = load_latest_data()
    if latest_data is None:
        logger.warning("Skipping due to data loading failure for Bengaluru.")
        return

    predicted_df = make_predictions_for_date(latest_data, target_date, use_lstm=(model is not None))
    if predicted_df is None:
        logger.warning("Skipping due to prediction failure for Bengaluru.")
        return

    format_and_save_output(predicted_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict weather for a specific date in the future for Bengaluru only.")
    parser.add_argument('--date', type=str, required=True, help="Target date in YYYY-MM-DD format (e.g., 2025-05-15). Must be after the current date (2025-04-28) for Bengaluru.")
    args = parser.parse_args()

    run_prediction_for_date(args.date)