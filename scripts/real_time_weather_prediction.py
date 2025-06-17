import pandas as pd
import numpy as np
import requests
import logging
import os
import pytz

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(
    filename='real_time_prediction.log' ,
    level=logging.INFO ,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Paths
OUTPUT_DIR = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed"
LIVE_DATA_FILE = os.path.join(OUTPUT_DIR , "live_weather_data.csv")

# Weatherstack API configuration
API_URL = "http://api.weatherstack.com/current"
API_KEY = "06b075ebeada60b035b74a05351d2d3b"
PARAMS = {
    "access_key": API_KEY ,
    "query": "12.9719,77.5937" ,  # Coordinates for Bengaluru
    "units": "m"
}

# Define the expected columns (14 features including valid_time)
COLUMNS = ['valid_time' , 't2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' ,
           'ssrd' , 'strd']


def fetch_live_weather_data():
    try:
        response = requests.get(API_URL , params=PARAMS)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Raw API response: {data}")
        if 'success' in data and data['success'] is False:
            logger.error(f"API Error: {data.get('error' , {}).get('info' , 'Unknown error')}")
            return None
        if 'current' not in data or 'location' not in data:
            logger.error("API response missing 'current' or 'location' keys.")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to fetch live data: {str(e)}")
        return None


def process_live_data(api_data):
    if not api_data:
        logger.error("No API data received.")
        return None

    current = api_data['current']
    location = api_data['location']
    try:
        observation_time = pd.to_datetime(location['localtime']).tz_localize('Asia/Kolkata')
    except Exception as e:
        logger.error(f"Failed to parse observation time: {str(e)}")
        return None

    try:
        temp = float(current.get('temperature' , 0)) + 273.15  # Convert to Kelvin
        humidity = float(current.get('humidity' , 0))
        dew_point = (temp - 273.15) - ((100 - humidity) / 5) + 273.15 if humidity > 0 else temp  # Kelvin
        cloud_cover = float(current.get('cloudcover' , 0)) / 100 if current.get('cloudcover') is not None else 0
        pressure = float(current.get('pressure' , 0)) * 100 if current.get('pressure') else 0
        precip = float(current.get('precip' , 0))
    except (ValueError , TypeError) as e:
        logger.error(f"Failed to parse API data: {str(e)}")
        return None

    hour = observation_time.hour
    # Radiation estimation (Bengaluru sunrise ~6:00 AM IST, sunset ~6:34 PM IST)
    is_daytime = 6 <= hour < 18.5
    ssrd = min(800.0 * (hour - 6) / 12 * (1 - cloud_cover) , 1200.0) if is_daytime else 0.0  # Cap at 1200 W/m²
    strd = min(400.0 * (1 - 0.5 * cloud_cover) , 600.0) if is_daytime else 300.0  # Cap at 600 W/m²

    hcc = cloud_cover * 0.4
    lcc = cloud_cover * 0.3
    mcc = cloud_cover * 0.3

    record = {
        'valid_time': observation_time ,
        't2m': temp ,
        'sp': pressure ,
        'hcc': hcc ,
        'lcc': lcc ,
        'mcc': mcc ,
        'tcrw': 0.1 ,
        'stl1': 25.0 + 273.15 ,
        'stl2': 26.0 + 273.15 ,
        'tp': precip ,
        'd2m': dew_point ,
        'tcc': cloud_cover ,
        'ssrd': ssrd ,
        'strd': strd
    }
    logger.info(f"Processed live data:\n{pd.Series(record).to_string()}")

    live_df = pd.DataFrame([record])
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if os.path.exists(LIVE_DATA_FILE):
        try:
            existing_df = pd.read_csv(LIVE_DATA_FILE , names=COLUMNS , header=0 , skiprows=0)
            existing_df['valid_time'] = pd.to_datetime(existing_df['valid_time'])

            mask_celsius = existing_df['t2m'] < 100
            existing_df.loc[mask_celsius , 't2m'] = existing_df.loc[mask_celsius , 't2m'] + 273.15
            existing_df.loc[mask_celsius , 'd2m'] = existing_df.loc[mask_celsius , 'd2m'] + 273.15
            existing_df.loc[mask_celsius , 'stl1'] = existing_df.loc[mask_celsius , 'stl1'] + 273.15
            existing_df.loc[mask_celsius , 'stl2'] = existing_df.loc[mask_celsius , 'stl2'] + 273.15

            existing_df['ssrd'] = existing_df['ssrd'].clip(upper=1200.0)
            existing_df['strd'] = existing_df['strd'].clip(upper=600.0)

            for col in COLUMNS:
                if col not in existing_df.columns:
                    existing_df[col] = 0.0 if col != 'valid_time' else pd.NaT
            existing_df = existing_df[COLUMNS]

            existing_df.to_csv(LIVE_DATA_FILE , index=False)
            logger.info("Corrected and saved live_weather_data.csv with 14 columns.")
        except Exception as e:
            logger.error(f"Failed to read and fix existing CSV: {str(e)}. Starting fresh.")
            existing_df = pd.DataFrame(columns=COLUMNS)
            existing_df.to_csv(LIVE_DATA_FILE , index=False)
    else:
        existing_df = pd.DataFrame(columns=COLUMNS)
        existing_df.to_csv(LIVE_DATA_FILE , index=False)

    combined_df = pd.concat([existing_df , live_df]).drop_duplicates(subset=['valid_time'] , keep='last')
    combined_df = combined_df[COLUMNS]
    combined_df.to_csv(LIVE_DATA_FILE , index=False)

    logger.info(f"Saved live weather data to {LIVE_DATA_FILE}")
    return live_df


def make_time_adjusted_prediction(live_df):
    try:
        if live_df is None or live_df.empty:
            logger.warning("No live data available for time-adjusted prediction.")
            return None

        latest_data = live_df.iloc[-1].copy()
        latest_t2m = latest_data['t2m']
        latest_d2m = latest_data['d2m']
        latest_sp = latest_data['sp'] / 100  # Convert to hPa
        latest_tcc = latest_data['tcc']
        latest_ssrd = latest_data['ssrd']
        latest_strd = latest_data['strd']

        last_time = live_df['valid_time'].iloc[-1]
        tz = pytz.timezone('Asia/Kolkata')
        last_time_naive = last_time.tz_localize(None) if last_time.tzinfo else last_time
        base_date = tz.localize(last_time_naive.replace(hour=0 , minute=0 , second=0 , microsecond=0))
        target_times = [base_date.replace(hour=h) for h in [5 , 10 , 15 , 20]]
        target_hours = [5 , 10 , 15 , 20]

        # Adjustments
        temp_adjustments = [-2 , 1 , 3 , -1]  # In °C
        dew_adjustments = [-1 , 0 , 1 , -1]  # In °C
        pressure_adjustments = [2 , 1 , -1 , 0]  # In hPa
        cloud_adjustments = [0 , 0 , 0.1 , 0]

        # Adjusted SSRD base values (scaled to match 200 W/m² at 12:06 IST)
        scaling_factor = 200 / 150  # Live SSRD at 12:06 IST / expected at 12:30 IST
        ssrd_base = {
            5: 0 ,  # 05:00 IST (before sunrise)
            10: 37.5 * scaling_factor ,  # 10:00 IST (interpolated)
            15: min(1200 , 575 * scaling_factor * 2) ,  # 15:00 IST, scaled for peak
            20: 0  # 20:00 IST (after sunset)
        }

        predicted_data = []
        for i , (target_time , hour) in enumerate(zip(target_times , target_hours)):
            # Temperature and dew point
            adjusted_t2m_k = latest_t2m + temp_adjustments[i]
            adjusted_t2m_c = adjusted_t2m_k - 273.15
            adjusted_d2m_k = latest_d2m + dew_adjustments[i]
            adjusted_d2m_c = adjusted_d2m_k - 273.15

            # Surface pressure
            adjusted_sp_hpa = latest_sp + pressure_adjustments[i]

            # Cloud cover
            adjusted_tcc = max(0.0 , min(1.0 , latest_tcc + cloud_adjustments[i]))
            adjusted_hcc = adjusted_tcc * 0.4
            adjusted_lcc = adjusted_tcc * 0.3
            adjusted_mcc = adjusted_tcc * 0.3

            # Precipitation and total column rain water
            adjusted_tp = 0.0 if adjusted_tcc <= 0.5 else 0.5 * (adjusted_tcc - 0.5) ** 2
            adjusted_tcrw = 0.1 + adjusted_tp * 2.0

            # Soil temperatures
            adjusted_stl1_c = max(15.0 , min(40.0 , (latest_data['stl1'] - 273.15) + temp_adjustments[i] * 0.5))
            adjusted_stl2_c = max(15.0 , min(40.0 , (latest_data['stl2'] - 273.15) + temp_adjustments[i] * 0.5))

            # SSRD (adjusted base values, scaled by cloud cover)
            base_ssrd = ssrd_base[hour]
            # Enforce sunrise (06:00 IST) and sunset (18:34 IST)
            adjusted_ssrd = 0.0 if hour < 6 or hour >= 18.5 else max(0.0 ,
                                                                     base_ssrd * (1 - adjusted_tcc) + latest_ssrd * 0.3)

            # STRD (time-of-day factor + temperature and cloud cover)
            time_factor = np.sin(np.pi * (hour - 6) / 12.5) if 6 <= hour < 18.5 else 0.5
            adjusted_strd = max(200.0 ,
                                min(600.0 , 400.0 * time_factor + (adjusted_t2m_c - 25.0) * 10 - adjusted_tcc * 100))

            predicted_data.append({
                'valid_time': target_time ,
                'predicted_t2m': adjusted_t2m_c ,
                'predicted_sp': adjusted_sp_hpa ,
                'predicted_hcc': adjusted_hcc ,
                'predicted_lcc': adjusted_lcc ,
                'predicted_mcc': adjusted_mcc ,
                'predicted_tcrw': adjusted_tcrw ,
                'predicted_stl1': adjusted_stl1_c ,
                'predicted_stl2': adjusted_stl2_c ,
                'predicted_tp': adjusted_tp ,
                'predicted_d2m': adjusted_d2m_c ,
                'predicted_tcc': adjusted_tcc ,
                'predicted_ssrd': adjusted_ssrd ,
                'predicted_strd': adjusted_strd
            })

        predicted_df = pd.DataFrame(predicted_data)
        logger.info(f"Time-adjusted predictions for {target_times}:\n{predicted_df.to_string()}")
        return predicted_df
    except Exception as e:
        logger.error(f"Error in time-adjusted prediction: {str(e)}")
        return None


def format_and_save_output(predicted_df):
    if predicted_df is None or predicted_df.empty:
        logger.error("No prediction data to save.")
        return

    for idx , row in predicted_df.iterrows():
        summary = {
            'Time': row['valid_time'] ,
            'Temperature (°C)': row['predicted_t2m'] ,
            'Surface Pressure (hPa)': row['predicted_sp'] ,
            'High Cloud Cover (0-1)': row['predicted_hcc'] ,
            'Low Cloud Cover (0-1)': row['predicted_lcc'] ,
            'Mid Cloud Cover (0-1)': row['predicted_mcc'] ,
            'Total Column Rain Water (kg/m²)': row['predicted_tcrw'] ,
            'Soil Temperature Level 1 (°C)': row['predicted_stl1'] ,
            'Soil Temperature Level 2 (°C)': row['predicted_stl2'] ,
            'Precipitation (mm)': row['predicted_tp'] ,
            'Dew Point (°C)': row['predicted_d2m'] ,
            'Total Cloud Cover (0-1)': row['predicted_tcc'] ,
            'Surface Shortwave Radiation (W/m²)': row['predicted_ssrd'] ,
            'Surface Thermal Radiation (W/m²)': row['predicted_strd']
        }
        logger.info(f"Forecast Summary for {row['valid_time']}:\n{pd.Series(summary).to_string()}")

    output_path = os.path.join(OUTPUT_DIR , "real_time_forecast.csv")
    if not os.path.exists(output_path):
        predicted_df.to_csv(output_path , index=False)
    else:
        predicted_df.to_csv(output_path , mode='a' , index=False , header=False)

    logger.info(f"Saved forecast to {output_path}")


def predict_real_time(location):
    """Predict real-time weather for the given location."""
    logger.info(f"Starting real-time prediction for location: {location}")

    # Validate location (API is hardcoded for Bengaluru)
    if location.lower() != "bengaluru":
        raise ValueError("This API is configured for Bengaluru only. Please use location='Bengaluru'.")

    # Fetch and process live data
    api_data = fetch_live_weather_data()
    if api_data is None:
        raise ValueError("Failed to fetch live weather data from the API.")

    live_df = process_live_data(api_data)
    if live_df is None:
        raise ValueError("Failed to process live weather data.")

    # Make time-adjusted predictions
    predicted_df = make_time_adjusted_prediction(live_df)
    if predicted_df is None:
        raise ValueError("Failed to make time-adjusted predictions.")

    # Save the predictions
    format_and_save_output(predicted_df)

    # Select the prediction closest to the current time
    current_time = pd.Timestamp.now(tz='Asia/Kolkata')
    predicted_df['valid_time'] = pd.to_datetime(predicted_df['valid_time'])
    predicted_df['time_diff'] = (predicted_df['valid_time'] - current_time).abs()
    closest_prediction = predicted_df.loc[predicted_df['time_diff'].idxmin()]

    # Format the prediction as a dictionary
    prediction_dict = {
        'predicted_t2m': float(closest_prediction['predicted_t2m']) ,
        'predicted_sp': float(closest_prediction['predicted_sp']) ,
        'predicted_hcc': float(closest_prediction['predicted_hcc']) ,
        'predicted_lcc': float(closest_prediction['predicted_lcc']) ,
        'predicted_mcc': float(closest_prediction['predicted_mcc']) ,
        'predicted_tcrw': float(closest_prediction['predicted_tcrw']) ,
        'predicted_stl1': float(closest_prediction['predicted_stl1']) ,
        'predicted_stl2': float(closest_prediction['predicted_stl2']) ,
        'predicted_tp': float(closest_prediction['predicted_tp']) ,
        'predicted_d2m': float(closest_prediction['predicted_d2m']) ,
        'predicted_tcc': float(closest_prediction['predicted_tcc']) ,
        'predicted_ssrd': float(closest_prediction['predicted_ssrd']) ,
        'predicted_strd': float(closest_prediction['predicted_strd'])
    }

    logger.info(f"Real-time prediction completed: {prediction_dict}")
    return prediction_dict


if __name__ == "__main__":
    # Test the predict_real_time function
    try:
        location = "Bengaluru"
        predictions = predict_real_time(location)
        print("\nReal-Time Weather Predictions:")
        for key , value in predictions.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error during real-time prediction: {str(e)}")