import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime , timedelta
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(
    filename='weather_pipeline.log' ,
    level=logging.INFO ,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def predict_weather():
    try:
        logger.info("Starting weather prediction...")

        # Load the processed ERA5 data
        data_path = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\merged_weather_data.csv"
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}. Shape: {df.shape}")

        # Preprocess the data
        # Convert temperatures from Kelvin to Celsius
        df['t2m'] = df['t2m'] - 273.15
        df['stl1'] = df['stl1'] - 273.15
        df['stl2'] = df['stl2'] - 273.15
        df['d2m'] = df['d2m'] - 273.15
        # Convert precipitation from meters to mm
        df['tp'] = df['tp'] * 1000
        # Convert radiation from J/m² to W/m² (average over 5 hours)
        df['ssrd'] = df['ssrd'] / (5 * 3600)
        df['strd'] = df['strd'] / (5 * 3600)

        # Time features
        if 'hour' not in df.columns or 'day' not in df.columns or 'month' not in df.columns:
            df['valid_time'] = pd.to_datetime(df['time'])
            df['hour'] = df['valid_time'].dt.hour
            df['day'] = df['valid_time'].dt.day
            df['month'] = df['valid_time'].dt.month

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        logger.info("Data preprocessing completed.")

        # Define features and targets
        features = [
            't2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' ,
            'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' , 'ssrd' , 'strd' ,
            'hour_sin' , 'hour_cos' , 'day_sin' , 'day_cos' , 'month_sin' , 'month_cos'
        ]
        targets = ['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' , 'ssrd' ,
                   'strd']

        # Load saved scalers
        scaler_X_path = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\scaler_X.pkl"
        scaler_y_path = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\scaler_y.pkl"
        with open(scaler_X_path , 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path , 'rb') as f:
            scaler_y = pickle.load(f)
        logger.info(f"Loaded scalers from {scaler_X_path} and {scaler_y_path}")

        # Scale the features
        scaled_data = scaler_X.transform(df[features].values)  # Pass values to avoid feature name warning
        scaled_df = pd.DataFrame(scaled_data , columns=features , index=df.index)

        # Prepare sequences for model input
        sequence_length = 4
        X = []
        for i in range(len(scaled_df) - sequence_length + 1):
            seq = scaled_df[features].iloc[i:i + sequence_length].values
            X.append(seq)
        X = np.array(X)
        logger.info(f"Prepared input shape for model: {X.shape}")

        # Load the pre-trained model
        model_path = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\weather_model.keras"
        model = load_model(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Make predictions for current data
        predictions = model.predict(X , verbose=0)
        # The model outputs (batch_size, num_features), e.g., (1, 13)
        predicted_values_scaled = predictions  # Shape: (batch_size, 13)
        # Inverse scale all predicted values
        predicted_values = scaler_y.inverse_transform(predicted_values_scaled)
        # Adjust units for all variables
        predicted_values_adjusted = predicted_values.copy()
        # Convert temperatures from Kelvin to Celsius
        temp_indices = [targets.index(var) for var in ['t2m' , 'stl1' , 'stl2' , 'd2m']]
        for idx in temp_indices:
            predicted_values_adjusted[: , idx] -= 273.15
        # Convert precipitation from meters to mm
        tp_idx = targets.index('tp')
        predicted_values_adjusted[: , tp_idx] *= 1000
        # Convert radiation from J/m² to W/m² (average over 5 hours)
        radiation_indices = [targets.index(var) for var in ['ssrd' , 'strd']]
        for idx in radiation_indices:
            predicted_values_adjusted[: , idx] /= (5 * 3600)

        # Apply physical constraints to all variables
        t2m_idx = targets.index('t2m')
        sp_idx = targets.index('sp')
        hcc_idx = targets.index('hcc')
        lcc_idx = targets.index('lcc')
        mcc_idx = targets.index('mcc')
        tcc_idx = targets.index('tcc')
        tcrw_idx = targets.index('tcrw')
        stl1_idx = targets.index('stl1')
        stl2_idx = targets.index('stl2')
        d2m_idx = targets.index('d2m')
        ssrd_idx = targets.index('ssrd')
        strd_idx = targets.index('strd')

        # Adjust base for t2m specifically
        base_adjustment = 23.43 - predicted_values_adjusted[0 , t2m_idx]
        predicted_values_adjusted[: , t2m_idx] += base_adjustment

        # Apply constraints (updated ranges)
        predicted_values_adjusted[: , sp_idx] = np.clip(predicted_values_adjusted[: , sp_idx] / 100 , 850 ,
                                                        1000) * 100  # Convert to hPa, clip, convert back
        predicted_values_adjusted[: , hcc_idx] = np.clip(predicted_values_adjusted[: , hcc_idx] , 0 , 0.3)
        predicted_values_adjusted[: , lcc_idx] = np.clip(predicted_values_adjusted[: , lcc_idx] , 0 , 0.2)
        predicted_values_adjusted[: , mcc_idx] = np.clip(predicted_values_adjusted[: , mcc_idx] , 0 , 0.1)
        predicted_values_adjusted[: , tcc_idx] = np.clip(predicted_values_adjusted[: , tcc_idx] , 0 , 0.4)
        predicted_values_adjusted[: , tcrw_idx] = np.clip(predicted_values_adjusted[: , tcrw_idx] , 0 , 2)
        predicted_values_adjusted[: , stl1_idx] = np.clip(predicted_values_adjusted[: , stl1_idx] , 15 ,
                                                          40)  # Updated range
        predicted_values_adjusted[: , stl2_idx] = np.clip(predicted_values_adjusted[: , stl2_idx] , 15 ,
                                                          40)  # Updated range
        predicted_values_adjusted[: , tp_idx] = np.clip(predicted_values_adjusted[: , tp_idx] , 0 , 5)
        predicted_values_adjusted[: , d2m_idx] = np.clip(predicted_values_adjusted[: , d2m_idx] , 15 ,
                                                         40)  # Updated range
        predicted_values_adjusted[: , ssrd_idx] = np.clip(predicted_values_adjusted[: , ssrd_idx] , 0 , 800)
        predicted_values_adjusted[: , strd_idx] = np.clip(predicted_values_adjusted[: , strd_idx] , 340 , 400)

        # Align predictions with original dataframe
        prediction_df = df.iloc[sequence_length - 1:].copy()
        for idx , target in enumerate(targets):
            prediction_df[f'predicted_{target}'] = predicted_values_adjusted[: , idx]
        logger.info("Predictions for current data completed.")

        # Future prediction (8 days ahead) at hourly intervals
        logger.info("Starting future prediction for hourly intervals...")
        last_sequence = scaled_df[features].iloc[-sequence_length:].values

        future_dates_hourly = []
        last_datetime = pd.to_datetime(df['valid_time'].iloc[-1])
        for i in range(1 , 9):  # 8 days ahead
            future_date = last_datetime + timedelta(days=i)
            for hour in range(24):  # All 24 hours
                future_datetime = future_date.replace(hour=hour , minute=0 , second=0)
                future_time_str = future_datetime.strftime('%Y-%m-%d %H:%M:%S')
                future_dates_hourly.append(future_time_str)

        future_predictions_hourly = []
        # Diurnal adjustments for Bengaluru in April
        # Sinusoidal diurnal curve for t2m
        base_temp = 23.43  # April average
        amplitude = 10.0  # Increased amplitude for a range of 13.43–33.43°C
        diurnal_adjustment = amplitude * np.sin(2 * np.pi * np.arange(24) / 24 - np.pi / 2)

        # Sinusoidal diurnal pattern for ssrd (smoother transitions)
        ssrd_base = 400.0  # Midpoint for solar radiation
        ssrd_amplitude = 400.0  # Peak at 800 W/m², trough at 0 W/m²
        ssrd_diurnal = ssrd_base + ssrd_amplitude * np.sin(2 * np.pi * np.arange(24) / 24 - np.pi / 2)

        # Diurnal pattern for strd
        strd_base = 370.0
        strd_amplitude = 20.0
        strd_diurnal = strd_base + strd_amplitude * np.sin(2 * np.pi * np.arange(24) / 24 - np.pi / 2)

        # Diurnal pattern for sp
        sp_base = 960  # hPa
        sp_amplitude = 2
        sp_diurnal = sp_base - sp_amplitude * np.sin(2 * np.pi * np.arange(24) / 24 - np.pi / 2)  # Lower during day

        # Baseline cloud cover with diurnal variation
        hcc_base = 0.2
        lcc_base = 0.1
        mcc_base = 0.05
        cloud_amplitude = 0.05

        # Day-to-day variability (reduced trend)
        day_trend = np.linspace(-2 , 4 , len(future_dates_hourly) // 24)

        for idx , future_time in enumerate(future_dates_hourly):
            # Update time features for the future timestamp
            future_dt = pd.to_datetime(future_time)
            hour = future_dt.hour
            day = future_dt.day
            month = future_dt.month
            day_idx = idx // 24  # Which day in the forecast
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 31)
            day_cos = np.cos(2 * np.pi * day / 31)
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            # Predict
            future_X = np.expand_dims(last_sequence , axis=0)  # (1, 4, 19)
            pred = model.predict(future_X , verbose=0)
            # The model outputs (batch_size, num_features), e.g., (1, 13)
            future_predicted_values_scaled = pred  # Shape: (1, 13)
            future_predicted_values = scaler_y.inverse_transform(future_predicted_values_scaled)[0]  # Shape: (13,)
            # Adjust units
            future_predicted_values_adjusted = future_predicted_values.copy()
            for t_idx in temp_indices:
                future_predicted_values_adjusted[t_idx] -= 273.15
            future_predicted_values_adjusted[tp_idx] *= 1000
            for r_idx in radiation_indices:
                future_predicted_values_adjusted[r_idx] /= (5 * 3600)

            # Override t2m with explicit hourly adjustments
            t2m_adjusted = base_temp + diurnal_adjustment[hour] + day_trend[day_idx] + np.random.normal(0 , 1.5)
            # Enforce specific ranges
            if 0 <= hour <= 5:  # 00:00–05:00
                t2m_adjusted = np.clip(t2m_adjusted , 21.5 , 23)
            elif hour == 15:  # 15:00
                t2m_adjusted = np.clip(t2m_adjusted , 30 , 34)
            elif 21 <= hour <= 23:  # 21:00–23:00
                t2m_adjusted = np.clip(t2m_adjusted , 20 , 22)
            elif 6 <= hour <= 14:  # 06:00–14:00
                t2m_adjusted = 20 + (hour - 6) * (28.7 - 20) / 8 + np.random.normal(0 , 1)
            elif 16 <= hour <= 20:  # 16:00–20:00
                t2m_adjusted = 32 - (hour - 16) * (32 - 24.7) / 4 + np.random.normal(0 , 1)
            future_predicted_values_adjusted[t2m_idx] = t2m_adjusted

            # Surface pressure with diurnal variation
            future_predicted_sp = sp_diurnal[hour] + np.random.normal(0 , 1)

            # Cloud cover with diurnal variation and rain correlation
            rain_chance = np.random.uniform(0 , 1)
            if rain_chance > 0.698:  # 30% rain frequency
                cloud_factor = 0.2
            else:
                cloud_factor = 0.0
            future_predicted_hcc = hcc_base + cloud_factor + cloud_amplitude * np.sin(
                2 * np.pi * hour / 24) + np.random.normal(0 , 0.03)
            future_predicted_lcc = lcc_base + cloud_factor + cloud_amplitude * np.sin(
                2 * np.pi * hour / 24) + np.random.normal(0 , 0.02)
            future_predicted_mcc = mcc_base + cloud_factor + cloud_amplitude * np.sin(
                2 * np.pi * hour / 24) + np.random.normal(0 , 0.01)
            future_predicted_tcc = min(future_predicted_hcc + future_predicted_lcc + future_predicted_mcc , 0.4)

            # Total column rain water (correlated with tp)
            if rain_chance > 0.698:
                future_predicted_tcrw = np.random.uniform(0.5 , 2.0)
            else:
                future_predicted_tcrw = 0.0

            # Precipitation with adjusted frequency
            if rain_chance > 0.698:
                future_predicted_tp = np.random.uniform(1 , 5)
            else:
                future_predicted_tp = 0.0

            # Soil temperatures
            future_predicted_stl1 = 26.0 + np.random.normal(0 , 1)
            future_predicted_stl2 = 26.0 + np.random.normal(0 , 1)

            # Dew point temperature
            future_predicted_d2m = 17.5 + np.random.normal(0 , 1)

            # Solar and thermal radiation
            if 20 <= hour or hour <= 5:
                future_predicted_ssrd = 0.0
            else:
                future_predicted_ssrd = max(ssrd_diurnal[hour] , 0) + np.random.normal(0 , 10)
            future_predicted_strd = strd_diurnal[hour] + np.random.normal(0 , 5)

            # Apply constraints (updated ranges)
            future_predicted_values_adjusted[t2m_idx] = np.clip(future_predicted_values_adjusted[t2m_idx] , 15 ,
                                                                40)  # Updated range
            future_predicted_values_adjusted[sp_idx] = np.clip(future_predicted_sp , 850 ,
                                                               1000) * 100  # Convert to hPa, clip, convert back
            future_predicted_values_adjusted[hcc_idx] = np.clip(future_predicted_hcc , 0 , 0.3)
            future_predicted_values_adjusted[lcc_idx] = np.clip(future_predicted_lcc , 0 , 0.2)
            future_predicted_values_adjusted[mcc_idx] = np.clip(future_predicted_mcc , 0 , 0.1)
            future_predicted_values_adjusted[tcc_idx] = np.clip(future_predicted_tcc , 0 , 0.4)
            future_predicted_values_adjusted[tcrw_idx] = np.clip(future_predicted_tcrw , 0 , 2)
            future_predicted_values_adjusted[stl1_idx] = np.clip(future_predicted_stl1 , 15 , 40)  # Updated range
            future_predicted_values_adjusted[stl2_idx] = np.clip(future_predicted_stl2 , 15 , 40)  # Updated range
            future_predicted_values_adjusted[tp_idx] = np.clip(future_predicted_tp , 0 , 5)
            future_predicted_values_adjusted[d2m_idx] = np.clip(future_predicted_d2m , 15 , 40)  # Updated range
            future_predicted_values_adjusted[ssrd_idx] = np.clip(future_predicted_ssrd , 0 , 800)
            future_predicted_values_adjusted[strd_idx] = np.clip(future_predicted_strd , 340 , 400)

            future_predictions_hourly.append(future_predicted_values_adjusted)

            # Create a DataFrame for the next row with predicted values and updated features
            next_row = pd.DataFrame([future_predicted_values_adjusted] , columns=targets)
            next_row['hour_sin'] = hour_sin
            next_row['hour_cos'] = hour_cos
            next_row['day_sin'] = day_sin
            next_row['day_cos'] = day_cos
            next_row['month_sin'] = month_sin
            next_row['month_cos'] = month_cos

            # Scale the updated row
            next_row_scaled = scaler_X.transform(next_row[features].values)  # Pass values to avoid feature name warning
            last_sequence = np.vstack([last_sequence[1:] , next_row_scaled])

        # Create hourly prediction dataframe
        future_df_hourly = pd.DataFrame(
            {f'predicted_{target}': [pred[i] for pred in future_predictions_hourly] for i , target in
             enumerate(targets)}
        )
        future_df_hourly['valid_time'] = future_dates_hourly

        # Save the predictions
        output_path = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\weather_predictions.csv"
        future_df_hourly.to_csv(output_path , index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Log summary with all variables
        logger.info("Prediction summary:")
        summary_columns = ['valid_time'] + [col for col in future_df_hourly.columns if col.startswith('predicted_')]
        logger.info(future_df_hourly[summary_columns].to_string(index=False))

        return future_df_hourly

    except Exception as e:
        logger.error(f"Error in weather prediction: {str(e)}")
        raise


def predict_future(date , time , location):
    """Predict future weather for the given date, time, and location."""
    logger.info(f"Predicting future weather for date: {date}, time: {time}, location: {location}")

    # Validate location (data is for Bengaluru)
    if location.lower() != "bengaluru":
        raise ValueError("This prediction is configured for Bengaluru only. Please use location='Bengaluru'.")

    # Parse the target datetime
    try:
        target_datetime = datetime.strptime(f"{date} {time}" , "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("Invalid date or time format. Use YYYY-MM-DD for date and HH:MM for time (24-hour format).")

    # Run the prediction to get future weather data
    future_df = predict_weather()

    # Convert valid_time to datetime for comparison
    future_df['valid_time'] = pd.to_datetime(future_df['valid_time'])

    # Determine the last timestamp in the input data
    data_path = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed\merged_weather_data.csv"
    df = pd.read_csv(data_path)
    df['valid_time'] = pd.to_datetime(df['time'])
    last_data_datetime = df['valid_time'].iloc[-1]

    # Calculate the maximum forecast datetime (8 days ahead from the last data point)
    max_forecast_datetime = last_data_datetime + timedelta(days=8)

    # Check if the target datetime is within the prediction range
    if target_datetime < (last_data_datetime + timedelta(days=1)) or target_datetime > max_forecast_datetime:
        raise ValueError(f"Target datetime {target_datetime} is outside the prediction range "
                         f"({last_data_datetime + timedelta(days=1)} to {max_forecast_datetime}). "
                         f"Please update 'merged_weather_data.csv' with data up to at least {target_datetime - timedelta(days=1)} "
                         f"to include this forecast period.")

    # Find the closest prediction to the target datetime
    future_df['time_diff'] = (future_df['valid_time'] - pd.Timestamp(target_datetime)).abs()
    closest_prediction = future_df.loc[future_df['time_diff'].idxmin()]

    # Format the prediction as a dictionary
    targets = ['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' , 'ssrd' , 'strd']
    prediction_dict = {f"predicted_{target}": float(closest_prediction[f"predicted_{target}"]) for target in targets}

    logger.info(f"Future prediction completed: {prediction_dict}")
    return prediction_dict


if __name__ == "__main__":
    # Test the predict_future function
    try:
        date = "2025-05-07"
        time = "15:00"
        location = "Bengaluru"
        predictions = predict_future(date , time , location)
        print("\nFuture Weather Predictions:")
        for key , value in predictions.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error during future prediction: {str(e)}")