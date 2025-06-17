import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime

# Define paths
BASE_DIR = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru"
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = PROCESSED_DATA_DIR
HISTORICAL_FILE = os.path.join(PROCESSED_DATA_DIR, "preprocessed_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "weather_model.keras")
SCALER_X_FILE = os.path.join(PROCESSED_DATA_DIR, "scaler_X.pkl")
SCALER_Y_FILE = os.path.join(PROCESSED_DATA_DIR, "scaler_y.pkl")

def preprocess_data_for_prediction(df):
    """Preprocess the data using the same steps as in the original preprocessing script."""
    print("Preprocessing data...")
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df['hour'] = df['valid_time'].dt.hour
    df['day'] = df['valid_time'].dt.day
    df['month'] = df['valid_time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    print("Preprocessed DataFrame:\n", df.head())
    return df

def create_sequences_for_prediction(df, features, seq_length=4):
    """Create sequences for prediction using the last `seq_length` time steps."""
    print("Creating sequences...")
    X = df[features].values
    Xs = []
    for i in range(len(X) - seq_length + 1):
        Xs.append(X[i:(i + seq_length)])
    sequences = np.array(Xs) if Xs else np.array([])
    print("Sequences shape:", sequences.shape)
    print("Sample sequence:\n", sequences[0] if sequences.size > 0 else "No sequences created")
    return sequences

def predict_historical(date, time, location, historical_file=HISTORICAL_FILE, model_file=MODEL_FILE, seq_length=4):
    """Predict weather variables for a given date and time using historical data."""
    print(f"\nStarting prediction for date: {date}, time: {time}, location: {location}")
    try:
        user_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        print("User datetime:", user_datetime)
    except ValueError:
        raise ValueError("Invalid date or time format. Use YYYY-MM-DD for date and HH:MM for time (24-hour format).")

    print("Loading historical data...")
    df = pd.read_csv(historical_file)
    df = preprocess_data_for_prediction(df)

    print("Loading scalers...")
    with open(SCALER_X_FILE, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y_FILE, 'rb') as f:
        scaler_y = pickle.load(f)
    print("Scaler X:", scaler_X)
    print("Scaler y:", scaler_y)

    features = ['t2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd',
                'strd', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
    targets = ['t2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd', 'strd']
    print("Features:", features)
    print("Targets:", targets)

    # Define expected physical ranges (same as preprocessing script)
    physical_ranges = {
        't2m': (293, 308),  # Kelvin
        'sp': (90000, 100000),  # Pascals
        'hcc': (0, 1),  # Fraction
        'lcc': (0, 1),
        'mcc': (0, 1),
        'tcrw': (0, 0.1),  # kg/m²
        'stl1': (293, 308),  # Kelvin
        'stl2': (293, 308),
        'tp': (0, 0.01),  # Meters
        'd2m': (283, 303),  # Kelvin
        'tcc': (0, 1),  # Fraction
        'ssrd': (0, 1e7),  # J/m²
        'strd': (0, 1e7)  # J/m²
    }

    df['valid_time'] = pd.to_datetime(df['valid_time'])
    target_time = user_datetime

    # Find the closest row to the target time for actual values
    df['time_diff'] = (df['valid_time'] - target_time).abs()
    closest_row = df.loc[df['time_diff'].idxmin()]
    print("Closest row to target time:\n", closest_row)

    # Improved past data selection: Select rows before target time, prioritizing temporal diversity
    past_data = df[df['valid_time'] < target_time].copy()
    if len(past_data) < seq_length:
        raise ValueError(
            f"Not enough historical data for prediction. Need {seq_length} time steps, found {len(past_data)}.")

    # Sort by time difference and select rows to ensure diversity (e.g., same time of day on previous days)
    past_data['time_diff'] = (past_data['valid_time'] - target_time).abs()
    past_data = past_data.sort_values('time_diff')

    # Prefer rows with the same hour but from different days to capture daily variations
    target_hour = user_datetime.hour
    same_hour_data = past_data[past_data['hour'] == target_hour].tail(seq_length)
    if len(same_hour_data) >= seq_length:
        past_data = same_hour_data
    else:
        # If not enough same-hour data, fall back to the closest rows
        past_data = past_data.head(seq_length)

    print("Past data for sequence (after improved selection):\n", past_data[['valid_time', 'hour', 'day', 'month'] + targets])

    input_data = create_sequences_for_prediction(past_data, features, seq_length)
    if input_data.size == 0:
        raise ValueError("Failed to create input sequences.")

    original_shape = input_data.shape
    print("Original input data shape before scaling:", original_shape)
    input_data_2d = input_data.reshape(-1, len(features))
    print("Input data reshaped for scaling:", input_data_2d.shape)
    print("Sample input data (2D):\n", input_data_2d)

    input_data_scaled = scaler_X.transform(input_data_2d)
    print("Scaled input data (2D):\n", input_data_scaled)

    input_data_scaled = input_data_scaled.reshape(original_shape)
    print("Scaled input data reshaped back to 3D:", input_data_scaled.shape)
    print("Sample scaled input data (3D):\n", input_data_scaled)

    print("Loading model...")
    if not os.path.exists(model_file):
        raise ValueError(
            f"File not found: filepath={model_file}. Please ensure the file is an accessible `.keras` zip file.")
    model = load_model(model_file)
    print("Model summary:")
    model.summary()

    print("Making prediction...")
    prediction = model.predict(input_data_scaled)
    print("Raw prediction shape:", prediction.shape)
    print("Raw prediction:\n", prediction)

    # Fix indexing: Use [0, :] for 2D array (batch_size, num_targets)
    prediction = prediction[0, :]  # Shape: (13,)
    print("Prediction (first step):", prediction)

    # Inverse transform to normalized units [0, 1]
    prediction = scaler_y.inverse_transform([prediction])[0]
    print("Prediction after inverse transform (normalized units):", prediction)

    # Map normalized values to physical units and convert to standard units
    prediction_dict = {}
    for i, feat in enumerate(targets):
        norm_value = prediction[i]
        norm_value = max(0, min(1, norm_value))  # Clip to [0, 1]
        min_val, max_val = physical_ranges[feat]
        value = min_val + norm_value * (max_val - min_val)
        value = max(min_val, min(max_val, value))  # Ensure within range

        # Convert to standard units
        if feat in ['t2m', 'stl1', 'stl2', 'd2m']:
            value_std = value - 273.15  # Kelvin to Celsius
        elif feat == 'sp':
            value_std = value / 100  # Pascals to hPa
        elif feat == 'tp':
            value_std = value * 1000  # Meters to mm
        elif feat in ['ssrd', 'strd']:
            value_std = value / 1000000  # J/m² to MJ/m²
        else:
            value_std = value  # No conversion for fractions (hcc, lcc, mcc, tcc, tcrw)

        prediction_dict[f"predicted_{feat}"] = value_std

    # Enforce nighttime rule for ssrd and strd
    target_hour = user_datetime.hour
    if target_hour in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]:
        prediction_dict['predicted_ssrd'] = 0
        prediction_dict['predicted_strd'] = 0

    print("Prediction dictionary (standard units):", prediction_dict)

    # Map actual normalized values to physical units and convert to standard units
    actual_values = scaler_y.inverse_transform([closest_row[targets].values])[0]
    print("Actual values after inverse transform (normalized units):", actual_values)

    actual_physical = []
    for i, feat in enumerate(targets):
        norm_value = actual_values[i]
        norm_value = max(0, min(1, norm_value))
        min_val, max_val = physical_ranges[feat]
        value = min_val + norm_value * (max_val - min_val)
        value = max(min_val, min(max_val, value))

        # Convert to standard units
        if feat in ['t2m', 'stl1', 'stl2', 'd2m']:
            value_std = value - 273.15  # Kelvin to Celsius
        elif feat == 'sp':
            value_std = value / 100  # Pascals to hPa
        elif feat == 'tp':
            value_std = value * 1000  # Meters to mm
        elif feat in ['ssrd', 'strd']:
            value_std = value / 1000000  # J/m² to MJ/m²
        else:
            value_std = value  # No conversion for fractions

        actual_physical.append(value_std)

    # Enforce nighttime rule for actual ssrd and strd
    if target_hour in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]:
        actual_physical[targets.index('ssrd')] = 0
        actual_physical[targets.index('strd')] = 0

    print("Actual values in standard units:", actual_physical)

    # Calculate normalized MAE for accuracy using original physical units
    predicted_physical = []
    for i, feat in enumerate(targets):
        norm_value = prediction[i]
        norm_value = max(0, min(1, norm_value))
        min_val, max_val = physical_ranges[feat]
        value = min_val + norm_value * (max_val - min_val)
        value = max(min_val, min(max_val, value))
        predicted_physical.append(value)

    actual_physical_orig = []
    for i, feat in enumerate(targets):
        norm_value = actual_values[i]
        norm_value = max(0, min(1, norm_value))
        min_val, max_val = physical_ranges[feat]
        value = min_val + norm_value * (max_val - min_val)
        value = max(min_val, min(max_val, value))
        actual_physical_orig.append(value)

    # Compute errors in physical units
    errors = []
    for i, feat in enumerate(targets):
        min_val, max_val = physical_ranges[feat]
        range_val = max_val - min_val
        if range_val == 0:
            continue
        error = abs(predicted_physical[i] - actual_physical_orig[i]) / range_val
        errors.append(error)

    if errors:
        mean_normalized_error = np.mean(errors)
        accuracy = max(0, 100 * (1 - mean_normalized_error))
    else:
        accuracy = 0

    print("Mean normalized error:", mean_normalized_error)
    print("Accuracy (%):", accuracy)

    # Convert prediction_dict values to native Python floats for clean printing
    formatted_predictions = {key: float(value) if isinstance(value, (np.floating, np.integer)) else value
                            for key, value in prediction_dict.items()}

    return formatted_predictions, accuracy

if __name__ == "__main__":
    try:
        # Prompt user for date, time, and location
        print("Enter the date for prediction (format: YYYY-MM-DD, e.g., 2025-04-30):")
        date = input().strip()

        print("Enter the time for prediction (format: HH:MM in 24-hour format, e.g., 05:00):")
        time = input().strip()

        print("Enter the location for prediction (e.g., Bengaluru):")
        location = input().strip()

        # Basic validation for date and time format
        try:
            datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(
                "Invalid date or time format. Use YYYY-MM-DD for date and HH:MM for time (24-hour format).")

        # Validate location (basic check to ensure it's not empty)
        if not location:
            raise ValueError("Location cannot be empty.")

        predictions, accuracy = predict_historical(date, time, location)
        print("\nFinal Predictions:", predictions)
        print("Final Accuracy:", accuracy)
    except ValueError as e:
        print("Error:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)