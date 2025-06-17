import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pickle
import xarray as xr
import zipfile
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO , format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure required packages are installed
try:
    import xarray
except ImportError:
    raise ImportError("Please install xarray: pip install xarray")
try:
    import netCDF4
except ImportError:
    raise ImportError("Please install netCDF4: pip install netCDF4")

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR , "data")
OUTPUT_DIR = os.path.join(DATA_DIR , "processed")
LIVE_DATA_DIR = os.path.join(DATA_DIR , "live")
MODEL_FILE = os.path.join(OUTPUT_DIR , "weather_model.keras")
SCALER_X_FILE = os.path.join(OUTPUT_DIR , "scaler_X.pkl")
SCALER_Y_FILE = os.path.join(OUTPUT_DIR , "scaler_y.pkl")
OUTPUT_PRED_FILE = os.path.join(OUTPUT_DIR , "realtime_predictions.csv")
LIVE_DATA_ZIP = os.path.join(LIVE_DATA_DIR , "live_era5_data.zip")
LIVE_DATA_FILE = os.path.join(LIVE_DATA_DIR , "live_era5_data.nc")

# Create directories if they don't exist
os.makedirs(LIVE_DATA_DIR , exist_ok=True)
os.makedirs(OUTPUT_DIR , exist_ok=True)

# Check if required files exist
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file {MODEL_FILE} not found. Please run train_model.py first.")
if not os.path.exists(SCALER_X_FILE) or not os.path.exists(SCALER_Y_FILE):
    raise FileNotFoundError(f"Scaler files not found in {OUTPUT_DIR}. Please run train_model.py first.")
if not os.path.exists(LIVE_DATA_ZIP):
    raise FileNotFoundError(f"ZIP file {LIVE_DATA_ZIP} not found. Please run download_era5.py first.")

# Load model and scalers
try:
    model = load_model(MODEL_FILE)
    logger.info(f"Model loaded from {MODEL_FILE}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError(f"Error loading model: {e}")

try:
    with open(SCALER_X_FILE , 'rb') as f:
        scaler_X = pickle.load(f)
    with open(SCALER_Y_FILE , 'rb') as f:
        scaler_y = pickle.load(f)
    logger.info("Scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading scalers: {e}")
    raise FileNotFoundError(f"Error loading scalers: {e}")


# Function to safely delete a file with retry logic
def safe_delete(file_path , max_retries=3 , delay_seconds=1):
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to delete {file_path}: {e}. Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"Failed to delete {file_path} after {max_retries} attempts: {e}")
                raise


# Function to extract and merge NetCDF files from ZIP
def extract_and_merge_zip(zip_path):
    try:
        # Clean up any existing NetCDF files
        for nc_file in os.listdir(LIVE_DATA_DIR):
            if nc_file.endswith('.nc'):
                safe_delete(os.path.join(LIVE_DATA_DIR , nc_file))

        # Extract the ZIP file
        instant_file = None
        accum_file = None
        with zipfile.ZipFile(zip_path , 'r') as zip_ref:
            nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
            if not nc_files:
                raise ValueError("No .nc file found in the downloaded ZIP archive.")
            logger.info(f"Found .nc files in ZIP: {nc_files}")
            for nc_file in nc_files:
                zip_ref.extract(nc_file , LIVE_DATA_DIR)
                extracted_path = os.path.join(LIVE_DATA_DIR , nc_file)
                if 'instant' in nc_file:
                    instant_file = extracted_path
                elif 'accum' in nc_file:
                    accum_file = extracted_path

        if not instant_file or not accum_file:
            raise ValueError("Did not find both instant and accum NetCDF files in ZIP.")

        # Load the two NetCDF files to inspect their dimensions
        with xr.open_dataset(instant_file , engine='netcdf4') as ds_instant , \
                xr.open_dataset(accum_file , engine='netcdf4') as ds_accum:
            logger.info(f"Instant dataset dimensions: {ds_instant.dims}")
            logger.info(f"Accum dataset dimensions: {ds_accum.dims}")
            # Merge datasets with outer join to preserve all time steps
            ds = xr.merge([ds_instant , ds_accum] , join='outer')
            logger.info("Merged instant and accum NetCDF datasets")
            # Log dataset structure for debugging
            logger.info(f"Dataset dimensions: {ds.dims}")
            logger.info(f"Dataset coordinates: {list(ds.coords)}")
            logger.info(f"Dataset variables: {list(ds.variables)}")
            # Drop any rows with missing data for critical variables
            ds = ds.dropna(dim='valid_time' , subset=['t2m' , 'tp'])
            logger.info(f"Dataset dimensions after dropping NA: {ds.dims}")
            ds.to_netcdf(LIVE_DATA_FILE)
            logger.info(f"Saved merged NetCDF file to {LIVE_DATA_FILE}")

        # Clean up temporary files
        safe_delete(instant_file)
        safe_delete(accum_file)
        safe_delete(zip_path)
        logger.info("Cleaned up temporary files")

        return LIVE_DATA_FILE
    except Exception as e:
        logger.error(f"Error extracting and merging ZIP: {e}")
        raise


# Function to preprocess live data
def preprocess_live_data(file_path):
    try:
        # Load NetCDF file with explicit engine
        with xr.open_dataset(file_path , engine='netcdf4') as ds:
            logger.info(f"NetCDF file loaded: {file_path}")

            # Log dataset structure for debugging
            logger.info(f"Dataset dimensions: {ds.dims}")
            logger.info(f"Dataset coordinates: {list(ds.coords)}")
            logger.info(f"Dataset variables: {list(ds.variables)}")

            # Convert to DataFrame and average over latitude/longitude
            df = ds.mean(dim=['latitude' , 'longitude']).to_dataframe().reset_index()

            # Ensure required variables exist
            required_vars = ['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' ,
                             'ssrd' , 'strd']
            if not all(var in df.columns for var in required_vars):
                missing = [var for var in required_vars if var not in df.columns]
                raise ValueError(f"Missing variables in NetCDF: {missing}")

            # Find the time dimension
            possible_time_dims = ['time' , 'valid_time' , 'datetime']
            time_dim = None
            for dim in possible_time_dims:
                if dim in df.columns:
                    time_dim = dim
                    break
            if time_dim is None:
                raise ValueError(f"Time dimension not found in DataFrame. Available columns: {df.columns}")

            logger.info(f"Using time dimension: {time_dim}")

            # Check the number of time steps
            num_time_steps = len(df)
            required_time_steps = 4
            if num_time_steps < required_time_steps:
                logger.warning(f"Only {num_time_steps} time steps available, but {required_time_steps} are required.")
                # Pad the data by repeating the last time step
                last_row = df.iloc[-1:].copy()
                while len(df) < required_time_steps:
                    df = pd.concat([df , last_row] , ignore_index=True)
                    logger.info(f"Padded data to {len(df)} time steps by repeating the last time step.")

            # Add temporal features
            df['valid_time'] = pd.to_datetime(df[time_dim])
            df['hour'] = df['valid_time'].dt.hour
            df['day'] = df['valid_time'].dt.day
            df['month'] = df['valid_time'].dt.month
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            # Select features in the correct order
            features = ['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' ,
                        'ssrd' , 'strd' ,
                        'hour_sin' , 'hour_cos' , 'day_sin' , 'day_cos' , 'month_sin' , 'month_cos']
            X = df[features]

            # Normalize using the same scaler as training
            X_scaled = scaler_X.transform(X)
            logger.info("Live data preprocessed successfully")
            return X_scaled[-4:]  # Last 4 time steps for prediction
    except Exception as e:
        logger.error(f"Error preprocessing live data: {e}")
        raise RuntimeError(f"Error preprocessing live data: {e}")


# Function to create prediction sequence
def predict_next_24h(model , X_recent):
    try:
        X_seq = np.array([X_recent])
        predictions = model.predict(X_seq , verbose=0)
        predictions = scaler_y.inverse_transform(predictions[0])
        logger.info("Predictions generated successfully")
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise RuntimeError(f"Error during prediction: {e}")


# Main execution
if __name__ == "__main__":
    try:
        logger.info("Step 1: Extracting and merging ZIP file...")
        file_path = extract_and_merge_zip(LIVE_DATA_ZIP)

        logger.info("Step 2: Preprocessing live data...")
        X_recent = preprocess_live_data(file_path)

        logger.info("Step 3: Predicting next 24 hours...")
        predictions = predict_next_24h(model , X_recent)

        logger.info("Step 4: Saving predictions...")
        # Create timestamps for predictions (next 24 hours, 6-hour intervals)
        current_time = pd.Timestamp.now()
        prediction_times = [current_time + pd.Timedelta(hours=i * 6) for i in range(4)]

        # Save predictions to CSV
        pred_df = pd.DataFrame(
            predictions ,
            columns=['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' , 'ssrd' ,
                     'strd'] ,
            index=prediction_times
        )
        pred_df.to_csv(OUTPUT_PRED_FILE)
        logger.info(f"Predictions saved to {OUTPUT_PRED_FILE}")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        exit(1)