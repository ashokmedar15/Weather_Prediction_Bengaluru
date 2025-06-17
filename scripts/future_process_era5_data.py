import os
import xarray as xr
import pandas as pd
import logging
import sys

# Set environment variable for CDSAPI_RC
os.environ['CDSAPI_RC'] = 'C:\\Users\\91905\\PycharmProjects\\WeatherPrediction_Bengaluru\\.cdsapirc'

# Set up logging
logging.basicConfig(
    filename='weather_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LIVE_DIR = os.path.join(DATA_DIR, "live")
DOWNLOAD_DIR = os.path.join(LIVE_DIR, "download")
MERGED_DIR = os.path.join(LIVE_DIR, "merged")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
LIVE_DATA_FILE = os.path.join(MERGED_DIR, "live_era5_data.nc")
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, "merged_weather_data.csv")

# Create directories if they don't exist
for dir_path in [MERGED_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Function to process NC files
def process_era5_data():
    try:
        logger.info("Starting data processing and merging...")

        # List all NC files in DOWNLOAD_DIR
        nc_files = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.nc')]
        if not nc_files:
            logger.error(f"No NetCDF files found in the download directory: {DOWNLOAD_DIR}")
            sys.exit(1)  # Exit with non-zero code to indicate failure

        logger.info(f"Found {len(nc_files)} NetCDF files: {nc_files}")

        # Merge all NC files
        datasets = []
        for nc_file in nc_files:
            ds = xr.open_dataset(nc_file, engine='netcdf4')
            datasets.append(ds)
            logger.info(f"Loaded dataset from {nc_file}: {list(ds.variables)}")

        merged_ds = xr.merge(datasets, compat='override')
        for ds in datasets:
            ds.close()
        logger.info(f"Merged dataset variables: {list(merged_ds.variables)}")

        # Define variable mapping from CDS API names to short names
        variable_mapping = {
            '2t': 't2m',
            'sp': 'sp',
            'hcc': 'hcc',
            'lcc': 'lcc',
            'mcc': 'mcc',
            'tcrw': 'tcrw',
            'stl1': 'stl1',
            'stl2': 'stl2',
            'tp': 'tp',
            'd2m': 'd2m',
            'tcc': 'tcc',
            'ssrd': 'ssrd',
            'strd': 'strd'
        }

        # Rename variables to match expected short names
        rename_dict = {}
        for cds_name, short_name in variable_mapping.items():
            if cds_name in merged_ds.variables:
                rename_dict[cds_name] = short_name
        if rename_dict:
            merged_ds = merged_ds.rename(rename_dict)
            logger.info(f"Renamed variables: {rename_dict}")

        # Verify required variables
        required_vars = ['t2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd', 'strd']
        missing_vars = [var for var in required_vars if var not in merged_ds.variables]
        if missing_vars:
            logger.error(f"Missing required variables in dataset: {missing_vars}")
            raise ValueError(f"Dataset does not contain required variables: {missing_vars}. Available variables: {list(merged_ds.variables)}")

        # Save the merged dataset
        merged_ds.to_netcdf(LIVE_DATA_FILE)
        logger.info(f"Final merged NetCDF file saved to {LIVE_DATA_FILE}")

        # Convert to DataFrame and save as CSV
        df = merged_ds.mean(dim=['latitude', 'longitude']).to_dataframe().reset_index()
        logger.info(f"Converted merged dataset to DataFrame. Columns: {df.columns}")

        # Ensure all required variables are in the DataFrame
        if not all(var in df.columns for var in required_vars):
            missing = [var for var in required_vars if var not in df.columns]
            logger.error(f"Missing variables in DataFrame: {missing}")
            raise ValueError(f"Missing variables in DataFrame: {missing}")

        # Rename time dimension if necessary
        time_dim = next((dim for dim in ['time', 'valid_time', 'datetime'] if dim in df.columns), None)
        if time_dim is None:
            logger.error(f"Time dimension not found in DataFrame. Available columns: {df.columns}")
            raise ValueError(f"Time dimension not found in DataFrame. Available columns: {df.columns}")
        df = df.rename(columns={time_dim: 'time'})

        # Select only the required columns
        df = df[['time'] + required_vars]
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        logger.info(f"Merged data saved to {OUTPUT_CSV_FILE}")
        if os.path.exists(OUTPUT_CSV_FILE):
            logger.info(f"Verified: {OUTPUT_CSV_FILE} exists.")
        else:
            logger.error(f"Failed to create {OUTPUT_CSV_FILE}")
            raise FileNotFoundError(f"Failed to create {OUTPUT_CSV_FILE}")

        logger.info("Data processing and merging completed successfully.")
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise

if __name__ == "__main__":
    process_era5_data()