import os
import cdsapi
import netCDF4 as nc
import logging
import time

# Set environment variable for CDSAPI_RC
os.environ['CDSAPI_RC'] = 'C:\\Users\\91905\\PycharmProjects\\WeatherPrediction_Bengaluru\\.cdsapirc'

# Set up logging
logging.basicConfig(
    filename='weather_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LIVE_DIR = os.path.join(DATA_DIR, "live")
DOWNLOAD_DIR = os.path.join(LIVE_DIR, "download")

# Create directories if they don't exist
for dir_path in [LIVE_DIR, DOWNLOAD_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Function to validate if a file is a valid NetCDF4 file
def is_valid_netcdf(file_path):
    try:
        with nc.Dataset(file_path, 'r'):
            return True
    except Exception:
        return False

# Function to download NC files
def download_era5_data():
    try:
        logger.info("Starting data download process...")

        # Clean up previous downloads
        logger.info("Cleaning up previous downloads...")
        if os.path.exists(DOWNLOAD_DIR):
            for file in os.listdir(DOWNLOAD_DIR):
                file_path = os.path.join(DOWNLOAD_DIR, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")

        # CDS API client
        c = cdsapi.Client()

        # Use the latest available date (2025-04-29)
        request_date = datetime(2025, 4, 29)

        # Define variable groups for chunked download
        variable_groups = [
            ['2m_temperature', 'surface_pressure', 'high_cloud_cover'],
            ['low_cloud_cover', 'medium_cloud_cover', 'total_column_rain_water'],
            ['soil_temperature_level_1', 'soil_temperature_level_2', 'total_precipitation'],
            ['2m_dewpoint_temperature', 'total_cloud_cover', 'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards']
        ]

        # Download each chunk as a NetCDF file
        max_retries = 5
        for i, variables in enumerate(variable_groups):
            temp_nc = os.path.join(DOWNLOAD_DIR, f"live_era5_data_part_{i}.nc")
            request = {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': str(request_date.year),
                'month': str(request_date.month).zfill(2),
                'day': str(request_date.day).zfill(2),
                'time': ['05:00', '10:00', '15:00', '20:00'],
                'area': [13.5, 77.0, 12.5, 78.0],  # North, West, South, East (Bengaluru region)
                'grid': '0.25/0.25',
            }

            # Try downloading the chunk
            for attempt in range(max_retries):
                try:
                    logger.info(f"Submitting CDS API request for part {i}, attempt {attempt + 1}: {request}")
                    c.retrieve('reanalysis-era5-single-levels', request, temp_nc)
                    if is_valid_netcdf(temp_nc):
                        logger.info(f"ERA5 data part {i} fetched and saved to {temp_nc}")
                        break
                    else:
                        file_size = os.path.getsize(temp_nc) if os.path.exists(temp_nc) else 0
                        logger.warning(f"Downloaded file {temp_nc} is not a valid NetCDF file on attempt {attempt + 1}. File size: {file_size} bytes")
                        with open(temp_nc, 'rb') as f:
                            first_bytes = f.read(100)
                            logger.warning(f"First 100 bytes of {temp_nc}: {first_bytes}")
                        os.remove(temp_nc)
                        if attempt == max_retries - 1:
                            logger.error(f"Failed to download a valid NetCDF file for part {i} after {max_retries} attempts. Trying individual variables...")
                            # Fallback: Try downloading each variable individually
                            for var in variables:
                                temp_var_nc = os.path.join(DOWNLOAD_DIR, f"live_era5_data_part_{i}_{var}.nc")
                                request['variable'] = [var]
                                for var_attempt in range(max_retries):
                                    try:
                                        logger.info(f"Submitting CDS API request for variable {var} (part {i}), attempt {var_attempt + 1}")
                                        c.retrieve('reanalysis-era5-single-levels', request, temp_var_nc)
                                        if is_valid_netcdf(temp_var_nc):
                                            logger.info(f"Variable {var} (part {i}) fetched and saved to {temp_var_nc}")
                                            break
                                        else:
                                            file_size = os.path.getsize(temp_var_nc) if os.path.exists(temp_var_nc) else 0
                                            logger.warning(f"Downloaded file {temp_var_nc} is not a valid NetCDF file on attempt {var_attempt + 1}. File size: {file_size} bytes")
                                            with open(temp_var_nc, 'rb') as f:
                                                first_bytes = f.read(100)
                                                logger.warning(f"First 100 bytes of {temp_var_nc}: {first_bytes}")
                                            os.remove(temp_var_nc)
                                            if var_attempt == max_retries - 1:
                                                raise ValueError(f"Failed to download variable {var} (part {i}) after {max_retries} attempts.")
                                            time.sleep(30)
                                    except Exception as e:
                                        logger.warning(f"Attempt {var_attempt + 1} failed for variable {var} (part {i}): {e}")
                                        if var_attempt == max_retries - 1:
                                            raise ValueError(f"Failed to download variable {var} (part {i}) after {max_retries} attempts: {str(e)}")
                                        time.sleep(30)
                            break
                        time.sleep(30)  # Increased delay to avoid rate limits
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for part {i}: {e}")
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to download a valid NetCDF file for part {i} after {max_retries} attempts: {str(e)}")
                    time.sleep(30)

        logger.info("Data download process completed successfully.")
    except Exception as e:
        logger.error(f"Error in data download: {str(e)}")
        raise

if __name__ == "__main__":
    from datetime import datetime
    download_era5_data()