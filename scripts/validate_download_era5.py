import os
from cdsapi import Client
from datetime import datetime, timedelta
import time
import logging
import xarray as xr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure required packages are installed
try:
    import cdsapi
except ImportError:
    raise ImportError("Please install cdsapi: pip install cdsapi")
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
DATA_DIR = os.path.join(BASE_DIR, "data")
LIVE_DATA_DIR = os.path.join(DATA_DIR, "live")
LIVE_DATA_ZIP = os.path.join(LIVE_DATA_DIR, "live_era5_data.zip")
CDSAPI_RC_FILE = os.path.join(BASE_DIR, ".cdsapirc")

# Create directories if they don't exist
os.makedirs(LIVE_DATA_DIR, exist_ok=True)

# Check if .cdsapirc exists
if not os.path.exists(CDSAPI_RC_FILE):
    raise FileNotFoundError(f"CDSAPI configuration file {CDSAPI_RC_FILE} not found. Please create .cdsapirc in the project root with url and key.")

# Function to fetch live data from CDS ERA5 API with retry logic
def download_era5_data(max_retries=3, delay_seconds=10):
    # Use a date 6 days prior due to ERA5 data delay (increased from 5 to ensure data availability)
    fetch_date = datetime.now() - timedelta(days=6)
    year = str(fetch_date.year)
    month = str(fetch_date.month).zfill(2)
    day = str(fetch_date.day).zfill(2)

    logger.info(f"Fetching ERA5 data for {year}-{month}-{day}...")
    request = {
        'product_type': 'reanalysis',
        'variable': [
            '2m_temperature', 'surface_pressure', 'high_cloud_cover',
            'low_cloud_cover', 'medium_cloud_cover', 'total_column_rain_water',
            'soil_temperature_level_1', 'soil_temperature_level_2', 'total_precipitation',
            '2m_dewpoint_temperature', 'total_cloud_cover', 'surface_solar_radiation_downwards',
            'surface_thermal_radiation_downwards'
        ],
        'year': year,
        'month': month,
        'day': day,
        'time': ['05:00', '10:00', '15:00', '20:00'],
        'area': [13.0827, 77.5850, 12.9139, 77.6381],  # Bengaluru lat/lon bounds
        'format': 'netcdf'
    }
    logger.debug(f"API request: {request}")

    for attempt in range(max_retries):
        try:
            # Clean up existing ZIP file
            if os.path.exists(LIVE_DATA_ZIP):
                os.remove(LIVE_DATA_ZIP)
                logger.info(f"Removed existing file: {LIVE_DATA_ZIP}")

            # Set environment variable to use project-specific .cdsapirc
            os.environ['CDSAPI_RC'] = CDSAPI_RC_FILE
            c = Client(timeout=300)
            c.retrieve(
                'reanalysis-era5-single-levels',
                request,
                LIVE_DATA_ZIP
            )
            logger.info(f"ZIP file downloaded to {LIVE_DATA_ZIP}")
            return LIVE_DATA_ZIP
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"All {max_retries} attempts failed: {e}")
                raise RuntimeError(f"All {max_retries} attempts failed: {e}. Ensure CDSAPI configuration in {CDSAPI_RC_FILE} is correct and ERA5 license is accepted.")

# Main execution
if __name__ == "__main__":
    try:
        zip_file = download_era5_data(max_retries=3, delay_seconds=10)
        logger.info("Download completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        exit(1)