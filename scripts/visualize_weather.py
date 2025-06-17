import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='visualization.log',
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

def visualize_weather():
    try:
        logger.info("Starting visualization...")

        # Load the predictions
        data_path = "C:\\Users\\91905\\PycharmProjects\\WeatherPrediction_Bengaluru\\data\\processed\\weather_predictions.csv"
        df = pd.read_csv(data_path)
        logger.info(f"Raw data columns: {df.columns.tolist()}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"First few rows:\n{df.head().to_string()}")
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        logger.info(f"First few valid_time values: {df['valid_time'].head().tolist()}")

        # Variables to plot
        variables = {
            'predicted_t2m': ('Temperature at 2m (°C)', 'Temperature'),
            'predicted_sp': ('Surface Pressure (Pa)', 'Surface Pressure'),
            'predicted_hcc': ('High Cloud Cover (fraction)', 'High Cloud Cover'),
            'predicted_lcc': ('Low Cloud Cover (fraction)', 'Low Cloud Cover'),
            'predicted_mcc': ('Medium Cloud Cover (fraction)', 'Medium Cloud Cover'),
            'predicted_tcc': ('Total Cloud Cover (fraction)', 'Total Cloud Cover'),
            'predicted_tcrw': ('Total Column Rain Water (kg/m²)', 'Total Column Rain Water'),
            'predicted_stl1': ('Soil Temperature Level 1 (°C)', 'Soil Temperature L1'),
            'predicted_stl2': ('Soil Temperature Level 2 (°C)', 'Soil Temperature L2'),
            'predicted_tp': ('Total Precipitation (mm)', 'Total Precipitation'),
            'predicted_d2m': ('Dew Point Temperature (°C)', 'Dew Point Temperature'),
            'predicted_ssrd': ('Solar Radiation (W/m²)', 'Solar Radiation'),
            'predicted_strd': ('Thermal Radiation (W/m²)', 'Thermal Radiation')
        }

        # Verify all columns exist
        missing_cols = [var for var in variables if var not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        # Set up the plot
        logger.info("Setting up the plot...")
        fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(12, 5 * len(variables)), sharex=True)
        fig.suptitle('Weather Predictions for Bengaluru (2025-04-22 to 2025-04-29)', fontsize=16)

        # Plot each variable
        for idx, (var, (ylabel, title)) in enumerate(variables.items()):
            ax = axes[idx]
            logger.info(f"Plotting {var} with {len(df[var])} data points, min: {df[var].min()}, max: {df[var].max()}")
            ax.plot(df['valid_time'], df[var], label=title, color='blue')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
            # Format x-axis
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save the plot as PNG and PDF
        output_png = "C:\\Users\\91905\\PycharmProjects\\WeatherPrediction_Bengaluru\\data\\processed\\weather_predictions.png"
        output_pdf = "C:\\Users\\91905\\PycharmProjects\\WeatherPrediction_Bengaluru\\data\\processed\\weather_predictions.pdf"
        plt.savefig(output_png, format='png', bbox_inches='tight')
        logger.info(f"Visualization saved as PNG to {output_png}")
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
        logger.info(f"Visualization saved as PDF to {output_pdf}")

        # Display the plot (comment out if running non-interactively)
        plt.show()

        plt.close()

    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise

if __name__ == "__main__":
    visualize_weather()