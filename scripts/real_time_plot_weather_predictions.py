import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
OUTPUT_DIR = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\processed"
FORECAST_FILE = os.path.join(OUTPUT_DIR, "real_time_forecast.csv")

# Define expected columns
COLUMNS = ['valid_time', 'predicted_t2m', 'predicted_sp', 'predicted_hcc', 'predicted_lcc',
           'predicted_mcc', 'predicted_tcrw', 'predicted_stl1', 'predicted_stl2',
           'predicted_tp', 'predicted_d2m', 'predicted_tcc', 'predicted_ssrd', 'predicted_strd']

# Read the forecast data, skipping bad lines and using expected columns
df = pd.read_csv(FORECAST_FILE, usecols=COLUMNS, on_bad_lines='skip')

# Convert valid_time to datetime, handling potential format issues
try:
    df['valid_time'] = pd.to_datetime(df['valid_time'], errors='coerce')
except Exception as e:
    print(f"Error converting valid_time to datetime: {e}")
    raise

# Filter for the latest predictions
# Extract date and hour from valid_time
df['date'] = df['valid_time'].dt.date
df['hour'] = df['valid_time'].dt.hour

# Target hours for predictions (05:00, 10:00, 15:00, 20:00 IST)
target_hours = [5, 10, 15, 20]

# Get the latest date in the dataset
latest_date = df['date'].max()

# Filter for the latest date and target hours
df_filtered = df[(df['date'] == latest_date) & (df['hour'].isin(target_hours))]

# Debug: Print filtered data to verify
print("Filtered DataFrame:")
print(df_filtered)

# Drop temporary columns
df_filtered = df_filtered.drop(columns=['date', 'hour'])

# Ensure there are exactly 4 rows (one for each target hour)
if len(df_filtered) != 4:
    raise ValueError(f"Expected 4 rows for target hours on {latest_date}, but found {len(df_filtered)} rows.")

# Sort by valid_time to ensure correct order
df_filtered = df_filtered.sort_values('valid_time')

# Unit corrections
# Convert temperatures from Kelvin to Celsius if values are > 100
temp_cols = ['predicted_t2m', 'predicted_stl1', 'predicted_stl2', 'predicted_d2m']
for col in temp_cols:
    mask = df_filtered[col] > 100  # Likely in Kelvin
    df_filtered.loc[mask, col] = df_filtered.loc[mask, col] - 273.15

# Clip radiation values to reasonable ranges
df_filtered['predicted_ssrd'] = df_filtered['predicted_ssrd'].clip(upper=1200.0)  # Max SSRD
df_filtered['predicted_strd'] = df_filtered['predicted_strd'].clip(upper=600.0)   # Max STRD

# Plotting all variables
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20))
axes = axes.flatten()

# List of variables to plot with their labels and colors
variables = [
    ('predicted_t2m', 'Temperature (°C)', 'orange'),
    ('predicted_sp', 'Surface Pressure (hPa)', 'purple'),
    ('predicted_hcc', 'High Cloud Cover (0-1)', 'cyan'),
    ('predicted_lcc', 'Low Cloud Cover (0-1)', 'blue'),
    ('predicted_mcc', 'Mid Cloud Cover (0-1)', 'teal'),
    ('predicted_tcrw', 'Total Column Rain Water (kg/m²)', 'magenta'),
    ('predicted_stl1', 'Soil Temperature Level 1 (°C)', 'red'),
    ('predicted_stl2', 'Soil Temperature Level 2 (°C)', 'darkred'),
    ('predicted_tp', 'Precipitation (mm)', 'lightblue'),
    ('predicted_d2m', 'Dew Point (°C)', 'gold'),
    ('predicted_tcc', 'Total Cloud Cover (0-1)', 'green'),
    ('predicted_ssrd', 'Surface Shortwave Radiation (W/m²)', 'blue'),
    ('predicted_strd', 'Surface Thermal Radiation (W/m²)', 'darkblue')
]

# Plot each variable
for idx, (var, label, color) in enumerate(variables):
    axes[idx].plot(df_filtered['valid_time'], df_filtered[var], marker='o', label=label, color=color)
    axes[idx].set_title(label)
    axes[idx].set_xlabel('Time')
    axes[idx].set_ylabel(label)
    axes[idx].grid(True)
    axes[idx].legend()

# Remove empty subplot (if any)
for idx in range(len(variables), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'weather_forecast_all_variables.png'))
plt.show()