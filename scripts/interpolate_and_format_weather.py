import pandas as pd
import numpy as np
from datetime import datetime , timedelta
from scipy.interpolate import interp1d

# Load the predictions
try:
    # Use 'Unnamed: 0' as the index to avoid creating an extra column
    predictions_df = pd.read_csv('realtime_predictions.csv' , index_col='Unnamed: 0')
except FileNotFoundError:
    print("Error: 'realtime_predictions.csv' file not found. Please ensure the file exists in the working directory.")
    exit(1)

# Debug: Print the columns to verify the structure
print("Columns in the CSV file:" , predictions_df.columns.tolist())

# Verify that 'valid_time' column exists
if 'valid_time' not in predictions_df.columns:
    print("Error: 'valid_time' column not found in the CSV file. Available columns:" , predictions_df.columns.tolist())
    exit(1)

# Convert 'valid_time' to datetime with a specific format
try:
    predictions_df['valid_time'] = pd.to_datetime(predictions_df['valid_time'] , format='%Y-%m-%d %H:%M:%S' ,
                                                  dayfirst=False)
except ValueError as e:
    print("Error: Failed to parse 'valid_time' column. Please check the data for invalid timestamps.")
    print("First few rows of 'valid_time':")
    print(predictions_df['valid_time'].head())
    print("Error message:" , str(e))
    exit(1)

# Define the variables to interpolate
variables = ['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' , 'ssrd' , 'strd']

# Verify that all expected variables are present
missing_vars = [var for var in variables if var not in predictions_df.columns]
if missing_vars:
    print(f"Error: The following variables are missing in the CSV: {missing_vars}")
    exit(1)

# Convert timestamps to numeric values for interpolation
time_numeric = (predictions_df['valid_time'] - predictions_df['valid_time'].iloc[
    0]).dt.total_seconds() / 3600  # Hours since first timestamp

# Prepare for interpolation
interpolated_data = {'valid_time': []}
for var in variables:
    interpolated_data[var] = []

# Define the hourly time range (from the first timestamp to 24 hours later)
start_time = predictions_df['valid_time'].iloc[0]
end_time = start_time + timedelta(hours=24)
current_time = start_time
hourly_times = []
while current_time <= end_time:
    hourly_times.append(current_time)
    current_time += timedelta(hours=1)

# Interpolate each variable to hourly intervals
for var in variables:
    # Use linear interpolation
    interpolator = interp1d(time_numeric , predictions_df[var] , kind='linear' , fill_value='extrapolate')
    hourly_values = interpolator([(t - start_time).total_seconds() / 3600 for t in hourly_times])

    # For ssrd and strd, set to 0 during nighttime (20:00 to 05:00)
    if var in ['ssrd' , 'strd']:
        for i , t in enumerate(hourly_times):
            hour = t.hour
            if hour >= 20 or hour < 5:  # Nighttime
                hourly_values[i] = 0

    interpolated_data[var] = hourly_values

interpolated_data['valid_time'] = hourly_times

# Create a DataFrame with interpolated data
interpolated_df = pd.DataFrame(interpolated_data)


# Function to determine weather condition based on tcc and tp
def get_weather_condition(tcc , tp):
    tcc_percent = tcc * 100
    if tcc_percent <= 20:
        condition = "Sunny/Clear"
    elif tcc_percent <= 50:
        condition = "Partly Cloudy"
    elif tcc_percent <= 80:
        condition = "Mostly Cloudy"
    else:
        condition = "Overcast"

    if tp > 0:
        condition += " with Rain"
    return condition


# Format the output in weather app style
output_lines = []
current_day = None

for _ , row in interpolated_df.iterrows():
    timestamp = row['valid_time']
    day = timestamp.strftime('%B %d, %Y')
    hour = timestamp.strftime('%H:%M')

    # Group by day
    if day != current_day:
        if current_day is not None:
            output_lines.append("")  # Add a blank line between days
        output_lines.append(f"## {day}")
        output_lines.append("")
        current_day = day

    # Summary line
    temp_celsius = row['t2m'] - 273.15  # Convert Kelvin to Celsius
    precip_mm = row['tp'] * 1000  # Convert meters to millimeters
    cloud_cover_percent = row['tcc'] * 100  # Convert fraction to percentage
    weather_condition = get_weather_condition(row['tcc'] , row['tp'])
    summary = f"{hour} | Temp: {temp_celsius:.1f}°C | {weather_condition} | Precip: {precip_mm:.1f} mm | Cloud Cover: {cloud_cover_percent:.0f}%"
    output_lines.append(summary)

    # Detailed values
    details = "Details: "
    details += f"t2m: {row['t2m']:.2f} K, "
    details += f"sp: {row['sp']:.2f} Pa, "
    details += f"hcc: {row['hcc']:.2f}, "
    details += f"lcc: {row['lcc']:.2f}, "
    details += f"mcc: {row['mcc']:.2f}, "
    details += f"tcrw: {row['tcrw']:.2f} kg/m², "
    details += f"stl1: {row['stl1']:.2f} K, "
    details += f"stl2: {row['stl2']:.2f} K, "
    details += f"tp: {row['tp']:.4f} m, "
    details += f"d2m: {row['d2m']:.2f} K, "
    details += f"tcc: {row['tcc']:.2f}, "
    details += f"ssrd: {row['ssrd']:.2f} J/m², "
    details += f"strd: {row['strd']:.2f} J/m²"
    output_lines.append(details)
    output_lines.append("")  # Add a blank line after each hour

# Save the output to a Markdown file
output_text = "\n".join(output_lines)
with open('weather_forecast.md' , 'w') as f:
    f.write("# Bengaluru Weather Forecast\n\n")
    f.write(output_text)

print("Weather forecast saved to 'weather_forecast.md'")