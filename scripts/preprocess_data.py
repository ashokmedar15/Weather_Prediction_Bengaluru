import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Define paths
DATA_DIR = "data"
OUTPUT_DIR = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data/processed"
DATA_FILE = os.path.join(DATA_DIR, r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru\data\final_merged_project.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "preprocessed_data.csv")
SEQUENCES_FILE = os.path.join(OUTPUT_DIR, "sequences.npz")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# Step 2: Data Validation
print("Validating data...")
# Define expected ranges for variables
RANGES = {
    't2m': (293, 308),  # Kelvin, typical for Bengaluru
    'sp': (90000, 100000),  # Pascals
    'hcc': (0, 1),  # Cloud cover fraction
    'lcc': (0, 1),
    'mcc': (0, 1),
    'tcc': (0, 1),
    'tcrw': (0, 0.1),  # kg/m², adjust based on data
    'stl1': (293, 308),  # Kelvin
    'stl2': (293, 308),
    'tp': (0, 0.01),  # Meters, expected to be 0 in dataset
    'd2m': (283, 303),  # Kelvin, dew point temperature
    'ssrd': (0, 1e7),  # J/m²
    'strd': (0, 1e7),  # J/m²
}

# Check for anomalies
for column, (min_val, max_val) in RANGES.items():
    if column in df:
        out_of_range = df[(df[column] < min_val) | (df[column] > max_val)][column]
        if not out_of_range.empty:
            print(f"Warning: {column} has {len(out_of_range)} values outside range [{min_val}, {max_val}]:")
            print(out_of_range.head())

# Step 3: Data Cleaning
print("Cleaning data...")
# Drop irrelevant columns
df = df.drop(columns=['sst', 'number', 'expver', 'expver_satellite'], errors='ignore')

# Convert valid_time to datetime
df['valid_time'] = pd.to_datetime(df['valid_time'])

# Handle missing values for ssrd and strd (0 at night)
df['hour'] = df['valid_time'].dt.hour
df.loc[df['hour'].isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]), ['ssrd', 'strd']] = 0

# Impute other missing values with mean
for column in ['t2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd', 'strd']:
    df[column] = df[column].fillna(df[column].mean())

# Aggregate by time (average across coordinates)
df = df.groupby('valid_time').mean().reset_index()

# Step 4: Feature Engineering
print("Performing feature engineering...")
# Extract temporal features
df['hour'] = df['valid_time'].dt.hour
df['day'] = df['valid_time'].dt.day
df['month'] = df['valid_time'].dt.month

# Cyclical encoding for temporal features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Step 5: Normalization
print("Normalizing data...")
# Define features and targets
features = ['t2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd', 'strd',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
targets = ['t2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd', 'strd']

# Normalize features and targets
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(df[features])
y = scaler_y.fit_transform(df[targets])

# Convert back to DataFrame for easier handling
X = pd.DataFrame(X, columns=features, index=df.index)
y = pd.DataFrame(y, columns=targets, index=df.index)

# Add valid_time back for reference
X['valid_time'] = df['valid_time']
y['valid_time'] = df['valid_time']

# Save preprocessed data
preprocessed_df = pd.concat([X, y.drop(columns='valid_time')], axis=1)
preprocessed_df.to_csv(OUTPUT_FILE, index=False)
print(f"Preprocessed data saved to {OUTPUT_FILE}")

# Step 6: Create Sequences
print("Creating sequences...")
def create_sequences(X, y, time_steps=4, future_steps=4):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - future_steps + 1):
        Xs.append(X.iloc[i:(i + time_steps)][features].values)
        ys.append(y.iloc[(i + time_steps):(i + time_steps + future_steps)][targets].values)
    return np.array(Xs), np.array(ys)

time_steps = 4  # Past 24 hours (4 time steps at 6-hour intervals)
future_steps = 4  # Predict next 24 hours (4 time steps)
X_seq, y_seq = create_sequences(X, y, time_steps, future_steps)

# Save sequences for training
np.savez(SEQUENCES_FILE, X_seq=X_seq, y_seq=y_seq)
print(f"Sequences saved to {SEQUENCES_FILE}")
print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")

# Save scalers for later use (e.g., denormalization)
import pickle
with open(os.path.join(OUTPUT_DIR, 'scaler_X.pkl'), 'wb') as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(OUTPUT_DIR, 'scaler_y.pkl'), 'wb') as f:
    pickle.dump(scaler_y, f)
print("Scalers saved for later use.")