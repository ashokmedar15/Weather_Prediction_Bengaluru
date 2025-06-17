import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input , LSTM , Dense , Attention , Concatenate , GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

# Define paths
BASE_DIR = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru"
DATA_DIR = os.path.join(BASE_DIR , "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR , "processed")
MODEL_DIR = PROCESSED_DATA_DIR
PREPROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR , "preprocessed_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR , "weather_model.keras")
SCALER_X_FILE = os.path.join(PROCESSED_DATA_DIR , "scaler_X.pkl")
SCALER_Y_FILE = os.path.join(PROCESSED_DATA_DIR , "scaler_y.pkl")
HISTORY_FILE = os.path.join(PROCESSED_DATA_DIR , "training_history.pkl")


def load_and_preprocess_data(file_path):
    """Load and preprocess the data for training."""
    print("Loading data...")
    df = pd.read_csv(file_path)

    # Ensure valid_time is in datetime format
    df['valid_time'] = pd.to_datetime(df['valid_time'])

    # Extract time-based features
    df['hour'] = df['valid_time'].dt.hour
    df['day'] = df['valid_time'].dt.day
    df['month'] = df['valid_time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    print("Preprocessed DataFrame:\n" , df.head())
    return df


def create_sequences(df , features , targets , seq_length=4):
    """Create sequences for training the LSTM model."""
    print("Creating sequences...")
    X , y = [] , []
    data = df[features + targets].values
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length) , :len(features)])
        y.append(data[i + seq_length , len(features):])  # Predict the next time step
    X = np.array(X)
    y = np.array(y)
    print("X shape:" , X.shape)
    print("y shape:" , y.shape)
    return X , y


def build_model(input_shape , output_dim):
    """Build the LSTM model with attention mechanism."""
    inputs = Input(shape=input_shape)

    # LSTM layer with dropout
    lstm_out , hidden_state , cell_state = LSTM(50 , return_sequences=True , return_state=True , dropout=0.2)(inputs)

    # Attention mechanism
    attention = Attention()([lstm_out , lstm_out])

    # Concatenate LSTM output and attention output
    combined = Concatenate()([lstm_out , attention])

    # Reduce sequence dimension to a single vector
    pooled = GlobalAveragePooling1D()(combined)

    # Dense layer for output (single prediction)
    outputs = Dense(output_dim)(pooled)

    # Build model
    model = Model(inputs=inputs , outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001) , loss='mse')
    print("Model summary:")
    model.summary()
    return model


def train_model():
    """Train the weather prediction model and save it along with scalers and history."""
    # Load and preprocess data
    df = load_and_preprocess_data(PREPROCESSED_FILE)

    # Define features and targets
    features = ['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' , 'ssrd' ,
                'strd' ,
                'hour_sin' , 'hour_cos' , 'day_sin' , 'day_cos' , 'month_sin' , 'month_cos']
    targets = ['t2m' , 'sp' , 'hcc' , 'lcc' , 'mcc' , 'tcrw' , 'stl1' , 'stl2' , 'tp' , 'd2m' , 'tcc' , 'ssrd' , 'strd']
    print("Features:" , features)
    print("Targets:" , targets)

    # Create sequences
    X , y = create_sequences(df , features , targets , seq_length=4)

    # Scale the data
    print("Scaling data...")
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Reshape for scaling
    X_reshaped = X.reshape(-1 , len(features))
    y_reshaped = y.reshape(-1 , len(targets))

    X_scaled = scaler_X.fit_transform(X_reshaped)
    y_scaled = scaler_y.fit_transform(y_reshaped)

    # Reshape back to 3D for LSTM
    X_scaled = X_scaled.reshape(X.shape)
    print("Scaled X shape:" , X_scaled.shape)
    print("Scaled y shape:" , y_scaled.shape)

    # Build and train the model
    model = build_model(input_shape=(X.shape[1] , X.shape[2]) , output_dim=len(targets))
    history = model.fit(X_scaled , y_scaled , epochs=50 , batch_size=32 , validation_split=0.2 , verbose=1)

    # Save the model
    print("Saving model...")
    model.save(MODEL_FILE)

    # Save the scalers
    print("Saving scalers...")
    with open(SCALER_X_FILE , 'wb') as f:
        pickle.dump(scaler_X , f)
    with open(SCALER_Y_FILE , 'wb') as f:
        pickle.dump(scaler_y , f)

    # Save the training history
    print("Saving training history...")
    with open(HISTORY_FILE , 'wb') as f:
        pickle.dump(history.history , f)

    print("Training complete!")


if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print("An error occurred during training:" , e)