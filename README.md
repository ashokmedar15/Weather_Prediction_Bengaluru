## Step 1: Data Collection and Preprocessing
- Dataset validated, cleaned, and preprocessed using `scripts/preprocess_data.py`.
- Preprocessed data saved to `data/processed/preprocessed_data.csv`.
- Sequences for training saved to `data/processed/sequences.npz`.
- Scalers saved for denormalization in later steps.

## Step 2: Model Development and Training
- LSTM model with attention trained using `scripts/train_model.py`.
- Model saved to `data/processed/weather_model.h5`.
- Training history saved to `data/processed/training_history.pkl`.