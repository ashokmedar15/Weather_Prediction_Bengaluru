import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Define paths
BASE_DIR = r"C:\Users\91905\PycharmProjects\WeatherPrediction_Bengaluru"
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
HISTORY_FILE = os.path.join(PROCESSED_DATA_DIR, "training_history.pkl")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

# Create plots directory if it doesn't exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Plot 1: Training and Validation Loss
def plot_training_history():
    """Plot the training and validation loss from the training history."""
    print("Loading training history...")
    with open(HISTORY_FILE, 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, 'training_loss_plot.png'))
    plt.close()
    print("Training loss plot saved to:", os.path.join(PLOT_DIR, 'training_loss_plot.png'))

# Plot 2: Predicted vs Actual Values
def plot_predicted_vs_actual():
    """Plot predicted vs actual values for key weather variables."""
    # Define the predicted and actual values from the latest run
    targets = ['t2m', 'sp', 'hcc', 'lcc', 'mcc', 'tcrw', 'stl1', 'stl2', 'tp', 'd2m', 'tcc', 'ssrd', 'strd']
    predicted_values = [15, 900, 0.358409583568573, 0.11752863228321075, 0.2063121646642685, 0, 15, 15, 9.948632679879665, 15, 0.3765895664691926, 0, 5.3887214097711776e-06]
    actual_values = [15, 900, 0.1665408025, 0.02147479325, 0.3616577975, 0.0049479158974129, 15, 15, 10, 15, 0.3976285475000002, 2.7575979229493417e-05, 0.0002554069900929903]

    # Labels for plotting
    labels = ['Temperature (°C)', 'Surface Pressure (hPa)', 'High Cloud Cover', 'Low Cloud Cover',
              'Medium Cloud Cover', 'Total Column Rain Water (kg/m²)', 'Soil Temp L1 (°C)',
              'Soil Temp L2 (°C)', 'Total Precipitation (mm)', 'Dew Point Temp (°C)',
              'Total Cloud Cover', 'Solar Radiation (W/m²)', 'Thermal Radiation (W/m²)']

    x = np.arange(len(targets))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, predicted_values, width, label='Predicted', color='skyblue')
    plt.bar(x + width/2, actual_values, width, label='Actual', color='salmon')
    plt.xlabel('Weather Variables')
    plt.ylabel('Values')
    plt.title('Predicted vs Actual Weather Variables for 2025-04-16 15:01')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'predicted_vs_actual_plot.png'))
    plt.close()
    print("Predicted vs Actual plot saved to:", os.path.join(PLOT_DIR, 'predicted_vs_actual_plot.png'))

if __name__ == "__main__":
    try:
        plot_training_history()
        plot_predicted_vs_actual()
        print("All plots generated successfully!")
    except Exception as e:
        print("An error occurred while generating plots:", e)