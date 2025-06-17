Weather_Prediction_Bengaluru

An AI-driven weather forecasting system for Bengaluru using ERA5 climate data and deep learning models. This project integrates data preprocessing, LSTM-based prediction, visualization, and a user-facing Flask dashboard for real-time, historical, and future forecasts.

────────────────────────────────────────────────────────────

Key Features

- Accurate 2m temperature prediction using ERA5 reanalysis data
- Real-time, future, and historical weather forecasting modules
- Deep learning (LSTM) model trained with climate variables
- Interactive Flask web interface with multiple forecast options
- Modular Python scripts for automation and reproducibility

────────────────────────────────────────────────────────────

Directory Structure

Weather_Prediction_Bengaluru/
│
├── .gitignore
├── .cdsapirc                (for CDS API key)
├── README.md
│
├── data/
│   ├── live/
│   │   ├── download/        (raw netCDF files)
│   │   ├── extracted/       (intermediate netCDF)
│   │   └── merged/          (merged ERA5 .nc files and zip)
│   └── processed/           (final forecast-ready .csv files and models)
│
├── scripts/
│   ├── app.py
│   ├── train_model.py
│   ├── preprocess_data.py
│   ├── real_time_weather_prediction.py
│   ├── custom_date_weather_prediction.py
│   ├── historical_weather_prediction.py
│   ├── future_download_era5_data.py
│   ├── future_process_era5_data.py
│   ├── future_predict_weather.py
│   ├── validate_download_era5.py
│   ├── validate_process_era5.py
│   ├── validate_weather_with_api.py
│   ├── interpolate_and_format_weather.py
│   ├── historical_plot_results.py
│   └── real_time_plot_weather_predictions.py
│
├── logs/
│   ├── custom_date_prediction.log
│   ├── real_time_prediction.log
│   ├── validation_2024.log
│   ├── weather_pipeline.log
│   ├── validation_with_api.log
│   └── validation_with_openweathermap.log
│
├── templates/
│   ├── index.html
│   ├── info.html
│   ├── realtime.html
│   ├── historical.html
│   └── future.html
│
├── static/
│   └── css/
│       └── styles.css
│
├── plots/
│   ├── predicted_vs_actual_plot.png
│   └── training_loss_plot.png
│
└── Models & Results/
    ├── weather_model.keras
    ├── scaler_X.pkl
    ├── scaler_y.pkl
    ├── sequences.npz
    ├── training_history.pkl
    ├── validation_metrics.csv
    ├── weather_predictions.csv
    ├── weather_predictions.pdf
    ├── weather_predictions.png
    ├── weather_forecast_all_variables.png
    └── weather_forecast_plots.png

────────────────────────────────────────────────────────────

Setup Instructions

Step 1: Clone the repository

git clone https://github.com/ashokmedar15/Weather_Prediction_Bengaluru.git
cd Weather_Prediction_Bengaluru

Step 2: Create a virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

Linux/macOS:
python3 -m venv venv
source venv/bin/activate

Step 3: Install dependencies

pip install -r requirements.txt

Step 4: Configure APIs and Environment

CDS API (.cdsapirc file):
url: https://cds.climate.copernicus.eu/api/v2
key: your-uid:your-api-key

Weatherstack (optional):
Insert key into app.py where required

Step 5: Prepare Data

Run scripts to:
- Download ERA5 data
- Merge and preprocess
- Add time-based cyclic features
- Scale using saved .pkl files
- Train model and save as .keras

Step 6: Start Flask App

python scripts/app.py
Visit http://127.0.0.1:5020 in browser

────────────────────────────────────────────────────────────

Model Info

Type       : LSTM
Framework  : TensorFlow/Keras
Inputs     : tp, ssrd, strd, stl1, stl2, d2m, tcc, tcrw, sp, t2m + cyclical time features
Output     : t2m (temperature at 2m height)
Metrics    : RMSE, MAE

Forecast Types

- Real-time forecast via live data
- Historical forecast using stored actuals
- Future prediction based on ERA5 time series

────────────────────────────────────────────────────────────

Troubleshooting

- Missing modules: pip install -r requirements.txt
- Missing files: check processed/ directory
- API failures: verify key, rate limit
- Future date errors: ensure <= 8 days from last available date
- Logs available in /logs/*.log for detailed errors

────────────────────────────────────────────────────────────

Contributing

- Fork repository
- Create new feature branch
- Commit and push changes
- Open Pull Request

License

MIT License

Contact

Ashok Kumar Medara
Email: medaraashok15@gmail.com
GitHub: https://github.com/ashokmedar15

