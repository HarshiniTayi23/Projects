import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
from arima_model import train_arima, predict_arima
from xgboost_model import prepare_features, train_xgboost, predict_xgboost
from hybrid_forecasting import combine_forecasts
from evaluation import evaluate_model

FILE_PATH = 'path_to_your_csv.csv'

series = preprocess_data(FILE_PATH)

# Train ARIMA
arima_model = train_arima(series)
arima_pred = predict_arima(arima_model, start=3, end=len(series)-1)

# Prepare XGBoost on ARIMA residuals
residuals = series[3:] - arima_pred
X, y = prepare_features(series)
xgb_model = train_xgboost(X, residuals)
xgb_pred = predict_xgboost(xgb_model, X)

# Hybrid forecast
hybrid_pred = combine_forecasts(arima_pred, xgb_pred)

# Evaluate
metrics = evaluate_model(y, hybrid_pred)
print(f"Hybrid Model Performance: {metrics}")

# Plot
plt.figure(figsize=(10,6))
plt.plot(series[3:].values, label='Actual')
plt.plot(hybrid_pred, label='Hybrid Forecast')
plt.legend()
plt.title('Hybrid Forecast vs Actual')
plt.show()
