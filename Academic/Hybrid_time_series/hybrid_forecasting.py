import numpy as np

def combine_forecasts(arima_pred, xgb_pred):
    return np.array(arima_pred) + np.array(xgb_pred)
