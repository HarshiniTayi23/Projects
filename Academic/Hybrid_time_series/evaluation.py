from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

def evaluate_model(true_values, predictions):
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    mape = mean_absolute_percentage_error(true_values, predictions)
    return {'RMSE': rmse, 'MAPE': mape}
