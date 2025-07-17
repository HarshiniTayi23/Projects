import pandas as pd
from statsmodels.tsa.stattools import adfuller

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}

def difference_series(series):
    return series.diff().dropna()

def preprocess_data(file_path):
    data = load_data(file_path)
    series = data['Close']
    stationary_check = check_stationarity(series)
    if stationary_check['p-value'] > 0.05:
        series = difference_series(series)
    return series
