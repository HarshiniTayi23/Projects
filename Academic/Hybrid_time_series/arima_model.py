from statsmodels.tsa.arima.model import ARIMA

def train_arima(series, order=(5,1,0)):
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    return fitted_model

def predict_arima(fitted_model, start, end):
    return fitted_model.predict(start=start, end=end, typ='levels')
