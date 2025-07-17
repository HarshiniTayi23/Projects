import xgboost as xgb
import numpy as np

def prepare_features(series):
    X, y = [], []
    for i in range(3, len(series)):
        X.append([series[i-3], series[i-2], series[i-1]])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X_test):
    return model.predict(X_test)
