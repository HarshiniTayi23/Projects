from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        RandomForestRegressor: Best trained model
    """
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=101)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=20, cv=3, scoring='neg_mean_squared_error',
        random_state=101, n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    print(f"Best hyperparameters: {random_search.best_params_}")
    
    return random_search.best_estimator_

def evaluate_model(model, X_train, y_train):
    """
    Evaluate the model on training data.
    
    Args:
        model: Trained model
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        tuple: MSE, R2 score, and predictions
    """
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    return mse_train, r2_train, y_train_pred

def evaluate_model_test(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target
        
    Returns:
        tuple: MSE, R2 score, and predictions
    """
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    return mse_test, r2_test, y_test_pred

def plot_feature_importance(model, X_train):
    """
    Plot feature importances from the trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        X_train (pandas.DataFrame): Training features
    """
    feature_importance = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=features_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Random Forest")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_actual, y_pred, title="Predictions vs Actual Values"):
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        y_actual (pandas.Series): Actual values
        y_pred (array): Predicted values
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.7)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 
             color='red', linestyle='--', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()