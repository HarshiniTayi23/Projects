import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def split_data(df, test_size=0.3, random_state=101):
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pandas.DataFrame): The input dataframe
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    y = df['yield']
    X = df.drop('yield', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def apply_pca_feature(X_train, X_test, features):
    """
    Apply PCA to fruit-related features (fruitset, fruitmass, seeds).
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        features (list): List of feature names to apply PCA to
        
    Returns:
        tuple: Modified X_train and X_test with PCA feature
    """
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[features])
    X_test_scaled = scaler.transform(X_test[features])
    
    # Apply PCA
    pca = PCA(n_components=1)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Add PCA feature and remove original features
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train['pca_feature'] = X_train_pca.flatten()
    X_test['pca_feature'] = X_test_pca.flatten()
    
    X_train = X_train.drop(columns=features)
    X_test = X_test.drop(columns=features)
    
    return X_train, X_test

def apply_pca_temperature(X_train, X_test, temp_features):
    """
    Apply PCA to temperature-related features.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        temp_features (list): List of temperature feature names
        
    Returns:
        tuple: Modified X_train and X_test with PCA temperature feature
    """
    # Apply PCA to temperature variables
    pca = PCA(n_components=2)  # Start with 2 to check explained variance
    X_temp_pca = pca.fit_transform(X_train[temp_features])
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Temperature PCA - Explained variance by each component: {explained_variance}")
    
    # Since the first component explains most variance, use only the first component
    pca_single = PCA(n_components=1)
    X_train_temp_pca = pca_single.fit_transform(X_train[temp_features])
    X_test_temp_pca = pca_single.transform(X_test[temp_features])
    
    # Add PCA feature and remove original features
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train['pca_temperature'] = X_train_temp_pca.flatten()
    X_test['pca_temperature'] = X_test_temp_pca.flatten()
    
    X_train = X_train.drop(columns=temp_features)
    X_test = X_test.drop(columns=temp_features)
    
    return X_train, X_test

def apply_pca_rain(X_train, X_test, rain_features):
    """
    Apply PCA to rain-related features.
    
    Args:
        X_train (pandas.DataFrame): Training features
        X_test (pandas.DataFrame): Testing features
        rain_features (list): List of rain feature names
        
    Returns:
        tuple: Modified X_train and X_test with PCA rain feature
    """
    # Apply PCA to rain variables
    pca = PCA(n_components=1)
    X_rain_pca = pca.fit_transform(X_train[rain_features])
    X_test_rain_pca = pca.transform(X_test[rain_features])
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Rain PCA - Explained variance by component: {explained_variance}")
    
    # Add PCA feature and remove original features
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train['pca_rain'] = X_rain_pca.flatten()
    X_test['pca_rain'] = X_test_rain_pca.flatten()
    
    X_train = X_train.drop(columns=rain_features)
    X_test = X_test.drop(columns=rain_features)
    
    return X_train, X_test