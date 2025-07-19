import os
import sys

# Add the src directory to the path if using a src structure
# Uncomment the following lines if you organize files in a src/ directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import functions from our modules
from data_loader import load_data
from preprocessing import split_data, apply_pca_feature, apply_pca_temperature, apply_pca_rain
from model import train_model, evaluate_model, evaluate_model_test, plot_feature_importance, plot_predictions

def main():
    """Main function to execute the blueberry yield prediction pipeline."""
    
    # Data path - update this to match your data location
    DATA_PATH = r'C:\Users\tayis\OneDrive\Desktop\Projects\Academic\Model_Blueberry\Data\WildBlueberryPollinationSimulationData.csv'
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable with the correct path to your CSV file.")
        return
    
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Define feature groups
    fruit_features = ['fruitset', 'fruitmass', 'seeds']
    temp_features = ['MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
                     'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange']
    rain_features = ['RainingDays', 'AverageRainingDays']
    
    print("\nApplying PCA transformations...")
    
    # Apply PCA to different feature groups
    print("Applying PCA to fruit features...")
    X_train, X_test = apply_pca_feature(X_train, X_test, fruit_features)
    
    print("Applying PCA to temperature features...")
    X_train, X_test = apply_pca_temperature(X_train, X_test, temp_features)
    
    print("Applying PCA to rain features...")
    X_train, X_test = apply_pca_rain(X_train, X_test, rain_features)
    
    print(f"Final feature shape - Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"Final features: {list(X_train.columns)}")
    
    # Train the model
    print("\nTraining Random Forest model with hyperparameter tuning...")
    model = train_model(X_train, y_train)
    
    # Evaluate on training set
    print("\nEvaluating model on training set...")
    train_mse, train_r2, y_train_pred = evaluate_model(model, X_train, y_train)
    print(f'Training MSE: {train_mse:.2f}')
    print(f'Training R²: {train_r2:.4f}')
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_mse, test_r2, y_test_pred = evaluate_model_test(model, X_test, y_test)
    print(f'Test MSE: {test_mse:.2f}')
    print(f'Test R²: {test_r2:.4f}')
    
    # Plot feature importance
    print("\nGenerating feature importance plot...")
    plot_feature_importance(model, X_train)
    
    # Plot predictions vs actual values with descriptive titles
    print("Generating prediction plots...")
    plot_predictions(y_train, y_train_pred, "Training Set: Predicted vs Actual Blueberry Yield")
    plot_predictions(y_test, y_test_pred, "Test Set: Predicted vs Actual Blueberry Yield")
    
    print("\nAnalysis completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY RESULTS")
    print("="*50)
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()