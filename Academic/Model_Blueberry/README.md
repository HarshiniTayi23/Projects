# Wild Blueberry Yield Prediction

This project implements a machine learning pipeline to predict the yield of wild blueberries based on agricultural, meteorological, and pollination data. The data pipeline includes data preprocessing, dimensionality reduction using PCA, training a Random Forest Regressor, and evaluating model performance.

## Project Structure

├── main.py # Entry point to run the pipeline
├── src/
│ ├── data_loader.py # Functions for data loading
│ ├── preprocessing.py # Data preprocessing and PCA transformations
│ └── model.py # Model training, evaluation, and visualization
└── README.md # Project documentation

## Dataset

- **Source:** Kaggle - Wild Blueberry Yield Prediction Dataset
- **Features include:**

  - **Pollinators:** honeybee, bumbles, andrena, osmia
  - **Weather Metrics:** Max/Min/Average of Upper and Lower Temperature Ranges, Rainfall Days
  - **Plant Data:** clonesize, fruitset, fruitmass, seeds
  - **Target:** yield

- **Shape:** 777 samples, 17 features (after cleaning)

## Requirements

Python 3.x
pandas
numpy
scikit-learn
seaborn
matplotlib

pip install pandas numpy scikit-learn seaborn matplotlib

pgsql
Copy code

## Usage

Update `DATA_PATH` in `main.py` with the path to your CSV dataset.

```python
DATA_PATH = 'path_to_data.csv'
Then run:

css
Copy code
python main.py
Pipeline Overview
Data Loading: Load and clean data, dropping irrelevant columns.

Data Splitting: Split dataset into training and test sets (70:30 split).

Feature Engineering:

PCA on fruit-related features (fruitset, fruitmass, seeds)

PCA on meteorological temperature features

PCA on rainfall features

Model Training: Random Forest Regressor tuned via RandomizedSearchCV.

Evaluation:

Training MSE and R²

Feature importance visualization

Predicted vs Actual yield scatter plot

Results
Best Hyperparameters
bash
Copy code
{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 10}
Model Performance
yaml
Copy code
Mean Squared Error (MSE) - Training: 5196.56
R² - Training: 0.997
Key Features
Feature importance ranked the following as most significant:

clonesize

pca_feature (fruit metrics)

pca_temperature (weather metrics)

pca_rain (rainfall metrics)

Visual Outputs
Feature Importance Plot: Visual representation of feature contributions.

Actual vs Predicted Plot: Displays prediction accuracy.
```
