# 🫐 Wild Blueberry Yield Prediction

This project implements a complete **machine learning pipeline** to predict the **yield of wild blueberries** 🍇 based on agricultural, meteorological, and pollination data. It covers preprocessing, **dimensionality reduction (PCA)**, model training using **Random Forest Regressor**, and performance evaluation with visualizations 📊.

---

## 📁 Project Structure

├── main.py # 🚀 Entry point to run the pipeline

├── Data/ 

│ ├── WildBlueberryPollinationSimulationData.csv

├── src/

│ ├── data_loader.py # 📦 Functions for data loading

│ ├── preprocessing.py # 🧹 Data cleaning and PCA transformation

│ └── model.py # 🧠 Model training, evaluation, and plots

├── requirements.txt

└── README.md # 📄 Project documentation


---

## 📊 Dataset

- **📌 Source:** Kaggle – *Wild Blueberry Yield Prediction Dataset*
- **🧬 Features:**

  - **Pollinators:** `honeybee`, `bumbles`, `andrena`, `osmia` 🐝  
  - **Weather Metrics:** Max/Min/Avg Upper & Lower Temps 🌡️, Rainfall Days 🌧️  
  - **Plant Data:** `clonesize`, `fruitset`, `fruitmass`, `seeds` 🌱  
  - **🎯 Target:** `yield`

- **🧾 Shape:** `777 samples`, `17 features` (after cleaning)

---

## 🧰 Requirements

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```
---
### ▶️ Usage
1. Update DATA_PATH in main.py:
```python
DATA_PATH = 'path_to_data.csv'
```
2. Run the pipeline:
```bash
python main.py
```
---

### 🔄 Pipeline Overview

-📥 Data Loading: Load and clean dataset, remove irrelevant columns

-🧪 Data Splitting: 70:30 train-test split
---

### ⚙️ Feature Engineering (PCA):

-PCA on fruit-related features: fruitset, fruitmass, seeds

-PCA on temperature-related features

-PCA on rainfall-related features

🌲 Model Training: Random Forest Regressor with hyperparameter tuning via RandomizedSearchCV

## 📈 Evaluation:

-Training MSE and R²

-Feature importance visualization

-Actual vs Predicted scatter plot

### 🏁 Results
## ✅ Best Hyperparameters
```json
{
  "n_estimators": 200,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_depth": 10
}
```
## 📉 Model Performance

```yaml
Mean Squared Error (Training): 5196.56
R² (Training): 0.997
```

### 🌟 Key Features by Importance
-clonesize

-pca_feature (fruit metrics)

-pca_temperature (weather metrics)

-pca_rain (rainfall metrics)

---

### 🖼️ Visual Outputs
📊 Feature Importance Plot – Shows key contributors to prediction

🔵 Actual vs Predicted Plot – Highlights prediction accuracy

