# 📈 Hybrid Time Series Forecasting using ARIMA and XGBoost

## 🚀 Project Overview

This project implements an **end-to-end Hybrid Time Series Forecasting Model** that combines the strengths of:
- **ARIMA**: To model the **linear patterns** in time series data.
- **XGBoost**: To capture the **non-linear residuals** left by the ARIMA model.

By integrating both statistical and machine learning approaches, the hybrid model aims to deliver **superior forecasting accuracy** on time series data compared to standalone methods.

---

## 🎯 Key Features
- Automated data preprocessing including **stationarity checks**.
- **ARIMA modeling** for linear dependencies in the data.
- **XGBoost modeling** on residuals to capture non-linearities.
- Model evaluation using **RMSE** and **MAPE** metrics.
- Visualization of **actual vs predicted values**.
- Modular codebase for ease of understanding, extension, and maintenance.

---

## 🛠️ Tech Stack

| Technology     | Purpose                  |
|----------------|--------------------------|
| **Python 3.x** | Programming Language     |
| **pandas**     | Data handling            |
| **numpy**      | Numerical computations   |
| **statsmodels**| ARIMA modeling           |
| **xgboost**    | Machine Learning (Boosting) |
| **scikit-learn** | Evaluation metrics      |
| **matplotlib** | Visualization            |

---

## 🧩 How it Works

1. **Preprocessing**
   - Load the time series data.
   - Perform stationarity check using **ADF Test**.
   - Apply differencing if data is non-stationary.

2. **Modeling**
   - Train an **ARIMA model** to predict the main time series trend.
   - Compute **residuals** (actual - ARIMA prediction).
   - Train an **XGBoost regressor** on lagged residuals to model non-linearities.

3. **Hybrid Forecast**
   - Final forecast = **ARIMA predictions + XGBoost residual predictions**.

4. **Evaluation**
   - Use **Root Mean Squared Error (RMSE)** and **Mean Absolute Percentage Error (MAPE)** to evaluate performance.

5. **Visualization**
   - Plot actual vs hybrid forecasted values.

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd projects/academic/Hybrid_time_series
