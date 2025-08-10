# Essential libraries for data analysis
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns           # Statistical visualizations
import sqlite3                  # SQLite database
from sqlalchemy import create_engine  # Database connections
from datetime import datetime   # Date handling
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#Load directly if you have the files
train_df = pd.read_csv(r'C:\Users\tayis\OneDrive\Desktop\Projects\Personal\Dynamic_Business_Dashboard_with_Predictive_Analysis\Data\train.csv')
store_df = pd.read_csv(r'C:\Users\tayis\OneDrive\Desktop\Projects\Personal\Dynamic_Business_Dashboard_with_Predictive_Analysis\Data\store.csv')
test_df = pd.read_csv(r'C:\Users\tayis\OneDrive\Desktop\Projects\Personal\Dynamic_Business_Dashboard_with_Predictive_Analysis\Data\test.csv')

print("Dataset shapes:")
print(f"Training data: {train_df.shape}")
print(f"Store data: {store_df.shape}")
print(f"Test data: {test_df.shape}")

# Examine training data structure
print("=== TRAINING DATA OVERVIEW ===")
print(train_df.head())
print("\nData types:")
print(train_df.dtypes)
print("\nBasic statistics:")
print(train_df.describe())

# Check for missing values
print("\n=== MISSING VALUES ===")
print("Training data missing values:")
print(train_df.isnull().sum())
print("\nStore data missing values:")
print(store_df.isnull().sum())

# Convert date column to datetime
train_df['Date'] = pd.to_datetime(train_df['Date'])

# Extract date features
train_df['Year'] = train_df['Date'].dt.year
train_df['Month'] = train_df['Date'].dt.month
train_df['Day'] = train_df['Date'].dt.day
train_df['DayOfWeek'] = train_df['Date'].dt.dayofweek
train_df['WeekOfYear'] = train_df['Date'].dt.isocalendar().week

# Handle missing values in training data
print("Handling missing values...")
train_df['Open'].fillna(1, inplace=True)  # Assume open if not specified

# Remove rows where store was closed (Sales = 0)
train_df = train_df[train_df['Open'] == 1]

print(f"After cleaning: {train_df.shape}")

# Handle missing values in store data
print("=== CLEANING STORE DATA ===")

# Fill missing competition data
store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median(), inplace=True)
store_df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
store_df['CompetitionOpenSinceYear'].fillna(0, inplace=True)

# Fill missing promo data
store_df['Promo2SinceWeek'].fillna(0, inplace=True)
store_df['Promo2SinceYear'].fillna(0, inplace=True)
store_df['PromoInterval'].fillna('None', inplace=True)

print("Store data after cleaning:")
print(store_df.isnull().sum())

# Merge training data with store information
print("=== MERGING DATASETS ===")
merged_df = train_df.merge(store_df, on='Store', how='left')

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Columns: {list(merged_df.columns)}")

# Create comprehensive data dictionary
data_dictionary = {
    'Store': 'Unique store identifier (1-1115)',
    'Date': 'Date of sales record',
    'Sales': 'Daily revenue (target variable)',
    'Customers': 'Number of customers that day',
    'Open': 'Store open status (0=closed, 1=open)',
    'Promo': 'Daily promotion active (0=no, 1=yes)',
    'StateHoliday': 'Public holiday (0=no, a=public, b=Easter, c=Christmas)',
    'SchoolHoliday': 'School holiday indicator (0=no, 1=yes)',
    'StoreType': 'Store format (a,b,c,d)',
    'Assortment': 'Product assortment (a=basic, b=extra, c=extended)',
    'CompetitionDistance': 'Distance to nearest competitor (meters)',
    'CompetitionOpenSinceMonth': 'Month competitor opened',
    'CompetitionOpenSinceYear': 'Year competitor opened',
    'Promo2': 'Continuous promotion participation (0=no, 1=yes)',
    'Promo2SinceWeek': 'Week continuous promotion started',
    'Promo2SinceYear': 'Year continuous promotion started',
    'PromoInterval': 'Months when Promo2 is active'
}

# Save data dictionary
dict_df = pd.DataFrame(list(data_dictionary.items()), 
                      columns=['Column', 'Description'])
dict_df.to_csv('data_dictionary.csv', index=False)
print("Data dictionary saved!")

# Create SQLite database
print("=== CREATING DATABASE ===")
engine = create_engine('sqlite:///rossmann_sales.db')

# Upload cleaned data to database
merged_df.to_sql('sales_data', engine, if_exists='replace', index=False)
store_df.to_sql('store_info', engine, if_exists='replace', index=False)

print("Data uploaded to SQLite database!")

# Test database connection
connection = sqlite3.connect('rossmann_sales.db')
test_query = "SELECT COUNT(*) as total_records FROM sales_data"
result = pd.read_sql_query(test_query, connection)
print(f"Database contains {result['total_records'][0]} records")
connection.close()

# Key insights about the dataset
print("=== DATASET INSIGHTS ===")
print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
print(f"Number of stores: {merged_df['Store'].nunique()}")
print(f"Total sales records: {len(merged_df)}")
print(f"Average daily sales: ${merged_df['Sales'].mean():.2f}")
print(f"Total revenue: ${merged_df['Sales'].sum():,.2f}")

# Store performance summary
store_summary = merged_df.groupby('Store').agg({
    'Sales': ['mean', 'sum', 'count'],
    'Customers': 'mean'
}).round(2)

print("\nTop 5 performing stores by total sales:")
print(store_summary.sort_values(('Sales', 'sum'), ascending=False).head())

