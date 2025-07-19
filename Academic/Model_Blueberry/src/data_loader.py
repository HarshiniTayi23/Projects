import pandas as pd

def load_data(data_path):
    """
    Load the blueberry dataset and perform initial cleaning.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with Row# column removed
    """
    df = pd.read_csv(data_path)
    # Remove the irrelevant Row# column
    df = df.drop(columns='Row#')
    return df