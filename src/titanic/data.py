"""
Load, preprocess, prepare, and save the Titanic dataset.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
import os

def load_data(file_name: str):
    """
    Load the Titanic dataset from a CSV file.
    
    Returns:
        DataFrame: The loaded Titanic dataset.
    """
    DATA_DIR = os.environ.get("DATA_DIR")

    df  = pd.read_csv(os.path.join(DATA_DIR, file_name + ".csv"),index_col=0)
    return df
    

       
def clean_data(df):
    """
    clean the Titanic dataset.
    
    Args:
        df (DataFrame): The Titanic dataset.
        
    Returns:
        DataFrame: The preprocessed Titanic dataset.
    """
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    imputer = SimpleImputer().set_output(transform="pandas")
    imputer.fit(train_df[['Age']])
    train_df[['Age']] = imputer.transform(train_df[['Age']])


def prepare_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: 
    """
    Prepare the Titanic dataset for training.
    
    Args:
        df (DataFrame): The preprocessed Titanic dataset.
        
    Returns:
        tuple: A tuple containing [X,y] the features DataFrame and the target Series.
    """
    pass