"""
Load, preprocess, prepare, and save the Titanic dataset.
"""

import pandas as pd
import os

from sklearn.calibration import LabelEncoder

def load_data():
    """
    Load the Titanic dataset from a CSV file.
    
    Returns:
        DataFrame: The loaded Titanic dataset.
    """
    DATA_DIR = os.environ.get("DATA_DIR")
    if DATA_DIR is None:
        raise ValueError("DATA_DIR is not defined.")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR,"train.csv"),index_col=0)
    return train_df

       
def clean_data(df):
    """
    clean the Titanic dataset.
    
    Args:
        df (DataFrame): The Titanic dataset.
        
    Returns:
        DataFrame: The preprocessed Titanic dataset.
    """
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    for col in ['Sex', 'Embarked']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


def prepare_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: 
    """
    Prepare the Titanic dataset for training.
    
    Args:
        df (DataFrame): The preprocessed Titanic dataset.
        
    Returns:
        tuple: A tuple containing [X,y] the features DataFrame and the target Series.
    """
    pass