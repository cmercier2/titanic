"""
Load, preprocess, prepare, and save the Titanic dataset.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
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
    df_copy = df.copy()
    imputer.fit(df_copy[['Age']])
    df[['Age']] = imputer.transform(df_copy[['Age']])
    df["Embarked"].fillna("S", inplace=True)
    return df


def prepare_data(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: 
    """
    Prepare the Titanic dataset for training.
    
    Args:
        df (DataFrame): The preprocessed Titanic dataset.
        
    Returns:
        tuple: A tuple containing [X,y] the features DataFrame and the target Series.
    """
    numeric_features = ['Age', 'Fare']
    categorical_features =  ["Sex", "Embarked"]

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first').set_output(transform="pandas")

    df_scaled = df.copy()

    df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])
    df_encoded = encoder.fit_transform(df[categorical_features])

    df_final = pd.concat([df_scaled, df_encoded], axis=1).drop(columns=['Sex', 'Embarked'])

    return (df_final.drop(columns=["Survived"]), df_final["Survived"])
