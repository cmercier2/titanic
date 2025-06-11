"""
Load, preprocess, prepare, and split the Titanic dataset.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(data_dir=None) -> pd.DataFrame:
    """
    Load the Titanic dataset from a CSV file.

    Args:
        data_dir (str, optional): Path to the data directory. 
                                  If None, uses the DATA_DIR environment variable.

    Returns:
        pd.DataFrame: Loaded Titanic dataset.

    Raises:
        ValueError: If DATA_DIR is not defined.
        FileNotFoundError: If train.csv is not found.
    """
    data_dir = data_dir or os.environ.get("DATA_DIR")
    if not data_dir:
        raise ValueError("DATA_DIR is not defined (environment variable or argument).")
    
    csv_path = os.path.join(data_dir, "train.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No such file: {csv_path}")
    
    return pd.read_csv(csv_path, index_col=0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the Titanic dataset.

    Args:
        df (pd.DataFrame): Raw Titanic dataset.

    Returns:
        pd.DataFrame: Cleaned and encoded Titanic dataset.
    """
    df = df.copy()
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')
    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df


def prepare_data(df: pd.DataFrame):
    """
    Split the Titanic dataset into training and test sets.

    Args:
        df (pd.DataFrame): Cleaned dataset including the 'Survived' column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)
