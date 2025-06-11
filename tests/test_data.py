import pandas as pd
from titanic.data import load_data, clean_data, prepare_data
from dotenv import load_dotenv

load_dotenv()

def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_clean_data():
    df = load_data()
    df_clean = clean_data(df.copy())
    assert 'Cabin' not in df_clean.columns
    assert df_clean['Age'].isnull().sum() == 0
