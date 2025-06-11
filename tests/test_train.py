import pytest
from titanic.data import load_data, clean_data, prepare_data
from titanic.train import train_model, evaluate_model

def test_train_model():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = prepare_data(df)
    model, y_pred = train_model(X_train, y_train, X_test)
    assert len(y_pred) == len(y_test)
    assert hasattr(model, "predict")

def test_evaluate_model(capsys):
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = prepare_data(df)
    model, y_pred = train_model(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred)
    captured = capsys.readouterr()
    assert "Accuracy" in captured.out
