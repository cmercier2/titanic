import os
import tempfile
from titanic.data import load_data, clean_data, prepare_data
from titanic.train import train_model
from titanic.registry import save_model, load_model


def test_save_and_load_model():
    df = clean_data(load_data())
    X_train, X_test, y_train, y_test = prepare_data(df)
    model, _ = train_model(X_train, y_train, X_test)

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ["MODELS_DIR"] = tmp_dir
        save_model(model, "test_model.pkl")
        loaded_model = load_model("test_model.pkl")
        assert hasattr(loaded_model, "predict")
