"""
Save and load models, preprocessors
"""

import pickle, os


def save_model(model, path: str):
    if not os.path.exists("models"):
        os.makedirs("models")

    with open("models/" + path, "wb") as f:
        pickle.dump(model, f)


    """
    Save a model to the specified path.
    
    Args:
        model: The model to save.
        path (str): The file path where the model will be saved.
    """
    pass

def load_model(path: str):
    """
    Load a model from the specified path.
    
    Args:
        path (str): The file path from which the model will be loaded.
        
    Returns:
        The loaded model.
    """
    if not os.path.exists(path):
        return
    with open("model/" + path, r) as f:
        pickle.load(f)