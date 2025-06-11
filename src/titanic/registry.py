"""
Save and load models, preprocessors
"""

import pickle, os


def save_model(model, modelName: str):
    """
    Save a model to the specified path.
    
    Args:
        model: The model to save.
        path (str): The file path where the model will be saved.
    """
    models_dir = models_dir or os.environ.get("MODELS_DIR")
    if not models_dir:
        raise ValueError("MODELS_DIR is not defined (environment variable or argument).")
    
    full_path = os.path.join(models_dir, modelName)
    with open(full_path, "wb") as f:
        pickle.dump(model,f)

def load_model(path: str):
    """
    Load a model from the specified path.
    
    Args:
        path (str): The file path from which the model will be loaded.
        
    Returns:
        The loaded model.
    """
    pass