"""

"""

from titanic.data import load_data,clean_data,prepare_data
from titanic.registry import save_model
from titanic.train import train_model, evaluate_model, optimize_model

from dotenv import load_dotenv
import os

load_dotenv()  # Charge les variables d'environnement depuis .env

data_dir = os.environ.get("DATA_DIR")
models_dir = os.environ.get("MODELS_DIR")

print(f"DATA_DIR = {data_dir}")
print(f"MODELS_DIR = {models_dir}")
