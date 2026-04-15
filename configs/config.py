import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Preprocessing parameters
MIN_SEQ_LEN = 10
MAX_SEQ_LEN = 50
MMSEQS_SIMILARITY_THRESHOLD = 0.80

# ESM-2 parameters
ESM2_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
MAX_LENGTH = MAX_SEQ_LEN + 2  # Including CLS and EOS tokens

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
RANDOM_SEED = 42

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
