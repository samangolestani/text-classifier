# config.py

# Dataset
DATASET_PATH = 'data/synthetic_dataset.csv'

# Model save path
MODEL_PATH = 'models/model.h5'

# Tokenizer and vocab settings
VOCAB_SIZE = 1000
OOV_TOKEN = "<OOV>"

# Training settings
EPOCHS = 20
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 16