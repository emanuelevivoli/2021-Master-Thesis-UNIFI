import os

# RAW_DIR = "data_raw"
# DATASETS_DIR = "datasets"
# TRAINED_MODELS_DIR = "trained_models"
CACHE_DIR = ".cache"

# Path and constants
MAIN_PATH = '/home/vivoli/Thesis'
DATA_PATH = '/home/vivoli/Thesis/data'
OUT_PATH = '/home/vivoli/Thesis/output'
ARGS_PATH = '/home/vivoli/Thesis/args'
ARGS_FILE = 'arguments.json'

DICTIONARY_FIELD_NAMES = dict(
    train=['train'],
    test=['test', 'debug', 'dev'],
    validation=['validation', 'valid']
)

_factory_MODELS = {
    'bert':       'bert-base-uncased',
    'scibert':    'allenai/scibert_scivocab_uncased',
    'paraphrase': 'paraphrase-distilroberta-base-v1',
    'distilbert': 'distilbert-base-nli-mean-tokens',
}
