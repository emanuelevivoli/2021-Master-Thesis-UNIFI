import os
from thesis.datasets.s2orc.mag_field import mag_field_dict
from thesis.datasets.journals.mag_field import ICDAR_field, ICPR_field, ICPR_subfield, IJDAR_field


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

LABEL_DICT = {
    's2orc': mag_field_dict,
    'icdar_19': ICDAR_field,
    'icpr_20': ICPR_field,
    'ijdar_20': IJDAR_field
}

SUBLABEL_DICT = {
    's2orc': None,
    'icdar_19': None,
    'icpr_20': ICPR_subfield,
    'ijdar_20': None
}
