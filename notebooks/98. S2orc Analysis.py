#!/usr/bin/env python
# coding: utf-8

# # S2orc (exploration, clustering & visualization)
# ---
# ---
# For presenting some results we need to analyze (and rapidly compare) some of the methods we used untill now in order to discriminates between paper's `field_of_study` based on their `title` and `abstract`.
# This notebook is an extention of some previous work done by Master's students from University of Florence (cite here).

# ## Dataset

# From each scientific paper we took the `title` and the `abstract`, as well as a property identifying the field in witch the article pertrains.
# The dataset (only 1000 elements) has been selected randomly from a full-version of 80M papers from different fields.
# The field of studies (that are called in the dataset `mag_field_of_study`) are the following:

# | Field of study | All papers | Full text |
# |----------------|------------|-----------|
# | Medicine       | 12.8M      | 1.8M      |
# | Biology        | 9.6M       | 1.6M      |
# | Chemistry      | 8.7M       | 484k      |
# | n/a            | 7.7M       | 583k      |
# | Engineering    | 6.3M       | 228k      |
# | Comp Sci       | 6.0M       | 580k      |
# | Physics        | 4.9M       | 838k      |
# | Mat Sci        | 4.6M       | 213k      |
# | Math           | 3.9M       | 669k      |
# | Psychology     | 3.4M       | 316k      |
# | Economics      | 2.3M       | 198k      |
# | Poli Sci       | 1.8M       | 69k       |
# | Business       | 1.8M       | 94k       |
# | Geology        | 1.8M       | 115k      |
# | Sociology      | 1.6M       | 93k       |
# | Geography      | 1.4M       | 58k       |
# | Env Sci        | 766k       | 52k       |
# | Art            | 700k       | 16k       |
# | History        | 690k       | 22k       |
# | Philosophy     | 384k       | 15k       |


# Note for reproducibility: `data` is a `DatasetDict` object composed by `Dataset` object for every key (in `train`, `test`, `valid`):

# ```python
# {
#     "train": Dataset,
#     "test" : Dataset,
#     "valid": Dataset
# }
# ```

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import json
from thesis.utils.parsers.args_parser import parse_args
from thesis.utils.general import load_dataset_wrapper
import logging
MAIN_PATH = '/home/vivoli/Thesis'
DATA_PATH = '/home/vivoli/Thesis/data'
OUT_PATH = '/home/vivoli/Thesis/outputs/'
ARGS_PATH = '/home/vivoli/Thesis/'


# In[3]:


# Imports


DICTIONARY_FIELD_NAMES = dict(
    train=['train'],
    test=['test', 'debug', 'dev'],
    validation=['validation', 'valid']
)


# ## Getting the dataset
# ---
# In order to get the dataset we need to create a dictionary with the DatasetArguments (params) and use our "library" called `thesis`.

# In[4]:


# ------------------
# Creating Arguments
# ------------------

# create arguments dictionary
args = dict(

    # nocache = "True",

    # DatasetArguments
    model_name_or_path="allenai/scibert_scivocab_uncased",
    dataset_name="s2orc",  # "keyphrase",
    dataset_config_name="sample",  # "inspec",

    # TrainingArguments
    seed='1234',
    output_dir="/home/vivoli/Thesis/output",

    num_train_epochs='1',
    # 16 and 32 end with "RuntimeError: CUDA out of memory."
    per_device_train_batch_size="8",
    # 16 and 32 end with "RuntimeError: CUDA out of memory."
    per_device_eval_batch_size="8",
    max_seq_length='512',

    # S2orcArguments & KeyPhArguments
    dataset_path="/home/vivoli/Thesis/data",
    data="abstract",
    target="title",
    classes="mag_field_of_study",  # "keywords",

    # S2orcArguments
    idxs='0',
    zipped=True,
    mag_field_of_study='',  # "Computer Science" but we want all
    keep_none_papers=False,
    keep_unused_columns=False,

    # RunArguments
    run_name="scibert-s2orc",
    run_number='0',
    run_iteration='0',

    # LoggingArguments
    verbose=True,
    debug_log=False,
    time=True,
    callbacks="WandbCallback,CometCallback,TensorBoardCallback",
)

# save dictionary to file

ARGS_FILE = 'arguments.json'

with open(os.path.join(ARGS_PATH, ARGS_FILE), 'w') as fp:
    json.dump(args, fp)

print(args)


# In[5]:


# ------------------
# Parsing the Arguments
# ------------------

dataset_args, training_args, model_args, run_args, log_args, embedding_args = parse_args(
    os.path.join(ARGS_PATH, ARGS_FILE))


# In[6]:


# ------------------
# Getting the datasets
# ------------------

# Getting the load_dataset wrapper that manages huggingface dataset and the custom ones
custom_load_dataset = load_dataset_wrapper()
# Loading the raw data based on input (and default) values of arguments
raw_datasets = custom_load_dataset(
    dataset_args, training_args, model_args, run_args, log_args, embedding_args)


# The Datasets in the raw form can have different form of key names (depending on the configuration).
# We need all datasets to contain 'train', 'test', 'validation' keys, if not we change the dictionary keys' name
# based on the `names_tuple` and conseguently on `names_map`.
def format_key_names(raw_datasets):
    # The creation of `names_map` happens to be here
    # For every element in the values lists, one dictionary entry is added
    # with (k,v): k=Value of the list, v=Key such as 'train', etc.
    def names_dict_generator(names_tuple: dict):
        names_map = dict()
        for key, values in names_tuple.items():
            for value in values:
                names_map[value] = key
        return names_map
    names_map = names_dict_generator(DICTIONARY_FIELD_NAMES)
    split_names = raw_datasets.keys()
    for split_name in split_names:
        new_split_name = names_map.get(split_name)
        if split_name != new_split_name:
            raw_datasets[new_split_name] = raw_datasets.pop(split_name)
    return raw_datasets


logging.info(f"Formatting DatasetDict keys")
datasets = format_key_names(raw_datasets)


# In[ ]:
print(datasets)
