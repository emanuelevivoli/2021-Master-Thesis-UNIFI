import os
import logging

from thesis.parsers.args_parser import parse_args
from thesis.utils.load_dataset import load_dataset_wrapper
from thesis.utils.constants import DICTIONARY_FIELD_NAMES


def getting_dataset_splitted(args):

    # ------------------
    # Parsing the Arguments
    # ------------------
    # visual_args
    dataset_args, training_args, model_args, run_args, log_args, embedding_args, visual_args = parse_args(
        args)

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

    return datasets
