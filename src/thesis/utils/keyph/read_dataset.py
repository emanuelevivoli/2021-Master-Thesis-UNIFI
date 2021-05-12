from tqdm.auto import tqdm  # custom progress bar
import json
import os
from typing import List, Dict

# 🤗 Datasets
from datasets import DatasetDict
from datasets.load import load_dataset

from ..config.datasets import KeyPHConfig
from ..config.execution import LogConfig
from ..config.base import fingerprints
from ..cache import _caching

import logging
import pandas as pd

from datasets import Dataset as hfDataset


def read_json_file(file_json_path: str, keyph_type: str):

    json_list_of_dict = []
    with open(file_json_path, "r") as input_json:
        for json_line in input_json:
            json_dict = json.loads(json_line)

            if keyph_type == "stackexchange":
                json_dict["abstract"] = json_dict["question"]
                json_dict["keywords"] = json_dict["tags"]
                del json_dict["question"]
                del json_dict["tags"]

            keywords = json_dict["keywords"]

            if isinstance(keywords, str):
                keywords = keywords.split(";")
                json_dict["keywords"] = keywords

            json_list_of_dict.append(json_dict)

    return json_list_of_dict


def keyph_dataset_read(
    dataset_config: KeyPHConfig, log_config: LogConfig, keyph_type: str
):

    if log_config.verbose:
        logging.info(f"[INFO] Dataset reading  : {keyph_type}")

    # `file_name` is not necessary as we need all the train/valid/test
    # files for a dataset to be called as `{dataset_name}_train/valid/test`
    _datasets_dict: dict = {}
    splits: List[str] = dataset_config.get_splits_by_key(keyph_type)

    for split in splits:
        if log_config.verbose:
            logging.info(f"This dataset has {split} split")
        json_path = os.path.join(
            dataset_config.path, keyph_type, f"{keyph_type}_{split}.json"
        )
        json_list_of_dict = read_json_file(json_path, keyph_type)
        _datasets_dict[split] = json_list_of_dict

    # For splits that doesn't exists, just add empty Dataset
    for no_split in set(['train', 'test', 'valid']).difference(set(splits)):
        if log_config.verbose:
            logging.info(f"This dataset has NOT {no_split} split")
        _datasets_dict[no_split] = []
    return _datasets_dict


def keyph_convert_to_dataset(log_config: LogConfig, datasets_dict: dict):
    # Create an empty Dictionary
    dataset: DatasetDict = dict()
    # Fill the dictionary
    for _split, _dataset in datasets_dict.items():
        _dataset = pd.DataFrame(_dataset)
        # Convert `list` to `Dataset` with `Dataset.from_pandas`
        dataset_ = hfDataset.from_pandas(_dataset)
        dataset[_split] = dataset_
    # Return the DatasetDict with 'train', 'test', 'valid' Dictionary (empty are allowed)
    return dataset


def json_keyphrase_read(dataset_config: KeyPHConfig, log_config: LogConfig):
    """ Reads the keyphrases datasets specified and return a list of Datasets.      \\
    Args:                                                                           \\
        - `dataset_config`: (KeyPHConfig), dataset configuration object             \\
        - `log_config`: (LogConfig), logging configuration                          \\
                                                                                    \\
    Return:                                                                         \\
        json_list (list of dict): List of dictionaries, each one with the fields    \\
        - `title` (string)                                                          \\
        - `abstract` (string)                                                       \\
        - `fulltext` (string | '')                                                  \\
        - `keywords` (list)                                                         \
    """
    if log_config.verbose:
        logging.info("[INFO-START] Multi dataset read")

    # **(dataset_config.get_configuration())
    @_caching(**fingerprints(dataset_config), function_name="json_keyphrase_read")
    def _json_keyphrase_read(dataset_config: KeyPHConfig, log_config: LogConfig):

        multidatasets_lists: List[dict] = []

        for dataset in tqdm(dataset_config.keyph_type):

            # `dataset_name` from kyphrase dataset is a list of dataset names
            # so we can mix those dataset to create one single dataset
            _dataset_dict = keyph_dataset_read(
                dataset_config, log_config, dataset)

            _dataset = keyph_convert_to_dataset(
                log_config, _dataset_dict)

            multidatasets_lists.append(_dataset)

        return multidatasets_lists

    multidatasets_lists = _json_keyphrase_read(dataset_config, log_config)

    if log_config.verbose:
        logging.info("[INFO-END] Multi dataset read")

    return multidatasets_lists
