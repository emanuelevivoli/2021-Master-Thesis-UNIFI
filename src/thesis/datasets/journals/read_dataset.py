from tqdm.auto import tqdm  # custom progress bar
import json
import os
from typing import List, Dict

# 🤗 Datasets
from datasets import DatasetDict, Dataset as hfDataset
from datasets.load import load_dataset


from thesis.datasets.journals.preprocessing import get_dataset
from thesis.config.datasets import JouRNConfig
from thesis.config.execution import LogConfig, RunConfig
from thesis.config.base import fingerprints
from thesis.utils.cache import _caching, no_caching

import logging
import pandas as pd


def load_jsonl(dataset_config: JouRNConfig, log_config: LogConfig):
    import json
    import logging

    # just open as usual
    json_path = os.path.join(
        dataset_config.path, f"{dataset_config.journ_type}.jsonl"
    )
    input_json = open(json_path, "r")
    if log_config.verbose:
        logging.info("You choose to only use unzipped files")

    json_list = []
    with input_json:
        for json_line in input_json.readlines():
            json_list.append(json.loads(json_line))

    return json_list


def json_journal_read(dataset_config: JouRNConfig, run_config: RunConfig, log_config: LogConfig):
    """ Reads the journal datasets specified and return a list of Datasets.         \\
    Args:                                                                           \\
        - `dataset_config`: (JouRNConfig), dataset configuration object             \\
        - `log_config`: (LogConfig), logging configuration                          \\
                                                                                    \\
    Return:                                                                         \\
        json_list (list of dict): List of dictionaries, each one with the fields    \\
        - `title` (string)                                                          \\
        - `abstract` (string)                                                       \\
        - `fulltext` (string | '')                                                  \\
        - `keywords` (list) or `custom_field` (str)                                  \
    """
    if log_config.verbose:
        logging.info("[INFO-START] Journal Dataset read")

    # **(dataset_config.get_configuration())
    @no_caching(**fingerprints(dataset_config), function_name="json_journal_read")
    def _json_journal_read(dataset_config: JouRNConfig, run_config: RunConfig, log_config: LogConfig):

        # `dataset_name` from kyphrase dataset is a list of dataset names
        # so we can mix those dataset to create one single dataset
        _dataset_list = load_jsonl(
            dataset_config, log_config)

        _dataset = get_dataset(
            _dataset_list, dataset_config, run_config, log_config)

        return _dataset

    dataset = _json_journal_read(dataset_config, run_config, log_config)

    if log_config.verbose:
        logging.info("[INFO-END] Multi dataset read")

    return dataset
