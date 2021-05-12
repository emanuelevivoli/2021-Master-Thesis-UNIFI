import pandas as pd

# typing
from typing import Dict, List, Union

# 🤗 Transformers
from transformers import PreTrainedTokenizer

# 🤗 Datasets
from datasets import DatasetDict, Dataset as hfDataset

# torch
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np

from typing import Dict, List

from ..cache import no_caching, _caching

from ..config.base import fingerprints, Config  # , SingleChunk
from ..config.datasets import JurNLConfig
from ..config.execution import RunConfig, LogConfig


def jurnl_convert_to_dataset(log_config: LogConfig, datasets_dict: dict):
    # Load the Dictionary
    _dataset = pd.DataFrame(datasets_dict)
    # Convert `list` to `Dataset` with `Dataset.from_pandas`
    dataset_ = hfDataset.from_pandas(_dataset)
    return dataset_


def get_dataset(
    dataset: List,
    dataset_config: JurNLConfig,
    run_config: RunConfig,
    log_config: LogConfig,
) -> Dict[str, DataLoader]:
    """Given an input file, prepare the train, test, validation dataloaders.
    :param single_chunk: `SingleChunk`, input file related to one chunk (format list)
    :param dataset_config: `JurNLConfig`, pretrained tokenizer that will prepare the data, i.e. convert tokens into IDs
    :param run_config: `RunConfig`, if set, seed for split train/val/test
    :param log_config: `LogConfig`, batch size for the dataloaders
    :return: a dictionary containing train, test, validation dataloaders
    """
    # **(dataset_config.get_fingerprint()), **(run_config.get_fingerprint()), **(log_config.get_fingerprint())
    @_caching(
        key_value_sort(single_chunk["meta_key_idx"]),
        key_value_sort(single_chunk["pdf_key_idx"]),
        **fingerprints(dataset_config, run_config, log_config),
        function_name="get_dataset",
    )
    def _get_dataset(
        dataset: List,
        dataset_config: JurNLConfig,
        run_config: RunConfig,
        log_config: LogConfig,
    ) -> DatasetDict:

        ## ------------------ ##
        ## -- LOAD DATASET -- ##
        ## ------------------ ##
        if log_config.time:
            start = time.time()
        if log_config.time:
            start_load = time.time()

        # execution
        dataset_dict = jurnl_convert_to_dataset(log_config, dataset)

        #  print(dataset_dict)

        if log_config.debug:
            print(dataset_dict)

        if log_config.time:
            end_load = time.time()
        if log_config.time:
            print(f"[TIME] load_dataset: {end_load - start_load}")

        ## ------------------ ##
        ## ---- MANAGING ---- ##
        ## ------------------ ##
        if log_config.time:
            start_selection = time.time()

        # execution
        dataset = dataset_dict  # ['train']

        if log_config.time:
            end_selection = time.time()
        if log_config.time:
            print(
                f"[TIME] dataset_train selection: {end_selection - start_selection}")
        if log_config.debug:
            print(dataset)

        ## ------------------ ##
        ## --- SPLIT 1.    -- ##
        ## ------------------ ##
        if log_config.time:
            start_first_split = time.time()

        # 80% (train), 20% (test + validation)
        # execution
        train_testvalid = dataset.train_test_split(
            test_size=0.2, seed=run_config.seed)

        if log_config.time:
            end_first_split = time.time()
        if log_config.time:
            print(
                f"[TIME] first [train-(test-val)] split: {end_first_split - start_first_split}"
            )

        ## ------------------ ##
        ## --- SPLIT 2.    -- ##
        ## ------------------ ##
        if log_config.time:
            start_second_split = time.time()

        # 10% of total (test), 10% of total (validation)
        # execution
        test_valid = train_testvalid["test"].train_test_split(
            test_size=0.5, seed=run_config.seed
        )

        if log_config.time:
            end_second_split = time.time()
        if log_config.time:
            print(
                f"[TIME] second [test-val] split: {end_second_split - start_second_split}"
            )

        # execution
        dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "test": test_valid["test"],
                "valid": test_valid["train"],
            }
        )
        if log_config.time:
            end = time.time()
        if log_config.time:
            print(f"[TIME] TOTAL: {end - start}")

        return dataset

    dataset = _get_dataset(
        dataset, dataset_config, run_config, log_config
    )

    return dataset
