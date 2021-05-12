# 🤗 Datasets
from datasets import concatenate_datasets, DatasetDict, Dataset

# 🤗 Tranformers
from transformers import PreTrainedTokenizer

from typing import List

from .read_dataset import json_keyphrase_read

# Dataset configuration files
from ..config.datasets import KeyPHConfig

from ..config.execution import RunConfig, LogConfig

from ..config.base import fingerprints

from ..cache import no_caching, _caching


def keyph_loader(
    dataset_config: KeyPHConfig,
    run_config: RunConfig,
    log_config: LogConfig,
    *args,
    **kwarg
) -> DatasetDict:
    """ Loader function for the Keyphrase dataset. It can load muiltiple datasets, concatenate them and return as one single dataset:
    + Args:
        - dataset_config: `KeyPHConfig`, configuration for keyPhrase dataset.
        - run_config: `RunConfig`, configuration for running experiments.
        - *args: `args list`, some extra params not used.
        - **kwargs: `kwargs dict`, some extra dictionary params not used.
    + Return:
        - all_datasets: `DatasetDict`, dictionary with fields `train`, `test`, `valid` and `Dataset` values.
    """

    # For everychunk we get an element composed by 4 elements:
    datasets = json_keyphrase_read(dataset_config, log_config)

    # TODO #
    def filter_dataset(
        dataset_config: KeyPHConfig, log_config: LogConfig, datasets: List[Dataset]
    ):
        # TODO # Filter datasets elements based on some arguments
        # TODO # (mag_field_of_studies or keyphrases)
        return datasets

    # Filter some paper based on specific arguments
    datasets = filter_dataset(dataset_config, log_config, datasets)

    print(datasets)

    # Concatenation of all dataset to form one single dataset
    all_datasets: DatasetDict = DatasetDict(
        {
            "train": concatenate_datasets([dataset["train"] for dataset in datasets]),
            "test": concatenate_datasets([dataset["test"] for dataset in datasets]),
            "valid": concatenate_datasets([dataset["valid"] for dataset in datasets]),
        }
    )

    return all_datasets
