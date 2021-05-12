# 🤗 Datasets
from datasets import concatenate_datasets, DatasetDict, Dataset

# 🤗 Tranformers
from transformers import PreTrainedTokenizer

from typing import List

from .read_dataset import json_journal_read

# Dataset configuration files
from ..config.datasets import JurNLConfig

from ..config.execution import RunConfig, LogConfig

from ..config.base import fingerprints

from ..cache import no_caching, _caching


def journal_loader(
    dataset_config: JurNLConfig,
    run_config: RunConfig,
    log_config: LogConfig,
    *args,
    **kwarg
) -> DatasetDict:
    """ Loader function for the journal dataset. It can load muiltiple datasets, concatenate them and return as one single dataset:
    + Args:
        - dataset_config: `JurNLConfig`, configuration for journal dataset.
        - run_config: `RunConfig`, configuration for running experiments.
        - *args: `args list`, some extra params not used.
        - **kwargs: `kwargs dict`, some extra dictionary params not used.
    + Return:
        - dataset: `DatasetDict`, dictionary with fields `train`, `test`, `valid` and `Dataset` values.
    """

    # For everychunk we get an element composed by 4 elements:
    dataset = json_journal_read(dataset_config, log_config, run_config)

    # TODO #
    def filter_dataset(
        dataset_config: JurNLConfig, log_config: LogConfig, datasets: List[Dataset]
    ):
        # TODO # Filter datasets elements based on some arguments
        # TODO # (mag_field_of_studies or journals)
        return datasets

    # Filter some paper based on specific arguments
    dataset = filter_dataset(dataset_config, log_config, dataset)

    print(dataset)

    return dataset
