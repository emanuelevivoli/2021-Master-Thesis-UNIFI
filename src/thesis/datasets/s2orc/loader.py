# 🤗 Datasets
from datasets import (
    concatenate_datasets,
    DatasetDict,
)

from thesis.datasets.s2orc.preprocessing import get_dataset

from thesis.datasets.s2orc.read_dataset import s2orc_multichunk_read

# Dataset configuration files
from thesis.config.datasets import S2orcConfig, KeyPHConfig

from thesis.config.execution import RunConfig, LogConfig

from thesis.config.base import fingerprints

from thesis.utils.cache import no_caching, _caching


def s2ortc_loader(
    dataset_config: S2orcConfig,
    run_config: RunConfig,
    log_config: LogConfig,
    *args,
    **kwarg
) -> DatasetDict:
    """
    Args: \\
        - dataset_config: `S2orcConfig`, configuration for s2orc dataset. \\
        - run_config: `RunConfig`, configuration for running experiments. \\
        - *args: `args list`, some extra params not used. \\
        - **kwargs: `kwargs dict`, some extra dictionary params not used. \\
    \\   
    Return: \\
        - all_datasets: `DatasetDict`, dictionary with fields `train`, `test`, `valid` and `Dataset` values. \
    """

    # print(dataset_config)

    toread_meta_s2orc, toread_pdfs_s2orc = dataset_config.memory_save_pipelines(
        log_config.verbose)

    # print(toread_meta_s2orc, toread_pdfs_s2orc)

    # for everychunk we get an element composed by 4 elements:
    multichunks_lists = s2orc_multichunk_read(
        dataset_config, log_config, toread_meta_s2orc, toread_pdfs_s2orc
    )

    # get dictionary input from config
    dictionary_input = dataset_config.get_dictionary_input()
    dictionary_columns = sum(dictionary_input.values(), [])

    # **(dataset_config.get_fingerprint()), **(run_config.get_fingerprint()), **(log_config.get_fingerprint())
    @_caching(
        dictionary_columns,
        **fingerprints(dataset_config, run_config, log_config),
        function_name='s2ortc_loader'
    )
    def custom_to_dataset_list(
        multichunks_lists, dataset_config, run_config, log_config, dictionary_columns
    ):
        return [
            get_dataset(
                single_chunk,
                dataset_config,
                run_config,
                log_config,
                data_field=dictionary_columns,
            )
            for single_chunk in multichunks_lists
        ]

    # for every chunk we fuse and create a dataset
    datasets = custom_to_dataset_list(
        multichunks_lists, dataset_config, run_config, log_config, dictionary_columns
    )

    # print(datasets)

    # concatenation of all dataset to form one single dataset
    all_datasets: DatasetDict = DatasetDict(
        {
            "train": concatenate_datasets([dataset["train"] for dataset in datasets]),
            "test": concatenate_datasets([dataset["test"] for dataset in datasets]),
            "valid": concatenate_datasets([dataset["valid"] for dataset in datasets]),
        }
    )

    return all_datasets
