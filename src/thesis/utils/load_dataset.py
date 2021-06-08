import logging

from thesis.config.datasets import S2orcConfig, KeyPHConfig, JouRNConfig
from thesis.config.execution import RunConfig, LogConfig

from thesis.datasets.s2orc.loader import s2ortc_loader
from thesis.datasets.keyph.loader import keyph_loader
from thesis.datasets.journals.loader import journal_loader

# Parsers
from thesis.parsers.classes import Args
from thesis.parsers.utils import split_args

from thesis.utils.cache import _caching


def custom_load_dataset(all_args: Args, *args, **kwargs):
    dataset_args = split_args(all_args)[0]

    @_caching(
        **dataset_args.to_dict(),
        function_name='custom_load_dataset'
    )
    def _custom_load_dataset(all_args: Args, *args, **kwargs):
        loader = load_dataset_wrapper()
        return loader(all_args, *args, **kwargs)

    return _custom_load_dataset(all_args, *args, **kwargs)


def load_dataset_wrapper():
    def __call__(
        all_args: Args,
        *args,
        **kwargs,
    ):
        dataset_args, training_args, model_args, embedding_args, visual_args, run_args, log_args = split_args(
            all_args)

        logging.info(f"The wrapper has started")
        run_config: RunConfig = RunConfig(
            *args, **run_args.to_dict(), **training_args.to_dict(), **kwargs
        )
        log_config: LogConfig = LogConfig(
            *args, **log_args.to_dict(), **training_args.to_dict(), **kwargs
        )

        if dataset_args.dataset_name == "s2orc":
            logging.info(
                f"S2ORC: depending on configuration, jsonl files are used or jsonl.gz files are extracting end used on the fly."
            )
            # analyze s2orc dataset configuration
            dataset_config: S2orcConfig = S2orcConfig(
                *args, **dataset_args.to_dict(), **model_args.to_dict(), **kwargs
            )

            raw_datasets = s2ortc_loader(
                dataset_config, run_config, log_config, *args, **kwargs
            )

        elif dataset_args.dataset_name == "keyphrase":
            logging.info(
                f"KeyPH: you can choose one or multiple datasets to load all in one."
            )
            # analyze keyph dataset configuration
            dataset_config: KeyPHConfig = KeyPHConfig(
                *args, **dataset_args.to_dict(), **model_args.to_dict(), **kwargs
            )

            raw_datasets = keyph_loader(
                dataset_config, run_config, log_config, *args, **kwargs
            )

        elif dataset_args.dataset_name == "journal":
            logging.info(
                f"JouRN: you can choose one datasets from those pertaining a Journal."
            )
            # analyze s2orc dataset configuration
            dataset_config: JouRNConfig = JouRNConfig(
                *args, **dataset_args.to_dict(), **model_args.to_dict(), **kwargs
            )

            raw_datasets = journal_loader(
                dataset_config, run_config, log_config, *args, **kwargs
            )

        else:
            from datasets.load import load_dataset
            logging.info(
                f"Huggingface: downloading and loading a dataset from the hub."
            )

            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                dataset_args.dataset_name, dataset_args.dataset_config_name
            )
            # print(raw_datasets)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    dataset_args.dataset_name,
                    dataset_args.dataset_config_name,
                    split=f"train[:{dataset_args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    dataset_args.dataset_name,
                    dataset_args.dataset_config_name,
                    split=f"train[{dataset_args.validation_split_percentage}%:]",
                )

        logging.info(f"The wrapper has ended")

        return raw_datasets  # , (dataset_config, run_config, log_config)

    return __call__
