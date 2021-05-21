import logging

from datasets.load import load_dataset
from dataclasses import asdict

from thesis.config.datasets import S2orcConfig, KeyPHConfig, JouRNConfig
from thesis.config.execution import RunConfig, LogConfig

from thesis.datasets.s2orc.loader import s2ortc_loader
from thesis.datasets.keyph.loader import keyph_loader
from thesis.datasets.journals.loader import journal_loader

# Parsers
from thesis.parsers.defaults import (
    DataTrainingArguments,
    TrainingArguments,
)
from thesis.parsers.customs import (
    ModelArguments,
    RunArguments,
    LoggingArguments,
    EmbeddingArguments,
)


def load_dataset_wrapper():
    def __call__(
        dataset_args: DataTrainingArguments,
        training_args: TrainingArguments,
        # s2orc_args: S2orcArguments,
        # keyph_args: KeyPhArguments,
        # journ_args: JournArguments,
        model_args: ModelArguments,
        run_args: RunArguments,
        log_args: LoggingArguments,
        embedding_args: EmbeddingArguments,
        *args,
        **kwargs,
    ):

        logging.info(f"The wrapper has started")
        run_config: RunConfig = RunConfig(
            *args, **asdict(run_args), **asdict(training_args), **kwargs
        )
        log_config: LogConfig = LogConfig(
            *args, **asdict(log_args), **asdict(training_args), **kwargs
        )

        if dataset_args.dataset_name == "s2orc":
            logging.info(
                f"S2ORC: depending on configuration, jsonl files are used or jsonl.gz files are extracting end used on the fly."
            )
            # analyze s2orc dataset configuration
            dataset_config: S2orcConfig = S2orcConfig(
                *args, **asdict(dataset_args), **asdict(model_args), **kwargs
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
                *args, **asdict(dataset_args), **asdict(model_args), **kwargs
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
                *args, **asdict(dataset_args), **asdict(model_args), **kwargs
            )

            raw_datasets = journal_loader(
                dataset_config, run_config, log_config, *args, **kwargs
            )

        else:
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

        return raw_datasets

    return __call__
