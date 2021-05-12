import logging
from .config.datasets import S2orcConfig, KeyPHConfig, JurNLConfig
from .config.execution import RunConfig, LogConfig
from datasets.load import load_dataset
from .s2orc.loader import s2ortc_loader
from .keyph.loader import keyph_loader
from .journals.loader import journal_loader
from dataclasses import asdict

# Parsers
from .parsers.defaults import (
    #  ModelArguments,
    DatasetArguments,
    TrainingArguments,
)
from .parsers.customs import (
    S2orcArguments,
    KeyPhArguments,
    RunArguments,
    LoggingArguments,
    EmbeddingArguments,
)


def range_from_N(s2orc_type, _n, _to, _into):
    if s2orc_type == "sample":

        if _n is not None:
            logging.warning(
                f"You set 'sample' but you also set `N` for full bucket range. \n The N selection will be discarded as only `sample` element will be used."
            )
            _n = 0
            list_range = [_n]

    elif s2orc_type == "full":

        if _n is None:
            logging.warning(
                f"You set 'full' but no bucket index was specified. \n We'll use the index 0, so the first bucket will be used."
            )
            _n = 0
            list_range = [_n]

        elif type(_n) is list:
            if _into:
                list_range = range(_n[0], _n[1])
                logging.warning(
                    f"The range is intended as [{_n[0]}, {_n[1]}] (start {_n[0]}, end {_n[1]})"
                )
            else:
                list_range = _n
                logging.warning(f"The element list is intended as: {_n}")

        elif type(_n) is int:
            if _to:
                list_range = range(0, _n)
                logging.warning(
                    f"The range is intended as [ 0, {_n}] (start 0, end {_n})"
                )
            else:
                list_range = [_n]
                logging.warning(f"The element list is intended as: [{_n}]")

    else:
        raise NameError(
            f"You must select an existed S2ORC dataset \n \
                    You selected {s2orc_type}, but options are ['sample' or 'full']"
        )

    return list_range


def load_dataset_wrapper():
    def __call__(
        dataset_args: DatasetArguments,
        training_args: TrainingArguments,
        s2orc_args: S2orcArguments,
        keyph_args: KeyPhArguments,
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
                *args, **asdict(dataset_args), **asdict(s2orc_args), **kwargs
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
                *args, **asdict(dataset_args), **asdict(keyph_args), **kwargs
            )

            raw_datasets = keyph_loader(
                dataset_config, run_config, log_config, *args, **kwargs
            )

        elif dataset_args.dataset_name == "journal":
            logging.info(
                f"JurNL: you can choose one datasets from those pertaining a Journal."
            )
            # analyze s2orc dataset configuration
            dataset_config: JurNLConfig = JurNLConfig(
                *args, **asdict(dataset_args), **asdict(s2orc_args), **kwargs
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
            print(raw_datasets)
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
