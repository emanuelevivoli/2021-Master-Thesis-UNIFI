# 🤗 Transformers
from transformers import HfArgumentParser

# Parsers
from .defaults import (
    #  ModelArguments,
    DatasetArguments,
    TrainingArguments,
)
from .customs import (
    S2orcArguments,
    KeyPhArguments,
    RunArguments,
    LoggingArguments,
    EmbeddingArguments,
)

# Generics
import sys
import os
from typing import Union, List


def parse_args(
    args_list: List,
) -> Union[
    DatasetArguments,
    TrainingArguments,
    S2orcArguments,
    KeyPhArguments,
    RunArguments,
    LoggingArguments,
    EmbeddingArguments,
]:
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (
            DatasetArguments,
            TrainingArguments,
            S2orcArguments,
            KeyPhArguments,
            RunArguments,
            LoggingArguments,
            EmbeddingArguments,
        )
    )
    # parser = HfArgumentParser((DatasetArguments, TrainingArguments, S2orcArguments, KeyPhArguments))

    #  if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    if (
        len(args_list) == 2
        and sys.argv[0] == "params_path"
        and sys.argv[1].endswith(".json")
    ):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            dataset_args,
            training_args,
            s2orc_args,
            keyph_args,
            run_args,
            log_args,
            embedding_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            dataset_args,
            training_args,
            s2orc_args,
            keyph_args,
            run_args,
            log_args,
            embedding_args,
        ) = parser.parse_args_into_dataclasses(args_list)

    # Sanity checks
    if (
        dataset_args.dataset_name is None
        and dataset_args.train_file is None
        and dataset_args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if dataset_args.train_file is not None:
            extension = dataset_args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if dataset_args.validation_file is not None:
            extension = dataset_args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    if training_args.output_dir is not None:
        os.makedirs(training_args.output_dir, exist_ok=True)

    return (
        dataset_args,
        training_args,
        s2orc_args,
        keyph_args,
        run_args,
        log_args,
        embedding_args,
    )
