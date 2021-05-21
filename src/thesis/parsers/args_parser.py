# 🤗 Transformers
from transformers import HfArgumentParser

# Parsers
from thesis.parsers.defaults import (
    #  ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
)
from thesis.parsers.customs import (
    EmbeddingArguments,
    LoggingArguments,
    ModelArguments,
    RunArguments,
    VisualArguments
)

# Generics
import sys
import os
from typing import Union, List, Dict
import logging


def parse_args(
    args_: Union[List, Dict, str],
) -> Union[
    DataTrainingArguments,
    TrainingArguments,
    ModelArguments,
    RunArguments,
    LoggingArguments,
    EmbeddingArguments,
    VisualArguments
]:
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (
            DataTrainingArguments,
            TrainingArguments,
            ModelArguments,
            RunArguments,
            LoggingArguments,
            EmbeddingArguments,
            VisualArguments
        )
    )
    # parser = HfArgumentParser((DataTrainingArguments, TrainingArguments, S2orcArguments, KeyPhArguments))

    # Let's divide the args_ based on type:
    # - List: list of all arguments
    # - Dict: dict with all arguments
    # - str : name of the file containing the `json` dict
    # - None: arguments are in sys.argv
    from_file = False
    file_name = None

    if (type(args_) is list) or (type(args_) is None):
        args_ = sys.argv
        print(
            f"type(args_) is {type(args_) }: args_:{args_}")
        (
            dataset_args,
            training_args,
            model_args,
            run_args,
            log_args,
            embedding_args,
            visual_args
        ) = parser.parse_args_into_dataclasses(args_)

    elif type(args_) is dict:
        print(
            f"type(args_) is dict: args_={args_}")
        (
            dataset_args,
            training_args,
            model_args,
            run_args,
            log_args,
            embedding_args,
            visual_args
        ) = parser.parse_dict(args_)

    elif type(args_) is str:
        file_name = args_
        print(
            f"type(args_) is str: from_file={from_file}; file_name=args_={file_name}")
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            dataset_args,
            training_args,
            model_args,
            run_args,
            log_args,
            embedding_args,
            visual_args
        ) = parser.parse_json_file(json_file=os.path.abspath(file_name))

    # Sanity checks
    if (
        dataset_args.dataset_name is None
        and dataset_args.train_file is None
        and dataset_args.validation_file is None
    ):
        raise ValueError(
            "Need either a dataset name or a training/validation file.")
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
        model_args,
        run_args,
        log_args,
        embedding_args,
        visual_args
    )
