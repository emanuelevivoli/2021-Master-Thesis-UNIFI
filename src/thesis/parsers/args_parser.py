from simple_parsing import ArgumentParser, ConflictResolution

# Parsers
from thesis.parsers.classes import Args, CallBacks, DataTrainingArguments, EmbeddingArguments, LoggingArguments, ModelArguments, PartialChoice, RunArguments, TrainingArguments, VisualArguments

# Generics
import sys
import os
from typing import Union, List, Dict
import logging
import enum

_factory_args: {
    PartialChoice.DATATRAIN: DataTrainingArguments,
    PartialChoice.TRAINING: TrainingArguments,
    PartialChoice.MODEL: ModelArguments,
    PartialChoice.EMBEDDS: EmbeddingArguments,
    PartialChoice.VISUAL: VisualArguments,
    PartialChoice.RUNS: RunArguments,
    PartialChoice.LOGS: LoggingArguments,
    PartialChoice.ALL: Args,
}


def parse_args(
    args_: Union[List, Dict, str],
) -> Union[Dict]:

    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Let's divide the args_ based on type:
    # - List: list of all arguments
    # - Dict: dict with all arguments
    # - str : name of the file containing the `json` dict
    # - None: arguments are in sys.argv
    if type(args_) is str:
        file_name = args_
        print(
            f"[   Str    ] type(args_) is str: file_name=args_={file_name}")
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.

        parses_args = Args()
        parses_args = parses_args.load(os.path.abspath(file_name))
        print(
            f"[   Str    ] type(args_) is {type(args_) }: parses_args")  # :{parses_args}")

    elif type(args_) is dict:
        print(f"[   Dict   ] type(args_) is dict: args_={args_}")
        # parses_args = parser.parse_args(args_)
        parses_args = Args()
        parses_args = parses_args.from_dict(args_)
        print(
            f"[   Dict   ] type(args_) is {type(args_) }: parses_args")  # :{parses_args}")

    elif (type(args_) is list) or (type(args_) is type(None)):
        parser = ArgumentParser(
            conflict_resolution=ConflictResolution.EXPLICIT,
            add_dest_to_option_strings=True
        )
        parser.add_arguments(DataTrainingArguments, "datatrain")
        parser.add_arguments(TrainingArguments, "training")
        parser.add_arguments(ModelArguments, "model")
        parser.add_arguments(EmbeddingArguments, "embedds")
        parser.add_arguments(VisualArguments, "visual")
        parser.add_arguments(RunArguments, "runs")
        parser.add_arguments(LoggingArguments, "logs")
        # parser = Args()
        if type(args_) is type(None):
            args_ = sys.argv[1:]
        print(
            f"[List | None] type(args_) is {type(args_) }: args_d")  # :{args_}")
        # parses_args = parser.parse_args(args_)
        parses_args = parser.parse_args()
        print(
            f"[List | None] type(args_) is {type(args_) }: parses_args")  # :{parses_args}")
    else:
        print(args_)
        print(type(args_))
        raise ValueError(
            'Args passed to parse_args is neither of type: None, List, Dict, Str')
        parses_args = None

    # Sanity check (List)
    def sanity_check_list(item_type, string):
        if any(map(lambda x: isinstance(x, enum.Enum) or isinstance(x, int) or isinstance(x, list), string)) or \
                not any(map(lambda x: x.find(','), string)):
            app = string
        else:
            def flatten(t): return [item for sublist in t for item in sublist]
            app = string.split(',') if type(string) is str else map(
                lambda x: x.split(','), string)
            app = list(app)
            app = flatten(app)
            app: List[item_type] = list(map(
                lambda x: item_type(x), app))

        return app

    parses_args.visual.fields = sanity_check_list(
        str, parses_args.visual.fields)

    parses_args.datatrain.s2orc.idxs = sanity_check_list(
        int, parses_args.datatrain.s2orc.idxs)

    parses_args.datatrain.s2orc.mag_field_of_study = sanity_check_list(
        str, parses_args.datatrain.s2orc.mag_field_of_study)

    parses_args.datatrain.data = sanity_check_list(
        str, parses_args.datatrain.data)

    parses_args.datatrain.target = sanity_check_list(
        str, parses_args.datatrain.target)

    parses_args.datatrain.classes = sanity_check_list(
        str, parses_args.datatrain.classes)

    parses_args.logs.callbacks = sanity_check_list(
        CallBacks, parses_args.logs.callbacks)

    # Sanity checks
    if (
        parses_args.datatrain.dataset_name is None
        and parses_args.datatrain.train_file is None
        and parses_args.datatrain.validation_file is None
    ):
        raise ValueError(
            "Need either a dataset name or a training/validation file.")
    else:
        if parses_args.datatrain.train_file is not None:
            extension = parses_args.datatrain.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if parses_args.datatrain.validation_file is not None:
            extension = parses_args.datatrain.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    if parses_args.training.output_dir is not None:
        os.makedirs(parses_args.training.output_dir, exist_ok=True)

    return parses_args


def parse_partial(
    partial_selection: PartialChoice,
    partial_args_: Union[List, Dict, str],
) -> Union[Dict]:

    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Let's divide the partial_args_ based on type:
    # - List: list of all arguments
    # - Dict: dict with all arguments
    # - str : name of the file containing the `json` dict
    # - None: arguments are in sys.argv
    if type(partial_args_) is str:
        file_name = partial_args_
        print(
            f"[   Str    ] type(partial_args_) is str: file_name=partial_args_={file_name}")
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.

        parses_args = _factory_args[PartialChoice]()
        parses_args = parses_args.load(os.path.abspath(file_name))
        print(
            f"[   Str    ] type(partial_args_) is {type(partial_args_) }: parses_args:{parses_args}")

    elif type(partial_args_) is dict:
        print(
            f"[   Dict   ] type(partial_args_) is dict: partial_args_={partial_args_}")
        # parses_args = parser.parse_args(partial_args_)
        parses_args = _factory_args[PartialChoice]()
        parses_args = parses_args.from_dict(os.path.abspath(file_name))
        print(
            f"[   Dict   ] type(partial_args_) is {type(partial_args_) }: parses_args:{parses_args}")

    elif (type(partial_args_) is list) or (type(partial_args_) is None):
        parser = ArgumentParser()
        parser.add_arguments(_factory_args[PartialChoice], PartialChoice.value)
        if type(partial_args_) is None:
            partial_args_ = sys.argv
        print(
            f"[List | None] type(partial_args_) is {type(partial_args_) }: partial_args_:{partial_args_}")
        parses_args = parser.parse_args(partial_args_)
        print(
            f"[List | None] type(partial_args_) is {type(partial_args_) }: parses_args:{parses_args}")

    if parses_args.training.output_dir is not None:
        os.makedirs(parses_args.training.output_dir, exist_ok=True)

    return parses_args
