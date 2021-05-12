from dataclasses import field, dataclass
from typing import Optional

# ----------------------------- #
# Definition of Custom Parsers  #
#    - S2orcArguments           #
#    - KeyPhArguments           #
#    - (?) EmbeddingArguments   #
# ----------------------------- #


@dataclass
class S2orcArguments:
    """
    Arguments pertaining to S2ORC dataset, and what data we are going to input our model for training and eval.
    """

    # It can be:
    # - 🤗 HuggingFace dataset from Hub
    # - Custom dataset like: 's2orc', 'keyphrase'
    # ! 'dataset_name'          --> DatasetArguments
    # dataset_name: Optional[str] = field(
    #     default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    # For 's2orc' it is either 'full' or 'sample'
    # For 'keyph' it can be either single choice ('kp20k') or multiple choices ('inspec,krapivin,nus,kp20k')
    # ! 'dataset_config_name'   --> DatasetArguments
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library). option 's2orc' ['full' or 'sample']. option 'keyph' +['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']"}
    # )
    # If there is multiple files for a specific dataset
    idxs: Optional[str] = field(
        default="0",
        metadata={
            "help": "List of chunks indexes. (e.g. --chunk_indexes=0,1,2) [def. 0]."
        },
    )
    zipped: Optional[bool] = field(
        default=True,
        metadata={
            "help": "if False, only idxs for unzipped files. if True, we extract idxs files."
        },
    )
    mag_field_of_study: Optional[str] = field(
        default="",
        metadata={
            "help": "options are empty '' or str(List(string)) like '`Computer Science`;`Mathematics`' [def. '']."
        },
    )
    # preprocessing of data
    keep_none_papers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if False, remove papers with None eather in abstract or title."
        },
    )
    keep_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "if False, remove columns not in dictionary."}
    )

    # as descripted here [https://docs.python.org/3/library/dataclasses.html#mutable-default-values]
    # we need `default_factory=list` option for field to prevent mutable default values
    # for the following elements:
    # - dictionary input default ["data"]
    data: Optional[str] = field(
        # default_factory=lambda: ["abstract"],
        # default=["abstract"],
        default="abstract",
        metadata={
            "help": "The data to token, and then used for training (feedforward)."
        },
    )
    # - dictionary input default ["target"]
    target: Optional[str] = field(
        # default_factory=lambda: ["title"],
        # default=["title"],
        default="title",
        metadata={
            "help": "The target to token, and then used for training (backprop)."
        },
    )
    # - dictionary input default ["mag_field_of_study"]
    classes: Optional[str] = field(
        # default_factory=lambda: ["mag_field_of_study"],
        #  default=["mag_field_of_study"],
        default="mag_field_of_study",
        metadata={"help": "The data field name for labels."},
    )


@dataclass
class KeyPhArguments:
    """
    Arguments pertaining to KeyPhrase dataset, and what data we are going to input our model for training and eval.
    """

    # there is the 'dataset_config_name' variable for that :=)
    # keyph_type: Optional[str] = field(
    #     default=None, metadata={"help": "Type of keyphrases dataset. Vary Options"}
    # )


@dataclass
class EmbeddingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    pooling: str = field(
        default="none",
        metadata={"help": "Pooling method: none, first, mean, sum (default:none)"},
    )
    batch_size: int = field(
        default=32, metadata={"help": "Batch size to embed in batch"}
    )
    # max_seq_length: int = field(
    #     default=512,
    #     metadata={
    #         "help": "The maximum total input sequence length after tokenization. Sequences longer "
    #         "than this will be truncated, sequences shorter will be padded."
    #     },
    # )


@dataclass
class RunArguments:
    """
    Arguments related run configurations.
    """

    # there is the 'run_name' in TrainingArguments for that :=)
    # run_name: Optional[str] = field(
    #     default="s2orc-run", metadata={"help": "string name for the run"}
    # )
    run_number: Optional[int] = field(
        default=False, metadata={"help": "number for that specific run"}
    )
    run_iteration: Optional[int] = field(
        default=False,
        metadata={"help": "iteration number for that specific run number"},
    )


@dataclass
class LoggingArguments:
    """
    Arguments for logging and callbacks management.
    """

    verbose: Optional[bool] = field(
        default=False, metadata={"help": "If True, all verbose logging will be logged"}
    )
    # there is the 'debug' in TrainingArguments for that :=)
    # debug: Optional[bool] = field(
    #     default=False, metadata={"help": "If True, all debug logging will be logged"}
    # )
    time: Optional[bool] = field(
        default=False, metadata={"help": "If True, all timing logging will be logged"}
    )
    callback: Optional[str] = field(
        default="WandbCallback",
        metadata={"help": "options are 'wandb' and 'tensorboard'"},
    )
