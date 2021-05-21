from dataclasses import field, dataclass
from typing import Optional

# 🤗 Transformers
from transformers import SchedulerType
from transformers import TrainingArguments as HfTrainingArguments

# ----------------------------- #
# Definition of Custom Parsers  #
#    - DataTrainingArguments    #
#    - TrainingArguments        #
# ----------------------------- #


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default="/home/vivoli/Thesis/data",
        metadata={"help": "Path for custom datasets."},
    )
    # It can be:
    # - 🤗 HuggingFace dataset from Hub
    # - Custom dataset like: 's2orc', 'keyphrase'
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."},
    )
    # For 's2orc' it is either 'full' or 'sample'
    # For 'keyph' it can be either single choice ('kp20k') or multiple choices ('inspec,krapivin,nus,kp20k')
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library). option 's2orc' ['full' or 'sample']. option 'keyph' +['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']"
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    validation_split_percentage: int = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # ! already in ModelArguments
    # model_name_or_path: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Path to pretrained model or model identifier from huggingface.co/models."
    #     },
    # )
    # ! already in ModelArguments
    # config_name: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Pretrained config name or path if not the same as model_name"
    #     },
    # )
    # ! already in ModelArguments
    # tokenizer_name: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Pretrained tokenizer name or path if not the same as model_name."
    #     },
    # )
    use_slow_tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."
        },
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    # def __post_init__(self):
    #     if self.dataset_name is None and self.train_file is None and self.validation_file is None:
    #         raise ValueError(
    #             "Need either a dataset name or a training/validation file.")
    #     else:
    #         if self.train_file is not None:
    #             extension = self.train_file.split(".")[-1]
    #             assert extension in [
    #                 "csv", "json", "jsonl", "txt"], "`train_file` should be a csv, a json, a jsonl or a txt file."
    #         if self.validation_file is not None:
    #             extension = self.validation_file.split(".")[-1]
    #             assert extension in [
    #                 "csv", "json", "json", "txt"], "`validation_file` should be a csv, a json, a jsonl or a txt file."


@dataclass
class TrainingArguments(HfTrainingArguments):
    """
    Arguments for training
    """

    # ------------------------ #
    #        Training           #
    # ------------------------ #
    # do_train: bool = field(default=False, metadata={
    #                     "help": "Whether to run training."})
    # do_eval: bool = field(default=False, metadata={
    #                     "help": "Whether to run eval on the dev set."})
    # do_predict: bool = field(default=False, metadata={
    #     "help": "Whether to run predictions on the test set."})

    per_device_train_batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size (per device) for the training dataloader."},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size (per device) for the evaluation dataloader."},
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={
            "help": "Initial learning rate (after the potential warmup period) to use."
        },
    )
    weight_decay: float = field(default=0.0, metadata={
                                "help": "Weight decay to use."})
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    max_train_steps: int = field(
        default=None,
        metadata={
            "help": "Total number of training steps to perform. If provided, overrides num_train_epochs."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    num_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for the warmup in the lr scheduler."},
    )
    output_dir: str = field(
        default=None, metadata={"help": "Where to store the final model."}
    )
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "If True, it automatically train from scratch (it overwrites the output folder!) ."}
    )
    seed: int = field(
        default=None, metadata={"help": "A seed for reproducible training."}
    )
    # max_seq_length: int = field(
    #     default=None,
    #     metadata={
    #         "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
    #     },
    # )
    # line_by_line: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
    #     },
    # )
    # preprocessing_num_workers: int = field(
    #     default=None,
    #     metadata={"help": "The number of processes to use for the preprocessing."},
    # )
    # overwrite_cache: bool = field(
    #     default=False,
    #     metadata={"help": "Overwrite the cached training and evaluation sets"},
    # )
    # mlm_probability: float = field(
    #     default=0.15,
    #     metadata={
    #         "help": "Ratio of tokens to mask for masked language modeling loss"},
    # )
    logging_dir: str = field(
        default="./logs",
        metadata={
            "help": "Directory for storing logs. TensorBoard log directory. [def. './logs']"
        },
    )

    def __post_init__(self):
        super().__post_init__()
