from simple_parsing.helpers import list_field
from simple_parsing import field
from simple_parsing.helpers import Serializable, encode

from datetime import datetime
import enum
from dataclasses import field, dataclass, fields
from typing import Optional, TypeVar, List, Union, Type, Dict
# from transformers.trainer_utils import IntervalStrategy

# 🤗 Transformers
from transformers import (
    # SchedulerType,
    TrainingArguments as HfTrainingArguments,
    MODEL_FOR_MASKED_LM_MAPPING
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
DIM_REDUCTION = ['none', 'umap', 'pca', 'tsne']


# ----------------------------- ----------------------------- #
# Definition of Custom Parsers
#    - DataTrainingArguments (HfTrainingArguments)
#        - S2orcArguments
#        - KeyPhArguments
#        - JournArguments
#    - ModelArguments
#    - EmbeddingArguments
#    - DimRedArguments
#        - UMAPArguments
#        - TSNEArguments
#        - PCAArguments
#    - ClustArguments
#        - KMEANSArguments
#        - HDBSCANArguments
#        - HIERARCHICALArguments
#    - VisualArguments
#        - DimRedArguments
#        - ClustArguments
#    - RunArguments
#    - LoggingArguments
# ----------------------------- ----------------------------- #


@dataclass
class S2orcArguments(Serializable):
    """
    Arguments pertaining to S2ORC dataset, and what data we are going to input our model for training and eval.
    """

    idxs: Optional[List[int]] = list_field(
        0,  # choices=[i for i in range(100)],
        metadata={
            "help": "options are from 0 to 99"
            "List of chunks indexes. (e.g. --chunk_indexes= 0 1 2)"
        },
    )
    zipped: Optional[bool] = field(
        default=True,
        metadata={
            "help": "if False, only idxs for unzipped files. if True, we extract idxs files."
        },
    )
    mag_field_of_study: Optional[List[str]] = list_field(
        # default=[],
        metadata={
            "help": "options are empty '' or str(List(string)) like '`Computer Science`;`Mathematics`'"
        },
    )
    keep_none_papers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if False, remove papers with None eather in abstract or title."
        },
    )
    keep_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if False, remove columns not in dictionary."
        }
    )


@dataclass
class KeyPhArguments(Serializable):
    """
    Arguments pertaining to KeyPhrase dataset, and what data we are going to input our model for training and eval.
    """

    # there is the 'dataset_config_name' variable for that :=)
    # keyph_type: Optional[str] = field(
    #     default=None, metadata={"help": "Type of keyphrases dataset. Vary Options"}
    # )


@dataclass
class JournArguments(Serializable):
    """
    Arguments pertaining to (custom) Journal dataset, and what data we are going to input our model for training and eval.
    """

    # there is the 'dataset_config_name' variable for that :=)
    # keyph_type: Optional[str] = field(
    #     default=None, metadata={"help": "Type of keyphrases dataset. Vary Options"}
    # )


@dataclass
class DataTrainingArguments(Serializable):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default="/home/vivoli/Thesis/data",
        metadata={
            "help": "Path for custom datasets."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
            " - 🤗 HuggingFace dataset from Hub"
            " - Custom dataset like: 's2orc', 'keyphrase'"
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
            " - 's2orc' (exclusive) ['full' or 'sample'];"
            " - 'keyph' (not exclusive) ['inspec', 'krapivin', 'nus', 'semeval', 'kp20k', 'duc', 'stackexchange']"
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the training data."
        },
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "A csv or a json file containing the validation data."
        },
    )
    validation_split_percentage: int = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    no_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether to avoid loading previously created results. "
            "If False, will load the cached file, if exists; if True won't load anything at all."
        },
    )
    no_splits: bool = field(
        default=False,
        metadata={
            "help": "Whether to avoid splitting in train/test/valid. "
            "If False, will split the samples in those 3 'default' splits; if True won't split at all."
        },
    )
    data: List[str] = list_field(
        "abstract",
        metadata={
            "help": "The data to token, and then used for training (feedforward)."
        },
    )
    target: List[str] = list_field(
        "title",
        metadata={
            "help": "The target to token, and then used for training (backprop)."
        },
    )
    classes: List[str] = list_field(
        "mag_field_of_study",
        metadata={
            "help": "The data field name for labels."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
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
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask for masked language modeling loss"
        }
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

    # s2orc: TypeVar['Dataclass'] = S2orcArguments
    # keyph: TypeVar['Dataclass'] = KeyPhArguments
    # journ: TypeVar['Dataclass'] = JournArguments

    s2orc: S2orcArguments = S2orcArguments()
    keyph: KeyPhArguments = KeyPhArguments()
    journ: JournArguments = JournArguments()

    """
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "jsonl", "txt"], "`train_file` should be a csv, a json, a jsonl or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json", "json", "txt"], "`validation_file` should be a csv, a json, a jsonl or a txt file."
    """


@dataclass
class TrainingArguments(Serializable):  # (HfTrainingArguments):
    """
    Arguments for training
    """

    seed: int = field(
        default=42, metadata={"help": "A seed for reproducible training."}
    )
    do_train: bool = field(
        default=False,
        metadata={
            "help": "Whether to run training."
        }
    )
    do_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to run eval on the dev set."
        }
    )
    do_predict: bool = field(
        default=False,
        metadata={
            "help": "Whether to run predictions on the test set."
        }
    )
    output_dir: str = field(
        default='outputs',
        metadata={
            "help": "Where to store the final model."
        }
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "If True, it automatically train from scratch (it overwrites the output folder!) ."
        }
    )
    # no_cache: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to avoid loading previously created results. "
    #         "If False, will load the cached file, if exists; if True won't load anything at all."
    #     },
    # )
    num_train_epochs: int = field(
        default=3,
        metadata={
            "help": "Total number of training epochs to perform."
        }
    )
    max_train_steps: int = field(
        default=5,
        metadata={
            "help": "Total number of training steps to perform. If provided, overrides num_train_epochs."
        },
    )
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
    weight_decay: float = field(
        default=0.0,
        metadata={
            "help": "Weight decay to use."
        }
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    # lr_scheduler_type: SchedulerType = field(
    #     default='linear',
    #     metadata={
    #         "help": "The scheduler type to use."
    #         "options [linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup]"
    #     },
    # )
    num_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of steps for the warmup in the lr scheduler."
        },
    )
    logging_dir: str = field(
        default="./logs",
        metadata={
            "help": "Directory for storing logs. TensorBoard log directory."
        },
    )
    eval_steps: int = field(
        default=100,
        metadata={
            "help": "Run an evaluation every X steps."
        }
    )

    # def __post_init__(self):
    #     super().__post_init__()
    def to_dict(self, discard: List[str] = list([])) -> Dict:
        partial_dict = super().to_dict()
        for key in discard:
            try:
                del partial_dict[key]
            except:
                print(f"Error: key {key} doesn't exists")
                continue

        return partial_dict


@dataclass
class ModelArguments(Serializable):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='allenai/scibert_scivocab_uncased',
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list:"
            ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    # no_cache: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to avoid loading previously created results. "
    #         "If False, will load the cached file, if exists; if True won't load anything at all."
    #     },
    # )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class EmbeddingArguments(Serializable):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # no_cache: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to avoid loading previously created results. "
    #         "If False, will load the cached file, if exists; if True won't load anything at all."
    #     },
    # )
    pooling: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pooling method: none, first, mean, sum (default:none)"
        },
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to embed in batch"
        }
    )


# class DimRedChoice(enum.Enum):
#     UMAP = 'UMAP'
#     TSNE = 'TSNE'
#     PCA = 'PCA'


# class ClustChoice(enum.Enum):
#     KMEANS = 'KMEANS'
#     HDBSCAN = 'HDBSCAN'
#     HIERARCHICAL = 'HIERARCHICAL'


@dataclass
class UMAPArguments(Serializable):
    """
    Arguments pertaining the UMAP dimentionality reduction algorith
    """

    n_neighbors: int = 15
    metric: str = 'cosine'
    n_components: int = 50
    min_dist: float = 0.0


@dataclass
class TSNEArguments(Serializable):
    """
    Arguments pertaining the TSNE dimentionality reduction algorith
    """

    metric: str = 'cosine'
    n_components: int = 15
    perplexity: int = 50


@dataclass
class PCAArguments(Serializable):
    """
    Arguments pertaining the PCA dimentionality reduction algorith
    """

    n_components: int = 15


@dataclass
class DimRedArguments(Serializable):
    """
    Arguments pertaining the dimentionality reduction algorith for pre/post processing
    """

    choice: str = 'UMAP'  # DimRedChoice = DimRedChoice.UMAP

    umap: UMAPArguments = UMAPArguments()
    tsne: TSNEArguments = TSNEArguments()
    pca: PCAArguments = PCAArguments()


@dataclass
class KMEANSArguments(Serializable):
    """
    Arguments pertaining the KMEANS dimentionality reduction algorith
    """

    n_clusters: int = 10
    n_init: int = 10
    max_iter: int = 300


@dataclass
class HDBSCANArguments(Serializable):
    """
    Arguments pertaining the HDBSCANA dimentionality reduction algorith
    """

    min_cluster_size: int = 5
    metric: str = 'euclidean'
    cluster_selection_method: str = 'eom'


@dataclass
class HIERARCHICALArguments(Serializable):
    """
    Arguments pertaining the HIERARCHICAL dimentionality reduction algorith
    """

    affinity: str = 'euclidean'
    linkage: str = 'ward'


@dataclass
class ClustArguments(Serializable):
    """
    Arguments pertaining the dimentionality reduction algorith for pre/post processing
    """

    choice: str = 'KMEANS'   # ClustChoice = ClustChoice.KMEANS

    kmeans: KMEANSArguments = KMEANSArguments()
    hdbscan: HDBSCANArguments = HDBSCANArguments()
    hierarchical: HIERARCHICALArguments = HIERARCHICALArguments()


@dataclass
class VisualArguments(Serializable):
    """
    Arguments pertaining to how we want to visualize data (embeddings).
    """
    fields: List[str] = list_field(
        ['abstract'],
        metadata={
            "help": "Field from dataset to use for embedding generation"
        },
    )
    # no_cache: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to avoid loading previously created results. "
    #         "If False, will load the cached file, if exists; if True won't load anything at all."
    #     },
    # )
    pre: DimRedArguments = DimRedArguments()

    clust: ClustArguments = ClustArguments()

    post: DimRedArguments = DimRedArguments()


@dataclass
class RunArguments(Serializable):
    """
    Arguments related run configurations.
    """
    # no_cache: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to avoid loading previously created results. "
    #         "If False, will load the cached file, if exists; if True won't load anything at all."
    #     },
    # )
    run_name: Optional[str] = field(
        default=f"s2orc-run-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}",
        metadata={
            "help": "string name for the run"
            "there is the `run_name` in TrainingArguments for that 😀 "
        }
    )
    run_number: Optional[int] = field(
        default=0,
        metadata={
            "help": "number for that specific run"
        }
    )
    run_iteration: Optional[int] = field(
        default=0,
        metadata={
            "help": "iteration number for that specific run number"
        },
    )

    def to_dict(self, discard: List[str] = list([])) -> Dict:
        partial_dict = super().to_dict()
        for key in discard:
            try:
                del partial_dict[key]
            except:
                print(f"Error: key {key} doesn't exists")
                continue

        return partial_dict


class CallBacks(enum.Enum):
    wandb = 'WandbCallback'
    comet = 'CometCallback'
    tensb = 'TensorBoardCallback'

    def __call__(self, string):
        return self[string]


@dataclass
class LoggingArguments(Serializable):
    """
    Arguments for logging and callbacks management.
    """

    verbose: Optional[bool] = field(
        default=False, metadata={"help": "If True, all verbose logging will be logged"}
    )
    # there is the 'debug' in TrainingArguments for that :=)
    debug_log: Optional[bool] = field(
        default=False, metadata={"help": "If True, all debug logging will be logged"}
    )
    time: Optional[bool] = field(
        default=False, metadata={"help": "If True, all timing logging will be logged"}
    )
    callbacks: List[Union[str, CallBacks]] = list_field(
        CallBacks.wandb,
        # choices=[e.name for e in CallBacks],
        metadata={"help": "options are 'wandb' and 'tensorboard'"},
    )


@dataclass
class Args(Serializable):

    datatrain: DataTrainingArguments = DataTrainingArguments()
    training: TrainingArguments = TrainingArguments()
    model: ModelArguments = ModelArguments()
    embedds: EmbeddingArguments = EmbeddingArguments()
    visual: VisualArguments = VisualArguments()
    runs: RunArguments = RunArguments()
    logs: LoggingArguments = LoggingArguments()


class PartialChoice(enum.Enum):
    DATATRAIN = 'DATATRAIN'
    TRAINING = 'TRAINING'
    MODEL = 'MODEL'
    EMBEDDS = 'EMBEDDS'
    VISUAL = 'VISUAL'
    RUNS = 'RUNS'
    LOGS = 'LOGS'
    ALL = 'ALL'
