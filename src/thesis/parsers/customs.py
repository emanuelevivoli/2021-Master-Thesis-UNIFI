from dataclasses import field, dataclass
from typing import Optional

# 🤗 Transformers
from transformers import MODEL_FOR_MASKED_LM_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
DIM_REDUCTION = ['none', 'umap', 'pca', 'tsne']

# ----------------------------- #
# Definition of Custom Parsers  #
#    - ModelArguments           #
#    - S2orcArguments           #
#    - KeyPhArguments           #
#    - JournArguments           #
#    - EmbeddingArguments       #
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
class JournArguments:
    """
    Arguments pertaining to (custom) Journal dataset, and what data we are going to input our model for training and eval.
    """

    # there is the 'dataset_config_name' variable for that :=)
    # keyph_type: Optional[str] = field(
    #     default=None, metadata={"help": "Type of keyphrases dataset. Vary Options"}
    # )


@dataclass
class ModelArguments(S2orcArguments, KeyPhArguments, JournArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
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

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    # ?
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    # ?
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # ? opposite to the above ?
    # use_slow_tokenizer: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."
    #     },
    # )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class EmbeddingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    pooling: str = field(
        default="none",
        metadata={
            "help": "Pooling method: none, first, mean, sum (default:none)"},
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
class VisualArguments:
    """
    Arguments pertaining to how we want to visualize data (embeddings).
    """
    fields: str = field(
        default='abstract',
        metadata={
            "help": f"Field from dataset to use for embedding generation"},
    )
    # same as `model_name_or_path` in ModelArguments
    model_name: Optional[str] = field(
        default='bho',
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    # ------------------
    # PRE
    # ------------------
    pre_alg: str = field(
        default='umap',
        metadata={
            "help": f"Algorithm use for reducing the dimentionality (pre-clustering) [options {DIM_REDUCTION}]"},
    )
    pre_n_neighbors: int = field(
        default=15,
        metadata={"help": f"Preprocessing alg n_neighbors."},
    )
    # only for umap, tsne
    pre_metric: str = field(
        default='cosine',
        metadata={"help": f"umap and tsne field for metric."},
    )
    # UMAP & PCA
    pre_n_components: int = field(
        default=5,
        metadata={
            "help": f"UMAP & PCA field for n_components, output space size (components)."},
    )
    # t-SNE
    pre_perplexity: float = field(
        default=30.0,
        metadata={"help": f"t-SNE field for perplexity."},
    )
    # ------------------
    # CLUSTERING
    # ------------------
    clustering_alg: str = field(
        default='kmeans',
        metadata={"help": f"Clustering algorithm."},
    )
    n_clusters: int = field(
        default=10,
        metadata={"help": f"number of clusters."},
    )
    min_cluster_size: int = field(
        default=5,
        metadata={"help": f"HDBSCAN min cluster size"},
    )
    metric: str = field(
        default='euclidean',
        metadata={"help": f"HDBSCAN metric choice"},
    )
    cluster_selection_method: str = field(
        default='eom',
        metadata={"help": f"HDBSCAN cluster_selection_method"},
    )
    n_init: int = field(
        default=10,
        metadata={"help": f"KMEANS number of init (??)."},
    )
    max_iter: int = field(
        default=300,
        metadata={"help": f"KMEANS number of max iteration."},
    )
    affinity: str = field(
        default='euclidean',
        metadata={"help": f"HDBSCAN affinity"},
    )
    linkage: str = field(
        default='ward',
        metadata={"help": f"HDBSCAN affinity"},
    )
    # ------------------
    # POST
    # ------------------
    post_alg: str = field(
        default='umap',
        metadata={
                "help": f"Algorithm use for reducing the dimentionality (pre-clustering) [options {DIM_REDUCTION}]"},
    )
    post_n_neighbors: int = field(
        default=15,
        metadata={"help": f"Postprocessing alg n_neighbors."},
    )
    # only for umap, tsne
    post_metric: str = field(
        default='cosine',
        metadata={"help": f"umap and tsne field for metric."},
    )
    # UMAP & PCA
    post_n_components: int = field(
        default=5,
        metadata={
            "help": f"UMAP & PCA field for n_components, output space size (components)."},
    )
    # t-SNE
    post_perplexity: float = field(
        default=30.0,
        metadata={"help": f"t-SNE field for perplexity."},
    )
    post_min_dist: float = field(
        default=0.0,
        metadata={"help": f"UMAP field for min_dist."},
    )


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
    debug_log: Optional[bool] = field(
        default=False, metadata={"help": "If True, all debug logging will be logged"}
    )
    time: Optional[bool] = field(
        default=False, metadata={"help": "If True, all timing logging will be logged"}
    )
    callbacks: Optional[str] = field(
        default="WandbCallback",
        metadata={"help": "options are 'wandb' and 'tensorboard'"},
    )
