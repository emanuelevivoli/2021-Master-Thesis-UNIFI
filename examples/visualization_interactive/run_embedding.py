import logging
import wandb
import numpy as np
import pandas as pd
from sklearn import metrics
from thesis.utils.cache import no_caching, _caching
from thesis.utils.load_dataset import custom_load_dataset
from thesis.topic.utils import extract_top_n_words_per_topic, extract_topic_sizes
from thesis.clusters.utils import c_tf_idf
from thesis.visualization.utils import visualization
from thesis.visualization.clusterization import clusterization
from thesis.visualization.dim_reduction import pre_reduction, post_reduction
from thesis.visualization.embedding import embedd
from thesis.datasets.s2orc.mag_field import mag_field_dict
from thesis.datasets.utils import format_key_names, fuse_datasets_splits, select_concat_fields, split_and_replicate
from thesis.parsers.utils import tag_generation
from thesis.parsers.args_parser import parse_args
import keyword
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def main(args=None):
    # ------------------
    # Creating the
    # Tensorboard Writer
    # ------------------
    writer = SummaryWriter()

    # ------------------
    # Parsing the Arguments
    # ------------------
    args = parse_args(args)

    # ------------------
    # Init the wandb
    # ------------------
    tags = tag_generation(args)
    # Pass them to wandb.init
    wandb.init(
        project='example-visualization',
        notes="Try out the visualization baselines",
        tags=["visualization"]+tags,
        config=args.to_dict(),
    )
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    # from thesis.utils.classes import DotDict
    # artifacts = DotDict({
    #     "labels_true": None,
    #     "labels": None,
    #     "x": None,
    #     "y": None,
    #     "fig": None,
    #     "df": None,
    #     "words_c_tfidf": None,
    #     "tfidf_c_tfidf": None,
    #     "rand_score": None,
    #     "adj_rand_score": None
    # })

    # ------------------
    # Getting the datasets
    # ------------------
    @_caching(
        args.visual.fields,
        args.datatrain.to_dict(),
        args.runs.to_dict(discard=['run_name']),
        function_name='get_dataset'
    )
    def get_dataset(args, keep_fields=['title', 'abstract'], label_field=['mag_field_of_study']):
        # Getting the load_dataset wrapper that manages huggingface dataset and the custom ones
        # Loading the raw data based on input (and default) values of arguments
        raw_dataset = custom_load_dataset(args)

        # The Datasets in the raw form can have different form of key names (depending on the configuration).
        # We need all datasets to contain 'train', 'test', 'validation' keys, if not we change the dictionary keys' name
        # based on the `names_tuple` and conseguently on `names_map`.
        logging.info(f"Formatting DatasetDict keys")
        dataset = format_key_names(raw_dataset)

        # The Dataset comes with train/test/validation splits.
        # As we need only one field the next step is to fude them together
        dataset = fuse_datasets_splits(dataset)

        # The Papers that comes with multiple labels ('mag_field_of_study' or 'keyphrases')
        # are replicated in the dataset with each 'mag_field'/'keyphrase'
        dataset = split_and_replicate(dataset, keep_fields, label_field)

        if label_field is not None:
            labels_true = np.asarray([mag_field_dict[mag.pop()]
                                      for mag in dataset[label_field]])
        else:
            labels_true = None

        return dataset, labels_true

    # We want to access to dictionary with Dot notation (dict.field)
    # instead of the String one (dict['field'])
    # config_ = DotDict(config)
    args_ = args

    dataset, labels_true = get_dataset(
        args_, args_.datatrain.data + args_.datatrain.target, args_.datatrain.classes)
    # artifacts.labels_true = labels_true
    corpus = select_concat_fields(dataset, args.visual.fields)

    # ------------------
    # Clustering Creation
    # & Visualization
    # ------------------
    embeddings = embedd(args_, corpus)
    print(
        f"[ BASE EMB ]: {embeddings.shape}   | {args_.model.model_name_or_path} on {args_.visual.fields}")

    labels = clusterization(args_.visual, embeddings)
    # artifacts.labels = labels
    print(f"[ labels   ]: {labels.shape}       | {args_.visual.clust.choice}")

    first_line = ["true", "clust", "mag", "title", "corpus"]
    line_by_line = [
        [t, f, m, ti, c] for t, f, m, ti, c in zip(
            [str(x) for x in labels_true],
            [str(x) for x in labels],
            list(
                map(lambda x: x[0], dataset['mag_field_of_study'])),
            dataset['title'],
            corpus
        )
    ]

    writer.add_embedding(
        embeddings,
        metadata=line_by_line,  # first_line+line_by_line,
        metadata_header=first_line,
        tag='emb_mag_numbers')

    writer.close()


if __name__ == '__main__':
    import os
    # from thesis.utils.constants import ARGS_PATH
    # main(os.path.join(ARGS_PATH, 'args.yaml'))
    main('/home/vivoli/Thesis/examples/visualization_interactive/embedding_args.yaml')
