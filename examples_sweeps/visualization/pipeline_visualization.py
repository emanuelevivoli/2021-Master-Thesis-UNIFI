from thesis.parsers.args_parser import parse_args
from thesis.parsers.utils import tag_generation

from thesis.datasets.utils import format_key_names, fuse_datasets_splits, select_concat_fields, split_and_replicate
from thesis.datasets.s2orc.mag_field import mag_field_dict

from thesis.visualization.embedding import embedd
from thesis.visualization.dim_reduction import pre_reduction, post_reduction
from thesis.visualization.clusterization import clusterization
from thesis.visualization.utils import visualization

from thesis.clusters.utils import c_tf_idf
from thesis.topic.utils import extract_top_n_words_per_topic, extract_topic_sizes

from thesis.utils.load_dataset import custom_load_dataset
from thesis.utils.cache import no_caching, _caching

from sklearn import metrics
import pandas as pd
import numpy as np
import wandb

import logging
logger = logging.getLogger(__name__)


def main(args=None):

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
        project='real-sweeps',
        notes="Try out the visualization baselines",
        tags=["visualization", "sweeps"]+tags,
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
    def get_dataset(args, label_field='mag_field_of_study'):
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

        # The Papers that comes with multiple labels ('mag_field_of_study')
        # are replicated in the dataset with each 'mag_field'
        dataset = split_and_replicate(dataset)

        labels_true = np.asarray([mag_field_dict[mag.pop()]
                                  for mag in dataset[label_field]])

        return dataset, labels_true

    # We want to access to dictionary with Dot notation (dict.field)
    # instead of the String one (dict['field'])
    # config_ = DotDict(config)
    args_ = args

    dataset, labels_true = get_dataset(args_)
    # artifacts.labels_true = labels_true
    corpus = select_concat_fields(dataset, args.visual.fields)

    # ------------------
    # Clustering Creation
    # & Visualization
    # ------------------
    embeddings = embedd(args_, corpus)
    print(
        f"[ BASE EMB ]: {embeddings.shape}   | {args_.model.model_name_or_path} on {args_.visual.fields}")

    embeddings = pre_reduction(args_.visual, embeddings)
    print(f"[ PRE  EMB ]: {embeddings.shape}   | {args_.visual.pre.choice}")

    labels = clusterization(args_.visual, embeddings)
    # artifacts.labels = labels
    print(f"[ labels   ]: {labels.shape}       | {args_.visual.clust.choice}")

    x, y = post_reduction(args_.visual, embeddings)
    # artifacts.x = x
    # artifacts.y = y
    print(f"[ POST EMB ]: {x.shape}, {y.shape} | {args_.visual.post.choice}")
    wandb.log({
        "scatter":
        wandb.Table(
            data=list(zip(x, y, labels, labels_true)),
            columns=["x", "y", "Predicted Label", "True Label"])
    })

    name, plt = visualization(args_,  x, y, labels)
    # artifacts.fig = wandb.Image(plt)
    print(f"[ pic  EMB ]: {name}")
    wandb.log({
        "fig":
        wandb.Image(plt)
    })

    # ------------------
    # Table Creation
    # ------------------
    # clustered_sentences = [[] for i in range(len(set(labels)))]
    # for sentence_id, cluster_id in enumerate(labels):
    #     clustered_sentences[cluster_id].append(corpus[sentence_id])

    dataframe = {
        "mag": mag_field_dict.keys(),
        "TOT": [0 for _ in mag_field_dict.keys()]
    }

    for cluster_id in list(set(labels)):
        dataframe[f'cluster {cluster_id}'] = [0 for _ in mag_field_dict.keys()]

    df = pd.DataFrame(dataframe)

    for cluster_id, mag_list in zip(labels, dataset['mag_field_of_study']):
        for mag in mag_list:
            # df.loc['TOT', mag_field_dict[mag]] = df['TOT'][mag_field_dict[mag]] + 1
            # df.loc['TOT', mag_field_dict[mag]] += 1
            df['TOT'][mag_field_dict[mag]] += 1
            # df.loc[f'cluster {cluster_id}', mag_field_dict[mag]] = df[f'cluster {cluster_id}'][mag_field_dict[mag]] + 1
            # df.loc[f'cluster {cluster_id}', mag_field_dict[mag]] += 1
            df[f'cluster {cluster_id}'][mag_field_dict[mag]] += 1

    app = df.append(df.sum(axis=0), ignore_index=True)
    # app.loc['mag', 20] = 'TOT'
    app['mag'][20] = 'TOT'
    # artifacts.df = wandb.Table(dataframe=app)
    wandb.log({
        "df":
        wandb.Table(dataframe=app)
    })

    # ------------------
    # Clustered
    # TF-IDF
    # ------------------
    docs_df = pd.DataFrame(corpus, columns=["Doc"])
    docs_df['Topic'] = labels
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(
        ['Topic'], as_index=False).agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(corpus))

    words_top_n_words, tfidf_top_n_words = extract_top_n_words_per_topic(
        tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df)

    words_c_tfidf = pd.DataFrame.from_records(words_top_n_words)
    # artifacts.words_c_tfidf = wandb.Table(dataframe=words_c_tfidf)
    wandb.log({
        "words_c_tfidf":
        wandb.Table(dataframe=words_c_tfidf)
    })

    tfidf_c_tfidf = pd.DataFrame.from_records(tfidf_top_n_words)
    # artifacts.tfidf_c_tfidf = wandb.Table(dataframe=tfidf_c_tfidf)
    wandb.log({
        "tfidf_c_tfidf":
        wandb.Table(dataframe=tfidf_c_tfidf)
    })

    # artifacts.rand_score = metrics.rand_score(labels_true, labels)
    wandb.log({
        "rand_score": metrics.rand_score(labels_true, labels)
    })
    # artifacts.adj_rand_score = metrics.adjusted_rand_score(labels_true, labels)
    wandb.log({
        "adj_rand_score": metrics.adjusted_rand_score(labels_true, labels)
    })

    # wandb.log({"artifacts": artifacts})


if __name__ == '__main__':

    # import os
    # from thesis.utils.constants import ARGS_PATH
    # main(os.path.join(ARGS_PATH, 'args.yaml'))
    main()
