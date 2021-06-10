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
from thesis.utils.constants import LABEL_DICT, SUBLABEL_DICT
from torch.utils.data.dataset import Subset

logger = logging.getLogger(__name__)

FILTER_NONE = True


def main(args=None, pipeline=None):
    # ------------------
    # Creating the
    # Tensorboard Writer
    # ------------------
    writer = SummaryWriter(comment="")

    # ------------------
    # Parsing the Arguments
    # ------------------
    args = parse_args(args)

    # ------------------
    # Init the wandb
    # ------------------
    tags = tag_generation(args)

    if False:
        # Pass them to wandb.init
        wandb.init(
            project='example-visualization',
            notes="Try out the visualization baselines",
            tags=["visualization"]+tags,
            config=args.to_dict(),
        )
        # Access all hyperparameter values through wandb.config
        config = wandb.config.as_dict()

    from thesis.utils.classes import DotDict
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
    def get_dataset(args, keep_fields=['title', 'abstract'], label_fields=['group', 'subgroup']):
        def get_dataset_labels(dataset_name, dataset_config_name):
            if dataset_name == 'journal':
                dataset_label_dict = LABEL_DICT[dataset_config_name]
            else:
                dataset_label_dict = LABEL_DICT[dataset_name]
            return dataset_label_dict

        def get_dataset_sublabels(dataset_name, dataset_config_name):
            if dataset_name == 'journal':
                dataset_sublabel_dict = SUBLABEL_DICT[dataset_config_name]
            else:
                dataset_sublabel_dict = SUBLABEL_DICT[dataset_name]
            return dataset_sublabel_dict

        label_field = label_fields[0]
        sublabel_field = label_fields[1]

        dataset_label_dict = get_dataset_labels(
            args.datatrain.dataset_name,
            args.datatrain.dataset_config_name)

        dataset_sublabel_dict = get_dataset_sublabels(
            args.datatrain.dataset_name,
            args.datatrain.dataset_config_name)

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
        if args.datatrain.dataset_name != 'journal':
            dataset = split_and_replicate(dataset, keep_fields, [label_field])

        if args.datatrain.dataset_name != 'keyphrase':
            labels_true = np.asarray([dataset_label_dict.get(mag[0] if type(mag) == list else mag, -1)
                                      for mag in dataset[label_field]])
            if dataset_sublabel_dict is not None:
                sublabels_true = np.asarray([dataset_sublabel_dict.get(mag[0] if type(mag) == list else mag, -1)
                                             for mag in dataset[sublabel_field]])
            else:
                sublabels_true = None

        else:
            labels_true = None
            sublabels_true = None

        return dataset, labels_true, sublabels_true

    # We want to access to dictionary with Dot notation (dict.field)
    # instead of the String one (dict['field'])
    # config_ = DotDict(config)
    args_ = args  # DotDict(config)
    pipeline = DotDict(pipeline)

    print(f"[ GET DATASETS ]: starting getting data")

    dataset, labels_true, sublabels_true = get_dataset(
        args_, args_.datatrain.data + args_.datatrain.target, args_.datatrain.classes)

    if FILTER_NONE:
        idxs = labels_true != -1
        dataset = dataset.filter(
            (lambda x, ids: bool(idxs[ids])), with_indices=True)
        labels_true = labels_true[idxs]
        sublabels_true = sublabels_true[idxs]

    print(f"[ CONCAT FIELD ]: starting concatenating fields")

    # artifacts.labels_true = labels_true
    corpus = select_concat_fields(
        dataset, args_.datatrain.data + args_.datatrain.target)

    def model_utilization(args_, dataset, labels_true, corpus):

        print(f"[ START EMBEDD ]: starting embedd")

        embeddings = embedd(args_, corpus)
        print(
            f"[ BASE EMB ]: {embeddings.shape}   | {args_.model.model_name_or_path} on {args_.visual.fields}")

        # ------------------
        # Pre Dimentionality
        # Reduction
        # ------------------
        visual = DotDict(dict(
            no_cache=True,
            fields=["title,abstract"],
            pre=DotDict(dict(
                choice='UMAP',
                umap=DotDict(dict(
                    n_neighbors=15,
                    metric='cosine',
                    n_components=25,  # 768 -> 50
                ))
                # tsne=dict(
                #     n_neighbors=15,
                #     metric='cosine',
                #     perplexity=30.0,
                # ),
                # pca=dict(
                #     n_neighbors=15
                # )
            ))
        ))
        pre_reduced = pre_reduction(visual, embeddings)
        print(f"[ PRE  EMB ]: {pre_reduced.shape}   | {visual.pre.choice}")

        # ------------------
        # Clustering Creation
        # & Visualization
        # ------------------
        clusters_labels = []

        def get_cluster_labels(clust_alg, pre_reduced):
            if clust_alg == 'kmeans':
                # ------------------
                # KMEANS
                # ------------------
                clust = DotDict(dict(
                    choice='KMEANS',
                    kmeans=DotDict(dict(
                        n_clusters=5,
                        n_init=10,
                        max_iter=300,
                    ))
                ))
            elif clust_alg == 'hdbscan':
                # ------------------
                # HDBSCAN
                # ------------------
                clust = DotDict(dict(
                    choice='HDBSCAN',
                    hdbscan=DotDict(dict(
                        min_cluster_size=2,
                        metric='euclidean',
                        cluster_selection_method='eom',
                    ))
                ))
            elif clust_alg == 'optics':
                # ------------------
                # OPTICS
                # ------------------
                clust = DotDict(dict(
                    choice='OPTICS',
                    optics=DotDict(dict(
                        min_cluster_size=.05,
                        # metric='minkowski',
                        # cluster_method='xi', # “xi” and “dbscan”
                        xi=.05,
                        min_samples=5,
                    ))
                ))
            else:
                raise ValueError(f'{clust_alg} algorithm not supported!')

            visual = DotDict(dict(
                clust=clust
            ))

            cluster_label = clusterization(visual, pre_reduced)
            # artifacts.labels = labels
            print(
                f"[ labels   ]: {cluster_label.shape}       | {visual.clust.choice}")
            return [str(x) for x in cluster_label]

        # iterate over clust choice and apply them
        for clust_alg in pipeline.clusts:
            cluster_label = get_cluster_labels(clust_alg, pre_reduced)
            clusters_labels.append(cluster_label)

        label_field = args_.datatrain.classes[0]
        sublabel_field = args_.datatrain.classes[1]

        first_line = pipeline.clusts + \
            ["label", "title", "corpus", "n_label"]

        to_zip = [
            *clusters_labels,
            list(
                map(lambda x: x[0] if type(x) == list else x, dataset[label_field])),
            dataset['title'],
            corpus,
            [str(x) for x in labels_true]
        ]

        if sublabels_true is not None:
            first_line.append('sublabel')
            first_line.append('n_sublabel')

            to_zip.append(list(
                map(lambda x: x[0] if type(x) == list else x, dataset[sublabel_field])))
            to_zip.append([str(x) for x in sublabels_true])

        line_by_line = [
            list(_) for _ in zip(
                *to_zip
            )
        ]

        return embeddings, line_by_line, first_line

    # ------------------
    # SciBERT
    # ------------------
    if 'scibert' in pipeline.models:
        args.model.model_name_or_path = 'allenai/scibert_scivocab_uncased'

        embeddings, line_by_line, first_line = model_utilization(
            args_, dataset, labels_true, corpus)

        writer.add_embedding(
            embeddings,
            metadata=line_by_line,  # first_line+line_by_line,
            metadata_header=first_line,
            tag='scibert')

    # ------------------
    # ParaPHRASE
    # ------------------
    if 'paraphrase' in pipeline.models:
        args.model.model_name_or_path = 'paraphrase-distilroberta-base-v1'

        embeddings, line_by_line, first_line = model_utilization(
            args_, dataset, labels_true, corpus)

        writer.add_embedding(
            embeddings,
            metadata=line_by_line,  # first_line+line_by_line,
            metadata_header=first_line,
            tag='paraphrase')

    # ------------------
    # DistilBERT
    # ------------------
    if 'distilbert' in pipeline.models:
        args.model.model_name_or_path = 'distilbert-base-nli-mean-tokens'

        embeddings, line_by_line, first_line = model_utilization(
            args_, dataset, labels_true, corpus)

        writer.add_embedding(
            embeddings,
            metadata=line_by_line,  # first_line+line_by_line,
            metadata_header=first_line,
            tag='distilbert')

    # ------------------
    # BERT base
    # ------------------
    if 'bert_base' in pipeline.models:
        args.model.model_name_or_path = 'bert-base-uncased'

        embeddings, line_by_line, first_line = model_utilization(
            args_, dataset, labels_true, corpus)

        writer.add_embedding(
            embeddings,
            metadata=line_by_line,  # first_line+line_by_line,
            metadata_header=first_line,
            tag='bert_base')

    writer.close()


if __name__ == '__main__':
    # import os
    # from thesis.utils.constants import ARGS_PATH
    # main(os.path.join(ARGS_PATH, 'args.yaml'))
    # main('/home/vivoli/Thesis/examples/visualization_interactive/embedding_args.yaml')

    args = dict(
        # DatasetArguments
        datatrain=dict(
            no_cache=True,
            dataset_path='/home/vivoli/Thesis/data',
            dataset_name='journal',
            dataset_config_name='icpr_20',
            # ? train_file
            # ? validation_file
            # ? validation_split_percentage
            data=['title', 'abstract'],
            # ? target=[],
            classes=['group', 'subgroup'],
            # ? pad_to_max_length
            # ? use_slow_tokenizer
            # ? overwrite_cache
            max_seq_length='512'
            # ? preprocessing_num_workers
            # ? mlm_probability
            # ? line_by_line
            # ? max_train_samples
            # ? max_eval_samples
        ),
        training=dict(
            # ? no_cache=True,
            seed=1234,
            # ? do_train=False,
            # ? do_eval=False,
            # ? do_predict=False,
            output_dir='/home/vivoli/Thesis/output',
            # ? overwrite_output_dir
            # ? num_train_epochs=1,
            # ? max_train_steps
            # 16 and 32 end with "RuntimeError: CUDA out of memory."
            # ? per_device_train_batch_size=8,
            # 16 and 32 end with "RuntimeError: CUDA out of memory."
            # ? per_device_eval_batch_size=8,
            # ? learning_rate
            # ? weight_decay
            # ? gradient_accumulation_steps
            # ? lr_scheduler_type
            # ? num_warmup_steps
            # ? logging_dir
        ),
        visual=dict(
            #     no_cache=True,
            #     # ? model_name_or_path
            #     # ? model_type
            #     # ? config_name
            fields=["title,abstract"],
            #     # pre=dict(
            #     #     choice='UMAP',
            #     #     umap=dict(
            #     #         n_neighbors=15,
            #     #         metric='cosine',
            #     #         n_components=5,  # 768 -> 50
            #     #     ),
            #     #     # tsne=dict(
            #     #     #     n_neighbors=15,
            #     #     #     metric='cosine',
            #     #     #     perplexity=30.0,
            #     #     # ),
            #     #     # pca=dict(
            #     #     #     n_neighbors=15
            #     #     # )
            #     # ),
            #     clust=dict(
            #         choice='KMEANS',
            #         kmeans=dict(
            #             n_clusters=20,
            #             n_init=10,
            #             max_iter=300,
            #         ),
            #         # hdbscan=dict(
            #         #     min_cluster_size=5,
            #         #     metric='euclidean',
            #         #     cluster_selection_method='eom',
            #         # ),
            #         # hierarchical=dict(
            #         #     affinity='euclidean',  # “l1” “l2” “manhattan” “cosine” or “precomputed”
            #         #     linkage='ward',  # ‘complete’ ‘average’ ‘single’
            #         # )
            #     )
            #     # post=dict(
            #     #     choice='UMAP',
            #     #     umap=dict(
            #     #         n_neighbors=15,
            #     #         metric='cosine',
            #     #         n_components=2,  # 50 -> 2
            #     #     ),
            #     #     # tsne=dict(
            #     #     #     n_neighbors=15,
            #     #     #     metric='cosine',
            #     #     #     perplexity=30.0,
            #     #     # ),
            #     #     # pca=dict(
            #     #     #     n_neighbors=15
            #     #     # )
            #     # ),
        ),
        model=dict(
            # ? no_cache=True,
            model_name_or_path='allenai/scibert_scivocab_uncased',
            # ? model_type
            # ? config_name
            # ? tokenizer_name
            # ? cache_dir
            # ? use_fast_tokenizer
            # ? model_revision
            # ? use_auth_token
        ),
        runs=dict(
            run_name='journl-s2orc',
            run_number=0,
            run_iteration=0,
        ),
        logs=dict(
            verbose=False,
            debug_log=False,
            time=False,
            callbacks=[]
        )
    )

    pipeline = dict(
        models=['scibert', 'paraphrase', 'distilbert', 'bert_base'],
        clusts=['kmeans', 'hdbscan', 'optics']
    )

    main(args, pipeline)
