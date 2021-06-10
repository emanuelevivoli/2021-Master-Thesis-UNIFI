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
from thesis.utils.constants import LABEL_DICT

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
    # Getting the datasets
    # ------------------
    @_caching(
        args.visual.fields,
        args.datatrain.to_dict(),
        args.runs.to_dict(discard=['run_name']),
        function_name='get_dataset'
    )
    def get_dataset(args, keep_fields=['title', 'abstract'], label_fields=['mag_field_of_study']):
        def get_dataset_labels(dataset_name, dataset_config_name):
            if dataset_name == 'journal':
                dataset_label_dict = LABEL_DICT[dataset_config_name]
            else:
                dataset_label_dict = LABEL_DICT[dataset_name]
            return dataset_label_dict

        label_field = label_fields.pop()
        dataset_label_dict = get_dataset_labels(
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
            labels_true = np.asarray([dataset_label_dict.get(mag.pop() if type(mag) == list else mag, -1)
                                      for mag in dataset[label_field]])
        else:
            labels_true = None

        idxs = labels_true != -1

        return dataset[idxs], labels_true[idxs]

    # We want to access to dictionary with Dot notation (dict.field)
    # instead of the String one (dict['field'])
    # config_ = DotDict(config)
    args_ = args  # DotDict(config)

    dataset, labels_true = get_dataset(
        args_, args_.datatrain.data + args_.datatrain.target, args_.datatrain.classes)
    # artifacts.labels_true = labels_true
    corpus = select_concat_fields(
        dataset, args_.datatrain.data + args_.datatrain.target)

    print('All done correctly')


if __name__ == '__main__':

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
            data=['abstract'],
            target=['title'],
            classes=['group'],
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
        # training=dict(
        #     # no_cache=True,
        #     seed=1234,
        #     do_train=False,
        #     do_eval=False,
        #     do_predict=False,
        #     output_dir='/home/vivoli/Thesis/output',
        #     # ? overwrite_output_dir
        #     num_train_epochs=1,
        #     # ? max_train_steps
        #     # 16 and 32 end with "RuntimeError: CUDA out of memory."
        #     per_device_train_batch_size=8,
        #     # 16 and 32 end with "RuntimeError: CUDA out of memory."
        #     per_device_eval_batch_size=8,
        #     # ? learning_rate
        #     # ? weight_decay
        #     # ? gradient_accumulation_steps
        #     # ? lr_scheduler_type
        #     # ? num_warmup_steps
        #     # ? logging_dir
        # ),
        # model=dict(
        #     # no_cache=True,
        #     model_name_or_path='allenai/scibert_scivocab_uncased',
        #     # ? model_type
        #     # ? config_name
        #     # ? tokenizer_name
        #     # ? cache_dir
        #     # ? use_fast_tokenizer
        #     # ? model_revision
        #     # ? use_auth_token
        # ),
        runs=dict(
            run_name='journl-s2orc',
            run_number=0,
            run_iteration=0,
        ),
        logs=dict(
            verbose=False,
            debug_log=False,
            time=False,
            callbacks=['WandbCallback', 'CometCallback', 'TensorBoardCallback']
        )

    )

    main(args)
