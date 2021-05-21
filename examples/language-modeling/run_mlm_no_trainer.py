#!/usr/bin/env python
# coding: utf-8

# In[1]:

# !python -m venv .env
# !source .env/bin/activate
# !pip install transformers
# !pip install accelerate

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[4]:

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TODO Add some generic comment ...
"""

# ----------------------------------- #
#           All Imports
# ----------------------------------- #

import argparse
import logging
import os
import io
import sys
import math
import random

from tqdm.auto import tqdm

# 🤗 Datasets
import datasets
from datasets import load_dataset, load_metric
from thesis.utils.load_dataset import load_dataset_wrapper

# parsing
from thesis.parsers.args_parser import parse_args

# dataset managements
from torch.utils.data import Dataset, DataLoader

# data managements
import json  # load/write data
import torch
import numpy as np
import pandas as pd

# dataclasses and types
from dataclasses import field, dataclass
from typing import Dict, List, Union, Optional

# accelerator for speed up experiments/runs
from accelerate import Accelerator

# 🤗 Tranformers
import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    TrainingArguments as HfTrainingArguments,
    SchedulerType,
    get_scheduler,
    set_seed,
)

logger = logging.getLogger(__name__)


# ----------------------------------- #
#           Imports for mlm
# ----------------------------------- #

"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""

# 🤗 Tranformers


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DICTIONARY_FIELD_NAMES = dict(
    train=['train'],
    test=['test', 'debug', 'dev'],
    validation=['validation', 'valid']
)


# In[7]:

def main(args_list):

    # ------------------
    # Parsing arguments
    # ------------------

    # Pass the args_list to parse_args. We will use it for args = parse_args(args_list)
    dataset_args, training_args, s2orc_args, keyph_args, run_args, log_args, embedding_args = parse_args(
        args_list)

    # ------------------
    # Logging definition
    # ------------------

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.info(
        f"Accelerator local main process: {accelerator.is_local_main_process}")
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # TODO Adding the logger:logging.Logger to log_args as *args
    # TODO allowing the LoggingConfig to logs on that Logger
    log_args.logger = logger

    # ------------------
    # Setting seed
    # ------------------

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # ------------------
    # Getting the datasets
    # ------------------

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    # You can also provide the name of some other dataset ('S2orc', 'Keyphrase') that we customly support.
    #
    # For the 'S2orc' dataset we provide a caching mechanisms in order to speed-up the preprocessing.
    # The cache files changes (and are recalculated) everytime the configurations changes.
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if dataset_args.dataset_name is not None:
        # Getting the load_dataset wrapper that manages huggingface dataset and the custom ones
        custom_load_dataset = load_dataset_wrapper()
        # Loading the raw data based on input (and default) values of arguments
        raw_datasets = custom_load_dataset(
            dataset_args, training_args, s2orc_args, keyph_args, run_args, log_args, embedding_args)
    else:
        # If the files 'train_file' and 'validation_file' are specified
        # data_files is composed by those elements.
        data_files = {}
        if dataset_args.train_file is not None:
            data_files["train"] = dataset_args.train_file
        if dataset_args.validation_file is not None:
            data_files["validation"] = dataset_args.validation_file
        extension = dataset_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        elif extension == "jsonl":  # jsonl files are file with json element per row
            extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # The Datasets in the raw form can have different form of key names (depending on the configuration).
    # We need all datasets to contain 'train', 'test', 'validation' keys, if not we change the dictionary keys' name
    # based on the `names_tuple` and conseguently on `names_map`.
    def format_key_names(raw_datasets):
        # The creation of `names_map` happens to be here
        # For every element in the values lists, one dictionary entry is added
        # with (k,v): k=Value of the list, v=Key such as 'train', etc.
        def names_dict_generator(names_tuple: dict):
            names_map = dict()
            for key, values in names_tuple.items():
                for value in values:
                    names_map[value] = key
            return names_map
        names_map = names_dict_generator(DICTIONARY_FIELD_NAMES)
        split_names = raw_datasets.keys()
        for split_name in split_names:
            new_split_name = names_map.get(split_name)
            if split_name != new_split_name:
                raw_datasets[new_split_name] = raw_datasets.pop(split_name)
        return raw_datasets

    logger.info(f"Formatting DatasetDict keys")
    raw_datasets = format_key_names(raw_datasets)

    # ------------------
    # Load tokenizer and
    # pretrained model
    # ------------------

    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if dataset_args.config_name:
        config = AutoConfig.from_pretrained(dataset_args.config_name)
    elif dataset_args.model_name_or_path:
        config = AutoConfig.from_pretrained(dataset_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[training_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if dataset_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            dataset_args.tokenizer_name, use_fast=not dataset_args.use_slow_tokenizer)
    elif dataset_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            dataset_args.model_name_or_path, use_fast=not dataset_args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if dataset_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            dataset_args.model_name_or_path,
            from_tf=bool(".ckpt" in dataset_args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # ------------------
    # Preprocessing the
    # datasets
    # ------------------

    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    if "text" in column_names:
        text_column_name = "text"
        logger.info(
            f"Dataset 'train' has 'text' field. It'll be used by the model!")
    else:
        text_column_name = column_names[0]
        logger.info(
            f"Dataset 'train' hasn't 'text' field. Field {text_column_name} will be used by the model!")

    if dataset_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if dataset_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({dataset_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(dataset_args.max_seq_length,
                             tokenizer.model_max_length)

    if dataset_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if dataset_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"]
                                if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=dataset_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not dataset_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=dataset_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not dataset_args.overwrite_cache,
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=dataset_args.preprocessing_num_workers,
            load_from_cache_file=not dataset_args.overwrite_cache,
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    # ------------------
    # Data collator
    # ------------------

    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=dataset_args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)

    # ------------------
    # Optimizer
    # ------------------

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=training_args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_train_steps is None:
        training_args.max_train_steps = training_args.num_train_epochs * \
            num_update_steps_per_epoch
    else:
        training_args.num_train_epochs = math.ceil(
            training_args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.max_train_steps,
    )

    # ------------------
    # 🚀 🚀 🚀 Train !
    # ------------------

    total_batch_size = training_args.per_device_train_batch_size * \
        accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(
        f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(training_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= training_args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(
                training_args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        perplexity = math.exp(torch.mean(losses))

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, save_function=accelerator.save)


# In[ ]:

# Arguments are passed either by script arguments call
# or by an `args_list` as bellow.
if __name__ == "__main__":
    args_list = [

        # DatasetArguments
        "--model_name_or_path", "allenai/scibert_scivocab_uncased",
        "--dataset_name", "s2orc",
        "--dataset_config_name", "full",

        # TrainingArguments
        "--seed", '1234',  # seed for reproducibility of experiments
        "--output_dir", "output",
        "--debug", 'False',

        "--run_name", "scibert-s2orc",
        "--num_train_epochs", '1',
        "--per_device_train_batch_size", "32",
        "--per_device_eval_batch_size", "32",
        "--max_seq_length", '512',  # cistom added

        # S2orcArguments & KeyPhArguments
        "--dataset_path", "/home/vivoli/Thesis/data",

        # S2orcArguments
        "--idxs", '0',
        "--zipped", 'True',
        "--mag_field_of_study", "Computer Science",  # list
        "--data", "abstract",  # list
        "--target", "title",  # list
        # Field for classification
        "--classes", "mag_field_of_study",  # list
        # Cleaning dataset with none fields & unused columns
        "--keep_none_papers", 'False',
        "--keep_unused_columns", 'False',

        # RunArguments
        # "--run_name"                     , "scibert-s2orc",
        "--run_number", '0',
        "--run_iteration", '0',

        # LoggingArguments
        "--verbose", 'True',
        # "--debug"                        , False,
        "--time", 'False',
        "--callback", "WandbCallback",

        # EmbeddingArguments
        # "--max_seq_length"               , '512',
        # "--pooling"                      , 'none',
        # "--batch_size"                   , '32'
    ]

    main(args_list)
