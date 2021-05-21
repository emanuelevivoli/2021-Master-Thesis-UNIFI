# Parsers
from thesis.parsers.defaults import (
    DataTrainingArguments,
    TrainingArguments,
)

# 🤗 Transformers
from transformers import AutoTokenizer
from datasets import Dataset

from typing import List

from thesis.utils.cache import _caching, no_caching


def speed_tokenization(dataset_args: DataTrainingArguments,
                       training_args: TrainingArguments,
                       tokenizer: AutoTokenizer,
                       max_seq_length: int,
                       datasets: Dataset,
                       text_column_name: str,
                       column_names: List[str]) -> Dataset:

    @no_caching(
        function_name="speed_tokenization"
    )
    def _speed_tokenization(dataset_args: DataTrainingArguments,
                            training_args: TrainingArguments,
                            tokenizer: AutoTokenizer,
                            max_seq_length: int,
                            datasets: Dataset,
                            text_column_name: str,
                            column_names: List[str]):

        if dataset_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if training_args.pad_to_max_length else False

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

            tokenized_datasets = datasets.map(
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

            tokenized_datasets = datasets.map(
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
                total_length = len(
                    concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                total_length = (
                    total_length // max_seq_length) * max_seq_length
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

        return tokenized_datasets

    return _speed_tokenization(dataset_args, training_args, tokenizer, max_seq_length, datasets, text_column_name, column_names)
