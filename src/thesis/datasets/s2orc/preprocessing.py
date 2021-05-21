import pandas as pd

# typing
from typing import Dict, List, Union

# 🤗 Transformers
from transformers import PreTrainedTokenizer

# 🤗 Datasets
from datasets import DatasetDict, Dataset as hfDataset

# torch
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np

from thesis.datasets.s2orc.mag_field import mag_field_dict

from thesis.config.base import fingerprints, Config
from thesis.config.datasets import S2orcConfig
from thesis.config.execution import RunConfig, LogConfig

from thesis.utils.cache import no_caching, _caching

import logging


def key_value_sort(dictionary: Dict) -> List:
    return sorted([[k, v] for k, v in dictionary.items()])


def fuse_dictionaries(
    single_chunk: dict, data_field: List[str], log_config: LogConfig
) -> Dataset:

    # definition of **single_chunk**
    # {'metadata': [], 'pdf_parses': [], 'meta_key_idx': {}, 'pdf_key_idx': {}}

    @no_caching(
        key_value_sort(single_chunk["meta_key_idx"]),
        key_value_sort(single_chunk["pdf_key_idx"]),
        data_field,
    )
    def _fuse_dictionaries(
        single_chunk: dict, data_field: List[str], log_config: LogConfig
    ) -> Dataset:

        if log_config.verbose:
            logging.info(
                f"[INFO] len meta single_chunk: {len(single_chunk['metadata'])}")
        if log_config.verbose:
            logging.info(
                f"[INFO] len pdfs single_chunk: {len(single_chunk['pdf_parses'])}")

        paper_list = []
        for key in single_chunk["meta_key_idx"]:

            if log_config.debug:
                logging.info(
                    f"[INFO] Analyse metadata dictionary for paper {key}")
            # get metadata dictionary for paper with paper_id: key
            meta_index = single_chunk["meta_key_idx"].get(key, None)
            if log_config.debug:
                logging.info(f"       meta_index: {meta_index}")
            metadata = (
                single_chunk["metadata"][meta_index]
                if meta_index is not None
                else dict()
            )

            if log_config.debug:
                logging.info(
                    f"[INFO] Analyse pdf_parses dictionary for paper {key}")
            # get pdf_parses dictionary for paper with paper_id: key
            pdf_index = single_chunk["pdf_key_idx"].get(key, None)
            if log_config.debug:
                logging.info(f"       pdf_index: {pdf_index}")
            pdf_parses = (
                single_chunk["pdf_parses"][pdf_index]
                if pdf_index is not None
                else dict()
            )

            def not_None(element):
                """
                    Here we see if the element is None, '' or [] 
                    considering it to be Falsy type, in python.
                """
                if element == None:
                    return False
                elif type(element) == str and element is "":
                    return False
                elif type(element) == list and element is []:
                    return False
                return True

            def fuse_field(meta_field, pdf_field):
                """
                    With inspiration from https://docs.python.org/3/library/stdtypes.html#truth-value-testing
                    both '' and `None` seems to be Falsy type, in python.
                """

                class s2orcBaseElement:
                    """
                        'section': str,
                        'text': str,
                        'cite_spans': list,
                        'ref_spans': list                
                    """

                    def __init__(self, dictionary):
                        self.section: str = dictionary["section"]
                        self.text: str = dictionary["text"]
                        self.cite_spans: list = dictionary["cite_spans"]
                        self.ref_spans: list = dictionary["ref_spans"]

                    def get_text(self):
                        return self.text

                if type(pdf_field) == list:
                    pdf_field = " ".join(
                        [s2orcBaseElement(elem).get_text()
                         for elem in pdf_field]
                    )

                return meta_field if not_None(meta_field) else pdf_field

            if log_config.debug:
                logging.info(f"[INFO] Start fusion for paper {key}")
            paper = dict()
            for field in data_field:
                if log_config.debug:
                    logging.info(
                        f"[INFO] Fusing field {field} for meta ({metadata.get(field, None)}) and pdf_parses ({pdf_parses.get(field, None)})"
                    )
                paper[field] = fuse_field(
                    metadata.get(field, None), pdf_parses.get(field, None)
                )

            paper_list.append(paper)

            if log_config.debug:
                logging.info(f"[INFO] Deleting meta and pdf for paper {key}")
            #  if meta_index is not None: del single_chunk['metadata'][meta_index]
            # if pdf_index is not None: del single_chunk['pdf_parses'][pdf_index]

        # Dataset.from_pandas(my_dict) could be a good try if we only convert our paper_list to Pandas Dataframes
        paper_df = pd.DataFrame(paper_list)

        return hfDataset.from_pandas(paper_df)

    dataset = _fuse_dictionaries(single_chunk, data_field, log_config)

    return dataset


def get_dataset(
    single_chunk: dict,
    dataset_config: S2orcConfig,
    run_config: RunConfig,
    log_config: LogConfig,
    data_field: List[str] = ["title", "abstract"],
) -> Dict[str, DataLoader]:
    """Given an input file, prepare the train, test, validation dataloaders.
    :param single_chunk: `SingleChunk`, input file related to one chunk (format list)
    :param dataset_config: `S2orcConfig`, pretrained tokenizer that will prepare the data, i.e. convert tokens into IDs
    :param run_config: `RunConfig`, if set, seed for split train/val/test
    :param log_config: `LogConfig`, batch size for the dataloaders
    :param data_field: `List[str] `, number of CPU workers to use during dataloading. On Windows this must be zero
    :return: a dictionary containing train, test, validation dataloaders
    """
    # **(dataset_config.get_fingerprint()), **(run_config.get_fingerprint()), **(log_config.get_fingerprint())
    @no_caching(
        key_value_sort(single_chunk["meta_key_idx"]),
        key_value_sort(single_chunk["pdf_key_idx"]),
        **fingerprints(dataset_config, run_config, log_config),
        function_name="get_dataset",
    )
    def _get_dataset(
        single_chunk: dict,
        dataset_config: S2orcConfig,
        run_config: RunConfig,
        log_config: LogConfig,
        data_field: List[str],
    ) -> DatasetDict:

        ## ------------------ ##
        ## -- LOAD DATASET -- ##
        ## ------------------ ##
        if log_config.time:
            start = time.time()
        if log_config.time:
            start_load = time.time()

        # execution
        dataset_dict = fuse_dictionaries(single_chunk, data_field, log_config)

        #  logging.info(dataset_dict)

        if log_config.debug:
            logging.info(dataset_dict)

        if log_config.time:
            end_load = time.time()
        if log_config.time:
            logging.info(f"[TIME] load_dataset: {end_load - start_load}")

        ## ------------------ ##
        ## ---- MANAGING ---- ##
        ## ------------------ ##
        if log_config.time:
            start_selection = time.time()

        # execution
        dataset = dataset_dict  # ['train']

        if log_config.time:
            end_selection = time.time()
        if log_config.time:
            logging.info(
                f"[TIME] dataset_train selection: {end_selection - start_selection}")
        if log_config.debug:
            logging.info(dataset)

        ## ------------------ ##
        ## --- REMOVE none -- ##
        ## ------------------ ##
        if log_config.time:
            start_removing = time.time()
        # clean input removing papers with **None** as abstract/title
        if not dataset_config.keep_none_papers:

            ## --------------------- ##
            ## --- REMOVE.indexes -- ##
            ## --------------------- ##
            if log_config.time:
                start_removing_indexes = time.time()
            if log_config.debug:
                logging.info(data_field)

            # execution
            none_papers_indexes = {}
            for field in data_field:
                none_indexes = [
                    idx_s for idx_s, s in enumerate(dataset[f"{field}"]) if s is None
                ]
                none_papers_indexes = {
                    **none_papers_indexes,
                    **dict.fromkeys(none_indexes, False),
                }

            if log_config.time:
                end_removing_indexes = time.time()
            if log_config.time:
                logging.info(
                    f"[TIME] remove.indexes: {end_removing_indexes - start_removing_indexes}"
                )
            if log_config.debug:
                logging.info(none_papers_indexes)

            ## --------------------- ##
            ## --- REMOVE.concat --- ##
            ## --------------------- ##
            if log_config.time:
                start_removing_concat = time.time()

            # execution
            to_remove_indexes = list(none_papers_indexes.keys())

            if log_config.time:
                end_removing_concat = time.time()
            if log_config.time:
                logging.info(
                    f"[TIME] remove.concat: {end_removing_concat - start_removing_concat}"
                )
            if log_config.debug:
                logging.info(to_remove_indexes)
            if log_config.debug:
                logging.info([dataset["abstract"][i]
                             for i in to_remove_indexes])

            ## --------------------- ##
            ## --- REMOVE.filter --- ##
            ## --------------------- ##
            if log_config.time:
                start_removing_filter = time.time()

            # execution
            dataset = dataset.filter(
                (lambda x, ids: none_papers_indexes.get(ids, True)), with_indices=True
            )

            if log_config.time:
                end_removing_filter = time.time()
            if log_config.time:
                logging.info(
                    f"[TIME] remove.filter: {end_removing_filter - start_removing_filter}"
                )
            if log_config.debug:
                logging.info(dataset)

        if log_config.time:
            end_removing = time.time()
        if log_config.time:
            logging.info(
                f"[TIME] remove None fields: {end_removing - start_removing}")

        ## --------------------- ##
        ## --- REMOVE.column --- ##
        ## --------------------- ##
        if log_config.time:
            start_remove_unused_columns = time.time()
        if not dataset_config.keep_unused_columns:

            for column in dataset.column_names:
                if column not in data_field:
                    if log_config.debug:
                        logging.info(f"{column}")
                    dataset.remove_columns_(column)

        if log_config.time:
            end_remove_unused_columns = time.time()
        if log_config.time:
            logging.info(
                f"[TIME] remove.column: {end_remove_unused_columns - start_remove_unused_columns}"
            )

        ## ------------------ ##
        ## --- SPLIT 1.    -- ##
        ## ------------------ ##
        if log_config.time:
            start_first_split = time.time()

        # 80% (train), 20% (test + validation)
        # execution
        train_testvalid = dataset.train_test_split(
            test_size=0.2, seed=run_config.seed)

        if log_config.time:
            end_first_split = time.time()
        if log_config.time:
            logging.info(
                f"[TIME] first [train-(test-val)] split: {end_first_split - start_first_split}"
            )

        ## ------------------ ##
        ## --- SPLIT 2.    -- ##
        ## ------------------ ##
        if log_config.time:
            start_second_split = time.time()

        # 10% of total (test), 10% of total (validation)
        # execution
        test_valid = train_testvalid["test"].train_test_split(
            test_size=0.5, seed=run_config.seed
        )

        if log_config.time:
            end_second_split = time.time()
        if log_config.time:
            logging.info(
                f"[TIME] second [test-val] split: {end_second_split - start_second_split}"
            )

        # execution
        dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "test": test_valid["test"],
                "valid": test_valid["train"],
            }
        )
        if log_config.time:
            end = time.time()
        if log_config.time:
            logging.info(f"[TIME] TOTAL: {end - start}")

        return dataset

    dataset = _get_dataset(
        single_chunk, dataset_config, run_config, log_config, data_field
    )

    return dataset


def data_target_preprocess(
    *sentences_by_column,
    data,
    target,
    classes,
    max_seq_length,
    tokenizer,
    debug=False,
    print_all_debug=False,
    time_debug=False,
    print_some_debug=False,
    **kwargs,
):  # -> Dict[str, Union[list, Tensor]]:
    """Preprocess the raw input sentences from the text file.
    :param sentences: a list of sentences (strings)
    :return: a dictionary of "input_ids"
    """

    if debug:
        logging.info(
            f"[INFO-START] Preprocess on data: {data}, target: {target}")

    assert data == ["abstract"], "data should be ['abstract']"
    if debug:
        logging.info(data)
    assert target == ["title"], "target should be ['title']"
    if debug:
        logging.info(target)

    data_columns_len = len(data)
    target_columns_len = len(target)
    columns_len = data_columns_len + target_columns_len

    assert data_columns_len == 1, "data length should be 1"
    if debug:
        logging.info(data_columns_len)
    assert target_columns_len == 1, "target length should be 1"
    if debug:
        logging.info(target_columns_len)

    sentences_by_column = np.asarray(sentences_by_column)
    input_columns_len = len(sentences_by_column)

    if debug:
        logging.info(
            f"all sentences (len {input_columns_len}): {sentences_by_column}")

    if target_columns_len == 0:
        raise NameError(
            "No target variable selected, \
                    are you sure you don't want any target?"
        )

    data_sentences = sentences_by_column[0]
    target_sentences = sentences_by_column[
        1
    ]  # if columns_len == input_columns_len else sentences_by_column[data_columns_len:-1]

    if debug:
        logging.info(data_sentences)
    if debug:
        logging.info(target_sentences)

    # The sequences are not padded here. we leave that to the dataloader in a collate_fn
    # ----------------------------------------------- #
    # -------- TODO include the `collate_fn` -------- #
    # ----------------------------------------------- #
    # That means: a bit slower processing, but a smaller saved dataset size
    if print_some_debug:
        logging.info(max_seq_length)

    data_encoded_d = tokenizer(
        text=data_sentences.tolist(),
        # add_special_tokens=False,
        # is_pretokenized=True,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_token_type_ids=False,
        return_attention_mask=False,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
        return_tensors="np",
    )

    target_encoded_d = tokenizer(
        text=target_sentences.tolist(),
        # add_special_tokens=False,
        #  is_pretokenized=True,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_token_type_ids=False,
        return_attention_mask=False,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
        return_tensors="np",
    )

    if debug:
        logging.info(data_encoded_d["input_ids"].shape)
    if debug:
        logging.info(target_encoded_d["input_ids"].shape)
    # return encoded_d

    return {
        "data_input_ids": data_encoded_d["input_ids"],
        "target_input_ids": target_encoded_d["input_ids"],
    }


def mag_preprocess(*mags):
    """Preprocess the raw input sentences from the text file.
    :param sentences: a list of sentences (strings)
    :return: a dictionary of "input_ids"
    """
    debug = False

    if debug:
        logging.info(f"[INFO-START] Mag Preprocess")

    mag_field = np.array(mags)
    input_columns_len = mag_field.shape
    if debug:
        logging.info(f"pre flatten (len {input_columns_len}): {mag_field}")
    if debug:
        logging.info(f"pre types: {[type(ele) for ele in mag_field]}")
    if debug:
        logging.info(f"pre types: {type(mag_field)}")

    mag_field = mag_field.flatten()
    input_columns_len = mag_field.shape
    if debug:
        logging.info(f"after flatten (len {input_columns_len}): {mag_field}")
    if debug:
        logging.info(f"after types: {[type(ele) for ele in mag_field]}")
    if debug:
        logging.info(f"after types: {type(mag_field)}")

    mag_field = np.array(
        [ele if type(ele) == str else list(ele)[0] for ele in mag_field]
    )

    if input_columns_len == 0:
        raise NameError(
            "No mag variable selected, \
                    are you sure you don't want any target?"
        )

    if debug:
        logging.info(mag_field)
    if debug:
        logging.info(mag_field_dict)
    if debug:
        logging.info(
            [
                mag_field_dict.get(real_mag_field_value, 3)
                for real_mag_field_value in mag_field
            ]
        )

    mag_index = np.asarray(
        [
            mag_field_dict.get(real_mag_field_value, 3)
            for real_mag_field_value in mag_field
        ]
    )

    if debug:
        logging.info(mag_index)

    return {"mag_index": mag_index}


def preprocessing(
    all_datasets,
    dictionary_input,
    dictionary_columns,
    tokenizer,
    model,
    *args,
    **kwargs,
):  # model -> max_seq_length
    """
    Args:
        - `all_datasets`: DatasetDict element divided into 'train', 'test', 'valid'.
        - `dictionary_input`: dict element with fields 'data', 'target' and 'classes'.
        - `dictionary_columns`: list of values of the `dictionary_input` dict.
        - `tokenizer`: tokenizer used for tokenize words.
        - `model`: model used to get 'max_position_embeddings' value.
        - `*args`: some extra params not used
        - `**kwargs`: some extra dictionary params not used    
    """

    all_datasets_map = all_datasets.map(
        data_target_preprocess,
        input_columns=dictionary_columns,
        fn_kwargs={
            **dictionary_input,
            "max_seq_length": model.config.max_position_embeddings,
            "tokenizer": tokenizer,
        },
        batched=True,
    )

    all_dataset_mag_map = all_datasets_map.map(
        mag_preprocess, input_columns=dictionary_input["classes"], batched=True
    )

    return all_dataset_mag_map
