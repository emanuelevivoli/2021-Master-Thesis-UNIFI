import io
from tqdm.auto import tqdm  # custom progress bar
import json
import os
from typing import List

from thesis.config.datasets import S2orcConfig
from thesis.config.execution import LogConfig
from thesis.config.base import fingerprints

from thesis.utils.cache import _caching, no_caching

import logging


def read_meta_json_list_dict(
    dataset_config: S2orcConfig, log_config: LogConfig, file_path: str
):
    """
    Args: \\
        - `config`: instance of tha class S2orcConfig       \\
        - `file_path`: path to the `json` file              \\
        - `verbose` [def. False]: `print` all loggs         \

    """

    if log_config.verbose:
        logging.info(f"file_path: {file_path}")
    # list of dictionaries, one for each row in pdf_parses
    json_list_of_dict = []
    # create a list index
    json_list_of_dict_idx = 0
    # dictionary of indexes, to obtain the object from list, starting from the `paper_id`
    json_dict_of_index = {}

    if dataset_config.zipped:
        # import for unzip
        import gzip

        # open by firstly unzipping it
        gz = gzip.open(file_path, "rb")
        input_json = io.BufferedReader(gz)
    else:
        # just open as usual
        input_json = open(file_path, "r")
        if log_config.verbose:
            logging.info("You choose to only use unzipped files")

    # check if ["mag_field_of_study"] is in dataset_config, and is valid
    mag_field_filter = False
    mag_field_all = False
    mag_field_dict = {}
    if type(dataset_config.mag_field_of_study) is list:
        if log_config.verbose:
            logging.info("Mag field is a list!")
        mag_field_filter = True
        if not dataset_config.mag_field_of_study:
            if log_config.verbose:
                logging.info("Mag field List is empty!")
            mag_field_all = True
        else:
            if log_config.verbose:
                logging.info(
                    "Mag field List is not empty: inizializing dictionary")
            for field in dataset_config.mag_field_of_study:
                mag_field_dict[field] = True
    elif type(dataset_config.mag_field_of_study) is str:
        if log_config.verbose:
            logging.info("Mag field is a str!")
        if (
            "," not in dataset_config.mag_field_of_study
            and "[" not in dataset_config.mag_field_of_study
            and "]" not in dataset_config.mag_field_of_study
        ):
            if log_config.verbose:
                logging.info("Mag field Str is only one element!")
            mag_field_dict[config.mag_field_of_study] = True
            dataset_config.mag_field_of_study = [
                dataset_config.mag_field_of_study]
        else:
            if log_config.verbose:
                logging.info(
                    "Mag field Str is corrupted, all mag field will be taken!")
            mag_field_all = True
            dataset_config.mag_field_of_study = []
    else:
        if log_config.verbose:
            logging.info(
                "You have choosen a wrong format for magfield! Skipped control, all field will be taken"
            )
        mag_field_all = True
        dataset_config.mag_field_of_study = []

    assert (
        dataset_config.mag_field_of_study is not list
    ), "config.mag_field_of_study must be *list* type"

    with input_json:
        for index, json_line in tqdm(enumerate(input_json)):
            json_dict = json.loads(json_line)

            mag_field_filter_pass = (
                sum(
                    [
                        True if json_field in mag_field_dict else False
                        for json_field in json_dict["mag_field_of_study"]
                    ]
                )
                >= 1
                if json_dict["mag_field_of_study"] is not None
                else False
            )

            if not mag_field_filter or mag_field_all or mag_field_filter_pass:
                # append the dictionary to the dictionaries' list
                json_list_of_dict.append(json_dict)
                # insert (paper_id, index) pair as (key, value) to the dictionary
                json_dict_of_index[json_dict["paper_id"]
                                   ] = json_list_of_dict_idx
                # increment the list index
                json_list_of_dict_idx += 1

    return json_list_of_dict, json_dict_of_index


def read_pdfs_json_list_dict(
    dataset_config, log_config, file_path, json_dict_of_index_meta
):
    """
    Args: \\
        - `config`: instance of tha class S2orcConfig       \\
        - `file_path`: path to the `json` file              \\
        - `json_dict_of_index_meta`: ?                      \\
        - `verbose` [def. False]: `print` all loggs         \

    """

    if log_config.verbose:
        logging.info(f"file_path: {file_path}")
    # list of dictionaries, one for each row in pdf_parses
    json_list_of_dict = []
    # create a list index
    json_list_of_dict_idx = 0
    # dictionary of indexes, to obtain the object from list, starting from the `paper_id`
    json_dict_of_index = {}

    if dataset_config.zipped:
        # import for unzip
        import gzip

        # open by firstly unzipping it
        gz = gzip.open(file_path, "rb")
        input_json = io.BufferedReader(gz)
    else:
        # just open as usual
        input_json = open(file_path, "r")
        if log_config.verbose:
            logging.info("You choose to only use unzipped files")

    with input_json:
        for index, json_line in tqdm(enumerate(input_json)):
            json_dict = json.loads(json_line)

            # if log_config.verbose: logging.info(json_dict.keys())

            # if the metadata has been selected
            if json_dict["paper_id"] in json_dict_of_index_meta:
                # append the dictionary to the dictionaries' list
                json_list_of_dict.append(json_dict)
                # insert (paper_id, index) pair as (key, value) to the dictionary
                json_dict_of_index[json_dict["paper_id"]
                                   ] = json_list_of_dict_idx
                # increment the list index
                json_list_of_dict_idx += 1

    return json_list_of_dict, json_dict_of_index


def s2orc_chunk_read(
    dataset_config: S2orcConfig,
    log_config: LogConfig,
    meta_s2orc_single_file,
    pdfs_s2orc_single_file,
) -> dict:
    """
    Args:   \\
        - `s2orc_path` (string): \\
            Path to the Dataset directory (es. '{data}/s2orc-{sample|full}-20200705v1/{sample|full}').  \\
        - `meta_s2orc_single_file` (string): \\
            Filenames with extentions (es. 'sample_0.jsonl') present in `{dataset_path}/metadata`.      \\
        - `pdfs_s2orc_single_file` (string): 
            Filenames with extentions (es. 'sample_0.jsonl') present in `{dataset_path}/pdf_parses`.    \\
        - `extention` (string | None):
            String element either `jsonl` (only decompressed files) or `jsonl.gz` (to decompress files).\\
        - `args` (dict):
            Dictionary containing some config params.                                                   \\
        - `log_config.verbose` [def. False]                                                                        \\
                                                                                                        \\
    Return: \\
        - `json_dict` (list of dict): 
            Dictionary such as: { 'metadata': [...], 'pdf_parses': [...], 'meta_key_idx': {...}, 'pdf_key_idx': {...}} with objects of type 
            metadata_CLASS and pdf_parses_CLASS respectively.                                           \
    """

    if log_config.verbose:
        logging.info("[INFO-START] Metadata Chunk read  : ",
                     meta_s2orc_single_file)
    if log_config.verbose:
        logging.info("[          ] Pdf parses Chunk read: ",
                     pdfs_s2orc_single_file)

    @no_caching(
        dataset_config=dataset_config,
        meta_s2orc_single_file=meta_s2orc_single_file,
        pdfs_s2orc_single_file=pdfs_s2orc_single_file,
        function_name="s2orc_chunk_read",
    )
    def _s2orc_chunk_read(
        dataset_config, log_config, meta_s2orc_single_file, pdfs_s2orc_single_file
    ) -> dict:

        json_dict_of_list: dict = {
            "metadata": [],
            "pdf_parses": [],
            "meta_key_idx": {},
            "pdf_key_idx": {},
        }

        if log_config.verbose:
            logging.info("[INFO] Metadata read: ", meta_s2orc_single_file)
        path_metadata = os.path.join(
            dataset_config.path, "metadata", meta_s2orc_single_file
        )
        if log_config.verbose:
            logging.info(f"{path_metadata}")

        json_list_metadata, json_dict_of_index_meta = read_meta_json_list_dict(
            dataset_config, log_config, path_metadata
        )

        json_dict_of_list["metadata"] = json_list_metadata
        json_dict_of_list["meta_key_idx"] = json_dict_of_index_meta

        if log_config.verbose:
            logging.info("[INFO] Pdf_Parses read: ", pdfs_s2orc_single_file)
        path_pdf_parses = os.path.join(
            dataset_config.path, "pdf_parses", pdfs_s2orc_single_file
        )
        if log_config.verbose:
            logging.info(f"{path_pdf_parses}")

        json_list_pdf_parses, json_dict_of_index_pdf = read_pdfs_json_list_dict(
            dataset_config, log_config, path_pdf_parses, json_dict_of_index_meta
        )

        json_dict_of_list["pdf_parses"] = json_list_pdf_parses
        json_dict_of_list["pdf_key_idx"] = json_dict_of_index_pdf

        if log_config.verbose:
            logging.info(
                f"[INFO] json_dict_of_list len: {len(json_dict_of_list)}")
        if log_config.verbose:
            logging.info(
                "[INFO-END  ] Chunk read: ",
                meta_s2orc_single_file,
                pdfs_s2orc_single_file,
                "\n\n\n",
            )

        return json_dict_of_list

    json_dict_of_list = _s2orc_chunk_read(
        dataset_config, log_config, meta_s2orc_single_file, pdfs_s2orc_single_file
    )

    if log_config.verbose:
        logging.info("[        ] Metadata Chunk read  : ",
                     meta_s2orc_single_file)
    if log_config.verbose:
        logging.info("[INFO-END] Pdf parses Chunk read: ",
                     pdfs_s2orc_single_file)

    return json_dict_of_list


def s2orc_multichunk_read(
    dataset_config: S2orcConfig,
    log_config: LogConfig,
    toread_meta_s2orc,
    toread_pdfs_s2orc,
):
    """
    Args:   \\
        - `config` (string): 
            Object containing all params for s2orc.                                                         \\
        - `toread_meta_s2orc` (list of string): 
            List of filenames with extentions (es. ['metadata_0.jsonl', 'metadata_1.jsonl'])
            present in `{dataset_path}/metadata`.                                                           \\
        - `toread_pdfs_s2orc` (list of string): 
            List of filenames with extentions (es. ['pdf_parses_0.jsonl', 'pdf_parses_1.jsonl'])
            present in `{dataset_path}/pdf_parses`.                                                         \\
        - extention (string | None):
            String element either `jsonl` (only decompressed files) or `jsonl.gz` (to decompress files).    \\
        - `verbose` [def. False]                                                                            \\
                                                                                                            \\
    Return: \\
        - `multichunks_lists` (list of dict): 
            List of Dictionary such as: 
            { 'metadata': [...], 'pdf_parses': [...], 'meta_key_idx': {}, 'pdf_key_idx': {} } 
            with objects of type metadata_CLASS and pdf_parses_CLASS respectively.                          \
    """

    if log_config.verbose:
        logging.info("[INFO-START] Multichunk read")
    if log_config.verbose:
        logging.info(f"[INFO] Metadata reading  : {toread_meta_s2orc}")
    if log_config.verbose:
        logging.info(f"[INFO] Pdf Parses reading: {toread_pdfs_s2orc}")

    assert len(toread_meta_s2orc) == len(
        toread_pdfs_s2orc
    ), "Files list (metadata and pdfs) must be the same length!"
    assert dataset_config.extention is not None, "Extention must be set!"

    if dataset_config.s2orc_type == 'full' and log_config.verbose:
        logging.info(
            f"[INFO] Data read selection : \n \
                            [{'x' if not dataset_config.zipped else ' '}] Extracted \n \
                            [{'x' if dataset_config.zipped else ' '}] To Extract \n \
                        Only files already extracted will be analyzed."
        )

    # **(dataset_config.get_configuration())
    @no_caching(
        sorted(toread_meta_s2orc),
        sorted(toread_pdfs_s2orc),
        **fingerprints(dataset_config),
        function_name="s2orc_multichunk_read",
    )
    def _s2orc_multichunk_read(
        dataset_config, log_config, toread_meta_s2orc, toread_pdfs_s2orc
    ):

        multichunks_lists: List[dict] = []

        for meta_s2orc_single_file, pdfs_s2orc_single_file in tqdm(
            zip(toread_meta_s2orc, toread_pdfs_s2orc)
        ):

            chunk_list = s2orc_chunk_read(
                dataset_config,
                log_config,
                meta_s2orc_single_file,
                pdfs_s2orc_single_file,
            )

            multichunks_lists.append(chunk_list)

        return multichunks_lists

    multichunks_lists = _s2orc_multichunk_read(
        dataset_config, log_config, toread_meta_s2orc, toread_pdfs_s2orc
    )

    if log_config.verbose:
        logging.info("[INFO-END] Multichunk read")

    return multichunks_lists
