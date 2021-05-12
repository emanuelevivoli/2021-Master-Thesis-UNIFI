import os
import subprocess
import logging
from typing import List, Union, Dict
from ..cache import _caching
from .base import Config

DATASET_PATH = "/home/vivoli/Thesis/data"
CUSTOM_DATASETS = ["s2orc", "keyphrase"]

S2ORC_TYPES = ["sample", "full"]

KEYPH_TYPES = ["inspec", "krapivin", "nus",
               "semeval", "kp20k", "duc", "stackexchange"]

JURNL_TYPES = ["icdar_19", "icpr_20", "ijdar_20"]


class CustomConfig(Config):
    """Configuration class, base for `S2orcConfig` and `KeyPHConfig`:
    - dataset_path `dataset_path` (string), the path for the dataset
    """

    # global variable initialization
    # the only common variable is the main dataset path
    def __init__(self, *args, **kwargs):
        self.dataset_path: str = kwargs["dataset_path"]

    def str_to_list(self, string: str) -> List[Union[str, int]]:
        return (
            string.split(",") if "," in string else [string]
        )  # string.split(' ') if ' ' in string else


class S2orcConfig(CustomConfig):
    """S2orc Configuration class.

    Properties:
    - types (`unused`), `S2ORC_TYPES` list composed (eg. 'sample', 'full')
    - s2orc_type 'dataset_config_name' (`str`) (one of the `types` above)
    - idxs 'idxs' (str,str... -> `List[str]`) (eg. '0,1,2,3' -> ['0', '1', '2', '3'])
    - zipped 'zipped' (`bool`)
    - mag_field_of_study 'mag_field_of_study' (str,str... -> `List[str]`), mag value (see documentation) for keeping only papers from selected mag (field of study)
        - (eg. 'Computer Science', 'Mathematics', 'Physics')
    - dictionary_input `dict`, composed by field `data`, `target` and `classes`:
        - data `List[str]`, usually 'abstract' (for summarization task)
        - target `List[str]`, usually 'title' (for summarization task)
        - classes `List[str]`, usually 'mag_field_of_study' (for classification task)
    - keep_none_papers `bool`, if False, we discard none papers from the `Dataset`
    - keep_unused_columns `bool`, if False, we delete `Dataset` columns that are not in Union[data, target, classes]
    - path (str), actually, depending on the `s2orc_type` choosed, it is: "`dataset_path`/s2orc-`s2orc_type`-20200705v1/`s2orc_type`"
    - extention `str`, can be 'jsonl' or 'jsonl.gz' for compressed files

    Calculated Properties:
    - metadata_filenames `List` and pdf_parses_filenames `List` 
        - (eg. [metadata_0, metadata_1] and  [pdf_parses_0, pdf_parses_1] )
    - completed_metadata_filenames `List` and completed_pdf_parses_filenames `List` 
        - (eg. [metadata_0.jsonl, metadata_1.jsonl] and [pdf_parses_0.jsonl, pdf_parses_1.jsonl] )
    - metadata_output `List` and pdf_parses_output `List` 
        - (eg. [metadata_0.jsonl, metadata_1.jsonl, ..., metadata_99.jsonl] and [pdf_parses_0.jsonl, pdf_parses_1.jsonl, ...,  pdf_parses_99.jsonl])
    - toread_meta_s2orc `List` and toread_pdfs_s2orc `List` 
        - (eg. [metadata_0.jsonl, metadata_1.jsonl]  [pdf_parses_0.jsonl, pdf_parses_1.jsonl] )
    """

    types: List[str] = S2ORC_TYPES

    extention: str
    dictionary_input: dict

    # --- without extention
    # e.g. [metadata_0, metadata_1]  [pdf_parses_0, pdf_parses_1]
    metadata_filenames: List[str]
    pdf_parses_filenames: List[str]

    # --- with extention
    # e.g. [metadata_0.jsonl, metadata_1.jsonl]  [pdf_parses_0.jsonl, pdf_parses_1.jsonl]
    completed_metadata_filenames: List[str]
    completed_pdf_parses_filenames: List[str]

    # --- extisted in the folder
    # e.g. [metadata_0.jsonl, metadata_1.jsonl, ..., metadata_99.jsonl]
    # e.g. [pdf_parses_0.jsonl, pdf_parses_1.jsonl, ...,  pdf_parses_99.jsonl]
    metadata_output: List[str]
    pdf_parses_output: List[str]

    # --- intersection between what we want (`completed_metadata_filenames`) and what there is (`metadata_output`)
    # e.g. [metadata_0.jsonl, metadata_1.jsonl]  [pdf_parses_0.jsonl, pdf_parses_1.jsonl]
    toread_meta_s2orc: List[str]
    toread_pdfs_s2orc: List[str]

    def __init__(self, *args, **kwargs):
        # inizialize the main dataset path
        super().__init__(*args, **kwargs)

        self.s2orc_type: str = kwargs[
            "dataset_config_name"
        ]  # options are 'sample' or 'full'
        self.idxs: list = self.str_to_list(
            kwargs["idxs"]
        )  # options are empty List() [for sample] or List(int) [for full]
        self.zipped: bool = kwargs[
            "zipped"
        ]  # if False, only idxs for unzipped files. if True, we extract idxs files
        self.mag_field_of_study: list = self.str_to_list(
            kwargs["mag_field_of_study"]
        )  # options are empty List() or List(string)
        self.dictionary_input: dict = {
            "data": self.str_to_list(kwargs["data"]),
            "target": self.str_to_list(kwargs["target"]),
            "classes": self.str_to_list(kwargs["classes"]),
        }

        # boolean flags for elaborate papers fields
        self.keep_none_papers: bool = kwargs["keep_none_papers"]
        self.keep_unused_columns: bool = kwargs["keep_unused_columns"]

        # path for the dataset
        self.path: str = f"{self.dataset_path}/s2orc-{self.s2orc_type}-20200705v1/{self.s2orc_type}"

        self.extention = "jsonl.gz" if self.zipped else "jsonl"

        # print(self.idxs)
        # print(self.mag_field_of_study)
        # print(self.dictionary_input)

    def get_fingerprint(self) -> Dict:
        return dict(
            {
                "s2orc_type": self.s2orc_type,
                "idxs": self.idxs,
                "zipped": self.zipped,
                "mag_field_of_study": self.mag_field_of_study,
                "dictionary_input": dict(self.dictionary_input),
                "path": self.path,
                "extention": self.extention,
                "keep_none_papers": self.keep_none_papers,
                "keep_unused_columns": self.keep_unused_columns,
            }
        )

    def reset(self):
        # extention left
        #  self.extention = None

        self.metadata_filenames = []
        self.pdf_parses_filenames = []

        # --- with extention
        # e.g. [metadata_0.jsonl, metadata_1.jsonl]  [pdf_parses_0.jsonl, pdf_parses_1.jsonl]
        self.completed_metadata_filenames = []
        self.completed_pdf_parses_filenames = []

        # --- extisted in the folder
        # e.g. [metadata_0.jsonl, metadata_1.jsonl, ..., metadata_99.jsonl]
        # e.g. [pdf_parses_0.jsonl, pdf_parses_1.jsonl, ...,  pdf_parses_99.jsonl]
        self.metadata_output = []
        self.pdf_parses_output = []

        # toread_meta_s2orc and toread_pdfs_s2orc left
        # self.toread_meta_s2orc = []
        # self.toread_pdfs_s2orc = []

    def get_filenames(self, verbose=False):
        @_caching(
            idxs=self.idxs, s2orc_type=self.s2orc_type, function_name="get_filenames"
        )
        def _get_filenames(idxs, s2orc_type):

            metadata_filenames = (
                [f"metadata_{n}" for n in idxs]
                if s2orc_type == "full"
                else ["sample"]
            )
            pdf_parses_filenames = (
                [f"pdf_parses_{n}" for n in idxs]
                if s2orc_type == "full"
                else ["sample"]
            )

            return metadata_filenames, pdf_parses_filenames

        self.metadata_filenames, self.pdf_parses_filenames = _get_filenames(
            self.idxs, self.s2orc_type
        )

        if verbose:
            logging.info(
                f"get_filenames:\n metadata_filenames: {self.metadata_filenames}\n pdf_parses_filenames: {self.pdf_parses_filenames}\n"
            )

        return self.metadata_filenames, self.pdf_parses_filenames

    def get_extention(self, verbose=False):
        # if it's not set yet
        if self.extention is None:
            self.extention = "jsonl.gz" if self.zipped else "jsonl"

        if verbose:
            logging.info(f"extention: {self.extention}")
        return self.extention

    def get_completed_filenames(self, verbose=False):

        # prerequisites
        self.get_extention()
        self.get_filenames()

        @_caching(
            extention=self.extention,
            metadata_filenames=self.metadata_filenames,
            pdf_parses_filenames=self.pdf_parses_filenames,
            function_name="get_completed_filenames",
        )
        def _get_completed_filenames(
            extention, metadata_filenames, pdf_parses_filenames
        ):
            # update filenames with extention
            completed_metadata_filenames = [
                f"{metadata_filename}.{extention}"
                for metadata_filename in metadata_filenames
            ]
            completed_pdf_parses_filenames = [
                f"{pdf_parses_filename}.{extention}"
                for pdf_parses_filename in pdf_parses_filenames
            ]
            return completed_metadata_filenames, completed_pdf_parses_filenames

        (
            self.completed_metadata_filenames,
            self.completed_pdf_parses_filenames,
        ) = _get_completed_filenames(
            self.extention, self.metadata_filenames, self.pdf_parses_filenames
        )

        if verbose:
            logging.info(
                f"get_completed_filenames:\n completed_metadata_filenames: {self.completed_metadata_filenames}\n completed_pdf_parses_filenames: {self.completed_pdf_parses_filenames}\n"
            )

        return self.completed_metadata_filenames, self.completed_pdf_parses_filenames

    def get_existance(self, verbose=False):
        # if no extention has been calculated
        self.get_extention(verbose)

        def filter_by_extention(files_list):
            return list(
                filter(
                    lambda file_name: file_name[-len(self.extention)
                                                     :] == self.extention,
                    files_list,
                )
            )

        meta_list_files = os.listdir(f"{self.path}/metadata")
        self.metadata_output = filter_by_extention(meta_list_files)
        if verbose:
            logging.info(f"metadata len: {len(self.metadata_output)}")

        pdfs_list_files = os.listdir(f"{self.path}/pdf_parses")
        self.pdf_parses_output = filter_by_extention(pdfs_list_files)
        if verbose:
            logging.info(f"pdf_parses len: {len(self.pdf_parses_output)}")

        return self.metadata_output, self.pdf_parses_output

    def get_toread_chunks(self, verbose=False):
        self.get_existance(verbose)
        # metadata_output, pdf_parses_output = self.get_existance(verbose)
        if verbose:
            logging.info(
                f"Meta_output: {self.metadata_output} and Pdfs_output: {self.pdf_parses_output}"
            )

        self.get_completed_filenames()
        # completed_metadata_filenames, completed_pdf_parses_filenames = self.get_completed_filenames()
        if verbose:
            logging.info(
                f"Meta_completed: {self.completed_metadata_filenames} and Pdfs_completed:{self.completed_pdf_parses_filenames}"
            )

        @_caching(
            metadata_output=self.metadata_output,
            pdf_parses_output=self.pdf_parses_output,
            completed_metadata_filenames=self.completed_metadata_filenames,
            completed_pdf_parses_filenames=self.completed_pdf_parses_filenames,
            function_name="get_toread_chunks",
        )
        def _get_toread_chunks(
            metadata_output,
            pdf_parses_output,
            completed_metadata_filenames,
            completed_pdf_parses_filenames,
        ):

            toread_meta_s2orc = sorted(
                list(set(completed_metadata_filenames) & set(metadata_output))
            )
            toread_pdfs_s2orc = sorted(
                list(set(completed_pdf_parses_filenames)
                     & set(pdf_parses_output))
            )

            return toread_meta_s2orc, toread_pdfs_s2orc

        self.toread_meta_s2orc, self.toread_pdfs_s2orc = _get_toread_chunks(
            self.metadata_output,
            self.pdf_parses_output,
            self.completed_metadata_filenames,
            self.completed_pdf_parses_filenames,
        )

        return self.toread_meta_s2orc, self.toread_pdfs_s2orc

    def memory_save_pipelines(self, verbose=False):

        self.get_toread_chunks(verbose)

        self.reset()

        return self.toread_meta_s2orc, self.toread_pdfs_s2orc

    def get_dictionary_input(self):
        return self.dictionary_input


class KeyPHConfig(CustomConfig):
    """S2orc Configuration class.

    Properties:
    - types (`unused`), `KEYPH_TYPES` list composed (eg. 'inspec', 'krapivin', 'nus', 'semeval', 'magkp', 'kp20k', 'duc', 'stackexchange')
    - keyph_type 'dataset_config_name' (`List[str]`) ( one of the `types` above)
    - path (str), actually, depending on the `s2orc_type` choosed, it is: "`dataset_path`/keyphrases/`type` for every s2orc_type (if multiple)"
    """

    types: List[str] = KEYPH_TYPES

    types_to_task: dict = dict(
        # filename, title, abstract, keywords
        inspec=['clas', 'regr', 'emb', 'summ', 'simp'],
        krapivin=['clas', 'regr', 'emb', 'summ', 'simp'],
        nus=['multitask-clas', 'clas',
             'multitask-regr', 'regr', 'emb', 'summ', 'simp'],
        semeval=['clas', 'regr', 'emb', 'summ', 'simp'],
        kp20k=['clas', 'regr', 'emb', 'summ', 'simp'],
        duc=['clas', 'regr', 'emb', 'summ', 'simp'],
        stackexchange=['clas', 'regr', 'emb', 'summ', 'simp'],
        magkp=['clas', 'regr', 'emb', 'summ', 'simp'],
    )

    splits_map: Dict = dict(
        inspec=["test", "valid"],
        krapivin=["test", "valid"],
        nus=["test"],
        semeval=["test", "valid"],
        kp20k=["test", "valid", "train"],
        duc=["test"],
        stackexchange=["test", "valid", "train"],
        magkp=["train"]
    )

    def __init__(self, *args, **kwargs):
        # inizialize the main dataset path
        super().__init__(*args, **kwargs)

        self.keyph_type: List[str] = self.str_to_list(
            kwargs["dataset_config_name"]
        )  # options are single choice 'inspec' or multiple 'krapivin,nus,semeval,kp20k'
        # path for the dataset
        # need to be appended {type} for every self.keyph_type
        self.path: str = f"{self.dataset_path}/keyphrase/json"

    def get_fingerprint(self) -> Dict:
        return dict({"keyph_type": self.keyph_type, "path": self.path})

    def get_splits_by_key(self, key: str):
        return self.splits_map[key]

    def get_tasks_by_key(self, key: str):
        return self.types_to_task[key]


class JurNLConfig(CustomConfig):
    """JurNLConfig Configuration class (For journal papers).

    Properties:
    - types (`unused`), `JURNL_TYPES` list composed (eg. 'icdar_19' 'icpr_20', 'ijdar_20')
    - jurnl_type 'dataset_config_name' (`List[str]`) ( one of the `types` above)
    - path (str), actually, depending on the `s2orc_type` choosed, it is: "`dataset_path`/keyphrases/`type` for every s2orc_type (if multiple)"
    """

    types: List[str] = JURNL_TYPES
    types_to_task: dict = dict(
        # filename, title, abstract, custom_class
        icdar_19=['clas', 'regr', 'emb', 'summ'],
        ijdar_20=['clas', 'regr', 'emb', 'summ'],
        # filename, title, abstract, keywords
        icpr_20=['multitask-clas', 'clas',
                 'multitask-regr', 'regr', 'emb', 'summ']
    )

    def __init__(self, *args, **kwargs):
        # inizialize the main dataset path
        super().__init__(*args, **kwargs)

        self.jurnl_type: List[str] = self.str_to_list(
            kwargs["dataset_config_name"]
        )  # options are single choice 'inspec' or multiple 'krapivin,nus,semeval,kp20k'
        # path for the dataset
        # need to be appended {type} for every self.jurnl_type
        self.path: str = f"{self.dataset_path}/s2orc-journal/"

    def get_fingerprint(self) -> Dict:
        return dict({"jurnl_type": self.jurnl_type, "path": self.path})

    def get_tasks_by_key(self, key: str):
        return self.types_to_task[key]
