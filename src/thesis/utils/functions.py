import logging
from datasets import concatenate_datasets
import yaml
import json


def range_from_N(s2orc_type, _n, _to, _into):
    if s2orc_type == "sample":

        if _n is not None:
            logging.warning(
                f"You set 'sample' but you also set `N` for full bucket range. \n The N selection will be discarded as only `sample` element will be used."
            )
            _n = 0
            list_range = [_n]

    elif s2orc_type == "full":

        if _n is None:
            logging.warning(
                f"You set 'full' but no bucket index was specified. \n We'll use the index 0, so the first bucket will be used."
            )
            _n = 0
            list_range = [_n]

        elif type(_n) is list:
            if _into:
                list_range = range(_n[0], _n[1])
                logging.warning(
                    f"The range is intended as [{_n[0]}, {_n[1]}] (start {_n[0]}, end {_n[1]})"
                )
            else:
                list_range = _n
                logging.warning(f"The element list is intended as: {_n}")

        elif type(_n) is int:
            if _to:
                list_range = range(0, _n)
                logging.warning(
                    f"The range is intended as [ 0, {_n}] (start 0, end {_n})"
                )
            else:
                list_range = [_n]
                logging.warning(f"The element list is intended as: [{_n}]")

    else:
        raise NameError(
            f"You must select an existed S2ORC dataset \n \
                    You selected {s2orc_type}, but options are ['sample' or 'full']"
        )

    return list_range


def fuse_datasets_splits(datasets):
    return concatenate_datasets([datasets[key] for key in datasets.keys()])


def get_dict_args(path_to_yaml):
    return yaml.load(open(path_to_yaml), Loader=yaml.FullLoader)


def save_dict_to_json(dict_, path_to_json):
    with open(path_to_json, 'w') as fp:
        json.dump(dict_, fp)
