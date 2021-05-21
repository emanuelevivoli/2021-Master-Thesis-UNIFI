from typing import List, Dict, Union


def fingerprints(*args) -> List[Dict]:
    dictionary: Dict = dict()
    for config in args:
        if type(config) is Config:
            for k, v in config.get_fingerprint().items():
                dictionary[k] = v
    return dictionary


class Config:
    """Base Config `interface`, it expose the method:
    - get_fingerprint `abstract method` that return a `Dict`
    """

    def str_to_list(self, string: str) -> List[Union[str, int]]:
        result = None
        if len(string) == 0:
            result = []
        elif "," in string:
            result = string.split(",")
        else:
            result = [string]
        return result

    def get_fingerprint(self) -> Dict:
        pass
