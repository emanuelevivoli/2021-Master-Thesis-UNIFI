from typing import List, Dict


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

    def get_fingerprint(self) -> Dict:
        pass


# SingleChunk = {
#     # {'metadata': [], 'pdf_parses': [], 'meta_key_idx': {}, 'pdf_key_idx': {}}
#     'metadata':  [],
#     'pdf_parses': [],
#     'meta_key_idx': {},
#     'pdf_key_idx': {}
# }
