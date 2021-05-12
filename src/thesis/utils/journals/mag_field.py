#
# The key null is actually null, not "null":str
#
#      real_mag_field_value = paper_metadata['mag_field_of_study']
#
# so we could return the id 3 if it not contained as key of dictionary
#
#      mag_field_dict.get(real_mag_field_value, 3)
#

from typing import Dict

ICDAR_field: Dict = {
    "binarization": 0,
    'comics': 1,
    'equations': 2,
    'handwritings': 3,
    'historical': 4,
    'interesting': 5,
    'invoices': 6,
    'other': 7
}

ICPR_field: Dict = {
}

IJDAR_field: Dict = {
    'ijdar1': 0,
    'ijdar2': 1,
    'ijdar3': 2,
    'ijdar4': 3,
    'ijdar5': 4,
    'ijdar6': 5,
    'ijdar7': 6,
    'ijdar8': 7,
    'ijdar9': 8,
    'ijdar10': 9,
    'ijdar11': 10,
    'ijdar12': 11,
    'ijdar13': 12,
    'ijdar14': 13,
    'ijdar15': 14,
    'ijdar16': 15,
    'ijdar17': 16,
    'ijdar18': 17,
    'ijdar19': 18,
    'ijdar20': 19,
}
