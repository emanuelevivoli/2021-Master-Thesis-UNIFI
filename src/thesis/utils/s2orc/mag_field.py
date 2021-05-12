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

mag_field_dict: Dict = {
    "Medicine": 0,
    "Biology": 1,
    "Chemistry": 2,
    "Engineering": 4,
    "Computer Science": 5,
    "Physics": 6,
    "Materials Science": 7,
    "Mathematics": 8,
    "Psychology": 9,
    "Economics": 10,
    "Political Science": 11,
    "Business": 12,
    "Geology": 13,
    "Sociology": 14,
    "Geography": 15,
    "Environmental Science": 16,
    "Art": 17,
    "History": 18,
    "Philosophy": 19
    # "null":         3,
}
