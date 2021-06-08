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

ICDAR_custom_field: Dict = {
    'binarization': 0,
    'comics': 1,
    'equations': 2,
    'handwritings': 3,
    'historical': 4,
    'interesting': 5,
    'invoices': 6,
    'other': 7
}

ICDAR_field: Dict = {
    "Handwritten Text Recognition": 0,
    "Document Image Processing": 1,
    "Document Understanding": 2,
    "Table Analysis": 3,
    "Text Detection and Recognition": 4,
    "Mathematical Expression and Text Recognition": 5,
    "Layout Analysis": 6,
    "Applications of Document Analysis": 7,
    "Script Identification and Authentication": 8,
    "Script Identification and Authentication": 9,
}

ICPR_field: Dict = {
    "Designing Machine Learning": 0,
    "Machine Learning": 1,
    "Human Behaviour Understanding": 2,
    "Computer Vision": 3,
    "Document Analysis": 4,
    "Image processing": 5,
    "Clustering for Pattern Analysis": 6,
    "Image Processing and Segmentation": 7,
    "Machine Learning and Features for Pattern Analysis": 8,
    "Computer Vision, Robotics and Tracking": 9,
    "Image Processing and Denoising": 10,
    "Neural Networks and Self-Attention": 11,
    "Face Analysis": 12,
    "Stereo and 3D Vision": 13,
    "Signal processing and Compression": 14,
    "Pattern Analysis and Applications": 15,
    "Action and Activity Recognition Oral": 16,
    "Understanding Deep Learning": 17,
    "Human Poses, Faces and Fingerprints": 18,
    "Object Detection, Localization, and Classification": 19,
    "Computer Vision and Neural Networks Applications": 20,
    "Scene Text Detection and Recognition": 21,
    "Medical Imaging": 22,
    "Scene Analysis, Learning, and Datasets": 23,
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
