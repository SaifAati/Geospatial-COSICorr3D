"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import json
import numpy as np
from decimal import *
from typing import Dict


class CosiCorr3DEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        
        return json.JSONEncoder.default(self, obj)


def read_json_as_dict(json_file: str) -> Dict:
    f = open(json_file)
    return json.loads(f.read())


class Payload(object):
    def __init__(self, json_file):
        f = open(json_file)
        self.__dict__ = json.loads(f.read())
