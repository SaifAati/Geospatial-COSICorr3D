"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import json
import numpy as np
from decimal import *
from typing import Dict
import psutil
import logging


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


def log_available_memory(component_name: str):
    memory_stats = psutil.virtual_memory()
    total_memory_gb = memory_stats.total / (1024 ** 3)
    available_memory = memory_stats.available / (1024 ** 3)
    logging.info(f'{component_name}: _memory [Gb] : {available_memory:.2f} / {total_memory_gb:.3f}')
    return  available_memory
