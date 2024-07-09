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
    return available_memory


def compute_tile_fp(east_arr, north_arr, grid_epsg, tile_num):
    import json
    min_east = np.min(east_arr)
    max_east = np.max(east_arr)
    min_north = np.min(north_arr)
    max_north = np.max(north_arr)

    footprint_corners = [
        [min_east, max_north],  # Top-left
        [max_east, max_north],  # Top-right
        [max_east, min_north],  # Bottom-right
        [min_east, min_north],  # Bottom-left
        [min_east, max_north]
    ]
    geojson_object = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [footprint_corners]
                },
                "properties": {}
            }
        ],
        "crs": {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{grid_epsg}"
            }
        }
    }
    output_file_path = f'footprint_{tile_num}.geojson'
    with open(output_file_path, 'w') as f:
        json.dump(geojson_object, f)
