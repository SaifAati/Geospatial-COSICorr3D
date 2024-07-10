"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
import json
import os
from decimal import *
import psutil
import logging
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import geoCosiCorr3D.georoutines.geo_utils as geoRT
import geoCosiCorr3D.geoCore.constants as C


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


# def write_patches(ref_patches: List[C.Patch], target_patches: [C.Patch], o_folder: str, prefix=None, png=False):
#     # TODO add the geo-referencing for each patch, which should be the same (patch Map Grid).
#     for ref_patch in ref_patches:
#         patch_id = ref_patch.id
#         ref_ortho_patch = ref_patch.array
#         raw_ortho_patch = ref_ortho_patch * 0
#         for target_patch in target_patches:
#             if target_patch.id == patch_id:
#                 raw_ortho_patch = target_patch.array
#                 break
#
#         if prefix is not None:
#             patches_raster_path = os.path.join(o_folder, f'{prefix}_{patch_id}_patches.tif')
#         else:
#             patches_raster_path = os.path.join(o_folder, f'{patch_id}_patches.tif')
#         geoRT.cRasterInfo.write_raster(output_raster_path=patches_raster_path,
#                                        array_list=[ref_ortho_patch, raw_ortho_patch],
#                                        descriptions=['ref_patch', 'raw_ortho_patch'])
#         if png:
#             fig, (ax1, ax2) = plt.subplots(1, 2)
#             ax1.imshow(ref_ortho_patch, cmap="gray")
#             ax1.set_title("ref_ortho_patch")
#
#             ax2.imshow(raw_ortho_patch, cmap="gray")
#             ax2.set_title("raw_ortho_patch")
#             plt.savefig(f'{patches_raster_path}.png')
#
#             plt.clf()
#             plt.close(fig)
#
#     return
def write_patches(correspondences: List[Tuple[C.Patch, C.Patch]], o_folder: str, prefix=None, png=False):
    # TODO add the geo-referencing for each patch, which should be the same (patch Map Grid).
    for (ref_patch, target_patch) in correspondences:
        patch_id = ref_patch.id
        # ref_ortho_patch = ref_patch.array
        # raw_ortho_patch = ref_ortho_patch * 0
        # for target_patch in target_patches:
        #     if target_patch.id == patch_id:
        #         raw_ortho_patch = target_patch.array
        #         break

        if prefix is not None:
            patches_raster_path = os.path.join(o_folder, f'{prefix}_{patch_id}_patches.tif')
        else:
            patches_raster_path = os.path.join(o_folder, f'{patch_id}_patches.tif')

        geoRT.cRasterInfo.write_raster(output_raster_path=patches_raster_path,
                                       array_list=[ref_patch.data, target_patch.data],
                                       descriptions=['ref_patch', 'raw_ortho_patch'])
        if png:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(ref_patch.data, cmap="gray")
            ax1.set_title("ref_ortho_patch")

            ax2.imshow(target_patch.data, cmap="gray")
            ax2.set_title("raw_ortho_patch")
            plt.savefig(f'{patches_raster_path}.png')

            plt.clf()
            plt.close(fig)

    return
