"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
import numpy as np


def build_flatten_data_cube(data_cube: np.ndarray) -> np.ndarray:
    """
    Flattens a 3D data cube into a 2D array where each row corresponds to a flattened 2D slice of the cube.
    Args:
        data_cube: 3D array with shape (bands, height, width)

    Returns:
        2D array with shape (height * width, bands)

    """
    data_cube_shape = data_cube.shape
    return data_cube.reshape(data_cube_shape[0], -1).T
