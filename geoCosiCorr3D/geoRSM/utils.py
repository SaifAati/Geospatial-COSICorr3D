"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

from typing import List, Optional

import numpy as np
from scipy import interpolate
from scipy.interpolate import *

from geoCosiCorr3D.geoCore.constants import INTERPOLATION_TYPES
import geopandas
from scipy.spatial.transform import Rotation


def is_monotonic(A, tol=1e-9):
    """
    Determine if the array `A` is monotonically increasing, decreasing, or neither.

    Args:
        A (array-like): Input array to be checked.
        tol (float): Tolerance level for floating-point comparisons.

    Returns:
        str: "monotonically increasing" if A is increasing,
             "monotonically decreasing" if A is decreasing,
             "not monotonic" otherwise.
    """
    # Check for small arrays
    if len(A) < 2:
        raise ValueError("Input array must have at least 2 elements.")

    # Check for NaN or infinite values
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        raise ValueError("Input array must not contain NaN or infinite values.")

    # Check monotonicity
    diff = np.diff(A)
    if np.all(diff > -tol) and np.any(diff > tol):
        return "monotonically increasing"
    elif np.all(diff < tol) and np.any(diff < -tol):
        return "monotonically decreasing"
    else:
        return "not monotonic"


def locate_values(vector, values):
    """
    Finds the indices of `vector` that bracket `values`.

    Args:
        vector (array-like): A monotonic array.
        values (array-like): Array of values to locate within `vector`.

    Returns:
        np.array: Indices of `vector` that bracket `values`.

    Notes:
        The function finds the intervals within a given monotonic `vector`
        that bracket a set of `values`.
    """
    if is_monotonic(vector) not in ["monotonically increasing", "monotonically decreasing"]:
        raise ValueError("Input vector must be monotonic.")

    inds = np.digitize(values, vector)
    adjust = 1 if is_monotonic(vector) == "monotonically increasing" else -1

    for i, vec_ in enumerate(values):
        tmp = inds[i]
        if tmp == len(vector):
            tmp = len(vector) - 1
        if adjust * (vec_ - vector[tmp]) < 0:
            inds[i] = inds[i] - 1
        inds[i] = max(0, min(inds[i], len(vector) - 2))

    return inds


def normalize_array(input_array: np.ndarray, order: Optional[int] = 2) -> np.ndarray:
    """
    Normalize the rows of a 2D array.

    Args:
        input_array (np.ndarray): 2D array to be normalized by row.
        order (int, optional): Order of the norm. Default is 2 (Euclidean norm).

    Returns:
        np.ndarray: Normalized array.

    Raises:
        ValueError: If input_array is not 2D.
        ValueError: If any norm result is zero to prevent division by zero.
    """
    # Validate input
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    # Compute norms
    norms = np.linalg.norm(input_array, axis=1, ord=order).reshape(-1, 1)

    # Check for zero norms to avoid division by zero
    if np.any(norms == 0):
        raise ValueError("Zero norm encountered. Cannot normalize.")

    # Normalize and return
    return input_array / norms


class Convert:
    @staticmethod
    def quat_to_rotation(quat_xyzs: List[float]) -> np.ndarray:
        quat_sxyz = np.array([quat_xyzs[3], quat_xyzs[0], quat_xyzs[1], quat_xyzs[2]])
        rot = Rotation.from_quat(quat_sxyz)
        return rot.as_matrix()

    @staticmethod
    def custom_quat_to_rotation(quat_xyzs) -> np.ndarray:
        """
        Convert a quaternion to a rotation matrix.

        Args:
            quat_xyzs: A 4-element array [x, y, z, w] representing a quaternion,
                  where [x, y, z] is the vector part and w is the scalar part.

        Returns:
            A 3x3 rotation matrix as a NumPy ndarray.
        """
        # Ensure the quaternion is normalized
        quat = quat_xyzs / np.linalg.norm(quat_xyzs)

        x, y, z, w = quat
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        xw, yw, zw = x * w, y * w, z * w

        return np.array([[1 - 2 * (yy + zz), 2 * (xy + zw), 2 * (xz - yw)],
                         [2 * (xy - zw), 1 - 2 * (xx + zz), 2 * (yz + xw)],
                         [2 * (xz + yw), 2 * (yz - xw), 1 - 2 * (xx + yy)]])

    @staticmethod
    def ecef_to_orbital(eph: np.ndarray):
        """
        Convert satellite pos from an Earth-Centered Earth-Fixed (ECEF) coordinate system to an
        orbital frame (often referred to as the Local Vertical Local Horizontal (LVLH)
        or Radial-Intrack-Crosstrack (RIC) frame).
        """
        sat_pos_ecef = eph[:, 0:3]
        sat_vel_ecef = eph[:, 3:6]
        sat_z_pos_orbital = normalize_array(sat_pos_ecef)
        sat_x_pos_orbital = normalize_array(np.cross(sat_vel_ecef, sat_z_pos_orbital))
        sat_y_pos_orbital = np.cross(sat_z_pos_orbital, sat_x_pos_orbital)

        return np.hstack((sat_x_pos_orbital, sat_y_pos_orbital, sat_z_pos_orbital))


class Interpolate:
    @staticmethod
    def linear(array: np.ndarray, location):
        """

        Args:
            array:
            location:

        Returns:

        """

        loc = int(location)
        tmpArr = array[loc]
        if loc + 1 < len(array):
            res = (array[loc + 1] - tmpArr) * (location - loc) + tmpArr
        else:
            res = (array[-1] - tmpArr) * (location - loc) + tmpArr
        return res

    @staticmethod
    def custom_linear_interpolation(VV, XX, x_out=None):
        """
        Linear interpolation of values.

        Args:
            VV (array-like): Y-coordinates of data points.
            XX (array-like): X-coordinates of data points.
            xOut (list, optional): X-coordinates at which to evaluate the interpolated function.

        Returns:
            array: Interpolated values at `xOut`.
        """
        if x_out is None:
            x_out = []

        v = np.copy(VV)
        x = np.copy(XX)
        m = np.size(v)

        if np.size(x) != m:
            raise ValueError("VV and XX must have the same number of elements.")

        if np.size(x_out) > 0:
            s = locate_values(vector=x, values=x_out)
            diff = v[s + 1] - v[s]
            p = (x_out - x[s]) * diff / (x[s + 1] - x[s]) + v[s]
            return p
        else:
            raise NotImplementedError("Interpolation on a regular grid is not implemented.")


def calculate_haversine_distance(pt1_lat, pt1_lon, pt2_lat, pt2_lon):
    """
    Calculate approximate distance between two lat/long points and return
    distance in meters.

    :param pt1_lat: The latitude of point one.
    :param pt1_lon: The longitude of point one.
    :param pt2_lat: The latitude of point two.
    :param pt2_lon: The longitude of point two.

    :returns: The approximate distance between two lat/long points in meters.
    """
    # from geopy.distance import distance
    #
    # point1 = (lat1, lon1)
    # point2 = (lat2, lon2)
    #
    # dist = distance(point1, point2).km
    EARTH_MEAN_RADIUS = 6371000
    pt1_lat = np.array(pt1_lat)
    pt1_lon = np.array(pt1_lon)
    pt2_lat = np.array(pt2_lat)
    pt2_lon = np.array(pt2_lon)

    p1_lat_radians = np.radians(pt1_lat)
    p2_lat_radians = np.radians(pt2_lat)
    diff_lat = np.radians(pt2_lat - pt1_lat)
    diff_lon = np.radians(pt2_lon - pt1_lon)

    term1 = np.sin(diff_lat / 2.) ** 2
    term2 = np.cos(p1_lat_radians) * \
            np.cos(p2_lat_radians) * np.sin(diff_lon / 2.) ** 2

    a = term1 + term2
    c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1. - a))
    dist = EARTH_MEAN_RADIUS * c

    return dist


def mean_distance_between_fps(gpd_fp1: geopandas.GeoDataFrame, gpd_fp2: geopandas.GeoDataFrame):
    top_coord1 = max(gpd_fp1.loc[0, 'geometry'].exterior.coords, key=lambda coord: coord[1])
    top_coord2 = max(gpd_fp2.loc[0, 'geometry'].exterior.coords, key=lambda coord: coord[1])
    d = calculate_haversine_distance(pt1_lat=top_coord1[1], pt1_lon=top_coord1[0],
                                     pt2_lat=top_coord2[1], pt2_lon=top_coord2[0])
    return d
