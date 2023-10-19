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
#######################################################
def isMonotonic(A):
    """

    Args:
        A:

    Returns:

    """
    # if all(A[i] <= A[i + 1] for i in range(len(A) - 1)):
    if np.all(np.diff(A) > 0):
        return "monotonically increasing"
    # elif (A[i] >= A[i + 1] for i in range(len(A) - 1)):
    elif np.all(np.diff(A) < 0):
        return "monotonically decreasing"
    else:
        return 0


def Value_Locate(vector, values):
    """

    Args:
        vector:
        values:

    Returns:
    Notes:
        https://www.harrisgeospatial.com/docs/VALUE_LOCATE.html
    The VALUE_LOCATE function finds the intervals within a given monotonic vector that brackets a given set of one
    or more search values. This function is useful for interpolation and table-lookup,
    and is an adaptation of the locate() routine in Numerical Recipes. V
    ALUE_LOCATE uses the bisection method to locate the interval.
    """

    # print(isMonotonic(vector))
    inds = np.digitize(values, vector)
    if isMonotonic(vector) == "monotonically increasing":
        for i, vec_ in enumerate(values):
            tmp = inds[i]
            if tmp == len(vector):
                tmp = len(vector) - 1
            if vec_ < vector[tmp]:
                inds[i] = inds[i] - 1
            if inds[i] < 0:
                inds[i] = 0
            if inds[i] >= len(vector) - 1:
                inds[i] = len(vector) - 2
    elif isMonotonic(vector) == "monotonically decreasing":
        for i, vec_ in enumerate(values):
            tmp = inds[i]
            if tmp == len(vector):
                tmp = len(vector) - 1
            if vec_ > vector[tmp]:
                inds[i] = inds[i] - 1
            if inds[i] < 0:
                inds[i] = 0
            if inds[i] >= len(vector) - 1:
                inds[i] = len(vector) - 2
    else:
        print("Waring Vector is not monotonic !!! ")
    return inds


def Interpol(VV, XX, xOut: Optional[List] = None):

    if xOut is None:
        xOut = []
    # print(np.size(xOut))
    if np.size(xOut) > 0:
        regular = 0
        # print("Not regular grid !")
    else:
        regular = 1
        # print("Regular grid !")
    ## Make a copy so we dont overwrite the outputs
    v = np.copy(VV)
    x = np.copy(XX)
    m = np.size(v)  ## Nbr of inputs points


    if regular == 1:
        raise NotImplementedError

    if np.size(x) != m:
        raise EOFError("Error! v and X array must have the same nb of elts")
    else:
        s = Value_Locate(vector=x, values=xOut)

        diff = v[s + 1] - v[s]
        p = (xOut - x[s]) * diff / (x[s + 1] - x[s]) + v[s]
        return p


def LinearIterpolation(array, location):
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


def Interpolate2D(inArray, x, y, kind=INTERPOLATION_TYPES.CUBIC):
    """
    Procedure for 2D interpolation
    Args:
        inArray:
        x: list
        y: list
        kind: RectBivariateSpline,quintic

    Returns:
    Notes:
        1- By default, RectBivariateSpline uses a degree 3 spline. By providing only 3 points along the y-axis it
        cannot do that. Adding ky=2 to the argument list fixes the problem, as does having more data.
        2- Add the possibility to perform sinc interpolation :
                Hints:  https://stackoverflow.com/questions/1851384/resampling-interpolating-matrix
                        https://www.programcreek.com/python/example/58004/numpy.sinc
                        https://gist.github.com/gauteh/8dea955ddb1ed009b48e
                        https://gist.github.com/endolith/1297227
                        https://www.harrisgeospatial.com/docs/INTERPOLATE.html
    """

    shape = np.shape(inArray)
    lin = shape[0]
    col = shape[1]
    # print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    # print(inArray.shape,x,y)

    if kind == "RectBivariateSpline":
        KX = 3
        KY = 3

        if lin <= KX:
            KX = lin - 1
        if col <= KY:
            KY = col - 1
        f = RectBivariateSpline(np.arange(0, lin, 1), np.arange(0, col, 1), inArray, kx=KX, ky=KY, s=0)

        if len(x) > 1 and len(y) > 1:
            res = []
            for x_, y_ in zip(x, y):
                res.append(f(x_, y_).item())
            return res
        else:
            return [f(x, y).item()]
    if kind == "cubic":
        f = interp2d(np.arange(0, lin, 1), np.arange(0, col, 1), np.ndarray.flatten(inArray), kind=kind)
        if len(x) > 1 and len(y) > 1:
            res = []
            for x_, y_ in zip(x, y):
                res.append(f(x_, y_).item())

            return res
        else:
            # print(f(x,y))
            return [f(x, y).item()]
    if kind == "linear" or kind == "nearest":
        # 'linear in 1d = binlinear in 2D'
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html

        f = interpolate.RegularGridInterpolator(points=(np.arange(0, lin, 1), np.arange(0, col, 1)),
                                                values=inArray,
                                                method=kind,
                                                bounds_error=False,
                                                fill_value=np.nan)

        return f(np.array([x, y]).T)


def Interpoate1D(X, Y, xCord, kind=INTERPOLATION_TYPES.LINEAR):
    # print(X,Y)
    f = interp1d(X, Y, kind=kind, fill_value="extrapolate")
    ynew = f(xCord)  # use interpolation function returned by `inter

    if len(ynew) > 1:
        return ynew
    else:
        return [ynew.item()]


if __name__ == '__main__':
    # print(np.searchsorted([1, 2, 3, 4, 5], 3.1,side="left"))

    vector = [-3, -5, -8, -9.2, -9.5, -10]
    values = [-10, -5.1, 9.3, -10]
    vector = np.asarray(vector)
    values = np.asarray(values)
    print(Value_Locate(vector=vector, values=values))
