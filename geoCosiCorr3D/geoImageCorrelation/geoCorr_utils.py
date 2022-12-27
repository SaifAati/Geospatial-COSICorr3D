"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import sys
from osgeo import gdal

gdal.UseExceptions()

def trycatch(f, x, exception):
    """Inline trycatch on call expression."""
    try:
        return f(x)
    except exception:
        return None

#make a function callable twice before it runs
splitcall = lambda f: lambda *args, **kargs: lambda *args2, **kargs2: f(*args, *args2, **kargs, **kargs2)
#make a function callable thrice before it runs
splitcall2 = lambda f: splitcall(splitcall(f))

#check if a value is a power of 2
pow2 = lambda val: (val & (val-1) == 0)

#helper function to mark progress (out of 100) and communicate between layers
__mark_progress__ = lambda x: None

def clamp(num, min_value, max_value):
    """Clamp a value between a min and a max."""
    return max(min(num, max_value), min_value)

def const(val):
    """Returns a function which ignores its parameter and returns val."""
    return lambda _: val

def project_path(path):
    """Gets a path relative to the location of the running script."""
    return os.path.join(sys.path[0], path)

def setdefaultattr(obj, name, value):
    """Sets an attribute on obj named name to value if it doesn't already exist."""
    if not hasattr(obj, name):
        setattr(obj, name, value)
    return getattr(obj, name)

def get_bands(path):
    """Checks if the specified path is a raaster image and returns a dictionary of bands"""
    #TODO use georoutines instead
    try:
        raster = gdal.Open(path, gdal.GA_ReadOnly)
    except RuntimeError:
        return None
    bands = {raster.GetRasterBand(i).GetDescription(): i for i in range(1, raster.RasterCount + 1)}
    bands = {name if name else str(i): i for name, i in bands.items()}
    del raster
    return bands