"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import warnings
import rasterio


def wrIgnoreNotGeoreferencedWarning():
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def wrSubsetOutsideBoundries(dataName, arg=''):
    msg = "Optimization subset is outside " + dataName + " boundaries" + arg
    warnings.warn(msg)


def wrNoDEM(h=0):
    msg = "No DEM is used, h=" + str(h)
    warnings.warn(msg)


def wrNoRSMCorrection(line=""):
    msg = "No RSM correction file, initial physical model will be used for the ortho-rectification process"
    if line:
         msg+=" line:"+ line

    warnings.warn(msg)


def wrInvaliDEM():
    msg = "Invalid DEM file for Direct Model pixel projection"
    warnings.warn(msg)
