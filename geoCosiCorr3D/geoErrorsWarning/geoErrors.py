"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os, sys


def erReadTpFile():
    msg = "Can't read the tie point file !!! "
    raise ValueError(msg)
def erReadCorrectionFile():
    msg = "Can't read correction File !!! "
    raise ValueError(msg)

def erLibNotFound(libPath):
    msg = "Unable to find the specified library:" + libPath
    raise ValueError(msg)


def erLibLoading(libPath):
    msg = "Unable to load the system C library" + libPath
    raise ValueError(msg)


def erRSMmodel():
    raise ValueError("Error! RSM information not found !!")


def erSensorNotSupported():
    raise ValueError("Sensor not supported !!")

def erDEM_vs_rOrtho_prjSystem():
    raise ValueError("DEM and rOrtho must have the same projection system !!")

def erNotImplemented(routineName):
    msg = routineName + " not implemented, on going work !!"
    raise ValueError(msg)

def erNotIdenticalDatum(msg=None):
    if msg!= None:
        raise ValueError(msg)
    else:
        raise ValueError("Different Datum and projection system")

