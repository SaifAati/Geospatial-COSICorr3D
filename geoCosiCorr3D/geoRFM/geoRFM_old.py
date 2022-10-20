"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import numpy as np


def DMS2DD(d, m, s):
    """
    Convert degree minute second to decimal degree
    Args:
        d:
        m:
        s:

    Returns:

    """
    dd = np.abs(d) + m / 60 + s / 3600
    if d < 0:
        return -1 * dd
    else:
        return dd


def DD2DMS(dd):
    """

    Args:
        dd:

    Returns:

    """
    mnt, sec = divmod(np.abs(dd) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    if dd < 0:
        return -1 * deg, mnt, sec
    else:
        return deg, mnt, sec

def ConvertRFM2MicMac(workPath, imgPath, RFMPath, outputDir, chsys="", degre=1):
    """

    Args:
        imgPath:
        RFMPath:
        outputDir:
        degre:

    Returns:

    """

    # subprocess.call("cd /home/cosi/5-Temp_SA/TestMicMAc")
    ##TODO: fix micmac path, get from cfg file
    micmacPath = "//home/cosicorr/Desktop/micmac/bin"
    mm3dPath = os.path.join(micmacPath, "mm3d")

    command = mm3dPath + " Convert2GenBundle " + imgPath + " " + RFMPath + " " + outputDir + " ChSys=WGS84toUTM.xml" + " Degre=" + str(
        degre)
    print(command)
    os.chdir(workPath)
    os.system(command)
    # subprocess.call([mm3dPath,"Convert2GenBundle",imgPath,RFMPath ,outputDir," Degre=1"])

    return





