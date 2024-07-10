"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
# This is a configuration file, containing all the parameters of the geoCosiCorr3D workflow.

import sys
from geoCosiCorr3D.georoutines.file_cmd_routines import CreateDirectory
from geoCosiCorr3D.geoCore.constants import *

geoCfg = {}


def Set_geoStatCorrLib():
    if "win" in sys.platform:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib\libgeoStatCorr.dll")
    if 'linux' in sys.platform:
        return CORRELATION.STAT_CORR_LIB


geoCfg['geoStatCorrLib'] = Set_geoStatCorrLib()

geoCfg['geoFreqCorrLib'] = CORRELATION.FREQ_CORR_LIB

geoCfg["geoCosiCorr3DLib"] = SOFTWARE.GEO_COSI_CORR3D_LIB

# path to MicMac
geoCfg["MicMacLib"] = os.path.join(SOFTWARE.PARENT_FOLDER, "lib/mmlibs/bin/mm3d")
# path to ASP
# geoCfg["ASPLib"] = "/home/cosicorr/anaconda3/envs/asp/bin"
# define tile size
geoCfg['tileSize'] = SOFTWARE.TILE_SIZE_MB  # Mb

# define
geoCfg["semiMajor"] = EARTH.SEMIMAJOR
geoCfg["semiMinor"] = EARTH.SEMIMINOR

geoCfg["geoSIFTLib"] = os.path.join(SOFTWARE.PARENT_FOLDER, 'lib/libSIFT_v0.1.1.so')
geoCfg["geoRANSACLib"] = os.path.join(SOFTWARE.PARENT_FOLDER, 'lib/libransac.so')


class cgeoCfg:
    def __init__(self):
        self.tileSize = geoCfg['tileSize']
        self.geoFreqCorrLib = geoCfg['geoFreqCorrLib']
        self.geoStatCorrLib = geoCfg['geoStatCorrLib']
        self.mmLib = geoCfg["MicMacLib"]
        # self.aspLib = geoCfg["ASPLib"]

        self.geoCosiCorr3DLib = geoCfg["geoCosiCorr3DLib"]
        self.semiMajor = geoCfg["semiMajor"]
        self.semiMinor = geoCfg["semiMinor"]

        self.geoRansacLib = geoCfg["geoRANSACLib"]
        self.geoSIFTLib = geoCfg["geoSIFTLib"]

    def CreateTempFolder(self):
        geoCfg["mmTempFolder"] = CreateDirectory(SOFTWARE.WKDIR, "TMP_mmTp", cal="y")
        self.mmTempFolder = geoCfg["mmTempFolder"]
        return
