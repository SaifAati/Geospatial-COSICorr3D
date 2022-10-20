"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from enum import Enum
from pathlib import Path
import numpy as np
import os
from geoCosiCorr3D.georoutines.file_cmd_routines import CreateDirectory
from dataclasses import dataclass


class SOFTWARE:
    AUTHOR = 'saif@caltech.edu||saifaati@gmail.com'
    SOFTWARE_NAME = "geoCosiCorr3D"
    VERSION = "2.1"
    TILE_SIZE_MB = 128
    PARENT_FOLDER = str(Path(os.getcwd()).parents[2])
    # PARENT_FOLDER = str(Path.home())

    WKDIR = CreateDirectory(PARENT_FOLDER, "geoCosiCorr3D_WKDIR", cal="n")
    LOGDIR = CreateDirectory(WKDIR, "geoCosiCorr3D_log", cal="n")
    geoCosiCorr3DOrientation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    CORR_CONFIG = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'geoCosiCorrBaseCfg/correlation.yaml')
    CORR_PARAMS_CONFIG = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                      'geoCosiCorrBaseCfg/corr_params.json')


@dataclass
class SATELLITE_MODELS:
    RSM: str = 'RSM'
    RFM: str = 'RFM'


class ResamplingMethods(Enum):
    SINC = 'sinc'
    BILINEAR = 'bilinear'


@dataclass
class Resampling_Methods:
    SINC = ResamplingMethods.SINC.value
    BILINEAR = ResamplingMethods.BILINEAR.value


GEOCOSICORR3D_SATELLITE_MODELS = [SATELLITE_MODELS.RFM, SATELLITE_MODELS.RSM]
GEOCOSICORR3D_RESAMLING_METHODS = [ResamplingMethods.SINC.value, ResamplingMethods.BILINEAR.value]


class WRITERASTER:
    COMPRESS = "LZW"
    DRIVER = 'GTiff'


class EARTH:
    SEMIMAJOR = 6378137.0
    SEMIMINOR = 6356752.3
    EARTH_MAX_RADIUS = 6384000
    EARTH_MEAN_RADIUS = 6371000
    SPEED_OF_LIGHT = 299792458
    EARTH_ROTATION_RATE = 72921151.467064 * 1e-12


class SENSORS(Enum):
    SPOT1 = "Spot1"
    SPOT2 = "Spot2"
    SPOT3 = "Spot3"
    SPOT4 = "Spot4"
    SPOT5 = "Spot5"
    SPOT1_5 = "Spot15"
    SPOT6 = "Spot6"
    SPOT7 = "Spot7"
    SPOT67 = 'Spot67'
    WV1 = "WV1"
    WV2 = "WV2"
    WV3 = "WV3"
    WV4 = "WV4"
    GE = "GE"
    QB = "QB"
    DG = "Dg"
    SENSOR_LIST = [SPOT1, SPOT2, SPOT3, SPOT4, SPOT5, SPOT1_5, SPOT6, SPOT7, SPOT67, WV1, WV2, WV3, WV4, GE, QB, DG]


GEOCOSICORR3D_SENSORS_LIST = SENSORS.SENSOR_LIST.value


class SENSOR:
    SPOT1 = SENSORS.SPOT1.value
    SPOT2 = SENSORS.SPOT2.value
    SPOT3 = SENSORS.SPOT3.value
    SPOT4 = SENSORS.SPOT4.value
    SPOT5 = SENSORS.SPOT5.value
    SPOT1_5 = SENSORS.SPOT1_5.value
    SPOT6 = SENSORS.SPOT6.value
    SPOT7 = SENSORS.SPOT7.value
    SPOT67 = SENSORS.SPOT67.value
    WV1 = SENSORS.WV1.value
    WV2 = SENSORS.WV2.value
    WV3 = SENSORS.WV3.value
    WV4 = SENSORS.WV4.value
    GE = SENSORS.GE.value
    QB = SENSORS.QB.value
    DG = SENSORS.DG.value
    SENSOR_LIST = SENSORS.SENSOR_LIST.value


GEOCOSICORR3D_SENSOR_SPOT_15 = [SENSOR.SPOT1, SENSOR.SPOT2, SENSOR.SPOT3, SENSOR.SPOT4, SENSOR.SPOT5, SENSOR.SPOT1_5]
GEOCOSICORR3D_SENSOR_SPOT_67 = [SENSOR.SPOT6, SENSOR.SPOT7, SENSOR.SPOT67]
GEOCOSICORR3D_SENSOR_DG = [SENSOR.WV1, SENSOR.WV2, SENSOR.WV3, SENSOR.WV4, SENSOR.GE, SENSOR.DG, SENSOR.QB]


class CORR_METHODS(Enum):
    FREQUENCY_CORR = "frequency"
    SPATIAL_CORR = "spatial"
    MM_CORR = "micmac"
    OPTICAL_FLOW = "optical_flow"
    NCC_CORR = "ncc"
    SGM_CORR = "sgm"
    CORR_METHODS_LIST = [FREQUENCY_CORR, SPATIAL_CORR, MM_CORR]


class CORR_LIBS(Enum):
    FREQ_CORR_LIB = os.path.join(os.path.dirname(__file__), "libs/lgeoFreqCorr_v1.so")
    STAT_CORR_LIB = os.path.join(os.path.dirname(__file__), "libs/libgeoStatCorr.so.1")


class CORRELATION:
    PIXEL_MEMORY_FOOTPRINT = 32
    TILE_SIZE_MB = 256
    FREQUENCY_CORRELATOR = CORR_METHODS.FREQUENCY_CORR.value
    SPATIAL_CORRELATOR = CORR_METHODS.SPATIAL_CORR.value
    CONFIG_FILE = SOFTWARE.CORR_CONFIG
    CORR_PARAMS_CONFIG = SOFTWARE.CORR_PARAMS_CONFIG
    FREQ_CORR_LIB = CORR_LIBS.FREQ_CORR_LIB.value
    STAT_CORR_LIB = CORR_LIBS.STAT_CORR_LIB.value


@dataclass
class RENDERING:
    GCP_COLOR = '#BDB76B'
    GCP_SZ = 25
    GCP_MARKER = '*'

@dataclass
class TP_DETECTION_METHODS:

    ASIFT = 'ASIFT'
    CVTP = 'cvTP'
    GEOSIFT = 'geoSIFT'

