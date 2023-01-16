"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import numpy as np
import os
import configparser
from enum import Enum
from dataclasses import dataclass
from typing import Dict
from osgeo import gdal

GEOCOSICORR3D_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
GEOCOSICORR3D_SETUP_CFG = os.path.join(GEOCOSICORR3D_PACKAGE_DIR, 'setup.cfg')

config = configparser.ConfigParser()
config.read(GEOCOSICORR3D_SETUP_CFG)


@dataclass(frozen=True)
class SOFTWARE:
    AUTHOR = 'saif@caltech.edu||saifaati@gmail.com'
    SOFTWARE_NAME = config['metadata']['name']
    VERSION = config['metadata']['version']
    TILE_SIZE_MB = 128
    PARENT_FOLDER = GEOCOSICORR3D_PACKAGE_DIR
    WKDIR = os.path.join(os.path.dirname(GEOCOSICORR3D_PACKAGE_DIR), 'GEO_COSI_CORR_3D_WD/')
    geoCosiCorr3DOrientation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    CORR_CONFIG = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'geoCosiCorrBaseCfg/correlation.yaml')
    CORR_PARAMS_CONFIG = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                      'geoCosiCorrBaseCfg/corr_params.json')

    GEO_COSI_CORR3D_LIB = os.path.join(PARENT_FOLDER, "geoCosiCorr3D/lib/lfgeoCosiCorr3D.so")


@dataclass(frozen=True)
class SATELLITE_MODELS:
    RSM: str = 'RSM'
    RFM: str = 'RFM'


class ResamplingMethods(Enum):
    SINC = 'sinc'
    BILINEAR = 'bilinear'


@dataclass(frozen=True)
class Resampling_Methods:
    SINC = ResamplingMethods.SINC.value
    BILINEAR = ResamplingMethods.BILINEAR.value


GEOCOSICORR3D_SATELLITE_MODELS = [SATELLITE_MODELS.RFM, SATELLITE_MODELS.RSM]
GEOCOSICORR3D_RESAMLING_METHODS = [ResamplingMethods.SINC.value, ResamplingMethods.BILINEAR.value]


@dataclass(frozen=True)
class WRITERASTER:
    COMPRESS = "LZW"
    DRIVER = 'GTiff'


@dataclass(frozen=True)
class RASTER_TYPE():
    GDAL_UINT16 = gdal.GDT_UInt16
    GDAL_FLOAT32 = gdal.GDT_Float32
    GDAL_FLOAT64 = gdal.GDT_Float64


@dataclass(frozen=True)
class INTERPOLATION_TYPES():
    RectBivariateSpline = "RectBivariateSpline"
    CUBIC = 'cubic'
    LINEAR = 'linear'


@dataclass(frozen=True)
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
    DG = "DG"
    SENSOR_LIST = [SPOT1, SPOT2, SPOT3, SPOT4, SPOT5, SPOT1_5, SPOT6, SPOT7, SPOT67, WV1, WV2, WV3, WV4, GE, QB, DG]


GEOCOSICORR3D_SENSORS_LIST = SENSORS.SENSOR_LIST.value


@dataclass(frozen=True)
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
    FREQ_CORR_LIB = os.path.join(SOFTWARE.PARENT_FOLDER, "geoCosiCorr3D/lib/lgeoFreqCorr_v1.so")
    STAT_CORR_LIB = os.path.join(SOFTWARE.PARENT_FOLDER, "geoCosiCorr3D/lib/libgeoStatCorr.so.1")


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TP_DETECTION_METHODS:
    ASIFT = 'ASIFT'
    CVTP = 'cvTP'
    GEOSIFT = 'geoSIFT'


@dataclass(frozen=True)
class ASIFT_TP_PARAMS:
    CONV_PARAMS = gdal.TranslateOptions(
        gdal.ParseCommandLine(
            f"-ot UInt16 -of {WRITERASTER.DRIVER} -co BIGTIFF=YES -co COMPRESS={WRITERASTER.COMPRESS} -b 1 -co NBITS=16"))
    SCALE_FACTOR = 1 / 8
    MODE = 'All'
    IMG_SIZE = 1000
    MM_LIB = os.path.join(SOFTWARE.PARENT_FOLDER, "geoCosiCorr3D/lib/mmlibs/bin/mm3d")


class TEST_CONFIG:
    FREQ_CORR_PARAMS = {
        "window_size": [64, 64, 64, 64],
        "step": [8, 8],
        "grid": True,
        "mask_th": 0.95,
        "nb_iters": 4
    }
    SPA_CORR_PARAMS = {
        "window_size": 4 * [64],
        "step": [16, 16],
        "grid": True,
        "search_range": [10, 10]
    }
    FREQ_CORR_CONFIG: Dict = {"correlator_name": CORRELATION.FREQUENCY_CORRELATOR,
                              "correlator_params": FREQ_CORR_PARAMS}
    SPA_CORR_CONFIG = {"correlator_name": CORRELATION.SPATIAL_CORRELATOR,
                       "correlator_params": SPA_CORR_PARAMS
                       }
    GCP_OPT_CONFIG = {'nb_loops': 3, 'snr_th': 0.9, 'mean_error_th': 1 / 20,
                      'resampling_method': Resampling_Methods.SINC}
