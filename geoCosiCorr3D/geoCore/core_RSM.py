"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import geopandas
import numpy as np
from shapely import Polygon

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
from geoCosiCorr3D.geoCore.base.base_RSM import BaseRSM
from geoCosiCorr3D.geoRSM.Pixel2GroundDirectModel import cPix2GroundDirectModel


class RSM(BaseRSM):

    def __init__(self):
        self.time = None
        self.date = None
        self.gsd = None
        self.platform = None
        self.date_time_obj = None

    def interp_eph(self):
        pass

    def interp_attitude(self):
        pass

    @staticmethod
    def build_RSM(metadata_file, sensor_name, debug=False):

        ## Check if the folder containing the necessary information to build the RSM file
        ## and build the RSM file. For the moment we support the following platforms: Spot6+, WV1+, GE, QuickBird
        if metadata_file is None:
            geoErrors.erRSMmodel()

        if sensor_name in C.GEOCOSICORR3D_SENSOR_DG:
            from geoCosiCorr3D.geoRSM.DigitalGlobe_RSM import cDigitalGlobe
            rsm_sensor_model = cDigitalGlobe(dgFile=metadata_file, debug=True)
            rsm_file = os.path.join(os.path.dirname(metadata_file), Path(metadata_file).stem + ".pkl")
            RSM.write_rsm(rsm_file, rsm_sensor_model)
        elif sensor_name in C.GEOCOSICORR3D_SENSOR_SPOT_67:
            # "model Path should be the XML file "
            from geoCosiCorr3D.geoRSM.Spot_RSM import cSpot67
            logging.info(f'{metadata_file}')
            rsm_sensor_model = cSpot67(dmpXml=metadata_file, debug=True)

            img_path = os.path.join(os.path.dirname(metadata_file),
                                    "IMG_" + Path(metadata_file).stem.split("DIM_")[1] + ".tif")
            rsm_file = os.path.join(os.path.dirname(metadata_file), Path(img_path).stem + ".pkl")
            RSM.write_rsm(rsm_file, rsm_sensor_model)
        elif sensor_name in C.GEOCOSICORR3D_SENSOR_SPOT_15:
            # "model Path should be the XML file "
            from geoCosiCorr3D.geoRSM.Spot_RSM import cSpot15
            rsm_sensor_model = cSpot15(dmpFile=metadata_file, debug=debug)
            rsm_file = os.path.join(os.path.dirname(metadata_file), Path(metadata_file).stem + ".pkl")
            RSM.write_rsm(rsm_file, rsm_sensor_model)
        else:
            raise sys.exit(f'Sensor {sensor_name} not supported by {C.SOFTWARE.SOFTWARE_NAME} v{C.SOFTWARE.VERSION}')

        with open(rsm_file, 'rb') as f:
            rsm_model = pickle.load(f)
        logging.info(f'RSM file: {rsm_file}')
        return rsm_model

    @staticmethod
    def write_rsm(output_rsm_file, rsm_model):
        with open(output_rsm_file, "wb") as output:
            pickle.dump(rsm_model, output, pickle.HIGHEST_PROTOCOL)
        return output_rsm_file

    @staticmethod
    def compute_rsm_footprint(rsm_model, dem_file: Optional[str] = None,
                              rsm_corr_model: Optional[np.ndarray] = None, hMean: Optional[float] = None):

        if rsm_corr_model is None:
            rsm_corr_model = np.zeros((3, 3))

        xBBox = [0, rsm_model.nbCols - 1, 0, rsm_model.nbCols - 1]
        yBBox = [0, 0, rsm_model.nbRows - 1, rsm_model.nbRows - 1]

        extent_coords = []
        for xVal, yVal in zip(xBBox, yBBox):
            # print("\n----- xVal:{},yVal:{}".format(xVal, yVal))
            arg = (rsm_model, xVal, yVal, rsm_corr_model, dem_file, hMean)
            pix2Ground_obj = cPix2GroundDirectModel(*arg)

            extent_coords.append(pix2Ground_obj.geoCoords)  # lon , Lat, ALT
        ground_extent = {'ul': extent_coords[0], 'ur': extent_coords[1], 'lf': extent_coords[2], 'lr': extent_coords[3]}
        fp = {'type': 'Polygon',
              'coordinates': [[ground_extent['ul'][0], ground_extent['ul'][1], ground_extent['ul'][2]],
                              [ground_extent['ur'][0], ground_extent['ur'][1], ground_extent['ur'][2]],
                              [ground_extent['lr'][0], ground_extent['lr'][1], ground_extent['lr'][2]],
                              [ground_extent['lf'][0], ground_extent['lf'][1], ground_extent['lf'][2]]],
              }

        fp_poly = Polygon(fp['coordinates'])
        fp_gdf = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[fp_poly])
        return fp, xBBox, yBBox, fp_gdf

    @staticmethod
    def read_geocosicorr3d_rsm_file(pkl_file):
        with open(pkl_file, "rb") as output:
            geocosicorr3d_rsm_model = pickle.load(output)
        return geocosicorr3d_rsm_model
