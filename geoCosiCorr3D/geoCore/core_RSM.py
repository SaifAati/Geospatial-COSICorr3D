"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import logging
import sys
from pathlib import Path
import pickle
import numpy as np
from geoCosiCorr3D.geoCore.base.base_RSM import BaseRSM

from geoCosiCorr3D.geoCore.constants import GEOCOSICORR3D_SENSOR_DG, GEOCOSICORR3D_SENSOR_SPOT_15, \
    GEOCOSICORR3D_SENSOR_SPOT_67, SOFTWARE
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors


class RSM(BaseRSM):

    def ComputeAttitude(self):
        pass

    def Interpolate_position_velocity_attitude(self):
        pass

    @staticmethod
    def build_RSM(metadata_file, sensor_name, debug=False):

        ## Check if the folder containing the necessary information to build the RSM file
        ## and build the RSM file. For the moment we support the following platforms: Spot6+, WV1+, GE, QuickBird
        if metadata_file is None:
            geoErrors.erRSMmodel()

        if sensor_name in GEOCOSICORR3D_SENSOR_DG:
            from geoCosiCorr3D.geoRSM.DigitalGlobe_RSM import cDigitalGlobe
            rsm_sensor_model = cDigitalGlobe(dgFile=metadata_file, debug=True)
            rsm_file = os.path.join(os.path.dirname(metadata_file), Path(metadata_file).stem + ".pkl")
            RSM.write_rsm(rsm_file, rsm_sensor_model)
        elif sensor_name in GEOCOSICORR3D_SENSOR_SPOT_67:
            # "model Path should be the XML file "
            from geoCosiCorr3D.geoRSM.Spot_RSM import cSpot67
            rsm_sensor_model = cSpot67(dmpXml=metadata_file, debug=True)

            img_path = os.path.join(os.path.dirname(metadata_file),
                                    "IMG_" + Path(metadata_file).stem.split("DIM_")[1] + ".tif")
            rsm_file = os.path.join(os.path.dirname(metadata_file), Path(img_path).stem + ".pkl")
            RSM.write_rsm(rsm_file, rsm_sensor_model)
        elif sensor_name in GEOCOSICORR3D_SENSOR_SPOT_15:
            # "model Path should be the XML file "
            from geoCosiCorr3D.geoRSM.Spot_RSM import cSpot15
            rsm_sensor_model = cSpot15(dmpFile=metadata_file, debug=debug)
            rsm_file = os.path.join(os.path.dirname(metadata_file), Path(metadata_file).stem + ".pkl")
            RSM.write_rsm(rsm_file, rsm_sensor_model)
        else:
            raise sys.exit(f'Sensor {sensor_name} not supported by {SOFTWARE.SOFTWARE_NAME} v{SOFTWARE.VERSION}')

        with open(rsm_file, 'rb') as f:
            rsm_model = pickle.load(f)
        logging.info(f'RSM file: {rsm_file}')
        return rsm_model

    @staticmethod
    def write_rsm(output_rsm_file, rsm_model):
        with open(output_rsm_file, "wb") as output:
            pickle.dump(rsm_model, output, pickle.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def rsm_footprint(rsmModel, demFile=None, hMean=None, pointingCorrection=np.zeros((3, 3))):
        """
        Computing the footprint of raster using the RSM.
        Args:
            rsmModel:
            demFile:
            hMean:
            pointingCorrection:

        Returns:  array(5,3)
            rows : ul,ur,lr,lf,ul
            cols: [lon ,lat, alt]
        """
        from geoCosiCorr3D.geoRSM.Pixel2GroundDirectModel import cPix2GroundDirectModel
        #TODO this function is not tested yet.
        rasterWidth = rsmModel.nbCols
        rasterHeight = rsmModel.nbRows
        xPixList = [0, rasterWidth, rasterWidth, 0, 0]
        yPixList = [0, 0, rasterHeight, rasterHeight, 0]
        resTemp_ = []
        for i in range(len(xPixList)):
            pix2GroundObj = cPix2GroundDirectModel(rsmModel=rsmModel,
                                                   xPix=xPixList[i],
                                                   yPix=yPixList[i],
                                                   demFile=demFile,
                                                   hMean=hMean,
                                                   rsmCorrectionArray=pointingCorrection
                                                   )
            resTemp_.append(pix2GroundObj)

        resTemp = [item.geoCoords for item in resTemp_]
        return np.asarray(resTemp)
