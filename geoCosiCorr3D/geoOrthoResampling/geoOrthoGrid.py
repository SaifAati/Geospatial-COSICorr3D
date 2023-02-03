"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import numpy as np
from typing import Optional

import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoCore.constants import SATELLITE_MODELS


class cGetSatMapGrid:
    def __init__(self,
                 rasterInfo: geoRT.cRasterInfo,
                 modelData,
                 modelType,
                 modelCorr: Optional[np.ndarray] = None,
                 rDEM=None,
                 newRes=None,
                 gridEPSG=None,
                 debug=True):
        if modelCorr is None:
            modelCorr = np.zeros((3, 3))
        self.gridEPSG = gridEPSG
        self.rasterInfo = rasterInfo
        self.modelData = modelData
        self.modelType = modelType
        self.rDEM = rDEM
        self.debug = debug
        if self.rDEM is not None:
            self.demInfo = geoRT.cRasterInfo(self.rDEM)
        else:
            self.demInfo = None
        self.modelCorr = modelCorr
        self.upLeftEW = None
        self.botRightEW = None
        self.upLeftNS = None
        self.botRightNS = None
        self.gridEPSG = None
        self.resNS = None
        self.resEW = None

        self.oUpLeftEW = None
        self.oBotRightEW = None
        self.oUpLeftNS = None
        self.oBotRightNS = None

        self.__GetSatMapGrid()
        if newRes is None:
            self.oRes = np.min([self.resEW, self.resNS])
        else:
            self.oRes = newRes
        self.__ComputeoMapGrid()

        return

    def __ComputeImgGSD(self):
        global geoGround
        xPixList = [self.rasterInfo.raster_width / 2, self.rasterInfo.raster_width / 2 + 1]
        yPixList = [self.rasterInfo.raster_height / 2, self.rasterInfo.raster_height / 2 + 1]

        if self.modelType == SATELLITE_MODELS.RSM:
            from geoCosiCorr3D.geoRSM.Pixel2GroundDirectModel import cPix2GroundDirectModel

            geoCoordList = []

            for xVal, yVal in zip(xPixList, yPixList):
                pix2Ground_obj = cPix2GroundDirectModel(rsmModel=self.modelData,
                                                        xPix=xVal,
                                                        yPix=yVal,
                                                        rsmCorrectionArray=self.modelCorr,
                                                        demFile=self.rDEM)

                geoCoordList.append(pix2Ground_obj.geoCoords)
            geoGround = np.asarray(geoCoordList)

        if self.modelType == SATELLITE_MODELS.RFM:
            from geoCosiCorr3D.geoRFM.RFM import RFM
            model_data: RFM = self.modelData
            # TODO use getGSD from RFM class
            lonBBox_RFM, latBBox_RFM, altBBox_RFM = model_data.Img2Ground_RFM(col=xPixList,
                                                                              lin=yPixList,
                                                                              demInfo=self.demInfo,
                                                                              corrModel=self.modelCorr)

            geoGround = np.array([lonBBox_RFM, latBBox_RFM, altBBox_RFM]).T

        # ## Convert footprint coord to the grid projection system
        utmGround = geoRT.ConvCoordMap1ToMap2_Batch(X=list(geoGround[:, 1]),
                                                    Y=list(geoGround[:, 0]),
                                                    Z=list(geoGround[:, -1]),
                                                    targetEPSG=self.gridEPSG)

        self.resEW = np.abs(utmGround[0][0] - utmGround[0][1])
        self.resNS = np.abs(utmGround[1][0] - utmGround[1][1])

        return

    def __GetSatMapGrid(self):
        """
        Compute the image extent and projection system using the input model information.

        Returns:

        """
        # self.rasterInfo.rasterWidth,self.modelData.nbCols
        # self.rasterInfo.rasterHeight, self.modelData.nbRows

        global geoGround
        xBBox = [0, self.rasterInfo.raster_width - 1, 0, self.rasterInfo.raster_width - 1]
        yBBox = [0, 0, self.rasterInfo.raster_height - 1, self.rasterInfo.raster_height - 1]

        if self.debug:
            logging.info("... Computing Grid Extent ...")
        if self.modelType == SATELLITE_MODELS.RSM:
            from geoCosiCorr3D.geoRSM.Pixel2GroundDirectModel import cPix2GroundDirectModel

            geoCoordList = []

            for xVal, yVal in zip(xBBox, yBBox):
                pix2Ground_obj = cPix2GroundDirectModel(rsmModel=self.modelData,
                                                        xPix=xVal,
                                                        yPix=yVal,
                                                        rsmCorrectionArray=self.modelCorr,
                                                        demFile=self.rDEM)
                geoCoordList.append(pix2Ground_obj.geoCoords)
            geoGround = np.asarray(geoCoordList)

        if self.modelType == SATELLITE_MODELS.RFM:
            from geoCosiCorr3D.geoRFM.RFM import RFM
            model_data: RFM = self.modelData

            lonBBox_RFM, latBBox_RFM, altBBox_RFM = model_data.Img2Ground_RFM(col=xBBox,
                                                                              lin=yBBox,
                                                                              demInfo=self.demInfo,
                                                                              corrModel=self.modelCorr)
            geoGround = np.array([lonBBox_RFM, latBBox_RFM, altBBox_RFM]).T

        if self.gridEPSG is None:
            ##Compute the output UTM epsg code
            self.gridEPSG = geoRT.ComputeEpsg(lon=geoGround[0, 0], lat=geoGround[0, 1])
            if self.debug:
                logging.info(f'Grid projection:{self.gridEPSG}')

        # ## Convert footprint coord to the grid projection system
        utmGround = geoRT.ConvCoordMap1ToMap2_Batch(X=list(geoGround[:, 1]),
                                                    Y=list(geoGround[:, 0]),
                                                    Z=list(geoGround[:, -1]),
                                                    targetEPSG=self.gridEPSG)

        # topLeftGround = [utmGround[0][0], utmGround[1][0], utmGround[2][0]]
        # topRightGround = [utmGround[0][1], utmGround[1][1], utmGround[2][1]]
        # bottomLeftGround = [utmGround[0][2], utmGround[1][2], utmGround[2][2]]
        # bottomRightGround = [utmGround[0][3], utmGround[1][3], utmGround[2][3]]

        self.upLeftEW = utmGround[0][0]  # np.min(utmGround[0])
        self.upRightEW = utmGround[0][1]
        self.botLeftEW = utmGround[0][2]
        self.botRightEW = utmGround[0][3]  # np.max(utmGround[0])

        self.upLeftNS = utmGround[1][0]  # np.max(utmGround[1])
        self.upRightNS = utmGround[1][1]
        self.botLeftNS = utmGround[1][2]
        self.botRightNS = utmGround[1][3]  # np.min(utmGround[1])
        self.__ComputeImgGSD()
        return

    def __ComputeoMapGrid(self):
        from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import ComputeoMapGrid
        ewBBox = [self.upLeftEW, self.upRightEW, self.botLeftEW, self.botRightEW]
        nsBBox = [self.upLeftNS, self.upRightNS, self.botLeftNS, self.botRightNS]

        self.oUpLeftEW, self.oUpLeftNS, self.oBotRightEW, self.oBotRightNS = ComputeoMapGrid(upLeftEW=np.min(ewBBox),
                                                                                             upLeftNS=np.max(nsBBox),
                                                                                             botRightEW=np.max(ewBBox),
                                                                                             botRightNS=np.min(nsBBox),
                                                                                             oRes=self.oRes)
        return

    def __repr__(self):
        return """
        # Grid Information :
            projEPSG = {}
            -- ESTIMATED GRID --
            res (ew,ns) = [{:.3f} , {:.3f}] m
            upLeft= [{:.5f} , {:.5f}]
            upRight = [{:.5f} , {:.5f}]
            botLeft= [{:.5f} , {:.5f}]
            botRight = [{:.5f} , {:.5f}]
            -- OUTPUT GRID --
            oUpleftEW={:.5f}
            oBotrightEW={:.5f}
            oUpleftNS={:.5f}
            oBotrightNS={:.5f}
            oRes= {:.5f}
            ________________________""".format(self.gridEPSG,
                                               self.resEW, self.resNS,
                                               self.upLeftEW, self.upLeftNS,
                                               self.upRightEW, self.upRightNS,
                                               self.botLeftEW, self.botLeftNS,
                                               self.botRightEW, self.botRightNS,
                                               self.oUpLeftEW, self.oBotRightEW,
                                               self.oUpLeftNS, self.oBotRightNS,
                                               self.oRes)
