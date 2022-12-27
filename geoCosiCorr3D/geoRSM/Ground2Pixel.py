"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import numpy as np
import logging

from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import EstimateGeoTransformation, get_dem_dims, ComputeFootprint
from geoCosiCorr3D.geoOrthoResampling.geoOrtho import RSMOrtho


class RSMG2P:
    def __init__(self,
                 rsmModel,
                 xMap,
                 yMap,
                 projEPSG,
                 rsmCorrection=None,
                 demInfo=None,
                 hMean=None,
                 debug=False):
        """
        Compute pixel coordinate from a list of ground coordinates using RSM- for pushbroom system.

        Args:
            rsmModel: RSM file object
            xMap: list of easting coordinates : list
            yMap: list of northing coordinates : list
            projEPSG: EPSG code of the provided coordinates : int
            rsmCorrection: RSM correction array
            demInfo: dem raster information : geoRT.RasterInfo() object
            hMean:

        Notes:
            - we use a two-point step size (TPP) gradient algorithm for minimization.
            - coordinates should be in UTM projection
            TODO: check if projEPSG=4326 then transform coordinates to UTM
        """

        self.rsmModel = rsmModel
        self.rsmCorrection = rsmCorrection
        if self.rsmCorrection is None:
            self.rsmCorrection = np.zeros((3, 3))
        self.demInfo = demInfo
        self.projEPSG = projEPSG
        self.xMap = xMap
        self.yMap = yMap
        self.debug = debug
        self.h_mean = hMean
        oColsNb = len(yMap)
        oRowsNb = len(xMap)

        xBBox = [xMap[0], xMap[oColsNb - 1], xMap[0], xMap[oColsNb - 1]]
        yBBox = [yMap[0], yMap[0], yMap[oRowsNb - 1], yMap[oRowsNb - 1]]

        demDims = np.zeros((1, 4))
        if self.demInfo:
            demDims = get_dem_dims(xBBox=xBBox, yBBox=yBBox, demInfo=self.demInfo)
            logging.info(demDims)

        ## Estimate Initial solution only for the first row
        self.get_init_sol()

        ## Compute the mesh-grid of ground coordinates grid
        self.eastArr = np.tile(xMap, (len(yMap), 1))

        tempNorthing = [yMap[i] for i in range(len(yMap))]
        self.northArr = np.tile(tempNorthing, (len(xMap), 1)).T

        self.DEM_interpolation(demDims=demDims)

        self.oArray = self.compute_pix_coords()

    def DEM_interpolation(self, demDims):
        from geoCosiCorr3D.geoRSM.misc import HeightInterpolation

        self.h_new = HeightInterpolation.DEM_interpolation(demInfo=self.demInfo,
                                                           demDims=demDims,
                                                           tileCurrent=None,
                                                           eastArr=self.eastArr,
                                                           northArr=self.northArr)

        if self.h_new is None:
            if self.h_mean is not None:
                self.h_new = np.ones(self.eastArr.shape) * self.h_mean
                # warnings.warn("H will be set to hMean:{}".format(self.h_mean))
                logging.warning(f'H will be set to the provided hMean {self.h_mean}')
            else:
                self.h_new = np.zeros(self.eastArr.shape)
                logging.warning('H will be set to 0')
                # warnings.warn("H will be set to 0")
        return

    def get_init_sol(self):
        """

        Returns:
        Notes:
            Define an array of corresponding pixel coordinates 1A <--> Ground coordinates UTM
            Compute the 2D-affine transformation <==> geotrans for thi task we don't need the h component
        """
        easting = self.xMap
        northing = self.yMap
        oColsNb = len(easting)
        topLeftGround, topRightGround, bottomLeftGround, bottomRightGround, iXPixList, iYPixList = \
            ComputeFootprint(rsmModel=self.rsmModel,
                             demInfo=self.demInfo,
                             rsmCorrectionArray=self.rsmCorrection,
                             oProj=self.projEPSG)
        pixObs = np.array([iXPixList, iYPixList]).T
        groundObs = np.array([topLeftGround, topRightGround, bottomLeftGround, bottomRightGround])
        xBBox = [0, self.rsmModel.nbCols - 1, 0, self.rsmModel.nbCols - 1]
        yBBox = [0, 0, self.rsmModel.nbRows - 1, self.rsmModel.nbRows - 1]
        pix_obs = np.array((xBBox, yBBox)).T

        geoAffineTrans = EstimateGeoTransformation(pixObs=pixObs, groundObs=groundObs)
        if self.debug:
            logging.info(f'approx geo_transform{geoAffineTrans}')
        eastingNorthingMatrix = np.array([easting, oColsNb * [northing[0]], oColsNb * [1]])
        xyPixelInit = np.dot(eastingNorthingMatrix.T, geoAffineTrans.T)

        self.xPixInit = xyPixelInit[:, 0]
        self.yPixInit = xyPixelInit[:, 1]
        return

    def fG2Px(self):

        outX, outY, oRowsNb = RSMOrtho.rsm_g2p_minimization(rsmModel=self.rsmModel,
                                                            rsmCorrectionArray=self.rsmCorrection,
                                                            nbColsOut=len(self.yMap),
                                                            xPixInit=self.xPixInit,
                                                            yPixInit=self.yPixInit,
                                                            eastArr=self.eastArr,
                                                            northArr=self.northArr,
                                                            nbRowsOut=len(self.xMap),
                                                            hNew=self.h_new,
                                                            debug=self.debug,
                                                            target_epsg=self.projEPSG)
        return outX, outY, oRowsNb


    def compute_pix_coords(self):
        outX, outY, oRowsNb = self.fG2Px()
        oArray = np.zeros((2, outX.shape[0], outX.shape[1]))
        oArray[0, :, :] = outX
        oArray[1, :, :] = outY
        return oArray
    def get_pix_coords(self):

        return self.oArray
