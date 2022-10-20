"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import ctypes, warnings
import numpy as np

import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarns
from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import EstimateGeoTransformation, GetDEM_dims, ComputeFootprint
from geoCosiCorr3D.geoConfig import cgeoCfg
from geoCosiCorr3D.geoRSM.Interpol import Interpolate2D
from geoCosiCorr3D.georoutines.georoutines import ConvCoordMap1ToMap2_Batch

from geoCosiCorr3D.geoCore.constants import EARTH, SOFTWARE

geoCfg = cgeoCfg()


class cGround2Pix:
    __semiMajor = EARTH.SEMIMAJOR
    __semiMinor = EARTH.SEMIMINOR
    __libPath = geoCfg.geoCosiCorr3DLib
    __imgTileSizemb = SOFTWARE.TILE_SIZE_MB

    def __init__(self,
                 rsmModel,
                 xMap,
                 yMap,
                 projEPSG,
                 rsmCorrection=np.zeros((3, 3)),
                 demInfo=None,
                 hMean=None,
                 ):
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
        self.demInfo = demInfo
        self.projEPSG = projEPSG
        self.xMap = xMap
        self.yMap = yMap
        oColsNb = len(yMap)
        oRowsNb = len(xMap)

        xBBox = [xMap[0], xMap[oColsNb - 1], xMap[0], xMap[oColsNb - 1]]
        yBBox = [yMap[0], yMap[0], yMap[oRowsNb - 1], yMap[oRowsNb - 1]]

        demDims = np.zeros((1, 4))
        if self.demInfo:
            demDims = GetDEM_dims(xBBox=xBBox, yBBox=yBBox, demInfo=self.demInfo)

        ## Estimate Intial solution only for the first row
        self.EstimateInitalSol()

        ## Compute the meshgrid of ground coordinates grid
        self.eastArr = np.tile(xMap, (len(yMap), 1))
        tempNorthing = [yMap[i] for i in range(len(yMap))]
        self.northArr = np.tile(tempNorthing, (len(xMap), 1)).T

        # self.geoCoords = self.callPix2GroundDirectModel()
        # print(self.geoCoords)

        try:
            demCol = (self.eastArr - self.demInfo.xOrigin) / np.abs(self.demInfo.pixelWidth) - demDims[0]
            demRow = (self.demInfo.yOrigin - self.northArr) / np.abs(self.demInfo.pixelHeight) - demDims[2]
            # tempWindow = demDims[ :]
            # print("tempWindow",tempWindow)
            demSubset = self.demInfo.ImageAsArray_Subset(xOffsetMin=int(demDims[0]), xOffsetMax=int(demDims[1]) + 1,
                                                         yOffsetMin=int(demDims[2]), yOffsetMax=int(demDims[3]) + 1)
            hNew_flatten = Interpolate2D(inArray=demSubset, x=demRow.flatten(), y=demCol.flatten(),
                                         kind="linear")
            self.hNew = np.reshape(hNew_flatten, demCol.shape)
            # print("hNew:",hNew.shape)
        except:
            demCol = 0 * self.eastArr

            geoWarns.wrSubsetOutsideBoundries("rDem")
            warnings.warn("Optimization subset is outside rDem boundaries")
            self.hNew = np.zeros(demCol.shape)

        outX, outY, oRowsNb = self.fGround2Pixel()

        self.oArray = np.zeros((2, outX.shape[0], outX.shape[1]))
        self.oArray[0, :, :] = outX
        self.oArray[1, :, :] = outY

    def EstimateInitalSol(self):
        """

        Returns:

        """

        ## Define an array of corresponding pixel coordinates 1A <--> Ground coordinates UTM
        ## Compute the the 2D-affine transformation <==> geotrans
        ## for thi task we dont need the h component
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
        # print(groundObs)
        geoAffineTrans = EstimateGeoTransformation(pixObs=pixObs, groundObs=groundObs)
        # print("geoAffineTrans:\n", geoAffineTrans)
        eastingNorthingMatrix = np.array([easting, oColsNb * [northing[0]], oColsNb * [1]])
        xyPixelInit = np.dot(eastingNorthingMatrix.T, geoAffineTrans.T)

        self.xPixInit = xyPixelInit[:, 0]
        self.yPixInit = xyPixelInit[:, 1]

        return

    def fGround2Pixel(self):
        s2n00 = np.array(self.rsmModel.satToNavMat[:, 0, 0], dtype=np.float)
        s2n10 = np.array(self.rsmModel.satToNavMat[:, 0, 1], dtype=np.float)
        s2n20 = np.array(self.rsmModel.satToNavMat[:, 0, 2], dtype=np.float)

        s2n01 = np.array(self.rsmModel.satToNavMat[:, 1, 0], dtype=np.float)
        s2n11 = np.array(self.rsmModel.satToNavMat[:, 1, 1], dtype=np.float)
        s2n21 = np.array(self.rsmModel.satToNavMat[:, 1, 2], dtype=np.float)

        s2n02 = np.array(self.rsmModel.satToNavMat[:, 2, 0], dtype=np.float)
        s2n12 = np.array(self.rsmModel.satToNavMat[:, 2, 1], dtype=np.float)
        s2n22 = np.array(self.rsmModel.satToNavMat[:, 2, 2], dtype=np.float)

        ccd0 = np.array(self.rsmModel.CCDLookAngle[:, 0], dtype=np.float)
        ccd1 = np.array(self.rsmModel.CCDLookAngle[:, 1], dtype=np.float)
        ccd2 = np.array(self.rsmModel.CCDLookAngle[:, 2], dtype=np.float)

        orbX0 = np.array(self.rsmModel.orbitalPos_X[:, 0], dtype=np.float)
        orbY0 = np.array(self.rsmModel.orbitalPos_Y[:, 0], dtype=np.float)
        orbZ0 = np.array(self.rsmModel.orbitalPos_Z[:, 0], dtype=np.float)

        orbX1 = np.array(self.rsmModel.orbitalPos_X[:, 1], dtype=np.float)
        orbY1 = np.array(self.rsmModel.orbitalPos_Y[:, 1], dtype=np.float)
        orbZ1 = np.array(self.rsmModel.orbitalPos_Z[:, 1], dtype=np.float)

        orbX2 = np.array(self.rsmModel.orbitalPos_X[:, 2], dtype=np.float)
        orbY2 = np.array(self.rsmModel.orbitalPos_Y[:, 2], dtype=np.float)
        orbZ2 = np.array(self.rsmModel.orbitalPos_Z[:, 2], dtype=np.float)

        satPos0 = np.array(self.rsmModel.interpSatPosition[:, 0], dtype=np.float)
        satPos1 = np.array(self.rsmModel.interpSatPosition[:, 1], dtype=np.float)
        satPos2 = np.array(self.rsmModel.interpSatPosition[:, 2], dtype=np.float)

        cartPlane = np.array(self.rsmCorrection.T.flatten(), dtype=np.float)

        (lat_fl, long_fl) = ConvCoordMap1ToMap2_Batch(X=self.eastArr.flatten(),
                                                      Y=self.northArr.flatten(),
                                                      targetEPSG=4326,
                                                      sourceEPSG=self.projEPSG)
        latArr = np.reshape(lat_fl, self.eastArr.shape)
        longArr = np.reshape(long_fl, self.northArr.shape)
        longArr_f = np.array(longArr * (np.pi / 180), dtype=np.float).T
        latArr_f = np.array(latArr * (np.pi / 180), dtype=np.float).T

        oRowsNb, oColsNb = self.eastArr.shape[0], self.eastArr.shape[1]
        outX = np.zeros((oRowsNb, oColsNb), dtype=np.float)
        outY = np.zeros((oRowsNb, oColsNb), dtype=np.float)
        outX_f = outX.T
        outY_f = outY.T

        oColsNb_f = ctypes.c_int(oColsNb)
        oRowsNb_f = ctypes.c_int(oRowsNb)
        nbcolanc_f = ctypes.c_int(self.rsmModel.nbCols)
        nbrowanc_f = ctypes.c_int(self.rsmModel.nbRows)

        xPixInit_f = np.array(self.xPixInit, dtype=np.float)
        yPixInit_f = np.array(self.yPixInit, dtype=np.float)

        h_f = np.array(self.hNew, dtype=np.float).T
        semiMajor_f = ctypes.c_double(self.__semiMajor)
        semiMinor_f = ctypes.c_double(self.__semiMinor)

        libPath_ = ctypes.util.find_library(self.__libPath)

        if not libPath_:
            geoErrors.erLibNotFound(libPath=self.__libPath)
        try:
            fLib = ctypes.CDLL(libPath_)
        except OSError:
            geoErrors.erLibLoading(libPath=libPath_)

        fLib.ground2pixel_(ctypes.byref(oColsNb_f),
                           ctypes.byref(oRowsNb_f),
                           s2n00.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n10.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n20.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n01.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n11.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n21.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n02.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n12.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           s2n22.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           ccd0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           ccd1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           ccd2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbX0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbY0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbZ0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbX1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbY1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbZ1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbX2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbY2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           orbZ2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           satPos0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           satPos1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           satPos2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           cartPlane.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),

                           ctypes.byref(nbcolanc_f),
                           ctypes.byref(nbrowanc_f),
                           xPixInit_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           yPixInit_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           longArr_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           latArr_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           h_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           ctypes.byref(semiMajor_f), ctypes.byref(semiMinor_f),
                           outX_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           outY_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           )
        outX = outX_f.T
        outY = outY_f.T
        return outX, outY, oRowsNb
