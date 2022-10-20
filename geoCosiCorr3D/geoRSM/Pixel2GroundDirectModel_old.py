"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import warnings

import numpy as np
import geoCosiCorr3D.georoutines.georoutines as geoRT

import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarns
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors

from geoCosiCorr3D.geoConfig import cgeoCfg
from geoCosiCorr3D.geoCore.constants import EARTH

from geoCosiCorr3D.geoCore.geoRawRSMMapping import RawPix2GroundDirectModel

geoCfg = cgeoCfg()


class cPix2GroundDirectModel(RawPix2GroundDirectModel):

    def __init__(self, rsmModel, xPix, yPix, rsmCorrectionArray=np.zeros((3, 3)), demFile=None, hMean=None,
                 oProjEPSG=None, semiMajor=EARTH.SEMIMAJOR, semiMinor=EARTH.SEMIMINOR):
        """

        Args:
            rsmModel:
            xPix:
            yPix:
            rsmCorrectionArray:
            demFile:
            hMean:
            oProjEPSG:
            semiMajor:
            semiMinor:
        """
        super().__init__(rsmModel, xPix, yPix, rsmCorrectionArray, demFile, hMean, oProjEPSG, semiMajor, semiMinor)
        self.rsmModel = rsmModel
        self.xPix = xPix
        self.yPix = yPix
        self.rsmCorrection = rsmCorrectionArray
        self.demFile = demFile
        self.hMean = hMean
        self.oProjEPSG = oProjEPSG
        self.semiMajor = semiMajor
        self.semiMinor = semiMinor

        self.geoCoords = self.compute_pix2ground_direct_model()

    def compute_pix2ground_direct_model(self):

        """
        Note:
            If no DEM and oProjEPSG are passed along the ground location of the pixel is expressed in 
            oProjEPSG = WGS-84 for earth.
            If DEM and oProjEPSG are passed along, check if the datum are the same on the 2 projection
             if True --> oProjEPSG == demInfo.EPSG_Code
              else errorMsg
        """

        withDemFlag = False
        if self.demFile != None:
            self.demInfo = geoRT.RasterInfo(self.demFile, False)

            if self.demInfo.validMapInfo:
                withDemFlag = True
                if self.oProjEPSG != None:
                    if geoRT.ProjDatumIdentical(proj1=self.demInfo.EPSG_Code, proj2=self.oProjEPSG) == False:
                        geoErrors.erNotIdenticalDatum(
                            msg="Datum of DEM and oProjection of the pixel ground location must be identical")
                else:
                    self.oProjEPSG = self.demInfo.EPSG_Code

        if withDemFlag == False:
            geoWarns.wrInvaliDEM()
            self.demInfo = None
            self.oProjEPSG = 4326
        if self.xPix != 0:
            self.xPix = self.xPix - 1
        if self.yPix != 0:
            self.yPix = self.yPix - 1

        if self.xPix < 0 or self.xPix > self.rsmModel.nbCols:
            print("We need to extrapolate")
            geoErrors.erNotImplemented(routineName="Extrapolation")
        u1 = self.compute_U1(xPix=self.xPix, rsm_model=self.rsmModel)

        u2, u2_norm = self.compute_U2(yPix=self.yPix,
                                      rsm_model=self.rsmModel,
                                      u1=u1)
        # print(" u2_X:{}\n u2_Y:{}\n u2_Z:{}\n u2_norm:{}".format(u2_X, u2_Y, u2_Z, u2_norm))
        u3 = self.compute_U3(yPix=self.yPix,
                             rsm_model=self.rsmModel,
                             u2=u2,
                             u2_norm=u2_norm)

        # print("u1={}".format(u1))
        # print("u2={}".format(u2))
        # print("u3={}".format(u3))
        corr = self.compute_Dl_correction(rsmCorrectionArray=self.rsmCorrection,
                                          rsm_model=self.rsmModel,
                                          xPix=self.xPix,
                                          yPix=self.yPix)
        # print(" dx:{}\n dy:{}\n dz:{}".format(dx, dy, dz))
        ## ========================##
        u3 = u3 + corr
        satPosX, satPosY, satPosZ = self.compute_sat_pos(rsm_model=self.rsmModel, yPix=self.yPix)
        # print("satPosX:{}, satPosY:{}, satPosZ:{}".format(satPosX, satPosY, satPosZ))

        ## ========================##
        # 3Computing the new geocentric coordinates taking into account the scene topography
        ## cf. spot geometry hanbook page 41, 42, Using a DEM algorithm
        u3_X, u3_Y, u3_Z = u3[0], u3[1], u3[2]
        alphaNum1 = u3_X ** 2 + u3_Y ** 2
        alphaNum2 = u3_Z ** 2
        betaNum1 = 2 * (satPosX * u3_X + satPosY * u3_Y)
        betaNum2 = 2 * (satPosZ * u3_Z)
        # print("alphaNum1:{} ,alphaNum2:{},betaNum1:{},betaNum2{} ".format(alphaNum1, alphaNum2, betaNum1, betaNum2))

        ## If a mean height is provided, initialize the altitude with it --> otherwise set to 0
        h = 0
        if self.hMean != None:
            h = self.hMean

        loopAgain = 1
        nbLoop = 0
        while loopAgain == 1:
            # print("\nLoop:", nbLoop, "\n")
            A_2 = (self.semiMajor + h) ** 2
            B_2 = (self.semiMinor + h) ** 2
            # print(self.semiMajor, self.semiMinor, h)
            # print("A_2,B_2", A_2, B_2)
            ## Terrestrial Coordinate System pixel location
            ##computes intersection of u3 with the ellipsoid at altitude h
            alpha_ = (alphaNum1 / A_2) + (alphaNum2 / B_2)
            beta_ = (betaNum1 / A_2) + (betaNum2 / B_2)
            gamma_ = ((satPosX ** 2 + satPosY ** 2) / A_2 + (satPosZ ** 2) / B_2) - 1
            delta_ = np.sqrt(beta_ ** 2 - 4 * alpha_ * gamma_)

            temp1 = ((-beta_ - delta_) / (2 * alpha_))
            temp2 = ((-beta_ + delta_) / (2 * alpha_))
            mu = temp2
            if temp1 < temp2:
                mu = temp1
            # print("mu=", mu)
            OM3_X = u3_X * mu + satPosX
            OM3_Y = u3_Y * mu + satPosY
            OM3_Z = u3_Z * mu + satPosZ
            # print("OM3_X:{}, OM3_Y:{}, OM3_Z:{}".format(OM3_X, OM3_Y, OM3_Z))
            self.geoCoords = geoRT.ConvertCartesian2Geo(x=OM3_X, y=OM3_Y, z=OM3_Z)
            tempEPSG = geoRT.ComputeEpsg(lon=self.geoCoords[0], lat=self.geoCoords[1])

            if withDemFlag == True:
                if tempEPSG != self.demInfo.EPSG_Code:
                    msg = "Reproject DEM from {}-->{}".format(self.demInfo.EPSG_Code, tempEPSG)
                    warnings.warn(msg)
                    prjDemPath = geoRT.ReprojectRaster(iRasterPath=self.demInfo.rasterPath, oPrj=tempEPSG, vrt=True)
                    self.demInfo = geoRT.RasterInfo(prjDemPath)
                hNew = self.InterpolateHeightFromDEM(geoCoords=self.geoCoords, h=h, demInfo=self.demInfo)
                if np.max(np.abs(h - np.asarray(hNew))) < 1e-3:
                    loopAgain = 0
                else:
                    h = hNew[0]
            else:
                loopAgain = 0
            nbLoop += 1
            if nbLoop > 10:
                loopAgain = 0

        return self.geoCoords

    def __repr__(self):
        pass

    def __str__(self):
        pass


def RSM_Footprint(rsmModel, demFile=None, hMean=None, pointingCorrection=np.zeros((3, 3))):
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
