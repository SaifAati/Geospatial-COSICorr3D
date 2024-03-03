"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import warnings
from typing import List, Optional

import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarns
import geoCosiCorr3D.georoutines.geo_utils as geoRT
import numpy as np
from geoCosiCorr3D.geoCore.constants import EARTH
from geoCosiCorr3D.geoCore.geoRawRSMMapping import RawPix2GroundDirectModel

INTERSECTON_TOL = 1e-3
MAX_ITER = 10


class cPix2GroundDirectModel(RawPix2GroundDirectModel):

    def __init__(self, rsmModel,
                 xPix,
                 yPix,
                 rsmCorrectionArray: Optional[np.ndarray] = None,
                 demFile: Optional[str] = None,
                 hMean: Optional[float] = None,
                 oProjEPSG=None,
                 semiMajor=EARTH.SEMIMAJOR,
                 semiMinor=EARTH.SEMIMINOR,
                 debug=False):
        if rsmCorrectionArray is None:
            rsmCorrectionArray = np.zeros((3, 3))
        super().__init__(rsmModel, xPix, yPix, rsmCorrectionArray, demFile, hMean, oProjEPSG, semiMajor, semiMinor,
                         debug)

        self.geoCoords = self.compute_pix2ground_direct_model()  # LON, LAT, ALT

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
        if self.demFile is not None:
            self.demInfo = geoRT.cRasterInfo(self.demFile)

            if self.demInfo.valid_map_info:
                withDemFlag = True
                if self.oProjEPSG is not None:
                    if self.demInfo.epsg_code != self.oProjEPSG:
                        geoErrors.erNotIdenticalDatum(
                            msg="Datum of DEM and oProjection of the pixel ground location must be identical")
                else:
                    self.oProjEPSG = self.demInfo.epsg_code

        if withDemFlag == False:
            geoWarns.wrInvaliDEM()
            self.demInfo = None
            self.oProjEPSG = 4326
        if self.xPix != 0:
            self.xPix = self.xPix - 1
        if self.yPix != 0:
            self.yPix = self.yPix - 1

        if self.xPix < 0 or self.xPix > self.rsmModel.nbCols:
            logging.warning(' NEED TO EXTRAPOLATE!!')
            geoErrors.erNotImplemented(routineName="Extrapolation")
        u1 = self.compute_U1(xPix=self.xPix, rsm_model=self.rsmModel)

        if self.debug:
            logging.info(f'u1:{u1}')
        u2, u2_norm = self.compute_U2(yPix=self.yPix,
                                      rsm_model=self.rsmModel,
                                      u1=u1)
        if self.debug:
            logging.info(" u2:{}  -->  u2_norm:{}".format(u2, u2_norm))
        u3 = self.compute_U3(yPix=self.yPix,
                             rsm_model=self.rsmModel,
                             u2=u2,
                             u2_norm=u2_norm)
        if self.debug:
            logging.info(f'u3:{u3}')

        corr = self.compute_Dl_correction(rsmCorrectionArray=self.rsmCorrection,
                                          rsm_model=self.rsmModel,
                                          xPix=self.xPix,
                                          yPix=self.yPix)
        if self.debug:
            logging.info(f'dl_corr:{corr}')
        u3 = u3 + corr
        if self.debug:
            logging.info(f'u3_corr:{u3}')
        satPosX, satPosY, satPosZ = self.compute_sat_pos(rsm_model=self.rsmModel, yPix=self.yPix)
        if self.debug:
            logging.info(f'sat_pos:{satPosX, satPosY, satPosZ}')

        # Computing the new geocentric coordinates taking into account the scene topography
        ## cf. spot geometry hanbook page 41, 42, Using a DEM algorithm
        u3_X, u3_Y, u3_Z = u3[0], u3[1], u3[2]
        alphaNum1 = u3_X ** 2 + u3_Y ** 2
        alphaNum2 = u3_Z ** 2
        betaNum1 = 2 * (satPosX * u3_X + satPosY * u3_Y)
        betaNum2 = 2 * (satPosZ * u3_Z)
        # print("alphaNum1:{} ,alphaNum2:{},betaNum1:{},betaNum2{} ".format(alphaNum1, alphaNum2, betaNum1, betaNum2))

        ## If a mean height is provided, initialize the altitude with it --> otherwise set to 0
        h = 0
        if self.hMean is not None:
            h = self.hMean

        loopAgain = 1
        nbLoop = 0
        geoCoords: List = []
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
            geoCoords = geoRT.Convert.cartesian_2_geo(x=OM3_X, y=OM3_Y, z=OM3_Z)  # LON, LAT,

            tempEPSG = geoRT.ComputeEpsg(lon=geoCoords[0], lat=geoCoords[1])

            if withDemFlag == True:
                if tempEPSG != self.demInfo.epsg_code:
                    msg = "Reproject DEM from {}-->{}".format(self.demInfo.epsg_code, tempEPSG)
                    warnings.warn(msg)
                    prjDemPath = geoRT.ReprojectRaster(input_raster_path=self.demInfo.input_raster_path, o_prj=tempEPSG,
                                                       vrt=True)
                    self.demInfo = geoRT.cRasterInfo(prjDemPath)
                hNew = self.InterpolateHeightFromDEM(geoCoords=geoCoords, h=h, demInfo=self.demInfo)
                if self.debug:
                    logging.info(f'h_new:{hNew} loop:{nbLoop}')
                if np.max(np.abs(h - np.asarray(hNew))) < INTERSECTON_TOL:
                    loopAgain = 0
                else:
                    h = hNew[0]
            else:
                loopAgain = 0
            nbLoop += 1
            if nbLoop > MAX_ITER:
                loopAgain = 0

        return geoCoords

    def __repr__(self):
        pass

    def __str__(self):
        pass
