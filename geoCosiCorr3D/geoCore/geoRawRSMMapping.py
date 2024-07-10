"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
import numpy as np
from geoCosiCorr3D.geoCore.base.base_RSM_Mapping import \
    BasePix2GroundDirectModel
from geoCosiCorr3D.geoCore.constants import EARTH
from geoCosiCorr3D.geoRSM.Interpol import LinearIterpolation
# from geoCosiCorr3D.geoRSM.misc import HeightInterpolation
from geoCosiCorr3D.geoCore.geoDEM import HeightInterpolation


class RawPix2GroundDirectModel(BasePix2GroundDirectModel):

    def __init__(self, rsmModel, xPix, yPix, rsmCorrectionArray=np.zeros((3, 3)), demFile=None, hMean=None,
                 oProjEPSG=None, semiMajor=EARTH.SEMIMAJOR, semiMinor=EARTH.SEMIMINOR, debug=True):

        super().__init__(rsmModel, xPix, yPix, rsmCorrectionArray, demFile, hMean, oProjEPSG, semiMajor, semiMinor,
                         debug)

    @staticmethod
    def compute_U1(xPix, rsm_model):

        angle_x = LinearIterpolation(array=rsm_model.CCDLookAngle[:, 0], location=xPix)
        angle_y = LinearIterpolation(array=rsm_model.CCDLookAngle[:, 1], location=xPix)
        angle_z = LinearIterpolation(array=rsm_model.CCDLookAngle[:, 2], location=xPix)
        u1 = np.array([angle_x, angle_y, angle_z]).T

        return u1

    @staticmethod
    def compute_U2(yPix, rsm_model, u1):
        """
        Compute the look direction of the pixel p(xPix,yPix) in the orbital coordinate system
        U1 in the satellite coordinate system
        """

        if yPix < 0 or yPix > rsm_model.nbRows:
            geoErrors.erNotImplemented(routineName="U1 Extrapolation")

        else:
            rotMat_0_x = LinearIterpolation(rsm_model.satToNavMat[:, 0, 0], yPix)
            rotMat_1_x = LinearIterpolation(rsm_model.satToNavMat[:, 0, 1], yPix)
            rotMat_2_x = LinearIterpolation(rsm_model.satToNavMat[:, 0, 2], yPix)

            rotMat_0_y = LinearIterpolation(rsm_model.satToNavMat[:, 1, 0], yPix)
            rotMat_1_y = LinearIterpolation(rsm_model.satToNavMat[:, 1, 1], yPix)
            rotMat_2_y = LinearIterpolation(rsm_model.satToNavMat[:, 1, 2], yPix)

            rotMat_0_z = LinearIterpolation(rsm_model.satToNavMat[:, 2, 0], yPix)
            rotMat_1_z = LinearIterpolation(rsm_model.satToNavMat[:, 2, 1], yPix)
            rotMat_2_z = LinearIterpolation(rsm_model.satToNavMat[:, 2, 2], yPix)

            R_sat_to_orbit = np.array([[rotMat_0_x, rotMat_1_x, rotMat_2_x],
                                       [rotMat_0_y, rotMat_1_y, rotMat_2_y],
                                       [rotMat_0_z, rotMat_1_z, rotMat_2_z]])
            u2 = R_sat_to_orbit @ u1

            u2_norm = np.sqrt(u2[0] ** 2 + u2[1] ** 2 + u2[2] ** 2)

            return u2, u2_norm

    @staticmethod
    def compute_U3(yPix, rsm_model, u2, u2_norm):

        if yPix < 0 or yPix > rsm_model.nbRows:
            geoErrors.erNotImplemented(routineName="U3 Extrapolation")

        posMat_0_x = LinearIterpolation(rsm_model.orbitalPos_X[:, 0], yPix)
        posMat_1_x = LinearIterpolation(rsm_model.orbitalPos_Y[:, 0], yPix)
        posMat_2_x = LinearIterpolation(rsm_model.orbitalPos_Z[:, 0], yPix)

        posMat_0_y = LinearIterpolation(rsm_model.orbitalPos_X[:, 1], yPix)
        posMat_1_y = LinearIterpolation(rsm_model.orbitalPos_Y[:, 1], yPix)
        posMat_2_y = LinearIterpolation(rsm_model.orbitalPos_Z[:, 1], yPix)

        posMat_0_z = LinearIterpolation(rsm_model.orbitalPos_X[:, 2], yPix)
        posMat_1_z = LinearIterpolation(rsm_model.orbitalPos_Y[:, 2], yPix)
        posMat_2_z = LinearIterpolation(rsm_model.orbitalPos_Z[:, 2], yPix)

        R_orbit_ECEF = np.array([[posMat_0_x, posMat_1_x, posMat_2_x],
                                 [posMat_0_y, posMat_1_y, posMat_2_y],
                                 [posMat_0_z, posMat_1_z, posMat_2_z]])
        u3 = R_orbit_ECEF @ u2 / u2_norm
        return u3

    @staticmethod
    def compute_Dl_correction(rsmCorrectionArray, rsm_model, yPix, xPix):

        dl = np.dot(rsmCorrectionArray.T, np.array([xPix, yPix, 1]))
        if yPix < 0 or yPix > rsm_model.nbRows:
            geoErrors.erNotImplemented(routineName="DL Extrapolation")

        return dl

    @staticmethod
    def compute_sat_pos(rsm_model, yPix):

        if yPix < 0 or yPix > rsm_model.nbRows:
            geoErrors.erNotImplemented(routineName="SAT pos Extrapolation")

        satPosX = LinearIterpolation(array=rsm_model.interpSatPosition[:, 0], location=yPix)

        satPosY = LinearIterpolation(array=rsm_model.interpSatPosition[:, 1], location=yPix)

        satPosZ = LinearIterpolation(array=rsm_model.interpSatPosition[:, 2], location=yPix)

        return satPosX, satPosY, satPosZ

    @staticmethod
    def __checkDEMSubset(xdemPix, ydemPix, demInfo):
        """

        Args:
            xdemPix:
            ydemPix:
            demInfo:

        Returns:

        """
        return HeightInterpolation.check_DEM_subset(xdemPix, ydemPix, demInfo, step=3)

    @staticmethod
    def InterpolateHeightFromDEM(geoCoords, h, demInfo):

        return HeightInterpolation.get_h_from_DEM(geo_coords=geoCoords, h=h, demInfo=demInfo)

    def compute_pix2ground_direct_model(self):

        """
        Note:
            If no DEM and oProjEPSG are passed along the ground location of the pixel is expressed in
            oProjEPSG = WGS-84 for earth.
            If DEM and oProjEPSG are passed along, check if the datum are the same on the 2 projection
             if True --> oProjEPSG == demInfo.EPSG_Code
              else errorMsg
        """
        pass
