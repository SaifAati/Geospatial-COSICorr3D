"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import warnings
from typing import List, Optional, Union, Tuple

import numpy as np

import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarns
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoCore.constants import EARTH
from geoCosiCorr3D.geoCore.geoRawRSMMapping import RawPix2GroundDirectModel

# from geoCosiCorr3D.geoRSM.Interpol import LinearIterpolation
import geoCosiCorr3D.geoRSM.utils as utils
from geoCosiCorr3D.geoRSM.geoRSM_metadata.SatMetadata import SatAnc, cGetDGMetadata
import geopandas
from shapely import Polygon

INTERSECTON_TOL = 1e-3
MAX_ITER = 10


# TODO:
## CLI geoCCtransform anc_file lin col --> lla
##
class Transformer:
    def I2G(self, lin, col):
        return

    def G2I(self, lon, lat, alt):
        return


class RSMTransformer:

    def __init__(self,
                 sat_model: SatAnc,
                 rsm_corr_model=None,
                 elevation: Union[str, float] = None,
                 debug=True):

        self.sat_model = sat_model
        self.elevation = elevation
        self.debug = debug
        self.rsm_corr_model = rsm_corr_model
        if self.rsm_corr_model is None:
            self.rsm_corr_model = np.zeros((3, 3))

    def compute_U1(self, col):
        angle_x = utils.Interpolate.linear(array=self.sat_model.look_angles[:, 0], location=col)
        angle_y = utils.Interpolate.linear(array=self.sat_model.look_angles[:, 1], location=col)
        angle_z = utils.Interpolate.linear(array=self.sat_model.look_angles[:, 2], location=col)
        u1 = np.array([angle_x, angle_y, angle_z]).T

        return u1

    def compute_U2(self, line):
        """
        Compute the look direction of the pixel p(xPix,yPix) in the orbital coordinate system
        U1 in the satellite coordinate system
        """

        if line < 0 or line > self.sat_model.nb_rows:
            raise NotImplementedError('Extrapolation not implemented for U2')

        else:
            rotMat_0_x = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 0, 0], line)
            rotMat_1_x = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 0, 1], line)
            rotMat_2_x = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 0, 2], line)

            rotMat_0_y = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 1, 0], line)
            rotMat_1_y = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 1, 1], line)
            rotMat_2_y = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 1, 2], line)

            rotMat_0_z = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 2, 0], line)
            rotMat_1_z = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 2, 1], line)
            rotMat_2_z = utils.Interpolate.linear(self.sat_model.att_rot_sat_to_orbital[:, 2, 2], line)

            R_sat_to_orbit = np.array([[rotMat_0_x, rotMat_1_x, rotMat_2_x],
                                       [rotMat_0_y, rotMat_1_y, rotMat_2_y],
                                       [rotMat_0_z, rotMat_1_z, rotMat_2_z]])
            u2 = R_sat_to_orbit @ self.u1

            u2_norm = np.sqrt(u2[0] ** 2 + u2[1] ** 2 + u2[2] ** 2)

            return u2, u2_norm

    def compute_U3(self, line):

        if line < 0 or line > self.sat_model.nb_rows:
            raise NotImplementedError('Extrapolation not implemented for U3')

        posMat_0_x = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 0], line)
        posMat_1_x = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 3], line)
        posMat_2_x = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 6], line)

        posMat_0_y = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 1], line)
        posMat_1_y = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 4], line)
        posMat_2_y = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 7], line)

        posMat_0_z = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 2], line)
        posMat_1_z = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 5], line)
        posMat_2_z = utils.Interpolate.linear(self.sat_model.sat_pos_orbital[:, 8], line)

        R_orbit_ECEF = np.array([[posMat_0_x, posMat_1_x, posMat_2_x],
                                 [posMat_0_y, posMat_1_y, posMat_2_y],
                                 [posMat_0_z, posMat_1_z, posMat_2_z]])
        u3 = R_orbit_ECEF @ self.look_direction / self.look_direction_norm
        return u3

    def I2G(self, col, line):

        """
        Note:
            If no DEM and oProjEPSG are passed along the ground location of the pixel is expressed in
            oProjEPSG = WGS-84 for earth.
            If DEM and oProjEPSG are passed along, check if the datum are the same on the 2 projection
             if True --> oProjEPSG == demInfo.EPSG_Code
              else errorMsg
        """
        global lla
        self.u1 = self.compute_U1(col)

        self.look_direction, self.look_direction_norm = self.compute_U2(line)

        self.u3 = self.compute_U3(line)

        self.corr = self.apply_correction(col=col, lin=line)
        h_mean = None  # FIXME: compute hmean based on elevation file else hmean = elevation if exists
        if isinstance(self.elevation, str):
            self.dem_info = geoRT.cRasterInfo(self.elevation)

        #     if self.demInfo.valid_map_info:
        #
        #         if self.oProjEPSG is not None:
        #             if self.demInfo.epsg_code != self.oProjEPSG:
        #                 geoErrors.erNotIdenticalDatum(
        #                     msg="Datum of DEM and oProjection of the pixel ground location must be identical")
        #         else:
        #             self.oProjEPSG = self.demInfo.epsg_code
        #
        # if withDemFlag == False:
        #     geoWarns.wrInvaliDEM()
        #     self.demInfo = None
        #     self.oProjEPSG = 4326
        # if col != 0:
        #     col = col - 1
        # if line != 0:
        #     line = line - 1
        #
        # if self.xPix < 0 or self.xPix > self.rsmModel.nbCols:
        #     logging.warning(' NEED TO EXTRAPOLATE!!')
        #     geoErrors.erNotImplemented(routineName="Extrapolation")

        # if self.debug:
        #     logging.info(" u2:{}  -->  u2_norm:{}".format(u2, u2_norm))
        # u3 = self.compute_U3(yPix=self.yPix,
        #                      rsm_model=self.rsmModel,
        #                      u2=u2,
        #                      u2_norm=u2_norm)
        # if self.debug:
        #     logging.info(f'u3:{u3}')

        u3 = self.u3 + self.corr

        satPosX, satPosY, satPosZ = self.sat_pos_at_line(rsm_model=self.sat_model, line=line)
        # Computing the new geocentric coordinates taking into account the scene topography
        u3_X, u3_Y, u3_Z = u3[0], u3[1], u3[2]
        alphaNum1 = u3_X ** 2 + u3_Y ** 2
        alphaNum2 = u3_Z ** 2
        betaNum1 = 2 * (satPosX * u3_X + satPosY * u3_Y)
        betaNum2 = 2 * (satPosZ * u3_Z)

        ## If a mean height is provided, initialize the altitude with it --> otherwise set to 0
        h = 0
        if h_mean is not None:
            h = h_mean

        loopAgain = 1
        nbLoop = 0

        while loopAgain == 1:
            A_2 = (EARTH.SEMIMAJOR + h) ** 2
            B_2 = (EARTH.SEMIMINOR + h) ** 2
            # Computes intersection of u3 with the ellipsoid at altitude h
            alpha_ = (alphaNum1 / A_2) + (alphaNum2 / B_2)
            beta_ = (betaNum1 / A_2) + (betaNum2 / B_2)
            gamma_ = ((satPosX ** 2 + satPosY ** 2) / A_2 + (satPosZ ** 2) / B_2) - 1
            delta_ = np.sqrt(beta_ ** 2 - 4 * alpha_ * gamma_)

            temp1 = ((-beta_ - delta_) / (2 * alpha_))
            temp2 = ((-beta_ + delta_) / (2 * alpha_))
            mu = temp2
            if temp1 < temp2:
                mu = temp1
            OM3_X = u3_X * mu + satPosX
            OM3_Y = u3_Y * mu + satPosY
            OM3_Z = u3_Z * mu + satPosZ
            # print("OM3_X:{}, OM3_Y:{}, OM3_Z:{}".format(OM3_X, OM3_Y, OM3_Z))
            lla = geoRT.Convert.cartesian_2_geo(x=OM3_X, y=OM3_Y, z=OM3_Z)  # LON, LAT,

            # tempEPSG = geoRT.ComputeEpsg(lon=geoCoords[0], lat=geoCoords[1])

            # if withDemFlag == True:
            #     if tempEPSG != self.demInfo.epsg_code:
            #         msg = "Reproject DEM from {}-->{}".format(self.demInfo.epsg_code, tempEPSG)
            #         warnings.warn(msg)
            #         prjDemPath = geoRT.ReprojectRaster(input_raster_path=self.demInfo.input_raster_path, o_prj=tempEPSG,
            #                                            vrt=True)
            #         self.demInfo = geoRT.cRasterInfo(prjDemPath)
            #     hNew = self.InterpolateHeightFromDEM(geoCoords=geoCoords, h=h, demInfo=self.demInfo)
            #
            #     if np.max(np.abs(h - np.asarray(hNew))) < INTERSECTON_TOL:
            #         loopAgain = 0
            #     else:
            #         h = hNew[0]
            # else:
            #     loopAgain = 0
            nbLoop += 1
            if nbLoop > MAX_ITER:
                loopAgain = 0

        return lla

    def G2I(self):
        return

    def apply_correction(self, lin: float, col: float):

        dl = np.dot(self.rsm_corr_model.T, np.array([col, lin, 1]))
        if lin < 0 or lin > self.sat_model.nb_rows:
            raise NotImplementedError(('Extrapolation not implemented for DL'))

        return dl

    @staticmethod
    def sat_pos_at_line(rsm_model: SatAnc, line) -> List:

        if line < 0 or line > rsm_model.nb_rows:
            geoErrors.erNotImplemented(routineName="SAT pos Extrapolation")

        sat_pos_x = utils.Interpolate.linear(array=rsm_model.interp_eph[:, 0], location=line)
        sat_pos_y = utils.Interpolate.linear(array=rsm_model.interp_eph[:, 1], location=line)
        sat_pos_z = utils.Interpolate.linear(array=rsm_model.interp_eph[:, 2], location=line)

        return [sat_pos_x, sat_pos_y, sat_pos_z]

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


def compute_rsm_footprint(rsm_model: SatAnc,
                          dem_file: Optional[str] = None,
                          rsm_corr_model: Optional[np.ndarray] = None,
                          fp_fn: str = None) -> Tuple[geopandas.GeoDataFrame, Polygon]:
    if rsm_corr_model is None:
        rsm_corr_model = np.zeros((3, 3))

    colBBox = [0, rsm_model.nb_cols - 1, rsm_model.nb_cols - 1, 0]
    linBBox = [0, 0, rsm_model.nb_rows - 1, rsm_model.nb_rows - 1]
    tr = RSMTransformer(sat_model=rsm_model, rsm_corr_model=rsm_corr_model, elevation=dem_file)
    extent_coords = [tr.I2G(col=x, line=y) for x, y in zip(colBBox, linBBox)]
    coordinates = [[coord[0], coord[1], coord[2]] for coord in extent_coords]
    fp_gdf = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[Polygon(coordinates)])
    if fp_fn is None:
        fp_fn = f"{rsm_model.platform}_{rsm_model.time_str}_rsm_fp.geojson"
    fp_gdf.to_file(fp_fn, driver='GeoJSON')
    poly_fp = fp_gdf.iloc[0]['geometry']
    return fp_gdf, poly_fp


if __name__ == '__main__':
    # from geoCosiCorr3D.geoRSM.DigitalGlobe_RSM import cDigitalGlobe
    # from geoCosiCorr3D.geoRSM.Pixel2GroundDirectModel import \
    #     cPix2GroundDirectModel
    #
    # rsm_sensor_model = cDigitalGlobe(
    #     dgFile='/home/saif/PycharmProjects/Geospatial-COSICorr3D/tests/test_dataset/WV2.XML', debug=True)
    #
    # pix2Ground_obj = cPix2GroundDirectModel(rsmModel=rsm_sensor_model, xPix=500, yPix=500)
    # lla1 = pix2Ground_obj.geoCoords
    # print(lla1)
    # print('###########################################')
    sat_anc = cGetDGMetadata(anc_file='/home/saif/PycharmProjects/Geospatial-COSICorr3D/tests/test_dataset/WV2.XML',
                             debug=True)
    anc_fp = sat_anc.get_anc_footprint()

    rsm_fp, _ = compute_rsm_footprint(rsm_model=sat_anc)
    # TODO the best why is to compare with respect to RPC based FP
    d = utils.mean_distance_between_fps(anc_fp, rsm_fp)
    print(d)
    pass
