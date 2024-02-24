"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import ctypes
# import ctypes.util
import logging
import os
import sys
from inspect import currentframe
from typing import Dict, Optional

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarns
import geoCosiCorr3D.georoutines.geo_utils as geoRT
import numpy as np
import psutil
from geoCosiCorr3D.geoCore.core_RSM import RSM
from geoCosiCorr3D.geoCore.geoRawInvOrtho import RawInverseOrtho
from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import \
    EstimateGeoTransformation
from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import SatMapGrid
from geoCosiCorr3D.geoOrthoResampling.geoResampling import Resampling
from geoCosiCorr3D.geoRFM.RFM import RFM

geoWarns.wrIgnoreNotGeoreferencedWarning()
process = psutil.Process(os.getpid())
G2P_LIB_Path = C.SOFTWARE.GEO_COSI_CORR3D_LIB

# TODO CHANGE RSM class location
Converter = geoRT.Convert()


class Ortho():
    pass


class RSMOrtho(RawInverseOrtho):

    def __init__(self,
                 input_l1a_path: str,
                 output_ortho_path: str,
                 ortho_params: Optional[Dict] = None,
                 output_trans_path: Optional[str] = None,
                 dem_path: Optional[str] = None,
                 debug: bool = True):

        if ortho_params is None:
            ortho_params = {}
        super().__init__(input_l1a_path, output_ortho_path, output_trans_path, ortho_params, dem_path, debug)
        self.check_sensor_type()
        self.orthorectify()
        return

    def orthorectify(self):
        self.model = self._get_rsm_model()

        self._get_correction_model()
        self.ortho_grid = self._set_ortho_grid()
        if self.debug:
            if self.debug:
                self.ortho_grid.grid_fp(o_folder=None)  # TODO set wd_folder
                logging.info(f'{self.__class__.__name__}:{repr(self.ortho_grid)}')
        self._check_dem_prj()
        self.ortho_geo_transform = self.set_ortho_geo_transform()
        if self.debug:
            logging.info(f'{self.__class__.__name__}:ortho_geo_transform:{self.ortho_geo_transform}')

        need_loop = True
        index = 1
        yOff = 0

        self.compute_num_tiles()
        if self.debug:
            logging.info(
                f'{self.__class__.__name__}:Raster: W:{self.o_raster_w}, H:{self.o_raster_h}, #tiles:{self.n_tiles}')

        if self.n_tiles > 1:
            self.write_ortho_per_tile()

        dem_dims, easting, northing, nbTiles, tiles = self.ortho_tiling(
            nbRowsPerTile=self.nb_rows_per_tile,
            nbRowsOut=self.o_raster_h,
            nbColsOut=self.o_raster_w,
            need_loop=need_loop,
            oUpLeftEW=self.ortho_grid.o_up_left_ew,
            oUpLeftNS=self.ortho_grid.o_up_left_ns,
            xRes=self.ortho_grid.o_res,
            yRes=self.ortho_grid.o_res,
            demInfo=self.dem_raster_info,
        )
        xPixInit, yPixInit = self.compute_initial_approx(modelData=self.model,
                                                         oGrid=self.ortho_grid,
                                                         easting=easting,
                                                         northing=northing,
                                                         oRasterW=self.o_raster_w)
        ortho_data = {"easting": easting,
                      "northing": northing,
                      "current_tile": 0,
                      "xPixInit": xPixInit,
                      "yPixInit": yPixInit,
                      "dem_dims": dem_dims,
                      "nbTiles": nbTiles,
                      "tiles": tiles,
                      }

        while need_loop is True:
            if self.debug:
                if self.debug:
                    logging.info(f'========= Tile:{index} /{self.n_tiles} =============')
            matTile, need_loop, ortho_data = self.compute_transformation_matrix(ortho_data=ortho_data)
            if self.debug:
                logging.info(f'{self.__class__.__name__}:mat_tile.shape:{matTile.shape}')
                logging.info(f'{self.__class__.__name__}:Resampling::{self.resampling_method}')

            resample = Resampling(input_raster_info=self.l1a_raster_info, transformation_mat=matTile,
                                  resampling_params={'method': self.resampling_method})
            oOrthoTile = resample.resample()

            if self.debug:
                logging.info(f'{self.__class__.__name__}: rss = {process.memory_info().rss * 1e-6} Mb')

            yOff = self.write_ortho_rasters(oOrthoTile=oOrthoTile, matTile=matTile, yOff=yOff)
            index = index + 1

        if self.n_tiles > 1:
            self.oOrthoRaster = None
            self.oTranRaster = None
        return

    def _get_rsm_model(self):
        return RSM.build_RSM(metadata_file=self.metadata, sensor_name=self.sensor, debug=self.debug)
        # return geoRSMMisc.CheckRSM(modelPath=self.metadata,
        #                            rawImgPath=self.input_l1a_path,
        #                            sensor=self.sensor)

    def _get_correction_model(self):
        if self.corr_model_file is None:
            cf = currentframe()
            geoWarns.wrNoRSMCorrection(line=str(cf.f_back.f_lineno))
            self.corr_model = np.zeros((3, 3))
            if self.debug:
                logging.warning(
                    'No RSM correction file, initial physical model will be used for the ortho-rectification process')
                logging.info(f'Correction model file:{self.corr_model_file}')
                logging.info(f'Correction model:{self.corr_model}')
        else:
            try:
                self.corr_model = np.loadtxt(self.corr_model_file)
                if self.debug:
                    logging.info(f'Correction model file:{self.corr_model_file}')
                    logging.info(f'Correction model:{self.corr_model}')
            except:
                geoErrors.erReadCorrectionFile()

        return

    def compute_transformation_matrix(self, ortho_data):
        """
        Core process of computing the orthorectification matrices of satellite images.
        Inputs are in-memory whereas output can be  written onto a file or returned in memory
        """

        easting = ortho_data["easting"]
        northing = ortho_data["northing"]
        xPixInit = ortho_data["xPixInit"]
        yPixInit = ortho_data["yPixInit"]
        tiles = ortho_data["tiles"]
        current_tile = ortho_data["current_tile"]
        if self.debug:
            logging.info(f'{self.__class__.__name__}:Current Tile:{current_tile + 1}')
        ## We assume that eastArry and northArray are in the same coordinate system as the DEM if exist.
        ## No conversion or proj system is needed
        nbRowsOut = tiles[current_tile + 1] - tiles[current_tile]
        eastArr = np.tile(easting, (nbRowsOut, 1))

        tempNorthing = [northing[tiles[current_tile] + i] for i in range(nbRowsOut)]
        northArr = np.tile(tempNorthing, (self.o_raster_w, 1)).T
        hNew = self.elev_interpolation(ortho_data['dem_dims'], current_tile, eastArr, northArr)
        outX, outY, nbRowsOut = self.rsm_g2p_minimization(rsmModel=self.model,
                                                          rsmCorrectionArray=self.corr_model,
                                                          nbColsOut=self.o_raster_w,
                                                          xPixInit=xPixInit,
                                                          yPixInit=yPixInit,
                                                          eastArr=eastArr,
                                                          northArr=northArr,
                                                          nbRowsOut=nbRowsOut,
                                                          hNew=hNew,
                                                          debug=self.debug,
                                                          target_epsg=self.ortho_grid.grid_epsg)
        if current_tile != self.n_tiles - 1:
            need_loop = True

            ortho_data["current_tile"] = current_tile + 1
            ortho_data["xPixInit"] = outX[nbRowsOut - 1, :]
            ortho_data["yPixInit"] = outY[nbRowsOut - 1, :]
        else:
            need_loop = False

        oArray = np.zeros((2, outX.shape[0], outX.shape[1]))
        oArray[0, :, :] = outX
        oArray[1, :, :] = outY
        return oArray, need_loop, ortho_data

    @staticmethod
    def rsm_g2p_minimization(rsmModel,
                             rsmCorrectionArray,
                             nbColsOut,
                             xPixInit,
                             yPixInit,
                             eastArr,
                             northArr,
                             nbRowsOut,
                             hNew,
                             debug,
                             target_epsg):
        ## Define pointers to Fortran subroutines
        if debug:
            logging.info('... ortho optimization ...')
        s2n00 = np.array(rsmModel.satToNavMat[:, 0, 0], dtype=np.float64)
        s2n10 = np.array(rsmModel.satToNavMat[:, 0, 1], dtype=np.float64)
        s2n20 = np.array(rsmModel.satToNavMat[:, 0, 2], dtype=np.float64)

        s2n01 = np.array(rsmModel.satToNavMat[:, 1, 0], dtype=np.float64)
        s2n11 = np.array(rsmModel.satToNavMat[:, 1, 1], dtype=np.float64)
        s2n21 = np.array(rsmModel.satToNavMat[:, 1, 2], dtype=np.float64)

        s2n02 = np.array(rsmModel.satToNavMat[:, 2, 0], dtype=np.float64)
        s2n12 = np.array(rsmModel.satToNavMat[:, 2, 1], dtype=np.float64)
        s2n22 = np.array(rsmModel.satToNavMat[:, 2, 2], dtype=np.float64)

        ccd0 = np.array(rsmModel.CCDLookAngle[:, 0], dtype=np.float64)
        ccd1 = np.array(rsmModel.CCDLookAngle[:, 1], dtype=np.float64)
        ccd2 = np.array(rsmModel.CCDLookAngle[:, 2], dtype=np.float64)

        orbX0 = np.array(rsmModel.orbitalPos_X[:, 0], dtype=np.float64)
        orbY0 = np.array(rsmModel.orbitalPos_Y[:, 0], dtype=np.float64)
        orbZ0 = np.array(rsmModel.orbitalPos_Z[:, 0], dtype=np.float64)

        orbX1 = np.array(rsmModel.orbitalPos_X[:, 1], dtype=np.float64)
        orbY1 = np.array(rsmModel.orbitalPos_Y[:, 1], dtype=np.float64)
        orbZ1 = np.array(rsmModel.orbitalPos_Z[:, 1], dtype=np.float64)

        orbX2 = np.array(rsmModel.orbitalPos_X[:, 2], dtype=np.float64)
        orbY2 = np.array(rsmModel.orbitalPos_Y[:, 2], dtype=np.float64)
        orbZ2 = np.array(rsmModel.orbitalPos_Z[:, 2], dtype=np.float64)

        satPos0 = np.array(rsmModel.interpSatPosition[:, 0], dtype=np.float64)
        satPos1 = np.array(rsmModel.interpSatPosition[:, 1], dtype=np.float64)
        satPos2 = np.array(rsmModel.interpSatPosition[:, 2], dtype=np.float64)

        cartPlane = np.array(rsmCorrectionArray.T.flatten(), dtype=np.float64)

        (lat_fl, long_fl) = geoRT.ConvCoordMap1ToMap2_Batch(X=eastArr.flatten(),
                                                            Y=northArr.flatten(),
                                                            targetEPSG=4326,
                                                            sourceEPSG=target_epsg)

        # NOTE: DO NOT REMOVE THIS CODE, FOR DEBUG PURPOSE
        # np.savez_compressed(
        #     "/home/cosicorr/0-WorkSpace/geoCosiCorr3D_TestWorksapce/Ground2Pix_dev/SPOT/Post_SPOT_VS_2000-07-12-09-02-45-Spot-4-HRVIR-2-M-10"
        #     "/TestData_" + str(self.oRes) + ".npz",
        #     eastArr=eastArr,
        #     northArr=northArr,
        #     xPixInit=xPixInit,
        #     yPixInit=yPixInit,
        #     hNew=hNew,
        #     nbRowsOut=nbRowsOut,
        #     nbColsOut=nbColsOut,
        #     semiMajor=self.__semiMajor,
        #     semiMinor=self.__semiMinor,
        #     gridEPSG=self.oGrid.gridEPSG
        # )

        latArr = np.reshape(lat_fl, eastArr.shape)
        longArr = np.reshape(long_fl, northArr.shape)

        outX = np.zeros((nbRowsOut, nbColsOut), dtype=np.float64)
        outY = np.zeros((nbRowsOut, nbColsOut), dtype=np.float64)

        # libPath_ = ctypes.util.find_library(G2P_LIB_Path)
        # if not libPath_:
        #     msg = "Unable to find the specified library:" + G2P_LIB_Path
        #     sys.exit(msg)
        try:
            fLib = ctypes.CDLL(G2P_LIB_Path)
        except OSError:
            sys.exit("Unable to load the system C library")
        if debug:
            logging.info(f'lib path:{G2P_LIB_Path}')

        nbColsOut_f = ctypes.c_int(nbColsOut)
        nbRowsOut_f = ctypes.c_int(nbRowsOut)
        nbcolanc_f = ctypes.c_int(rsmModel.nbCols)
        nbrowanc_f = ctypes.c_int(rsmModel.nbRows)

        xPixInit_f = np.array(xPixInit, dtype=np.float64)
        yPixInit_f = np.array(yPixInit, dtype=np.float64)
        longArr_f = np.array(longArr * (np.pi / 180), dtype=np.float64).T
        latArr_f = np.array(latArr * (np.pi / 180), dtype=np.float64).T
        h_f = np.array(hNew, dtype=np.float64).T
        semiMajor_f = ctypes.c_double(C.EARTH.SEMIMAJOR)
        semiMinor_f = ctypes.c_double(C.EARTH.SEMIMINOR)
        outX_f = outX.T
        outY_f = outY.T
        fLib.ground2pixel_(ctypes.byref(nbColsOut_f),
                           ctypes.byref(nbRowsOut_f),
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

        return outX, outY, nbRowsOut

    @staticmethod
    def compute_initial_approx(modelData, oGrid: SatMapGrid, easting, northing, oRasterW):
        """
        Define an array of corresponding pixel coordinates 1A <--> Ground coordinates UTM
        Compute the 2D-affine transformation <==> geotrans for this task we don't need the h component
        """

        xBBox = [0, modelData.nbCols - 1, 0, modelData.nbCols - 1]
        yBBox = [0, 0, modelData.nbRows - 1, modelData.nbRows - 1]

        topLeftGround = [oGrid.up_left_ew, oGrid.up_left_ns]
        topRightGround = [oGrid.up_right_ew, oGrid.up_right_ns]
        bottomLeftGround = [oGrid.bot_left_ew, oGrid.bot_left_ns]
        bottomRightGround = [oGrid.bot_right_ew, oGrid.bot_right_ns]

        pixObs = np.array([xBBox, yBBox]).T
        groundObs = np.array([topLeftGround, topRightGround, bottomLeftGround, bottomRightGround])
        geoAffineTrans = EstimateGeoTransformation(pixObs=pixObs, groundObs=groundObs)

        eastingNorthingMatrix = np.array([easting, oRasterW * [northing[0]], oRasterW * [1]])
        xyPixelInit = np.dot(eastingNorthingMatrix.T, geoAffineTrans.T)
        xPixInit = xyPixelInit[:, 0]
        yPixInit = xyPixelInit[:, 1]

        return xPixInit, yPixInit

    def check_sensor_type(self):
        if self.sensor in C.GEOCOSICORR3D_SENSORS_LIST:
            logging.info(f'Satellite sensor:{self.sensor}')
        else:
            msg = f'Satellite sensor:{self.sensor} not supported by {C.SOFTWARE.SOFTWARE_NAME}_v{C.SOFTWARE.VERSION}'
            logging.error(msg)
            sys.exit(msg)
        pass


class RFMOrtho(RawInverseOrtho):
    def __init__(self,
                 input_l1a_path: str,
                 output_ortho_path: str,
                 ortho_params: Optional[Dict] = None,
                 output_trans_path: Optional[str] = None,
                 dem_path: Optional[str] = None,
                 debug: bool = False):
        if ortho_params is None:
            ortho_params = {}
        super().__init__(input_l1a_path, output_ortho_path, output_trans_path, ortho_params, dem_path, debug)
        self.orthorectify()
        return

    def orthorectify(self):

        self.model = RFM(self.metadata, debug=self.debug)
        self.mean_h = self.model.altOff
        self._get_correction_model()

        self.ortho_grid = self._set_ortho_grid()
        if self.debug:
            self.ortho_grid.grid_fp(o_folder=None)  # TODO set wd_folder
            logging.info(f'{self.__class__.__name__}:{repr(self.ortho_grid)}')
        self._check_dem_prj()
        self.ortho_geo_transform = self.set_ortho_geo_transform()
        logging.info(f'{self.__class__.__name__}:ortho_geo_transform:{self.ortho_geo_transform}')

        need_loop = True
        index = 1
        yOff = 0
        self.compute_num_tiles()
        if self.debug:
            logging.info(
                f'{self.__class__.__name__}:Raster: W:{self.o_raster_w}, H:{self.o_raster_h}, #tiles:{self.n_tiles}')

        if self.n_tiles > 1:
            self.write_ortho_per_tile()

        dem_dims, easting, northing, nbTiles, tiles = self.ortho_tiling(
            nbRowsPerTile=self.nb_rows_per_tile,
            nbRowsOut=self.o_raster_h,
            nbColsOut=self.o_raster_w,
            need_loop=need_loop,
            oUpLeftEW=self.ortho_grid.o_up_left_ew,
            oUpLeftNS=self.ortho_grid.o_up_left_ns,
            xRes=self.ortho_grid.o_res,
            yRes=self.ortho_grid.o_res,
            demInfo=self.dem_raster_info,
        )

        ortho_data = {"easting": easting,
                      "northing": northing,
                      "current_tile": 0,
                      "xPixInit": None,
                      "yPixInit": None,
                      "dem_dims": dem_dims,
                      "nbTiles": nbTiles,
                      "tiles": tiles,
                      }

        while need_loop is True:
            if self.debug:
                logging.info(f'========= Tile:{index} /{self.n_tiles} =============')

            matTile, need_loop, ortho_data = self.compute_transformation_matrix(ortho_data=ortho_data)

            if self.debug:
                logging.info(f'{self.__class__.__name__}:mat_tile.shape:{matTile.shape}')
                logging.info(f'{self.__class__.__name__}:Resampling::{self.resampling_method}')
            resample = Resampling(input_raster_info=self.l1a_raster_info, transformation_mat=matTile,
                                  resampling_params={'method': self.resampling_method})
            oOrthoTile = resample.resample()
            if self.debug:
                logging.info(f'{self.__class__.__name__}: rss = {process.memory_info().rss * 1e-6} Mb')
            yOff = self.write_ortho_rasters(oOrthoTile=oOrthoTile, matTile=matTile, yOff=yOff)
            index = index + 1

        if self.n_tiles > 1:
            self.oOrthoRaster = None
            self.oTranRaster = None

        return

    def _get_correction_model(self):
        if self.corr_model_file is None:
            cf = currentframe()
            # TODO: change to no RFM correction model
            geoWarns.wrNoRSMCorrection(line=str(cf.f_back.f_lineno))
            self.corr_model = np.zeros((3, 3))
        else:
            try:
                self.corr_model = np.loadtxt(self.corr_model_file)
            except:
                geoErrors.erReadCorrectionFile()
        return

    def compute_transformation_matrix(self, ortho_data):
        """
        Core process of computing the orthorectification matrices of satellite images.
        Inputs are in-memory whereas output can be written onto a file or returned in memory
        """
        current_tile = ortho_data['current_tile']
        easting = ortho_data['easting']
        northing = ortho_data['northing']
        tiles = ortho_data['tiles']
        if self.debug:
            logging.info(f'{self.__class__.__name__}:Current Tile:{current_tile + 1}')
        ## We assume that eastArry and northArray are in the same coordinate system as the DEM if exist.
        ## No conversion or proj system is needed
        nbRowsOut = tiles[current_tile + 1] - tiles[current_tile]
        eastArr = np.tile(easting, (nbRowsOut, 1))

        tempNorthing = [northing[tiles[current_tile] + i] for i in range(nbRowsOut)]
        northArr = np.tile(tempNorthing, (self.o_raster_w, 1)).T

        hNew = self.elev_interpolation(ortho_data['dem_dims'], current_tile, eastArr, northArr)

        eastArr_flat = eastArr.flatten()
        northArr_flat = northArr.flatten()
        hNew_flatten = list(hNew.flatten())
        if self.debug:
            logging.info(">>> converting UTM --> WGS84")
        (lat_flat, lon_flat, alt_flat) = Converter.coord_map1_2_map2(X=eastArr_flat,
                                                                     Y=northArr_flat,
                                                                     Z=hNew_flatten,
                                                                     sourceEPSG=self.ortho_grid.grid_epsg,
                                                                     targetEPSG=4326)
        if self.debug:
            logging.info(">>> RFM Ground 2 Pix >>> ")
        x_pix, y_pix = self.model.Ground2Img_RFM(lon=lon_flat,
                                                 lat=lat_flat,
                                                 alt=alt_flat,
                                                 corrModel=self.corr_model)

        outX = np.reshape(x_pix, eastArr.shape)
        outY = np.reshape(y_pix, northArr.shape)

        if current_tile != self.n_tiles - 1:
            need_loop = True

            ortho_data["current_tile"] = current_tile + 1
            ortho_data["xPixInit"] = outX[nbRowsOut - 1, :]
            ortho_data["yPixInit"] = outY[nbRowsOut - 1, :]
        else:
            need_loop = False
        oArray = np.zeros((2, outX.shape[0], outX.shape[1]))
        oArray[0, :, :] = outX
        oArray[1, :, :] = outY

        return oArray, need_loop, ortho_data


def orthorectify(input_l1a_path: str,
                 output_ortho_path: str,
                 ortho_params: Optional[Dict] = None,
                 output_trans_path: Optional[str] = None,
                 dem_path: Optional[str] = None,
                 debug: bool = False):
    method_type = ortho_params['method']['method_type']
    if method_type == C.SATELLITE_MODELS.RFM:
        RFMOrtho(input_l1a_path=input_l1a_path,
                 output_ortho_path=output_ortho_path,
                 dem_path=dem_path,
                 ortho_params=ortho_params,
                 output_trans_path=output_trans_path,
                 debug=debug)
    if method_type == C.SATELLITE_MODELS.RSM:
        RSMOrtho(input_l1a_path=input_l1a_path,
                 output_ortho_path=output_ortho_path,
                 output_trans_path=output_trans_path,
                 dem_path=dem_path,
                 ortho_params=ortho_params,
                 debug=debug)

    return
