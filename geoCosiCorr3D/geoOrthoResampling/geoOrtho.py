"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import psutil
import ctypes, ctypes.util, goto, sys
import warnings

from inspect import currentframe
from goto import with_goto
from dominate.tags import label
from typing import Optional

import geoCosiCorr3D.georoutines.geo_utils as geoRT
import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarns
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors

from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import EstimateGeoTransformation
from geoCosiCorr3D.geoRFM.RFM import RFM
from geoCosiCorr3D.geoCore.core_RSM import RSM
from geoCosiCorr3D.geoOrthoResampling.geoResampling import Resampling
from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import cGetSatMapGrid
from geoCosiCorr3D.geoCore.geoRawInvOrtho import RawInverseOrtho
from geoCosiCorr3D.geoCore.constants import *

geoWarns.wrIgnoreNotGeoreferencedWarning()
process = psutil.Process(os.getpid())
G2P_LIB_Path = SOFTWARE.GEO_COSI_CORR3D_LIB


# TODO CHANGE RSM class location


class RSMOrtho(RawInverseOrtho):

    def __init__(self,
                 input_l1a_path: str,
                 output_ortho_path: str,
                 ortho_params: Optional[Dict] = None,
                 output_trans_path: Optional[str] = None,
                 dem_path: Optional[str] = None,
                 debug: bool = True):
        self.hMean = None
        if ortho_params is None:
            ortho_params = {}
        super().__init__(input_l1a_path, output_ortho_path, output_trans_path, ortho_params, dem_path, debug)
        self.check_sensor_type()
        self.orthorectify()
        return

    def orthorectify(self):
        self.rsm_model = self._get_rsm_model()
        self._get_correction_model()
        self.ortho_grid = self._set_ortho_grid()
        self._check_dem_prj()
        self.ortho_geo_transform = self.set_ortho_geo_transform()
        if self.debug:
            logging.info("ortho_geo_transform:{}".format(self.ortho_geo_transform))
        need_loop = 1
        index = 1
        init_data = None
        yOff = 0

        self.compute_tiles()
        if self.debug:
            logging.info("oRasterW:{}, oRasterH:{}, nbTiles:{}".format(self.oRasterW, self.oRasterH, self.nbTiles))

        if self.nbTiles > 1:
            self.write_ortho_per_tile()

        while need_loop != 0:
            if self.debug:
                if self.debug:
                    logging.info("========= Tile:{} /{} =============".format(index, self.nbTiles))
            matTile, need_loop, init_data, nbTiles = self.compute_transformation_matrix(need_loop=need_loop,
                                                                                        init_data=init_data)
            if self.debug:
                logging.info('matTile.shape:{}'.format(matTile.shape))
                logging.info("... Resampling::{} ...".format(self.resampling_method))

            resample = Resampling(input_raster_info=self.l1a_raster_info, transformation_mat=matTile,
                                  resampling_params={'method': self.resampling_method})
            oOrthoTile = resample.resample()

            if self.debug:
                logging.info("---> rss = {} Mb".format(process.memory_info().rss * 1e-6))
                # plt.imshow(oOrthoTile, origin="lower", cmap="gray")
                # plt.show()
            # sys.exit()
            yOff = self.write_ortho_rasters(oOrthoTile=oOrthoTile, matTile=matTile, yOff=yOff)
            index = index + 1

        if self.nbTiles > 1:
            self.oOrthoRaster = None
            self.oTranRaster = None
        return

    def _set_ortho_grid(self) -> cGetSatMapGrid:
        ortho_grid = cGetSatMapGrid(rasterInfo=self.l1a_raster_info,
                                    modelData=self.rsm_model,
                                    modelType=self.ortho_type,
                                    rDEM=self.dem_path,
                                    newRes=self.o_res,
                                    modelCorr=self.corr_model,
                                    debug=self.debug)
        if self.debug:
            logging.info(repr(ortho_grid))
        return ortho_grid

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

    @with_goto
    def compute_transformation_matrix(self, need_loop, init_data):
        """
        Core process of computing the orthorectification matrices of satellite images.
        Inputs are in-memory whereas output can be  written onto a file or returned in memory
        Args:
            need_loop
            init_data

        Returns:

        """

        if need_loop != 0:
            # print("----- Potential GOTO ----")
            if need_loop == 1 and init_data is not None:
                ## ++> the first loop and step --> we need initialize
                # print("----- GOTO, inTiling ----")
                goto.inTiling
        else:
            need_loop = 0
        ## From the oGrid, define the matrix size
        demDims, easting, northing, nbTiles, tiles = self.ortho_tiling(nbRowsPerTile=self.nbRowsPerTile,
                                                                       nbRowsOut=self.oRasterH,
                                                                       nbColsOut=self.oRasterW,
                                                                       need_loop=need_loop,
                                                                       oUpLeftEW=self.ortho_grid.oUpLeftEW,
                                                                       oUpLeftNS=self.ortho_grid.oUpLeftNS,
                                                                       xRes=self.ortho_grid.oRes,
                                                                       yRes=self.ortho_grid.oRes,
                                                                       demInfo=self.dem_raster_info, )

        xPixInit, yPixInit = self.compute_initial_approx(modelData=self.rsm_model,
                                                         oGrid=self.ortho_grid,
                                                         easting=easting,
                                                         northing=northing,
                                                         oRasterW=self.oRasterW)

        if self.debug:
            logging.info(f'initial_approx xPixIinit:{xPixInit} {len(xPixInit)} ')
            logging.info(f'initial_approx yPixInit: {yPixInit} {len(yPixInit)}')

        init_data = {"nbColsOut": self.oRasterW,
                     "easting": easting,
                     "northing": northing,
                     "xPixInit": xPixInit,
                     "yPixInit": yPixInit,
                     "demInfo": self.dem_raster_info,
                     "demDims": demDims,
                     "ancillary": self.rsm_model,
                     "nbTiles": nbTiles,
                     "tiles": tiles,
                     "tileCurrent": 0}

        label.inTiling

        nbColsOut = init_data["nbColsOut"]
        easting = init_data["easting"]
        northing = init_data["northing"]
        xPixInit = init_data["xPixInit"]
        yPixInit = init_data["yPixInit"]
        demInfo = init_data["demInfo"]
        demDims = init_data["demDims"]
        nbTiles = init_data["nbTiles"]
        tiles = init_data["tiles"]
        tileCurrent = init_data["tileCurrent"]
        if self.debug:
            logging.info("Current Tile:{}".format(tileCurrent + 1))
        ## We assume that eastArry and northArray are in the same coordinate system as the DEM if exist.
        ## No conversion or proj system is needed
        nbRowsOut = tiles[tileCurrent + 1] - tiles[tileCurrent]
        eastArr = np.tile(easting, (nbRowsOut, 1))

        tempNorthing = [northing[tiles[tileCurrent] + i] for i in range(nbRowsOut)]
        northArr = np.tile(tempNorthing, (nbColsOut, 1)).T
        hNew = self.DEM_interpolation(demInfo, demDims, tileCurrent, eastArr, northArr, self.rsm_model)
        outX, outY, nbRowsOut = self.rsm_g2p_minimization(rsmModel=self.rsm_model,
                                                          rsmCorrectionArray=self.corr_model,
                                                          nbColsOut=nbColsOut,
                                                          xPixInit=xPixInit,
                                                          yPixInit=yPixInit,
                                                          eastArr=eastArr,
                                                          northArr=northArr,
                                                          nbRowsOut=nbRowsOut,
                                                          hNew=hNew,
                                                          debug=self.debug,
                                                          target_epsg=self.ortho_grid.gridEPSG)
        if tileCurrent != nbTiles - 1:
            need_loop = 1

            init_data["tileCurrent"] = tileCurrent + 1
            init_data["xPixInit"] = outX[nbRowsOut - 1, :]
            init_data["yPixInit"] = outY[nbRowsOut - 1, :]
        else:
            need_loop = 0
            init_data = None
        oArray = np.zeros((2, outX.shape[0], outX.shape[1]))
        oArray[0, :, :] = outX
        oArray[1, :, :] = outY
        return oArray, need_loop, init_data, nbTiles

    def DEM_interpolation(self, demInfo, demDims, tileCurrent, eastArr, northArr, modelData):
        from geoCosiCorr3D.geoRSM.misc import HeightInterpolation
        if self.debug:
            logging.info("  H (rDEM) interpolation ...")
        h_new = HeightInterpolation.DEM_interpolation(demInfo=demInfo, demDims=demDims, tileCurrent=tileCurrent,
                                                      eastArr=eastArr, northArr=northArr)
        if h_new is None:
            if self.hMean is not None:
                h_new = np.ones(eastArr.shape) * self.hMean
                warnings.warn("H will be set to hMean:{}".format(self.hMean))
                logging.warning(f'H will be set to the provided hMean {self.hMean}')
            else:
                h_new = np.zeros(eastArr.shape)
                logging.warning('H will be set to 0')
                warnings.warn("H will be set to 0")

        return h_new

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

        libPath_ = ctypes.util.find_library(G2P_LIB_Path)
        if not libPath_:
            msg = "Unable to find the specified library:" + G2P_LIB_Path
            sys.exit(msg)
        try:
            fLib = ctypes.CDLL(libPath_)
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
        semiMajor_f = ctypes.c_double(EARTH.SEMIMAJOR)
        semiMinor_f = ctypes.c_double(EARTH.SEMIMINOR)
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
    def compute_initial_approx(modelData, oGrid, easting, northing, oRasterW):
        """
        Define an array of corresponding pixel coordinates 1A <--> Ground coordinates UTM
        Compute the 2D-affine transformation <==> geotrans for this task we don't need the h component


        """

        xBBox = [0, modelData.nbCols - 1, 0, modelData.nbCols - 1]
        yBBox = [0, 0, modelData.nbRows - 1, modelData.nbRows - 1]

        topLeftGround = [oGrid.upLeftEW, oGrid.upLeftNS]
        topRightGround = [oGrid.upRightEW, oGrid.upRightNS]
        bottomLeftGround = [oGrid.botLeftEW, oGrid.botLeftNS]
        bottomRightGround = [oGrid.botRightEW, oGrid.botRightNS]

        pixObs = np.array([xBBox, yBBox]).T
        groundObs = np.array([topLeftGround, topRightGround, bottomLeftGround, bottomRightGround])
        geoAffineTrans = EstimateGeoTransformation(pixObs=pixObs, groundObs=groundObs)

        eastingNorthingMatrix = np.array([easting, oRasterW * [northing[0]], oRasterW * [1]])
        xyPixelInit = np.dot(eastingNorthingMatrix.T, geoAffineTrans.T)
        xPixInit = xyPixelInit[:, 0]
        yPixInit = xyPixelInit[:, 1]

        # eastingMat = np.tile(easting,(self.oRasterH,1))
        # northingMat = np.tile(northing, (self.oRasterW, 1)).T

        return xPixInit, yPixInit

    def check_sensor_type(self):
        if self.sensor in GEOCOSICORR3D_SENSORS_LIST:
            logging.info(f'Satellite sensor:{self.sensor}')
        else:
            msg = f'Satellite sensor:{self.sensor} not supported by {SOFTWARE.SOFTWARE_NAME}_v{SOFTWARE.VERSION}'
            logging.error(msg)
            sys.exit(msg)
        pass


class RFMOrtho(RawInverseOrtho):
    def __init__(self,
                 input_l1a_path: str,
                 output_ortho_path: str,
                 ortho_params: Optional[Dict] = None,
                 output_trans_path: Optional[str] = None,
                 dem_path: Optional[str] = None):
        if ortho_params is None:
            ortho_params = {}
        super().__init__(input_l1a_path, output_ortho_path, output_trans_path, ortho_params, dem_path)
        self.orthorectify()
        return

    def orthorectify(self):

        self.rfm_model = RFM(self.metadata, debug=True)

        self._get_correction_model()

        self.ortho_grid = self._set_ortho_grid()
        self._check_dem_prj()
        self.ortho_geo_transform = self.set_ortho_geo_transform()
        logging.info("ortho_geo_transform:{}".format(self.ortho_geo_transform))

        need_loop = 1
        index = 1
        init_data = None
        yOff = 0
        self.compute_tiles()
        if self.debug:
            logging.info("oRasterW:{}, oRasterH:{}, nbTiles:{}".format(self.oRasterW, self.oRasterH, self.nbTiles))

        if self.nbTiles > 1:
            self.write_ortho_per_tile()

        while need_loop != 0:
            if self.debug:
                logging.info("========= Tile:{} /{} =============".format(index, self.nbTiles))
            matTile, need_loop, init_data, nbTiles = self.compute_transformation_matrix(need_loop=need_loop,
                                                                                        init_data=init_data)

            if self.debug:
                logging.info('matTile.shape:{}'.format(matTile.shape))
            if self.debug:
                logging.info("... Resampling::{} ...".format(self.resampling_method))
            resample = Resampling(input_raster_info=self.l1a_raster_info, transformation_mat=matTile,
                                  resampling_params={'method': self.resampling_method})
            oOrthoTile = resample.resample()
            if self.debug:
                logging.info("---> rss = {} Mb".format(process.memory_info().rss * 1e-6))
                # plt.imshow(oOrthoTile, origin="lower", cmap="gray")
                # plt.show()
            # sys.exit()
            yOff = self.write_ortho_rasters(oOrthoTile=oOrthoTile, matTile=matTile, yOff=yOff)
            index = index + 1

        if self.nbTiles > 1:
            self.oOrthoRaster = None
            self.oTranRaster = None

        return

    def _set_ortho_grid(self):
        ortho_grid = cGetSatMapGrid(rasterInfo=self.l1a_raster_info,
                                    modelData=self.rfm_model,
                                    modelType=self.ortho_type,
                                    rDEM=self.dem_path,
                                    newRes=self.o_res,
                                    modelCorr=self.corr_model,
                                    debug=self.debug)
        if self.debug:
            logging.info(repr(ortho_grid))
        return ortho_grid

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

    @with_goto
    def compute_transformation_matrix(self, need_loop, init_data):
        """
        Core process of computing the orthorectification matrices of satellite images.
        Inputs are in-memory whereas output can be  written onto a file or returned in memory
        Returns:

        """

        if need_loop != 0:
            # print("----- Potential GOTO ----")
            if need_loop == 1 and init_data is not None:
                ## ++> the first loop and step --> we need initialize
                # print("----- GOTO, inTiling ----")
                goto.inTiling
        else:
            need_loop = 0
        ## From the oGrid, define the matrix size
        demDims, easting, northing, nbTiles, tiles = self.ortho_tiling(nbRowsPerTile=self.nbRowsPerTile,
                                                                       nbRowsOut=self.oRasterH,
                                                                       nbColsOut=self.oRasterW,
                                                                       need_loop=need_loop,
                                                                       oUpLeftEW=self.ortho_grid.oUpLeftEW,
                                                                       oUpLeftNS=self.ortho_grid.oUpLeftNS,
                                                                       xRes=self.ortho_grid.oRes,
                                                                       yRes=self.ortho_grid.oRes,
                                                                       demInfo=self.dem_raster_info, )

        xPixInit = None
        yPixInit = None

        init_data = {"nbColsOut": self.oRasterW,
                     "easting": easting,
                     "northing": northing,
                     "xPixInit": xPixInit,
                     "yPixInit": yPixInit,
                     "demInfo": self.dem_raster_info,
                     "demDims": demDims,
                     "ancillary": self.rfm_model,
                     "nbTiles": nbTiles,
                     "tiles": tiles,
                     "tileCurrent": 0}

        label.inTiling

        nbColsOut = init_data["nbColsOut"]
        easting = init_data["easting"]
        northing = init_data["northing"]
        xPixInit = init_data["xPixInit"]
        yPixInit = init_data["yPixInit"]
        demInfo = init_data["demInfo"]
        demDims = init_data["demDims"]
        nbTiles = init_data["nbTiles"]
        tiles = init_data["tiles"]
        tileCurrent = init_data["tileCurrent"]
        if self.debug:
            logging.info("Current Tile:{}".format(tileCurrent + 1))
        ## We assume that eastArry and northArray are in the same coordinate system as the DEM if exist.
        ## No conversion or proj system is needed
        nbRowsOut = tiles[tileCurrent + 1] - tiles[tileCurrent]
        eastArr = np.tile(easting, (nbRowsOut, 1))

        tempNorthing = [northing[tiles[tileCurrent] + i] for i in range(nbRowsOut)]
        northArr = np.tile(tempNorthing, (nbColsOut, 1)).T

        hNew = self.DEM_interpolation(demInfo, demDims, tileCurrent, eastArr, northArr, self.rfm_model)

        eastArr_flat = eastArr.flatten()
        northArr_flat = northArr.flatten()
        hNew_flatten = list(hNew.flatten())
        if self.debug:
            logging.info(">>> converting UTM --> WGS84")

        (lat_flat, lon_flat, alt_flat) = geoRT.ConvCoordMap1ToMap2_Batch(X=eastArr_flat,
                                                                         Y=northArr_flat,
                                                                         Z=hNew_flatten,
                                                                         sourceEPSG=self.ortho_grid.gridEPSG,
                                                                         targetEPSG=4326)
        if self.debug:
            logging.info(">>> RFM Ground 2 Pix >>> ")
        x_pix, y_pix = self.rfm_model.Ground2Img_RFM(lon=lon_flat,
                                                     lat=lat_flat,
                                                     alt=alt_flat,
                                                     corrModel=self.corr_model)

        outX = np.reshape(x_pix, eastArr.shape)
        outY = np.reshape(y_pix, northArr.shape)

        if tileCurrent != nbTiles - 1:
            need_loop = 1

            init_data["tileCurrent"] = tileCurrent + 1
            init_data["xPixInit"] = outX[nbRowsOut - 1, :]
            init_data["yPixInit"] = outY[nbRowsOut - 1, :]
        else:
            need_loop = 0
            init_data = None
        oArray = np.zeros((2, outX.shape[0], outX.shape[1]))
        oArray[0, :, :] = outX
        oArray[1, :, :] = outY
        return oArray, need_loop, init_data, nbTiles

    def DEM_interpolation(self, demInfo: geoRT.cRasterInfo, demDims, tileCurrent, eastArr, northArr, modelData):
        from geoCosiCorr3D.geoRSM.misc import HeightInterpolation
        if self.debug:
            logging.info("  H (rDEM) interpolation ...")

        h_new = HeightInterpolation.DEM_interpolation(demInfo=demInfo, demDims=demDims, tileCurrent=tileCurrent,
                                                      eastArr=eastArr, northArr=northArr)
        if h_new is None:
            h_new = np.ones(eastArr.shape) * modelData.altOff
            msg = "H will be set to alfOffset:{} m".format(modelData.altOff)
            warnings.warn(msg)

        return h_new
