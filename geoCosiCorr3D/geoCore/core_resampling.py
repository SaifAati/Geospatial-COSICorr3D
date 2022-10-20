"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import ctypes, ctypes.util
import math, warnings
import numpy as np
from typing import Optional, Dict, Any

import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
from geoCosiCorr3D.geoConfig import cgeoCfg
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from dataclasses import dataclass
from geoCosiCorr3D.geoCore.constants import GEOCOSICORR3D_RESAMLING_METHODS, Resampling_Methods

geoCfg = cgeoCfg()
SINC_KERNEL_SZ: int = 15


@dataclass
class SincResamlingConfig:
    # NOTE if kernelSz < 0 or kernelSz % 2 == 0
    # Kernel size must be positive odd number")
    kernel_sz: int = SINC_KERNEL_SZ


@dataclass
class BilinearResamlingConfig:
    pass


class ResamplingEngine:
    def __init__(self, resampling_params: Optional[Dict] = None, debug: Optional[bool] = False):
        self.debug = debug
        if resampling_params is None:
            self.resampling_params: Dict[Any] = {}
        else:
            self.resampling_params = resampling_params
        self.get_resampling_config()

    def get_resampling_config(self):
        self.method = self.resampling_params.get('method', Resampling_Methods.SINC)

        if self.method in GEOCOSICORR3D_RESAMLING_METHODS:
            if self.method == Resampling_Methods.SINC:
                kernel_sz = self.resampling_params.get("kernel_sz", SincResamlingConfig.kernel_sz)
                if kernel_sz < 0 or kernel_sz % 2 == 0:
                    logging.warning(f'RESAMPLING::Kernel size must be positive odd number')
                    kernel_sz = 15  # TODO add to constant file
                SincResamlingConfig.kernel_sz = kernel_sz
                if self.debug:
                    logging.info(f'RESAMPLING:: Resampling method:{self.method}, kz:{SincResamlingConfig.kernel_sz}')
                self.resampling_cfg = SincResamlingConfig

            elif self.method == Resampling_Methods.BILINEAR:
                self.resampling_cfg = BilinearResamlingConfig
        else:
            # TODO write a warning and set as default SINC
            logging.warning(
                f'RESAMPLING::Resampling method <<{self.method}>> not recognized by geoCosiCorr3D --> set to <<{Resampling_Methods.SINC}>>')
            self.method = Resampling_Methods.SINC
            self.resampling_cfg = SincResamlingConfig


class RawResampling(ResamplingEngine):
    def __init__(self, input_raster_info: geoRT.cRasterInfo, transformation_mat: np.ndarray,
                 resampling_params: Optional[Dict], debug: bool = False):
        """
        Resampling an image according to transformation matrices using the selected
        resampling kernel (e.g., sinc ,bilinear,bicubic,)
        Args:
            input_raster_info: object rasterInfo(geoRoutines) of the image to resample.
            transformation_mat: transformation matrix [2,nbCols,nbRows]

        Notes:
            TODO: handle multi-bands
        """
        super().__init__(resampling_params=resampling_params, debug=debug)
        self.raster_info = input_raster_info
        self.trans_matx = transformation_mat

    def compute_l1A_img_tile_subset(self, rasterInfo: geoRT.cRasterInfo, matrix_x, matrix_y, margin=3) -> Dict:
        """
        Define the necessary image subset Dimensions needed for the resampling,
        depending on the resample engine selected.
        Args:
            matrix_x: X-pixel coordinates : array
            matrix_y: Y-pixel coordinates : array
            margin: margin added around the subset for interpolation purpose :int

        Returns:

        """
        ### In order to define the necessary image subset to process the current matrix tile
        ### we need the matrix_x min and max,
        ### as well an estimate of the resampling distance in case of Sinc resampling engine.
        if np.isnan(matrix_x).all() or np.isnan(matrix_y).all():
            logging.error("RESAMPLING::MATRIX_X or MATRIX_Y ALL NANs")
            import sys
            sys.exit("MATRIX_X or MATRIX_Y ALL NANs")
        if self.debug:
            logging.info(
                f'Resampling:compute L1A subset img_extent--> X: {np.nanmin(matrix_x), np.nanmax(matrix_x)},'
                f' Y:{np.nanmin(matrix_y), np.nanmax(matrix_y)}')
        minX = math.floor(np.nanmin(matrix_x))
        maxX = math.ceil(np.nanmax(matrix_x))
        minY = math.floor(np.nanmin(matrix_y))
        maxY = math.ceil(np.nanmax(matrix_y))

        raw_img_pix_extent: Dict = {'col_pix_min': 0, 'col_pix_max': rasterInfo.raster_width - 1,
                                    'row_pix_min': 0, 'row_pix_max': rasterInfo.raster_height}
        ## Compute the necessary image subset dimension
        # initialize as the full image raw image size
        raw_img_subset_pix_extent = raw_img_pix_extent.copy()

        if (minX - margin) > raw_img_pix_extent['col_pix_min']:
            raw_img_subset_pix_extent['col_pix_min'] = minX - margin
        if (maxX + margin) < raw_img_pix_extent['col_pix_max']:
            raw_img_subset_pix_extent['col_pix_max'] = maxX + margin

        if (minY - margin) > raw_img_pix_extent['row_pix_min']:
            raw_img_subset_pix_extent['row_pix_min'] = minY + margin

        if (maxY + margin) < raw_img_pix_extent['row_pix_max']:
            raw_img_pix_extent['row_pix_max'] = maxY + margin

        if self.method == Resampling_Methods.SINC:
            borderX, borderY = SincResampler.compute_resampling_distance(matrix_x, matrix_y, matrix_x.shape,
                                                                         self.resampling_cfg.kernel_sz)
            if (minX - borderX) > raw_img_pix_extent['col_pix_min']:
                raw_img_subset_pix_extent['col_pix_min'] = minX - borderX
            if (maxX + borderX) < raw_img_pix_extent['col_pix_max']:
                raw_img_subset_pix_extent['col_pix_max'] = maxX + borderX
            if (minY - borderY) > raw_img_pix_extent['row_pix_min']:
                raw_img_subset_pix_extent['row_pix_min'] = minY - borderY
            if (maxY + borderY) < raw_img_pix_extent['row_pix_max']:
                raw_img_subset_pix_extent['row_pix_max'] = maxY + borderY

        if self.debug:
            logging.info(f'L1A Img subset img extent:{raw_img_subset_pix_extent}')
        # print("-- not sinc --- ")
        # print(dims1, dims2, dims3, dims4)
        ## Check for situation where the entire current matrice tile is outside image boundaries
        ## In that case need to output a zero array, either on file or in the output array, and
        ## continue to the next tile
        if (raw_img_subset_pix_extent['col_pix_min'] > raw_img_subset_pix_extent['col_pix_max']) or (
                raw_img_subset_pix_extent['row_pix_min'] > raw_img_subset_pix_extent['row_pix_max']):
            warnings.warn(
                f"ERROR:Raw image subset is out of the boundary of the the input L1A img{raw_img_subset_pix_extent}"
                f"--> out of :{raw_img_pix_extent}")
            logging.error(
                f"ERROR:Raw image subset is out of the boundary of the the input L1A img{raw_img_subset_pix_extent}"
                f"--> out of :{raw_img_pix_extent}")
            return dict.fromkeys(raw_img_subset_pix_extent, 0)

        if self.debug:
            logging.info(f'Raw subset img extent:dims:{raw_img_subset_pix_extent}')

        return raw_img_subset_pix_extent


class SincResampler:
    # Definition of the resampling kernel

    apodization = 1  # sinc kernel apodization (1 by default, with no option for user)

    @staticmethod
    def f_sinc_resampler(matrix_x, matrix_y, im1A, kernel_sz, weigthing=1):
        """

        Args:
            matrix_x:
            matrix_y:
            im1A:
            weigthing:

        Returns:

        """

        sz = matrix_x.shape

        libPath_ = ctypes.util.find_library(geoCfg.geoCosiCorr3DLib)

        if not libPath_:
            geoErrors.erLibNotFound(libPath=geoCfg.geoCosiCorr3DLib)
        try:
            sincLib = ctypes.CDLL(libPath_)
            matCol = np.array(matrix_y, dtype=np.float_)
            matRow = np.array(matrix_x, dtype=np.float_)

            # print("matrix_x[15,3]:{},matrix_y[15,3]:{}".format(matrix_x[15,3],matrix_y[15,3]))

            img = np.array(im1A, dtype=np.float_)
            width = ctypes.c_int(kernel_sz)
            oImg = np.zeros(sz, dtype=np.float)
            weighting = ctypes.c_int(weigthing)
            nbColMat = ctypes.c_int(sz[0])
            nbRowMat = ctypes.c_int(sz[1])

            nbColImg = ctypes.c_int(im1A.shape[0])  # dims[3] - dims[2] + 1)
            nbRowImg = ctypes.c_int(im1A.shape[1])  # dims[1] - dims[0] + 1)

            sincLib.main_sinc_adp_resampling_(
                matCol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                matRow.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(width),
                oImg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(weighting),
                ctypes.byref(nbColMat),
                ctypes.byref(nbRowMat),
                ctypes.byref(nbColImg),
                ctypes.byref(nbRowImg))

            return oImg
        except OSError:
            geoErrors.erLibLoading(geoCfg.geoCosiCorr3DLib)

    @staticmethod
    def compute_resampling_distance(matrix_x, matrix_y, sz, resampling_kernel_sz):

        ## Get the average resampling distance in X over the current tile in X, Y and diagonal directions.
        # Take the maximum
        dx1 = (matrix_x[0, 0] - matrix_x[0, sz[1] - 1]) / sz[1]
        dx2 = (matrix_x[0, 0] - matrix_x[sz[0] - 1, 0]) / sz[0]

        if sz[1] > sz[0]:
            dx3 = (matrix_x[0, 0] - matrix_x[sz[0] - 1, sz[0] - 1]) / sz[0]
        else:
            dx3 = (matrix_x[0, 0] - matrix_x[sz[1] - 1, sz[1] - 1]) / sz[1]
            ## Note:
            # 10 =max variability of the resampling distance,
            # 1.15= oversampling for kernel edge response
        borderX = int(np.ceil(np.max(np.abs([dx1, dx2, dx3])) * resampling_kernel_sz * 10 * 1.15))
        # print("dx1:{},dx2:{},dx3:{}".format(dx1, dx2,dx3))
        # print("borderX:", borderX)

        ## Get the average resampling distance in Y over the current tile in X, Y and diagonal directions.
        ## Take the maximum
        dy1 = (matrix_y[0, 0] - matrix_y[0, sz[1] - 1]) / sz[1]
        dy2 = (matrix_y[0, 0] - matrix_y[sz[0] - 1, 0]) / sz[0]

        if sz[1] > sz[0]:
            dy3 = (matrix_y[0, 0] - matrix_y[sz[0] - 1, sz[0] - 1]) / sz[0]
        else:
            dy3 = (matrix_y[0, 0] - matrix_y[sz[1] - 1, sz[1] - 1]) / sz[1]
            ## Note:
            # 10 =max variability of the resampling distance,
            # 1.15= oversampling for kernel edge response
        borderY = int(np.ceil(np.max(np.abs([dy1, dy2, dy3])) * resampling_kernel_sz * 10 * 1.15))
        # print("dy1:{},dy2:{},dy3:{}".format(dy1, dy2, dy3))
        # print("borderY:", borderY)
        return borderX, borderY

    @classmethod
    def resampling(cls, matrix_x, matrix_y, im1A, kernel_sz, weigthing=1):
        cls.f_sinc_resampler(matrix_x, matrix_y, im1A, kernel_sz, weigthing=weigthing)


class BilinearResampler:
    @classmethod
    def resampling(cls, matrix_x, matrix_y, im1A):
        # imgL3b_fl = interpRT.Interpolate2D(inArray=im1A, x=matrix_y.flatten(), y=matrix_x.flatten(), kind="linear")
        from scipy.interpolate import interpolate
        # print("Resampling ....")
        sz = matrix_x.shape
        nbRows, nbCols = im1A.shape[0], im1A.shape[1]
        ## Note: since we have corrected matrix_x nad matrix_y coordinates the interpolation is then done 0,nbRows and 0,nbCols
        f = interpolate.RegularGridInterpolator(
            # (np.arange(dims[2], dims[3] + 1, 1), np.arange(dims[0], dims[1] + 1, 1)),
            (np.arange(0, nbRows, 1), np.arange(0, nbCols, 1)),
            im1A,
            method="linear",
            bounds_error=False,
            fill_value=np.nan)
        imgL3b_fl = f(list(zip(matrix_y.flatten(), matrix_x.flatten())))
        imgL3b = np.reshape(imgL3b_fl, sz)
        # # print(imgL3b.shape)
        # plt.imshow(imgL3b, cmap="gray")
        # plt.show()
        # print("__________________________________ END Resampling _________________________________________")
        return imgL3b
