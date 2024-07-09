"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import ctypes
import ctypes.util
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
import geoCosiCorr3D.georoutines.geo_utils as geoRT
import geoCosiCorr3D.utils.misc as misc

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
        self.method = self.resampling_params.get('method', C.Resampling_Methods.SINC)

        if self.method in C.GEOCOSICORR3D_RESAMLING_METHODS:
            if self.method == C.Resampling_Methods.SINC:
                kernel_sz = self.resampling_params.get("kernel_sz", SincResamlingConfig.kernel_sz)
                if kernel_sz < 0 or kernel_sz % 2 == 0:
                    logging.warning(f'RESAMPLING::Kernel size must be positive odd number')
                    kernel_sz = 15  # TODO add to constant file
                SincResamlingConfig.kernel_sz = kernel_sz
                if self.debug:
                    logging.info(f'RESAMPLING:: Resampling method:{self.method}, kz:{SincResamlingConfig.kernel_sz}')
                self.resampling_cfg = SincResamlingConfig

            elif self.method == C.Resampling_Methods.BILINEAR:
                self.resampling_cfg = BilinearResamlingConfig
        else:
            # TODO write a warning and set as default SINC
            logging.warning(
                f'RESAMPLING::Resampling method <<{self.method}>> not recognized by geoCosiCorr3D --> set to <<{C.Resampling_Methods.SINC}>>')
            self.method = C.Resampling_Methods.SINC
            self.resampling_cfg = SincResamlingConfig


class RawResampling(ResamplingEngine):
    def __init__(self, input_raster_info: geoRT.cRasterInfo,
                 transformation_mat: np.ndarray,
                 resampling_params: Optional[Dict],
                 debug: bool = False):
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

    def compute_l1A_img_tile_subset(self, raster_info: geoRT.cRasterInfo, matrix_x, matrix_y, margin=15) -> Dict:
        """
        Define the necessary image subset Dimensions needed for the resampling,
        depending on the resample engine selected.
            matrix_x: X-pixel coordinates : array
            matrix_y: Y-pixel coordinates : array
            margin: margin added around the subset for interpolation purpose :int
        """
        if np.isnan(matrix_x).all() or np.isnan(matrix_y).all():
            logging.error("RESAMPLING::MATRIX_X or MATRIX_Y ALL NANs")
            raise ValueError("RESAMPLING::MATRIX_X or MATRIX_Y ALL NANs")

        minX = math.floor(np.nanmin(matrix_x))
        maxX = math.ceil(np.nanmax(matrix_x))
        minY = math.floor(np.nanmin(matrix_y))
        maxY = math.ceil(np.nanmax(matrix_y))

        raw_img_pix_extent: Dict = {'col_pix_min': 0,
                                    'col_pix_max': raster_info.raster_width - 1,
                                    'row_pix_min': 0,
                                    'row_pix_max': raster_info.raster_height}
        logging.info(f'{self.__class__.__name__}: Raw image extent:{raw_img_pix_extent}')

        # initialize as the full image raw image size
        tile_img_subset_pix_extent = raw_img_pix_extent.copy()

        # if (minX - margin) > raw_img_pix_extent['col_pix_min']:
        #     tile_img_subset_pix_extent['col_pix_min'] = minX - margin
        # if (maxX + margin) < raw_img_pix_extent['col_pix_max']:
        #     tile_img_subset_pix_extent['col_pix_max'] = maxX + margin

        if (minY - margin) > raw_img_pix_extent['row_pix_min']:
            tile_img_subset_pix_extent['row_pix_min'] = minY + margin

        if (maxY + margin) < raw_img_pix_extent['row_pix_max']:
            tile_img_subset_pix_extent['row_pix_max'] = maxY + margin

        if self.method == C.Resampling_Methods.SINC:
            borderX, borderY = SincResampler.compute_resampling_distance(matrix_x,
                                                                         matrix_y,
                                                                         matrix_x.shape,
                                                                         self.resampling_cfg.kernel_sz)
            if (minX - borderX) > raw_img_pix_extent['col_pix_min']:
                tile_img_subset_pix_extent['col_pix_min'] = minX - borderX
            if (maxX + borderX) < raw_img_pix_extent['col_pix_max']:
                tile_img_subset_pix_extent['col_pix_max'] = maxX + borderX
            if (minY - borderY) > raw_img_pix_extent['row_pix_min']:
                tile_img_subset_pix_extent['row_pix_min'] = minY - borderY
            if (maxY + borderY) < raw_img_pix_extent['row_pix_max']:
                tile_img_subset_pix_extent['row_pix_max'] = maxY + borderY

        ## Check for situation where the entire current matx tile is outside image boundaries
        ## In that case need to output a zero array, either on file or in the output array, and
        ## continue to the next tile
        if (tile_img_subset_pix_extent['col_pix_min'] > tile_img_subset_pix_extent['col_pix_max']) or (
                tile_img_subset_pix_extent['row_pix_min'] > tile_img_subset_pix_extent['row_pix_max']):
            warnings.warn(
                f"ERROR:Raw image subset is out of the boundary of the the input L1A img{tile_img_subset_pix_extent}"
                f"--> out of :{raw_img_pix_extent}")
            logging.error(
                f"ERROR:Raw image subset is out of the boundary of the the input L1A img{tile_img_subset_pix_extent}"
                f"--> out of :{raw_img_pix_extent}")
            return dict.fromkeys(tile_img_subset_pix_extent, 0)

        return tile_img_subset_pix_extent


class SincResampler:
    # Definition of the resampling kernel
    apodization = 1  # sinc kernel apodization (1 by default, with no option for user)

    @staticmethod
    def f_sinc_resampler(matrix_x, matrix_y, l1a_img, kernel_sz, weigthing=1):
        sz = matrix_x.shape
        misc.log_available_memory(f'f_sinc_resampler')
        try:
            sincLib = ctypes.CDLL(C.SOFTWARE.GEO_COSI_CORR3D_LIB)
            if matrix_x.dtype != np.float_:
                matrix_x = matrix_x.astype(np.float_, copy=False)
            if matrix_y.dtype != np.float_:
                matrix_y = matrix_y.astype(np.float_, copy=False)

            width = ctypes.c_int(kernel_sz)
            o_img = np.zeros(sz, dtype=float)
            weighting = ctypes.c_int(weigthing)
            n_col_mat = ctypes.c_int(sz[0])
            n_row_mat = ctypes.c_int(sz[1])

            n_col_img = ctypes.c_int(l1a_img.shape[0])  # dims[3] - dims[2] + 1)
            n_row_img = ctypes.c_int(l1a_img.shape[1])  # dims[1] - dims[0] + 1)

            sincLib.main_sinc_adp_resampling_(
                matrix_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                matrix_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                l1a_img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(width),
                o_img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(weighting),
                ctypes.byref(n_col_mat),
                ctypes.byref(n_row_mat),
                ctypes.byref(n_col_img),
                ctypes.byref(n_row_img))

            return o_img
        except OSError:
            geoErrors.erLibLoading(C.SOFTWARE.GEO_COSI_CORR3D_LIB)

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
        # 10= max variability of the resampling distance,
        # 1.15= oversampling for kernel edge response
        borderX = int(np.ceil(np.max(np.abs([dx1, dx2, dx3])) * resampling_kernel_sz * 10 * 1.15))

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

        return borderX, borderY

    @classmethod
    def resampling(cls, matrix_x, matrix_y, im1A, kernel_sz, weigthing=1):
        cls.f_sinc_resampler(matrix_x, matrix_y, im1A, kernel_sz, weigthing=weigthing)


class BilinearResampler:
    @classmethod
    def resampling(cls, matrix_x, matrix_y, im1A):
        sz = matrix_x.shape
        nbRows, nbCols = im1A.shape[0], im1A.shape[1]
        ## Note: since we have corrected matrix_x and matrix_y coordinates the interpolation is then done 0,nbRows and 0,nbCols
        f = RegularGridInterpolator(
            (np.arange(0, nbRows, 1), np.arange(0, nbCols, 1)),
            im1A,
            method="linear",
            bounds_error=False,
            fill_value=np.nan)
        imgL3b_fl = f(list(zip(matrix_y.flatten(), matrix_x.flatten())))
        imgL3b = np.reshape(imgL3b_fl, sz)

        return imgL3b
