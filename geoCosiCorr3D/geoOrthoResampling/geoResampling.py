"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import warnings
from typing import Dict, Optional

import numpy as np
import psutil

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.georoutines.geo_utils as geoRT

from geoCosiCorr3D.geoCore.core_resampling import (BilinearResampler,
                                                   RawResampling,
                                                   ResamplingEngine,
                                                   SincResampler)
import geoCosiCorr3D.utils.misc as misc


class Resampling(RawResampling):
    _cache = {}

    def __init__(self, input_raster_info: geoRT.cRasterInfo, transformation_mat: np.ndarray,
                 resampling_params: Optional[Dict] = None, debug: bool = False, tile_num: int = 0):
        misc.log_available_memory(self.__class__.__name__)
        self.tile_num = tile_num
        super().__init__(input_raster_info, transformation_mat, resampling_params, debug)

    def resample(self):

        if self.raster_info.band_number > 1:
            logging.warning(
                f'Multi-band orthorectification is not supported in {C.SOFTWARE.SOFTWARE_NAME}_v_{C.SOFTWARE.VERSION}')
        band_num = 1

        matrix_x = self.trans_matx[0, :, :]  # rows
        matrix_y = self.trans_matx[1, :, :]  # cols
        sz = matrix_x.shape

        raw_l1a_subset_img_extent = self.compute_l1A_img_tile_subset(self.raster_info, matrix_x, matrix_y)
        logging.info(f'L1A subset extent to orthorectify :{raw_l1a_subset_img_extent}')
        dims = list(raw_l1a_subset_img_extent.values())

        # Create a tuple of dimensions to use as a key for the cache
        dims_key = (dims[0], dims[1], dims[2], dims[3], band_num)

        if np.all(dims == 0):
            logging.error('ERROR: enable to orthorectify the subset ')
            return np.zeros(sz) * np.nan

        available_memory_gb = misc.log_available_memory(self.__class__.__name__)

        width = dims[1] - dims[0] + 1
        height = dims[3] - dims[2] + 1
        total_pixels = width * height
        data_type_size_bytes = 4  # Assuming float32
        number_of_bands = 1
        memory_required_bytes = total_pixels * data_type_size_bytes * number_of_bands
        memory_required_gb = memory_required_bytes / (1024 ** 3)
        logging.info(
            f'{self.__class__.__name__}:Ortho memory required [Gb]: {memory_required_gb:.2f} for {width}x{height}')
        if memory_required_gb > available_memory_gb:
            raise MemoryError(
                f"Insufficient memory: {memory_required_gb:.2f} required, "
                f"but only {available_memory_gb / (1024 ** 3):.2f} available.")

        # FIXME: there is something wrong here why I am loading the full image instead of loading the subset,
        #  dims points always to the full extent for the first 2 tiles
        l1a_img = self.raster_info.image_as_array_subset(*dims_key)

        # print('---------- Debug: Plotting ----------')
        # import tifffile
        # l1a_img_16bit = l1a_img.astype(np.uint16)
        # tifffile.imwrite(f'tile_{self.tile_num}.tif', l1a_img_16bit)
        # print('---------- Debug: Plotting ----------')

        if l1a_img.dtype != np.float32 or l1a_img.dtype != np.float_:
            if self.method == C.Resampling_Methods.SINC:
                if check_memory_before_conversion(l1a_img, size_float=8):
                    l1a_img = l1a_img.astype(np.float_, copy=False)
                else:
                    raise MemoryError("Insufficient memory for conversion to float 64, try with bilinear resampling")
            else:
                if check_memory_before_conversion(l1a_img, size_float=4):
                    l1a_img = l1a_img.astype(np.float32, copy=False)
                else:
                    raise MemoryError("Insufficient memory for conversion to float 32")

        ## Correct the matrices coordinates for the subsetting of the extracted image
        matrix_x -= dims[0]
        matrix_y -= dims[2]

        if self.method == C.Resampling_Methods.SINC:
            return SincResampler.f_sinc_resampler(matrix_x, matrix_y, l1a_img, self.resampling_cfg.kernel_sz)
        elif self.method == C.Resampling_Methods.BILINEAR:
            return BilinearResampler.resampling(matrix_x, matrix_y, l1a_img)
        else:
            raise ValueError(f"Resampling method {self.method} not supported")


def check_memory_before_conversion(array, size_float=4):
    """
    size_float32 = 4
    size_float64 = 8

    """
    expected_size_bytes = array.size * size_float
    available_memory = psutil.virtual_memory().available
    if expected_size_bytes > available_memory:
        print("Insufficient memory for conversion.")
        return False
    else:
        return True


if __name__ == '__main__':
    from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog

    log = geoCosiCorr3DLog("Test_Resampling")
    engine1 = ResamplingEngine(debug=True)
    logging.info('====================================================')
    engine2 = ResamplingEngine(resampling_params={'method': 'sinc', 'kernel_sz': 3}, debug=True)
    logging.info('====================================================')
    engine3 = ResamplingEngine(resampling_params={'method': 'sinc', 'kernel_sz': 4}, debug=True)
    logging.info('====================================================')
    engine4 = ResamplingEngine(resampling_params={'method': 'bilinea'}, debug=True)
