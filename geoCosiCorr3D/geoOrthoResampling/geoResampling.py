"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
from typing import Dict, Optional

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.georoutines.geo_utils as geoRT
import numpy as np
from geoCosiCorr3D.geoCore.core_resampling import (BilinearResampler,
                                                   RawResampling,
                                                   ResamplingEngine,
                                                   SincResampler)


class Resampling(RawResampling):
    def __init__(self, input_raster_info: geoRT.cRasterInfo, transformation_mat: np.ndarray,
                 resampling_params: Optional[Dict] = None, debug: bool = False):
        super().__init__(input_raster_info, transformation_mat, resampling_params, debug)

    def resample(self):
        if self.debug:
            logging.info(
                f"{self.__class__.__name__}:RESAMPLING::____________________ Resampling::{self.method} _______________")

        nb_bands = self.raster_info.band_number
        if nb_bands > 1:
            nb_bands = 1
            logging.warning(
                f'Multi-band orthorectification is not supported in {C.SOFTWARE.SOFTWARE_NAME}_v_{C.SOFTWARE.VERSION}')

        matrix_x = self.trans_matx[0, :, :]
        matrix_y = self.trans_matx[1, :, :]
        sz = matrix_x.shape
        if self.debug:
            logging.info(f'sz:{sz}')

        self.raw_l1a_subset_img_extent = self.compute_l1A_img_tile_subset(self.raster_info, matrix_x, matrix_y)

        if self.debug:
            logging.info(
                f'{self.__class__.__name__}:L1A subset extent to orthorectify :{self.raw_l1a_subset_img_extent}')
        dims = list(self.raw_l1a_subset_img_extent.values())

        if np.all(dims == 0):
            logging.error('ERROR: enable to orthorectify the subset ')
            return np.zeros(sz) * np.nan
        im1A = self.raster_info.image_as_array_subset(dims[0],
                                                      dims[1],
                                                      dims[2],
                                                      dims[3], band_number=nb_bands)

        ## Correct the matrices coordinates for the subsetting of the extracted image
        matrix_x = matrix_x - dims[0]
        matrix_y = matrix_y - dims[2]

        if self.method == C.Resampling_Methods.SINC:
            return SincResampler.f_sinc_resampler(matrix_x, matrix_y, im1A, self.resampling_cfg.kernel_sz)

        if self.method == C.Resampling_Methods.BILINEAR:
            return BilinearResampler.resampling(matrix_x, matrix_y, im1A)


if __name__ == '__main__':
    from geoCosiCorr3D.geoCosiCorr3dLogger import GeoCosiCorr3DLog

    log = GeoCosiCorr3DLog("Test_Resampling")
    engine1 = ResamplingEngine(debug=True)
    logging.info('====================================================')
    engine2 = ResamplingEngine(resampling_params={'method': 'sinc', 'kernel_sz': 3}, debug=True)
    logging.info('====================================================')
    engine3 = ResamplingEngine(resampling_params={'method': 'sinc', 'kernel_sz': 4}, debug=True)
    logging.info('====================================================')
    engine4 = ResamplingEngine(resampling_params={'method': 'bilinea'}, debug=True)
