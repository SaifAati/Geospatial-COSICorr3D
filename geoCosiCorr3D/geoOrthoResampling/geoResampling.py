"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import warnings
from typing import Optional

import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoCore.core_resampling import RawResampling, SincResampler, BilinearResampler, ResamplingEngine
from geoCosiCorr3D.geoCore.constants import *


class Resampling(RawResampling):
    def __init__(self, input_raster_info: geoRT.cRasterInfo, transformation_mat: np.ndarray,
                 resampling_params: Optional[Dict] = None, debug: bool = False):
        super().__init__(input_raster_info, transformation_mat, resampling_params, debug)

    def resample(self):
        if self.debug:
            logging.info(
                "RESAMPLING::____________________ Resampling::{} ____________________________".format(self.method))

        nbBands = self.raster_info.band_number
        ##Fixme: force band = 1
        if nbBands > 1:
            nbBands = 1
            msg = f"Multi-band image: This version does not support multi-band ortho-rectification, only band {nbBands} will be orthorectified "
            warnings.warn(msg)
            logging.warning(msg)

        # Definition of the matrices dimensions
        dims_geom = [0, self.trans_matx.shape[2] - 1, 0, self.trans_matx.shape[1] - 1]

        if nbBands == 1:
            oOrthoTile = np.zeros((dims_geom[3] - dims_geom[2] + 1, dims_geom[1] - dims_geom[0] + 1))
        else:
            # geoErrors.erNotImplemented(routineName="This version does not support multi-band ortho-rectification")
            logging.warning(
                f'Multi-band orthorectification is not supported in {SOFTWARE.SOFTWARE_NAME}_v_{SOFTWARE.VERSION}')

        matrix_x = self.trans_matx[0, :, :]
        matrix_y = self.trans_matx[1, :, :]
        sz = matrix_x.shape
        if self.debug:
            logging.info(f'sz:{sz}')

        raw_l1a_subset_img_extent = self.compute_l1A_img_tile_subset(self.raster_info, matrix_x, matrix_y)

        logging.info(f'L1A subset extent to orthorectify :{raw_l1a_subset_img_extent}')
        dims = list(raw_l1a_subset_img_extent.values())

        if np.all(dims == 0):
            logging.error('ERROR: enable to orthorectify the subset ')
            return np.zeros(sz) * np.nan
        im1A = self.raster_info.image_as_array_subset(dims[0],
                                                      dims[1],
                                                      dims[2],
                                                      dims[3], band_number=nbBands)

        ## Correct the matrices coordinates for the subsetting of the extracted image
        matrix_x = matrix_x - dims[0]
        matrix_y = matrix_y - dims[2]

        if self.method == Resampling_Methods.SINC:
            return SincResampler.f_sinc_resampler(matrix_x, matrix_y, im1A, self.resampling_cfg.kernel_sz)

        if self.method == Resampling_Methods.BILINEAR:
            return BilinearResampler.resampling(matrix_x, matrix_y, im1A)


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
