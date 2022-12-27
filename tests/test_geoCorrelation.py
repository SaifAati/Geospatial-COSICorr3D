"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import pytest
import tempfile

from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate
from geoCosiCorr3D.geoCore.constants import *
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo

folder = os.path.join(SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
img1 = os.path.join(folder, "BASE_IMG.TIF")
img2 = os.path.join(folder, "TARGET_IMG.TIF")


# TODO:
# Correlation between patches

@pytest.mark.parametrize('test_corr_config,expec_corr_path', [(TEST_CONFIG.FREQ_CORR_CONFIG, 'expec_freq_corr.TIF'),
                                                              (TEST_CONFIG.SPA_CORR_CONFIG, 'expec_spa_corr.tif')])
def test_corr(test_corr_config: Dict, expec_corr_path: str):
    expec_corr = cRasterInfo(os.path.join(folder, expec_corr_path)).raster_array
    with tempfile.TemporaryDirectory(dir=SOFTWARE.WKDIR, suffix='test_corr') as tmp_dir:
        corr_obj = Correlate(base_image_path=img1,
                             target_image_path=img2,
                             output_corr_path=tmp_dir,
                             corr_config=test_corr_config,
                             corr_show=False)

    mask = ~(np.isnan(corr_obj.ew_output) | np.isnan(expec_corr[0], ))
    assert np.allclose(corr_obj.ew_output[mask], expec_corr[0][mask])
    mask = ~(np.isnan(corr_obj.ns_output) | np.isnan(expec_corr[1], ))
    assert np.allclose(corr_obj.ns_output[mask], expec_corr[1][mask])
    mask = ~(np.isnan(corr_obj.snr_output) | np.isnan(expec_corr[2], ))
    assert np.allclose(corr_obj.snr_output[mask], expec_corr[2][mask])
