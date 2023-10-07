"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os
import tempfile
import unittest
from typing import Dict

import numpy as np

from geoCosiCorr3D.geoCore.constants import SOFTWARE, TEST_CONFIG
from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo


class TestCorrelation(unittest.TestCase):
    test_folder = None

    @classmethod
    def setUpClass(cls):
        cls.test_folder = os.path.join(SOFTWARE.PARENT_FOLDER.parent, "tests/test_dataset")
        cls.img1 = os.path.join(cls.test_folder, "BASE_IMG.TIF")
        cls.img2 = os.path.join(cls.test_folder, "TARGET_IMG.TIF")

    def corr_test_helper(self, test_corr_config: Dict, expec_corr_path: str):
        expec_corr = cRasterInfo(os.path.join(self.test_folder, expec_corr_path)).raster_array
        with tempfile.TemporaryDirectory(dir=SOFTWARE.WKDIR, suffix='test_corr') as tmp_dir:
            corr_obj = Correlate(base_image_path=self.img1,
                                 target_image_path=self.img2,
                                 output_corr_path=tmp_dir,
                                 corr_config=test_corr_config,
                                 corr_show=False)

        mask = ~(np.isnan(corr_obj.ew_output) | np.isnan(expec_corr[0], ))
        self.assertTrue(np.allclose(corr_obj.ew_output[mask], expec_corr[0][mask]))

        mask = ~(np.isnan(corr_obj.ns_output) | np.isnan(expec_corr[1], ))
        self.assertTrue(np.allclose(corr_obj.ns_output[mask], expec_corr[1][mask]))

        mask = ~(np.isnan(corr_obj.snr_output) | np.isnan(expec_corr[2], ))
        self.assertTrue(np.allclose(corr_obj.snr_output[mask], expec_corr[2][mask]))

    def test_freq_corr(self):
        self.corr_test_helper(TEST_CONFIG.FREQ_CORR_CONFIG, 'expec_freq_corr.TIF')

    def test_spa_corr(self):
        self.corr_test_helper(TEST_CONFIG.SPA_CORR_CONFIG, 'expec_spa_corr.tif')


if __name__ == '__main__':
    unittest.main()

#python -m unittest your_test_module_name.py
