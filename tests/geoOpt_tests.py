"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import tempfile
import unittest

import numpy as np
import pandas

from geoCosiCorr3D.geoCore.constants import (SATELLITE_MODELS, SENSOR,
                                             SOFTWARE, TEST_CONFIG)
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoOptimization.gcpOptimization import cGCPOptimization

folder = os.path.join(SOFTWARE.PARENT_FOLDER.parent, "tests/test_dataset")
gcp_file = os.path.join(folder, 'input_GCPs.csv')
raw_img_path = os.path.join(folder, 'SP2.TIF')
ref_img_path = os.path.join(folder, 'basemap.TIF')
dem_path = os.path.join(folder, 'DEM.TIF')
dmp_file = os.path.join(folder, 'SP-2.DIM')
expec_corr = np.loadtxt(os.path.join(folder, 'expec_rsm_corr.txt'))
expec_report = pandas.read_csv(os.path.join(folder, 'expec_gcp_opt_report.csv')).to_numpy()
sat_model_params = {"sat_model": SATELLITE_MODELS.RSM, "metadata": dmp_file, "sensor": SENSOR.SPOT2}


class TestGeoOpt(unittest.TestCase):

    def test_geoOpt(self):
        # Your setup code...

        with tempfile.TemporaryDirectory(dir=SOFTWARE.WKDIR, suffix='test_gcp_opt') as tmp_dir:
            geoCosiCorr3DLog("Test_GCP_OPTIMIZATION", log_dir=tmp_dir)
            opt = cGCPOptimization(gcp_file_path=gcp_file,
                                   raw_img_path=raw_img_path,
                                   ref_ortho_path=ref_img_path,
                                   sat_model_params=sat_model_params,
                                   dem_path=dem_path,
                                   opt_params=TEST_CONFIG.GCP_OPT_CONFIG,
                                   opt_gcp_file_path=tmp_dir,
                                   corr_config=TEST_CONFIG.FREQ_CORR_CONFIG,
                                   debug=False,
                                   svg_patches=False)
        opt_report = opt.opt_report_df.drop(columns=['GCP_ID']).to_numpy()

        # Replace the assert with a unittest-style assert
        self.assertTrue(np.allclose(expec_corr, opt.corr_model),
                        msg="Expected and actual correlation models do not match")


if __name__ == "__main__":
    unittest.main()
