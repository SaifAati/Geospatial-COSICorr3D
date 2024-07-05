"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os
import tempfile
import pytest

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoCosiCorr3dLogger import GeoCosiCorr3DLog
from geoCosiCorr3D.geoOptimization.gcpOptimization import cGCPOptimization

# folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
folder = '/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/OLD/'
gcp_file = os.path.join(folder, 'input_GCPs.csv')
raw_img_path = os.path.join(folder, 'SP2.TIF')
ref_img_path = os.path.join(folder, 'basemap.TIF')
dem_path = os.path.join(folder, 'DEM.TIF')
dmp_file = os.path.join(folder, 'SP-2.DIM')

params = C.SatModelParams(C.SATELLITE_MODELS.RSM, dmp_file, C.SENSOR.SPOT2)

# def test_geoOpt():
#     with tempfile.TemporaryDirectory(dir=C.SOFTWARE.WKDIR, suffix='test_gcp_opt') as tmp_dir:
#         geoCosiCorr3DLog("Test_GCP_OPTIMIZATION", log_dir=tmp_dir)
#         opt = cGCPOptimization(gcp_file_path=gcp_file,
#                                raw_img_path=raw_img_path,
#                                ref_ortho_path=ref_img_path,
#                                sat_model_params=params,
#                                dem_path=dem_path,
#                                opt_params=C.TEST_CONFIG.GCP_OPT_CONFIG,
#                                opt_gcp_file_path=tmp_dir,
#                                corr_config=C.TEST_CONFIG.FREQ_CORR_CONFIG,
#                                debug=False,
#                                svg_patches=False)
#         opt()

if __name__ == '__main__':
    GeoCosiCorr3DLog("GCP_OPTIMIZATION", log_dir=folder)

    opt = cGCPOptimization(gcp_file_path=gcp_file,
                           raw_img_path=raw_img_path,
                           ref_ortho_path=ref_img_path,
                           sat_model_params=params,
                           dem_path=dem_path,
                           opt_params=C.TEST_CONFIG.GCP_OPT_CONFIG,
                           opt_gcp_file_path='/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/OLD',
                           corr_config=C.TEST_CONFIG.FREQ_CORR_CONFIG,
                           debug=True,
                           svg_patches=True)
    opt()
