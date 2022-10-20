"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import geoCosiCorr3D
from geoCosiCorr3D.geoOptimization.gcpOptimization_v2 import cGCPOptimization
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoCore.constants import SOFTWARE

log = geoCosiCorr3DLog("Test_GCP_OPTIMIZATION")
folder = os.path.join(os.path.dirname(geoCosiCorr3D.__file__), 'Tests/3-geoGCP_optimization_Test/Sample2')

gcp_file = os.path.join(folder, 'rOrtho_VS_1999-07-10-09-07-26-Spot-2-HRV-2-P-10_GCP.csv')
raw_img_path = os.path.join(folder, '1999-07-10-09-07-26-Spot-2-HRV-2-P-10.TIF')
ref_img_path = os.path.join(folder, 'rOrtho.TIF')
dem_path = os.path.join(folder, 'rDEM.TIF')
dmp_file = os.path.join(folder, '1999-07-10-09-07-26-Spot-2-HRV-2-P-10.DIM')

sat_model_params = {"sat_model": "RSM", "metadata": dmp_file, "sensor": "Spot15"}

corr_config = {"correlator_name": "frequency",
               "correlator_params": {
                   "window_size": 4 * [64],
                   "step": [8, 8], "grid": True, "mask_th": 0.95, "nb_iters": 4}}
opt_params = {'nb_loops': 3, 'snr_th': 0.9, 'mean_error_th': 1 / 20, 'resampling_method': 'sinc'}

opt = cGCPOptimization(gcp_file_path=gcp_file,
                       raw_img_path=raw_img_path,
                       ref_ortho_path=ref_img_path,
                       sat_model_params=sat_model_params,
                       dem_path=dem_path,
                       opt_params=opt_params,
                       opt_gcp_file_path=SOFTWARE.WKDIR,
                       corr_config=corr_config,
                       debug=True,
                       svg_patches=False)
