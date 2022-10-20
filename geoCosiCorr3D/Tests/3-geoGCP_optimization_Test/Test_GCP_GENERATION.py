"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os
import geoCosiCorr3D
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoTiePoints.Tp2GCPs import TPsTOGCPS
from geoCosiCorr3D.geoCore.constants import SOFTWARE

log = geoCosiCorr3DLog("Test_TP2GCP")

folder = os.path.join(os.path.dirname(geoCosiCorr3D.__file__), 'Tests/3-geoGCP_optimization_Test/Sample')

match_file = os.path.join(folder, 'rOrtho_VS_1999-07-10-09-07-26-Spot-2-HRV-2-P-10.pts')
raw_img_path = os.path.join(folder, '1999-07-10-09-07-26-Spot-2-HRV-2-P-10.TIF')
ref_img_path = os.path.join(folder, 'rOrtho.TIF')
dem_path = os.path.join(folder, 'rDEM.TIF')

gcpObj = TPsTOGCPS(in_tp_file=match_file,
                   base_img_path=raw_img_path,
                   ref_img_path=ref_img_path,
                   dem_path=dem_path,
                   output_gcp_path=SOFTWARE.WKDIR,
                   debug=True)
