"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os

from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate

folder = os.path.join(os.path.dirname(__file__), "Samples/DG_Sample")
img1 = os.path.join(folder, "BASE_IMG.TIF")
img2 = os.path.join(folder, "TARGET_IMG.TIF")

###### TEST FREQUENCY CORRELATOR #######################
corr_config = {"correlator_name": "frequency",
               "correlator_params": {
                   "window_size": [64, 64, 64, 64],
                   "step": [8, 8],
                   "grid": True,
                   "mask_th": 0.95,
                   "nb_iters": 4
               }
               }

correlation = Correlate(base_image_path=img1,
                        target_image_path=img2,
                        base_band=1,
                        target_band=1,
                        output_corr_path=None,
                        corr_config=corr_config,
                        corr_show=True)

###### TEST SPATIAL CORRELATOR #######################
corr_config = {"correlator_name": "spatial",
               "correlator_params": {
                   "window_size": 4 * [64],
                   "step": [16, 16],
                   "grid": True,
                   "search_range": [10, 10]
               }
               }

Correlate(base_image_path=img1,
          target_image_path=img2,
          base_band=1,
          target_band=1,
          output_corr_path=None,
          corr_config=corr_config,
          corr_show=True)
