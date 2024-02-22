"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os
import tempfile
from typing import Dict

import pytest

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate

folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
img1 = os.path.join(folder, "BASE_IMG.TIF")
img2 = os.path.join(folder, "TARGET_IMG.TIF")


# TODO:
# Correlation between patches

@pytest.mark.parametrize('test_corr_config',
                         [C.TEST_CONFIG.FREQ_CORR_CONFIG, C.TEST_CONFIG.SPA_CORR_CONFIG])
def test_corr(test_corr_config: Dict):
    with tempfile.TemporaryDirectory(dir=C.SOFTWARE.WKDIR, suffix='test_corr') as tmp_dir:
        Correlate(base_image_path=img1,
                  target_image_path=img2,
                  output_corr_path=tmp_dir,
                  corr_config=test_corr_config,
                  corr_show=False)
