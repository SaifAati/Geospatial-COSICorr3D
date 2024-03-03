"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""

import os
import pytest
import logging

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoRFM.load_rfm import ReadRFM

test_folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset/test_ortho_dataset")

test_rfm_fn = os.path.join(test_folder, "SP2_RPC.txt")

log = geoCosiCorr3DLog("test_rfm_loading")


@pytest.mark.functional
@pytest.mark.parametrize('rfm_fn', [test_rfm_fn])
def test_loading_rfm(rfm_fn):
    rpc = ReadRFM(rfm_fn)
    logging.info(rpc.__repr__())
