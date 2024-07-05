"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""

import os
import pytest
import logging
import numpy as np
import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoCosiCorr3dLogger import GeoCosiCorr3DLog
from geoCosiCorr3D.geoRFM.RFM import RFM

folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset/test_ortho_dataset")
test_raw_img_fn = os.path.join(folder, "RAW_SP2.TIF")
test_dem_fn = os.path.join(folder, "DEM.tif")
test_rfm_fn = os.path.join(folder, "SP2_RPC.txt")
dmpFile = os.path.join(folder, "SP2_METADATA.DIM")

log = GeoCosiCorr3DLog("test_sat_models")

COLS = [0, 5999, 0, 5999, 3000, 3001]
LINS = [0, 0, 5999, 5999, 3000, 3001]


@pytest.mark.parametrize('rfm_fn', [test_rfm_fn])
def test_rfm_transformer(rfm_fn):
    model_data = RFM(test_rfm_fn, dem_fn=test_dem_fn, debug=True)
    lla = model_data.i2g(col=COLS, lin=LINS)
    lla = np.asarray(lla).T
    logging.info(f'lla:{lla}')
    res = model_data.g2i(lla[:, 0], lla[:, 1], lla[:, 2])
    res = np.asarray(res).T
    logging.info(f'res:{res}')
    expected_coords = np.array([COLS, LINS]).T
    logging.info(f'expected_coords:{expected_coords}')
    assert np.allclose(expected_coords, res, atol=1e-6)


@pytest.mark.functional
@pytest.mark.parametrize('rfm_fn', [test_rfm_fn])
def test_rfm_model_properties(rfm_fn):
    model_data = RFM(test_rfm_fn, dem_fn=test_dem_fn, debug=True)
    logging.info(f'attitude range:{model_data.get_altitude_range()}')
    logging.info(f'GSD:{model_data.get_gsd()}')
    logging.info(f'geoTransform:{model_data.get_geotransform()}')
