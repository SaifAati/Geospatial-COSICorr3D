"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""

import os
import pytest
import logging

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import SatMapGrid
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoRFM.RFM import RFM
from geoCosiCorr3D.geoCore.core_RSM import RSM

test_folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset/test_ortho_dataset")
test_raw_img_fn = os.path.join(test_folder, "RAW_SP2.TIF")
test_rfm_fn = os.path.join(test_folder, "SP2_RPC.txt")
test_dem_fn = os.path.join(test_folder, "DEM.tif")
test_dmp_fn = os.path.join(test_folder, "SP2_METADATA.DIM")


@pytest.mark.parametrize('o_res', [20, 5])
@pytest.mark.parametrize('model_data_type', [[RFM(test_rfm_fn, debug=True), C.SATELLITE_MODELS.RFM],
                                             [RSM.build_RSM(metadata_file=test_dmp_fn, sensor_name=C.SENSOR.SPOT1_5,
                                                            debug=True), C.SATELLITE_MODELS.RSM]])
def test_geoSatGrid(model_data_type, o_res):
    l1a_raster_info = geoRT.cRasterInfo(test_raw_img_fn)
    ortho_grid = SatMapGrid(raster_info=l1a_raster_info,
                            model_data=model_data_type[0],
                            model_type=model_data_type[1],
                            dem_fn=test_dem_fn,
                            new_res=o_res,
                            debug=True)
    logging.info(ortho_grid.__repr__())
