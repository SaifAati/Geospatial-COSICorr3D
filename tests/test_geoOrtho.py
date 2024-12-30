"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import tempfile

import pytest

import geoCosiCorr3D.geoCore.constants as const
from geoCosiCorr3D.geoOrthoResampling.geoOrtho import RFMOrtho, RSMOrtho
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo

folder = os.path.join(const.SOFTWARE.PARENT_FOLDER, "tests/test_dataset/test_ortho_dataset")
test_raw_img_fn = os.path.join(folder, "RAW_SP2.TIF")
test_dem_path = os.path.join(folder, "DEM.tif")
test_rfm_fn = os.path.join(folder, "SP2_RPC.txt")
test_dmp_fn = os.path.join(folder, "SP2_METADATA.DIM")


@pytest.mark.parametrize('o_gsd', [30, 15])
@pytest.mark.parametrize('resampling_method', const.GEOCOSICORR3D_RESAMLING_METHODS)
def test_ortho_rsm(o_gsd, resampling_method):
    with tempfile.TemporaryDirectory(dir=const.SOFTWARE.WKDIR, suffix='_test_rsm_ortho') as tmp_dir:
        ortho_params = {
            "method":
                {"method_type": const.SATELLITE_MODELS.RSM,
                 "metadata": test_dmp_fn,
                 "sensor": const.SENSOR.SPOT1_5,
                 "corr_model": None
                 },
            "GSD": o_gsd,
            "resampling_method": resampling_method}
        o_ortho_path = os.path.join(tmp_dir,
                                    f"ORTHO_{ortho_params['resampling_method']}_RSM_{ortho_params['GSD']}_GSD.tif")
        temp_ortho_obj = RSMOrtho(input_l1a_path=test_raw_img_fn,
                                  output_ortho_path=o_ortho_path,
                                  dem_path=test_dem_path,
                                  ortho_params=ortho_params)
        temp_ortho_obj()
        temp_ortho = cRasterInfo(temp_ortho_obj.output_ortho_path).geo_transform

        assert os.path.exists(o_ortho_path)
        if o_gsd == 15:
            assert temp_ortho == [273585.0, 15, 0, 4568250.0, 0, -15]
        if o_gsd == 30:
            assert temp_ortho == [273600.0, 30.0, 0.0, 4568250.0, 0.0, -30.0]

    return



@pytest.mark.parametrize('o_gsd', [30])
@pytest.mark.parametrize('resampling_method', [const.GEOCOSICORR3D_RESAMLING_METHODS[0]])
def test_ortho_rfm(o_gsd, resampling_method):
    with tempfile.TemporaryDirectory(dir=const.SOFTWARE.WKDIR, suffix='_test_rfm_ortho') as tmp_dir:
        ortho_params = {
            "method":
                {"method_type": const.SATELLITE_MODELS.RFM,
                 "metadata": test_rfm_fn,
                 "corr_model": None,
                 },
            "GSD": o_gsd,
            "resampling_method": resampling_method}
        o_ortho_path = os.path.join(tmp_dir,
                                    f"ORTHO_{ortho_params['resampling_method']}_RFM_{ortho_params['GSD']}_GSD.tif")
        temp_ortho_obj = RFMOrtho(input_l1a_path=test_raw_img_fn,
                                  output_ortho_path=o_ortho_path,
                                  dem_path=test_dem_path,
                                  ortho_params=ortho_params)
        temp_ortho_obj()
        temp_ortho = cRasterInfo(temp_ortho_obj.output_ortho_path).geo_transform
        assert os.path.exists(o_ortho_path)
        assert temp_ortho == [273570.0, 30, 0, 4568250.0, 0, -30]

    return
