"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import pytest
import tempfile
from geoCosiCorr3D.geoOrthoResampling.geoOrtho import RFMOrtho, RSMOrtho
from geoCosiCorr3D.geoCore.constants import *
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo

folder = os.path.join(SOFTWARE.PARENT_FOLDER, "tests/test_dataset/test_ortho_dataset")
rawImg = os.path.join(folder, "RAW_SP2.TIF")
demPath = os.path.join(folder, "DEM.tif")
rfmFile = os.path.join(folder, "SP2_RPC.txt")
dmpFile = os.path.join(folder, "SP2_METADATA.DIM")
expec_orthos = np.load(os.path.join(folder, 'Expec_orthos.npz'))


@pytest.mark.parametrize('o_gsd', [15, 30])
@pytest.mark.parametrize('resampling_method', GEOCOSICORR3D_RESAMLING_METHODS)
def test_ortho_rsm(o_gsd, resampling_method):
    with tempfile.TemporaryDirectory(dir=SOFTWARE.WKDIR, suffix='_test_rsm_ortho') as tmp_dir:
        oOrthoPath = os.path.join(tmp_dir, "temp_rsm_ortho.tif")
        ortho_params = {
            "method":
                {"method_type": SATELLITE_MODELS.RSM,
                 "metadata": dmpFile,
                 "sensor": SENSOR.SPOT1_5,
                 "corr_model": None
                 },
            "GSD": o_gsd,
            "resampling_method": resampling_method}

        temp_ortho_obj = RSMOrtho(input_l1a_path=rawImg,
                                  output_ortho_path=oOrthoPath,
                                  dem_path=demPath,
                                  ortho_params=ortho_params)
        temp_ortho = cRasterInfo(temp_ortho_obj.output_ortho_path).raster_array
    expec_ortho = expec_orthos[f"Expec_ORTHO_{ortho_params['resampling_method']}_RSM_{ortho_params['GSD']}_GSD.npy"]

    np.testing.assert_allclose(temp_ortho, expec_ortho, rtol=1e-1)

    return

@pytest.mark.parametrize('o_gsd', [15, 30])
@pytest.mark.parametrize('resampling_method', GEOCOSICORR3D_RESAMLING_METHODS)
def test_ortho_rfm(o_gsd, resampling_method):
    with tempfile.TemporaryDirectory(dir=SOFTWARE.WKDIR, suffix='_test_rfm_ortho') as tmp_dir:
        oOrthoPath = os.path.join(tmp_dir, "temp_rfm_ortho.tif")
        ortho_params = {
            "method":
                {"method_type": SATELLITE_MODELS.RFM, "metadata": rfmFile, "corr_model": None,
                 },
            "GSD": o_gsd,
            "resampling_method": resampling_method}
        temp_ortho_obj = RFMOrtho(input_l1a_path=rawImg,
                                  output_ortho_path=oOrthoPath,
                                  dem_path=demPath,
                                  ortho_params=ortho_params)

        temp_ortho = cRasterInfo(temp_ortho_obj.output_ortho_path).raster_array

    expec_ortho = expec_orthos[f"Expec_ORTHO_{ortho_params['resampling_method']}_RFM_{ortho_params['GSD']}_GSD.npy"]

    assert np.allclose(temp_ortho, expec_ortho)

    return
