"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import pandas
import pytest
import tempfile
from geoCosiCorr3D.geoTiePoints.MicMacTP import cMicMacTp
from geoCosiCorr3D.geoTiePoints.Tp2GCPs import TPsTOGCPS
from geoCosiCorr3D.geoCore.constants import *

folder = os.path.join(SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
img1 = os.path.join(folder, "BASE_IMG.TIF")
img2 = os.path.join(folder, "TARGET_IMG.TIF")
match_file = os.path.join(folder, 'basemap_VS_SP2.pts')
raw_img_path = os.path.join(folder, 'SP2.TIF')
ref_img_path = os.path.join(folder, 'basemap.TIF')
dem_path = os.path.join(folder, 'DEM.TIF')


@pytest.mark.functional
def test_tp_to_gcps():
    with tempfile.TemporaryDirectory(dir=SOFTWARE.WKDIR, suffix='_test_gcp') as tmp_dir:
        tp_2_gcp = TPsTOGCPS(in_tp_file=match_file,
                             base_img_path=raw_img_path,
                             ref_img_path=ref_img_path,
                             dem_path=dem_path,
                             output_gcp_path=tmp_dir)

    gcp_df = tp_2_gcp.gcp_df[['lon', 'lat', 'alt', 'xPix', 'yPix', 'x_map', 'y_map', 'epsg']]

    expec_gcp = pandas.read_csv(os.path.join(folder, 'basemap_VS_SP2_GCP_expec.csv'))
    expec_array = np.array([expec_gcp['lon'].values, expec_gcp['lat'].values, expec_gcp['alt'].values,
                            expec_gcp['xPix'].values, expec_gcp['yPix'].values, expec_gcp['x_map'].values,
                            expec_gcp['y_map'].values]).T
    gcp_array = np.array([gcp_df['lon'].values, gcp_df['lat'].values, gcp_df['alt'].values,
                          gcp_df['xPix'].values, gcp_df['yPix'].values, gcp_df['x_map'].values,
                          gcp_df['y_map'].values]).T
    assert np.allclose(expec_array, gcp_array)


@pytest.mark.functional
def test_asift_tps():
    with tempfile.TemporaryDirectory(dir=SOFTWARE.WKDIR, suffix='_test_asift') as tmp_dir:
        tp_obj = cMicMacTp(ref_img_path=img1,
                           raw_img_path=img2,
                           scale_factor=1 / 6,
                           plot_tps=False,
                           o_dir=tmp_dir
                           )
        expec_tps = np.loadtxt(os.path.join(folder, 'expec_mmtps.pts'))

        assert np.allclose(np.loadtxt(tp_obj.o_tp_path, comments=';'), expec_tps)
    return
