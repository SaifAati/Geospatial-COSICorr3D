"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import tempfile

import numpy as np
import pandas
import pytest

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoTiePoints.MicMacTP import cMicMacTp
from geoCosiCorr3D.geoTiePoints.Tp2GCPs import TPsTOGCPS

folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
img1 = os.path.join(folder, "BASE_IMG.TIF")
img2 = os.path.join(folder, "TARGET_IMG.TIF")
match_file = os.path.join(folder, 'basemap_VS_SP2.pts')
raw_img_path = os.path.join(folder, 'SP2.TIF')
ref_img_path = os.path.join(folder, 'basemap.TIF')
dem_path = os.path.join(folder, 'DEM.TIF')


def asift_tps():
    tp_obj = cMicMacTp(ref_img_path=img1,
                       raw_img_path=img2,
                       scale_factor=1 / 6,
                       plot_tps=True,
                       o_dir='/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD'
                       # o_dir=tmp_dir
                       )
    # expec_tps = np.loadtxt(os.path.join(folder, 'expec_mmtps.pts'))

    # assert np.allclose(np.loadtxt(tp_obj.o_tp_path, comments=';'), expec_tps)
    return


if __name__ == '__main__':
    asift_tps()
