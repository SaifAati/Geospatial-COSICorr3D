"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import os
import geoCosiCorr3D
from geoCosiCorr3D.geoOrthoResampling.geoOrtho import RFMOrtho, RSMOrtho
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoCore.constants import SOFTWARE

log = geoCosiCorr3DLog("Test_geoOrtho")

folder = os.path.join(os.path.dirname(geoCosiCorr3D.__file__), "Tests/4-geoOrtho_Test/Sample/Sample2")


def Test_geoOrthoResampling_RSM():
    rawImg = os.path.join(folder, "SPOT2.TIF")
    dmpFile = os.path.join(folder, "METADATA_SPOT2.DIM")
    demPath = os.path.join(folder, "SRTM_DEM_subset_o.tif")
    oRes = 10

    oOrthoPath = os.path.join(SOFTWARE.WKDIR, "oOrtho_SPOT2_RSM.TIF")

    params = {
        "method":
            {"method_type": "RSM",
             "metadata": dmpFile,
             "sensor": "Spot15",
             "corr_model": None
             },
        "GSD": oRes,
        "resampling_method": "sinc"}
    RSMOrtho(input_l1a_path=rawImg,
             output_ortho_path=oOrthoPath,
             dem_path=demPath,
             ortho_params=params)
    return


def Test_geoOrthoResampling_RFM():
    logging.info("_______RFM__________")
    rawImg = os.path.join(folder, "SPOT2.TIF")
    demPath = os.path.join(folder, "SRTM_DEM_subset_o.tif")
    oRes = 10
    rfmFile = os.path.join(folder, "SPOT2_RPC.txt")

    oOrthoPath = os.path.join(SOFTWARE.WKDIR, "oOrtho_SPOT2_RFM.TIF")
    rfm_params = {
        "method":
            {"method_type": "RFM", "metadata": rfmFile, "corr_model": None,
             },
        "GSD": oRes,
        "resampling_method": "sinc"}
    RFMOrtho(input_l1a_path=rawImg,
             output_ortho_path=oOrthoPath,
             dem_path=demPath,
             ortho_params=rfm_params)
    return


if __name__ == '__main__':
    Test_geoOrthoResampling_RSM()
    Test_geoOrthoResampling_RFM()
    #TODO:
    # 1- add correction matrix to RFM and test the output
    # 2- Implement RSM orthorectification
    # 3- Differentiate between unit/function test and sample examples