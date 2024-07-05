"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""

import logging
import os

import numpy as np
import pandas

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoCore.core_RSM import RSM
from geoCosiCorr3D.geoCosiCorr3dLogger import GeoCosiCorr3DLog
from geoCosiCorr3D.geoOptimization.model_refinement.sat_model_refinement import RsmRefinement

# folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
folder = '/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/NEW/'
log = GeoCosiCorr3DLog("GCP_OPTIMIZATION", log_dir=folder)
gcp_file = os.path.join(folder, 'input_GCPs.csv')
raw_img_path = os.path.join(folder, 'SP2.TIF')
ref_img_path = os.path.join(folder, 'basemap.TIF')
dem_path = os.path.join(folder, 'DEM.TIF')
dmp_file = os.path.join(folder, 'SP-2.DIM')
sensor = 'Spot2'
params = C.SatModelParams(C.SATELLITE_MODELS.RSM, dmp_file, C.SENSOR.SPOT2)
debug = False
gcps_df = pandas.read_csv(gcp_file)
sat_model = RSM.build_RSM(metadata_file=dmp_file,
                          sensor_name=sensor,
                          debug=debug)


def test_linear_refinement():
    rsm_refinement = RsmRefinement(sat_model=sat_model, gcps=gcps_df, debug=debug)
    rsm_refinement.refine()

    expected_correction = np.array([[5.72835604e-07, 1.68738324e-07, 5.61578545e-07],
                                    [1.66133358e-08, -6.20805542e-10, -1.89712215e-08],
                                    [-1.26942529e-02, 6.70568661e-02, -4.98777828e-03]])
    assert np.allclose(rsm_refinement.corr_model, expected_correction, atol=1e-6)
    logging.info(f"Test passed: {rsm_refinement.__class__.__name__}")

    return


def test_quadratic_refinement():
    rsm_refinement = RsmRefinement(sat_model=sat_model, gcps=gcps_df, debug=debug, refinement_model='quadratic')
    rsm_refinement.refine()

    expected_correction = np.array([[-1.11478749e-10, 3.26386710e-10, 4.39423897e-11],
                                    [4.81857063e-12, -1.67358404e-11, -1.38823378e-12],
                                    [1.54626372e-13, 4.14975682e-12, -1.15428706e-12],
                                    [6.93428247e-07, -1.86290440e-07, 5.15023123e-07],
                                    [1.30457862e-08, -2.02618034e-08, -1.03469703e-08],
                                    [-1.27240026e-02, 6.71842043e-02, -4.98592236e-03]])

    assert np.allclose(rsm_refinement.corr_model, expected_correction, atol=1e-6)
    logging.info(f"Test passed: {rsm_refinement.__class__.__name__}")
    return


def test_wlsq_gm_refinement():
    rsm_refinement = RsmRefinement(sat_model=sat_model, gcps=gcps_df, debug=debug, solver='wlsq_gm')
    rsm_refinement.refine()

    expected_correction = np.array([[5.72835604e-07, 1.68738324e-07, 5.61578545e-07],
                                    [1.66133358e-08, -6.20805542e-10, -1.89712215e-08],
                                    [-1.26942529e-02, 6.70568661e-02, -4.98777828e-03]])
    assert np.allclose(rsm_refinement.corr_model, expected_correction, atol=1e-6)
    logging.info(f"Test passed: {rsm_refinement.__class__.__name__}")

    return


if __name__ == '__main__':
    print('hi')

    test_linear_refinement()
    print('--------------------------------')
    test_quadratic_refinement()
    print('--------------------------------')
    test_wlsq_gm_refinement()
