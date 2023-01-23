"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os
import numpy as np
import pytest

import geoCosiCorr3D
from geoCosiCorr3D.geoRSM.geoRSM_generation import geoRSM_generation
from geoCosiCorr3D.geoCore.constants import SENSOR, SOFTWARE
from geoCosiCorr3D.geoRSM.Spot_RSM import cSpot67
from geoCosiCorr3D.utils.misc import read_json_as_dict

folder = os.path.join(SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
metadata_list = [os.path.join(folder, file) for file in
                 ['SPOT1.DIM', 'SPOT2.DIM', 'SPOT3.DIM', 'SPOT4.DIM', 'SPOT5.DIM', 'SPOT6.XML', 'WV1.XML', 'WV2.XML',
                  'WV3.XML']]
test_sensor_list = [SENSOR.SPOT1, SENSOR.SPOT2, SENSOR.SPOT3, SENSOR.SPOT4, SENSOR.SPOT5, SENSOR.SPOT6, SENSOR.WV1,
                    SENSOR.WV2, SENSOR.WV3]
rsm_models = [geoRSM_generation(sensor_name=sensor, metadata_file=metadata_file) for (sensor, metadata_file) in
              zip(test_sensor_list, metadata_list)]
expected_model_classes = 5 * [geoCosiCorr3D.geoRSM.Spot_RSM.cSpot15]
expected_model_classes.extend(
    [geoCosiCorr3D.geoRSM.Spot_RSM.cSpot67, geoCosiCorr3D.geoRSM.DigitalGlobe_RSM.cDigitalGlobe])
expected_model_classes.extend(2 * [geoCosiCorr3D.geoRSM.DigitalGlobe_RSM.cDigitalGlobe])


@pytest.mark.parametrize('test_input, expected',
                         [(model_, class_) for model_, class_ in zip(rsm_models, expected_model_classes)])
def test_rsm_generation(test_input, expected):
    assert isinstance(test_input, expected)


expec_res = read_json_as_dict(os.path.join(folder, 'expec_test_georsm.json'))

model: geoCosiCorr3D.geoRSM.Spot_RSM.cSpot67 = rsm_models[5]


@pytest.mark.parametrize('test_input, expected', [
    (model.nbCols, expec_res['nbCols']),
    (model.nbRows, expec_res['nbRows']),
    (model.platform, expec_res['platform']),
    (model.sunAz, expec_res['sunAz']),
    (model.sunElev, expec_res['sunElev']),
    (model.satElev, expec_res['satElev']),
    (model.satAz, expec_res['satAz']),
    (model.viewAngle, expec_res['viewAngle']),
    (model.incidenceAngle, expec_res['incidenceAngle']),
    (model.avgLineRate, expec_res['avgLineRate']),
    (model.scanDirection, expec_res['scanDirection']),
    (model.nbBands, expec_res['nbBands']),
    (model.gsd_ACT, expec_res['gsd_ACT']),
    (model.gsd_ALT, expec_res['gsd_ALT']),
    (model.meanGSD, expec_res['meanGSD']),
    (model.gsd, expec_res['gsd']),
    (model.time, expec_res['time']),
    (model.startTime, expec_res['startTime']),
    (model.date_time_obj, expec_res['date_time_obj']),
    (float(model.focal), float(expec_res['focal'])),
    (float(model.szCol), float(expec_res['szCol'])),
    (float(model.szRow), float(expec_res['szRow'])),
])
def test_satellite_anc_reading(test_input, expected):
    assert test_input == expected


@pytest.mark.parametrize('test_input, expected, msg', [
    (model.lineOfSight, expec_res['lineOfSight'], 'invalid lineOfSight '),
    (model.position, expec_res['position'], 'invalid position'),
    (model.velocity, expec_res['velocity'], 'invalid velocity'),
    (model.ephTime, expec_res['ephTime'], 'invalid ephTime'),
    (model.Q0, expec_res['Q0'], 'invalid Q0'),
    (model.Q1, expec_res['Q1'], 'invalid Q1'),
    (model.Q2, expec_res['Q2'], 'invalid Q2'),
    (model.Q3, expec_res['Q3'], 'invalid Q3'),
    (model.QTime, expec_res['QTime'], 'invalid QTime'),
    (model.linePeriod, expec_res['linePeriod'], 'invalid linePeriod'),
    (model.linesDate, expec_res['linesDate'], 'invalid linesDate'),
    (model.orbitalPos_Z, expec_res['orbitalPos_Z'], 'invalid orbitalPos_Z'),
    (model.orbitalPos_X, expec_res['orbitalPos_X'], 'invalid orbitalPos_X'),
    (model.orbitalPos_Y, expec_res['orbitalPos_Y'], 'invalid orbitalPos_Y'),
])
def test_rsm_attitude_motion_parsing(test_input, expected, msg):
    np.testing.assert_array_equal(test_input, expected, msg)


@pytest.mark.parametrize('test_input, expected, msg', [
    (model.satToNavMat, expec_res['satToNavMat'], 'invalid satToNavMat '),
    (model.interpSatPosition, expec_res['interpSatPosition'], 'invalid interpSatPosition'),
    (model.interpSatVelocity, expec_res['interpSatVelocity'], 'invalid interpSatVelocity'),
    (model.CCDLookAngle, expec_res['CCDLookAngle'], 'invalid CCDLookAngle'),
    (model.Q0interp, expec_res['Q0interp'], 'invalid Q0interp'),
    (model.Q1interp, expec_res['Q1interp'], 'invalid Q1interp'),
    (model.Q2interp, expec_res['Q2interp'], 'invalid Q2interp'),
    (model.Q3interp, expec_res['Q3interp'], 'invalid Q3interp'),
])
def test_build_rsm(test_input, expected, msg):
    np.testing.assert_allclose(test_input,expected, err_msg=msg)
