"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import sys

from geoCosiCorr3D.geoCore.constants import SENSOR
from geoCosiCorr3D.geoCore.core_RSM import RSM


def geoRSM_generation(sensor_name: str, metadata_file: str, debug: bool = True) -> RSM:
    if sensor_name not in SENSOR.SENSOR_LIST:
        raise sys.exit(f'Sensor {sensor_name} not listed in geoCosiCorr3D supported RSM sensor list: {sensor_name}')
    elif sensor_name in [SENSOR.SPOT1, SENSOR.SPOT2, SENSOR.SPOT3, SENSOR.SPOT4, SENSOR.SPOT5, SENSOR.SPOT1_5]:
        from geoCosiCorr3D.geoRSM.Spot_RSM import cSpot15
        return cSpot15(dmpFile=metadata_file, debug=debug)
    elif sensor_name in [SENSOR.SPOT6, SENSOR.SPOT7, SENSOR.SPOT67]:
        from geoCosiCorr3D.geoRSM.Spot_RSM import cSpot67
        return cSpot67(dmpXml=metadata_file, debug=True)
    elif sensor_name in [SENSOR.WV1, SENSOR.WV2, SENSOR.WV3, SENSOR.WV4, SENSOR.DG, SENSOR.QB, SENSOR.GE]:
        from geoCosiCorr3D.geoRSM.DigitalGlobe_RSM import cDigitalGlobe
        return cDigitalGlobe(dgFile=metadata_file, debug=True)
