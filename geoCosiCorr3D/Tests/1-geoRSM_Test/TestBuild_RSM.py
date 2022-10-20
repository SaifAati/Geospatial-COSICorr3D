"""
AUTHOR: Saif Aati (saif@caltech.edu)
"""
import logging
import os
from pathlib import Path

import geoCosiCorr3D
from geoCosiCorr3D.geoRSM.geoRSM import geoRSM
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog

folder = os.path.join(os.path.dirname(geoCosiCorr3D.__file__), "Tests/1-geoRSM_Test/Samples")
log = geoCosiCorr3DLog("TestBuild_RSM")

wv1 = os.path.join(folder, "WV1.XML")
wv2 = os.path.join(folder, "WV2.XML")
wv3 = os.path.join(folder, "WV3.XML")

logging.info("__________________________{}____________________".format(Path(wv1).stem))
geoRSM('WV1', wv1)
logging.info("__________________________{}____________________".format(Path(wv2).stem))
geoRSM('WV2', wv2)
logging.info("__________________________{}____________________".format(Path(wv3).stem))
geoRSM('WV3', wv3)


spot1 = os.path.join(folder, "SPOT1.DIM")
spot2 = os.path.join(folder, "SPOT2.DIM")
spot3 = os.path.join(folder, "SPOT3.DIM")
spot4 = os.path.join(folder, "SPOT4.DIM")
spot5 = os.path.join(folder, "SPOT5.DIM")
for dmpFile in [spot1, spot2, spot3, spot4, spot5]:
    logging.info("__________________________{}____________________".format(Path(dmpFile).stem))
    geoRSM('Spot15', dmpFile)

spot6 = os.path.join(folder, "SPOT6.XML")
logging.info("__________________________{}____________________".format(Path(spot6).stem))
geoRSM('Spot67', spot6)
