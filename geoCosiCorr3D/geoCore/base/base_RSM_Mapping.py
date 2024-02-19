"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from geoCosiCorr3D.geoConfig import cgeoCfg
from geoCosiCorr3D.geoCore.constants import EARTH

geoCfg = cgeoCfg()


class BasePix2GroundDirectModel(ABC):

    def __init__(self,
                 rsmModel,
                 xPix,
                 yPix,
                 rsmCorrectionArray: np.ndarray = np.zeros((3, 3)),
                 demFile: Optional[str] = None,
                 hMean: Optional[float] = None,
                 oProjEPSG: Optional[int] = None,
                 semiMajor=EARTH.SEMIMAJOR,
                 semiMinor=EARTH.SEMIMINOR,
                 debug: Optional[bool] = False):
        self.rsmModel = rsmModel
        self.xPix = xPix
        self.yPix = yPix
        self.rsmCorrection = rsmCorrectionArray
        self.demFile = demFile
        self.hMean = hMean
        self.oProjEPSG = oProjEPSG
        self.semiMajor = semiMajor
        self.semiMinor = semiMinor
        self.debug = debug

    @abstractmethod
    def compute_pix2ground_direct_model(self):
        """
        Note:
            If no DEM and oProjEPSG are passed along the ground location of the pixel is expressed in
            oProjEPSG = WGS-84 for earth.
            If DEM and oProjEPSG are passed along, check if the datum are the same on the 2 projection
             if True --> oProjEPSG == demInfo.EPSG_Code
              else errorMsg
        """
        pass

    def __repr__(self):
        pass

    def __str__(self):
        pass
