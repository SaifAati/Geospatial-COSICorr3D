"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from abc import ABC
import numpy as np


class BaseRFM(ABC):
    def __init__(self):
        self.init_RFM()

    def init_RFM(self):
        self.linOff = np.nan
        self.colOff = np.nan
        self.latOff = np.nan
        self.lonOff = np.nan
        self.altOff = np.nan
        self.linScale = np.nan
        self.colScale = np.nan
        self.latScale = np.nan
        self.lonScale = np.nan
        self.altScale = np.nan

        self.lonNum = [np.nan] * 20
        self.lonDen = [np.nan] * 20
        self.latNum = [np.nan] * 20
        self.latDen = [np.nan] * 20
        self.linNum = [np.nan] * 20
        self.linDen = [np.nan] * 20
        self.colNum = [np.nan] * 20
        self.colDen = [np.nan] * 20

    def __repr__(self):
        return """
    # Offsets and Scales
      linOffset = {}
      colOffset = {}
      latOffset = {}
      lonOffset = {}
      altOffset = {}
      rowScale = {}
      colScale = {}
      latScale = {}
      lonScale = {}
      altScale = {}

      # Inverse model functions coefficients
      linNum = {}
      linDen = {}
      colNum = {}
      colDen = {} """.format(self.linOff,
                             self.colOff,
                             self.latOff,
                             self.lonOff,
                             self.altOff,
                             self.linScale,
                             self.colScale,
                             self.latScale,
                             self.lonScale,
                             self.altScale,
                             ' '.join(['{: .8f}'.format(x) for x in self.linNum]),
                             ' '.join(['{: .8f}'.format(x) for x in self.linDen]),
                             ' '.join(['{: .8f}'.format(x) for x in self.colNum]),
                             ' '.join(['{: .8f}'.format(x) for x in self.colDen])
                             )
