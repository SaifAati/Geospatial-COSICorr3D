"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from abc import ABC
import numpy as np


class BaseRFM(ABC):
    def __init__(self):
        self.NB_NUM_COEF = 20
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

        self.lonNum = [np.nan] * self.NB_NUM_COEF
        self.lonDen = [np.nan] * self.NB_NUM_COEF
        self.latNum = [np.nan] * self.NB_NUM_COEF
        self.latDen = [np.nan] * self.NB_NUM_COEF
        self.linNum = [np.nan] * self.NB_NUM_COEF
        self.linDen = [np.nan] * self.NB_NUM_COEF
        self.colNum = [np.nan] * self.NB_NUM_COEF
        self.colDen = [np.nan] * self.NB_NUM_COEF

    def __repr__(self):
        return """
        {}
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
      colDen = {} """.format(
            self.__class__.__name__,
            self.linOff,
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
