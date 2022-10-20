"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from abc import ABC, abstractmethod


class BaseRSM(ABC):
    @abstractmethod
    def ComputeAttitude(self):
        pass

    @abstractmethod
    def Interpolate_position_velocity_attitude(self):
        pass
