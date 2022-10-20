# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2020

import numbers
from typing import TypeVar

import numpy as np

FloatOrArray = TypeVar('FloatOrArray', float, np.ndarray)  # float or numpy array
NumOrArray = TypeVar('NumOrArray', numbers.Real, np.ndarray)  # int, float or numpy array
