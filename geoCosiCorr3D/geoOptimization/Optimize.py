"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import numpy as np
from scipy.optimize import least_squares
from typing import Optional, List


def Optimize(xy_obs_with_model, xy_error, method="lm", ini_sol: Optional[List] = None):
    if ini_sol is None:
        ini_sol = [0, 0, 0, 0, 0, 0]

    def ModelFunction(xError_, yError_, xPix, yPix, params):

        f_col = params[0] * xPix + params[1] * yPix + params[2]
        f_row = params[3] * xPix + params[4] * yPix + params[5]
        xRes = xError_ - f_col
        yRes = yError_ - f_row
        return xRes, yRes

    def Fun(params):
        V = np.zeros(2 * xy_error.shape[0])
        for i, dx, dy, xPix, yPix in zip(np.arange(0, xy_error.shape[0], 1),
                                         xy_error[:, 0], xy_error[:, 1],
                                         xy_obs_with_model[:, 0], xy_obs_with_model[:, 1]):
            V[2 * i], V[2 * i + 1] = ModelFunction(xError_=dx, yError_=dy, xPix=xPix, yPix=yPix, params=params)

        return V

    X0 = np.array(ini_sol)
    res = least_squares(Fun, X0, method="trf", loss='cauchy')

    return res
