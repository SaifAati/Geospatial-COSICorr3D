"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import warnings
import numpy as np
from geoCosiCorr3D.geoCore.base.base_RFM import BaseRFM


class RfmModel(BaseRFM):
    __delta = 2
    __th = 1e-18
    __nbIterMax = 200

    def __init__(self):
        super().__init__()

    def build_RFM(self, x, y, z, num, den):
        """

        Args:
            x: x-coordinate: float
            y: y-coordinate: float
            z: z-coordinate: float
            num: numerator coefficient : list, len=20
            den: denominator coefficient : list, len=20

        Returns:

        """
        numPoly3d = self.__poly_3d(num, x, y, z)
        denPoly3d = self.__poly_3d(den, x, y, z)

        return numPoly3d / denPoly3d

    @staticmethod
    def __poly_3d(coefList, x, y, z):
        """

        Args:
            coefList: list of the 20 coefficients
            x: x-coord : float
            y: y-coord : float
            z: z-coord : float

        Returns:

        """

        return coefList[0] + \
            coefList[1] * y + coefList[2] * x + coefList[3] * z + \
            coefList[4] * y * x + coefList[5] * y * z + coefList[6] * x * z + \
            coefList[7] * y ** 2 + coefList[8] * x ** 2 + coefList[9] * z ** 2 + \
            coefList[10] * x * y * z + \
            coefList[11] * y ** 3 + \
            coefList[12] * y * x ** 2 + coefList[13] * y * z ** 2 + coefList[14] * x * y ** 2 + \
            coefList[15] * x ** 3 + \
            coefList[16] * x * z ** 2 + coefList[17] * z * y ** 2 + coefList[18] * z * x ** 2 + \
            coefList[19] * z ** 3

    def direct_model(self, colN, linN, altN, corr_model=np.zeros((3, 3))):
        """
        Invert the Inverse RFM model
        Args:
            colN: normalized column pixel coordinate, list or a single value
            linN: normalized line(row) pixel coordinate, list or a single value
            altN: normalized altitude coordinate, list or a single value

        Returns: normalized lon lat, list or single value
        Notes:
            Adapted version of RPCM numerical inversion.
            https://github.com/centreborelli/rpcm.git

        """

        self.__delta = 2
        nbIter = 0

        Xf = np.vstack([colN, linN]).T
        lon = -np.ones(len(Xf))
        lat = -np.ones(len(Xf))
        x0 = self.build_RFM(num=self.colNum, den=self.colDen, x=lat, y=lon, z=altN)
        y0 = self.build_RFM(num=self.linNum, den=self.linDen, x=lat, y=lon, z=altN)
        x1 = self.build_RFM(num=self.colNum, den=self.colDen, x=lat, y=lon + self.__delta, z=altN)
        y1 = self.build_RFM(num=self.linNum, den=self.linDen, x=lat, y=lon + self.__delta, z=altN)
        x2 = self.build_RFM(num=self.colNum, den=self.colDen, x=lat + self.__delta, y=lon, z=altN)
        y2 = self.build_RFM(num=self.linNum, den=self.linDen, x=lat + self.__delta, y=lon, z=altN)
        if not np.all((corr_model == 0)):
            # print(x0,y0)
            x0, y0 = self.apply_correction(corr_model, x0, y0)
            # print("____>",x0, y0)
            x1, y1 = self.apply_correction(corr_model, x1, y1)
            x2, y2 = self.apply_correction(corr_model, x2, y2)
        while not np.all((x0 - colN) ** 2 + (y0 - linN) ** 2 < self.__th):

            X0 = np.vstack([x0, y0]).T
            X1 = np.vstack([x1, y1]).T
            X2 = np.vstack([x2, y2]).T
            e1 = X1 - X0
            e2 = X2 - X0
            u = Xf - X0

            num = np.sum(np.multiply(u, e1), axis=1)
            den = np.sum(np.multiply(e1, e1), axis=1)
            a1 = np.divide(num, den)

            num = np.sum(np.multiply(u, e2), axis=1)
            den = np.sum(np.multiply(e2, e2), axis=1)
            a2 = np.divide(num, den)

            # use the coefficients a1, a2 to compute an approximation of the
            # point on the ground which in turn will give us the new X0
            lon += a1 * self.__delta
            lat += a2 * self.__delta

            # update X0, X1 and X2
            self.__delta = .1
            x0 = self.build_RFM(num=self.colNum, den=self.colDen, x=lat, y=lon, z=altN)
            y0 = self.build_RFM(num=self.linNum, den=self.linDen, x=lat, y=lon, z=altN)
            x1 = self.build_RFM(num=self.colNum, den=self.colDen, x=lat, y=lon + self.__delta, z=altN)
            y1 = self.build_RFM(num=self.linNum, den=self.linDen, x=lat, y=lon + self.__delta, z=altN)
            x2 = self.build_RFM(num=self.colNum, den=self.colDen, x=lat + self.__delta, y=lon, z=altN)
            y2 = self.build_RFM(num=self.linNum, den=self.linDen, x=lat + self.__delta, y=lon, z=altN)
            if not np.all((corr_model == 0)):
                x0, y0 = self.apply_correction(corr_model, x0, y0)
                x1, y1 = self.apply_correction(corr_model, x1, y1)
                x2, y2 = self.apply_correction(corr_model, x2, y2)
            nbIter += 1
            if nbIter > self.__nbIterMax:
                warnings.warn("RFM: nb iter max reached !!!!")
                break
        if np.size(lon) == 1 and np.size(lat) == 1:
            return lon[0], lat[0]
        else:
            return lon, lat

    def apply_correction(self, corr_model, colN, linN):

        in_array = np.array(
            [colN, linN, len(colN) * [1]]).T
        dxyArray = np.dot(in_array, corr_model.T)
        # ====================================== Apply correction in image space ==========================================#
        corr_colN = in_array[:, 0] + dxyArray[:, 0]
        corr_linN = in_array[:, 1] + dxyArray[:, 1]
        return corr_colN, corr_linN
