"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import numpy as np
import warnings
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoRSM.Interpol import Interpolate2D
from geoCosiCorr3D.geoCore.base.base_RFM import BaseRFM

class RawRFM(BaseRFM):
    __delta = 2
    __th = 1e-18
    __nbIterMax = 200

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
        numPoly3d = self.__Poly_3D(num, x, y, z)
        denPoly3d = self.__Poly_3D(den, x, y, z)

        return numPoly3d / denPoly3d

    @staticmethod
    def __Poly_3D(coefList, x, y, z):
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

    def ComputeDirectModel(self, colN, linN, altN, corrModel=np.zeros((3, 3))):
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
        if not np.all((corrModel == 0)):
            # print(x0,y0)
            x0, y0 = self.__ApplyCorrection(corrModel=corrModel, colN=x0, linN=y0)
            # print("____>",x0, y0)
            x1, y1 = self.__ApplyCorrection(corrModel=corrModel, colN=x1, linN=y1)
            x2, y2 = self.__ApplyCorrection(corrModel=corrModel, colN=x2, linN=y2)
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
            if not np.all((corrModel == 0)):
                x0, y0 = self.__ApplyCorrection(corrModel=corrModel, colN=x0, linN=y0)
                x1, y1 = self.__ApplyCorrection(corrModel=corrModel, colN=x1, linN=y1)
                x2, y2 = self.__ApplyCorrection(corrModel=corrModel, colN=x2, linN=y2)
            nbIter += 1
            if nbIter > self.__nbIterMax:
                warnings.warn("nbIterMax reached !!!!")
                break
        if np.size(lon) == 1 and np.size(lat) == 1:
            return lon[0], lat[0]
        else:
            return lon, lat

    def __ApplyCorrection(self, corrModel, colN, linN):
        # print("____RFM CORRECTION:\n", corrModel)
        inArray = np.array(
            [colN, linN, len(colN) * [1]]).T
        dxyArray = np.dot(inArray, corrModel.T)
        # ====================================== Apply correction in image space ==========================================#
        corr_colN = inArray[:, 0] + dxyArray[:, 0]
        corr_linN = inArray[:, 1] + dxyArray[:, 1]
        return corr_colN, corr_linN

    @staticmethod
    def ExtractAlt(lonVal, latVal, dem_info: geoRT.cRasterInfo, margin=3):

        epsg_code = geoRT.ComputeEpsg(lon=lonVal, lat=latVal)
        map_coord = geoRT.ConvCoordMap1ToMap2(x=latVal, y=lonVal, targetEPSG=epsg_code, sourceEPSG=4326)

        ## TODO check projections of lon,lat and DEM they should be in the same projection system ===> Done
        if epsg_code != dem_info.epsg_code:
            msg = "Reproject DEM from {}-->{}".format(dem_info.epsg_code, epsg_code)
            warnings.warn(msg)
            logging.warning(msg)

            reprj_dem_path = geoRT.ReprojectRaster(input_raster_path=dem_info.input_raster_path,
                                                   o_prj=epsg_code,
                                                   vrt=True)
            dem_info = geoRT.cRasterInfo(reprj_dem_path)

        xDemPix, yDemPix = dem_info.Map2Pixel(x=map_coord[0], y=map_coord[1])
        #         # TODO
        #         ## 1- need to verify if the window around the tie point is inside the DEM image --> Done wiht try Except
        #         ## 2- verify if h= -32767 or NaN
        #         ## 3- step value need to be a user defined parameter
        step = margin

        try:
            # demSubset = demArray[int(yDemPix) - step: int(yDemPix) + step, int(xDemPix) - step: int(xDemPix) + step]
            demSubset = dem_info.image_as_array_subset(col_off_min=int(xDemPix) - step,
                                                       col_off_max=int(xDemPix) + step,
                                                       row_off_min=int(yDemPix) - step,
                                                       row_off_max=int(yDemPix) + step)

            alt = Interpolate2D(inArray=demSubset, x=[yDemPix - (int(yDemPix) - step)],
                                y=[xDemPix - (int(xDemPix) - step)])[0]

        except:
            logging.info("Val:{:.3f},{:.3f} outside the DEM extent ".format(lonVal, latVal))
            alt = 0

        return alt