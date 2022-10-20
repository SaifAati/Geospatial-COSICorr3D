"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import sys
import warnings

import numpy as np
import affine6p

import geoCosiCorr3D.georoutines.georoutines as geoRT
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
from geoCosiCorr3D.geoRSM.Interpol import Interpolate2D
from typing import List


class cRFMModel:
    __delta = 2
    __th = 1e-18
    __nbIterMax = 200

    def __init__(self, RFMFile=None, debug=False):
        """

        Args:
            RFMFile:
            debug:
        """
        self.RFM_init()
        # print(self.__dict__)
        if RFMFile != None:
            self.RFM_Read(RFMFile=RFMFile)
        self.debug = debug
        if self.debug:
            print(self.__repr__())

    def RFM_init(self):
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

    def RFM_Read(self, RFMFile):
        """

        Args:
            RFMFile:

        Returns:

        """
        self.RFMFile = RFMFile
        if RFMFile.endswith('xml') or RFMFile.endswith('XML'):
            print("RFM file format: xml")
            self.RFM_Read_fromXML(RFMXML=RFMFile)
        elif RFMFile.lower().endswith('RPB'):
            print("RFM file format: RPB")
            self.RFM_Read_fromRPB(RFMRPB=RFMFile)
        elif RFMFile.lower().endswith(tuple(("txt", "TXT", "rpc"))):
            # print("RFM file format: txt")
            self.RFM_Read_fromTXT(RFMTXT=RFMFile)
        elif RFMFile.endswith(tuple(('TIF', 'NTF', "tif", "ntf", "JP2"))):
            # print("RFM file format: Raster")
            self.RFM_Read_fromRaster(rasterFile=RFMFile)
        else:

            sys.exit("RFM file invalid !!")

        return

    def parse_file(self, param, lines):
        from re import search
        val = None
        # print(param)
        for line_ in lines:
            if search(param, line_):
                val = float(line_.split(":")[1].split()[0])

        if val == None:
            msg = "ERROR in reading " + param + " from RFM txt file!"
            sys.exit(msg)
        return val

    def RFM_Read_fromTXT(self, RFMTXT):

        with open(RFMTXT) as f:
            fileContent = f.read()
        lines = fileContent.split('\n')
        self.linOff = self.parse_file(param="LINE_OFF", lines=lines)
        self.colOff = self.parse_file(param="SAMP_OFF", lines=lines)
        self.latOff = self.parse_file(param="LAT_OFF", lines=lines)
        self.lonOff = self.parse_file(param="LONG_OFF", lines=lines)
        self.altOff = self.parse_file(param="HEIGHT_SCALE", lines=lines)
        self.linScale = self.parse_file(param="LINE_SCALE", lines=lines)
        self.colScale = self.parse_file(param="SAMP_SCALE", lines=lines)
        self.latScale = self.parse_file(param="LAT_SCALE", lines=lines)
        self.lonScale = self.parse_file(param="LONG_SCALE", lines=lines)
        self.altScale = self.parse_file(param="HEIGHT_SCALE", lines=lines)

        ### Inverse model
        for i in range(20):
            self.linNum[i] = self.parse_file(param="LINE_NUM_COEFF_" + str(i + 1) + ":", lines=lines)
            self.linDen[i] = self.parse_file(param="LINE_DEN_COEFF_" + str(i + 1) + ":", lines=lines)
            self.colNum[i] = self.parse_file(param="SAMP_NUM_COEFF_" + str(i + 1) + ":", lines=lines)
            self.colDen[i] = self.parse_file(param="SAMP_DEN_COEFF_" + str(i + 1) + ":", lines=lines)
        # print(self.linNum)
        # TODO: check for direct model
        return

    def RFM_Read_fromXML(self, RFMXML):
        # TODO
        print("--- Read RFM form xML ---")
        print("--- Future work  ---")
        geoErrors.erNotImplemented(routineName="Read RFM from XML")
        return

    def RFM_Read_fromRPB(self, RFMRPB):
        # TODO
        print("--- Read RFM form RPB ---")
        print("--- Future work  ---")
        geoErrors.erNotImplemented(routineName="Read RFM from RPB")
        return

    def RFM_Read_fromRaster(self, rasterFile):
        ## Read the RPC coefficent from raster tag using GDAL and georoutines.
        self.rasterInfo = geoRT.RasterInfo(rasterFile)
        rasterInfo = self.rasterInfo
        if rasterInfo.rpcs:
            rfmInfo = rasterInfo.rpcs
            # print("RFM info :", rfmInfo)
            ## Scale and offset
            self.altOff = float(rfmInfo["HEIGHT_OFF"])
            # print("self.altOff:",self.altOff)
            self.altScale = float(rfmInfo["HEIGHT_SCALE"])

            self.latOff = float(rfmInfo["LAT_OFF"])
            self.latScale = float(rfmInfo["LAT_SCALE"])
            self.lonOff = float(rfmInfo["LONG_OFF"])
            self.lonScale = float(rfmInfo["LONG_SCALE"])

            self.linOff = float(rfmInfo["LINE_OFF"])
            self.linScale = float(rfmInfo["LINE_SCALE"])
            self.colOff = float(rfmInfo["SAMP_OFF"])
            self.colScale = float(rfmInfo["SAMP_SCALE"])

            ## Inverse model
            self.linNum = list(map(float, rfmInfo['LINE_NUM_COEFF'].split()))
            self.linDen = list(map(float, rfmInfo['LINE_DEN_COEFF'].split()))
            self.colNum = list(map(float, rfmInfo['SAMP_NUM_COEFF'].split()))
            self.colDen = list(map(float, rfmInfo['SAMP_DEN_COEFF'].split()))

            ## Direct model
            if 'LON_NUM_COEFF' in rfmInfo:
                self.lonNum = list(map(float, rfmInfo['LON_NUM_COEFF'].split()))
                self.lonDen = list(map(float, rfmInfo['LON_DEN_COEFF'].split()))
                self.latNum = list(map(float, rfmInfo['LAT_NUM_COEFF'].split()))
                self.latDen = list(map(float, rfmInfo['LAT_DEN_COEFF'].split()))
        else:

            sys.exit("RPCs not found in the metadata")
        return

    def __buildRFM(self, x, y, z, num, den):
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

    def __Poly_3D(self, coefList, x, y, z):
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

    def Ground2Img_RFM(self, lon, lat, alt: List = None, normalized=False, demInfo=None, corrModel=np.zeros((3, 3))):
        """
        Apply inverse RFM model to convert Ground coordinates to image coordinates

        Args:
            lon: longitude(s) of the input 3D point(s) : float or list
            lat: latitude(s) of the input 3D point(s) : float or list
            alt: altitude(s) of the input 3D point(s) : float or list
            corrModel

        Returns:

            float or list: horizontal image coordinate(s) (column index, ie x)
            float or list: vertical image coordinate(s) (row index, ie y)

        """
        if alt is None:
            alt = []
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        if np.array(alt).any() == True:
            alt = np.asarray(alt)
        else:
            if demInfo != None:
                warnings.warn("INTERPOLATE FROM DEM --> TODO")
            else:
                warnings.warn("NO alt  values and no DEM: alt will be set to:{}".format(self.altOff))
                alt = np.ones(lon.shape) * self.altOff

        lonN = (lon - self.lonOff) / self.lonScale
        latN = (lat - self.latOff) / self.latScale
        altN = (alt - self.altOff) / self.altScale
        colN = self.__buildRFM(num=self.colNum, den=self.colDen, x=latN, y=lonN, z=altN)
        linN = self.__buildRFM(num=self.linNum, den=self.linDen, x=latN, y=lonN, z=altN)
        if not np.all((corrModel == 0)):
            colN, linN = self.__ApplyCorrection(corrModel=corrModel, colN=colN, linN=linN)
        if normalized == True:
            return colN, linN
        else:
            col = colN * self.colScale + self.colOff
            row = linN * self.linScale + self.linOff

            return col, row

    def Img2Ground_RFM_(self, col, lin, altIni: List = None, demInfo=None, corrModel=np.zeros((3, 3)),
                        normalized=False):
        """
        Apply direct RFM model to convert image coordinates to ground coordinates

        Args:
            col: x-image coordinate(s) of the input point(s) : float or list
            lin: y-image coordinate(s) of the input point(s) : float or list
            altIni: altitude(s) of the input point(s) : float or list
            normalized:

        Returns: float or list: longitude(s) && float or list: latitude(s)
        """
        if altIni is None:
            altIni = None
        if isinstance(altIni, list):
            if len(altIni) == 0:
                if isinstance(col, list) and isinstance(lin, list):
                    altIni = len(col) * [self.altOff]
                else:
                    altIni = self.altOff
            elif len(altIni) != len(col) or len(altIni) != len(lin):
                ValueError("Invalid Initial Altitude values !")

        col = np.asarray(col)
        lin = np.asarray(lin)
        altIni_ = np.asarray(altIni)

        # Normalize input image coordinates
        colN = (col - self.colOff) / self.colScale
        linN = (lin - self.linOff) / self.linScale
        altIniN = (altIni_ - self.altOff) / self.altScale

        if self.lonNum == [np.nan] * 20:
            if self.debug:
                print("Computing Direct model ....")
                print("correction matrix:\n", corrModel)
                # print("colN,linN,altN", colN, linN, altN)
            lonN, latN = self.ComputeDirectModel(colN=colN, linN=linN, altN=altIniN, corrModel=corrModel)
        else:
            # print("Direct model provided in the RFM file will be used")
            lonN = self.__buildRFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altIniN)
            latN = self.__buildRFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altIniN)
        if not normalized:
            lon = lonN * self.lonScale + self.lonOff
            lat = latN * self.latScale + self.latOff
            # print(lon, lat, altIni)
            # ==== Apply correction if exist =====
            # if not np.all((modelCorr == 0)):
            #     lon, lat, altIni = ApplyCorrection(lon=lon, lat=lat, alt=altIni, col=col, lin=lin, modelCorr=modelCorr)

            if isinstance(altIni, list):
                alt = altIni
            else:
                alt = altIni

            ### Here we will use the computed lon & lat to interpolate the alt from the DEM if exist
            if demInfo != None:
                alt = []
                # print("______________________________")
                # TODO: loop untill convergence or no change in coordinates
                if isinstance(lon, np.ndarray) and isinstance(lat, np.ndarray):
                    for lonVal, latVal, altValIni in zip(lon, lat, altIni):
                        altVal = ExtractAlt(lonVal, latVal, demInfo)

                        if altVal == 0:
                            altVal = altValIni
                        alt.append(altVal)
                else:
                    altVal = ExtractAlt(lon, lat, demInfo)
                    if altVal == 0:
                        altVal = altIni
                    alt = altVal
                alt = np.asarray(alt)
                # Normalize input image coordinates
                colN = (col - self.colOff) / self.colScale
                linN = (lin - self.linOff) / self.linScale
                altN = (alt - self.altOff) / self.altScale
                if self.lonNum == [np.nan] * 20:
                    # print("Computing Direct model ....")
                    # print("colN,linN,altN", colN, linN, altN)
                    lonN, latN = self.ComputeDirectModel(colN=colN, linN=linN, altN=altN, corrModel=corrModel)
                else:
                    # print("Direct model provided in the RFM file will be used")
                    lonN = self.__buildRFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altN)
                    latN = self.__buildRFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altN)
                lon = lonN * self.lonScale + self.lonOff
                lat = latN * self.latScale + self.latOff
                # lon, lat, alt = ApplyCorrection(lon=lon, lat=lat, alt=alt, col=col, lin=lin, modelCorr=modelCorr)
            return lon, lat, alt

        else:
            return lonN, latN, None

    def Img2Ground_RFM(self, col, lin, altIni: List = None, demInfo=None, corrModel=np.zeros((3, 3)), normalized=False):
        """
        Apply direct RFM model to convert image coordinates to ground coordinates

        Args:
            col: x-image coordinate(s) of the input point(s) : float or list
            lin: y-image coordinate(s) of the input point(s) : float or list
            altIni: altitude(s) of the input point(s) : float or list
            normalized:

        Returns: float or list: longitude(s) && float or list: latitude(s)
        """
        if altIni is None:
            altIni = []
        if isinstance(altIni, list):
            if len(altIni) == 0:
                if isinstance(col, list) and isinstance(lin, list):
                    altIni = len(col) * [self.altOff]
                else:
                    altIni = self.altOff
            elif len(altIni) != len(col) or len(altIni) != len(lin):
                ValueError("Invalid Initial Altitude values !")

        col = np.asarray(col)
        lin = np.asarray(lin)
        altIni_ = np.asarray(altIni)

        # Normalize input image coordinates
        colN = (col - self.colOff) / self.colScale
        linN = (lin - self.linOff) / self.linScale
        altIniN = (altIni_ - self.altOff) / self.altScale

        if self.lonNum == [np.nan] * 20:
            if self.debug:
                print("Computing Direct model ....")
                print("correction matrix:\n", corrModel)
            # print("colN,linN,altN", colN, linN, altN)
            lonN, latN = self.ComputeDirectModel(colN=colN, linN=linN, altN=altIniN, corrModel=corrModel)
        else:
            # print("Direct model provided in the RFM file will be used")
            lonN = self.__buildRFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altIniN)
            latN = self.__buildRFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altIniN)
        if not normalized:
            lon = lonN * self.lonScale + self.lonOff
            lat = latN * self.latScale + self.latOff
            # print(lon, lat, altIni)
            # ==== Apply correction if exist =====
            # if not np.all((modelCorr == 0)):
            #     lon, lat, altIni = ApplyCorrection(lon=lon, lat=lat, alt=altIni, col=col, lin=lin, modelCorr=modelCorr)

            if isinstance(altIni, list):
                alt = altIni
            else:
                alt = altIni

            ### Here we will use the computed lon & lat to interpolate the alt from the DEM if exist
            if demInfo != None:
                alt = []
                # print("______________________________")
                # TODO: loop until convergence or no change in coordinates
                if isinstance(lon, np.ndarray) and isinstance(lat, np.ndarray):
                    for lonVal, latVal, altValIni in zip(lon, lat, altIni):
                        altVal = ExtractAlt(lonVal, latVal, demInfo)

                        if altVal == 0:
                            altVal = altValIni
                        alt.append(altVal)
                else:
                    altVal = ExtractAlt(lon, lat, demInfo)
                    if altVal == 0:
                        altVal = altIni
                    alt = altVal
                alt = np.asarray(alt)
                # Normalize input image coordinates
                colN = (col - self.colOff) / self.colScale
                linN = (lin - self.linOff) / self.linScale
                altN = (alt - self.altOff) / self.altScale
                if self.lonNum == [np.nan] * 20:
                    # print("Computing Direct model ....")
                    # print("colN,linN,altN", colN, linN, altN)
                    lonN, latN = self.ComputeDirectModel(colN=colN, linN=linN, altN=altN, corrModel=corrModel)
                else:
                    # print("Direct model provided in the RFM file will be used")
                    lonN = self.__buildRFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altN)
                    latN = self.__buildRFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altN)
                lon = lonN * self.lonScale + self.lonOff
                lat = latN * self.latScale + self.latOff
                # lon, lat, alt = ApplyCorrection(lon=lon, lat=lat, alt=alt, col=col, lin=lin, modelCorr=modelCorr)
            return lon, lat, alt

        else:
            return lonN, latN, None

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
        x0 = self.__buildRFM(num=self.colNum, den=self.colDen, x=lat, y=lon, z=altN)
        y0 = self.__buildRFM(num=self.linNum, den=self.linDen, x=lat, y=lon, z=altN)
        x1 = self.__buildRFM(num=self.colNum, den=self.colDen, x=lat, y=lon + self.__delta, z=altN)
        y1 = self.__buildRFM(num=self.linNum, den=self.linDen, x=lat, y=lon + self.__delta, z=altN)
        x2 = self.__buildRFM(num=self.colNum, den=self.colDen, x=lat + self.__delta, y=lon, z=altN)
        y2 = self.__buildRFM(num=self.linNum, den=self.linDen, x=lat + self.__delta, y=lon, z=altN)
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
            x0 = self.__buildRFM(num=self.colNum, den=self.colDen, x=lat, y=lon, z=altN)
            y0 = self.__buildRFM(num=self.linNum, den=self.linDen, x=lat, y=lon, z=altN)
            x1 = self.__buildRFM(num=self.colNum, den=self.colDen, x=lat, y=lon + self.__delta, z=altN)
            y1 = self.__buildRFM(num=self.linNum, den=self.linDen, x=lat, y=lon + self.__delta, z=altN)
            x2 = self.__buildRFM(num=self.colNum, den=self.colDen, x=lat + self.__delta, y=lon, z=altN)
            y2 = self.__buildRFM(num=self.linNum, den=self.linDen, x=lat + self.__delta, y=lon, z=altN)
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

    def GeoTransfromFromRFM(self, rasterPath):
        """

        Args:
            rasterPath:

        Returns:

        """

        ## Compute the fooot print box
        self.rasterinfo = geoRT.RasterInfo(rasterPath)
        BboxPix = [[0, 0],
                   [0, self.rasterInfo.rasterHeight],
                   [self.rasterInfo.rasterWidth, self.rasterInfo.rasterHeight],
                   [self.rasterInfo.rasterWidth, 0],
                   [0, 0]]
        w = self.rasterInfo.rasterWidth
        h = self.rasterInfo.rasterHeight
        z = self.altOff
        lons, lats, _ = self.Img2Ground_RFM(col=[0, 0, w, w, 0], lin=[0, h, h, 0, 0], altIni=[z, z, z, z, z],
                                            normalized=False)
        self.BboxMap = []
        for lon_, lat_ in zip(lons, lats):
            self.BboxMap.append([lon_, lat_])

        trans = affine6p.estimate(origin=BboxPix, convrt=self.BboxMap)
        mat = trans.get_matrix()  ## Homogenious represention of the affine transformation
        self.geoTrans_h = np.array(mat)
        self.geoTrans = [mat[0][-1], mat[0][0], mat[0][1], mat[1][-1], mat[1][0], mat[1][1]]
        return

    def ComputeGSDFromRFM(self, rasterPath):
        """

        Args:
            rasterPath:

        Returns:

        """

        rasterInfo = geoRT.RasterInfo(rasterPath, printInfo=False)

        ## Estimate GSD from RFM
        center = (int(rasterInfo.rasterHeight / 2), int(rasterInfo.rasterWidth / 2))
        center_plus = (center[0] + 1, center[1] + 1)
        prjCenter = self.Img2Ground_RFM(col=center[1], lin=center[0])
        prjCenter_plus = self.Img2Ground_RFM(col=center_plus[1], lin=center_plus[0])
        ## Estimate the UTM
        epsgCode = geoRT.ComputeEpsg(lon=prjCenter[0], lat=prjCenter[1])
        ## Convert tot UTM projection
        centerCoords = geoRT.ConvCoordMap1ToMap2_Batch(X=[prjCenter[1], prjCenter_plus[1]],
                                                       Y=[prjCenter[0], prjCenter_plus[0]],
                                                       targetEPSG=epsgCode)
        xGSD = np.abs(centerCoords[0][0] - centerCoords[0][1])
        yGSD = np.abs(centerCoords[1][0] - centerCoords[1][1])
        return (xGSD, yGSD)

    def AltitudeRange(self, scaleFactor=1):
        """

        Args:
            scaleFactor:

        Returns:

        """
        minAlt = self.altOff - scaleFactor * self.altScale
        maxAlt = self.altOff + scaleFactor * self.altScale
        return [minAlt, maxAlt]

    def __ApplyCorrection(self, corrModel, colN, linN):
        # print("____RFM CORRECTION:\n", corrModel)
        inArray = np.array(
            [colN, linN, len(colN) * [1]]).T
        dxyArray = np.dot(inArray, corrModel.T)
        # ====================================== Apply correction in image space ==========================================#
        corr_colN = inArray[:, 0] + dxyArray[:, 0]
        corr_linN = inArray[:, 1] + dxyArray[:, 1]
        return corr_colN, corr_linN

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



# ToDO: Implement the bellow functions
def ApplyCorrection(lon, lat, alt, col, lin, modelCorr):
    if isinstance(lon, np.ndarray) and isinstance(lat, np.ndarray):
        # for lonVal, latVal, altValIni in zip(lon, lat, altIni):
        #     altVal = ExtractAlt(lonVal, latVal, demInfo)
        utmEPSG = geoRT.ComputeEpsg(lon=lon[0], lat=lat[0])
        # print(utmEPSG)
        utmGround = geoRT.ConvCoordMap1ToMap2_Batch(X=list(lat),
                                                    Y=list(lon),
                                                    Z=list(alt),
                                                    targetEPSG=utmEPSG)
        XYZ_RFM_UTM_Array = np.asarray(utmGround).T
        # print(XYZ_RFM_UTM_Array)
        xyPix_array = np.ones((len(lin), 3))
        xyPix_array[:, 0] = col
        xyPix_array[:, 1] = lin

        XYZ_RFM_UTM_Array_corr = XYZ_RFM_UTM_Array + np.dot(xyPix_array, modelCorr)
        # print(XYZ_RFM_UTM_Array_corr)
        geoGround = geoRT.ConvCoordMap1ToMap2_Batch(X=XYZ_RFM_UTM_Array_corr[:, 0],
                                                    Y=XYZ_RFM_UTM_Array_corr[:, 1],
                                                    Z=XYZ_RFM_UTM_Array_corr[:, 2],
                                                    sourceEPSG=utmEPSG,
                                                    targetEPSG=4326)
        geoGroundCorr = np.asarray(geoGround).T
        lat = geoGroundCorr[:, 0]
        lon = geoGroundCorr[:, 1]
        alt = geoGroundCorr[:, 2]
    else:
        utmEPSG = geoRT.ComputeEpsg(lon=lon, lat=lat)
        # print(utmEPSG)
        utmGround = geoRT.ConvCoordMap1ToMap2(x=lat,
                                              y=lon,
                                              z=alt,
                                              targetEPSG=utmEPSG)
        XYZ_RFM_UTM_Array = np.asarray(utmGround).T
        # print(XYZ_RFM_UTM_Array)
        xyPix_array = np.array([col, lin, 1])

        XYZ_RFM_UTM_Array_corr = XYZ_RFM_UTM_Array + np.dot(xyPix_array, modelCorr)
        # print(XYZ_RFM_UTM_Array_corr)
        geoGround = geoRT.ConvCoordMap1ToMap2_Batch(X=XYZ_RFM_UTM_Array_corr[:, 0],
                                                    Y=XYZ_RFM_UTM_Array_corr[:, 1],
                                                    Z=XYZ_RFM_UTM_Array_corr[:, 2],
                                                    sourceEPSG=utmEPSG,
                                                    targetEPSG=4326)
        geoGroundCorr = np.asarray(geoGround).T
        lat = geoGroundCorr[:, 0]
        lon = geoGroundCorr[:, 1]
        alt = geoGroundCorr[:, 2]

    return lon, lat, alt


def ExtractAlt(lonVal, latVal, demInfo, margin=3):
    epsgCode = geoRT.ComputeEpsg(lon=lonVal, lat=latVal)
    mapCoord = geoRT.ConvCoordMap1ToMap2(x=latVal, y=lonVal, targetEPSG=epsgCode, sourceEPSG=4326)

    ## TODO check projections of lon,lat and DEM they should be in the same projection system ===> Done
    if epsgCode != demInfo.EPSG_Code:
        msg = "Reproject DEM from {}-->{}".format(demInfo.EPSG_Code, epsgCode)
        warnings.warn(msg)
        prjDemPath = geoRT.ReprojectRaster(iRasterPath=demInfo.rasterPath, oPrj=epsgCode, vrt=True)
        demInfo = geoRT.RasterInfo(prjDemPath)

    xDemPix, yDemPix = demInfo.Map2Pixel(x=mapCoord[0], y=mapCoord[1])

    #         # TODO
    #         ## 1- need to verify if the window around the tie point is inside the DEM image --> Done wiht try Except
    #         ## 2- verify if h= -32767 or NaN
    #         ## 3- step value need to be a user defined parameter
    step = margin

    try:
        # demSubset = demArray[int(yDemPix) - step: int(yDemPix) + step, int(xDemPix) - step: int(xDemPix) + step]
        demSubset = demInfo.ImageAsArray_Subset(xOffsetMin=int(xDemPix) - step,
                                                xOffsetMax=int(xDemPix) + step,
                                                yOffsetMin=int(yDemPix) - step,
                                                yOffsetMax=int(yDemPix) + step)
        alt = Interpolate2D(inArray=demSubset, x=[yDemPix - (int(yDemPix) - step)],
                            y=[xDemPix - (int(xDemPix) - step)])[0]

    except:
        print("Val:{:.3f},{:.3f} outside the DEM extent ".format(lonVal, latVal))
        alt = 0

    return alt


def Box(imgPath, roiPath):
    """
    Compute the BoundingBox2D in pixel using ROI in WGS projection
    Args:
        imgPath:
        roiPath:

    Returns:

    """

    rasterRFM = cRFMModel(RFMFile=imgPath, debug=False)
    roi = geoRT.ReadGeojson(geojsonPath=roiPath)
    try:
        roiCoords = roi.features[0]["geometry"]['coordinates']
    except:
        roiCoords = roi["geometry"]['coordinates']

    lons, lats = np.asarray(roiCoords).squeeze().T
    alt = len(lons) * [rasterRFM.altOff]
    ## Define the bouding box
    x, y = rasterRFM.Ground2Img_RFM(lons, lats, alt)
    pts = list(zip(x, y))
    box = np.round(geoRT.BoundingBox2D(pts)).astype(int)
    print("x:{}, y:{}, w:{}, h:{}".format(box[0], box[1], box[2], box[3]))
    return box
