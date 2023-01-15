"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import numpy as np
from typing import Optional
import sys
import warnings
import affine6p
import geoCosiCorr3D.georoutines.geo_utils as geoRT
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors

from typing import List
from geoCosiCorr3D.geoCore.core_RFM import RawRFM


class ReadRFM(RawRFM):
    def __init__(self, rfm_file):
        super().__init__()
        self.rfm_file = rfm_file
        self._ingest()

    def _ingest(self):

        if self.rfm_file.endswith('xml') or self.rfm_file.endswith('XML'):
            logging.info("RFM file format: xml")
            self.RFM_Read_fromXML(self.rfm_file)
        elif self.rfm_file.lower().endswith('RPB'):
            logging.info("RFM file format: RPB")
            self.RFM_Read_fromRPB(self.rfm_file)
        elif self.rfm_file.lower().endswith(tuple(("txt", "TXT", "rpc"))):
            # print("RFM file format: txt")
            self.RFM_Read_fromTXT(self.rfm_file)
        elif self.rfm_file.endswith(tuple(('TIF', 'NTF', "tif", "ntf", "JP2"))):
            logging.info("RFM file format: Raster")
            self.RFM_Read_fromRaster(self.rfm_file)
        else:
            sys.exit(f'RFM file:{self.rfm_file}  is not valid')

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

    def RFM_Read_fromTXT(self, rfm_txt_file):

        with open(rfm_txt_file) as f:
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

    def RFM_Read_fromXML(self, rfm_xml_file):
        # TODO
        logging.info("--- Read RFM form xML ---")
        logging.info("--- Future work  ---")
        geoErrors.erNotImplemented(routineName="Read RFM from XML")
        return

    def RFM_Read_fromRPB(self, rpb_file):
        # TODO
        logging.info("--- Read RFM form RPB ---")
        logging.info("--- Future work  ---")
        geoErrors.erNotImplemented(routineName="Read RFM from RPB")
        return

    def RFM_Read_fromRaster(self, raster_file):
        ## Read the RPC coefficent from raster tag using GDAL and georoutines.
        rasterInfo = geoRT.cRasterInfo(raster_file)
        if rasterInfo.rpcs:
            rfmInfo = rasterInfo.rpcs
            # print("RFM info :", rfmInfo)
            ## Scale and offset
            self.altOff = float(rfmInfo["HEIGHT_OFF"])
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
            sys.exit(f'RPCs not found in the raster {raster_file} metadata')
        return


class RFM(ReadRFM):

    def __init__(self, rfm_file: Optional[str] = None, debug: bool = False):
        self.init_RFM()
        if rfm_file is not None:
            super().__init__(rfm_file)
        self.debug = debug
        if self.debug:
            logging.info(self.__repr__())

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
                logging.warning("INTERPOLATE FROM DEM --> TODO")
            else:
                warnings.warn("NO alt  values and no DEM: alt will be set to:{}".format(self.altOff))
                logging.warning("NO alt  values and no DEM: alt will be set to:{}".format(self.altOff))
                alt = np.ones(lon.shape) * self.altOff

        lonN = (lon - self.lonOff) / self.lonScale
        latN = (lat - self.latOff) / self.latScale
        altN = (alt - self.altOff) / self.altScale
        colN = self.build_RFM(num=self.colNum, den=self.colDen, x=latN, y=lonN, z=altN)
        linN = self.build_RFM(num=self.linNum, den=self.linDen, x=latN, y=lonN, z=altN)
        if not np.all((corrModel == 0)):
            colN, linN = self.apply_correction(corrModel=corrModel, colN=colN, linN=linN)
        if normalized == True:
            return colN, linN
        else:
            col = colN * self.colScale + self.colOff
            row = linN * self.linScale + self.linOff

            return col, row

    def Img2Ground_RFM(self, col, lin,
                       altIni: List = None,
                       demInfo: geoRT.cRasterInfo = None,
                       corrModel=np.zeros((3, 3)),
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
                logging.warning("Computing Direct model ....")
                # print("correction matrix:\n", corrModel)
            # print("colN,linN,altN", colN, linN, altN)
            lonN, latN = self.ComputeDirectModel(colN=colN, linN=linN, altN=altIniN, corrModel=corrModel)
        else:
            # print("Direct model provided in the RFM file will be used")
            lonN = self.build_RFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altIniN)
            latN = self.build_RFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altIniN)
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
            if demInfo is not None:
                alt = []

                # TODO: loop until convergence or no change in coordinates
                if isinstance(lon, np.ndarray) and isinstance(lat, np.ndarray):
                    for lonVal, latVal, altValIni in zip(lon, lat, altIni):
                        altVal = self.ExtractAlt(lonVal, latVal, demInfo)
                        if altVal == 0:
                            altVal = altValIni
                        alt.append(altVal)

                else:
                    altVal = self.ExtractAlt(lon, lat, demInfo)
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
                    lonN = self.build_RFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altN)
                    latN = self.build_RFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altN)
                lon = lonN * self.lonScale + self.lonOff
                lat = latN * self.latScale + self.latOff
                # lon, lat, alt = ApplyCorrection(lon=lon, lat=lat, alt=alt, col=col, lin=lin, modelCorr=modelCorr)
            return lon, lat, alt

        else:
            return lonN, latN, None

    def get_geoTransform(self):
        ## Compute the fooot print box
        h = int(self.linOff * 2)
        w = int(self.colOff * 2)
        BBoxPix = [[0, 0],
                   [0, h],
                   [w, h],
                   [w, 0],
                   [0, 0]]

        z = self.altOff
        lons, lats, _ = self.Img2Ground_RFM(col=[0, 0, w, w, 0],
                                            lin=[0, h, h, 0, 0],
                                            altIni=[z, z, z, z, z],
                                            normalized=False)
        BBoxMap = []
        for lon_, lat_ in zip(lons, lats):
            BBoxMap.append([lon_, lat_])

        trans = affine6p.estimate(origin=BBoxPix, convrt=BBoxMap)
        mat = trans.get_matrix()  ## Homogenious represention of the affine transformation
        geoTrans_h = np.array(mat)
        geo_transform = [mat[0][-1], mat[0][0], mat[0][1], mat[1][-1], mat[1][0], mat[1][1]]
        return geo_transform

    def get_GSD(self):

        h = self.linOff * 2
        w = self.colOff * 2

        ## Estimate GSD from RFM
        center = (int(h / 2), int(w / 2))
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

    def get_altitude_range(self, scaleFactor=1):
        """

        Args:
            scaleFactor:

        Returns:

        """
        minAlt = self.altOff - scaleFactor * self.altScale
        maxAlt = self.altOff + scaleFactor * self.altScale
        return [minAlt, maxAlt]


if __name__ == '__main__':
    #TODO add to unit/functional tests
    img = '/home/cosicorr/0-WorkSpace/3-PycharmProjects/geoCosiCorr3D/geoCosiCorr3D/Tests/3-geoOrtho_Test/Sample/Sample1/SPOT2.TIF'
    # img = '/media/cosicorr/storage/Saif/Planet_project/PlanetScope_L1As/Ridgecrest/Dove-R/Ridgecrest.3284591/L1As/20200402_183354_92_105c_1A_AnalyticMS.tif'
    rfm = RFM(img, debug=True)
    print(f'attitude range:{rfm.get_altitude_range()}')
    print(f'GSD:{rfm.get_GSD()}')
    print(f'geoTransform:{rfm.get_geoTransform()}')
