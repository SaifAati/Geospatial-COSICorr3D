"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
import logging
import sys
import re
from typing import Optional

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoCore.core_RFM import RfmModel


class ReadRFM(RfmModel):
    def __init__(self, rfm_file: Optional[str] = None):
        super().__init__()
        if rfm_file is not None:
            self.rfm_file = rfm_file
            self._ingest()

    def _ingest(self):

        file_extension = self.rfm_file.lower().split('.')[-1]

        # Mapping of file extensions to processing methods
        file_handlers = {
            'xml': self.from_xml,
            'rpb': self.from_rpb,
            'txt': self.from_txt,
            'rpc': self.from_txt,
            'tif': self.from_raster,
            'ntf': self.from_raster,
            'jp2': self.from_raster
        }

        try:
            handler = file_handlers.get(file_extension)
            if handler:
                logging.info(
                    f"{self.__class__.__name__}:Processing RFM file format: {file_extension.upper()} - {self.rfm_file}")
                handler(self.rfm_file)
            else:
                logging.info(f"{self.__class__.__name__}:Unknown RFM file format, defaulting to TXT: {self.rfm_file}")
                self.from_txt(self.rfm_file)
        except Exception as e:
            # Log the error and re-raise with a more informative message
            logging.error(f"{self.__class__.__name__}:Error processing RFM file: {self.rfm_file}, {e}")
            raise IOError(
                f"{self.__class__.__name__}:RFM file: {self.rfm_file} is not valid or could not be processed") from e

    def parse_file(self, param, lines):
        for line_ in lines:
            if re.search(re.escape(param), line_):
                parts = line_.split(":", 1)
                if len(parts) > 1:
                    try:
                        val = float(parts[1].split()[0])
                        return val
                    except ValueError:
                        raise ValueError(f"Unable to convert {parts[1].split()[0]} to float for parameter {param}")

        raise ValueError(f"ERROR in reading {param} from RFM txt file!")

    def from_txt(self, rfm_txt_file):

        with open(rfm_txt_file) as f:
            fileContent = f.read()
        lines = fileContent.split('\n')
        self.linOff = self.parse_file(param=C.RfmKeys.LINE_OFF, lines=lines)
        self.colOff = self.parse_file(param=C.RfmKeys.SAMP_OFF, lines=lines)
        self.latOff = self.parse_file(param=C.RfmKeys.LAT_OFF, lines=lines)
        self.lonOff = self.parse_file(param=C.RfmKeys.LONG_OFF, lines=lines)
        self.altOff = self.parse_file(param=C.RfmKeys.HEIGHT_OFF, lines=lines)

        self.linScale = self.parse_file(param=C.RfmKeys.LINE_SCALE, lines=lines)
        self.colScale = self.parse_file(param=C.RfmKeys.SAMP_SCALE, lines=lines)
        self.latScale = self.parse_file(param=C.RfmKeys.LAT_SCALE, lines=lines)
        self.lonScale = self.parse_file(param=C.RfmKeys.LONG_SCALE, lines=lines)
        self.altScale = self.parse_file(param=C.RfmKeys.HEIGHT_SCALE, lines=lines)

        ### Inverse model
        for i in range(self.NB_NUM_COEF):
            self.linNum[i] = self.parse_file(param=f'{C.RfmKeys.LINE_NUM_COEFF}_{i + 1}:', lines=lines)
            self.linDen[i] = self.parse_file(param=f'{C.RfmKeys.LINE_DEN_COEFF}_{i + 1}:', lines=lines)

            self.colNum[i] = self.parse_file(param=f'{C.RfmKeys.SAMP_NUM_COEFF}_{i + 1}:', lines=lines)
            self.colDen[i] = self.parse_file(param=f'{C.RfmKeys.SAMP_DEN_COEFF}_{i + 1}:', lines=lines)

        # TODO: check for direct model
        return

    def from_xml(self, rfm_xml_file):
        # TODO
        logging.info("--- Read RFM form xML ---")
        geoErrors.erNotImplemented(routineName="Read RFM from XML")
        return

    def from_rpb(self, rpb_file):
        # TODO
        logging.info("--- Read RFM form RPB ---")
        geoErrors.erNotImplemented(routineName="Read RFM from RPB")
        return

    def from_raster(self, raster_file):
        # Read the RPC coefficent from raster tag using GDAL and georoutines.
        raster_info = geoRT.cRasterInfo(raster_file)
        if raster_info.rpcs:
            rfm_info = raster_info.rpcs
            self.altOff = float(rfm_info[C.RfmKeys.HEIGHT_OFF])
            self.altScale = float(rfm_info[C.RfmKeys.HEIGHT_SCALE])

            self.latOff = float(rfm_info[C.RfmKeys.LAT_OFF])
            self.latScale = float(rfm_info[C.RfmKeys.LAT_SCALE])

            self.lonOff = float(rfm_info[C.RfmKeys.LONG_OFF])
            self.lonScale = float(rfm_info[C.RfmKeys.LONG_SCALE])

            self.linOff = float(rfm_info[C.RfmKeys.LINE_OFF])
            self.linScale = float(rfm_info[C.RfmKeys.LINE_SCALE])

            self.colOff = float(rfm_info[C.RfmKeys.SAMP_OFF])
            self.colScale = float(rfm_info[C.RfmKeys.SAMP_SCALE])

            ## Inverse model
            self.linNum = list(map(float, rfm_info[C.RfmKeys.LINE_NUM_COEFF].split()))
            self.linDen = list(map(float, rfm_info[C.RfmKeys.LINE_DEN_COEFF].split()))
            self.colNum = list(map(float, rfm_info[C.RfmKeys.SAMP_NUM_COEFF].split()))
            self.colDen = list(map(float, rfm_info[C.RfmKeys.SAMP_DEN_COEFF].split()))

            ## Direct model
            if C.RfmKeys.LON_NUM_COEFF in rfm_info:
                self.lonNum = list(map(float, rfm_info[C.RfmKeys.LON_NUM_COEFF].split()))
                self.lonDen = list(map(float, rfm_info[C.RfmKeys.LON_DEN_COEFF].split()))
                self.latNum = list(map(float, rfm_info[C.RfmKeys.LAT_NUM_COEFF].split()))
                self.latDen = list(map(float, rfm_info[C.RfmKeys.LAT_DEN_COEFF].split()))
        else:
            sys.exit(f'RPCs not found in the raster {raster_file} metadata')
        return
