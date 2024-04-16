"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas
from osgeo import gdal

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.georoutines.file_cmd_routines as fileRT
from geoCosiCorr3D.geoCore.geoRawTp import RawMMTP
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo


class cMicMacTp(RawMMTP):

    def __init__(self, ref_img_path: str,
                 raw_img_path: str,
                 o_dir: Optional[str] = None,
                 mode: str = C.ASIFT_TP_PARAMS.MODE,
                 scale_factor: Optional[float] = None,
                 tmp_dir: Optional[str] = None,
                 plot_tps: bool = False,
                 tp_format: str = "COSI-Corr",
                 max_pts: Optional[int] = None):

        config = {"method": "mmSIFT", "scale_factor": scale_factor, "mode": mode,
                  "tp_format": tp_format, "mm_temp_folder": tmp_dir, 'max_pts': max_pts}

        super().__init__(config)

        self.raw_img_path = raw_img_path
        self.ref_img_path = ref_img_path
        self.plot_tps = plot_tps
        self.o_dir = o_dir
        if o_dir is None:
            self.o_dir = os.path.dirname(self.raw_img_path)
        self.o_tp_path = None
        self.cosi_corr_format = False
        if self.tp_format == "COSI-Corr":
            self.cosi_corr_format = True

        if self.tmp_dir is None:
            import tempfile
            with tempfile.TemporaryDirectory(dir=C.SOFTWARE.WKDIR, suffix='mm_temp_tp') as tmp_dir:
                self.tmp_dir = tmp_dir
                self.run_mm_tp()
        else:
            self.run_mm_tp()

    def generate_tmp_raster(self, input_raster_path):
        tmp_raster_path = os.path.join(self.tmp_dir, f"{Path(input_raster_path).stem}.tif")
        gdal.Translate(destName=tmp_raster_path, srcDS=input_raster_path, options=C.ASIFT_TP_PARAMS.CONV_PARAMS,
                       outputType=C.RASTER_TYPE.GDAL_UINT16)

        return tmp_raster_path

    def run_mm_tp(self):

        tmp_ref_img_path = self.generate_tmp_raster(self.ref_img_path)
        tmp_raw_img_path = self.generate_tmp_raster(self.raw_img_path)
        tmp_raster_info = cRasterInfo(tmp_raw_img_path)
        img_size = self.set_img_size(img_width=tmp_raster_info.raster_width, img_height=tmp_raster_info.raster_height)
        self.run_mm_tapioca(mm_lib_path=self.MM_LIB_PATH, mode=self.mode, in_imgs_folder=self.tmp_dir,
                            img_size=img_size)

        homol_path = os.path.join(self.tmp_dir, "Homol")
        try:
            self.tp_report_dic: Optional[Dict] = self.mm_tps(img_i=tmp_ref_img_path,
                                                             img_j=tmp_raw_img_path,
                                                             homol_dir=homol_path,
                                                             format_cosi_corr=self.cosi_corr_format,
                                                             max_tps = self.max_pts)
            if self.tp_report_dic is not None:
                self.tp: pandas.DataFrame = self.tp_report_dic["DataFrame"]
                self.o_tp_path = fileRT.CopyFile(inputFilePath=self.tp_report_dic["TpFile"], outputFolder=self.o_dir)
                logging.info("Copy to :{}".format(self.o_tp_path))

            if self.plot_tps:
                # self.plot_matches(img_i=refImg, img_j=img_, matches=self.tp, tpPath=self.tpPath)
                self.plot_matches_v2(img_i=tmp_ref_img_path, img_j=tmp_raw_img_path, matches_file=self.o_tp_path)

        except:
            logging.error("No matched Tie points ")
        return
