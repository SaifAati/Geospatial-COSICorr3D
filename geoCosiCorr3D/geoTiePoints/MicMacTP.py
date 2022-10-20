"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging

import os
import geoCosiCorr3D.georoutines.georoutines as geoRT
import geoCosiCorr3D.georoutines.file_cmd_routines as fileRT
from typing import List, Optional

from pathlib import Path
import gdal

from geoCosiCorr3D.geoConfig import cgeoCfg

geoCfg = cgeoCfg()

from geoCosiCorr3D.geoCore.geoRawTp import RawMMTP


class cMicMacTp(RawMMTP):
    __dtype = gdal.GDT_UInt16

    def __init__(self, refImgPath: str, iRasterPathList: List[str], oFolder: Optional[str] = None,
                 mode: Optional[str] = "All", scaleFactor: Optional[float] = 1 / 8, tempFolder: Optional[str] = None,
                 saveVisualizeTp: Optional[bool] = False, tpFomat: Optional[str] = "COSI-Corr"):

        config = {"method": "mmSIFT", "scale_factor": scaleFactor, "mode": mode,
                  "tp_format": tpFomat, "mm_temp_folder": tempFolder}
        super().__init__(config)

        self.iRasterPathList = iRasterPathList
        self.refImgPath = refImgPath
        self.scaleFactor = self.scale_factor
        self.saveVisualizeTp = saveVisualizeTp
        self.oFolder = oFolder
        self.tpPath = None

        if self.tp_format == "COSI-Corr":
            self.formatCosiCorr = True
        else:
            self.formatCosiCorr = False

        self.tempFolder = self.mm_temp_folder
        self.run_mm_tp()

    def run_mm_tp(self):
        refImg = os.path.join(self.tempFolder, Path(self.refImgPath).stem + ".tif")
        params = gdal.TranslateOptions(
            gdal.ParseCommandLine("-ot UInt16 -of Gtiff -co BIGTIFF=YES -co COMPRESS=LZW -b 1 -co NBITS=16"))
        gdal.Translate(destName=refImg, srcDS=self.refImgPath, options=params, outputType=self.__dtype)
        for index, img_ in enumerate(self.iRasterPathList):
            imgPath_orig = img_
            logging.info("imgPath_orig:{}".format(imgPath_orig))
            fileRT.ContentFolderDelete(self.tempFolder, exception=os.path.basename(refImg))
            # fileRT.CopyFile(inputFilePath=refImgs, outputFolder=tempPath, overWrite=False)
            logging.info(
                "-------- img:{}, : {}/{}".format(os.path.basename(img_), index + 1, len(self.iRasterPathList)))
            convImgPath = os.path.join(self.tempFolder, Path(img_).stem + ".tif")
            gdal.Translate(destName=convImgPath, srcDS=img_, options=params, outputType=self.__dtype)
            img_ = convImgPath
            rasterInfo = geoRT.RasterInfo(img_, printInfo=False)

            imageSize = self.set_img_size(img_width=rasterInfo.rasterWidth, img_height=rasterInfo.rasterHeight)
            self.run_mm_tapioca(mm_lib_path=self.MM_LIB_PATH, mode=self.mode, in_imgs_folder=self.tempFolder,
                                img_size=imageSize)
            homolPath = os.path.join(self.tempFolder, "Homol")

            if len(fileRT.FilesInDirectory(path=homolPath, displayFile=False)) != 0:
                resDic = self.mm_tps(img_i=refImg,
                                     img_j=img_,
                                     homolDirectory=homolPath,
                                     formatCosiCorr=self.formatCosiCorr)
                self.tp = resDic["DataFrame"]
                self.tpDic = resDic
                if self.oFolder != None:
                    out = fileRT.CopyFile(inputFilePath=resDic["TpFile"], outputFolder=self.oFolder)
                    logging.info("Copy to :{}".format(out))
                    self.tpPath = out
                else:
                    out = fileRT.CopyFile(inputFilePath=resDic["TpFile"], outputFolder=os.path.dirname(imgPath_orig))
                    logging.info("Copy to :{}".format(out))
                    self.tpPath = out
                if self.saveVisualizeTp:
                    # self.plot_matches(img_i=refImg, img_j=img_, matches=self.tp, tpPath=self.tpPath)
                    self.plot_matches_v2(img_i=refImg, img_j=img_, matches_file=self.tpPath)

            else:
                logging.error("No matched Tie points ")
