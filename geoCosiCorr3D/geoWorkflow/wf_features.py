"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os
from geoCosiCorr3D.geoCore.constants import SOFTWARE, TP_DETECTION_METHODS
from pathlib import Path

method = None


def wf_features(img1, img2, tp_params, output_folder=None):
    if output_folder is None:
        output_folder = SOFTWARE.WKDIR
    method = tp_params.get('method', TP_DETECTION_METHODS.ASIFT)
    if method == TP_DETECTION_METHODS.ASIFT:
        from geoCosiCorr3D.geoTiePoints.MicMacTP import cMicMacTp

        tp = cMicMacTp(refImgPath=img1,
                       iRasterPathList=[img2],
                       scaleFactor=1 / 6,
                       saveVisualizeTp=True,
                       oFolder=output_folder)
        return tp.tpPath

    if method == TP_DETECTION_METHODS.CVTP:
        from geoCosiCorr3D.geoTiePoints.geoTP import cTiePoints, cTpMatching

        geoTp1 = cTiePoints(rasterPath=img1,
                            kernel="openCV",
                            oTpPath=os.path.join(output_folder, Path(img1).stem + "_cv.npz"),
                            debug=False)
        geoTp2 = cTiePoints(rasterPath=img2,
                            kernel="openCV",
                            oTpPath=os.path.join(output_folder, Path(img2).stem + "_cv.npz"),
                            debug=False)

        match = cTpMatching(tpFile_i=geoTp1.oTpPath,
                            tpFile_j=geoTp2.oTpPath,
                            img_i=img1,
                            img_j=img2,
                            oFolder=output_folder,
                            matchFilter=False,
                            matchKernel="openCV")
        return match.matchFile

    if method == TP_DETECTION_METHODS.GEOSIFT:
        from geoCosiCorr3D.geoTiePoints.geoTP import cTiePoints, cTpMatching
        geoTp1 = cTiePoints(rasterPath=img1,
                            kernel="geoSIFT",
                            oTpPath=os.path.join(output_folder, Path(img1).stem + "_geoSIFT.npz"),
                            debug=False)
        geoTp2 = cTiePoints(rasterPath=img2,
                            kernel="geoSIFT",
                            oTpPath=os.path.join(output_folder, Path(img1).stem + "_geoSIFT.npz"),
                            debug=False)

        match = cTpMatching(tpFile_i=geoTp1.oTpPath,
                            tpFile_j=geoTp2.oTpPath,
                            img_i=img1,
                            img_j=img2,
                            oFolder=output_folder,
                            matchFilter=True,
                            matchKernel="SIMD")
        return match.matchFile
