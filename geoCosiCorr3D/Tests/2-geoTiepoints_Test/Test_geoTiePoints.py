"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os
import geoCosiCorr3D
from geoCosiCorr3D.geoCore.constants import SOFTWARE
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from pathlib import Path

log = geoCosiCorr3DLog("Test_geoTiePoints")
folder = os.path.join(os.path.dirname(geoCosiCorr3D.__file__), "Tests/2-geoTiepoints_Test/Sample")

def Test1_geoTP_mm():
    from geoCosiCorr3D.geoTiePoints.MicMacTP import cMicMacTp
    img1 = os.path.join(folder, "BASE_IMG.TIF")
    img2 = os.path.join(folder, "TARGET_IMG.TIF")
    cMicMacTp(refImgPath=img1,
              iRasterPathList=[img2],
              scaleFactor=1 / 6,
              saveVisualizeTp=True,
              oFolder=SOFTWARE.WKDIR)
    return

def Test2_geoTP_mm():
    from geoCosiCorr3D.geoTiePoints.MicMacTP import cMicMacTp
    img1 = os.path.join(folder, "Img_i.TIF")
    img2 = os.path.join(folder, "Img_j.TIF")
    cMicMacTp(refImgPath=img1,
              iRasterPathList=[img2],
              scaleFactor=1 / 2,
              saveVisualizeTp=True,
              oFolder=SOFTWARE.WKDIR)
    return


def Test_geoTP_OPENCV():
    from geoCosiCorr3D.geoTiePoints.geoTP import cTiePoints, cTpMatching
    img1 = os.path.join(folder, "Img_i.TIF")
    img2 = os.path.join(folder, "Img_j.TIF")

    geoTp1 = cTiePoints(rasterPath=img1,
                        kernel="openCV",
                        oTpPath=os.path.join(SOFTWARE.WKDIR, Path(img1).stem + "_cv.npz"),
                        debug=True)
    geoTp2 = cTiePoints(rasterPath=img2,
                        kernel="openCV",
                        oTpPath=os.path.join(SOFTWARE.WKDIR, Path(img2).stem + "_cv.npz"),
                        debug=True)

    match = cTpMatching(tpFile_i=geoTp1.oTpPath,
                        tpFile_j=geoTp2.oTpPath,
                        img_i=img1,
                        img_j=img2,
                        oFolder=SOFTWARE.WKDIR,
                        matchFilter=False,
                        matchKernel="openCV")
    return


def Test_geoTP_geoSIFT():
    from geoCosiCorr3D.geoTiePoints.geoTP import cTiePoints, cTpMatching
    img1 = os.path.join(folder, "Img_i.TIF")
    img2 = os.path.join(folder, "Img_j.TIF")
    geoTp1 = cTiePoints(rasterPath=img1,
                        kernel="geoSIFT",
                        oTpPath=os.path.join(SOFTWARE.WKDIR, Path(img1).stem + "_geoSIFT.npz"),
                        debug=True)
    geoTp2 = cTiePoints(rasterPath=img2,
                        kernel="geoSIFT",
                        oTpPath=os.path.join(SOFTWARE.WKDIR, Path(img1).stem + "_geoSIFT.npz"),
                        debug=True)

    match = cTpMatching(tpFile_i=geoTp1.oTpPath,
                        tpFile_j=geoTp2.oTpPath,
                        img_i=img1,
                        img_j=img2,
                        oFolder=SOFTWARE.WKDIR,
                        matchFilter=True,
                        matchKernel="SIMD")
    return


if __name__ == '__main__':
    Test1_geoTP_mm()
    Test2_geoTP_mm()

    Test_geoTP_OPENCV()
    # Test_geoTP_geoSIFT()# Note take so long to finish computing all possible pts/
