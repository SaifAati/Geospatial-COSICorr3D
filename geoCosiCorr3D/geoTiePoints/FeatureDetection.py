"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import os, sys, shutil
from pathlib import Path
from typing import List, Optional
import geoCosiCorr3D.georoutines.file_cmd_routines as fileRT


def PlotTp(matchesFile, matchArray, rOrthoPath_crop, rawImg):
    import numpy as np
    import matplotlib.pyplot as plt
    import rasterio
    from geoCosiCorr3D.georoutines.geoplt_misc import GenerateColors

    np.savetxt(matchesFile, matchArray)
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("# Matches:" + str(matchArray.shape[0]))
    src1 = rasterio.open(rOrthoPath_crop)
    axs[0].imshow(src1.read(1), cmap='gray')
    src2 = rasterio.open(rawImg)
    axs[1].imshow(src2.read(1), cmap='gray')
    colors = GenerateColors(matchArray.shape[0])
    for index, match_ in enumerate(matchArray):
        axs[0].scatter(match_[0], match_[1], marker="+", s=80, color=colors[index])
        axs[1].scatter(match_[2], match_[3], marker="+", s=80, color=colors[index])
    axs[0].axes.xaxis.set_visible(False)
    axs[0].axes.yaxis.set_visible(False)
    axs[1].axes.xaxis.set_visible(False)
    axs[1].axes.yaxis.set_visible(False)
    axs[0].set_title("Ref image")
    axs[1].set_title("Target image")
    fig.savefig(os.path.join(os.path.dirname(matchesFile), Path(matchesFile).stem + ".svg"), dpi=100)
    plt.close(fig)

    return


def Feature_detection(rawImg: str,
                      rOrtho: str,
                      workspaceFolder: str,
                      scaleFactor: float = 1 / 8,
                      baseName: str = None,
                      crop: bool = False,
                      padding: Optional[List[int]] = None,
                      minTp: Optional[int] = 40,
                      tpFomat: Optional[str] = "matches",
                      maxTp: Optional[int] = None,
                      rawImgFp=None,
                      opencv: bool = False):
    """

    Args:
        rawImg:
        rOrtho:
        workspaceFolder:
        baseName:
        crop:
        padding:
        tpFomat "COSI-Corr" matches

    Returns:

    """

    import geoCosiCorr3D.geoTiePoints.MicMacTP as mmTP
    from geoCosiCorr3D.georoutines.misc import CropRefOrtho
    from pathlib import Path
    import numpy as np

    if padding is None:
        padding = [700, 700, 700, 700]
    if baseName == None:
        baseName = Path(rawImg).stem

    oFolderPair = fileRT.CreateDirectory(directoryPath=workspaceFolder,
                                         folderName=Path(rOrtho).stem + "_VS_" + baseName,
                                         cal="n")
    rOrthoPath_crop = rOrtho
    if crop:
        fp = rawImgFp
        rOrthoPath_crop = os.path.join(oFolderPair, Path(rOrtho).stem + "_crp.tif")
        if os.path.exists(rOrthoPath_crop) == False:
            CropRefOrtho(refOrthoPath=rOrtho,
                         baseImgPath=rawImg,
                         oRefOrthCrop=rOrthoPath_crop,
                         padding=padding, basefp=fp)
    matchesFile = os.path.join(oFolderPair,
                               Path(rOrthoPath_crop).stem + "_VS_" + baseName + "_matches.pts")
    nbTp = 0
    if os.path.exists(matchesFile):
        matches = np.loadtxt(matchesFile)
        nbTp = matches.shape[0]
        print("nbTp:", nbTp)

    else:

        if os.path.exists(os.path.join(oFolderPair, baseName + ".tif")) == False:
            shutil.copyfile(rawImg, os.path.join(oFolderPair, baseName + ".tif"))
        rawImg = os.path.join(oFolderPair, baseName + ".tif")

        if opencv:
            from geoCosiCorr3D.geoTiePoints.geoTP import cTiePoints, cTpMatching
            # rawImg = os.path.join(oFolderPair,Path(self.validData["ImgPath"][index]).stem+".TIF")
            # print(rOrthoPath_crop)
            if maxTp != None:
                nbMax = 200 * maxTp
            else:
                nbMax = None
            nbMax = None
            geoTp1 = cTiePoints(rasterPath=rOrthoPath_crop, kernel="openCV", debug=True, nbMax=nbMax)

            geoTp2 = cTiePoints(rasterPath=rawImg, kernel="openCV", debug=True, nbMax=nbMax)

            match = cTpMatching(tpFile_i=geoTp1.oTpPath,
                                tpFile_j=geoTp2.oTpPath,
                                img_i=rOrthoPath_crop,
                                img_j=rawImg,
                                oFolder=oFolderPair)
        else:

            try:
                mmTp_obj = mmTP.cMicMacTp(refImgPath=rOrthoPath_crop, iRasterPathList=[rawImg],
                                          scaleFactor=scaleFactor,
                                          oFolder=oFolderPair,
                                          tpFomat=tpFomat,
                                          saveVisualizeTp=False)
                if tpFomat == "COSI-Corr":
                    comment = ";"
                    matches = np.loadtxt(mmTp_obj.tpPath, comments=comment)
                else:
                    matches = np.loadtxt(mmTp_obj.tpPath)

                if matches.shape[0] < minTp and scaleFactor != 1:
                    print("=========> Try with mmTp scale factor:1 ==========")
                    mmTp_obj = mmTP.cMicMacTp(refImgPath=rOrthoPath_crop,
                                              iRasterPathList=[rawImg],
                                              scaleFactor=1,
                                              oFolder=oFolderPair,
                                              tpFomat=tpFomat,
                                              saveVisualizeTp=False)
                del mmTp_obj
            except:
                try:
                    print("=========> Try with mmTp scalefactor:1 ==========")
                    mmTp_obj = mmTP.cMicMacTp(refImgPath=rOrthoPath_crop, iRasterPathList=[rawImg],
                                              scaleFactor=scaleFactor,
                                              oFolder=oFolderPair, tpFomat=tpFomat,
                                              saveVisualizeTp=False)
                    if tpFomat == "COSI-Corr":
                        comment = ";"
                        matches = np.loadtxt(mmTp_obj.tpPath, comments=comment)
                    else:
                        matches = np.loadtxt(mmTp_obj.tpPath)

                    if matches.shape[0] < minTp:
                        mmTp_obj = mmTP.cMicMacTp(refImgPath=rOrthoPath_crop, iRasterPathList=[rawImg],
                                                  scaleFactor=1,
                                                  oFolder=oFolderPair, tpFomat=tpFomat,
                                                  saveVisualizeTp=False)
                    del mmTp_obj
                except:
                    try:
                        print("========= Try with opencv ==========")
                        from geoCosiCorr3D.geoTiePoints.geoTP import cTiePoints, cTpMatching
                        # rawImg = os.path.join(oFolderPair,Path(self.validData["ImgPath"][index]).stem+".TIF")
                        # print(rOrthoPath_crop)
                        if maxTp != None:
                            nbMax = 200 * maxTp
                        else:
                            nbMax = None
                        geoTp1 = cTiePoints(rasterPath=rOrthoPath_crop, kernel="openCV", debug=True, nbMax=nbMax)

                        geoTp2 = cTiePoints(rasterPath=rawImg, kernel="openCV", debug=True, nbMax=nbMax)

                        match = cTpMatching(
                            tpFile_i=geoTp1.oTpPath,
                            tpFile_j=geoTp2.oTpPath,
                            img_i=rOrthoPath_crop,
                            img_j=rawImg,
                            oFolder=oFolderPair)
                    except:
                        sys.exit("ERROR to extract feature points between {} and {}".format(rOrtho, rawImg))

        if tpFomat == "COSI-Corr":
            comment = ";"
            matchArray = np.loadtxt(matchesFile, comments=comment)
        else:
            matchArray = np.loadtxt(matchesFile)
        nbTp = np.loadtxt(matchesFile).shape[0]
        if maxTp != None:
            if nbTp > maxTp:
                # print("... Sampling ....")
                samplingFactor = int(np.ceil(nbTp / maxTp))
                matchArray = np.loadtxt(matchesFile)[::samplingFactor]

        PlotTp(matchesFile=matchesFile, matchArray=matchArray, rOrthoPath_crop=rOrthoPath_crop, rawImg=rawImg)

    if os.path.exists(os.path.join(oFolderPair, baseName + ".tif")):
        os.remove(os.path.join(oFolderPair, baseName + ".tif"))
    return matchesFile
