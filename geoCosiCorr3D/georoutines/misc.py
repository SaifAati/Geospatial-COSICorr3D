import os, datetime

import gdal
import numpy as np
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


# import geoRoutines.FootPrint as fpRT
# import geoRoutines.georoutines as geoRT
# import geoRoutines.Routine_Lyr as lyrRT


def CropRefOrtho(refOrthoPath, baseImgPath, oRefOrthCrop, basefp=None, padding: List[int] = None):
    """
    Compute the foot print of the two images, then we use the intersection class to calculate the intersection
    Args:
        refOrthoPath:
        baseImgPath:
        oRefOrthCrop:
        padding: [pdLeft,pdRight,pdUp,pdDown] in meter

    Returns:

    """
    if padding is None:
        padding = [0, 0, 0, 0]
        ## Check if the FP exist:
    img1, img2 = refOrthoPath, baseImgPath

    if os.path.exists(os.path.join(os.path.dirname(img1), Path(img1).stem + "_fp.geojson")):
        fpPath1 = os.path.join(os.path.dirname(img1), Path(img1).stem + "_fp.geojson")
    else:
        _, fpPath1 = fpRT.RasterFootprint(rasterPath=img1, writeFp=True)

    fpPath2 = basefp
    if basefp == None:
        if os.path.exists(os.path.join(os.path.dirname(img2), Path(img2).stem + "_fp.geojson")):
            fpPath2 = os.path.join(os.path.dirname(img2), Path(img2).stem + "_fp.geojson")
        else:
            _, fpPath2 = fpRT.RasterFootprint(rasterPath=img2, writeFp=True)

    intersect = lyrRT.Intersection(fp1=fpPath1, fp2=fpPath2, dispaly=False)
    roi = intersect.intersection
    try:
        roiCoords = roi.features[0]["geometry"]['coordinates']
    except:
        roiCoords = roi["geometry"]['coordinates']
    roiCoords = np.asarray(roiCoords).squeeze()

    # lons, lats = np.asarray(roiCoords).squeeze().T
    rOrthoInfo = geoRT.RasterInfo(refOrthoPath)
    if rOrthoInfo.EPSG_Code != 4326:

        xMap, yMap = geoRT.ConvCoordMap1ToMap2_Batch(X=roiCoords[:, 1], Y=roiCoords[:, 0],
                                                     targetEPSG=rOrthoInfo.EPSG_Code,
                                                     sourceEPSG=4326)

        if np.all(padding == 0) == False:
            xMapMin = xMap.min() - padding[0]
            xMapMax = xMap.max() + padding[1]
            yMapMax = yMap.max() + padding[2]
            yMapMin = yMap.min() - padding[3]
            xMap = [xMapMin, xMapMax, xMapMax, xMapMin, xMapMin]
            yMap = [yMapMin, yMapMin, yMapMax, yMapMax, yMapMin]

        xPix, yPix = rOrthoInfo.Map2Pixel_Batch(X=xMap, Y=yMap)
    else:
        xPix, yPix = rOrthoInfo.Map2Pixel_Batch(X=roiCoords[:, 0], Y=roiCoords[:, 1])
    pts = list(zip(xPix, yPix))
    box = np.round(geoRT.BoundingBox2D(pts)).astype(int)
    # print("x:{}, y:{}, w:{}, h:{} :unit=[pix]".format(box[0], box[1], box[2], box[3]))
    xOrigin, yOrigin = rOrthoInfo.Pixel2Map(x=box[0], y=box[1])

    rOrthoPath_crop = oRefOrthCrop

    geoTransform = list(rOrthoInfo.geoTrans)
    geoTransform[0] = xOrigin
    geoTransform[3] = yOrigin
    geoRT.WriteRaster(oRasterPath=rOrthoPath_crop, geoTransform=tuple(geoTransform),
                      arrayList=[rOrthoInfo.ImageAsArray_Subset(xOffsetMin=box[0],
                                                                xOffsetMax=box[0] + box[2],
                                                                yOffsetMin=box[1],
                                                                yOffsetMax=box[1] + box[3])],
                      epsg=rOrthoInfo.EPSG_Code, dtype=gdal.GDT_UInt16)
    return oRefOrthCrop
