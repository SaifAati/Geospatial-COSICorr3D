"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import math
import os.path
import warnings

import affine6p
import geoCosiCorr3D.georoutines.geo_utils as geoRT
import numpy as np
import pandas


def EstimateGeoTransformation(pixObs, groundObs):
    trans = affine6p.estimate(np.array([groundObs[:, 0], groundObs[:, 1]]).T, pixObs)
    trans_ = np.asarray(trans.get_matrix())

    a0 = trans_[0, -1]
    a1 = trans_[0, 0]
    a2 = trans_[0, 1]
    b0 = trans_[1, -1]
    b1 = trans_[1, 0]
    b2 = trans_[1, 1]

    return trans_


def DispOptReport(reportPath, snrTh=0.9, debug=False, plotError=True):
    """

    :param reportPath:
    :param snrTh:
    :return:
    """
    df = pandas.read_csv(reportPath)

    print(df)
    totalNbLoop = list(df["nbLoop"])[-1]
    # print(totalNbLoop)
    loopList = []
    rmseList = []
    avgErrorList = []
    for loop_ in range(totalNbLoop + 1):
        if debug:
            print("------ Loop:{} -------".format(loop_))
        itemList = []
        dxPixList = []
        dyPixList = []
        snrList = []
        dxList = []
        dyList = []
        for item, dxPix_, dyPix_, snr_, dx_, dy_ in zip(list(df["nbLoop"]), list(df["dxPix"]), list(df["dyPix"]),
                                                        list(df["SNR"]), list(df["dx"]), list(df["dy"])):
            if item == loop_:
                itemList.append(item)
                dxPixList.append(dxPix_)
                dyPixList.append(dyPix_)
                snrList.append(snr_)
                dxList.append(dx_)
                dyList.append(dy_)

        nanList = [item_ for item_ in snrList if item_ == 0]
        snrThList = [item_ for item_ in snrList if item_ > snrTh]
        dxPixAvg = np.nanmean(np.asarray(dxPixList))
        dyPixAvg = np.nanmean(np.asarray(dyPixList))

        dxPixRMSE = np.nanstd(np.asarray(dxPixList))
        dyPixRMSE = np.nanstd(np.asarray(dyPixList))

        xyErrorAvg = np.sqrt(dxPixAvg ** 2 + dyPixAvg ** 2)
        xyRMSE = np.sqrt(dxPixRMSE ** 2 + dyPixRMSE ** 2)
        if debug:
            print("#GCPs:{} --> #NaNs:{} ; #snrTh >{}:{}".format(len(itemList), len(nanList), snrTh, len(snrThList)))
            print("dxPixAvg:{}  , xRMSE:{}".format("{0:.4f}".format(dxPixAvg),

                                                   "{0:.2f}".format(dxPixRMSE)))
            print("dyPixAvg:{}  , yRMSE:{}".format("{0:.4f}".format(dyPixAvg),

                                                   "{0:.2f}".format(dyPixRMSE)))
            print("xyErrorAvg:{}  , xyRMSE:{}".format("{0:.4f}".format(xyErrorAvg),

                                                      "{0:.2f}".format(xyRMSE)))
        loopList.append(loop_)
        rmseList.append(xyRMSE)
        avgErrorList.append(xyErrorAvg)
    indexMin = np.argmin(avgErrorList)
    # if debug:
    print("Loop of Min Error:{} --> RMSE:{:.3f} , avgErr:{:.3f}".format(loopList[indexMin], np.min(rmseList),
                                                                        np.min(avgErrorList)))
    if plotError:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        fig, ax = plt.subplots()

        ax.plot(loopList, rmseList, c="r", linestyle="--", marker="o", label="RMSE [pix]")

        ax.plot(loopList, avgErrorList, c="g", linestyle="-", marker="o", label="meanErr [pix]")

        ax.grid()
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.tick_params(which='both', width=2, direction="in")
        ax.set_xlabel('#iterations')
        ax.set_ylabel("Error [pix]")
        # plt.show()
        fig.savefig(os.path.join(os.path.dirname(reportPath), "CoregistrationError.svg"), dpi=400)

    return loopList[indexMin], totalNbLoop, np.min(avgErrorList)


def GetPatchDims(rOrthoInfo, gcpCorr, winSz, tol=1e-3):
    """
    Determining the Reference Image subset surrounding the gcp in the Reference Image.
    Determining the Geographic/UTM coordinates of the subset corners of the on-coming gcpPatch
    Args:
        rOrthoInfo: reference Ortho image information : geoRT.RasterInfo()
        gcpCorr: [xMap_gcp,yMap_gcp]
        winSz: [fullwinszX, fullwinszY] : patch dimension
        tol: 1e-3 : offset to be added for subpixel accuracy

    Returns:
        Returns: Subset pixel corners coordinates of the refImage and gcpPatch dimensions :
            rOrthoPatchDimsArray = [1, xRefMin_pix, xRefMax_pix, yRefMin_pix, yRefMax_pix]
            rawImgPatchX0Y0Array = [xCoordMinCor, yCoordMaxCor]
    """
    fullwinszX, fullwinszY = winSz[0], winSz[1]

    xRefPix_, yRefPix_ = rOrthoInfo.Map2Pixel(x=gcpCorr[0], y=gcpCorr[1])

    xRefPix = math.floor(xRefPix_)
    yRefPix = math.floor(yRefPix_)

    # Determination of the Reference Image subset surrounding the gcp in the Reference Image
    xRefMin_pix = xRefPix - fullwinszX / 2 + 1
    xRefMax_pix = xRefPix + fullwinszX / 2
    yRefMin_pix = yRefPix - fullwinszY / 2 + 1
    yRefMax_pix = yRefPix + fullwinszY / 2

    ## Check if the subset is fully inside the reference image boundaries
    if xRefMin_pix < 0 or xRefMax_pix > rOrthoInfo.rasterWidth \
            or yRefMin_pix < 0 or yRefMax_pix > rOrthoInfo.rasterHeight:
        msg = "Patch is outside reference image boundaries"
        # geoWarn.wrSubsetOutsideBoundries(dataName="rOrtho", arg=" for gcp nb:" + str(i + 1))
        warnings.warn(msg)
        return [0, 0, 0, 0, 0], [0, 0]
    else:
        # Store the subset pixel corners coordinates in the Reference image
        rOrthoPatchDimsArray = [1, xRefMin_pix, xRefMax_pix, yRefMin_pix, yRefMax_pix]

        deltaX_map = 0
        deltaY_map = 0

        if (xRefPix_ - xRefPix) > tol:
            # offset to be added for subpixel accuracy -
            deltaX_map = (xRefPix_ - xRefPix) * np.abs(rOrthoInfo.pixelWidth)
        if (yRefPix_ - yRefPix) > tol:
            # offset to be added for subpixel accuracy
            deltaY_map = (yRefPix_ - yRefPix) * np.abs(rOrthoInfo.pixelHeight)
        # print("xRefPix:{}, yRefPix:{}".format(xRefPix, yRefPix))
        # Determination of the Geographic/UTM coordinates of the subset corners of the on-coming orthorectified
        # subset of Slave image (with gcp corrected)
        xCoordCor = gcpCorr[0] - deltaX_map
        yCoordCor = gcpCorr[1] + deltaY_map

        xCoordMinCor = xCoordCor - (fullwinszX / 2 - 1) * np.abs(rOrthoInfo.pixelWidth)
        yCoordMaxCor = yCoordCor + (fullwinszY / 2 - 1) * np.abs(rOrthoInfo.pixelHeight)

        rawImgPatchX0Y0Array = [xCoordMinCor, yCoordMaxCor]

    return rOrthoPatchDimsArray, rawImgPatchX0Y0Array


def GetDEM_subsetDim(bboxCoords, demInfo: geoRT.cRasterInfo, margin=5):
    """
    Determining the DEM subset covering the patch area.
    The subset is enlarged of margin pixels for interpolation.
    Args:
        bboxCoords: [xRefMin_map, yRefMin_map, xRefMax_map, yRefMax_map]: map coordinates of the path dimension
        demInfo: DEM
        margin: padding margin ## TODO: add this parameter to configuration file.

    Returns: Subset pixel corners coordinates of the DEM :
            [1, xDemMinPix, xDemMaxPix, yDemMinPix, yDemMaxPix] or [0,0,0,0,0]

    """

    xRefMin_map, yRefMin_map, xRefMax_map, yRefMax_map = bboxCoords[0], bboxCoords[1], bboxCoords[2], bboxCoords[3]
    xDemMinPix, yDemMinPix = demInfo.Map2Pixel(x=xRefMin_map, y=yRefMin_map)
    xDemMaxPix, yDemMaxPix = demInfo.Map2Pixel(x=xRefMax_map, y=yRefMax_map)
    xDemMinPix = int(xDemMinPix - margin)
    yDemMinPix = int(yDemMinPix - margin)
    xDemMaxPix = int(xDemMaxPix + margin)
    yDemMaxPix = int(yDemMaxPix + margin)
    # Check if the subset is fully inside the DEM boundaries
    if xDemMinPix < 0 or xDemMaxPix > demInfo.raster_width or \
            yDemMinPix < 0 or yDemMaxPix > demInfo.raster_height:
        warnings.warn("DEM subset is outside DEM image boundaries")
        return [0, 0, 0, 0, 0]
    else:
        # Store the subset pixel corners coordinates in the DEM
        return [1, xDemMinPix, xDemMaxPix, yDemMinPix, yDemMaxPix]


def decimal_mod(value, param, precision=None):
    if precision is None:
        precision = 1e-5

    result = value % param

    if (np.abs(result) < precision) or (param - np.abs(result) < precision):
        result = 0
    return result


def get_dem_dims(xBBox, yBBox, demInfo: geoRT.cRasterInfo, margin=2):
    """
    Determining the DEM dimension needed to cover BBox extent .
    Args:
        xBBox:
        yBBox:
        demInfo:
        margin:

    Returns:dims = [xMin, xMax, yMin, yMax]

    """

    dims = [0, demInfo.raster_width, 0, demInfo.raster_height]
    ## Getting the BBox extent coordinates

    demCol, demRow = demInfo.Map2Pixel_Batch(X=xBBox, Y=yBBox)

    ## Get the needed subset to cover the input values.
    if int(np.min(demCol)) - margin > 0:
        dims[0] = int(np.min(demCol)) - margin
    if (np.ceil(np.max(demCol)) + margin) < demInfo.raster_width:
        dims[1] = np.ceil(np.max(demCol)) + margin
    if (int(np.min(demRow)) - margin) > 0:
        dims[2] = (int(np.min(demRow)) - margin)
    if (np.ceil(np.max(demRow)) + margin) < demInfo.raster_height:
        dims[3] = np.ceil(np.max(demRow)) + margin

    # TODO
    ## Accounting for situation where the grid footprint does not overlap with the dem
    ## this can happen either if the dem is out of the whole grid footprint (useless dem),
    ## or, more likely, when ortho is parallelized, the firsts or lasts tiles could be out of the dem.
    ## Need to handle that situation with other thing than an error message
    if (dims[0] > dims[1]) or (dims[2] > dims[3]):
        if (int(np.min(demCol)) - 1) > demInfo.raster_width:
            dims[0] = demInfo.raster_width - 4
            dims[1] = demInfo.raster_height
        if (np.ceil(np.max(demCol)) + 1) < 0:
            dims[0] = 0
            dims[1] = 4
        if (int(np.min(demRow)) - 1) > demInfo.raster_height:
            dims[2] = demInfo.raster_height - 4
            dims[3] = demInfo.raster_height
        if (np.ceil(np.max(demRow)) + 1) < 0:
            dims[2] = 0
            dims[3] = 4
    dims = [int(x) for x in dims]
    # print("dims=", dims)
    return dims


def GetDEM_dims_old(xBBox, yBBox, demInfo, margin=2):
    """
    Determining the DEM dimension needed to cover BBox extent .
    Args:
        xBBox:
        yBBox:
        demInfo:
        margin:

    Returns:dims = [xMin, xMax, yMin, yMax]

    """

    dims = [0, demInfo.rasterWidth, 0, demInfo.rasterHeight]
    ## Getting the BBox extent coordinates

    demCol, demRow = demInfo.Map2Pixel_Batch(X=xBBox, Y=yBBox)

    ## Get the needed subset to cover the input values.
    if int(np.min(demCol)) - margin > 0:
        dims[0] = int(np.min(demCol)) - margin
    if (np.ceil(np.max(demCol)) + margin) < demInfo.rasterWidth:
        dims[1] = np.ceil(np.max(demCol)) + margin
    if (int(np.min(demRow)) - margin) > 0:
        dims[2] = (int(np.min(demRow)) - margin)
    if (np.ceil(np.max(demRow)) + margin) < demInfo.rasterHeight:
        dims[3] = np.ceil(np.max(demRow)) + margin

    # TODO
    ## Accounting for situation where the grid footprint does not overlap with the dem
    ## this can happen either if the dem is out of the whole grid footprint (useless dem),
    ## or, more likely, when ortho is parallelized, the firsts or lasts tiles could be out of the dem.
    ## Need to handle that situation with other thing than an error message
    if (dims[0] > dims[1]) or (dims[2] > dims[3]):
        if (int(np.min(demCol)) - 1) > demInfo.rasterWidth:
            dims[0] = demInfo.rasterWidth - 4
            dims[1] = demInfo.rasterWidth
        if (np.ceil(np.max(demCol)) + 1) < 0:
            dims[0] = 0
            dims[1] = 4
        if (int(np.min(demRow)) - 1) > demInfo.rasterHeight:
            dims[2] = demInfo.rasterHeight - 4
            dims[3] = demInfo.rasterHeight
        if (np.ceil(np.max(demRow)) + 1) < 0:
            dims[2] = 0
            dims[3] = 4
    dims = [int(x) for x in dims]
    # print("dims=", dims)
    return dims
