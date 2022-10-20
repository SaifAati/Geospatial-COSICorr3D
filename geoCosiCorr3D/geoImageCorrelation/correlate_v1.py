"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import numpy as np
import sys, warnings, os
from ctypes import cdll
from pathlib import Path

import geoCosiCorr3D.georoutines.georoutines as geoRT
from geoCosiCorr3D.geoConfig import cgeoCfg

geoCfg = cgeoCfg()


class cCorrEngine:
    def __init__(self,
                 correlator="Frequency",
                 engine="Cosi-Corr",
                 debug=False):
        self.correlator = correlator
        self.debug = debug
        if correlator == "Frequency":
            self.freqParams(windowSize=4 * [64], steps=2 * [8], maskTh=0.9, resampling=False, nbIterations=4, grid=True)

        if correlator == "Spatial":
            self.spatialParams(windowSize=2 * [32], steps=2 * [16], searchRanges=2 * [10], grid=False)

        self.corrEngine = self.__dict__
        if self.debug:
            print(self.corrEngine)

    def spatialParams(self, windowSize=2 * [32], steps=2 * [16], searchRanges=2 * [10], grid=False):
        self.windowSizes = windowSize
        self.steps = steps
        self.searchRanges = searchRanges
        self.grid = grid

        return

    def freqParams(self, windowSize=4 * [64], steps=2 * [8], maskTh=0.9, resampling=False, nbIterations=4, grid=True):
        self.windowSizes = windowSize
        self.steps = steps
        self.maskTh = maskTh
        self.resampling = resampling
        self.nbIterations = nbIterations
        self.grid = grid

        return

    def PerformCorrelation(self, baseArray, targetArray):
        if self.corrEngine["correlator"] == "Frequency":
            freqCorrObj = cFreqCorr(corrName=self.correlator, windowSizes=self.windowSizes, steps=self.steps,
                                    iterations=self.nbIterations, maskTh=self.maskTh, grid=self.grid,
                                    resampling=self.resampling)
            ewArray, nsArray, snrArray = freqCorrObj.PerformCorrelation(baseArray=baseArray, targetArray=targetArray)

            return ewArray, nsArray, snrArray


class cCorrelate:

    def __init__(self, baseImagePath,
                 targetImagePath,
                 corrEngine,
                 baseBand=1,
                 targetBand=1,
                 oCorrPath=None,
                 userTileSize_mb=500,
                 visualize=False,
                 debug=True):
        """

        Args:
            baseImagePath: base image raster path :string
            targetImagePath: target image raster path : string
            corrEngine: dict of correlation parameters : dict
            baseBand: band number :int
            targetBand: band number :int
            oCorrPath: output correlation path : string
            userTileSize_mb:
            visualize:
            debug:
        """

        self.baseImagePath = baseImagePath
        self.targetImagePath = targetImagePath
        self.corrEngine = corrEngine

        self.userTileSize_mb = userTileSize_mb
        self.visualize = visualize
        self.baseBandNb = baseBand
        self.targetBandNb = targetBand
        self.oCorrPath = oCorrPath

        if oCorrPath == None:
            self.oCorrPath = os.path.join(os.path.dirname(self.baseImagePath),
                                          Path(self.baseImagePath).stem + "_VS_" +
                                          Path(self.targetImagePath).stem + "_" + corrEngine["correlator"] + "_w" +
                                          str(corrEngine["windowSizes"][0]) + "_Stp_" + str(
                                              corrEngine["steps"][0]) + ".tif")

        self.debug = debug
        printInfo = False
        if self.debug:
            printInfo = True
        self.baseInfo = geoRT.RasterInfo(inputRaster=self.baseImagePath, printInfo=printInfo)
        self.targetInfo = geoRT.RasterInfo(inputRaster=self.targetImagePath, printInfo=printInfo)

    def Correlate(self):

        if self.corrEngine["correlator"] == "Spatial":  # and self.corrEngine["Engine"] == "Cosi-Corr":
            self.corr = cSpatialCorr()
            self.corr.steps = self.corrEngine["steps"]
            self.corr.windowSizes = self.corrEngine["windowSizes"]
            self.corr.grid = self.corrEngine["grid"]
            self.corr.searchRanges = self.corrEngine["searchRanges"]
            self.corr.SetMargins()
        if self.corrEngine["correlator"] == "Frequency":  # and self.corrEngine["Engine"] == "Cosi-Corr":
            self.corr = cFreqCorr()
            self.corr.steps = self.corrEngine["steps"]
            self.corr.windowSizes = self.corrEngine["windowSizes"]
            self.corr.grid = self.corrEngine["grid"]
            self.corr.iterations = self.corrEngine["nbIterations"]
            self.corr.maskTh = self.corrEngine["maskTh"]
            self.corr.resampling = False
            self.corr.SetMargins()

        ##Check that the images have identical projection reference system

        self.CheckSameProjectionSystem()

        ## Check if the images have the same ground resolution

        self.CheckSameGroundResolution()

        # ## Check that the images are on geographically aligned grids (depends on origin and resolution)
        # ## verify if the difference between image origin is less than of resolution/1000

        self.CheckAlignedGrid()

        # ## Depending on the validity of the map information of the images, the pixel resolution is setup

        self.SetOutputResolution()

        self.SetWinArea()

        self.baseDims = [-1, 0, int(self.baseInfo.rasterWidth) - 1, 0, int(self.baseInfo.rasterHeight) - 1]
        self.targetDims = [-1, 0, int(self.targetInfo.rasterWidth) - 1, 0, int(self.targetInfo.rasterHeight) - 1]

        # Backup of the original master subset dimensions in case of a non gridded correlation
        if self.corr.grid == False:
            self.baseOriginalDims = self.baseDims

        """
            Cropping the images to the same size:
            Two conditions exist:
                1- if the map information are valid and identical: we define the overlapping area based on geo-referencing
                2- if map information invalid or different: we define the overlapping area based we define overlapping area
                    based on images size (pixel wise)
        """
        self.CroppingTmages2TheSameSize()

        """
           Adjusting cropped images according to a gridded/non-gridded output
           Two cases:
               1- If the user selected the gridded option
               2- If the user selected non-gridded option
       """
        self.AdjustingCroppedImagesAccording2GriddedOrNonGridded()

        if self.totCol <= 0 or self.totRow <= 0:
            sys.exit('Not enough overlapping area for correlation')
        if self.debug:
            print(self.baseDims)
            print(self.targetDims)

        self.Tiling()

        self.nbMeasurmentPerROI = self.nbCorrColPerROI * self.nbCorrRowPerROI * len(self.corr.bands)
        self.outputArray = np.empty(self.nbMeasurmentPerROI, dtype=np.float32)
        # print("output.shape", outputArray.shape)
        self.locArr = np.arange(self.nbCorrColPerROI * self.nbCorrRowPerROI) * len(self.corr.bands)
        # print("locArr.shape:", locArr.shape)

        ewArrayList = []
        nsArrayList = []
        snrArrayList = []

        from tqdm import tqdm
        for roi in tqdm(range(self.nbROI), desc="Correlation per tile"):
            if self.debug:
                print("Tile:{}/{} ".format(roi + 1, self.nbROI))

            baseSubset = self.baseInfo.ImageAsArray_Subset(xOffsetMin=self.dimsBaseTile[roi][:][1],
                                                           xOffsetMax=self.dimsBaseTile[roi][:][2],
                                                           yOffsetMin=self.dimsBaseTile[roi][:][3],
                                                           yOffsetMax=self.dimsBaseTile[roi][:][4],
                                                           bandNumber=self.baseBandNb)

            targetSubset = self.targetInfo.ImageAsArray_Subset(xOffsetMin=self.dimsTargetTile[roi][:][1],
                                                               xOffsetMax=self.dimsTargetTile[roi][:][2],
                                                               yOffsetMin=self.dimsTargetTile[roi][:][3],
                                                               yOffsetMax=self.dimsTargetTile[roi][:][4],
                                                               bandNumber=self.targetBandNb)
            if self.debug:
                print(self.dimsBaseTile[roi])
                print(self.dimsTargetTile[roi])

                print("baseSubset.size= ", baseSubset.shape)
                print("targetSubset.size= ", targetSubset.shape)

            if (self.nbROI > 1) and roi == (self.nbROI - 1):
                if self.debug:
                    print("LAST TILE")
                self.outputArray = np.zeros((self.nbCorrColPerROI * self.nbCorrRowLastROI * len(self.corr.bands)))
                self.locArr = np.arange(self.nbCorrColPerROI * self.nbCorrRowLastROI) * len(self.corr.bands)
                self.nbCorrRowPerROI = self.nbCorrRowLastROI

            ewArray, nsArray, snrArray = self.corr.PerformCorrelation(baseArray=baseSubset, targetArray=targetSubset)

            ewArrayList.append(ewArray * self.xRes)
            nsArrayList.append(nsArray * (-self.yRes))
            snrArrayList.append(snrArray)

        self.ewOutput = np.vstack(tuple(ewArrayList))
        self.nsOutput = np.vstack(tuple(nsArrayList))
        self.snrOutput = np.vstack(tuple(snrArrayList))

        if self.corr.grid == False:
            self.WriteBlankPixels()

        if all(self.flagList):
            self.SetGeoReferencing()
        else:
            warnings.warn("=== Pixel Based correlation ===")
            # print(self.flags)
            progress = False
            if self.debug:
                progress = True

            geoRT.WriteRaster(
                oRasterPath=self.oCorrPath,
                geoTransform=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                arrayList=[self.ewOutput, self.nsOutput, self.snrOutput],
                epsg=4326,
                metaData={"Base-image": Path(self.baseImagePath).stem,
                          "Target-image": Path(self.targetImagePath).stem},
                descriptions=self.corr.bands, progress=progress)

        if self.visualize:
            from geoRoutines.Plotting.Plotting_Routine import VisualizeCorrelation
            VisualizeCorrelation(iCorrPath=self.oCorrPath,
                                 ewArray=self.ewOutput,
                                 nsArray=self.nsOutput,
                                 factor=0.5,
                                 vmin=-5, vmax=5,
                                 dpi=300)

    def BlankArray(self, nbVals, nbBands=3):
        blankArray = np.zeros((nbVals * nbBands))
        for i in range(nbVals):
            blankArray[3 * i] = np.nan
            blankArray[3 * i + 1] = np.nan
            blankArray[3 * i + 2] = 0

        return blankArray

    def Decimalmod(self, value, param, precision=None):
        if precision == None:
            precision = 1e-5
        result = value % param

        if (np.abs(result) < precision) or (param - np.abs(result) < precision):
            result = 0
        return result

    def updateFlagList(self, flagDic):
        return list(flagDic.values())

    def CheckSameProjectionSystem(self):
        ##Check that the images have identical projection reference system
        self.flags = {"validMaps": False, "groundSpaceCorr": False, "continue": True}
        if self.baseInfo.validMapInfo and self.targetInfo.validMapInfo:
            self.flags["validMaps"] = True
            if self.baseInfo.EPSG_Code != self.targetInfo.EPSG_Code:
                warnings.warn(
                    "=== Input images have different map projection (!= EPSG code), Correlation will be pixel based! ===",
                    stacklevel=2)
                self.flags["groundSpaceCorr"] = False
            else:
                self.flags["groundSpaceCorr"] = True
        else:
            warnings.warn(
                "=== Input images are not geo-referenced. Correlation will be pixel based!,  ===",
                stacklevel=2)
            self.flags["groundSpaceCorr"] = False

        self.flagList = self.updateFlagList(self.flags)

    def CheckSameGroundResolution(self):
        ## Check if the images have the same ground resolution
        # (up to 1/1000 of the resolution to avoid precision error)
        if all(self.flagList):
            if (np.abs((self.baseInfo.pixelWidth / self.targetInfo.pixelWidth) - 1) > 0.001) or (
                    np.abs((self.baseInfo.pixelHeight / self.targetInfo.pixelHeight) - 1) > 0.001):
                self.flags["continue"] = False
                sys.exit("=== ERROR: Input data must have the same resolution to be correlated ===")
            else:
                if self.debug:
                    print("=== Images have the same GSD:{} ===".format(self.baseInfo.pixelWidth))
        self.flagList = self.updateFlagList(self.flags)

    def CheckAlignedGrid(self):
        ## Check that the imqges are on geographically aligned grids (depends on origin and resolution)
        ## verify if the difference between image origin is less than of resolution/1000
        if all(self.flagList):
            if self.Decimalmod(value=self.baseInfo.xOrigin - self.targetInfo.xOrigin, param=self.baseInfo.pixelWidth,
                               precision=self.baseInfo.pixelWidth / 1000) != 0 or self.Decimalmod(
                value=self.baseInfo.yOrigin - self.targetInfo.yOrigin, param=np.abs(self.baseInfo.pixelHeight),
                precision=np.abs(self.baseInfo.pixelHeight) / 1000) != 0:
                self.flags["overlap"] = False
                sys.exit(
                    "=== ERROR: --- Images cannot be overlapped due to their origin and resolution - "
                    "Origins difference great than 1/1000 of the pixel resolution ===")
            else:
                self.flags["overlap"] = True
        self.flagList = self.updateFlagList(self.flags)

    def SetOutputResolution(self):
        ## Depending on the validity of the map information of the images, the pixel resolution is setup
        # it will be the GSD if map info is valid , otherwise it will be 1 and the correlation will be pixel based
        if all(self.flagList):
            # the correlation will be map based not pixel based
            self.yRes = np.abs(self.baseInfo.pixelHeight)
            self.xRes = np.abs(self.baseInfo.pixelWidth)
        else:
            # Correlation will be pixel based
            self.xRes = 1.0
            self.yRes = 1.0

    def SetWinArea(self):
        ## Winarea is composed of the window correlation size + the necessary margin if any (i.e. search distance, resampling kernel,...)
        self.winAreaX = np.int(self.corr.windowSizes[0] + 2 * self.corr.margins[0])
        self.winAreaY = np.int(self.corr.windowSizes[1] + 2 * self.corr.margins[1])
        if self.debug:
            print("winAreaX:.{}, winAreaX:.{}".format(self.winAreaX, self.winAreaY))

    def CroppingTmages2TheSameSize(self):
        """
                    Cropping the images to the same size:
                    Two condition exist:
                        1- if the map information are valid and identical: we define the overlapping area based on geo-referencing
                        2- if map information invalid or different: we define the overlapping area based we define overlapping area
                            based on images size (pixel wise)
                """
        if all(self.flagList):
            # If map information are valid and identical, define the overlapping are based on geo-referencing.
            # Backup of the original master subset dimensions in case of a non gridded correlation
            # IF NOT grid THEN img1OriginalDims = baseDims
            offset = ((self.targetInfo.xOrigin + self.targetDims[1] * self.targetInfo.pixelWidth) - (
                    self.baseInfo.xOrigin + self.baseDims[1] * self.baseInfo.pixelWidth)) / self.baseInfo.pixelWidth
            if offset > 0:
                self.baseDims[1] = int(self.baseDims[1] + round(offset))
            else:
                self.targetDims[1] = int(self.targetDims[1] - round(offset))

            offset = ((self.targetInfo.xOrigin + self.targetDims[2] * self.targetInfo.pixelWidth) - (
                    self.baseInfo.xOrigin + self.baseDims[2] * self.baseInfo.pixelWidth)) / self.baseInfo.pixelWidth

            if offset < 0:
                self.baseDims[2] = int(self.baseDims[2] + round(offset))
            else:
                self.targetDims[2] = int(self.targetDims[2] - round(offset))

            offset = ((self.targetInfo.yOrigin - self.targetDims[3] * np.abs(self.targetInfo.pixelHeight)) - (
                    self.baseInfo.yOrigin - self.baseDims[3] * np.abs(self.baseInfo.pixelHeight))) / np.abs(
                self.baseInfo.pixelHeight)
            if offset < 0:
                self.baseDims[3] = int(self.baseDims[3] - round(offset))
            else:
                self.targetDims[3] = int(self.targetDims[3] + round(offset))

            offset = ((self.targetInfo.yOrigin - self.targetDims[4] * np.abs(self.targetInfo.pixelHeight)) - (
                    self.baseInfo.yOrigin - self.baseDims[4] * np.abs(self.baseInfo.pixelHeight))) / np.abs(
                self.baseInfo.pixelHeight)
            if offset > 0:
                self.baseDims[4] = int(self.baseDims[4] - round(offset))
            else:
                self.targetDims[4] = int(self.targetDims[4] + round(offset))

            if self.baseDims[0] >= self.baseDims[2] or self.baseDims[3] >= self.baseDims[4]:
                sys.exit(
                    "=== ERROR: Images do not have a geographic overlap ===")
        else:
            # If map information invalid or different, define overlapping area based on images size (pixel-wise)

            if (self.baseDims[2] - self.baseDims[1]) > (self.targetDims[2] - self.targetDims[1]):
                self.baseDims[2] = self.baseDims[1] + (self.targetDims[2] - self.targetDims[1])
            else:
                self.targetDims[2] = self.targetDims[1] + (self.baseDims[2] - self.baseDims[1])
            if (self.baseDims[4] - self.baseDims[3]) > (self.targetDims[4] - self.targetDims[3]):
                self.baseDims[4] = self.baseDims[3] + (self.targetDims[4] - self.targetDims[3])
            else:
                self.targetDims[4] = self.targetDims[3] + (self.baseDims[4] - self.baseDims[3])

    def AdjustingCroppedImagesAccording2GriddedOrNonGridded(self):
        # If the user selected the gridded option
        if self.corr.grid:
            if all(self.flagList):
                if self.Decimalmod(value=self.baseInfo.xOrigin, param=self.baseInfo.pixelWidth) != 0 or \
                        self.Decimalmod(value=self.baseInfo.yOrigin, param=np.abs(self.baseInfo.pixelHeight)) != 0:
                    sys.exit(
                        "=== ERROR: Images coordinates origins must be a multiple of the resolution for a gridded output' ===")
                ## Chek if the geo-coordinate of the first correlated pixel is multiple integer of the resolution
                ## If not adjust the image boundaries
                geoOffsetX = (self.baseInfo.xOrigin + (
                        self.baseDims[1] + self.corr.margins[0] + self.corr.windowSizes[
                    0] / 2) * self.baseInfo.pixelWidth) % \
                             (self.corr.steps[0] * self.baseInfo.pixelWidth)

                # print(self.baseDims[1], self.corr.margins[0], self.corr.windowSizes[0] / 2, self.baseInfo.pixelWidth)

                if np.round(geoOffsetX / self.baseInfo.pixelWidth) != 0:
                    self.baseDims[1] = int(
                        self.baseDims[1] + self.corr.steps[0] - np.round(geoOffsetX / self.baseInfo.pixelWidth))
                    self.targetDims[1] = int(
                        self.targetDims[1] + self.corr.steps[0] - np.round(geoOffsetX / self.baseInfo.pixelWidth))

                geoOffsetY = (self.baseInfo.yOrigin - (
                        self.baseDims[3] + self.corr.margins[1] + self.corr.windowSizes[1] / 2) * np.abs(
                    self.baseInfo.pixelHeight)) % \
                             (self.corr.steps[1] * np.abs(self.baseInfo.pixelHeight))

                if np.round(geoOffsetY / np.abs(self.baseInfo.pixelHeight)) != 0:
                    self.baseDims[3] = int(self.baseDims[3] + np.round(geoOffsetY / np.abs(self.baseInfo.pixelWidth)))
                    self.targetDims[3] = int(
                        self.targetDims[3] + np.round(geoOffsetY / np.abs(self.baseInfo.pixelWidth)))

            ## Define the number of column and rows of the ouput correlation
            self.totCol = int(np.floor(
                (self.baseDims[2] - self.baseDims[1] + 1 - self.corr.windowSizes[0] - 2 * self.corr.margins[0]) /
                self.corr.steps[0]) + 1)
            self.totRow = int(np.floor(
                (self.baseDims[4] - self.baseDims[3] + 1 - self.corr.windowSizes[1] - 2 * self.corr.margins[1]) /
                self.corr.steps[1]) + 1)
            if self.debug:
                print("tCols:{}, tRows:{}".format(self.totCol, self.totRow))

        else:
            # The non-gridded correlation will generate a correlation map whose first pixel corresponds
            # to the first master pixel

            # Define the total number of pixel of the correlation map in col and row
            self.totCol = int(np.floor((self.baseOriginalDims[2] - self.baseOriginalDims[1]) / self.corr.steps[0]) + 1)
            self.totRow = int(np.floor((self.baseOriginalDims[4] - self.baseOriginalDims[3]) / self.corr.steps[1]) + 1)
            if self.debug:
                print("tCols:{}, tRows:{}".format(self.totCol, self.totRow))

            # Compute the "blank" border in col and row. This blank border corresponds to the area of the
            # correlation map where no correlation values could be computed due to the patch characteristic of the
            # correlator
            self.borderColLeft = int(np.ceil(
                (self.baseDims[1] - self.baseOriginalDims[1] + self.corr.windowSizes[0] / 2 + self.corr.margins[
                    0]) / np.float(
                    self.corr.steps[0])))
            self.borderRowTop = int(np.ceil(
                (self.baseDims[3] - self.baseOriginalDims[3] + self.corr.windowSizes[1] / 2 + self.corr.margins[
                    1]) / np.float(
                    self.corr.steps[1])))
            if self.debug:
                print("borderColLeft:{}, borderRowTop:{}".format(self.borderColLeft, self.borderRowTop))
            # From the borders in col and row, compute the necessary cropping of the master and slave in row and col,
            # so the first patch retrived from the tile correponds to a step-wise position of the correlation grid origin
            offsetX = self.borderColLeft * self.corr.steps[0] - (
                    self.corr.windowSizes[0] / 2 + self.corr.margins[0]) - (
                              self.baseDims[1] - self.baseOriginalDims[1])
            offsetY = self.borderRowTop * self.corr.steps[1] - (self.corr.windowSizes[1] / 2 + self.corr.margins[1]) - (
                    self.baseDims[3] - self.baseOriginalDims[3])
            self.baseDims[1] = int(self.baseDims[1] + offsetX)
            self.targetDims[1] = int(self.targetDims[1] + offsetX)
            self.baseDims[3] = int(self.baseDims[3] + offsetY)
            self.targetDims[3] = int(self.targetDims[3] + offsetY)
            # Define the number of actual correlation, i.e., the total number of points in row and
            # column, minus the "blank" correlation on the border
            self.nbCorrCol = int(np.floor(
                (self.baseDims[2] - self.baseDims[1] + 1 - self.corr.windowSizes[0] - 2 * self.corr.margins[0]) /
                self.corr.steps[0]) + 1)
            self.nbCorrRow = int(np.floor(
                (self.baseDims[4] - self.baseDims[3] + 1 - self.corr.windowSizes[1] - 2 * self.corr.margins[1]) /
                self.corr.steps[1]) + 1)

            if self.debug:
                print("nbCorrCol:{}, nbCorrRow:{}".format(self.nbCorrCol, self.nbCorrRow))

            #  ;Define the blank border on the right side in column and bottom side in row
            self.borderColRight = int(self.totCol - self.borderColLeft - self.nbCorrCol)
            self.borderRowBottom = int(self.totRow - self.borderRowTop - self.nbCorrRow)
            if self.debug:
                print("borderColRight:{}, borderRowBottom:{}".format(self.borderColRight, self.borderRowBottom))

            # Define a "blank" (i.e., invalid) correlation line
            self.outputRowBlank = self.BlankArray(nbVals=self.totCol)
            ##Define blank column border left and right
            self.outputColLeftBlank = self.BlankArray(nbVals=self.borderColLeft)

            self.outputColRightBlank = self.BlankArray(nbVals=self.borderColRight)

    def Tiling(self):
        # Get number of pixel in column and row of the file subset to tile
        self.nbColImg = int(self.baseDims[2] - self.baseDims[1] + 1)
        self.nbRowImg = int(self.baseDims[4] - self.baseDims[3] + 1)
        if self.debug:
            print("nbColImg: {} || nbRowImg: {}".format(self.nbColImg, self.nbRowImg))
        ##  ### We assume that images are FLOAT32 type !!!! To be improved
        pixelMemoryFootprint = 32
        # Define number max of lines per tile
        self.maxRowsROI = int(
            np.floor((self.userTileSize_mb * 8 * 1024 * 1024) / (self.nbColImg * pixelMemoryFootprint)))
        if self.debug:
            print("maxRowsROI=", self.maxRowsROI)
        # Define number of correlation column and lines computed for one tile
        if self.maxRowsROI < self.nbRowImg:
            temp = self.maxRowsROI
            self.nbCorrRowPerROI = int((temp - self.winAreaY) / self.corr.steps[1] + 1)
        else:
            temp = self.nbRowImg
            self.nbCorrRowPerROI = int((temp - self.winAreaY) / self.corr.steps[1] + 1)

        self.nbCorrColPerROI = int((self.nbColImg - self.winAreaX) / self.corr.steps[0] + 1)

        self.nbROI = int(
            (self.nbRowImg - self.winAreaY + self.corr.steps[1]) / (
                    (self.nbCorrRowPerROI - 1) * self.corr.steps[1] + (self.winAreaY - self.corr.steps[1])))

        if self.nbROI < 1:
            ## At least one tile even if the ROI is Larger than the image
            self.nbROI = 1
        if self.debug:
            print("nbROI: {} || nbCorrRowPerROI: {} || nbCorrColPerROI: {}".format(self.nbROI, self.nbCorrRowPerROI,
                                                                                   self.nbCorrColPerROI))

        # Define the boundaries of all the tiles but the last one which will have a different size
        self.dimsBaseTile = np.zeros((self.nbROI, 5), dtype=np.int64)
        self.dimsTargetTile = np.zeros((self.nbROI, 5), dtype=np.int64)
        for i in range(self.nbROI):
            val = int(self.baseDims[3] + ((i + 1) * self.nbCorrRowPerROI - 1) * self.corr.steps[
                1] + self.winAreaY - 1)
            self.dimsBaseTile[i, :] = [-1,
                                       self.baseDims[1],
                                       self.baseDims[2],
                                       self.baseDims[3] + i * self.nbCorrRowPerROI * self.corr.steps[1],
                                       val]

            self.dimsTargetTile[i, :] = [-1,
                                         self.targetDims[1],
                                         self.targetDims[2],
                                         self.targetDims[3] + i * self.nbCorrRowPerROI * self.corr.steps[1],
                                         int(self.targetDims[3] + ((i + 1) * self.nbCorrRowPerROI - 1) *
                                             self.corr.steps[1] + self.winAreaY - 1)]

        # Define boundaries of the last tile and the number of correlation column and lines computed for the last tile
        # print(self.dimsBaseTile)
        # print(self.dimsTargetTile)
        self.nbRowsLeft = int((self.baseDims[4] - self.dimsBaseTile[self.nbROI - 1, 4] + 1) - 1)
        if self.debug:
            print("nbRowsLeft=", self.nbRowsLeft, "\n")
        if (self.nbRowsLeft >= self.corr.steps[1]):
            self.nbCorrRowLastROI = int(self.nbRowsLeft / self.corr.steps[1])

            self.dimsBaseTile = np.vstack((self.dimsBaseTile, np.array(
                [-1, self.baseDims[1], self.baseDims[2],
                 self.baseDims[3] + self.nbROI * self.nbCorrRowPerROI * self.corr.steps[1],
                 int(self.baseDims[3] + self.nbROI * self.nbCorrRowPerROI * self.corr.steps[1] + self.winAreaY - 1 + (
                         self.nbCorrRowLastROI - 1) * self.corr.steps[1])])))

            self.dimsTargetTile = np.vstack((self.dimsTargetTile, np.array(
                [-1, self.targetDims[1], self.targetDims[2],
                 self.targetDims[3] + self.nbROI * self.nbCorrRowPerROI * self.corr.steps[1],
                 int(self.targetDims[3] + self.nbROI * self.nbCorrRowPerROI * self.corr.steps[1] + self.winAreaY - 1 + (
                         self.nbCorrRowLastROI - 1) * self.corr.steps[1])])))
            self.nbROI = self.nbROI + 1
        else:
            self.nbCorrRowLastROI = self.nbCorrRowPerROI

    def WriteBlankPixels(self):
        ## In case of non-gridded correlation,write the top blank correlation lines
        # print(borderRowTop)
        # print(outputRowBlank.shape)
        # Define a "blank" (i.e., invalid) correlation line
        tempAdd = np.empty((self.borderRowTop, self.ewOutput.shape[1]))
        tempAdd[:] = np.nan
        self.ewOutput = np.vstack((tempAdd, self.ewOutput))
        self.nsOutput = np.vstack((tempAdd, self.nsOutput))
        self.snrOutput = np.vstack((tempAdd, self.snrOutput))

        # In case of non-gridded correlation, write the bottom blank correlation lines
        if self.borderRowBottom != 0:
            tempAdd = np.empty((self.borderRowBottom, self.ewOutput.shape[1]))
            tempAdd[:] = np.nan
            self.ewOutput = np.vstack((self.ewOutput, tempAdd))
            self.nsOutput = np.vstack((self.nsOutput, tempAdd))
            self.snrOutput = np.vstack((self.snrOutput, tempAdd))

        if self.borderColLeft != 0:
            ##Define blank column border left and right
            # outputColLeftBlank = BlankArray(nbVals=borderColLeft)
            tempAdd = np.empty((self.ewOutput.shape[0], self.borderRowBottom))
            tempAdd[:] = np.nan
            self.ewOutput = np.vstack((tempAdd.T, self.ewOutput.T)).T
            self.nsOutput = np.vstack((tempAdd.T, self.nsOutput.T)).T
            self.snrOutput = np.vstack((tempAdd.T, self.snrOutput.T)).T
        if self.borderColRight != 0:
            tempAdd = np.empty((self.ewOutput.shape[0], self.borderColRight))
            tempAdd[:] = np.nan
            self.ewOutput = np.vstack((self.ewOutput.T, tempAdd.T)).T
            self.nsOutput = np.vstack((self.nsOutput.T, tempAdd.T)).T
            self.snrOutput = np.vstack((self.snrOutput.T, tempAdd.T)).T

    def SetGeoReferencing(self):
        if self.corr.grid:
            originX = self.baseInfo.xOrigin + (
                    self.baseDims[1] + self.corr.margins[0] + self.corr.windowSizes[0] / 2) * self.baseInfo.pixelWidth
            originY = self.baseInfo.yOrigin - (
                    self.baseDims[3] + self.corr.margins[1] + self.corr.windowSizes[1] / 2) * np.abs(
                self.baseInfo.pixelHeight)
        else:
            originX = self.baseInfo.xOrigin + self.baseDims[1] * self.baseInfo.pixelWidth
            originY = self.baseInfo.yOrigin - self.baseDims[3] * np.abs(self.baseInfo.pixelHeight)

        geoTransform = [originX, self.baseInfo.pixelWidth * self.corr.steps[0], 0, originY, 0,
                        self.baseInfo.pixelHeight * self.corr.steps[1]]
        if self.debug:
            print("geoTransform:", geoTransform)

        progress = False
        if self.debug:
            progress = True
        geoRT.WriteRaster(oRasterPath=self.oCorrPath,
                          geoTransform=geoTransform,
                          arrayList=[self.ewOutput, self.nsOutput, self.snrOutput],
                          epsg=self.baseInfo.EPSG_Code,
                          metaData={"Base-image": Path(self.baseImagePath).stem,
                                    "Target-image": Path(self.targetImagePath).stem},
                          descriptions=self.corr.bands, progress=progress)


class cSpatialCorr:

    def __init__(self):
        self.corrName = "Spatial"
        self.windowSizes = [32, 32]
        self.steps = [16, 16]
        self.searchRanges = [50, 50]
        self.grid = False
        self.bands = ["East/West", "North/South", "SNR"]
        self.libPath = geoCfg.geoStatCorrLib
        self.SetMargins()

    def SetMargins(self):
        self.margins = self.searchRanges

        return

    def ComputeOutputRowsAndColumns(self, stepSizes, inputShape, windoSizes, rangeSizes):
        if (stepSizes[0] != 0):
            value = inputShape[1] - (windoSizes[0] + 2 * rangeSizes[0])
            outputCols = int((np.floor(value / stepSizes[0] + 1.0)))
        else:
            outputCols = 1
        if (stepSizes[1] != 0):

            value = (inputShape[0] - (windoSizes[1] + 2 * rangeSizes[1]))
            outputRows = int(np.floor(value / stepSizes[1] + 1.0))
        else:
            outputRows = 1
        return outputRows, outputCols

    def PerformCorrelation(self,
                           baseArray,
                           targetArray):
        # load the library
        libCstatCorr = cdll.LoadLibrary(self.libPath)
        libCstatCorr.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32)]

        inputShape = np.array(baseArray.shape, dtype=np.int32)
        windowSizes = np.array(self.windowSizes, dtype=np.int32)
        stepSizes = np.array(self.steps, dtype=np.int32)
        searchRanges = np.array(self.searchRanges, dtype=np.int32)
        baseArray = np.array(baseArray.flatten(), dtype=np.float32)
        targetArray = np.array(targetArray.flatten(), dtype=np.float32)

        outputRows, outputCols = self.ComputeOutputRowsAndColumns(stepSizes=stepSizes, inputShape=inputShape,
                                                                  windoSizes=windowSizes,
                                                                  rangeSizes=searchRanges)
        outputShape = np.array([outputRows, outputCols], dtype=np.int32)

        ewArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)
        nsArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)
        snrArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)

        libCstatCorr.InputData(inputShape, windowSizes, stepSizes, searchRanges, baseArray, targetArray, outputShape,
                               ewArray_fl, nsArray_fl, snrArray_fl)

        ewArray = np.asarray(ewArray_fl[:, 0]).reshape((outputRows, outputCols))
        nsArray = np.asarray(nsArray_fl[:, 0]).reshape((outputRows, outputCols))
        snrArray = np.asarray(snrArray_fl[:, 0]).reshape((outputRows, outputCols))

        return ewArray, nsArray, snrArray


class cFreqCorr:
    def __init__(self,
                 corrName="Frequency",
                 windowSizes=[64, 64, 32, 32],
                 steps=[16, 16],
                 iterations=4,
                 maskTh=0.9,
                 grid=False,
                 resampling=False,
                 bands=["East/West", "North/South", "SNR"]):
        """

        Args:
            corrName:
            windowSizes:
            steps:
            iterations:
            maskTh:
            grid:
            resampling:
            bands:
        """
        self.corrName = corrName
        self.windowSizes = windowSizes
        self.steps = steps
        self.iterations = iterations
        self.maskTh = maskTh
        self.grid = grid
        self.bands = bands
        self.resampling = resampling
        self.libPath = geoCfg.geoFreqCorrLib
        self.SetMargins()

    def SetMargins(self):
        if ~self.resampling:
            self.margins = [int(self.windowSizes[0] / 2), int(self.windowSizes[1] / 2)]
            # print("Margin:", self.margins)
        else:
            warnings.warn("Compute margin based on resampling Kernel ! ")
            sys.exit("Not implemented  ")

    def PerformCorrelation(self,
                           baseArray,
                           targetArray):
        # load the library
        libCfreqCorr = cdll.LoadLibrary(self.libPath)
        libCfreqCorr.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.int32)]
        iShape = baseArray.shape

        inputShape = np.array(baseArray.shape, dtype=np.int32)
        windowSizes = np.array(self.windowSizes, dtype=np.int32)
        stepSizes = np.array(self.steps, dtype=np.int32)
        baseArray = np.array(baseArray.flatten(), dtype=np.float32)
        targetArray = np.array(targetArray.flatten(), dtype=np.float32)
        ewArray_ = np.zeros((iShape[0] * iShape[1], 1), dtype=np.float32)
        nsArray_ = np.zeros((iShape[0] * iShape[1], 1), dtype=np.float32)
        snrArray_ = np.zeros((iShape[0] * iShape[1], 1), dtype=np.float32)
        iteration = np.array([self.iterations], dtype=np.int32)
        maskThreshold = np.array([self.maskTh], dtype=np.float32)
        oShape = np.array([0, 0], dtype=np.int32)
        libCfreqCorr.InputData(inputShape,
                               windowSizes,
                               stepSizes,
                               baseArray,
                               targetArray,
                               ewArray_,
                               nsArray_,
                               snrArray_,
                               iteration,
                               maskThreshold,
                               oShape)

        ewArrayfl = ewArray_[0:oShape[0] * oShape[1]]
        nsArrayfl = nsArray_[0:oShape[0] * oShape[1]]
        snrArrayfl = snrArray_[0:oShape[0] * oShape[1]]
        ewArray = np.asarray(ewArrayfl).reshape((oShape[0], oShape[1]))
        nsArray = np.asarray(nsArrayfl).reshape((oShape[0], oShape[1]))
        snrArray = np.asarray(snrArrayfl).reshape((oShape[0], oShape[1]))

        return ewArray, nsArray, snrArray


class cBatchCorrelate:
    def __init__(self, iBaseList, iTargetList, oFolder, corrEngine, corrStrat, band):
        self.iBaseList = iBaseList
        # self.iBaseBand = iBaseBandList
        self.iTargetList = iTargetList
        # self.iTargetBandList = iTargetBandList
        self.oFolder = oFolder
        self.corrEngine = corrEngine
        self.bandNb = band
        self.corrStrategy = corrStrat

    def CorrelationStrategy(self, baseImgList, targetImgList, corrStrat=2):
        corrStrategy = corrStrat

        if corrStrategy == 1:
            return baseImgList, targetImgList

        if corrStrategy == 2:
            newBaseList = []
            newTargetList = []
            # print(baseImgList)

            print("Correlation Strategy all")
            nbCorrelation = int(len(baseImgList) * (len(targetImgList) - 1) / 2)
            print("Number of correlation:", nbCorrelation)
            for i in range(len(baseImgList) - 1):
                newMList_ = (len(baseImgList) - 1 - i) * [baseImgList[i]]
                newBaseList.extend(newMList_)
                for j in range(i + 1, len(targetImgList)):
                    newTargetList.append(targetImgList[j])
            return newBaseList, newTargetList

        if corrStrategy == 3:
            newBaseList = []
            newTargetList = []
            print(baseImgList)
            print("Correlation Strategy all (Back and Forward)")
            nbCorrelation = int(len(baseImgList) * (len(baseImgList) - 1))
            print("Number of correlation:", nbCorrelation)
            for base_ in baseImgList:
                newTargetList_ = []
                for target_ in targetImgList:
                    if base_ != target_:
                        newBaseList.append(base_)
                        newTargetList_.append(target_)
                newTargetList.extend(newTargetList_)
            return newBaseList, newTargetList
        if corrStrategy == 4:
            from itertools import permutations
            newBaseList = []
            newTargetList = []
            print(baseImgList)
            print("#############################################")
            print("--Correlation Strategy all (Back and Forward)")
            print("--Base list is the same as the target list---")
            print("#############################################")
            # nbCorrelation = int(len(baseImgList) * (len(baseImgList) - 1))

            perm = permutations(baseImgList, 2)
            permList = []
            for i in list(perm):
                permList.append(i)
                newBaseList.append(i[0])
                newTargetList.append(i[1])
            return newBaseList, newTargetList
        return

    def WriteCorrFiles(self, filePath, infoList):
        with open(filePath, 'w') as f:
            for item in infoList:
                item_ = item
                f.write("%s\n" % item_)
        return

    def PrformBatchCorr(self):

        baseList, targetList = self.CorrelationStrategy(baseImgList=self.iBaseList,
                                                        targetImgList=self.iTargetList,
                                                        corrStrat=self.corrStrategy)
        self.WriteCorrFiles(filePath=os.path.join(self.oFolder, "0-Base.txt"), infoList=baseList)
        self.WriteCorrFiles(filePath=os.path.join(self.oFolder, "1-Target.txt"), infoList=targetList)
        oRasterPathList = []
        for baseInputPath, targetInputPath in zip(baseList, targetList):
            oRasterPath = ""
            temp = Path(baseInputPath).stem + "_VS_" + Path(targetInputPath).stem + "_"
            if self.corrEngine["Correlator"] == "Spatial":
                oRasterPath = os.path.join(self.oFolder, temp + "geoSpCorr_B" + str(self.bandNb) + "_W" + str(
                    self.corrEngine["WindowSize"][0]) + "_S" + str(self.corrEngine["Steps"][0]) + "_R" + str(
                    self.corrEngine["RangeArea"][0]) + ".tif")
                oRasterPathList.append(oRasterPath)
            if self.corrEngine["Correlator"] == "Frequency":
                oRasterPath = os.path.join(self.oFolder, temp + "geoFqCorr_B" + str(self.bandNb) + "_W" +
                                           str(self.corrEngine["WindowSize"][0]) + "_S" +
                                           str(self.corrEngine["Steps"][0]) + ".tif")
                oRasterPathList.append(oRasterPath)
        self.WriteCorrFiles(filePath=os.path.join(self.oFolder, "3-oCorr.txt"), infoList=oRasterPathList)

        for baseInputPath_, targetInputPath_, oRasterPath_ in zip(baseList, targetList, oRasterPathList):
            correlation = cCorrelate(baseImagePath=baseInputPath_, targetImagePath=targetInputPath_,
                                     corrEngine=self.corrEngine,
                                     oCorrPath=oRasterPath_, visualize=True, bandNumber=self.bandNb)
            correlation.Correlate()

        return
