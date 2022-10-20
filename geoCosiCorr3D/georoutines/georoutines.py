"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import os
import sys
import warnings

import numpy as np
import pyproj
import gdal, gdalconst
import ogr, osr
import rasterio

from scipy.stats import norm
from scipy import stats
from rasterio.merge import merge
from pathlib import Path
from typing import Optional, List


# import geoRoutines.FilesCommandRoutine as fileRT
# import geoRoutines.FootPrint as fpRT


class cgeoStat:
    def __init__(self, inputArray, displayValue=True):
        sample = np.ma.masked_invalid(inputArray)
        mask = np.ma.getmask(sample)

        # Remove mask and array to vector
        if isinstance(sample, np.ma.MaskedArray):  # check if the sample was masked using the class numpy.ma.MaskedArray
            sample = sample.compressed()  ## return all the non-masked values as 1-D array
        else:
            if sample.ndim > 1:  # if the dimension of the array more than 1 transform it to 1-D array
                sample = sample.flatten()
        self.sample = sample
        # Estimate initial sigma and RMSE
        (self.mu, self.sigma) = norm.fit(sample)
        self.sigma_ = '%.3f' % (self.sigma)
        temp = np.square(sample)
        temp = np.ma.masked_where(temp <= 0, temp)
        self.RMSE = '%.3f' % (np.ma.sqrt(np.ma.mean(temp)))

        self.max = '%.3f' % (np.nanmax(sample))
        self.min = '%.3f' % (np.nanmin(sample))
        self.std = '%.3f' % (np.nanstd(sample))
        self.mean = '%.3f' % (np.nanmean(sample))
        self.median = '%.3f' % (np.nanmedian(sample))
        self.mad = '%.3f' % (stats.median_absolute_deviation(sample))
        self.nmad = '%.3f' % (1.4826 * stats.median_absolute_deviation(sample))
        self.ce68 = stats.norm.interval(0.68, loc=self.mu, scale=self.sigma)
        self.ce90 = stats.norm.interval(0.9, loc=self.mu, scale=self.sigma)
        self.ce95 = stats.norm.interval(0.95, loc=self.mu, scale=self.sigma)
        self.ce99 = stats.norm.interval(0.99, loc=self.mu, scale=self.sigma)

        if displayValue:
            print("mu, sigma", self.mu, self.sigma)
            print("RMSE=", self.RMSE)
            print("max=", self.max, "min=", self.min, "std=", self.std, "mean=", self.mean, "median", self.median,
                  "mad=",
                  self.mad, "nmad=", self.nmad)
            print("CE68=", self.ce68, "CE90=", self.ce90, "CE95=", self.ce95, "CE99=", self.ce99)


class RasterInfo:
    def __init__(self, inputRaster, printInfo=False):
        """

        Args:
            inputRaster: raster path : string
            printInfo: Display raster information if True
        """
        self.error = False
        try:
            ## Open the image using GDAL
            self.rasterPath = inputRaster
            self.rasterName = os.path.basename(inputRaster)
            self.raster = gdal.Open(self.rasterPath)
            self.geoTrans = self.raster.GetGeoTransform()
            self.geoTransMatrix_pix2map = np.array([[self.geoTrans[1], self.geoTrans[2], self.geoTrans[0]],
                                                    [self.geoTrans[4], self.geoTrans[5], self.geoTrans[3]],
                                                    [0, 0, 1]])
            R = np.array([[self.geoTrans[1], self.geoTrans[2], 0],
                          [self.geoTrans[4], self.geoTrans[5], 0],
                          [0, 0, 1]])
            T = np.array([[0, 0, self.geoTrans[0]], [0, 0, self.geoTrans[3]], [0, 0, 0]])

            self.geoTransMatrix_map2pix = np.linalg.inv(R) - np.dot(np.linalg.inv(R), T)
            self.GetRasterInfo(printInfo)
        except:
            print('Problem when loading image with GDAL:', self.rasterPath, sys.exc_info())
            self.error = True
            return

    def GetRasterInfo(self, printInfo):
        if self.error == False:
            self.xOrigin = self.geoTrans[0]  ## UpLeftEW
            self.yOrigin = self.geoTrans[3]  ## UpLeftNs
            self.pixelWidth = self.geoTrans[1]
            self.pixelHeight = self.geoTrans[5]
            self.nbBand = self.raster.RasterCount
            self.rasterWidth = self.raster.RasterXSize
            self.rasterHeight = self.raster.RasterYSize
            self.rasterDataType = []
            for i in range(self.nbBand):
                band = self.raster.GetRasterBand(1)
                self.rasterDataType.append(gdal.GetDataTypeName(band.DataType))
                del band
            try:
                self.validMapInfo = True
                projection = self.raster.GetProjection()
                self.proj = osr.SpatialReference(wkt=self.raster.GetProjection())
                self.EPSG_Code = self.proj.GetAttrValue('AUTHORITY', 1)
                src = rasterio.open(self.rasterPath)
                crs = src.crs
                epsg_spare = int(str(crs).split(":")[1])
                if epsg_spare != self.EPSG_Code:
                    self.EPSG_Code = epsg_spare
                self.imgDimsMap, self.imgDimsPix = self.RasterDims()

            except:
                self.validMapInfo = False
                self.proj = None
                self.EPSG_Code = None
            self.rpcs = self.raster.GetMetadata('RPC')
            # dataInfo["RPC"] = rpcs
            self.metaData = self.raster.GetMetadata()
            self.bandInfo = []
            self.rasterDataType = self.raster.GetRasterBand(1).DataType
            for band_ in range(self.nbBand):
                self.bandInfo.append(self.raster.GetRasterBand(band_ + 1).GetDescription())
                self.noData = self.raster.GetRasterBand(band_ + 1).GetNoDataValue()

        if printInfo == True:
            print(repr(RasterInfo(self.rasterPath)))

    def ImageAsArray(self, bandNumber=1):
        """

        Args:
            bandNumber:

        Returns:

        """

        raster = self.raster
        # Transform image to array
        imageAsArray = np.array(raster.GetRasterBand(bandNumber).ReadAsArray())
        if self.noData != None:
            imageAsArray = np.ma.masked_where(imageAsArray <= self.noData, imageAsArray)
            try:
                imageAsArray = np.copy(imageAsArray.filled(fill_value=np.nan))
            except:
                imageAsArray = np.copy(imageAsArray)
        return imageAsArray

    def ImageAsArray_Subset(self, xOffsetMin, xOffsetMax, yOffsetMin, yOffsetMax, bandNumber=1):
        """

        Args:
            xOffsetMin:
            xOffsetMax:
            yOffsetMin:
            yOffsetMax:
            bandNumber:

        Returns:
        References: https://gdal.org/python/osgeo.gdal-pysrc.html#Band.ReadAsArray
        """

        raster = self.raster
        xSize = (xOffsetMax - xOffsetMin) + 1
        ySize = (yOffsetMax - yOffsetMin) + 1
        # print("--->",int(xOffsetMin), int(yOffsetMin), int(xSize), int(ySize))
        imageAsArray = np.array(
            raster.GetRasterBand(bandNumber).ReadAsArray(int(xOffsetMin), int(yOffsetMin), int(xSize), int(ySize)))
        return imageAsArray

    def MultiBandsRaster2Array(self):
        # TODO :  Improve with Xarray
        array = np.empty((self.nbBand, self.rasterHeight, self.rasterWidth))
        for i in range(self.nbBand):
            array[i] = self.ImageAsArray(bandNumber=i + 1)

        return array

    def Pixel2Map(self, x, y):
        """
        Convert pixel coordinate to map coordinate,
        Notes:
            The top Left coordinate of the image with GDAl correspond to (0,0)pix
        Args:
            x: xPixel coordinate : int or float
            y: yPixel coordinate: int or float

        Returns: xMap,yMap : tuple  (non integer coordinates)

        """

        rtnX = self.geoTrans[2]
        rtnY = self.geoTrans[4]
        ## Apply affine transformation
        mat = np.array([[self.pixelWidth, rtnX], [rtnY, self.pixelHeight]])
        trans = np.array([[self.xOrigin, self.yOrigin]])
        res = np.dot(mat, np.array([[x, y]]).T) + trans.T
        xMap = res[0].item()
        yMap = res[1].item()
        return (xMap, yMap)

    def Pixel2Map_Batch(self, X, Y):
        """
        Convert pixel coordinate to map coordinate,
        Args:
            X: list of xPixel coordinates : list of int of float
            Y: list of  yPixel coordinates:  list of int or float

        Returns: xMap,yMap : tuple  (non integer coordinates)

        """

        X_map = []
        Y_map = []
        for x, y in zip(X, Y):
            xMap_, yMap_ = self.Pixel2Map(x=x, y=y)
            X_map.append(xMap_)
            Y_map.append(yMap_)
        return (X_map, Y_map)

    def Map2Pixel(self, x, y):
        """
        Convert coordinate from map space to image space
        Args:
            x: xMap coordinate : int or float
            y: yMap coordinate: int or float

        Returns: coordinate in image space : tuple in pix

        """

        ## Apply inverse affine transformation
        rtnX = self.geoTrans[2]
        rtnY = self.geoTrans[4]
        # print(self.pixelWidth, self.pixelHeight)
        ## Apply affine transformation
        mat = np.array([[self.pixelWidth, rtnX], [rtnY, self.pixelHeight]])
        trans = np.array([[self.xOrigin, self.yOrigin]])

        temp = np.array([[x, y]]).T - trans.T
        res = np.dot(np.linalg.inv(mat), temp)
        xPx = res[0].item()
        yPx = res[1].item()

        if xPx < 0 or xPx > self.rasterWidth:
            warnings.warn("xPix outside the image dimension", DeprecationWarning, stacklevel=2)
        if yPx < 0 or yPx > self.rasterHeight:
            warnings.warn("yPix outside the image dimension", DeprecationWarning, stacklevel=2)

        return (xPx, yPx)

    def Map2Pixel_Batch(self, X, Y):
        """
        Convert coordinate from map space to image space
        Args:
            X: list of xMap coordinate : list of int or float
            Y: list of yMap coordinate: list of int or float

        Returns:
            coordinate in image space (X_pix, Y_pix) : tuple in pix

        """
        X_pix = []
        Y_pix = []
        for x, y in zip(X, Y):
            xPix, yPix = self.Map2Pixel(x=x, y=y)
            X_pix.append(xPix)
            Y_pix.append(yPix)
        return (X_pix, Y_pix)

    def RasterDims(self):
        """

        Returns:[upLeftEW,botRightEW,botRightNS,upLeftNS], [x0,xf,yf,y0]

        """

        ds = rasterio.open(self.rasterPath)
        bounds = ds.bounds
        imgDimsMap = {"x0Map": bounds.left,
                      "xfMap": bounds.right,
                      "y0Map": bounds.bottom,
                      "yfMap": bounds.top}
        x0, y0 = self.Map2Pixel(x=imgDimsMap.get("x0Map"), y=imgDimsMap.get("y0Map"))
        xf, yf = self.Map2Pixel(x=imgDimsMap.get("xfMap"), y=imgDimsMap.get("yfMap"))
        imgDimsPix = {"x0Pix": int(x0), "xfPix": int(xf), "y0Pix": int(y0), "yfPix": int(yf)}
        return list(imgDimsMap.values()), list(imgDimsPix.values())

    def __repr__(self):
        return """
        # Raster Information :
            Error = {}
            Raster = {}
            Number  of Bands = {}
            DataType = {}
        # Dimensions :
            Width = {}
            Height = {}
            Resolution = {}
        # Map Information :
            MapInfo = {}
            Geo Transformation  = {}
            EPSG = {}
            Projection = {}
            BandInfo ={}
            Metadata ={}
        # Raster location:
            Raster Name = {}
            Raster path = {}""".format(self.error,
                                       self.raster,
                                       self.nbBand,
                                       self.rasterDataType,
                                       self.rasterWidth,
                                       self.rasterHeight,
                                       self.pixelWidth,
                                       self.validMapInfo,
                                       self.geoTrans,
                                       self.EPSG_Code,
                                       self.proj,
                                       self.bandInfo,
                                       self.metaData,
                                       self.rasterName,
                                       self.rasterPath)


# =====================================================================================================================#

# def WriteRaster(oRasterPath,
#                 geoTransform,
#                 arrayList,
#                 epsg=4326,
#                 dtype=gdal.GDT_Float32,
#                 metaData: Optional[List] = None,
#                 resample_alg=gdal.GRA_Lanczos,
#                 descriptions=None,
#                 noData=None,
#                 progress=False,
#                 driver='GTiff'):
#     """
#
#     Args:
#         oRasterPath:
#         geoTransform:
#         arrayList:
#         epsg:
#         dtype:
#         metaData:
#         resample_alg:
#         descriptions:
#         noData:
#         progress:
#         driver:
#
#     Returns:
#     Notes:
#
#         https://gdal.org/python/osgeo.gdalconst-module.html
#         geoTransfrom it's an affine transformation
#         geoTransform = originX, pixelWidth, rtx, originY,rty, pixelHeight
#     """
#     driver = gdal.GetDriverByName(driver)
#     rows, cols = np.shape(arrayList[0])
#     # print(oRasterPath, cols, rows, len(arrayList), dtype)
#     outRaster = driver.Create(oRasterPath, cols, rows, len(arrayList), dtype,
#                               options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
#     outRaster.SetGeoTransform((geoTransform[0], geoTransform[1], geoTransform[2], geoTransform[3], geoTransform[4],
#                                geoTransform[5]))
#     # dst_ds = driver.CreateCopy(dst_filename, src_ds, strict=0,
#     #                            options=["TILED=YES", "COMPRESS=PACKBITS"])
#     ## Set the projection
#     if epsg != None:
#         outRasterSRS = osr.SpatialReference()
#         outRasterSRS.ImportFromEPSG(epsg)
#         outRaster.SetProjection(outRasterSRS.ExportToWkt())
#
#     outRaster.SetMetadataItem("Author", "SAIF AATI saif@caltech.edu")
#     ## Set the metadata
#     metaData_ = []
#     if metaData is not None:
#         if isinstance(metaData, dict):
#             for key, value in metaData.items():
#                 temp = [key, value]
#                 metaData_.append(temp)
#         elif isinstance(metaData, list):
#             metaData_ = metaData
#
#         for mm in metaData_:
#             # print("mm    ",mm)
#             if not isinstance(mm[1], dict):
#                 if not isinstance(mm[1], str):
#                     str_ = str(mm[1])
#                     outRaster.SetMetadataItem(mm[0], str_)
#                 else:
#                     outRaster.SetMetadataItem(mm[0], mm[1])
#     ## Write the data
#     for i in range(len(arrayList)):
#         outband = outRaster.GetRasterBand(i + 1)
#         outband.WriteArray(arrayList[i], resample_alg=resample_alg)
#         if noData != None:
#             if progress:
#                 print("No data=", noData)
#             outband.SetNoDataValue(noData)
#         if descriptions != None:
#             outband.SetDescription(descriptions[i])
#             # outBand.SetRasterCategoryNames(descriptions[i])
#         if progress:
#             print("Writing band number: ", i + 1, " ", i + 1, "/", len(arrayList))
#
#     outband.FlushCache()
#     outRaster = None
#     return oRasterPath


def GeoTransfomAsArray(geoTrans):
    xOrigin = geoTrans[0]
    yOrigin = geoTrans[3]
    pixelWidth = geoTrans[1]
    pixelHeight = geoTrans[5]
    rtnX = geoTrans[2]  #
    rtnY = geoTrans[4]  #
    ## Apply affine transformation
    trans = np.zeros((3, 3))
    trans[0, 0] = pixelWidth
    trans[0, 1] = rtnX
    trans[1, 0] = rtnY
    trans[1, 1] = pixelHeight
    trans[0, 2] = xOrigin
    trans[1, 2] = yOrigin
    trans[2, 2] = 1

    return trans


# =====================================================================================================================#

def DataMemory(dataType):
    gdalTypesLib = [gdal.GDT_Unknown, gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
                    gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32,
                    gdal.GDT_CFloat64]

    dataTypeLib = {"uint8": 1, "int8": 1, "uint16": 2, "int16": 3, "uint32": 4, "int32": 5,
                   "float32": 6, "float64": 7, "complex64": 10, "complex128": 1}
    dataTypeMemo = {"uint8": 8, "int8": 8, "uint16": 16, "int16": 16, "uint32": 32, "int32": 32,
                    "float32": 32, "float64": 64, "complex64": 64, "complex128": 128}
    for index, val in enumerate(dataTypeLib.items()):
        if dataType == val[1]:
            # print(index,val)
            memoType = dataTypeMemo.get(val[0])
    return memoType


def ParseResampMethod(method):
    """
    Parse resampling method
    Args:
        method:

    Returns:

    """

    if method == 'near':
        # Note: Nearest respects nodata when downsampling
        return gdal.GRA_NearestNeighbour
    elif method == 'bilinear':
        return gdal.GRA_Bilinear
    elif method == 'cubic':
        return gdal.GRA_Cubic
    elif method == 'cubicspline':
        return gdal.GRA_CubicSpline
    elif method == 'average':
        return gdal.GRA_Average
    elif method == 'lanczos':
        return gdal.GRA_Lanczos
    elif method == 'mode':
        # Note: Mode respects nodata when downsampling, but very slow
        return gdal.GRA_Mode
    elif method == 'sinc':
        print("Not implemented yet, it is the sinc method implmented in cosi-corr")
        sys.exit("Invalid resampling method")
    else:
        algo = None
        sys.exit("Invalid resampling method")


# def ComputeEpsg(lon, lat):
#     """
#     Compute the EPSG code of the UTM zone which contains
#     the point with given longitude and latitude
#
#     Args:
#         lon : longitude
#         lat : latitude
#
#     Returns:
#         EPSG code
#     Notes:
#         UTM zone number starts from 1 at longitude -180, and increments by 1 every 6 degrees of longitude
#
#         EPSG = CONST + ZONE where CONST is
#         - 32600 for positive latitudes
#         - 32700 for negative latitudes
#     """
#
#     zone = int((lon + 180) // 6 + 1)
#     const = 32600 if lat > 0 else 32700
#     return const + zone
#

def Set_crs(epsg=4326):
    """

    Args:
        epsg:

    Returns:

    """
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    print("-----", outRasterSRS)

    return outRasterSRS


def GetWindow(iRasterPath, windowGeo):
    """

    Args:
        iRasterPath:
        windowGeo:

    Returns:

    """
    # ShisperWindow = 74.467995439,74.653396304,36.302775965,36.485637092 [EPSG:4326]
    rasterInfo = RasterInfo(inputRaster=iRasterPath)
    topLeft = rasterInfo.Map2Pixel(windowGeo[0], windowGeo[-1])
    topRight = rasterInfo.Map2Pixel(windowGeo[1], windowGeo[-1])
    bottumLeft = rasterInfo.Map2Pixel(windowGeo[0], windowGeo[2])
    bottumRight = rasterInfo.Map2Pixel(windowGeo[1], windowGeo[2])
    width = np.abs(topRight[0] - topLeft[0])
    height = np.abs(topRight[1] - bottumRight[1])
    colOff = topLeft[0]
    rowOff = topLeft[1]

    return [int(colOff), int(rowOff), int(width), int(height)]


def ProjDatumIdentical(proj1, proj2):
    print("Check if both projection are identical")
    print("---- Working progress ----")
    print("--- Error! Sorry!---")
    return False


def CreateProj(epsgCode=4326):
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(epsgCode)
    # proj = target.GetProjection()
    return proj


# =====================================================================================================================#
def Create_MultiBandsRaster_form_MultiRasters(rastersList, outputPath, bandNumber=1):
    """

    Args:
        rastersList:
        outputPath:
        bandNumber:

    Returns:

    """

    arrayList = []
    bandDescription = []
    for img_ in rastersList:
        rasterInfo = RasterInfo(img_)
        array = rasterInfo.ImageAsArray(bandNumber=bandNumber)
        arrayList.append(array)
        bandDescription.append("Band" + str(bandNumber) + "_" + Path(img_).stem)
    WriteRaster(oRasterPath=outputPath, geoTransform=rasterInfo.geoTrans, arrayList=arrayList,
                epsg=rasterInfo.EPSG_Code, descriptions=bandDescription)
    # WriteRaster(refRasterPath=rastersList[0], newRasterPath=outputPath, Listarrays=arrayList,
    #             numberOfBands=len(rastersList), descriptions=bandDescription)
    return outputPath


# FIXME
def Create_MultiRasters_from_MultiBandsRaster(inputRaster, output):
    """

    Args:
        inputRaster:
        output:

    Returns:

    """

    # rasterInfo = GetRasterInfo(inputRaster)
    rasterInfo = RasterInfo(inputRaster)
    print(rasterInfo["NbBands"])
    for i in range(rasterInfo["NbBands"]):
        # array = ImageAsArray(rasterInfo, i + 1)

        array = rasterInfo.ImageAsArray(bandNumber=i + 1)
        WriteRaster(oRasterPath=os.path.join(output, "Img_Band_" + str(i + 1) + ".tif"),
                    geoTransform=rasterInfo.geoTrans, arrayList=[array],
                    epsg=rasterInfo.EPSG_Code)

    return


# =====================================================================================================================#
def SubsetRasters(rasterList, areaCoord, outputFolder=None, vrt=False, outputType=gdal.GDT_Float32):
    """

    Args:
        rasterList:
        areaCoord:
        outputFolder:
        vrt:
        outputType:

    Returns:

    """
    if vrt:
        format = "VRT"
    else:
        format = "GTiff"
    params = gdal.TranslateOptions(projWin=areaCoord, format=format, outputType=outputType, noData=-32767)
    path = os.path.dirname(rasterList[0])
    oList = []
    for img_ in rasterList:
        if outputFolder == None:
            outputFile = os.path.join(path, Path(img_).stem + "_crop")
        else:
            outputFile = os.path.join(outputFolder, Path(img_).stem + "_crop")

        if vrt == True:
            gdal.Translate(destName=os.path.join(outputFile + ".vrt"),
                           srcDS=gdal.Open(img_),
                           options=params)
            oList.append(os.path.join(outputFile + ".vrt"))

        else:
            gdal.Translate(destName=os.path.join(outputFile + ".tif"),
                           srcDS=gdal.Open(img_),
                           options=params)
            oList.append(os.path.join(outputFile + ".tif"))

    return oList


# =====================================================================================================================#
def ConvCoordMap1ToMap2(x, y, targetEPSG, z=None, sourceEPSG=4326, display=False):
    """
    convert point coordinates from source to target system
    Args:
        x:  map coordinate (e.g lon,lat)
        y:
        targetEPSG: target coordinate system of the point; integer
        z:
        sourceEPSG:  source coordinate system of the point; integer (default geographic coordinate )
        display:

    Returns:
        point in target coordinate system; list =[xCoord,yCoord,zCoord]
    Notes:
        - If the transformation from WGS to UTM, x= lat, y=lon ==> coord =(easting(xMap) ,northing(yMap))

    """

    ## Set the source system
    source = osr.SpatialReference()  # instance from SpatialReference Class
    source.ImportFromEPSG(int(sourceEPSG))  # create a projection system based on EPSG code

    ## Set the target system
    target = osr.SpatialReference()
    target.ImportFromEPSG(int(targetEPSG))

    transform = osr.CoordinateTransformation(source,
                                             target)  # instance of the class Coordinate Transformation
    if z == None:
        coord = transform.TransformPoint(x, y)

    else:
        coord = transform.TransformPoint(x, y, z)
    if display:
        print("{}, {}, EPSG={} ----> Easting:{}  , Northing:{} , EPSG={}".format(x, y, sourceEPSG, coord[0], coord[1],
                                                                                 targetEPSG))
    return coord


def ConvertRaster2WGS84(inputPath, outputPath=None):
    """

    Args:
        inputPath:
        outputPath:

    Returns:

    """
    if outputPath == None:
        outputPath = Path(inputPath).stem + "_conv_4326.tif"

    gdal.Warp(outputPath, inputPath, dstSRS='EPSG:4326')
    return outputPath


def ReprojectRaster(iRasterPath, oPrj, vrt=True, oRasterPath=None):
    """
    Reproject a raster to a neaw projection system
    Args:
        iRasterPath:
        oPrj:
        oRasterPath:

    Returns: oRasterPath

    """
    if oRasterPath == None:
        oRasterPath = os.path.join(os.path.dirname(iRasterPath), Path(iRasterPath).stem + "_" + str(oPrj) + ".tif")
        if vrt:
            oRasterPath = os.path.join(os.path.dirname(iRasterPath), Path(iRasterPath).stem + "_" + str(oPrj) + ".vrt")
    # print(oRasterPath)
    warpOptions = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs epsg:" + str(oPrj)))
    # gdal.Warp(oRasterPath, iRasterPath, dstSRS='EPSG:'+str(oPrj))#options=warpOptions)
    gdal.Warp(oRasterPath, iRasterPath, options=warpOptions)

    return oRasterPath


def ConvCoordMap1ToMap2_Batch(X, Y, targetEPSG, Z=[], sourceEPSG=4326):
    """
    Convert point coordinates from source to target system

    Args:
        X:map coordinate (e.g lon,lat)
        Y:map coordinate (e.g lon,lat)
        targetEPSG:target coordinate system of the point: integer
        Z:map coordinates
        sourceEPSG:source coordinate system of the point: integer (default geographic coordinate )

    Returns:         point in target coordinate system; list =[xCoord,yCoord,zCoord] or ([lats],[lons])

    Notes:
        - if the transformation from WGS to UTM, x= lat, y=lon ==> coord =(easting(xMap) ,northing(yMap))
        - if the transformation from UTM to WGS 84, x=easting, y=Notthing ==> lat, long

    """

    sourceEPSG_string = "epsg:" + str(sourceEPSG)
    targetEPSG_string = "epsg:" + str(targetEPSG)
    transformer = pyproj.Transformer.from_crs(sourceEPSG_string, targetEPSG_string)
    if len(Z) == 0:
        return transformer.transform(X, Y)

    else:
        return transformer.transform(X, Y, Z)


def ConvertGeo2Cartesian(Lon, Lat, Alt, method="pyprj"):
    """

    Args:
        Lon: list []
        Lat: list []
        Alt: list  []
        method: pyprj, custom

    Returns: X_cart, y_cart, Z_cart

    Notes:
        https://pyproj4.github.io/pyproj/dev/api/proj.html
    """
    # TODO: the conversion is performed using pyproj. House implementation could be used.
    ## (see IDL version: convert_geographic_to_cartesian)
    if method == "pyprj":
        transproj = pyproj.Transformer.from_crs("EPSG:4326", {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                                always_xy=True)
        X_cart, Y_cat, Z_cart = transproj.transform(Lon, Lat, Alt, radians=False)
        if len(Lon) == 1 and len(Lat) == 1 and len(Alt) == 1:
            return [X_cart[0], Y_cat[0], Z_cart[0]]
        return X_cart, Y_cat, Z_cart
    elif method == "custom":
        radLat = Lat * (np.pi / 180.0)
        radLon = Lon * (np.pi / 180.0)
        a = 6378137.0
        finv = 298.257223563
        f = 1 / finv
        e2 = 1 - (1 - f) * (1 - f)
        v = a / np.sqrt(1 - e2 * np.sin(radLat) * np.sin(radLat))

        X_cart = (v + Alt) * np.cos(radLat) * np.cos(radLon)
        Y_cart = (v + Alt) * np.cos(radLat) * np.sin(radLon)
        Z_cart = (v * (1 - e2) + Alt) * np.sin(radLat)
        if len(Lon) == 1 and len(Lat) == 1 and len(Alt) == 1:
            return [X_cart[0], Y_cart[0], Z_cart[0]]
        return X_cart, Y_cart, Z_cart
    else:
        sys.exit("Error: Conversion from WG84 --> Cartesian ! ")


def ConvertCartesian2Geo(x, y, z):
    """

    Args:
        x:
        y:
        z:

    Returns:
    Notes:
        # ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    # # print(ecef)
    # lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    # # print(lla)
    # # transformer = pyproj.Transformer.from_crs(lla, ecef)
    # x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
    # print( [x,y,z])
    https://pyproj4.github.io/pyproj/dev/api/proj.html
    """

    transproj = pyproj.Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, "EPSG:4326",
                                            always_xy=True)
    lon, lat, alt = transproj.transform(x, y, z, radians=False)
    return [lon, lat, alt]


def ConvertCartesian2Geo_Batch(X, Y, Z):
    transproj = pyproj.Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, "EPSG:4326",
                                            always_xy=True)
    Lon, Lat, Alt = transproj.transform(X, Y, Z, radians=False)
    return Lon, Lat, Alt


def BoundingBox2D(pts):
    """
    Rectangular bounding box for a list of 2D points.
    Args:
        pts:  list of 2D points represented as 2-tuples or lists of length 2 : list

    Returns:
        [x, y, w, h]: coordinates of the top-left corner, width and height of the bounding box : list

    """

    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return [bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]]


def ReadGeojson(geojsonPath):
    import geojson
    with open(geojsonPath) as f:
        data = geojson.load(f)
    return data


# TODO: move to constant as a class
def VectorDrivers(driver):
    if driver == "GeoJSON":
        return ".geojson"
    if driver == "KML":
        return ".kml"
    if driver == "ESRI Shapefile":
        return ".shp"


def WriteJson(features, outputFile, driver="GeoJSON"):
    """

    Args:
        features:
        outputFile:
        driver:

    Returns:

    """
    import geojson

    outputFile = outputFile + VectorDrivers(driver)
    # print(features)
    with open(outputFile, 'w') as outfile:
        geojson.dump(features, outfile, sort_keys=True)
    return outputFile


def check_same_projection(img_i_info: RasterInfo, img_j_info: RasterInfo):
    if img_i_info.EPSG_Code != img_j_info.EPSG_Code:
        warnings.warn("reference Ortho and reference DEM must have the same projection system")
        msg = "Reproject DEM from {}-->{}".format(img_i_info.EPSG_Code, img_j_info.EPSG_Code)
        warnings.warn(msg)
        # raise Exception("reference Ortho and reference DEM must have the same projection system ")
        return False
    else:
        return True


def reproject_to_same_epsg(img_i_info: RasterInfo, img_j_info: RasterInfo):
    """
    reproject img_j to the same projection system of img_i
    Args:
        img_i_info:
        img_j_info:

    Returns:

    """
    if check_same_projection(img_i_info, img_j_info) == False:
        prjDemPath = ReprojectRaster(iRasterPath=img_j_info.rasterPath,
                                     oPrj=img_i_info.EPSG_Code,
                                     vrt=True)
        return RasterInfo(prjDemPath)

    else:
        return img_j_info
