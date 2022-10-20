"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import ctypes, ctypes.util
import sys, math, warnings
import numpy as np

import geoCosiCorr3D.geoErrorsWarning.geoErrors as geoErrors
from geoCosiCorr3D.geoConfig import cgeoCfg
import geoCosiCorr3D.georoutines.geo_utils as geoRT

geoCfg = cgeoCfg()


class cResamplingEngine:

    def __init__(self, method="bilinear", kernelSz=15, debug=False):
        """

        Args:
            method: bilinear, bicubic, sinc
            kernelSz:
        """
        self.method = method
        if kernelSz < 0 or kernelSz % 2 == 0:
            sys.exit("Kernel size must be positive odd number")
        else:
            self.kernelSz = kernelSz
        # ToDo: engine: c++, scipy, ....
        self.engine = ""
        self.debug = debug
        if self.debug:
            print("cResamplingEngine::", self.method)

    def ComputeResamplingDistance(self, matrix_x, matrix_y, sz):

        ## Get the average resampling distance in X over the current tile in X, Y and diagonal directions.
        # Take the maximum
        dx1 = (matrix_x[0, 0] - matrix_x[0, sz[1] - 1]) / sz[1]
        dx2 = (matrix_x[0, 0] - matrix_x[sz[0] - 1, 0]) / sz[0]

        if sz[1] > sz[0]:
            dx3 = (matrix_x[0, 0] - matrix_x[sz[0] - 1, sz[0] - 1]) / sz[0]
        else:
            dx3 = (matrix_x[0, 0] - matrix_x[sz[1] - 1, sz[1] - 1]) / sz[1]
            ## Note:
            # 10 =max variability of the resampling distance,
            # 1.15= oversampling for kernel edge response
        borderX = int(np.ceil(np.max(np.abs([dx1, dx2, dx3])) * self.kernelSz * 10 * 1.15))
        # print("dx1:{},dx2:{},dx3:{}".format(dx1, dx2,dx3))
        # print("borderX:", borderX)

        ## Get the average resampling distance in Y over the current tile in X, Y and diagonal directions.
        ## Take the maximum
        dy1 = (matrix_y[0, 0] - matrix_y[0, sz[1] - 1]) / sz[1]
        dy2 = (matrix_y[0, 0] - matrix_y[sz[0] - 1, 0]) / sz[0]

        if sz[1] > sz[0]:
            dy3 = (matrix_y[0, 0] - matrix_y[sz[0] - 1, sz[0] - 1]) / sz[0]
        else:
            dy3 = (matrix_y[0, 0] - matrix_y[sz[1] - 1, sz[1] - 1]) / sz[1]
            ## Note:
            # 10 =max variability of the resampling distance,
            # 1.15= oversampling for kernel edge response
        borderY = int(np.ceil(np.max(np.abs([dy1, dy2, dy3])) * self.kernelSz * 10 * 1.15))
        # print("dy1:{},dy2:{},dy3:{}".format(dy1, dy2, dy3))
        # print("borderY:", borderY)
        return borderX, borderY

    def ComputeL1AImageTileSubset(self, rasterInfo: geoRT.cRasterInfo, matrix_x, matrix_y, margin=3):
        """
        Define the necessary image subset Dimensions needed for the resampling,
        depending on the resample engine selected.
        Args:
            matrix_x: X-pixel coordinates : array
            matrix_y: Y-pixel coordinates : array
            margin: margin added around the subset for interpolation purpose :int

        Returns:

        """
        ### In order to define the necessary image subset to process the current matrix tile
        ### we need the matrix_x min and max,
        ### as well an estimate of the resampling distance in case of Sinc resampling engine.
        if np.isnan(matrix_x).all() or np.isnan(matrix_y).all():
            sys.exit("MATRIX_X or MATRIX_Y ALL NANs")
        if self.debug:
            logging.info(
                f'Resampling: {np.nanmin(matrix_x), np.nanmax(matrix_x), np.nanmin(matrix_y), np.nanmax(matrix_y)}')
        minX = math.floor(np.nanmin(matrix_x))
        maxX = math.ceil(np.nanmax(matrix_x))
        minY = math.floor(np.nanmin(matrix_y))
        maxY = math.ceil(np.nanmax(matrix_y))
        if self.debug:
            print(np.nanmin(matrix_x), np.nanmax(matrix_x)), print(np.nanmin(matrix_y), np.nanmax(matrix_y))
            print("  minX:{}, minY:{}, maxX:{}, maxY:{}".format(minX, minY, maxX, maxY))

        sz = matrix_x.shape

        ni_x = rasterInfo.raster_width
        ni_y = rasterInfo.raster_height
        dims_img = [0, ni_x - 1, 0, ni_y - 1]
        ## Compute the necessary image subset dimension
        dims1 = dims_img[0]
        dims2 = dims_img[1]
        dims3 = dims_img[2]
        dims4 = dims_img[3]
        if (minX - margin) > dims_img[0]:
            dims1 = minX - margin
        if (maxX + margin) < dims_img[1]:
            dims2 = (maxX + margin)
        if (minY - margin) > dims_img[2]:
            dims3 = (minY - margin)
        if (maxY + margin) < dims_img[3]:
            dims4 = (maxY + margin)  # 3 border necessary for bi-linear or bicubic

        if self.method == "sinc":
            borderX, borderY = self.ComputeResamplingDistance(matrix_x, matrix_y, sz)
            # print("borderX:{},borderY:{}".format(borderX,borderY))
            if (minX - borderX) > dims_img[0]:
                dims1 = minX - borderX
            if (maxX + borderX) < dims_img[1]:
                dims2 = (maxX + borderX)
            if (minY - borderY) > dims_img[2]:
                dims3 = (minY - borderY)
            if (maxY + borderY) < dims_img[3]:
                dims4 = (maxY + borderY)

        # print("-- not sinc --- ")
        # print(dims1, dims2, dims3, dims4)
        ## Check for situation where the entire current matrice tile is outside image boundaries
        ## In that case need to output a zero array, either on file or in the output array, and
        ## continue to the next tile
        if (dims1 > dims2) or (dims3 > dims4):
            print(dims1, dims2, dims3, dims4)
            # TODO: add warning
            warnings.warn("# @SA: potential Error, I did not test this condition so far.")
            logging.warning("# @SA: potential Error, I did not test this condition so far.")
            return np.zeros(sz)
        dims = [int(dims1), int(dims2), int(dims3), int(dims4)]
        if self.debug:
            logging.info(f'dims:{dims}')
        return dims

    def fAdapSincResampling(self, matrix_x, matrix_y, im1A, dims, weigthing=1):
        """

        Args:
            matrix_x:
            matrix_y:
            im1A:
            dims:
            weigthing:

        Returns:

        """

        sz = matrix_x.shape
        if self.debug:
            print("sz:", sz)

        libPath_ = ctypes.util.find_library(geoCfg.geoCosiCorr3DLib)

        if not libPath_:
            geoErrors.erLibNotFound(libPath=geoCfg.geoCosiCorr3DLib)
        try:
            sincLib = ctypes.CDLL(libPath_)
            matCol = np.array(matrix_y, dtype=np.float_)
            matRow = np.array(matrix_x, dtype=np.float_)

            # print("matrix_x[15,3]:{},matrix_y[15,3]:{}".format(matrix_x[15,3],matrix_y[15,3]))

            img = np.array(im1A, dtype=np.float_)
            width = ctypes.c_int(self.kernelSz)
            oImg = np.zeros(sz, dtype=np.float)
            weighting = ctypes.c_int(weigthing)
            nbColMat = ctypes.c_int(sz[0])
            nbRowMat = ctypes.c_int(sz[1])

            nbColImg = ctypes.c_int(dims[3] - dims[2] + 1)
            nbRowImg = ctypes.c_int(dims[1] - dims[0] + 1)

            sincLib.main_sinc_adp_resampling_(
                matCol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                matRow.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(width),
                oImg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(weighting),
                ctypes.byref(nbColMat),
                ctypes.byref(nbRowMat),
                ctypes.byref(nbColImg),
                ctypes.byref(nbRowImg))

            return oImg
        except OSError:
            geoErrors.erLibLoading(geoCfg.geoCosiCorr3DLib)


class cResamapling:
    def __init__(self, rasterInfo: geoRT.cRasterInfo,
                 tranMatTile: np.ndarray,
                 resamplingEngine: cResamplingEngine = None,
                 debug: bool = False):
        """
        Resampling an image according to transformation matrices using the selected
        resampling kernel (e.g., sinc ,bilinear,bicubic,)
        Args:
            rasterInfo: object rasterInfo(geoRoutines) of the image to resample.
            tranMatTile: transformation matrix [2,nbCols,nbRows]
            resamplingEngine: resampling engine
                - bilinera : default selection
                -  bicubic: #TODO
                - sinc
        Notes:
            TODO: handle multi-bands
        """
        self.rasterInfo = rasterInfo
        self.tranMatTile = tranMatTile
        self.resamplingEngine = resamplingEngine
        self.debug = debug
        self.resamplingEngine = resamplingEngine
        if resamplingEngine is None:
            self.resamplingEngine = cResamplingEngine(debug=self.debug)

    def ResamplingImgCore(self):
        if self.debug:
            logging.info(
                "____________________ Resampling::{} ____________________________".format(self.resamplingEngine.method))

        nbBands = self.rasterInfo.band_number
        ##Fixme: force band = 1
        if nbBands > 1:
            msg = "Multi-band image: This version does not support multi-band ortho-rectification, only band 1 will be orthorectified "
            warnings.warn(msg)
            logging.warning(msg)
            nbBands = 1

        # Definition of the matrices dimensions
        dims_geom = [0, self.tranMatTile.shape[2] - 1, 0, self.tranMatTile.shape[1] - 1]

        # Definition of the resampling kernel

        apodization = 1  # sinc kernel apodization (1 by default, with no option for user)

        if nbBands == 1:
            oOrthoTile = np.zeros((dims_geom[3] - dims_geom[2] + 1, dims_geom[1] - dims_geom[0] + 1))

        else:
            geoErrors.erNotImplemented(routineName="This version does not support multi-band ortho-rectification")
        # print("  oOrthoTile.shape:", oOrthoTile.shape)

        matrix_x = self.tranMatTile[0, :, :]
        matrix_y = self.tranMatTile[1, :, :]
        sz = matrix_x.shape
        if self.debug:
            # print(matrix_x[5, :])
            # print(matrix_y[5, :])
            logging.info(f'sz:{sz}')

        dims = self.resamplingEngine.ComputeL1AImageTileSubset(self.rasterInfo, matrix_x, matrix_y)
        if self.debug:
            logging.info(f'Reseampling::dims:{dims}')

        im1A = self.rasterInfo.image_as_array_subset(dims[0],
                                                     dims[1],
                                                     dims[2],
                                                     dims[3], band_number=nbBands)

        ## Correct the matrices coordinates for the subsetting of the extracted image
        matrix_x = matrix_x - dims[0]
        matrix_y = matrix_y - dims[2]
        # print("im1A.shape=", im1A.shape)
        #
        # plt.figure()
        # plt.imshow(im1A, cmap="gray")  # , origin="lower")
        # plt.title("Image to resample: {},{}".format(im1A.shape[0], im1A.shape[1]))
        # plt.show()
        # print(im1A[1100, 1222])
        # sys.exit() # checkPoint: valid
        if self.resamplingEngine.method == "sinc":
            imgL3b = self.resamplingEngine.fAdapSincResampling(matrix_x, matrix_y, im1A, dims)
            # print("__________________________________ END Resampling _________________________________________")
            return imgL3b

        if self.resamplingEngine.method == "bilinear":
            # imgL3b_fl = interpRT.Interpolate2D(inArray=im1A, x=matrix_y.flatten(), y=matrix_x.flatten(), kind="linear")
            from scipy.interpolate import interpolate
            # print("Resampling ....")
            nbRows, nbCols = im1A.shape[0], im1A.shape[1]
            ## Note: since we have corrected matrix_x nad matrix_y coordinates the interpolation is then done 0,nbRows and 0,nbCols
            f = interpolate.RegularGridInterpolator(
                # (np.arange(dims[2], dims[3] + 1, 1), np.arange(dims[0], dims[1] + 1, 1)),
                (np.arange(0, nbRows, 1), np.arange(0, nbCols, 1)),
                im1A,
                method="linear",
                bounds_error=False,
                fill_value=np.nan)
            imgL3b_fl = f(list(zip(matrix_y.flatten(), matrix_x.flatten())))
            imgL3b = np.reshape(imgL3b_fl, sz)
            # # print(imgL3b.shape)
            # plt.imshow(imgL3b, cmap="gray")
            # plt.show()
            # print("__________________________________ END Resampling _________________________________________")
            return imgL3b
        if self.resamplingEngine == "bicubic":
            ## TODO: add warning
            geoErrors.erNotImplemented("bicubic")

        return


########################################################################################################################

class cResamplingEngine_Dev:

    def __init__(self, method="bilinear", kernelSz=15, debug=False):
        """

        Args:
            method: bilinear, bicubic, sinc
            kernelSz:
        """
        self.method = method
        if kernelSz < 0 or kernelSz % 2 == 0:
            sys.exit("Kernel size must be positive odd number")
        else:
            self.kernelSz = kernelSz
        # ToDo: engine: c++, scipy, ....
        self.engine = ""
        self.debug = debug
        if self.debug:
            print("cResamplingEngine::", self.method)

    def ComputeResamplingDistance(self, matrix_x, matrix_y, sz):

        ## Get the average resampling distance in X over the current tile in X, Y and diagonal directions.
        # Take the maximum
        dx1 = (matrix_x[0, 0] - matrix_x[0, sz[1] - 1]) / sz[1]
        dx2 = (matrix_x[0, 0] - matrix_x[sz[0] - 1, 0]) / sz[0]

        if sz[1] > sz[0]:
            dx3 = (matrix_x[0, 0] - matrix_x[sz[0] - 1, sz[0] - 1]) / sz[0]
        else:
            dx3 = (matrix_x[0, 0] - matrix_x[sz[1] - 1, sz[1] - 1]) / sz[1]
            ## Note:
            # 10 =max variability of the resampling distance,
            # 1.15= oversampling for kernel edge response
        borderX = int(np.ceil(np.max(np.abs([dx1, dx2, dx3])) * self.kernelSz * 10 * 1.15))
        # print("dx1:{},dx2:{},dx3:{}".format(dx1, dx2,dx3))
        # print("borderX:", borderX)

        ## Get the average resampling distance in Y over the current tile in X, Y and diagonal directions.
        ## Take the maximum
        dy1 = (matrix_y[0, 0] - matrix_y[0, sz[1] - 1]) / sz[1]
        dy2 = (matrix_y[0, 0] - matrix_y[sz[0] - 1, 0]) / sz[0]

        if sz[1] > sz[0]:
            dy3 = (matrix_y[0, 0] - matrix_y[sz[0] - 1, sz[0] - 1]) / sz[0]
        else:
            dy3 = (matrix_y[0, 0] - matrix_y[sz[1] - 1, sz[1] - 1]) / sz[1]
            ## Note:
            # 10 =max variability of the resampling distance,
            # 1.15= oversampling for kernel edge response
        borderY = int(np.ceil(np.max(np.abs([dy1, dy2, dy3])) * self.kernelSz * 10 * 1.15))
        # print("dy1:{},dy2:{},dy3:{}".format(dy1, dy2, dy3))
        # print("borderY:", borderY)
        return borderX, borderY

    def ComputeL1AImageTileSubset(self, rasterInfo, matrix_x, matrix_y, margin=3):
        """
        Define the necessary image subset Dimensions needed for the resampling,
        depending on the resample engine selected.
        Args:
            matrix_x: X-pixel coordinates : array
            matrix_y: Y-pixel coordinates : array
            margin: margin added around the subset for interpolation purpose :int

        Returns:

        """
        ### In order to define the necessary image subset to process the current matrix tile
        ### we need the matrix_x min and max,
        ### as well an estimate of the resampling distance in case of Sinc resampling engine.
        if np.isnan(matrix_x).all() or np.isnan(matrix_y).all():
            sys.exit("MATRIX_X or MATRIX_Y ALL NANs")
        if self.debug:
            print("Resampling::", np.nanmin(matrix_x), np.nanmax(matrix_x), np.nanmin(matrix_y), np.nanmax(matrix_y))
        minX = math.floor(np.nanmin(matrix_x))
        maxX = math.ceil(np.nanmax(matrix_x))
        minY = math.floor(np.nanmin(matrix_y))
        maxY = math.ceil(np.nanmax(matrix_y))
        if self.debug:
            print(np.nanmin(matrix_x), np.nanmax(matrix_x)), print(np.nanmin(matrix_y), np.nanmax(matrix_y))
            print("  minX:{}, minY:{}, maxX:{}, maxY:{}".format(minX, minY, maxX, maxY))

        sz = matrix_x.shape

        ni_x = rasterInfo.rasterWidth
        ni_y = rasterInfo.rasterHeight
        dims_img = [0, ni_x - 1, 0, ni_y - 1]
        ## Compute the necessary image subset dimension
        dims1 = dims_img[0]
        dims2 = dims_img[1]
        dims3 = dims_img[2]
        dims4 = dims_img[3]
        if (minX - margin) > dims_img[0]:
            dims1 = minX - margin
        if (maxX + margin) < dims_img[1]:
            dims2 = (maxX + margin)
        if (minY - margin) > dims_img[2]:
            dims3 = (minY - margin)
        if (maxY + margin) < dims_img[3]:
            dims4 = (maxY + margin)  # 3 border necessary for bi-linear or bicubic

        if self.method == "sinc":
            borderX, borderY = self.ComputeResamplingDistance(matrix_x, matrix_y, sz)
            # print("borderX:{},borderY:{}".format(borderX,borderY))
            if (minX - borderX) > dims_img[0]:
                dims1 = minX - borderX
            if (maxX + borderX) < dims_img[1]:
                dims2 = (maxX + borderX)
            if (minY - borderY) > dims_img[2]:
                dims3 = (minY - borderY)
            if (maxY + borderY) < dims_img[3]:
                dims4 = (maxY + borderY)

        # print("-- not sinc --- ")
        # print(dims1, dims2, dims3, dims4)
        ## Check for situation where the entire current matrice tile is outside image boundaries
        ## In that case need to output a zero array, either on file or in the output array, and
        ## continue to the next tile
        if (dims1 > dims2) or (dims3 > dims4):
            print(dims1, dims2, dims3, dims4)
            # TODO: add warning
            warnings.warn("# @SA: potential Error, I did not test this condition so far.")
            # geoErrors.erNotImplemented()
            return np.zeros(sz)
        dims = [int(dims1), int(dims2), int(dims3), int(dims4)]
        if self.debug:
            print("dims:", dims)
        return dims

    def fAdapSincResampling(self, matrix_x, matrix_y, im1A, dims, weigthing=1):
        """

        Args:
            matrix_x:
            matrix_y:
            im1A:
            dims:
            weigthing:

        Returns:

        """

        sz = matrix_x.shape
        if self.debug:
            print("sz:", sz)

        # libPath = "/home/cosicorr/0-WorkSpace/3-PycharmProjects/geoCosiCorr3D_Lib/SincResampler/adaptsinc_fortran/Test/lib.so"
        libPath_ = ctypes.util.find_library(geoCfg.geofAdapSincResamplingLib)
        if not libPath_:
            geoErrors.erLibNotFound(libPath=geoCfg.geofAdapSincResamplingLib)
        try:
            sincLib = ctypes.CDLL(libPath_)
            matCol = np.array(matrix_y, dtype=np.float_)
            matRow = np.array(matrix_x, dtype=np.float_)

            # print("matrix_x[15,3]:{},matrix_y[15,3]:{}".format(matrix_x[15,3],matrix_y[15,3]))

            img = np.array(im1A, dtype=np.float_)
            width = ctypes.c_int(self.kernelSz)
            oImg = np.zeros(sz, dtype=np.float)
            weighting = ctypes.c_int(weigthing)
            nbColMat = ctypes.c_int(sz[0])
            nbRowMat = ctypes.c_int(sz[1])

            nbColImg = ctypes.c_int(dims[3] - dims[2] + 1)
            nbRowImg = ctypes.c_int(dims[1] - dims[0] + 1)

            sincLib.main_sinc_adp_resampling_(
                matCol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                matRow.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(width),
                oImg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(weighting),
                ctypes.byref(nbColMat),
                ctypes.byref(nbRowMat),
                ctypes.byref(nbColImg),
                ctypes.byref(nbRowImg))

            return oImg
        except OSError:
            geoErrors.erLibLoading()


class cResamapling_Dev:
    def __init__(self, rasterInfo, tranMatTile, resamplingEngine=None, debug=False):
        """
        Resampling an image according to transformation matrices using the selected
        resampling kernel (e.g., sinc ,bilinear,bicubic,)
        Args:
            rasterInfo: object rasterInfo(geoRoutines) of the image to resample.
            tranMatTile: transformation matrix [2,nbCols,nbRows]
            resamplingEngine: resampling engine
                - bilinera : default selection
                -  bicubic: #TODO
                - sinc
        Notes:
            TODO: handle multi-bands
        """
        self.rasterInfo = rasterInfo
        self.tranMatTile = tranMatTile
        self.resamplingEngine = resamplingEngine
        self.debug = debug
        self.resamplingEngine = resamplingEngine
        if resamplingEngine == None:
            self.resamplingEngine = cResamplingEngine(debug=self.debug)

    def ResamplingImgCore(self):
        if self.debug:
            print(
                "____________________ Resampling::{} ____________________________".format(self.resamplingEngine.method))

        nbBands = self.rasterInfo.nbBand
        ##Fixme: force band = 1
        if nbBands > 1:
            warnings.warn(
                "Multi-band image: This version does not support multi-band ortho-rectification, "
                "only band 1 will be orthorectified ")
            nbBands = 1

        # Definition of the matrices dimensions
        dims_geom = [0, self.tranMatTile.shape[2] - 1, 0, self.tranMatTile.shape[1] - 1]

        # Definition of the resampling kernel

        apodization = 1  # sinc kernel apodization (1 by default, with no option for user)

        if nbBands == 1:
            oOrthoTile = np.zeros((dims_geom[3] - dims_geom[2] + 1, dims_geom[1] - dims_geom[0] + 1))

        else:
            geoErrors.erNotImplemented(routineName="This version does not support multi-band ortho-rectification")
        # print("  oOrthoTile.shape:", oOrthoTile.shape)

        matrix_x = self.tranMatTile[0, :, :]
        matrix_y = self.tranMatTile[1, :, :]
        sz = matrix_x.shape
        if self.debug:
            # print(matrix_x[5, :])
            # print(matrix_y[5, :])
            print("sz:", sz)
        dims = self.resamplingEngine.ComputeL1AImageTileSubset(self.rasterInfo, matrix_x, matrix_y)
        if self.debug:
            print("Reseampling::dims:", dims)
        im1A = self.rasterInfo.ImageAsArray_Subset(bandNumber=nbBands,
                                                   xOffsetMin=dims[0],
                                                   xOffsetMax=dims[1],
                                                   yOffsetMin=dims[2],
                                                   yOffsetMax=dims[3])

        ## Correct the matrices coordinates for the subsetting of the extracted image
        matrix_x = matrix_x - dims[0]
        matrix_y = matrix_y - dims[2]
        # print("im1A.shape=", im1A.shape)
        #
        # plt.figure()
        # plt.imshow(im1A, cmap="gray")  # , origin="lower")
        # plt.title("Image to resample: {},{}".format(im1A.shape[0], im1A.shape[1]))
        # plt.show()
        # print(im1A[1100, 1222])
        # sys.exit() # checkPoint: valid
        if self.resamplingEngine.method == "sinc":
            imgL3b = self.resamplingEngine.fAdapSincResampling(matrix_x, matrix_y, im1A, dims)
            # print("__________________________________ END Resampling _________________________________________")
            return imgL3b

        if self.resamplingEngine.method == "bilinear":
            # imgL3b_fl = interpRT.Interpolate2D(inArray=im1A, x=matrix_y.flatten(), y=matrix_x.flatten(), kind="linear")
            from scipy.interpolate import interpolate
            # print("Resampling ....")
            nbRows, nbCols = im1A.shape[0], im1A.shape[1]
            ## Note: since we have corrected matrix_x nad matrix_y coordinates the interpolation is then done 0,nbRows and 0,nbCols
            f = interpolate.RegularGridInterpolator(
                # (np.arange(dims[2], dims[3] + 1, 1), np.arange(dims[0], dims[1] + 1, 1)),
                (np.arange(0, nbRows, 1), np.arange(0, nbCols, 1)),
                im1A,
                method="linear",
                bounds_error=False,
                fill_value=np.nan)
            imgL3b_fl = f(list(zip(matrix_y.flatten(), matrix_x.flatten())))
            imgL3b = np.reshape(imgL3b_fl, sz)
            # # print(imgL3b.shape)
            # plt.imshow(imgL3b, cmap="gray")
            # plt.show()
            # print("__________________________________ END Resampling _________________________________________")
            return imgL3b
        if self.resamplingEngine == "bicubic":
            ## TODO: add warning
            geoErrors.erNotImplemented("bicubic")

        return
