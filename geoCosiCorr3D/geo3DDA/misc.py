"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""

import os
import numpy as np
from itertools import permutations, combinations, groupby
from typing import List

from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo, Convert
from geoCosiCorr3D.geoRSM.Interpol import Interpolate2D, Interpoate1D

from geoCosiCorr3D.geoCore.constants import INTERPOLATION_TYPES


class ImgInfo():
    def __init__(self):
        self.imgName = None
        self.orthoPath = None
        self.imgFolder = None
        self.rsmFile = None
        self.date = None
        self.warpRaster = None

    def __repr__(self):
        return """
        imgName = {}
        orthoPath = {}
        imgFolder = {}
        rsmFile = {}
        date = {}
        warpRaster = {}
          """.format(self.imgName,
                     self.orthoPath,
                     self.imgFolder,
                     self.rsmFile,
                     self.date,
                     self.warpRaster)


def generate_3DD_set_combination(pre_ortho_list: List, post_ortho_list: List):
    permPre = permutations(pre_ortho_list, 2)
    prePerm = [list(pre) for pre in list(permPre)]
    permPost = permutations(post_ortho_list, 2)
    postPerm = [list(post) for post in list(permPost)]
    combPre = combinations(pre_ortho_list, 2)
    preComb = [list(pre) for pre in list(combPre)]

    combPost = combinations(post_ortho_list, 2)
    postComb = [list(post) for post in list(combPost)]
    set_comb = []
    for preCombItem_ in prePerm:
        for postCombItem_ in postComb:
            set_comb.append(preCombItem_ + postCombItem_)
    for postCombItem_ in postPerm:
        for preCombItem_ in preComb:
            set_comb.append(postCombItem_ + preCombItem_)

    return set_comb


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class LOS:
    @staticmethod
    def compute_los(x_map, y_map, corr_info: cRasterInfo, dem_info: cRasterInfo, trx, y_trx_arr,
                    inter_sat_pos):
        XYZ_cart = LOS.xy_map_2_cart(xMap=x_map, yMap=y_map, prj=corr_info.epsg_code, dem_info=dem_info)
        # print(XYZ_cart)

        trx_info = cRasterInfo(trx)
        (xPix_yMat, yPix_yMat) = trx_info.Map2Pixel_Batch(X=x_map, Y=y_map)

        y_imagePix = Interpolate2D(inArray=y_trx_arr,
                                   x=yPix_yMat,
                                   y=xPix_yMat,
                                   kind=INTERPOLATION_TYPES.RectBivariateSpline)
        #
        satPos = []
        for i in range(3):
            satPos.append(Interpoate1D(X=np.arange(0, np.shape(inter_sat_pos[:, i])[0], 1), Y=inter_sat_pos[:, i],
                                       xCord=y_imagePix, kind=INTERPOLATION_TYPES.CUBIC))
        satPos = np.asarray(satPos).T
        XYZ_cart = np.asarray(XYZ_cart).T

        sightVector = np.asarray(satPos) - np.asarray(XYZ_cart)
        XYZ_cart = XYZ_cart.reshape(sightVector.shape)
        # print(type(XYZ_cart), XYZ_cart.shape)
        # print(type(sightVector), sightVector.shape)
        # sys.exit()
        return sightVector, XYZ_cart

    @staticmethod
    def xy_map_2_cart(xMap, yMap, prj, dem_info: cRasterInfo):
        """
        Get the elevation and  return the cartesian coordinates

        """

        # print(demPath)
        # dem_info = cRasterInfo(demPath)
        (xPixDem, yPixDem) = dem_info.Map2Pixel_Batch(X=xMap, Y=yMap)
        dem_array = dem_info.image_as_array()
        try:
            elev = Interpolate2D(inArray=dem_array, x=xPixDem, y=yPixDem, kind="RectBivariateSpline")
        except:
            import warnings
            warnings.warn("Error in interpolating the Elevation --> using hMean")
            elev = len(xMap) * [np.mean(dem_array)]

        (map_lat, map_lon, elev_lonLat) = Convert.coord_map1_2_map2(X=xMap, Y=yMap, Z=elev, sourceEPSG=prj,
                                                                    targetEPSG=4326)

        X_cart, Y_cart, Z_cart = Convert.geo_2_cartesian(Lon=map_lon, Lat=map_lat, Alt=elev_lonLat)
        XYZ = [X_cart, Y_cart, Z_cart]

        return XYZ


def merge_3dd_tiles(tiles_dir, o_dir):
    from geoCosiCorr3D.georoutines.file_cmd_routines import get_files_based_on_extensions
    from geoCosiCorr3D.georoutines.geo_utils import merge_tiles
    tilesList = []
    for x in os.walk(tiles_dir):
        dirTemp = x[0].split("/")[-1]
        if dirTemp == "3DDTiles":
            tilesList.append(x[0])
    for path_ in tilesList:
        imgList = get_files_based_on_extensions(path_)
        print("Number of Tiles:", len(imgList))
        merge_tiles(imgList,
                    os.path.join(o_dir, path_.split("/")[-2] + ".tif"))

    return
