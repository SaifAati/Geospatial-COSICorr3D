"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""

import numpy as np
import os
from pathlib import Path
from osgeo import gdal
from typing import List, Optional

from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo
from geoCosiCorr3D.geoCore.constants import SOFTWARE, RASTER_TYPE
from geoCosiCorr3D.georoutines.file_cmd_routines import get_files_based_on_extensions, CreateDirectory


class TilingRaster:
    def __init__(self, tile_sz=SOFTWARE.TILE_SIZE_MB):
        self.tile_sz = tile_sz
        pass

    @staticmethod
    def get_tile_info(col, lin, tileSize, nbCol, nbLine):
        if (col + tileSize) <= nbCol:
            if lin + tileSize > nbLine:
                resLin = nbLine - lin

                tempExtent = [lin, lin + resLin, col, col + tileSize]
            else:

                tempExtent = [lin, lin + tileSize, col, col + tileSize]
            col += tileSize
            return tempExtent, col
        elif (col + tileSize) > nbCol:
            restNbCol = nbCol - col
            if lin + tileSize > nbLine:
                resLin = nbLine - lin

                tempExtent = [lin, lin + resLin, col, col + restNbCol]
            else:

                tempExtent = [lin, lin + tileSize, col, col + restNbCol]
            col = nbCol
            return tempExtent, col

    def get_raster_tiles_extent(self, raster_shape):

        nbLine, nbCol = raster_shape
        colNbTile = int(nbCol / self.tile_sz)
        colRest = nbCol - colNbTile * self.tile_sz
        linNbTile = int(nbLine / self.tile_sz)
        linRest = nbLine - linNbTile * self.tile_sz
        # print("colNbTile, linNbTile: ", colNbTile, linNbTile)
        # print("colRest, linRest: ", colRest, linRest)
        print("NB tiles:", (colNbTile + 1) * (linNbTile + 1))

        tilesExtents = []
        col = 0
        lin = 0

        while lin < nbLine:
            while col < nbCol:
                tempExtent, col = self.get_tile_info(col, lin, self.tile_sz, nbCol, nbLine)
                tilesExtents.append(tempExtent)
            lin += self.tile_sz
            col = 0

        return tilesExtents

    @staticmethod
    def get_map_tiles_extent(pix_tiles_extent: List[List], raster_info: cRasterInfo):
        map_tiles_extent = []
        for pixExtent in pix_tiles_extent:
            xMapMin, yMapMin = raster_info.Pixel2Map(pixExtent[2], pixExtent[0])
            xMapMax, yMapMax = raster_info.Pixel2Map(pixExtent[3], pixExtent[1])
            map_tiles_extent.append([xMapMin, yMapMin, xMapMax, yMapMax])
        return map_tiles_extent

    @staticmethod
    def create_geo_tiles(in_raster_path, o_folder, map_tiles_extent, vrt=True, type=RASTER_TYPE.GDAL_FLOAT32,
                         tile_folder_suffix='_Tiles', tile_file_suffix='tile'):
        tiles_dir = CreateDirectory(directoryPath=o_folder,
                                    folderName=f'{Path(in_raster_path).stem}{tile_folder_suffix}')

        for index, tile_extent in enumerate(map_tiles_extent):
            format = 'vrt' if vrt == True else 'tif'
            o_tile_path = os.path.join(tiles_dir, f'{index}-{tile_file_suffix}_{index}.{format}')
            gdal.Translate(destName=o_tile_path, srcDS=gdal.Open(in_raster_path), projWin=tile_extent, outputType=type)

        return

    def __repr__(self):
        pass


def batch_tiling(in_folders: List, o_tiles_dir, ref_img_path, tile_sz=250,
                 ext_file_filter: Optional[List] = None):
    if ext_file_filter is None:
        ext_file_filter = ["*.tif", "*.vrt"]
    img_path_list = []
    for folder_ in in_folders:
        file_list = get_files_based_on_extensions(folder_, filter_list=ext_file_filter)
        img_path_list.extend(file_list)
    print(img_path_list, len(img_path_list))

    tmp_tiles_folder = CreateDirectory(directoryPath=o_tiles_dir, folderName="temp_tiles")

    raster_info = cRasterInfo(ref_img_path)
    raster_shape = (raster_info.raster_height, raster_info.raster_width)
    tile = TilingRaster(tile_sz=tile_sz)
    tiles_extent = tile.get_raster_tiles_extent(raster_shape)
    map_tiles_extent = tile.get_map_tiles_extent(tiles_extent, raster_info)

    for img_ in img_path_list:
        tile.create_geo_tiles(in_raster_path=img_, o_folder=tmp_tiles_folder, map_tiles_extent=map_tiles_extent)

    return


########################################################################################################################
## the function below extract the pixel extent and the values of the array
def TillingRaster_(array, tileSize=30):
    def GetTile(col, lin, tileSize, A, nbCol, nbLine):
        if (col + tileSize) <= nbCol:
            if lin + tileSize > nbLine:
                resLin = nbLine - lin
                tempTile = A[lin:lin + resLin, col:col + tileSize]
                tempExtent = [lin, lin + resLin, col, col + tileSize]
            else:
                tempTile = A[lin:lin + tileSize, col:col + tileSize]
                tempExtent = [lin, lin + tileSize, col, col + tileSize]
            col += tileSize
            return tempTile, tempExtent, col
        elif (col + tileSize) > nbCol:
            restNbCol = nbCol - col
            if lin + tileSize > nbLine:
                resLin = nbLine - lin
                tempTile = A[lin:lin + resLin, col:col + restNbCol]
                tempExtent = [lin, lin + resLin, col, col + restNbCol]
            else:
                tempTile = A[lin:lin + tileSize, col:col + restNbCol]
                tempExtent = [lin, lin + tileSize, col, col + restNbCol]
            col = nbCol
            return tempTile, tempExtent, col

    A = np.copy(array)

    nbLine, nbCol = np.shape(A)
    print("nbLine, nbCol", nbLine, nbCol)

    colNbTile = int(nbCol / tileSize)
    colRest = nbCol - colNbTile * tileSize
    linNbTile = int(nbLine / tileSize)
    linRest = nbLine - linNbTile * tileSize
    print("colNbTile, linNbTile: ", colNbTile, linNbTile)
    print("colRest, linRest: ", colRest, linRest)
    print("NB tiles:", (colNbTile + 1) * (linNbTile + 1))
    tiles = []
    tilesExtents = []
    col = 0
    lin = 0

    while lin < nbLine:
        while col < nbCol:
            tempTile, tempExtent, col = GetTile(col, lin, tileSize, A, nbCol, nbLine)
            # print(tempTile.shape)
            tiles.append(tempTile)
            tilesExtents.append(tempExtent)
        lin += tileSize
        col = 0

    return tiles, tilesExtents
