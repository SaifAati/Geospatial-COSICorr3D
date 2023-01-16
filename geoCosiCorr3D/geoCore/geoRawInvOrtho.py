"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
from typing import Dict, Optional, Type, List
import gdal, osr
import warnings, math
import numpy as np
import os
from pathlib import Path
from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import cGetSatMapGrid
from geoCosiCorr3D.geoCore.base.base_orthorectification import BaseInverseOrtho, BaseOrthoGrid, SatModel
from geoCosiCorr3D.geoCore.constants import SOFTWARE, SATELLITE_MODELS

from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import get_dem_dims
import geoCosiCorr3D.georoutines.geo_utils as geoRT

class InvalidOutputOrthoPath(Exception):
    pass
class RawInverseOrtho(BaseInverseOrtho):
    def __init__(self, input_l1a_path: str, output_ortho_path: str, output_trans_path: Optional[str],
                 ortho_params: Dict, dem_path: Optional[str], debug: bool = True):
        # self.ortho_grid = None
        super().__init__(input_l1a_path, output_ortho_path, output_trans_path, ortho_params, dem_path, debug)
        self._ingest()

    def _ingest(self):
        self.l1a_raster_info = geoRT.cRasterInfo(self.input_l1a_path)
        if self.dem_path is not None:
            self.dem_raster_info = geoRT.cRasterInfo(self.dem_path)
        else:
            logging.warning("DEM is set to NONE ")
            self.dem_raster_info = None

        self.o_res = self.ortho_params.get("GSD", None)
        self.ortho_method = self.ortho_params.get("method", None)

        self.ortho_type = self.ortho_method.get("method_type", None)

        self.metadata = self.ortho_method.get("metadata", None)

        if self.ortho_type == SATELLITE_MODELS.RSM:
            self.sensor = self.ortho_method.get("sensor", None)
            logging.info(f'Input RSM sensor: {self.sensor}')
            # TODO if sensor is not in supported sensor --> error
        else:
            self.sensor = None
        self.corr_model_file = self.ortho_method.get("corr_model", None)
        self.resampling_method = self.ortho_params.get("resampling_method", None)

        # self.resampling_engine = self._set_ortho_resampling_method(self.resampling_method)
        if self.output_ortho_path is None:
            raise InvalidOutputOrthoPath
        return

    def orthorectify(self):
        pass

    def _get_correction_model(self):
        """

        Returns:

        """
        pass

    def _check_dem_prj(self):

        if self.dem_raster_info is not None:
            if self.ortho_grid.gridEPSG != self.dem_raster_info.epsg_code:
                msg = "Reproject DEM from {}-->{}".format(self.dem_raster_info.epsg_code,
                                                          self.ortho_grid.gridEPSG)
                warnings.warn(msg)

                self.dem_path = geoRT.ReprojectRaster(input_raster_path=self.dem_raster_info.input_raster_path,
                                                      o_prj=self.ortho_grid.gridEPSG,
                                                      output_raster_path=os.path.join(
                                                          os.path.dirname(self.output_ortho_path),
                                                          Path(self.dem_raster_info.input_raster_path).stem + "_" + str(
                                                              self.ortho_grid.gridEPSG) + ".vrt"),
                                                      vrt=True)

                self.dem_raster_info = geoRT.cRasterInfo(self.dem_path)
        return

    def set_ortho_geo_transform(self) -> List[float]:

        # TODO transform to affine transformation, to ba compatible with rasterio
        return [self.ortho_grid.oUpLeftEW, self.ortho_grid.oRes, 0, self.ortho_grid.oUpLeftNS, 0,
                -1 * self.ortho_grid.oRes]

    def compute_tiles(self):
        """
        Compute the required number of tiles.
        Returns:

        """
        oNbCols = round(((self.ortho_grid.oBotRightEW - self.ortho_grid.oUpLeftEW) / self.ortho_grid.oRes) + 1)
        oNbRows = round(((self.ortho_grid.oUpLeftNS - self.ortho_grid.oBotRightNS) / self.ortho_grid.oRes) + 1)
        self.oRasterW = oNbCols
        self.oRasterH = oNbRows
        self.nbRowsPerTile = math.floor((SOFTWARE.TILE_SIZE_MB * 8 * 1024 * 1024) / (oNbCols * 32 * 2))
        if self.debug:
            logging.info("nbRowsPerTile: {}".format(self.nbRowsPerTile))
        self.nbTiles = int(oNbRows / self.nbRowsPerTile)

        if (self.nbTiles != 0):
            if oNbRows % self.nbRowsPerTile != 0:
                self.nbTiles += 1
        else:
            self.nbTiles = 1

        return

    def write_ortho_per_tile(self):

        driver = gdal.GetDriverByName("GTiff")
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(self.ortho_grid.gridEPSG)

        if self.output_trans_path is not None:
            self.oTranRaster = driver.Create(self.output_trans_path, self.oRasterW, self.oRasterH, 2, gdal.GDT_Float32,
                                             options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])

            self.oTranRaster.SetGeoTransform(
                (self.ortho_geo_transform[0], self.ortho_geo_transform[1], self.ortho_geo_transform[2],
                 self.ortho_geo_transform[3],
                 self.ortho_geo_transform[4], self.ortho_geo_transform[5]))
            self.oTranRaster.SetProjection(outRasterSRS.ExportToWkt())
            self.xMatBand = self.oTranRaster.GetRasterBand(1)
            self.yMatBand = self.oTranRaster.GetRasterBand(2)
            self.xMatBand.SetDescription("Transformation xMat")
            self.yMatBand.SetDescription("Transformation yMat")
            self.oTranRaster.SetMetadataItem("Author", "SAIF AATI saif@caltech.edu")

        self.oOrthoRaster = driver.Create(self.output_ortho_path, self.oRasterW, self.oRasterH, 1, gdal.GDT_UInt16,
                                          options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
        self.oOrthoRaster.SetGeoTransform(
            (self.ortho_geo_transform[0], self.ortho_geo_transform[1], self.ortho_geo_transform[2],
             self.ortho_geo_transform[3],
             self.ortho_geo_transform[4],
             self.ortho_geo_transform[5]))

        self.oOrthoRaster.SetProjection(outRasterSRS.ExportToWkt())
        self.oOrthoBand = self.oOrthoRaster.GetRasterBand(1)
        self.oOrthoBand.SetDescription("Ortho:" + str(self.ortho_grid.oRes) + "m")
        self.oOrthoRaster.SetMetadataItem("Author", "SAIF AATI saif@caltech.edu")

        return

    def write_ortho_rasters(self,
                            oOrthoTile,
                            matTile,
                            yOff: int) -> int:
        """

        Args:
            oOrthoTile:
            matTile:
            yOff:


        Returns:

        """
        logging.info(f"yOff:{yOff}")
        if self.nbTiles > 1:
            if self.debug:
                logging.info("... Tile saving ...")
                logging.info(oOrthoTile.shape)
            if self.output_trans_path is not None:
                self.xMatBand.WriteArray(matTile[0, :, :], xoff=0, yoff=yOff)
                self.yMatBand.WriteArray(matTile[1, :, :], xoff=0, yoff=yOff)
                self.xMatBand.FlushCache()
                self.yMatBand.FlushCache()
            self.oOrthoBand.WriteArray(oOrthoTile, xoff=0, yoff=yOff)
            self.oOrthoBand.FlushCache()
            self.oOrthoBand.SetNoDataValue(0)
            yOff += oOrthoTile.shape[0]

        else:
            progress = False
            if self.debug:
                progress = True
            oMat_x = matTile[0, :, :]
            oMat_y = matTile[1, :, :]
            oOrtho = oOrthoTile
            if self.output_trans_path is not None:
                geoRT.WriteRaster(oRasterPath=self.output_trans_path,
                                  geoTransform=self.ortho_geo_transform,
                                  arrayList=[oMat_x, oMat_y],
                                  descriptions=["Transformation xMat", "Transformation yMat"],
                                  epsg=self.ortho_grid.gridEPSG,
                                  progress=progress,
                                  dtype=gdal.GDT_Float32)
            geoRT.WriteRaster(oRasterPath=self.output_ortho_path,
                              geoTransform=self.ortho_geo_transform,
                              arrayList=[oOrtho],
                              descriptions=["Ortho:" + str(self.ortho_grid.oRes) + "m"],
                              epsg=self.ortho_grid.gridEPSG,
                              progress=progress,
                              dtype=gdal.GDT_UInt16,
                              noData=0)
        return yOff

    def _set_ortho_grid(self) -> cGetSatMapGrid:
        pass

    @staticmethod
    def ortho_tiling(nbRowsPerTile, nbRowsOut, nbColsOut, need_loop, oUpLeftEW, oUpLeftNS, xRes, yRes, demInfo):
        """
        Define the grid for each tile
        Args:
            oUpLeftEW:
            oUpLeftNS:
            nbRowsOut:
            nbColsOut:
            need_loop:

            xRes:
            yRes:
            demInfo:

        Returns:

        """
        # # Define number max of lines per "tile"
        # self.nbRowsPerTile = math.floor((self.__imgTileSizemb * 8 * 1024 * 1024) / (nbColsOut * 32 * 2))
        nbTiles = int(nbRowsOut / nbRowsPerTile)
        if (nbTiles != 0) and (need_loop == 1):  # if needLoop=0 -> compute full matrix in memory
            tiles = list(np.arange(nbTiles + 1) * nbRowsPerTile)
            if nbRowsOut % nbRowsPerTile != 0:
                tiles.append(nbRowsOut)
                nbTiles += 1
        else:
            nbTiles = 1
            tiles = [0, nbRowsOut]

        # Easting and Nothing array for the whole matrix
        easting = oUpLeftEW + np.arange(nbColsOut) * xRes
        northing = oUpLeftNS - np.arange(nbRowsOut) * yRes

        ## Definition for all the matrice tiles of the necessary DEM subset
        demDims = np.zeros((nbTiles, 4))  # initialization in case no dem

        if demInfo is not None:
            demDims = np.zeros((nbTiles, 4))
            for tile_ in range(nbTiles):
                xBBox = [easting[0], easting[nbColsOut - 1], easting[0], easting[nbColsOut - 1]]
                yBBox = [northing[tiles[tile_]], northing[tiles[tile_]], northing[tiles[tile_ + 1] - 1],
                         northing[tiles[tile_ + 1] - 1]]
                dims = get_dem_dims(xBBox=xBBox, yBBox=yBBox, demInfo=demInfo)
                demDims[tile_, :] = dims

        return demDims, easting, northing, nbTiles, tiles

    def DEM_interpolation(self, demInfo, demDims, tileCurrent, eastArr, northArr, modelData):
        """


        Args:
            demInfo:
            demDims:
            tileCurrent:
            eastArr:
            northArr:
            modelData:

        Returns:

        """
        pass

    def compute_transformation_matrix(self, need_loop, init_data):
        """

        Args:
            need_loop:
            init_data:

        Returns:

        """
        pass


class RawOrthoGrid(BaseOrthoGrid):
    def __init__(self, sat_model: Type['SatModel'], grid_epsg: int = None, gsd: float = None):
        # rasterInfo,
        super().__init__(sat_model, grid_epsg, gsd)

# TODO Notes:
#  Getting two child classes
#  1 one compute the raw grid of a satellite images
#  2 The second one generate new output ortho grid which is a child of the raw grid satellite image
