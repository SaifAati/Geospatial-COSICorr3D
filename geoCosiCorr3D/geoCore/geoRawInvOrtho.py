"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import math
import os
import warnings
from pathlib import Path
from typing import List, Type

import numpy as np
import psutil
from osgeo import gdal, osr

import geoCosiCorr3D.georoutines.geo_utils as geoRT
import geoCosiCorr3D.utils.misc as misc
from geoCosiCorr3D.geoCore.base.base_orthorectification import (
    BaseInverseOrtho, BaseOrthoGrid, SatModel)
from geoCosiCorr3D.geoCore.constants import SATELLITE_MODELS, SOFTWARE
from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import get_dem_dims
from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import SatMapGrid


class InvalidOutputOrthoPath(Exception):
    pass


class RawInverseOrtho(BaseInverseOrtho):
    def __init__(self, input_l1a_path, output_ortho_path, ortho_params, **kwargs):

        super().__init__(input_l1a_path, output_ortho_path, ortho_params, **kwargs)
        misc.log_available_memory(self.__class__.__name__)
        self._ingest()
        misc.log_available_memory(f'{self.__class__.__name__}: ingest')

    def _ingest(self):
        self.l1a_raster_info = geoRT.cRasterInfo(self.input_l1a_path)
        if self.dem_path is not None:
            self.dem_raster_info = geoRT.cRasterInfo(self.dem_path)
        else:
            logging.warning("DEM is set to NONE ")
            self.dem_raster_info = None

        self.o_res = self.ortho_params.get("GSD", None)
        self.ortho_method = self.ortho_params.get("method", {})

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
        pass

    def _check_dem_prj(self):

        if self.dem_raster_info is not None:
            if self.ortho_grid.grid_epsg != self.dem_raster_info.epsg_code:
                msg = "Reproject DEM from {}-->{}".format(self.dem_raster_info.epsg_code,
                                                          self.ortho_grid.grid_epsg)
                warnings.warn(msg)

                self.dem_path = geoRT.ReprojectRaster(input_raster_path=self.dem_raster_info.input_raster_path,
                                                      o_prj=self.ortho_grid.grid_epsg,
                                                      output_raster_path=os.path.join(
                                                          os.path.dirname(self.output_ortho_path),
                                                          Path(self.dem_raster_info.input_raster_path).stem + "_" + str(
                                                              self.ortho_grid.grid_epsg) + ".vrt"),
                                                      vrt=True)

                self.dem_raster_info = geoRT.cRasterInfo(self.dem_path)
        return

    def set_ortho_geo_transform(self) -> List[float]:

        # TODO transform to affine transformation, to be compatible with rasterio
        return [self.ortho_grid.o_up_left_ew, self.ortho_grid.o_res, 0, self.ortho_grid.o_up_left_ns, 0,
                -1 * self.ortho_grid.o_res]

    @staticmethod
    def compute_nb_rows_per_tile(o_raster_w, memory_usage_percent=0.5, bit_depth=32):
        """
        Dynamically computes the number of rows per tile based on available system memory and desired memory usage percentage.

        Args:
        - o_raster_w: The width of the raster.
        - memory_usage_percent: The percentage of available memory to use for computing the number of rows per tile.
        - bit_depth: The bit depth of the raster.

        Returns:
        - The number of rows per tile.
        """
        # Get available memory
        available_memory = psutil.virtual_memory().available
        memory_to_use = available_memory * memory_usage_percent
        bytes_per_pixel = 4 * 2
        # Calculate the number of pixels that can fit into the desired memory usage
        num_pixels = memory_to_use // bytes_per_pixel
        nb_rows_per_tile = num_pixels // (o_raster_w * bit_depth * 2)

        return math.floor(nb_rows_per_tile)

    def compute_num_tiles(self):
        """
        Compute the required number of tiles.

        """

        self.o_raster_w = round(
            ((self.ortho_grid.o_bot_right_ew - self.ortho_grid.o_up_left_ew) / self.ortho_grid.o_res) + 1)
        self.o_raster_h = round(
            ((self.ortho_grid.o_up_left_ns - self.ortho_grid.o_bot_right_ns) / self.ortho_grid.o_res) + 1)
        # self.nb_rows_per_tile = math.floor((SOFTWARE.TILE_SIZE_MB * 8 * 1024 * 1024) / (self.o_raster_w * 32 * 2))
        self.nb_rows_per_tile = self.compute_nb_rows_per_tile(self.o_raster_w, SOFTWARE.MEMORY_USAGE)
        if self.debug:
            logging.info(f'{self.__class__.__name__}: nb_rows_per_tile: {self.nb_rows_per_tile}')
        self.n_tiles = int(self.o_raster_h / self.nb_rows_per_tile)

        if (self.n_tiles != 0):
            if self.o_raster_h % self.nb_rows_per_tile != 0:
                self.n_tiles += 1
        else:
            self.n_tiles = 1

        return

    def write_ortho_per_tile(self):

        driver = gdal.GetDriverByName("GTiff")
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(self.ortho_grid.grid_epsg)

        if self.output_trans_path is not None:
            self.o_trf_raster = driver.Create(self.output_trans_path, self.o_raster_w, self.o_raster_h, 2,
                                              gdal.GDT_Float32,
                                              options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])

            self.o_trf_raster.SetGeoTransform(
                (self.ortho_geo_transform[0], self.ortho_geo_transform[1], self.ortho_geo_transform[2],
                 self.ortho_geo_transform[3],
                 self.ortho_geo_transform[4], self.ortho_geo_transform[5]))
            self.o_trf_raster.SetProjection(outRasterSRS.ExportToWkt())
            self.x_trf_band = self.o_trf_raster.GetRasterBand(1)
            self.y_trf_band = self.o_trf_raster.GetRasterBand(2)
            self.x_trf_band.SetDescription("Transformation xMat")
            self.y_trf_band.SetDescription("Transformation yMat")
            self.o_trf_raster.SetMetadataItem("Author", "SAIF AATI saif@caltech.edu")

        self.o_ortho_raster = driver.Create(self.output_ortho_path, self.o_raster_w, self.o_raster_h, 1,
                                            gdal.GDT_UInt16,
                                            options=["COMPRESS=LZW", "PREDICTOR=2", "BIGTIFF=YES"])
        self.o_ortho_raster.SetGeoTransform(
            (self.ortho_geo_transform[0], self.ortho_geo_transform[1], self.ortho_geo_transform[2],
             self.ortho_geo_transform[3],
             self.ortho_geo_transform[4],
             self.ortho_geo_transform[5]))

        self.o_ortho_raster.SetProjection(outRasterSRS.ExportToWkt())
        self.o_ortho_band = self.o_ortho_raster.GetRasterBand(1)
        self.o_ortho_band.SetDescription("Ortho:" + str(self.ortho_grid.o_res) + "m")
        self.o_ortho_raster.SetMetadataItem("Author", "SAIF AATI saif@caltech.edu")

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
        logging.info(f'{self.__class__.__name__}:y_off:{yOff}')
        if self.n_tiles > 1:
            if self.debug:
                logging.info("... Tile saving ...")
                logging.info(oOrthoTile.shape)
            if self.output_trans_path is not None:
                self.x_trf_band.WriteArray(matTile[0, :, :], xoff=0, yoff=yOff)
                self.y_trf_band.WriteArray(matTile[1, :, :], xoff=0, yoff=yOff)
                self.x_trf_band.FlushCache()
                self.y_trf_band.FlushCache()
            self.o_ortho_band.WriteArray(oOrthoTile, xoff=0, yoff=yOff)
            self.o_ortho_band.FlushCache()
            self.o_ortho_band.SetNoDataValue(0)
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
                                  epsg=self.ortho_grid.grid_epsg,
                                  progress=progress,
                                  dtype=gdal.GDT_Float32)

            geoRT.WriteRaster(oRasterPath=self.output_ortho_path,
                              geoTransform=self.ortho_geo_transform,
                              arrayList=[oOrtho],
                              descriptions=["Ortho:" + str(self.ortho_grid.o_res) + "m"],
                              epsg=self.ortho_grid.grid_epsg,
                              progress=progress,
                              dtype=gdal.GDT_UInt16,
                              noData=0)
        return yOff

    def _set_ortho_grid(self) -> SatMapGrid:
        ortho_grid = SatMapGrid(raster_info=self.l1a_raster_info,
                                model_data=self.model,
                                model_type=self.ortho_type,
                                dem_fn=self.dem_path,
                                new_res=self.o_res,
                                corr_model=self.corr_model,
                                debug=self.debug)
        return ortho_grid

    @staticmethod
    def ortho_tiling(nbRowsPerTile, nbRowsOut, nbColsOut, need_loop, oUpLeftEW, oUpLeftNS, xRes, yRes, demInfo):
        """
        Define the grid for each tile

        """
        # # Define number max of lines per "tile"
        n_tiles = int(nbRowsOut / nbRowsPerTile)
        if (n_tiles != 0) and (need_loop == 1):  # if needLoop=0 -> compute full matrix in memory
            tiles = list(np.arange(n_tiles + 1) * nbRowsPerTile)
            if nbRowsOut % nbRowsPerTile != 0:
                tiles.append(nbRowsOut)
                n_tiles += 1
        else:
            n_tiles = 1
            tiles = [0, nbRowsOut]

        # Easting and Nothing array for the whole matrix
        easting = oUpLeftEW + np.arange(nbColsOut) * xRes
        northing = oUpLeftNS - np.arange(nbRowsOut) * yRes

        ## Definition for all the matrice tiles of the necessary DEM subset
        demDims = np.zeros((n_tiles, 4))  # initialization in case no dem

        if demInfo is not None:
            demDims = np.zeros((n_tiles, 4))
            for tile_ in range(n_tiles):
                xBBox = [easting[0], easting[nbColsOut - 1], easting[0], easting[nbColsOut - 1]]
                yBBox = [northing[tiles[tile_]], northing[tiles[tile_]], northing[tiles[tile_ + 1] - 1],
                         northing[tiles[tile_ + 1] - 1]]
                dims = get_dem_dims(xBBox=xBBox, yBBox=yBBox, demInfo=demInfo)
                demDims[tile_, :] = dims

        return demDims, easting, northing, n_tiles, tiles

    def elev_interpolation(self, demDims, tileCurrent, eastArr, northArr):
        from geoCosiCorr3D.geoCore.geoDEM import HeightInterpolation
        if self.debug:
            logging.info(f"{self.__class__.__name__}: Elev interpolation ...")
        h_new = HeightInterpolation.DEM_interpolation(demInfo=self.dem_raster_info,
                                                      demDims=demDims,
                                                      tileCurrent=tileCurrent,
                                                      eastArr=eastArr,
                                                      northArr=northArr)
        if h_new is None:

            if self.mean_h is not None:
                h_new = np.ones(eastArr.shape) * self.mean_h
                logging.warning(f'H will be set to the mean_h {self.mean_h}')
            else:
                h_new = np.zeros(eastArr.shape)
                logging.warning('H will be set to 0')
        return h_new

    def compute_transformation_matrix(self, ortho_data):
        pass


class RawOrthoGrid(BaseOrthoGrid):
    def __init__(self, sat_model: Type['SatModel'], grid_epsg: int = None, gsd: float = None):
        super().__init__(sat_model, grid_epsg, gsd)

# TODO Notes:
#  Getting two child classes
#  1 one compute the raw grid of a satellite images
#  2 The second one generate new output ortho grid which is a child of the raw grid satellite image
