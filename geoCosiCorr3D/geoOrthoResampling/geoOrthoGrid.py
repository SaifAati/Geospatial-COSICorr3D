"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import os
from typing import Optional

import geoCosiCorr3D.georoutines.geo_utils as geoRT
import geojson
import numpy as np
from geoCosiCorr3D.geoCore.constants import SATELLITE_MODELS
from geoCosiCorr3D.geoOrthoResampling.geoOrtho_misc import decimal_mod

Converter = geoRT.Convert()


class SatMapGrid:
    def __init__(self,
                 raster_info: geoRT.cRasterInfo,
                 model_data,
                 model_type,
                 corr_model: Optional[np.ndarray] = None,
                 dem_fn=None,
                 new_res=None,
                 grid_epsg=None,
                 debug=True):

        self.grid_epsg = grid_epsg
        self.raster_info = raster_info
        self.model_data = model_data
        self.model_type = model_type
        self.dem_fn = dem_fn
        self.debug = debug

        self.dem_info = geoRT.cRasterInfo(self.dem_fn) if self.dem_fn is not None else None
        self.model_corr = corr_model if corr_model is not None else np.zeros((3, 3))

        self.up_left_ew = None
        self.bot_right_ew = None
        self.up_left_ns = None
        self.bot_right_ns = None
        self.grid_epsg = None
        self.res_ns = None
        self.res_ew = None

        self.o_up_left_ew = None
        self.o_bot_right_ew = None
        self.o_up_left_ns = None
        self.o_bot_right_ns = None

        self.__get_sat_map_grid()

        self.o_res = np.min([self.res_ew, self.res_ns]) if new_res is None else new_res

        self.__compute_o_map_grid()

        return

    def _transform_i2g_coords(self, cols, lins):

        if self.model_type == SATELLITE_MODELS.RSM:
            from geoCosiCorr3D.geoRSM.Pixel2GroundDirectModel import \
                cPix2GroundDirectModel

            geoCoordList = []

            for x, y in zip(cols, lins):
                pix2Ground_obj = cPix2GroundDirectModel(rsmModel=self.model_data,
                                                        xPix=x,
                                                        yPix=y,
                                                        rsmCorrectionArray=self.model_corr,
                                                        demFile=self.dem_fn)

                geoCoordList.append(pix2Ground_obj.geoCoords)
            return np.asarray(geoCoordList)

        elif self.model_type == SATELLITE_MODELS.RFM:
            from geoCosiCorr3D.geoRFM.RFM import RFM
            model_data: RFM = self.model_data
            # TODO use getGSD from RFM class
            lonBBox_RFM, latBBox_RFM, altBBox_RFM = model_data.i2g(col=cols,
                                                                   lin=lins)

            return np.array([lonBBox_RFM, latBBox_RFM, altBBox_RFM]).T
        else:
            raise ValueError(f'model_type:{self.model_type} is not supported')

    def __compute_gsd(self):

        geo_ground = self._transform_i2g_coords(
            cols=[self.raster_info.raster_width / 2, self.raster_info.raster_width / 2 + 1],
            lins=[self.raster_info.raster_height / 2, self.raster_info.raster_height / 2 + 1])
        utm_ground = Converter.coord_map1_2_map2(X=list(geo_ground[:, 1]),
                                                 Y=list(geo_ground[:, 0]),
                                                 Z=list(geo_ground[:, -1]),
                                                 targetEPSG=self.grid_epsg)
        self.res_ew = np.abs(utm_ground[0][0] - utm_ground[0][1])
        self.res_ns = np.abs(utm_ground[1][0] - utm_ground[1][1])

        return

    def __get_sat_map_grid(self):
        """
        Compute the image extent and projection system using the input model information.

        Returns:

        """

        logging.info(f'{self.__class__.__name__}... Computing Grid Extent ...')

        geo_ground = self._transform_i2g_coords(
            [0, self.raster_info.raster_width - 1, 0, self.raster_info.raster_width - 1],
            [0, 0, self.raster_info.raster_height - 1, self.raster_info.raster_height - 1]
        )
        if self.grid_epsg is None:
            self.grid_epsg = geoRT.ComputeEpsg(lon=geo_ground[0, 0], lat=geo_ground[0, 1])

            logging.info(f'{self.__class__.__name__}:Grid projection:{self.grid_epsg}')
        utm_ground = Converter.coord_map1_2_map2(X=list(geo_ground[:, 1]),
                                                 Y=list(geo_ground[:, 0]),
                                                 Z=list(geo_ground[:, -1]),
                                                 targetEPSG=self.grid_epsg)

        self.up_left_ew = utm_ground[0][0]
        self.up_right_ew = utm_ground[0][1]
        self.bot_left_ew = utm_ground[0][2]
        self.bot_right_ew = utm_ground[0][3]
        self.up_left_ns = utm_ground[1][0]
        self.up_right_ns = utm_ground[1][1]
        self.bot_left_ns = utm_ground[1][2]
        self.bot_right_ns = utm_ground[1][3]

        self.__compute_gsd()
        return

    def __compute_o_map_grid(self):

        ewBBox = [self.up_left_ew, self.up_right_ew, self.bot_left_ew, self.bot_right_ew]
        nsBBox = [self.up_left_ns, self.up_right_ns, self.bot_left_ns, self.bot_right_ns]

        self.o_up_left_ew, self.o_up_left_ns, self.o_bot_right_ew, self.o_bot_right_ns = self.compute_map_grid(
            up_left_ew=np.min(ewBBox),
            up_left_ns=np.max(nsBBox),
            bot_right_ew=np.max(ewBBox),
            bot_right_ns=np.min(nsBBox),
            o_res=self.o_res)
        return

    @staticmethod
    def compute_map_grid(up_left_ew, up_left_ns, bot_right_ew, bot_right_ns, o_res, grid_precision=1e5):
        """
        Compute regular output grid.
        """

        o_up_left_ew = up_left_ew
        o_bot_right_ew = bot_right_ew

        o_up_left_ns = up_left_ns
        o_bot_right_ns = bot_right_ns

        if decimal_mod(up_left_ew, o_res, o_res / grid_precision) != 0:
            if (up_left_ew - (up_left_ew % o_res)) > up_left_ew:
                o_up_left_ew = up_left_ew - (up_left_ew % o_res)
            else:
                o_up_left_ew = (up_left_ew - (up_left_ew % o_res) + o_res)

        if decimal_mod(bot_right_ew, o_res, o_res / grid_precision) != 0:
            if (bot_right_ew - (bot_right_ew % o_res)) < bot_right_ew:
                o_bot_right_ew = bot_right_ew - (bot_right_ew % o_res)
            else:
                o_bot_right_ew = (bot_right_ew - (bot_right_ew % o_res) + o_res)

        if decimal_mod(up_left_ns, o_res, o_res / grid_precision) != 0:
            if (up_left_ns - (up_left_ns % o_res)) < up_left_ns:
                o_up_left_ns = up_left_ns - (up_left_ns % o_res)
            else:
                o_up_left_ns = (up_left_ns - (up_left_ns % o_res) + o_res)

        if decimal_mod(bot_right_ns, o_res, o_res / grid_precision) != 0:
            if (bot_right_ns - (bot_right_ns % o_res)) > bot_right_ns:
                o_bot_right_ns = bot_right_ns - (bot_right_ns % o_res)
            else:
                o_bot_right_ns = (bot_right_ns - (bot_right_ns % o_res) + o_res)

        return o_up_left_ew, o_up_left_ns, o_bot_right_ew, o_bot_right_ns

    def grid_fp(self, o_folder=None):
        """
        Compute and write the footprint of the raw image (i.e., L1A img) using the orientation model (RFM ||RSM)


        """

        utm_x_bbox = [self.o_up_left_ew, self.o_bot_right_ew, self.o_bot_right_ew, self.o_up_left_ew, self.o_up_left_ew]
        utm_y_bbox = [self.o_up_left_ns, self.o_up_left_ns, self.o_bot_right_ns, self.o_bot_right_ns, self.o_up_left_ns]

        crs_geo = {"type": "name", "properties": {"name": "EPSG:4326"}}

        lats, lons = Converter.coord_map1_2_map2(X=utm_x_bbox, Y=utm_y_bbox,
                                                 sourceEPSG=self.grid_epsg,
                                                 targetEPSG=4326)

        footprint_geo = geojson.Feature(geometry=geojson.Polygon([list(zip(lons, lats))]),
                                        properties={"name": f"{self.__class__.__name__}_res_{self.o_res}"}, crs=crs_geo)

        feature_collection = geojson.FeatureCollection([footprint_geo])

        o_file_name = f"fp_{self.__class__.__name__}_{self.model_type}_res_{self.o_res}.geojson"
        o_fn = os.path.join(o_folder, o_file_name) if o_folder is not None else o_file_name
        with open(o_fn, "w") as file:
            geojson.dump(feature_collection, file)

        return footprint_geo

    def __repr__(self):
        return """
         # Grid Information :
            proj_epsg = {}
            -- ESTIMATED GRID --
            res (ew,ns) = [{:.3f} , {:.3f}] m
            upLeft= [{:.5f} , {:.5f}]
            upRight = [{:.5f} , {:.5f}]
            botLeft= [{:.5f} , {:.5f}]
            botRight = [{:.5f} , {:.5f}]
            -- OUTPUT GRID --
            o_up_left_ew={:.5f}
            o_bot_right_ew={:.5f}
            o_up_left_ns={:.5f}
            o_bot_right_ns={:.5f}
            o_res= {:.5f}
            ________________________""".format(self.grid_epsg,
                                               self.res_ew, self.res_ns,
                                               self.up_left_ew, self.up_left_ns,
                                               self.up_right_ew, self.up_right_ns,
                                               self.bot_left_ew, self.bot_left_ns,
                                               self.bot_right_ew, self.bot_right_ns,
                                               self.o_up_left_ew, self.o_bot_right_ew,
                                               self.o_up_left_ns, self.o_bot_right_ns,
                                               self.o_res)
