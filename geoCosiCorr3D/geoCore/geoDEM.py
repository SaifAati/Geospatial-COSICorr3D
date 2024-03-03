import warnings
from typing import List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform

import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoRSM.Interpol import Interpolate2D
from geoCosiCorr3D.georoutines.geo_utils import (Convert,
                                                 cRasterInfoGDAL)
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo


class DEM:
    def __init__(self, dem_fn, margin: int = 3):
        self.dem_fn = dem_fn
        self.margin = margin
        self.set_elevation_layer()

    def set_elevation_layer(self):
        self.dem_info = geoRT.cRasterInfo(self.dem_fn)
        print(self.dem_fn)
        if self.dem_info.epsg_code != '4326':
            dem_fn = geoRT.ReprojectRaster(input_raster_path=self.dem_info.input_raster_path,
                                           o_prj=4326,
                                           vrt=True)
            self.dem_fn = dem_fn
            self.dem_info = geoRT.cRasterInfo(self.dem_fn)
            print(self.dem_info.epsg_code)

    def get_elevation(self, lon, lat):

        col, lin = self.dem_info.Map2Pixel(x=lon, y=lat)
        window = self.dem_info.image_as_array_subset(col_off_min=int(col) - self.margin,
                                                     col_off_max=int(col) + self.margin,
                                                     row_off_min=int(lin) - self.margin,
                                                     row_off_max=int(lin) + self.margin)

        if window.size == 0:
            print("Val:{:.3f},{:.3f} outside the DEM extent ".format(lon, lat))
            return np.nan
        else:

            alt = Interpolate2D(inArray=window,
                                x=[lin - (int(lin) - self.margin)],
                                y=[col - (int(col) - self.margin)])

            return alt[0]

    def get_elevations(self, lons, lats):
        altitudes = []
        for lon, lat in zip(lons, lats):
            col, lin = self.dem_info.Map2Pixel(x=lon, y=lat)
            window = self.dem_info.image_as_array_subset(col_off_min=int(col) - self.margin,
                                                         col_off_max=int(col) + self.margin,
                                                         row_off_min=int(lin) - self.margin,
                                                         row_off_max=int(lin) + self.margin)
            if window.size == 0:
                print("Val:{:.3f},{:.3f} outside the DEM extent ".format(lon, lat))
                altitudes.append(np.nan)
            else:
                alt = Interpolate2D(inArray=window,
                                    x=[lin - (int(lin) - self.margin)],
                                    y=[col - (int(col) - self.margin)])
                altitudes.append(alt[0])
        return np.array(altitudes)

    def get_elevations2(self, lons, lats):

        elevations = []

        dem = self.dem_info.raster
        for lon, lat in zip(lons, lats):
            # Transform input coordinates to the DEM's coordinate system
            input_crs = 'EPSG:4326'  # Assuming input lon/lat is in WGS84
            transform_coords = transform(src_crs=input_crs,
                                         dst_crs=dem.crs,
                                         xs=[lon],
                                         ys=[lat]
                                         )
            x, y = transform_coords[0][0], transform_coords[1][0]
            row, col = dem.index(x, y)

            elevation = dem.read(1, window=rasterio.windows.Window(col, row, 1, 1), resampling=Resampling.bilinear)
            elevations.append(elevation[0, 0])

        return np.array(elevations)

    def get_elevations3(self, lons, lats):

        dem = self.dem_info.raster

        # Vectorized transform of coordinates
        input_crs = 'EPSG:4326'
        xs, ys = transform(src_crs=input_crs, dst_crs=dem.crs, xs=lons, ys=lats)

        # Vectorized conversion to row and column indices
        rows, cols = dem.index(xs, ys)

        # Since rasterio does not directly support vectorized window reads,
        # we'll read the minimum bounding box that contains all points and then index into it.
        min_col, min_row = np.min(cols), np.min(rows)
        max_col, max_row = np.max(cols), np.max(rows)
        window = rasterio.windows.Window.from_slices((min_row, max_row + 1), (min_col, max_col + 1))
        elevation_data = dem.read(1, window=window, resampling=Resampling.bilinear)

        # Extract elevations using the relative indices within the read window
        elevations = elevation_data[rows - min_row, cols - min_col]

        return np.array(elevations)

    def get_elevations4(self, lons, lats):

        elevations = []

        dem = self.dem_info.raster
        for lon, lat in zip(lons, lats):
            col, row = self.dem_info.Map2Pixel(x=lon, y=lat)
            elevation = dem.read(1, window=rasterio.windows.Window(col, row, 1, 1), resampling=Resampling.bilinear)
            elevations.append(elevation[0, 0])

        return np.array(elevations)

    def check_dem_subset(self, col, lin):
        if int(col) - self.margin > 0:
            temp = int(col) - self.margin
        else:
            temp = 0
        if temp < self.dem_info.raster_width - self.margin:
            col_min = temp
        else:
            col_min = (self.dem_info.raster_width - 1) - self.margin

        if np.ceil(col) + self.margin < self.dem_info.raster_width:
            temp = np.ceil(np.max(col)) + self.margin
        else:
            temp = self.dem_info.raster_width - 1
        if temp > self.margin:
            col_max = temp
        else:
            col_max = self.margin

        if int(lin) - self.margin > 0:
            temp = int(lin) - self.margin
        else:
            temp = 0

        if temp < self.dem_info.raster_height - self.margin:
            lin_min = temp
        else:
            lin_min = self.dem_info.raster_height - self.margin

        if np.ceil(lin) + self.margin < self.dem_info.raster_height:
            temp = np.ceil(lin) + self.margin
        else:
            temp = self.dem_info.raster_height - 1
        if temp > self.margin:
            lin_max = temp
        else:
            lin_max = self.margin
        return int(col_min), int(col_max), int(lin_min), int(lin_max)


class HeightInterpolation:
    @staticmethod
    def get_h_from_DEM(geo_coords: List, demInfo, h: float = None, step=3):
        """

        Args:
            geo_coords: [lon, lat]
            demInfo:
            h:
            step:

        Returns:

        """

        demCoord = Convert.coord_map1_2_map2(X=geo_coords[1],  # LAT
                                             Y=geo_coords[0],  # LON
                                             Z=h,
                                             targetEPSG=demInfo.epsg_code,
                                             sourceEPSG=4326)

        xdemCoord = demCoord[0]
        ydemCoord = demCoord[1]
        xdemPix = (xdemCoord - demInfo.x_map_origin) / np.abs(demInfo.pixel_width)
        ydemPix = (demInfo.y_map_origin - ydemCoord) / np.abs(demInfo.pixel_height)
        XdemMin, XdemMax, YdemMin, YdemMax = HeightInterpolation.check_DEM_subset(xdemPix=xdemPix,
                                                                                  ydemPix=ydemPix,
                                                                                  demInfo=demInfo,
                                                                                  step=step)

        demSubset = demInfo.image_as_array_subset(col_off_min=int(XdemMin),
                                                  col_off_max=int(XdemMax),
                                                  row_off_min=int(YdemMin),
                                                  row_off_max=int(YdemMax))

        # print(dem_subset, dem_subset.shape)
        return Interpolate2D(inArray=demSubset, x=[ydemPix - YdemMin], y=[xdemPix - XdemMin],
                             kind="RectBivariateSpline")

    @staticmethod
    def get_h_from_DEM_v2(geo_coords: List, dem_path: str, h: float = None, step=3):
        """

        Args:
            geo_coords: [lat,lon]
            demInfo:
            h:
            step:

        Returns:

        """

        demInfo = cRasterInfoGDAL(dem_path)
        demCoord = Convert.coord_map1_2_map2(X=geo_coords[0],
                                             Y=geo_coords[1],
                                             Z=h,
                                             targetEPSG=demInfo.epsg_code,
                                             sourceEPSG=4326)
        xdemCoord = demCoord[0]
        ydemCoord = demCoord[1]
        xdemPix = (xdemCoord - demInfo.x_map_origin) / np.abs(demInfo.pixel_width)
        ydemPix = (demInfo.y_map_origin - ydemCoord) / np.abs(demInfo.pixel_height)

        XdemMin, XdemMax, YdemMin, YdemMax = HeightInterpolation.check_DEM_subset(xdemPix=xdemPix,
                                                                                  ydemPix=ydemPix,
                                                                                  demInfo=demInfo,
                                                                                  step=step)

        demSubset = demInfo.image_as_array_subset(input_raster_path=dem_path,
                                                  col_off_min=int(XdemMin),
                                                  col_off_max=int(XdemMax),
                                                  row_off_min=int(YdemMin),
                                                  row_off_max=int(YdemMax))

        # print(dem_subset, dem_subset.shape)
        return Interpolate2D(inArray=demSubset, x=[ydemPix - YdemMin], y=[xdemPix - XdemMin],
                             kind="RectBivariateSpline")

    @staticmethod
    def check_DEM_subset(xdemPix, ydemPix, demInfo, step=3):
        if int(xdemPix) - step > 0:
            temp = int(xdemPix) - step
        else:
            temp = 0
        if temp < demInfo.raster_width - step:
            XdemMin = temp
        else:
            XdemMin = (demInfo.raster_width - 1) - step

        if np.ceil(xdemPix) + step < demInfo.raster_width:
            temp = np.ceil(np.max(xdemPix)) + step
        else:
            temp = demInfo.raster_width - 1
        if temp > step:
            XdemMax = temp
        else:
            XdemMax = step

        if int(ydemPix) - step > 0:
            temp = int(ydemPix) - step
        else:
            temp = 0

        if temp < demInfo.raster_height - step:
            YdemMin = temp
        else:
            YdemMin = demInfo.raster_height - step

        if np.ceil(ydemPix) + step < demInfo.raster_height:
            temp = np.ceil(ydemPix) + step
        else:
            temp = demInfo.raster_height - 1
        if temp > step:
            YdemMax = temp
        else:
            YdemMax = step
        return int(XdemMin), int(XdemMax), int(YdemMin), int(YdemMax)

    @staticmethod
    def DEM_interpolation(demInfo: cRasterInfo, demDims, eastArr, northArr, tileCurrent=None):
        h_new = None
        try:
            if tileCurrent is not None:
                tempWindow = demDims[tileCurrent, :]
                demCol = (eastArr - demInfo.x_map_origin) / np.abs(demInfo.pixel_width) - demDims[tileCurrent, 0]
                demRow = (demInfo.y_map_origin - northArr) / np.abs(demInfo.pixel_height) - demDims[tileCurrent, 2]
            else:
                demCol = (eastArr - demInfo.x_map_origin) / np.abs(demInfo.pixel_width) - demDims[0]
                demRow = (demInfo.y_map_origin - northArr) / np.abs(demInfo.pixel_height) - demDims[2]
                tempWindow = demDims

            demSubset = demInfo.image_as_array_subset(int(tempWindow[0]),
                                                      int(tempWindow[1]) + 1,
                                                      int(tempWindow[2]),
                                                      int(tempWindow[3]) + 1)

            hNew_flatten = Interpolate2D(inArray=demSubset,
                                         x=demRow.flatten(),
                                         y=demCol.flatten(),
                                         kind="linear")  # TODO: --> user-choice
            h_new = np.reshape(hNew_flatten, demCol.shape)

        except:
            warnings.warn("Enable to interpolate H form the input DEM ")

        return h_new
