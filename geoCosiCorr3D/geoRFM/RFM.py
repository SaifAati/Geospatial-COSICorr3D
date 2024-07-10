"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""
import logging
import warnings
from typing import List, Optional

import affine6p
import geopandas
import numpy as np
from shapely.geometry import Polygon

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoCore.base.base_sat_model import SatModel
from geoCosiCorr3D.geoRFM.load_rfm import ReadRFM

converter = geoRT.Convert()


class RFM(ReadRFM, SatModel):

    def __init__(self, rfm_file: Optional[str] = None,
                 corr_model: Optional[np.ndarray] = None,
                 debug: bool = False,
                 **kwargs):

        SatModel.__init__(self,
                          name=C.SATELLITE_MODELS.RFM,
                          corr_model=corr_model, **kwargs)
        ReadRFM.__init__(self, rfm_file)

        self.debug = debug
        if self.debug:
            logging.info(self.__repr__())

    def g2i(self, lon, lat, alt=None, normalized=False):

        if alt is None:
            alt = []
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        if np.array(alt).any() == True:
            alt = np.asarray(alt)
        else:
            if self.dem is not None:
                alt = self.dem.get_elevations3(lons=lon, lats=lat)
                print(alt)
            else:
                warnings.warn(
                    f"Warning: No altitude values available and no DEM detected. Alt will be set to the default offset value of {self.altOff}.")
                alt = np.ones(lon.shape) * self.altOff

        lonN = (lon - self.lonOff) / self.lonScale
        latN = (lat - self.latOff) / self.latScale
        altN = (alt - self.altOff) / self.altScale
        colN = self.build_RFM(num=self.colNum, den=self.colDen, x=latN, y=lonN, z=altN)
        linN = self.build_RFM(num=self.linNum, den=self.linDen, x=latN, y=lonN, z=altN)
        if not np.all((self.corr_model == 0)):
            colN, linN = self.apply_correction(self.corr_model, colN, linN)
        if normalized == True:
            return colN, linN
        else:
            col = colN * self.colScale + self.colOff
            row = linN * self.linScale + self.linOff

            return col, row

    def i2g(self, col, lin,
            init_alt: Optional[List] = None,
            normalized=False):
        col = np.atleast_1d(np.asarray(col))
        lin = np.atleast_1d(np.asarray(lin))

        if init_alt is None:
            init_alt = np.full_like(col, self.altOff)
        alt_init_ = np.copy(init_alt)

        # Normalize input image coordinates
        colN = (col - self.colOff) / self.colScale
        linN = (lin - self.linOff) / self.linScale
        altIniN = (alt_init_ - self.altOff) / self.altScale

        if self.lonNum == [np.nan] * self.NB_NUM_COEF:
            if self.debug:
                logging.warning("Computing Direct model ....")

            lonN, latN = self.direct_model(colN=colN, linN=linN, altN=altIniN, corr_model=self.corr_model)
        else:
            lonN = self.build_RFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altIniN)
            latN = self.build_RFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altIniN)
        if normalized:
            return lonN, latN, altIniN
        else:
            lon = np.atleast_1d(np.asarray(lonN * self.lonScale + self.lonOff))
            lat = np.atleast_1d(np.asarray(latN * self.latScale + self.latOff))
            alt = init_alt
            # Here we will use the computed lon & lat to interpolate the alt from the DEM if exist
            # Here we are performing one iteration --> TODO add more iter to refine the elevation
            if self.dem_fn is not None:
                alt = []
                # TODO: loop until convergence or no change in coordinates
                for lonVal, latVal, altValIni in zip(lon, lat, init_alt):
                    altVal = self.dem.get_elevation(lonVal, latVal)
                    if altVal == np.nan:
                        altVal = altValIni
                    alt.append(altVal)

                alt = np.atleast_1d(np.asarray(alt))

                # Normalize input image coordinates
                colN = (col - self.colOff) / self.colScale
                linN = (lin - self.linOff) / self.linScale
                altN = (alt - self.altOff) / self.altScale
                if self.lonNum == [np.nan] * self.NB_NUM_COEF:
                    lonN, latN = self.direct_model(colN=colN, linN=linN, altN=altN, corr_model=self.corr_model)
                else:
                    lonN = self.build_RFM(num=self.lonNum, den=self.lonDen, x=linN, y=colN, z=altN)
                    latN = self.build_RFM(num=self.latNum, den=self.latDen, x=linN, y=colN, z=altN)
                lon = lonN * self.lonScale + self.lonOff
                lat = latN * self.latScale + self.latOff
            return lon, lat, alt

    def get_geotransform(self):

        h = int(self.linOff * 2)
        w = int(self.colOff * 2)
        BBoxPix = [[0, 0],
                   [0, h],
                   [w, h],
                   [w, 0],
                   [0, 0]]

        z = self.altOff
        lons, lats, _ = self.i2g(col=[0, 0, w, w, 0],
                                 lin=[0, h, h, 0, 0],
                                 init_alt=[z, z, z, z, z],
                                 normalized=False)
        BBoxMap = []
        for lon_, lat_ in zip(lons, lats):
            BBoxMap.append([lon_, lat_])

        trans = affine6p.estimate(origin=BBoxPix, convrt=BBoxMap)
        mat = trans.get_matrix()  ## Homogenious represention of the affine transformation
        geoTrans_h = np.array(mat)
        geo_transform = [mat[0][-1], mat[0][0], mat[0][1], mat[1][-1], mat[1][0], mat[1][1]]
        return geo_transform

    def get_footprint(self) -> [Polygon, geopandas.GeoDataFrame]:
        h = int(self.linOff * 2)
        w = int(self.colOff * 2)
        z = self.altOff

        lons, lats, _ = self.i2g(col=[0, 0, w, w, 0],
                                 lin=[0, h, h, 0, 0],
                                 init_alt=[z, z, z, z, z],
                                 normalized=False)

        fp_poly_geom = Polygon(zip(lons, lats))
        gpd_polygon = geopandas.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[fp_poly_geom])

        return fp_poly_geom, gpd_polygon

    def get_gsd(self):
        h = self.linOff * 2
        w = self.colOff * 2

        center = (int(h / 2), int(w / 2))
        center_plus = (center[0] + 1, center[1] + 1)
        center_prj = self.i2g(col=center[1], lin=center[0])
        center_prj_plus = self.i2g(col=center_plus[1], lin=center_plus[0])
        utm_epsg_code = geoRT.ComputeEpsg(lon=center_prj[0], lat=center_prj[1])
        center_coords = converter.coord_map1_2_map2(X=[center_prj[1], center_prj_plus[1]],
                                                    Y=[center_prj[0], center_prj_plus[0]],
                                                    targetEPSG=utm_epsg_code)
        x_gsd = np.abs(center_coords[0][0] - center_coords[0][1])
        y_gsd = np.abs(center_coords[1][0] - center_coords[1][1])
        return (x_gsd, y_gsd)

    def get_altitude_range(self, scaleFactor=1):
        """

        Args:
            scaleFactor:

        Returns:

        """
        min_alt = self.altOff - scaleFactor * self.altScale
        max_alt = self.altOff + scaleFactor * self.altScale
        return [min_alt, max_alt]
