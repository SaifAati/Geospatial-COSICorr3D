"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas

import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoCore.core_geoGCPs import RawTP2GCP
from geoCosiCorr3D.geoCore.geoDEM import HeightInterpolation


class TpsToGcps(RawTP2GCP):
    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)
        self.ingest()

    def __call__(self, *args, **kwargs):
        self.run_tp_to_gcp()

    def run_tp_to_gcp(self) -> None:

        if self.dem_info is not None:
            if self.dem_info.epsg_code != self.ref_img_info.epsg_code:
                new_dem_path = geoRT.ReprojectRaster(
                    input_raster_path=self.dem_path,
                    o_prj=self.ref_img_info.epsg_code,
                    vrt=True,
                    output_raster_path=os.path.join(os.path.dirname(self.output_gcp_path),
                                                    Path(self.dem_path).stem + "_" + str(
                                                        self.ref_img_info.epsg_code) + ".vrt"))
                if self.debug:
                    logging.info(f'Convert input DEM from:{self.dem_info.epsg_code} ---> {self.ref_img_info.epsg_code}')
                self.dem_path = new_dem_path
                self.dem_info = geoRT.cRasterInfo(new_dem_path)

        ## Convert Tie points of ref Image to Map coordinate
        x_map_ref_img, y_map_ref_img = self.ref_img_info.Pixel2Map(cols=self.ref_img_tps[:, 0],
                                                                   rows=self.ref_img_tps[:, 1])
        self.xy_map_ref_img = np.vstack((np.asarray(x_map_ref_img), np.asarray(y_map_ref_img))).T
        if self.debug:
            logging.info(f'xy_map_ref_img:{self.xy_map_ref_img.shape}')
        ## Convert Map coordinate to lon,Lat coordinates
        if self.ref_img_info.epsg_code != 4326:
            # lat, lon = geoRT.ConvCoordMap1ToMap2_Batch(X=self.xyMapRefImg[:, 0], Y=self.xyMapRefImg[:, 1],
            #                                            sourceEPSG=self.ref_img_info.EPSG_Code, targetEPSG=4326)
            res = geoRT.Convert.coord_map1_2_map2(X=self.xy_map_ref_img[:, 0], Y=self.xy_map_ref_img[:, 1],
                                                  sourceEPSG=self.ref_img_info.epsg_code, targetEPSG=4326)
            lat, lon = res[0], res[1]
            self.lon_lat_ref_img = np.vstack((np.asarray(lon), np.asarray(lat))).T
        else:
            self.lon_lat_ref_img = self.xy_map_ref_img

        alt = self.set_gcp_alt()
        self.gcps = self.lon_lat_ref_img
        self.gcps = np.insert(self.gcps, 2, np.asarray(alt).T, axis=1)
        self.gcps = np.insert(self.gcps, 3, self.base_img_tps[:, 0], axis=1)
        self.gcps = np.insert(self.gcps, 4, self.base_img_tps[:, 1], axis=1)
        self.write_gcps()
        if self.debug:
            self.debug_plot()
            # Fixme: memory issue when the image is very large: downsample the image before plotting
            # self.plot_gcps(ref_ortho_path=self.ref_img_path, raw_img_path=self.base_img_path, gcp_df=self.gcp_df,
            #                output_gcp_path=self.output_gcp_path)
        return

    def set_gcp_alt(self) -> List:
        alt = [0.0] * self.nb_Tps
        if self.dem_info is None:
            logging.warning('No DEM used - Altitude set at 0.0')
            alt = [0.0] * self.nb_Tps
            # TODO: add SRTM4/COPDEM in case no DEM is provided by the user
            return alt

        else:
            from multiprocessing import Pool, cpu_count
            arg_list = []

            for val, val_map in zip(self.lon_lat_ref_img, self.xy_map_ref_img):
                arg = (val[0], val[1], self.dem_info.input_raster_path)
                arg_list.append(arg)
            logging.info(f'#CPU:{cpu_count()}')
            with Pool(processes=int(cpu_count() / 2)) as pool:
                alt = pool.starmap(self.get_gcp_alt, arg_list)
            logging.info(f'alts:{len(alt)}')
            return alt

    def write_gcps(self) -> None:

        self.gcps = np.insert(self.gcps, 5, self.nb_Tps * [1], axis=1)
        self.gcps = np.insert(self.gcps, 6, self.nb_Tps * [1], axis=1)
        self.gcps = np.insert(self.gcps, 7, self.nb_Tps * [0], axis=1)
        self.gcps = np.insert(self.gcps, 8, self.nb_Tps * [0], axis=1)
        self.gcps = np.insert(self.gcps, 9, self.nb_Tps * [0], axis=1)
        dict: Dict = {'lon': [], 'lat': [], 'alt': [], 'xPix': [], "yPix": [], "weight": [], "opti": [], "dE": [],
                      "dN": [],
                      "dA": [], "x_map": [], 'y_map': [], 'epsg': []}
        dict['lon'] = self.gcps[:, 0]
        dict['lat'] = self.gcps[:, 1]
        dict['alt'] = self.gcps[:, 2]
        dict['xPix'] = self.gcps[:, 3]
        dict['yPix'] = self.gcps[:, 4]
        dict['weight'] = self.gcps[:, 5]
        dict['opti'] = self.gcps[:, 6]
        dict['dE'] = self.gcps[:, 7]
        dict['dN'] = self.gcps[:, 8]
        dict['dA'] = self.gcps[:, 9]
        dict['ref_img'] = self.gcps.shape[0] * [self.ref_img_path]
        dict['dem'] = self.gcps.shape[0] * [self.dem_path]
        dict['raw_img'] = self.gcps.shape[0] * [self.base_img_path]
        dict['gcp_id'] = [uuid.uuid4() for i in range(self.gcps.shape[0])]

        epsg = geoRT.ComputeEpsg(self.gcps[0, 0], self.gcps[0, 1])
        X_map, Y_map = geoRT.Convert.coord_map1_2_map2(X=dict['lat'],
                                                       Y=dict['lon'],
                                                       targetEPSG=epsg,
                                                       sourceEPSG=4326)
        dict['x_map'] = X_map
        dict['y_map'] = Y_map
        dict['epsg'] = self.gcps.shape[0] * [epsg]

        self.gcp_df = pandas.DataFrame.from_dict(dict)
        self.gcp_df.to_csv(self.output_gcp_path)

        return

    @staticmethod
    def get_gcp_alt(lon: float, lat: float, dem_path):
        # TODO
        #  1- need to verify if the window around the tie point is inside the DEM image --> Done wiht try Except
        #  2- verify if h= -32767 or NaN
        #  3- step value need to be a user defined parameter
        try:
            alt_ = HeightInterpolation.get_h_from_DEM_v2(geo_coords=[lat, lon],
                                                         dem_path=dem_path)[0]
        except:
            logging.warning(f'Enable to get alt val for point ({lon},{lat}) ---> alt= 0')
            alt_ = 0.0
        return alt_

    def debug_plot(self):
        """

        Returns:
        Note: we assume that the DEM and the ref_ortho_img have the same projection.
        """

        # TODO add RAW img ground extent
        import matplotlib.pyplot as plt
        import rasterio
        import shapely.geometry

        from geoCosiCorr3D.geoCore.constants import RENDERING
        from geoCosiCorr3D.georoutines.geo_utils import Convert
        fig, ax = plt.subplots(1, 1)
        src_dem = rasterio.open(self.dem_info.input_raster_path)
        fp_shp_dem = shapely.geometry.box(*src_dem.bounds)
        fp_shp_dem = Convert.polygon(fp_shp_dem, src_dem.crs.to_epsg(), 4326)
        ax.plot(fp_shp_dem.exterior.xy[0], fp_shp_dem.exterior.xy[1], color='#6699cc', alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2, label="DEM")
        src_ref = rasterio.open(self.ref_img_info.input_raster_path)
        fp_shp_ref = shapely.geometry.box(*src_ref.bounds)
        fp_shp_ref = Convert.polygon(fp_shp_ref, src_ref.crs.to_epsg(), 4326)
        ax.plot(fp_shp_ref.exterior.xy[0], fp_shp_ref.exterior.xy[1], color='#F08080', alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2, label='REF_IMG')
        ax.scatter(self.gcp_df['lon'], self.gcp_df['lat'], label="GCPs", c=RENDERING.GCP_COLOR,
                   s=RENDERING.GCP_SZ, marker=RENDERING.GCP_MARKER)

        ax.set_title(f'GCPs [{self.gcp_df.shape[0]}] + REF_ORTHO + DEM')
        ax.set_xlabel('LON [$^\circ$]')
        ax.set_ylabel('LAT [$^\circ$]')
        plt.legend()
        plt.savefig(f'{self.output_gcp_path}.png')
        src_ref, src_dem = None, None
        plt.clf()
        plt.switch_backend('agg')
        plt.close(fig)

        return

    @staticmethod
    def plot_gcps(ref_ortho_path, raw_img_path, gcp_df, output_gcp_path):
        import matplotlib.pyplot as plt
        import rasterio
        from rasterio.plot import show

        from geoCosiCorr3D.geoCore.constants import RENDERING
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig2.suptitle('GCPs + DEM + Ref_Ortho')
        ref_source = rasterio.open(ref_ortho_path)
        raw_source = rasterio.open(raw_img_path)
        show((ref_source, 1), ax=ax1, transform=ref_source.transform, cmap='gray')
        show((raw_source, 1), ax=ax2, cmap='gray')
        ax1.set_title("ref_ortho")
        ax2.set_title("raw_img")
        ax1.scatter(gcp_df['x_map'], gcp_df['y_map'], c=RENDERING.GCP_COLOR,
                    s=RENDERING.GCP_SZ, marker=RENDERING.GCP_MARKER)
        ax2.scatter(gcp_df['xPix'], gcp_df['yPix'], c=RENDERING.GCP_COLOR,
                    s=RENDERING.GCP_SZ, marker=RENDERING.GCP_MARKER)
        ax1.set_xlabel('X_MAP_UTM')
        ax1.set_ylabel('Y_MAP_UTM')
        # plt.show()
        plt.savefig(f'{output_gcp_path}.vis.png')
        src_ref, src_dem = None, None
        plt.clf()
        plt.close(fig2)
        return
