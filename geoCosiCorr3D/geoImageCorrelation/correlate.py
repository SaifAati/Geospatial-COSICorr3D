"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Any, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import geoCosiCorr3D.geoImageCorrelation.geoCorr_utils as utils
from geoCosiCorr3D.geoCore.constants import CORRELATION
from geoCosiCorr3D.geoCore.core_correlation import RawCorrelation, FreqCorrelator, SpatialCorrelator
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo


class Correlate(RawCorrelation):

    def __init__(self, base_image_path: str,
                 target_image_path: str,
                 corr_config: Optional[Dict] = None,
                 base_band: int = 1,
                 target_band: int = 1,
                 output_corr_path: Optional[str] = None,
                 corr_show: Optional[bool] = True):
        self.corr_config = corr_config
        if self.corr_config is None:
            self.corr_config = {}

        super().__init__(base_image_path,
                         target_image_path,
                         self.corr_config,
                         base_band,
                         target_band,
                         output_corr_path)

        self._ingest()
        self.run_correlation()

        if corr_show:
            logging.info(f'{self.__class__.__name__}: Plotting correlation results')
            self.plt_cfg = self.get_corr_config.get_config['CORRELATION_PLOT']
            self.plot_correlation_map(corr_path=self.output_corr_path, cfg=self.plt_cfg,
                                      ground_space=self.flags.get("groundSpaceCorr"))
        return

    def run_correlation(self):

        ##Check that the images have identical projection reference system

        self.check_same_projection_system()

        ## Check if the images have the same ground resolution

        self.check_same_ground_resolution()
        # ## Check that the images are on geographically aligned grids (depends on origin and resolution)
        # ## verify if the difference between image origin is less than of resolution/1000

        self._check_aligned_grids()

        # ## Depending on the validity of the map information of the images, the pixel resolution is setup

        self.set_corr_map_resolution()

        self.win_area_x, self.win_area_y = self._set_win_area(self.corr_engine.corr_params.window_size,
                                                              self.margins)
        if self.debug:
            logging.info("winAreaX:.{}, winAreaX:.{}".format(self.win_area_x, self.win_area_y))

        # Backup of the original base (reference) subset dimensions in case of a non gridded correlation
        if self.corr_engine.corr_params.grid == False:
            self.base_original_dims: List[float] = self.base_dims_pix
        """
            Cropping the images to the same size:
            Two conditions exist:
                1- if the map information are valid and identical: we define the overlapping area based on geo-referencing
                2- if map information invalid or different: we define the overlapping area based we define overlapping area
                    based on images size (pixel wise)
        """
        self.crop_to_same_size()

        """
           Adjusting cropped images according to a gridded/non-gridded output
           Two cases:
               1- If the user selected the gridded option
               2- If the user selected non-gridded option
       """
        self.adjusting_cropped_images_according_2_grid_nongrid()

        if self.tot_col <= 0 or self.tot_row <= 0:
            logging.error('Not enough overlapping area for correlation')
            sys.exit('Not enough overlapping area for correlation')
        if self.debug:
            logging.info("base dims pix: {}".format(self.base_dims_pix))
            logging.info("target dims pix: {}".format(self.target_dims_pix))

        self.tiling()

        self.nb_measurment_per_roi = self.nb_corr_col_per_roi * self.nb_corr_row_per_roi * len(
            self.corr_engine.corr_bands)
        self.output_array = np.empty(self.nb_measurment_per_roi, dtype=np.float32)
        # print("output.shape", outputArray.shape)

        self.loc_Array = np.arange(self.nb_corr_col_per_roi * self.nb_corr_row_per_roi) * len(
            self.corr_engine.corr_bands)
        # print("locArr.shape:", locArr.shape)

        ew_array_list: List[Any] = []
        ns_array_list: List[Any] = []
        snr_array_list: List[Any] = []

        for roi in tqdm(range(self.nb_roi), desc="Correlation per tile"):
            utils.__mark_progress__(roi * 100 / self.nb_roi)
            if self.debug:
                logging.info("Tile:{}/{} ".format(roi + 1, self.nb_roi))

                logging.info("base tile dims :{}".format(self.dims_base_tile[roi]))
                logging.info("target tile info:{}".format(self.dims_target_tile[roi]))

            base_subset = self.base_info.image_as_array_subset(self.dims_base_tile[roi][:][1],
                                                               self.dims_base_tile[roi][:][2],
                                                               self.dims_base_tile[roi][:][3],
                                                               self.dims_base_tile[roi][:][4],
                                                               self.base_band_nb)

            target_subset = self.target_info.image_as_array_subset(self.dims_target_tile[roi][:][1],
                                                                   self.dims_target_tile[roi][:][2],
                                                                   self.dims_target_tile[roi][:][3],
                                                                   self.dims_target_tile[roi][:][4],
                                                                   self.target_band_nb)

            if self.debug:
                logging.info("baseSubset.size:{} ".format(base_subset.shape))
                logging.info("targetSubset.size:{} ".format(target_subset.shape))

            if (self.nb_roi > 1) and roi == (self.nb_roi - 1):
                if self.debug:
                    logging.info("--- LAST TILE ----")
                self.output_array = np.zeros(
                    (self.nb_corr_row_per_roi * self.nb_corr_row_last_roi * len(self.corr_engine.corr_bands)))
                self.loc_Array = np.arange(self.nb_corr_col_per_roi * self.nb_corr_row_last_roi) * len(
                    self.corr_engine.corr_bands)
                self.nb_corr_row_per_roi = self.nb_corr_row_last_roi

            ew_array, ns_array, snr_array = self.run_corr_engine(base_subset, target_subset)

            ew_array_list.append(ew_array * self.x_res)
            ns_array_list.append(ns_array * (-self.y_res))
            snr_array_list.append(snr_array)

        self.ew_output = np.vstack(tuple(ew_array_list))
        self.ns_output = np.vstack(tuple(ns_array_list))
        self.snr_output = np.vstack(tuple(snr_array_list))

        if self.corr_engine.corr_params.grid == False:
            self.write_blank_pixels()

        geo_transform, epsg_code = self.set_geo_referencing()

        cRasterInfo.write_raster(output_raster_path=self.output_corr_path,
                                 array_list=[self.ew_output, self.ns_output, self.snr_output],
                                 geo_transform=geo_transform,
                                 epsg_code=epsg_code,
                                 dtype="float32",
                                 descriptions=self.corr_engine.corr_bands)
        utils.__mark_progress__(100)

    def run_corr_engine(self, base_array, target_array):

        if self.corr_engine.correlator_name == CORRELATION.FREQUENCY_CORRELATOR:
            return FreqCorrelator.run_correlator(target_array=target_array,
                                                 base_array=base_array,
                                                 window_size=self.corr_engine.corr_params.window_size,
                                                 step=self.corr_engine.corr_params.step,
                                                 iterations=self.corr_engine.corr_params.nb_iter,
                                                 mask_th=self.corr_engine.corr_params.mask_th)

        if self.corr_engine.correlator_name == CORRELATION.SPATIAL_CORRELATOR:
            return SpatialCorrelator.run_correlator(base_array=base_array,
                                                    target_array=target_array,
                                                    window_size=self.corr_engine.corr_params.window_size,
                                                    step=self.corr_engine.corr_params.step,
                                                    search_range=self.corr_engine.corr_params.search_range)

    @classmethod
    def from_config(cls, config):
        correlator = config['correlator_name']
        params = config['correlator_params']
        if correlator == 'spatial':
            if "mask_th" in params: del params["mask_th"]
            if "nb_iters" in params: del params["nb_iters"]
        elif correlator == 'frequency':
            if "search_range" in params: del params["search_range"]

        print("Configuration generated successfully:")
        print()
        print(json.dumps(config, indent=4))
        print()
        print("Initiating correlation...")

        cls(base_image_path=config['base_image_path'],
            target_image_path=config['target_image_path'],
            base_band=config['base_band'],
            target_band=config['target_band'],
            output_corr_path=config['output_path'],
            corr_config=config)

    def corr_plot(self):

        from geoCosiCorr3D.georoutines.geoplt_misc import ColorBar_

        # TODO add to config the visualization params
        axs_labels = [self.plt_cfg['axs_ground']['x_label'], self.plt_cfg['axs_ground']['y_label']]
        cbar_label = self.plt_cfg['axs_ground']['cbar_label']
        if not self.flags.get("groundSpaceCorr"):
            axs_labels = [self.plt_cfg['axs_img']['x_label'], self.plt_cfg['axs_img']['y_label']]
            cbar_label = self.plt_cfg['axs_img']['cbar_label']

        cmap = self.plt_cfg['cmap']
        title = None
        dpi = self.plt_cfg['dpi']
        vmin = self.plt_cfg['cmap_range']['min']  # TODO set to the correlation resolution
        vmax = self.plt_cfg['cmap_range']['max']
        fig, axs = plt.subplots(1, 2)  # , figsize=(16, 9))

        imEW = axs[0].imshow(self.ew_output, cmap=cmap, vmin=vmin, vmax=vmax)
        imNS = axs[1].imshow(self.ns_output, cmap=cmap, vmin=vmin, vmax=vmax)
        # for ax, title_ in zip(axs, ["East/West", "North/South"]):
        for ax, title_ in zip(axs, axs_labels):
            ax.axis('off')
            ax.set_title(title_)
        ColorBar_(ax=axs[0], mapobj=imEW, cmap=cmap, vmin=vmin, vmax=vmax, orientation="vertical",
                  label=cbar_label)
        ColorBar_(ax=axs[1], mapobj=imNS, cmap=cmap, vmin=vmin, vmax=vmax, orientation="vertical",
                  label=cbar_label)
        if title is None:
            title = Path(self.output_corr_path).stem
        plt.suptitle(title)
        # fig.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(self.output_corr_path), title + '.' + self.plt_cfg['format']),
                    dpi=dpi, )
        # bbox_inches='tight')#,pad_inches = 2)
        fig.clear()
        plt.close(fig)
        return

    @staticmethod
    def plot_correlation_map(corr_path, cfg, ground_space=False, vmin=-2, vmax=2, title=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        from geoCosiCorr3D.geoImageCorrelation.plot import show
        from geoCosiCorr3D.georoutines.geo_utils import ReprojectRaster
        from matplotlib.ticker import (AutoMinorLocator, MaxNLocator)
        BORDER_PAD = 2
        corr_raster_info = cRasterInfo(corr_path)

        ew_array = corr_raster_info.raster_array[0]
        ns_array = corr_raster_info.raster_array[1]

        cmap = cfg['cmap']
        fig, axs = plt.subplots(1, 2)  # , layout="constrained")  # , figsize=(16, 9))

        if ground_space:
            cmap = 'RdBu'  # 'bwr'
            temp_raster = ReprojectRaster(input_raster_path=corr_raster_info.input_raster_path,
                                          o_prj=4326,
                                          vrt=True)
            corr_raster_info = cRasterInfo(temp_raster)

            _, im1 = show(ew_array, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, ax=axs[0],
                          transform=corr_raster_info.geo_transform_affine)
            _, im2 = show(ns_array, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, ax=axs[1],
                          transform=corr_raster_info.geo_transform_affine)

            axs[1].set_xlabel('lon[$^\circ$]', fontsize=12)
            axs[0].set_xlabel('lon[$^\circ$]', fontsize=12)
            axs[0].set_ylabel('lat[$^\circ$]', fontsize=12)

            for ax, title_, im in zip(axs, ["E-W [m]", "N-S [m]"], [im1, im2]):
                # ax.set_title(title_)
                axins = inset_axes(ax,
                                   width="60%",
                                   height="3%",
                                   loc='upper center',
                                   borderpad=-3
                                   )
                cbar1 = fig.colorbar(im, cax=axins, orientation="horizontal", extend='both')
                cbar1.ax.set_title(title_)

        else:
            _, im1 = show(ew_array, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, ax=axs[0])
            _, im2 = show(ns_array, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, ax=axs[1])

            # axs[1].set_xlabel('lon[$^\circ$]', fontsize=12)
            # axs[0].set_xlabel('lon[$^\circ$]', fontsize=12)
            # axs[0].set_ylabel('lat[$^\circ$]', fontsize=12)
            for ax, title_, im in zip(axs, ["dx [px]", "dy [px]"], [im1, im2]):
                # ax.set_title(title_)
                axins = inset_axes(ax,
                                   width="60%",
                                   height="3%",
                                   loc='upper center',
                                   borderpad=-BORDER_PAD
                                   )
                cbar1 = fig.colorbar(im, cax=axins, orientation="horizontal", extend='both')
                cbar1.ax.set_title(title_)

        axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        axs[0].xaxis.set_major_locator(MaxNLocator(3))
        axs[0].yaxis.set_major_locator(MaxNLocator(3))
        axs[0].tick_params(direction='in', which='minor', top=True, right=True)
        axs[0].tick_params(direction='in', which='major', top=True, right=True)

        axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[1].yaxis.set_minor_locator(AutoMinorLocator())
        axs[1].xaxis.set_major_locator(MaxNLocator(3))
        axs[1].yaxis.set_major_locator(MaxNLocator(3))
        axs[1].tick_params(direction='in', which='minor', top=True, right=True)
        axs[1].tick_params(direction='in', which='major', top=True, right=True)
        axs[1].set_yticklabels([])
        if title is None:
            title = Path(corr_path).stem

        fig.suptitle(title, fontsize=14)
        dpi = cfg['dpi']

        fig.tight_layout(pad=BORDER_PAD)  ##pad=4, w_pad=5, )

        plt.savefig(os.path.join(os.path.dirname(corr_path), Path(corr_path).stem + '.png'), dpi=dpi,
                    bbox_inches='tight')
        fig.clear()
        plt.close(fig)
        return
