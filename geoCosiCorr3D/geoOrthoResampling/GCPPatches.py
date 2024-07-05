"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoOrthoResampling.geoResampling import Resampling
from geoCosiCorr3D.geoRSM.Ground2Pixel import RSMG2P


class PatchMapGrid:
    def __init__(self, grid_epsg: int, grid_res: float, up_left_ns: float, up_left_ew: float, bot_right_ns: float,
                 bot_right_ew: float):
        self.grid_epsg = grid_epsg
        self.grid_res = grid_res
        self.up_left_ew = up_left_ew
        self.bot_right_ew = bot_right_ew
        self.up_left_ns = up_left_ns
        self.bot_right_ns = bot_right_ns
        self.ground_map_extents = {C.GrdMapExtents.UP_LEFT_EW: up_left_ew,
                                   C.GrdMapExtents.UP_LEFT_NS: up_left_ns,
                                   C.GrdMapExtents.BOT_RIGHT_EW: bot_right_ew,
                                   C.GrdMapExtents.BOT_RIGHT_NS: bot_right_ns}

    def __repr__(self):
        return """# Grid Information :
            projEPSG = {}
            res = {}
            ground_extent= {}""".format(self.grid_epsg,
                                        self.grid_res,
                                        self.ground_map_extents)


class GCPPatch:

    @staticmethod
    def set_patch_dim(x_map_coord, y_map_coord, gcp_id, patch_sz, l3b_img_path, dem_info):
        valid_gcp = True
        l3b_ortho_info = geoRT.cRasterInfo(l3b_img_path)
        dem_patch_dim = {'x_dem_min_pix': 0, 'x_dem_max_pix': 0, 'y_dem_min_pix': 0, 'y_dem_max_pix': 0}
        ref_ortho_patch_dim_pix, patch_map_extent = \
            GCPPatch.get_patch_dim_l3b(patch_sz_pix=[patch_sz[0], patch_sz[1]],
                                       gcp_coord=[x_map_coord, y_map_coord],
                                       l3b_raster_info=l3b_img_path)
        if np.all(ref_ortho_patch_dim_pix.values() == 0) and np.all(patch_map_extent.values() == 0):
            msg = f'Patch ref boundaries for GCP {gcp_id} is outside ref l3b image'
            logging.warning(msg)
            valid_gcp = False
        else:
            # Determine the subset map coordinates boundaries (in the Reference image projection)
            x_ref_min_map, y_ref_min_map = l3b_ortho_info.Pixel2Map(
                cols=ref_ortho_patch_dim_pix.get(C.ImgExtents.COL_MIN),
                rows=ref_ortho_patch_dim_pix.get(C.ImgExtents.LIN_MIN))
            x_ref_max_map, y_ref_max_map = l3b_ortho_info.Pixel2Map(
                cols=ref_ortho_patch_dim_pix.get(C.ImgExtents.COL_MAX),
                rows=ref_ortho_patch_dim_pix.get(C.ImgExtents.LIN_MAX))
            ref_bbox_map = [x_ref_min_map, y_ref_min_map, x_ref_max_map, y_ref_max_map]
            if dem_info is not None:
                # Determination of the pixel corners coordinates of the subset in the DEM (with gcp corrected)
                # Determination of the DEM subset covering the patch area -
                # The subset is enlarged of 5 pixels for the interpolating need

                dem_patch_dim = GCPPatch.get_dem_subset_dim(bbox_map_coord=ref_bbox_map, dem_info=dem_info,
                                                            margin=5)
                # Check if the subset is fully inside the DEM boundaries
                if np.all(dem_patch_dim.values() == 0):
                    msg = f'Patch DEM boundaries for GCP {gcp_id} is outside ref DEM'
                    logging.warning(msg)
        return ref_ortho_patch_dim_pix, patch_map_extent, dem_patch_dim, valid_gcp

    @staticmethod
    def get_dem_subset_dim(bbox_map_coord: List, dem_info: geoRT.cRasterInfo, margin: Optional[int] = 5) -> Dict:
        """
        Determining the DEM subset covering the patch area.
        The subset is enlarged of margin pixels for interpolation.
        Args:

            margin: padding margin ## TODO: add this parameter to configuration file.

        Returns: Subset pixel corners coordinates of the DEM :
                [1, xDemMinPix, xDemMaxPix, yDemMinPix, yDemMaxPix] or [0,0,0,0,0]

        """
        dem_bbox_pix = {'x_dem_min_pix': 0, 'x_dem_max_pix': 0, 'y_dem_min_pix': 0, 'y_dem_max_pix': 0}
        xRefMin_map, yRefMin_map, xRefMax_map, yRefMax_map = bbox_map_coord[0], bbox_map_coord[1], bbox_map_coord[2], \
            bbox_map_coord[3]
        xDemMinPix, yDemMinPix = dem_info.Map2Pixel(x=xRefMin_map, y=yRefMin_map)
        xDemMaxPix, yDemMaxPix = dem_info.Map2Pixel(x=xRefMax_map, y=yRefMax_map)
        xDemMinPix = int(xDemMinPix - margin)
        yDemMinPix = int(yDemMinPix - margin)
        xDemMaxPix = int(xDemMaxPix + margin)
        yDemMaxPix = int(yDemMaxPix + margin)
        # Check if the subset is fully inside the DEM boundaries
        if xDemMinPix < 0 or xDemMaxPix > dem_info.raster_width or \
                yDemMinPix < 0 or yDemMaxPix > dem_info.raster_height:
            # warnings.warn("DEM subset is outside DEM image boundaries")
            return dem_bbox_pix
        else:
            # Store the subset pixel corners coordinates in the DEM
            return {'x_dem_min_pix': xDemMinPix, 'x_dem_max_pix': xDemMaxPix, 'y_dem_min_pix': yDemMinPix,
                    'y_dem_max_pix': yDemMaxPix}

    @staticmethod
    def get_patch_dim_l3b(patch_sz_pix: List, gcp_coord: List, l3b_raster_info: geoRT.cRasterInfo, tol: float = 1e-3):
        # TODO update the description
        """
            Determining the Reference Image subset surrounding the gcp in the Reference Image.
            Determining the Geographic/UTM coordinates of the subset corners of the on-coming gcpPatch
            Args:
                tol: 1e-3 : offset to be added for subpixel accuracy
            Returns:
                Subset pixel corners coordinates of the refImage and gcpPatch dimensions
        """
        # l3b_raster_info = geoRT.cRasterInfo(l3b_img_path)
        fullwinszX, fullwinszY = patch_sz_pix[0], patch_sz_pix[1]
        x_ref_pix_, y_ref_pix_ = l3b_raster_info.Map2Pixel(x=gcp_coord[0], y=gcp_coord[1])
        x_ref_pix = math.floor(x_ref_pix_)
        y_ref_pix = math.floor(y_ref_pix_)
        # BBOX
        x_ref_min_pix = x_ref_pix - fullwinszX / 2 + 1
        x_ref_max_pix = x_ref_pix + fullwinszX / 2
        y_ref_min_pix = y_ref_pix - fullwinszY / 2 + 1
        y_ref_max_pix = y_ref_pix + fullwinszY / 2
        bbox_pix = {C.ImgExtents.COL_MIN: 0, C.ImgExtents.COL_MAX: 0, C.ImgExtents.LIN_MIN: 0, C.ImgExtents.LIN_MAX: 0}
        bbox_map = {C.GrdMapExtents.UP_LEFT_EW: 0,
                    C.GrdMapExtents.UP_LEFT_NS: 0,
                    C.GrdMapExtents.BOT_RIGHT_EW: 0,
                    C.GrdMapExtents.BOT_RIGHT_NS: 0}
        ## Check if the subset is fully inside the reference image boundaries
        if x_ref_min_pix < 0 or x_ref_max_pix > l3b_raster_info.raster_width \
                or y_ref_min_pix < 0 or y_ref_max_pix > l3b_raster_info.raster_height:
            return bbox_pix, bbox_map

        else:
            bbox_pix = {C.ImgExtents.COL_MIN: x_ref_min_pix,
                        C.ImgExtents.COL_MAX: x_ref_max_pix,
                        C.ImgExtents.LIN_MIN: y_ref_min_pix,
                        C.ImgExtents.LIN_MAX: y_ref_max_pix}
            deltaX_map = 0
            deltaY_map = 0

            if (x_ref_pix_ - x_ref_pix) > tol:
                # offset to be added for subpixel accuracy -
                deltaX_map = (x_ref_pix_ - x_ref_pix) * np.abs(l3b_raster_info.pixel_width)
            if (y_ref_pix_ - y_ref_pix) > tol:
                # offset to be added for subpixel accuracy
                deltaY_map = (y_ref_pix_ - y_ref_pix) * np.abs(l3b_raster_info.pixel_height)

            x_map_coord = gcp_coord[0] - deltaX_map
            y_map_coord = gcp_coord[1] + deltaY_map

            x_map_min = x_map_coord - (fullwinszX / 2 - 1) * np.abs(l3b_raster_info.pixel_width)
            y_map_max = y_map_coord + (fullwinszY / 2 - 1) * np.abs(l3b_raster_info.pixel_height)
            up_left_ew = x_map_min
            up_left_ns = y_map_max

            bot_right_ew = x_map_min + np.abs(l3b_raster_info.pixel_width) * fullwinszX
            bot_right_ns = y_map_max - np.abs(l3b_raster_info.pixel_height) * fullwinszY
            (bbox_map[C.GrdMapExtents.UP_LEFT_EW], bbox_map[C.GrdMapExtents.UP_LEFT_NS],
             bbox_map[C.GrdMapExtents.BOT_RIGHT_EW], bbox_map[C.GrdMapExtents.BOT_RIGHT_NS]) \
                = up_left_ew, up_left_ns, bot_right_ew, bot_right_ns
            return bbox_pix, bbox_map

    @staticmethod
    def get_patch_dim_l1a(patch_sz_pix: List, kp_coord: List, l1a_raster_info: geoRT.cRasterInfo, tol: float = 1e-3):
        # TODO
        pass


class OrthoPatch(GCPPatch):
    def __init__(self, ortho_info: geoRT.cRasterInfo, patch_sz: List[int], gcp_df: pandas.DataFrame):
        self.ortho_info = ortho_info
        self.gcp_df = gcp_df
        self.patch_sz = patch_sz
        self.patch_status = True

    def __call__(self, *args, **kwargs) -> C.Patch:
        self.compute_ortho_patch_extent()
        return C.Patch(id=self.gcp_df[C.GCPKeys.ID],
                       data=self.compute_ortho_patch(),
                       status=self.patch_status,
                       img_extent=self.ortho_patch_pix_extent,
                       ground_extent=self.ortho_patch_map_extent,
                       gsd=np.abs(self.ortho_info.pixel_width)
                       )

    def compute_ortho_patch(self):
        return self.ortho_info.image_as_array_subset(
            col_off_min=int(self.ortho_patch_pix_extent[C.ImgExtents.COL_MIN]),
            col_off_max=int(self.ortho_patch_pix_extent[C.ImgExtents.COL_MAX]),
            row_off_min=int(self.ortho_patch_pix_extent[C.ImgExtents.LIN_MIN]),
            row_off_max=int(self.ortho_patch_pix_extent[C.ImgExtents.LIN_MAX]))

    def compute_ortho_patch_extent(self):
        self.ortho_patch_pix_extent, self.ortho_patch_map_extent = \
            GCPPatch.get_patch_dim_l3b(patch_sz_pix=[self.patch_sz[0], self.patch_sz[1]],
                                       gcp_coord=[self.gcp_df[C.GCPKeys.X_MAP], self.gcp_df[C.GCPKeys.Y_MAP]],
                                       l3b_raster_info=self.ortho_info)

        if np.all(self.ortho_patch_pix_extent.values() == 0) and np.all(self.ortho_patch_map_extent.values() == 0):
            gcp_id = self.gcp_df.get(C.GCPKeys.ID, [self.gcp_df[C.GCPKeys.X_MAP], self.gcp_df[C.GCPKeys.Y_MAP]])
            msg = f'Patch ref boundaries for GCP {gcp_id} is outside ref l3b image'
            logging.warning(msg)
            self.patch_status = False
        return


class RawInverseOrthoPatch(GCPPatch):
    def __init__(self,
                 raw_img_info: geoRT.cRasterInfo,
                 gcp_df,
                 sat_model,
                 corr_model,
                 patch_sz,
                 patch_epsg,
                 patch_res,
                 patch_map_extent,
                 dem_path: Optional[str] = None,
                 model_type=C.SATELLITE_MODELS.RSM,
                 resampling_method=C.Resampling_Methods.SINC,

                 debug=False):
        self.debug = debug
        self.patch_sz = patch_sz
        self.raw_img_info = raw_img_info
        self.sat_model = sat_model
        self.corr_model = corr_model
        self.model_type = model_type
        self.dem_path = dem_path
        self.resampling_method = resampling_method
        self.patch_status = True  # TODO
        self.gcp_df = gcp_df
        self.ingest()
        self.raw_ortho_patch_map_grid = PatchMapGrid(patch_epsg,
                                                     patch_res,
                                                     patch_map_extent[C.GrdMapExtents.UP_LEFT_NS],
                                                     patch_map_extent[C.GrdMapExtents.UP_LEFT_EW],
                                                     patch_map_extent[C.GrdMapExtents.BOT_RIGHT_NS],
                                                     patch_map_extent[C.GrdMapExtents.BOT_RIGHT_EW])
        if self.debug:
            logging.info(f'{self.__class__.__name__}:GCP patch GCP:{repr(self.raw_ortho_patch_map_grid)}')

    def __call__(self, *args, **kwargs) -> C.Patch:
        patch_array = self.compute_ortho_patch()
        return C.Patch(id=self.gcp_df[C.GCPKeys.ID],
                       data=patch_array,
                       status=self.patch_status,
                       img_extent=self.patch_img_extent,
                       ground_extent=self.raw_ortho_patch_map_grid.ground_map_extents,
                       gsd=self.raw_ortho_patch_map_grid.grid_res)

    def ingest(self):
        if self.dem_path is not None:
            self.dem_info = geoRT.cRasterInfo(self.dem_path)
        else:
            self.dem_info = None
        return

    def compute_ortho_patch(self):
        if self.model_type == C.SATELLITE_MODELS.RSM:
            patch_ortho_obj = RsmOrthoPatch(raw_img_info=self.raw_img_info,
                                            rsm_model=self.sat_model,
                                            patch_map_grid=self.raw_ortho_patch_map_grid,
                                            rsm_corr_model=self.corr_model,
                                            dem_info=self.dem_info,
                                            resampling_method=self.resampling_method,
                                            debug=self.debug,
                                            transform_model_shape=self.patch_sz)
            self.patch_img_extent = patch_ortho_obj.patch_img_extent
            return patch_ortho_obj.ortho_patch
        elif self.model_type == C.SATELLITE_MODELS.RFM:
            raise NotImplementedError
        else:
            raise ValueError(f'{self.__class__.__name__}: Model type {self.model_type} is not supported')


class RsmOrthoPatch:
    def __init__(self,
                 raw_img_info,
                 rsm_model,
                 patch_map_grid: PatchMapGrid,
                 rsm_corr_model=None,
                 dem_info=None,
                 transform_model_shape=None,
                 resampling_method=C.Resampling_Methods.SINC,
                 debug=False):

        self.grid = patch_map_grid
        self.transform_model_shape = transform_model_shape
        self.rsm_corr_model = rsm_corr_model
        self.rsm_model = rsm_model
        self.dem_info = dem_info
        self.resampling_method = resampling_method
        self.raw_img_info = raw_img_info
        self.debug = debug
        self._ingest()
        self.compute_transformation_model()

        if self.debug:
            logging.info(f'{self.__class__.__name__}: patch tx_model:{self.transform_model.shape}')
        self.orthorectify()

    def _ingest(self):

        if self.rsm_corr_model is None:
            self.rsm_corr_model = np.zeros((3, 3))
        if self.transform_model_shape is None:
            nb_cols = round((self.grid.bot_right_ew - self.grid.up_left_ew) / self.grid.grid_res + 1)
            nb_rows = round((self.grid.up_left_ns - self.grid.bot_right_ns) / self.grid.grid_res + 1)
            self.transform_model_shape = (nb_rows, nb_cols)

        return

    def compute_transformation_model(self):

        easting = self.grid.up_left_ew + np.arange(self.transform_model_shape[1]) * self.grid.grid_res
        northing = self.grid.up_left_ns - np.arange(self.transform_model_shape[0]) * self.grid.grid_res

        transform_model_obj = RSMG2P(rsmModel=self.rsm_model,
                                     xMap=easting,
                                     yMap=northing,
                                     projEPSG=self.grid.grid_epsg,
                                     rsmCorrection=self.rsm_corr_model,
                                     demInfo=self.dem_info,
                                     debug=self.debug)

        self.transform_model = transform_model_obj.get_pix_coords()
        if self.debug:
            logging.info(
                f'Trx model:x_range:{np.min(self.transform_model[0]), np.max(self.transform_model[0])}, '
                f'y_range:{np.min(self.transform_model[1]), np.max(self.transform_model[1])}')
        del transform_model_obj
        return

    def orthorectify(self):
        resample = Resampling(input_raster_info=self.raw_img_info, transformation_mat=self.transform_model,
                              resampling_params={'method': self.resampling_method}, debug=self.debug)

        self.ortho_patch = resample.resample()
        self.patch_img_extent = resample.raw_l1a_subset_img_extent
        return


class RFMOrthorectifyPatch:
    # TODO
    pass
