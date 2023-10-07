"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import logging
import os
import sys
import warnings
from ctypes import cdll
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import geoCosiCorr3D.geoImageCorrelation.misc as misc
from geoCosiCorr3D.geoCore.base.base_correlation import (BaseCorrelation,
                                                         BaseCorrelationEngine,
                                                         BaseFreqCorr,
                                                         BaseSpatialCorr)
from geoCosiCorr3D.geoCore.constants import CORRELATION
from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo

FREQ_CORR_LIB = CORRELATION.FREQ_CORR_LIB
STAT_CORR_LIB = CORRELATION.STAT_CORR_LIB


# TODO:
#  1- add flag to perform pixel-based correlation
#  2- Base class for overlap based on projection and based on pixel, for the pixel based we need support x_off and y_off
#  3- Support: Optical flow correlation, MicMac, ASP, Sckit-image, OpenCV corr, ....
#  4- Data sets: geometry artifacts (PS, HiRISE,WV, GF), glacier , cloud detection, earthquake, landslide, dune

class InvalidCorrLib(Exception):
    pass


class RawFreqCorr(BaseFreqCorr):
    def __init__(self,
                 window_size: List[int] = None,
                 step: List[int] = None,
                 mask_th: float = None,
                 resampling: bool = False,
                 nb_iter: int = 4,
                 grid: bool = True):

        if window_size is None:
            self.window_size = 4 * [64]
        else:
            self.window_size = window_size

        if step is None:
            self.step = 2 * [8]
        else:
            self.step = step
        if mask_th is None:
            self.mask_th = 0.9
        else:
            self.mask_th = mask_th
        self.resampling = resampling
        self.nb_iter = nb_iter
        self.grid = grid
        return

    @staticmethod
    def ingest_freq_corr_params(params: Dict) -> List[Any]:
        """

        Args:
            params:

        Returns:

        """
        window_size = params.get("window_size", 4 * [64])
        step = params.get("step", 2 * [8])
        mask_th = params.get("mask_th", 0.9)
        nb_iters = params.get("nb_iters", 4)
        grid = params.get("grid", True)
        return [window_size, step, mask_th, nb_iters, grid]

    @staticmethod
    def set_margins(resampling: bool, window_size: List[int]) -> List[int]:
        """

        Args:
            resampling:
            window_size:

        Returns:

        """
        if ~resampling:
            margins = [int(window_size[0] / 2), int(window_size[1] / 2)]
            logging.info("corr margins: {}".format(margins))
            return margins
        else:
            logging.warning("Compute margin based on resampling Kernel ! ")
            raise NotImplementedError

    # TODO: change to static method or adapt to class method
    @classmethod
    def run_correlator(cls,
                       base_array: np.ndarray,
                       target_array: np.ndarray,
                       window_size: List[int],
                       step: List[int],
                       iterations: int,
                       mask_th: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            lib_cfreq_corr = cdll.LoadLibrary(FREQ_CORR_LIB)
        except:
            raise InvalidCorrLib
        lib_cfreq_corr.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.int32),
                                             np.ctypeslib.ndpointer(dtype=np.float32),
                                             np.ctypeslib.ndpointer(dtype=np.int32)]

        input_shape = np.array(base_array.shape, dtype=np.int32)
        window_sizes = np.array(window_size, dtype=np.int32)
        step_sizes = np.array(step, dtype=np.int32)
        base_array_ = np.array(base_array.flatten(), dtype=np.float32)
        target_array_ = np.array(target_array.flatten(), dtype=np.float32)
        ew_array_ = np.zeros((input_shape[0] * input_shape[1], 1), dtype=np.float32)
        ns_array_ = np.zeros((input_shape[0] * input_shape[1], 1), dtype=np.float32)
        snr_array_ = np.zeros((input_shape[0] * input_shape[1], 1), dtype=np.float32)
        iteration = np.array([iterations], dtype=np.int32)
        mask_threshold = np.array([mask_th], dtype=np.float32)
        output_shape = np.array([0, 0], dtype=np.int32)
        lib_cfreq_corr.InputData(input_shape,
                                 window_sizes,
                                 step_sizes,
                                 base_array_,
                                 target_array_,
                                 ew_array_,
                                 ns_array_,
                                 snr_array_,
                                 iteration,
                                 mask_threshold,
                                 output_shape)

        ew_array_fl = ew_array_[0:output_shape[0] * output_shape[1]]
        ns_array_fl = ns_array_[0:output_shape[0] * output_shape[1]]
        snr_array_fl = snr_array_[0:output_shape[0] * output_shape[1]]
        ew_array = np.asarray(ew_array_fl).reshape((output_shape[0], output_shape[1]))
        ns_array = np.asarray(ns_array_fl).reshape((output_shape[0], output_shape[1]))
        snr_array = np.asarray(snr_array_fl).reshape((output_shape[0], output_shape[1]))

        return ew_array, ns_array, snr_array


class RawSpatialCorr(BaseSpatialCorr):

    def __init__(self,
                 window_size: List[int] = None,
                 steps: List[int] = None,
                 search_range: List[int] = None,
                 grid: bool = False):
        """

        Args:
            window_size:
            steps:
            search_range:
            grid:
        """
        if window_size is None:
            self.window_size = 2 * [32]
        else:
            self.window_size = window_size
        if steps is None:
            self.step = 2 * [16]
        else:
            self.step = steps
        if search_range is None:
            self.search_range = 2 * [10]
        else:
            self.search_range = search_range

        self.grid = grid

        return

    @staticmethod
    def get_output_dims(step_size: List[int],
                        input_shape: Tuple[int, int],
                        window_size: List[int],
                        range_size: List[int]) -> Tuple[int, int]:
        """

        Args:
            step_size:
            input_shape:
            window_size:
            range_size:

        Returns:

        """

        if (step_size[0] != 0):
            value = input_shape[1] - (window_size[0] + 2 * range_size[0])
            output_cols = int((np.floor(value / step_size[0] + 1.0)))
        else:
            output_cols = 1
        if (step_size[1] != 0):

            value = (input_shape[0] - (window_size[1] + 2 * range_size[1]))
            output_rows = int(np.floor(value / step_size[1] + 1.0))
        else:
            output_rows = 1
        return (output_rows, output_cols)

    @classmethod
    def run_correlator(cls, base_array: np.ndarray,
                       target_array: np.ndarray,
                       window_size: List[int],
                       step: List[int],
                       search_range: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            libCstatCorr = cdll.LoadLibrary(STAT_CORR_LIB)
        except:
            raise InvalidCorrLib

        libCstatCorr.InputData.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.int32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32),
                                           np.ctypeslib.ndpointer(dtype=np.float32)]

        inputShape = np.array(base_array.shape, dtype=np.int32)
        windowSizes = np.array(window_size, dtype=np.int32)
        stepSizes = np.array(step, dtype=np.int32)
        searchRanges = np.array(search_range, dtype=np.int32)
        baseArray = np.array(base_array.flatten(), dtype=np.float32)
        targetArray = np.array(target_array.flatten(), dtype=np.float32)

        outputRows, outputCols = cls.get_output_dims(step_size=step,
                                                     input_shape=base_array.shape,
                                                     window_size=window_size,
                                                     range_size=search_range)
        outputShape = np.array([outputRows, outputCols], dtype=np.int32)

        ewArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)
        nsArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)
        snrArray_fl = np.zeros((outputShape[0] * outputShape[1], 1), dtype=np.float32)

        libCstatCorr.InputData(inputShape, windowSizes, stepSizes, searchRanges, baseArray, targetArray, outputShape,
                               ewArray_fl, nsArray_fl, snrArray_fl)

        ew_array = np.asarray(ewArray_fl[:, 0]).reshape((outputRows, outputCols))
        ns_Array = np.asarray(nsArray_fl[:, 0]).reshape((outputRows, outputCols))
        snr_array = np.asarray(snrArray_fl[:, 0]).reshape((outputRows, outputCols))

        return ew_array, ns_Array, snr_array

    @staticmethod
    def ingest_spatial_corr_params(params: Dict) -> List[Any]:
        window_size = params.get("window_size", [64, 64, 64, 64])
        step = params.get("step", 2 * [8])
        search_range = params.get("search_range", 2 * [10])
        grid = params.get("grid", True)
        return [window_size, step, search_range, grid]


class RawCorrelationEngine(BaseCorrelationEngine):
    def __init__(self,
                 correlator_name: str = None,
                 params=None,
                 debug=False):

        self.correlator_name = correlator_name

        if self.correlator_name is None:
            self.correlator_name = CORRELATION.FREQUENCY_CORRELATOR
        self.corr_params = params
        self.debug = debug
        self.corr_bands: List[str] = ["East/West", "North/South", "SNR"]

        self._get_corr_params()

    def _get_corr_params(self):
        if self.correlator_name == CORRELATION.FREQUENCY_CORRELATOR:
            if self.corr_params is None:
                self.corr_params = self.get_freq_params()
            else:
                self.corr_params = self.get_freq_params(window_size=self._ingest_params()[0],
                                                        step=self._ingest_params()[1],
                                                        mask_th=self._ingest_params()[2],
                                                        nb_iters=self._ingest_params()[3],
                                                        grid=self._ingest_params()[4])
        if self.correlator_name == CORRELATION.SPATIAL_CORRELATOR:
            if self.corr_params is None:
                self.corr_params = self.get_spatial_params()
            else:
                self.corr_params = self.get_spatial_params(window_size=self._ingest_params()[0],
                                                           step=self._ingest_params()[1],
                                                           search_range=self._ingest_params()[2],
                                                           grid=self._ingest_params()[3])

        if self.debug:
            logging.info(self.__dict__)
        return

    @staticmethod
    def get_spatial_params(window_size: List[int] = None, step: List[int] = None,
                           search_range: List[int] = None, grid: bool = False) -> RawSpatialCorr:

        return RawSpatialCorr(window_size, step, search_range, grid)

    @staticmethod
    def get_freq_params(window_size: List[int] = None, step: List[int] = None, mask_th: float = None,
                        resampling: bool = False, nb_iters: int = 4, grid: bool = True) -> RawFreqCorr:

        return RawFreqCorr(window_size, step, mask_th, resampling, nb_iters, grid)

    def _ingest_params(self):
        if self.correlator_name == CORRELATION.FREQUENCY_CORRELATOR:
            return RawFreqCorr.ingest_freq_corr_params(params=self.corr_params)
        if self.correlator_name == CORRELATION.SPATIAL_CORRELATOR:
            return RawSpatialCorr.ingest_spatial_corr_params(params=self.corr_params)

    def correlate(self):
        pass


class CorrelationEngine(RawCorrelationEngine):
    def __init__(self, correlator_name: str = None,
                 params=None,
                 debug=False):
        super().__init__(correlator_name,
                         params,
                         debug)
        pass


class RawCorrelation(BaseCorrelation):

    def __init__(self,
                 base_image_path: str,
                 target_image_path: str,
                 corr_config: Optional[Dict] = None,
                 base_band: Optional[int] = 1,
                 target_band: Optional[int] = 1,
                 output_corr_path: Optional[str] = None,
                 tile_size_mb: Optional[int] = CORRELATION.TILE_SIZE_MB,
                 visualize: Optional[bool] = False,
                 debug: Optional[bool] = True,
                 pixel_based_correlation: Optional[bool] = None):

        self.snr_output = None
        self.ns_output = None
        self.ew_output = None
        self.win_area_x = None
        self.win_area_y = None
        self.base_img_path = base_image_path
        self.target_img_path = target_image_path
        self.corr_engine = CorrelationEngine(correlator_name=corr_config.get("correlator_name", None),
                                             params=corr_config.get("correlator_params", None))

        self.tile_size_mb = tile_size_mb
        self.visualize = visualize
        self.base_band_nb = base_band
        self.target_band_nb = target_band
        self.output_corr_path = output_corr_path
        self.debug = debug
        self.pixel_based_correlation = pixel_based_correlation

    def _ingest(self):

        self.x_res: float
        self.y_res: float
        self.win_area_x: int
        self.win_area_y: int
        self.base_dims_pix: List[float]
        self.target_dims_pix: List[float]
        self.tot_col: int
        self.tot_row: int

        if self.output_corr_path is None:
            self.output_corr_path = self.make_output_path(os.path.dirname(self.base_img_path), self.base_img_path,
                                                          self.target_img_path, self.corr_engine.correlator_name,
                                                          self.corr_engine.corr_params.window_size[0],
                                                          self.corr_engine.corr_params.step[0])
        else:
            if os.path.isdir(self.output_corr_path):
                self.output_corr_path = self.make_output_path(self.output_corr_path, self.base_img_path,
                                                              self.target_img_path, self.corr_engine.correlator_name,
                                                              self.corr_engine.corr_params.window_size[0],
                                                              self.corr_engine.corr_params.step[0])

        if self.debug:
            logging.info("Correlation engine:{} , params:{}".format(self.corr_engine.correlator_name,
                                                                    self.corr_engine.corr_params.__dict__))

        self.base_info = cRasterInfo(self.base_img_path)
        self.target_info = cRasterInfo(self.target_img_path)
        # TODo remove the -1 form the list and use rasterio or bbox_pix bbox_map instead
        self.base_dims_pix = [-1, 0, int(self.base_info.raster_width) - 1, 0,
                              int(self.base_info.raster_height) - 1]
        self.target_dims_pix = [-1, 0, int(self.target_info.raster_width) - 1, 0,
                                int(self.target_info.raster_height) - 1]

        self.base_original_dims: List[float] = self.base_dims_pix
        self.margins = self.set_margins()
        logging.info(f'{self.__class__.__name__}:correlation margins:{self.margins}')
        if self.pixel_based_correlation is None:
            self.pixel_based_correlation = False
        return

    def set_margins(self) -> List[int]:
        if self.corr_engine.correlator_name == CORRELATION.FREQUENCY_CORRELATOR:
            return RawFreqCorr.set_margins(self.corr_engine.corr_params.resampling,
                                           self.corr_engine.corr_params.window_size)
        if self.corr_engine.correlator_name == CORRELATION.SPATIAL_CORRELATOR:
            return self.corr_engine.corr_params.search_range

    def check_same_projection_system(self):
        # TODO: move this function misc
        ##Check that the images have identical projection reference system
        self.flags = {"validMaps": False, "groundSpaceCorr": False, "continue": True}
        if self.pixel_based_correlation:
            logging.info(' USER: PIXEL-BASED CORRELATION ')
            self.flagList = self.updateFlagList(self.flags)
            return
        if self.base_info.valid_map_info and self.target_info.valid_map_info:
            self.flags["validMaps"] = True
            if self.base_info.epsg_code != self.target_info.epsg_code:
                # TODO: Check based on the projection not only epsg_code
                # TODO: add the possibility to reproject to the same projection system + same resolution
                logging.warning(
                    "=== Input images have different map projection (!= EPSG code), Correlation will be pixel based! ===")
                # warnings.warn(
                #     "=== Input images have different map projection (!= EPSG code), Correlation will be pixel based! ===",
                #     stacklevel=2)
                self.flags["groundSpaceCorr"] = False
            else:
                self.flags["groundSpaceCorr"] = True
        else:
            logging.warning("=== Input images are not geo-referenced. Correlation will be pixel based!,  ===")
            warnings.warn(
                "=== Input images are not geo-referenced. Correlation will be pixel based!,  ===",
                stacklevel=2)
            self.flags["groundSpaceCorr"] = False

        self.flagList = self.updateFlagList(self.flags)

    def check_same_ground_resolution(self):
        ## Check if the images have the same ground resolution
        # (up to 1/1000 of the resolution to avoid precision error)
        if all(self.flagList):
            same_res = misc.check_same_gsd(base_img_info=self.base_info, target_img_info=self.target_info)
            if same_res == False:
                self.flags["continue"] = False
                # TODO: add error output -> raise error
                logging.error("=== Images have the same GSD:{} ===".format(self.base_info.pixel_width))
                logging.error("=== ERROR: Input data must have the same resolution to be correlated ===")
                sys.exit("=== ERROR: Input data must have the same resolution to be correlated ===")
            else:
                if self.debug:
                    logging.warning("=== Images have the same GSD:{} ===".format(self.base_info.pixel_width))
        self.flagList = self.updateFlagList(self.flags)

    def _check_aligned_grids(self):
        ## Check that the imqges are on geographically aligned grids (depends on origin and resolution)
        ## verify if the difference between image origin is less than of resolution/1000

        if all(self.flagList):
            if misc.check_aligned_grids(base_img_info=self.base_info, target_img_info=self.target_info) == False:
                self.flags["overlap"] = False
                # TODO raise ana error
                ## Add the possibility to align the inpu grids
                error_msg = "=== ERROR: --- Images cannot be overlapped due to their origin and resolution - " \
                            "Origins difference great than 1/1000 of the pixel resolution ==="
                logging.error(error_msg)
                sys.exit(error_msg)
            else:
                self.flags["overlap"] = True
        self.flagList = self.updateFlagList(self.flags)

    def set_corr_map_resolution(self):
        ## Depending on the validity of the map information of the images, the pixel resolution is setup
        # it will be the GSD if map info is valid , otherwise it will be 1 and the correlation will be pixel based
        if all(self.flagList):
            # the correlation will be map based not pixel based
            self.y_res = np.abs(self.base_info.pixel_height)
            self.x_res = np.abs(self.base_info.pixel_width)
        else:
            # Correlation will be pixel based
            self.x_res = 1.0
            self.y_res = 1.0

    @staticmethod
    def _set_win_area(window_sizes: List[int], margins: List[int]):
        """

        Args:
            window_sizes:
            margins:

        Returns:

        """
        win_area_x = int(window_sizes[0] + 2 * margins[0])
        win_area_y = int(window_sizes[1] + 2 * margins[1])

        return win_area_x, win_area_y

    @staticmethod
    def _blank_array_func(nbVals: int, nbBands: Optional[int] = 3) -> Any:
        blank_arr = np.zeros((nbVals * nbBands))
        for i in range(nbVals):
            blank_arr[3 * i] = np.nan
            blank_arr[3 * i + 1] = np.nan
            blank_arr[3 * i + 2] = 0

        return blank_arr

    @staticmethod
    def updateFlagList(flagDic):
        return list(flagDic.values())

    def crop_to_same_size(self):
        # TODO: move this function to misc
        """
        Cropping the images to the same size:
        Two condition exist:
            1- if the map information are valid and identical: we define the overlapping area based on geo-referencing
            2- if map information invalid or different: we define the overlapping area based we define overlapping area
                based on images size (pixel wise)
                """
        if all(self.flagList):
            # If map information are valid and identical, define the overlapping are based on geo-referencing.
            # Backup of the original master subset dimensions in case of a non gridded correlation
            # IF NOT grid THEN img1OriginalDims = base_dims_pix
            offset = ((self.target_info.x_map_origin + self.target_dims_pix[1] * self.target_info.pixel_width) - (
                    self.base_info.x_map_origin + self.base_dims_pix[
                1] * self.base_info.pixel_width)) / self.base_info.pixel_width
            if offset > 0:
                self.base_dims_pix[1] = int(self.base_dims_pix[1] + round(offset))
            else:
                self.target_dims_pix[1] = int(self.target_dims_pix[1] - round(offset))

            offset = ((self.target_info.x_map_origin + self.target_dims_pix[2] * self.target_info.pixel_width) - (
                    self.base_info.x_map_origin + self.base_dims_pix[
                2] * self.base_info.pixel_width)) / self.base_info.pixel_width

            if offset < 0:
                self.base_dims_pix[2] = int(self.base_dims_pix[2] + round(offset))
            else:
                self.target_dims_pix[2] = int(self.target_dims_pix[2] - round(offset))

            offset = ((self.target_info.y_map_origin - self.target_dims_pix[3] * np.abs(
                self.target_info.pixel_height)) - (
                              self.base_info.y_map_origin - self.base_dims_pix[3] * np.abs(
                          self.base_info.pixel_height))) / np.abs(
                self.base_info.pixel_height)
            if offset < 0:
                self.base_dims_pix[3] = int(self.base_dims_pix[3] - round(offset))
            else:
                self.target_dims_pix[3] = int(self.target_dims_pix[3] + round(offset))

            offset = ((self.target_info.y_map_origin - self.target_dims_pix[4] * np.abs(
                self.target_info.pixel_height)) - (
                              self.base_info.y_map_origin - self.base_dims_pix[4] * np.abs(
                          self.base_info.pixel_height))) / np.abs(
                self.base_info.pixel_height)
            if offset > 0:
                self.base_dims_pix[4] = int(self.base_dims_pix[4] - round(offset))
            else:
                self.target_dims_pix[4] = int(self.target_dims_pix[4] + round(offset))

            if self.base_dims_pix[0] >= self.base_dims_pix[2] or self.base_dims_pix[3] >= self.base_dims_pix[4]:
                logging.error("=== ERROR: Images do not have a geographic overlap ===")
                sys.exit(
                    "=== ERROR: Images do not have a geographic overlap ===")

        else:
            # If map information invalid or different, define overlapping area based on images size (pixel-wise)

            if (self.base_dims_pix[2] - self.base_dims_pix[1]) > (self.target_dims_pix[2] - self.target_dims_pix[1]):
                self.base_dims_pix[2] = self.base_dims_pix[1] + (self.target_dims_pix[2] - self.target_dims_pix[1])
            else:
                self.target_dims_pix[2] = self.target_dims_pix[1] + (self.base_dims_pix[2] - self.base_dims_pix[1])
            if (self.base_dims_pix[4] - self.base_dims_pix[3]) > (self.target_dims_pix[4] - self.target_dims_pix[3]):
                self.base_dims_pix[4] = self.base_dims_pix[3] + (self.target_dims_pix[4] - self.target_dims_pix[3])
            else:
                self.target_dims_pix[4] = self.target_dims_pix[3] + (self.base_dims_pix[4] - self.base_dims_pix[3])

    def adjusting_cropped_images_according_2_grid_nongrid(self):
        # TODO: refactoring + split into 2 functions or class
        """
          Adjusting cropped images according to a gridded/non-gridded output
          Two cases:
              1- If the user selected the gridded option
              2- If the user selected non-gridded option
              """
        # If the user selected the gridded option
        if self.corr_engine.corr_params.grid:
            if all(self.flagList):
                if misc.decimal_mod(value=self.base_info.x_map_origin, param=self.base_info.pixel_width) != 0 or \
                        misc.decimal_mod(value=self.base_info.y_map_origin,
                                         param=np.abs(self.base_info.pixel_height)) != 0:
                    logging.error(
                        "=== ERROR: Images coordinates origins must be a multiple of the resolution for a gridded output' ===")
                    sys.exit(
                        "=== ERROR: Images coordinates origins must be a multiple of the resolution for a gridded output' ===")
                ## Chek if the geo-coordinate of the first correlated pixel is multiple integer of the resolution
                ## If not adjust the image boundaries
                geoOffsetX = (self.base_info.x_map_origin + (
                        self.base_dims_pix[1] + self.margins[0] + self.corr_engine.corr_params.window_size[
                    0] / 2) * self.base_info.pixel_width) % \
                             (self.corr_engine.corr_params.step[0] * self.base_info.pixel_width)

                # print(self.baseDims[1], self.corr.margins[0], self.corr.windowSizes[0] / 2, self.base_info.pixelWidth)

                if np.round(geoOffsetX / self.base_info.pixel_width) != 0:
                    self.base_dims_pix[1] = int(
                        self.base_dims_pix[1] + self.corr_engine.corr_params.step[0] - np.round(
                            geoOffsetX / self.base_info.pixel_width))
                    self.target_dims_pix[1] = int(
                        self.target_dims_pix[1] + self.corr_engine.corr_params.step[0] - np.round(
                            geoOffsetX / self.base_info.pixel_width))

                geoOffsetY = (self.base_info.y_map_origin - (
                        self.base_dims_pix[3] + self.margins[1] + self.corr_engine.corr_params.window_size[
                    1] / 2) * np.abs(
                    self.base_info.pixel_height)) % \
                             (self.corr_engine.corr_params.step[1] * np.abs(self.base_info.pixel_height))

                if np.round(geoOffsetY / np.abs(self.base_info.pixel_height)) != 0:
                    self.base_dims_pix[3] = int(
                        self.base_dims_pix[3] + np.round(geoOffsetY / np.abs(self.base_info.pixel_width)))
                    self.target_dims_pix[3] = int(
                        self.target_dims_pix[3] + np.round(geoOffsetY / np.abs(self.base_info.pixel_width)))

            ## Define the number of column and rows of the ouput correlation
            self.tot_col = int(np.floor(
                (self.base_dims_pix[2] - self.base_dims_pix[1] + 1 - self.corr_engine.corr_params.window_size[0] - 2
                 * self.margins[0]) / self.corr_engine.corr_params.step[0]) + 1)
            self.tot_row = int(np.floor(
                (self.base_dims_pix[4] - self.base_dims_pix[3] + 1 - self.corr_engine.corr_params.window_size[1] - 2 *
                 self.margins[
                     1]) /
                self.corr_engine.corr_params.step[1]) + 1)
            if self.debug:
                logging.info("tCols:{}, tRows:{}".format(self.tot_col, self.tot_row))


        else:
            # The non-gridded correlation will generate a correlation map whose first pixel corresponds
            # to the first master pixel

            # Define the total number of pixel of the correlation map in col and row
            self.tot_col = int(
                np.floor((self.base_original_dims[2] - self.base_original_dims[1]) / self.corr_engine.corr_params.step[
                    0]) + 1)
            self.tot_row = int(
                np.floor((self.base_original_dims[4] - self.base_original_dims[3]) / self.corr_engine.corr_params.step[
                    1]) + 1)
            if self.debug:
                logging.info("tCols:{}, tRows:{}".format(self.tot_col, self.tot_row))

            # Compute the "blank" border in col and row. This blank border corresponds to the area of the
            # correlation map where no correlation values could be computed due to the patch characteristic of the
            # correlator
            self.border_col_left: int = int(np.ceil(
                (self.base_dims_pix[1] - self.base_original_dims[1] + self.corr_engine.corr_params.window_size[0] / 2 +
                 self.margins[0]) / float(self.corr_engine.corr_params.step[0])))
            self.border_row_top: int = int(np.ceil(
                (self.base_dims_pix[3] - self.base_original_dims[3] + self.corr_engine.corr_params.window_size[1] / 2 +
                 self.margins[1]) / float(self.corr_engine.corr_params.step[1])))
            if self.debug:
                logging.info("borderColLeft:{}, borderRowTop:{}".format(self.border_col_left, self.border_row_top))
            # From the borders in col and row, compute the necessary cropping of the master and slave in row and col,
            # so the first patch retrived from the tile correponds to a step-wise position of the correlation grid origin
            offsetX = self.border_col_left * self.corr_engine.corr_params.step[0] - (
                    self.corr_engine.corr_params.window_size[0] / 2 + self.margins[0]) - (
                              self.base_dims_pix[1] - self.base_original_dims[1])
            offsetY = self.border_row_top * self.corr_engine.corr_params.step[1] - (
                    self.corr_engine.corr_params.window_size[1] / 2 + self.margins[1]) - (
                              self.base_dims_pix[3] - self.base_original_dims[3])
            self.base_dims_pix[1] = int(self.base_dims_pix[1] + offsetX)
            self.target_dims_pix[1] = int(self.target_dims_pix[1] + offsetX)
            self.base_dims_pix[3] = int(self.base_dims_pix[3] + offsetY)
            self.target_dims_pix[3] = int(self.target_dims_pix[3] + offsetY)
            # Define the number of actual correlation, i.e., the total number of points in row and
            # column, minus the "blank" correlation on the border
            self.nb_corr_col: int = int(np.floor(
                (self.base_dims_pix[2] - self.base_dims_pix[1] + 1 - self.corr_engine.corr_params.window_size[0] - 2 *
                 self.margins[
                     0]) /
                self.corr_engine.corr_params.step[0]) + 1)
            self.nb_corr_row: int = int(np.floor(
                (self.base_dims_pix[4] - self.base_dims_pix[3] + 1 - self.corr_engine.corr_params.window_size[1] - 2 *
                 self.margins[
                     1]) /
                self.corr_engine.corr_params.step[1]) + 1)

            if self.debug:
                logging.info("nbCorrCol:{}, nbCorrRow:{}".format(self.nb_corr_col, self.nb_corr_row))

            #  ;Define the blank border on the right side in column and bottom side in row
            self.border_col_right: int = int(self.tot_col - self.border_col_left - self.nb_corr_col)
            self.border_row_bottom: int = int(self.tot_row - self.border_row_top - self.nb_corr_row)
            if self.debug:
                logging.info(
                    "borderColRight:{}, borderRowBottom:{}".format(self.border_col_right, self.border_row_bottom))

            # Define a "blank" (i.e., invalid) correlation line
            self.output_row_blank = self._blank_array_func(nbVals=self.tot_col)
            ##Define blank column border left and right
            self.output_col_left_blank = self._blank_array_func(nbVals=self.border_col_left)

            self.output_col_right_blank = self._blank_array_func(nbVals=self.border_col_right)

    def tiling(self):
        # Get number of pixel in column and row of the file subset to tile
        self.nb_col_img: int = int(self.base_dims_pix[2] - self.base_dims_pix[1] + 1)
        self.nb_row_img: int = int(self.base_dims_pix[4] - self.base_dims_pix[3] + 1)
        if self.debug:
            logging.info("nbColImg: {} || nbRowImg: {}".format(self.nb_col_img, self.nb_row_img))

        # Define number max of lines per tile
        self.max_rows_roi: int = int(
            np.floor((self.tile_size_mb * 8 * 1024 * 1024) / (self.nb_col_img * CORRELATION.PIXEL_MEMORY_FOOTPRINT)))
        if self.debug:
            logging.info("maxRowsROI:{}".format(self.max_rows_roi))
        # Define number of correlation column and lines computed for one tile
        if self.max_rows_roi < self.nb_row_img:
            temp = self.max_rows_roi
            self.nb_corr_row_per_roi: int = int((temp - self.win_area_y) / self.corr_engine.corr_params.step[1] + 1)
        else:
            temp = self.nb_row_img
            self.nb_corr_row_per_roi = int((temp - self.win_area_y) / self.corr_engine.corr_params.step[1] + 1)

        self.nb_corr_col_per_roi: int = int(
            (self.nb_col_img - self.win_area_x) / self.corr_engine.corr_params.step[0] + 1)

        # TODO change ROI per tile
        self.nb_roi: int = int(
            (self.nb_row_img - self.win_area_y + self.corr_engine.corr_params.step[1]) / (
                    (self.nb_corr_row_per_roi - 1) * self.corr_engine.corr_params.step[1] + (
                    self.win_area_y - self.corr_engine.corr_params.step[1])))

        if self.nb_roi < 1:
            ## At least one tile even if the ROI is Larger than the image
            self.nb_roi = 1
        if self.debug:
            logging.info("nbROI: {} || nbCorrRowPerROI: {} || nbCorrColPerROI: {}".format(self.nb_roi,
                                                                                          self.nb_corr_row_per_roi,
                                                                                          self.nb_corr_col_per_roi))

        # Define the boundaries of all the tiles but the last one which will have a different size
        self.dims_base_tile = np.zeros((self.nb_roi, 5), dtype=np.int64)
        self.dims_target_tile = np.zeros((self.nb_roi, 5), dtype=np.int64)
        for i in range(self.nb_roi):
            val = int(
                self.base_dims_pix[3] + ((i + 1) * self.nb_corr_row_per_roi - 1) * self.corr_engine.corr_params.step[
                    1] + self.win_area_y - 1)
            self.dims_base_tile[i, :] = [-1,
                                         self.base_dims_pix[1],
                                         self.base_dims_pix[2],
                                         self.base_dims_pix[3] + i * self.nb_corr_row_per_roi *
                                         self.corr_engine.corr_params.step[1],
                                         val]

            self.dims_target_tile[i, :] = [-1,
                                           self.target_dims_pix[1],
                                           self.target_dims_pix[2],
                                           self.target_dims_pix[3] + i * self.nb_corr_row_per_roi *
                                           self.corr_engine.corr_params.step[1],
                                           int(self.target_dims_pix[3] + ((i + 1) * self.nb_corr_row_per_roi - 1) *
                                               self.corr_engine.corr_params.step[1] + self.win_area_y - 1)]

        # Define boundaries of the last tile and the number of correlation column and lines computed for the last tile
        self.nb_rows_left: int = int((self.base_dims_pix[4] - self.dims_base_tile[self.nb_roi - 1, 4] + 1) - 1)
        if self.debug:
            logging.info("nbRowsLeft:{}".format(self.nb_rows_left, "\n"))
        if (self.nb_rows_left >= self.corr_engine.corr_params.step[1]):
            self.nb_corr_row_last_roi: int = int(self.nb_rows_left / self.corr_engine.corr_params.step[1])

            self.dims_base_tile = np.vstack((self.dims_base_tile, np.array(
                [-1, self.base_dims_pix[1], self.base_dims_pix[2],
                 self.base_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi * self.corr_engine.corr_params.step[1],
                 int(self.base_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi * self.corr_engine.corr_params.step[
                     1] + self.win_area_y - 1 + (
                             self.nb_corr_row_last_roi - 1) * self.corr_engine.corr_params.step[1])])))

            self.dims_target_tile = np.vstack((self.dims_target_tile, np.array(
                [-1, self.target_dims_pix[1], self.target_dims_pix[2],
                 self.target_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi * self.corr_engine.corr_params.step[
                     1],
                 int(self.target_dims_pix[3] + self.nb_roi * self.nb_corr_row_per_roi *
                     self.corr_engine.corr_params.step[
                         1] + self.win_area_y - 1 + (
                             self.nb_corr_row_last_roi - 1) * self.corr_engine.corr_params.step[1])])))
            self.nb_roi = self.nb_roi + 1
        else:
            self.nb_corr_row_last_roi = self.nb_corr_row_per_roi

        return

    def write_blank_pixels(self):
        # In case of non-gridded correlation,write the top blank correlation lines
        # Define a "blank" (i.e., invalid) correlation line
        temp_add = np.empty((self.border_row_top, self.ew_output.shape[1]))
        temp_add[:] = np.nan
        self.ew_output = np.vstack((temp_add, self.ew_output))
        self.ns_output = np.vstack((temp_add, self.ns_output))
        self.snr_output = np.vstack((temp_add, self.snr_output))

        # In case of non-gridded correlation, write the bottom blank correlation lines
        if self.border_row_bottom != 0:
            temp_add = np.empty((self.border_row_bottom, self.ew_output.shape[1]))
            temp_add[:] = np.nan
            self.ew_output = np.vstack((self.ew_output, temp_add))
            self.ns_output = np.vstack((self.ns_output, temp_add))
            self.snr_output = np.vstack((self.snr_output, temp_add))

        if self.border_col_left != 0:
            ##Define blank column border left and right
            # outputColLeftBlank = BlankArray(nbVals=borderColLeft)
            temp_add = np.empty((self.ew_output.shape[0], self.border_row_bottom))
            temp_add[:] = np.nan
            self.ew_output = np.vstack((temp_add.T, self.ew_output.T)).T
            self.ns_output = np.vstack((temp_add.T, self.ns_output.T)).T
            self.snr_output = np.vstack((temp_add.T, self.snr_output.T)).T
        if self.border_col_right != 0:
            temp_add = np.empty((self.ew_output.shape[0], self.border_col_right))
            temp_add[:] = np.nan
            self.ew_output = np.vstack((self.ew_output.T, temp_add.T)).T
            self.ns_output = np.vstack((self.ns_output.T, temp_add.T)).T
            self.snr_output = np.vstack((self.snr_output.T, temp_add.T)).T

    def set_geo_referencing(self):
        # TODO change this function to a class method
        if all(self.flagList):
            if self.corr_engine.corr_params.grid:
                x_map_origin = self.base_info.x_map_origin + (
                        self.base_dims_pix[1] + self.margins[0] + self.corr_engine.corr_params.window_size[
                    0] / 2) * self.base_info.pixel_width
                y_map_origin = self.base_info.y_map_origin - (
                        self.base_dims_pix[3] + self.margins[1] + self.corr_engine.corr_params.window_size[
                    1] / 2) * np.abs(
                    self.base_info.pixel_height)
            else:
                x_map_origin = self.base_info.x_map_origin + self.base_dims_pix[1] * self.base_info.pixel_width
                y_map_origin = self.base_info.y_map_origin - self.base_dims_pix[3] * np.abs(self.base_info.pixel_height)

            geo_transform = [x_map_origin, self.base_info.pixel_width * self.corr_engine.corr_params.step[0], 0,
                             y_map_origin, 0,
                             -1 * self.base_info.pixel_height * self.corr_engine.corr_params.step[1]]
            if self.debug:
                logging.info("correlation geo. transformation :{}".format(geo_transform))
            return geo_transform, self.base_info.epsg_code
        else:
            logging.warning("=== Pixel Based correlation ===")
            return [0.0, 1.0, 0.0, 0.0, 0.0, -1.0], 4326

    def set_corr_debug(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(os.path.dirname(self.output_corr_path), 'correlation.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )

    @staticmethod
    def make_output_path(path, base_img_path, target_img_path, correlator_name, window_size, step):
        return os.path.join(path,
                            Path(base_img_path).stem + "_VS_" +
                            Path(target_img_path).stem + "_" +
                            correlator_name + "_wz_" +
                            str(window_size) + "_step_" +
                            str(step) + ".tif")


class FreqCorrelator(RawFreqCorr):
    pass


class SpatialCorrelator(RawSpatialCorr):
    pass
