# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from geoCosiCorr3D.geoCore.constants import CORRELATION
from geoCosiCorr3D.geoCore.geoCosiCorrBaseCfg.BaseReadConfig import \
    ConfigReader


class BaseFreqCorr(ABC):

    @staticmethod
    def ingest_freq_corr_params(params: Dict) -> List:
        """

        Args:
            params:

        Returns:

        """

    pass

    @staticmethod
    def set_margins(resampling: bool, window_size: List[int]) -> List[int]:
        """

        Args:
            resampling:
            window_size:

        Returns:

        """
        pass

    # TODO: change to static method or adapt to class method
    @classmethod
    def run_correlator(cls, base_array: np.ndarray, target_array: np.ndarray, window_size: List[int],
                       step: List[int],
                       iterations: int, mask_th: float):
        pass


class BaseSpatialCorr(ABC):

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

        pass

    @classmethod
    def run_correlator(cls, base_array: np.ndarray,
                       target_array: np.ndarray,
                       window_size: List[int],
                       step: List[int],
                       search_range: List[int]):
        pass

    @staticmethod
    def ingest_spatial_corr_params(params: Dict) -> List:
        pass


class BaseCorrelationEngine(ABC):

    @abstractmethod
    def _get_corr_params(self):
        pass

    @abstractmethod
    def correlate(self):
        pass

    @abstractmethod
    def _ingest_params(self):
        pass


class BaseCorrelation(ABC):

    @abstractmethod
    def _ingest(self):
        pass

    @abstractmethod
    def set_margins(self) -> List[int]:
        pass

    @abstractmethod
    def check_same_projection_system(self):
        # TODO: move this function misc
        ##Check that the images have identical projection reference system
        pass

    @abstractmethod
    def check_same_ground_resolution(self):
        ## Check if the images have the same ground resolution
        # (up to 1/1000 of the resolution to avoid precision error)
        pass

    @abstractmethod
    def _check_aligned_grids(self):
        ## Check that the imqges are on geographically aligned grids (depends on origin and resolution)
        ## verify if the difference between image origin is less than of resolution/1000

        pass

    @abstractmethod
    def set_corr_map_resolution(self):
        pass

    @abstractmethod
    def crop_to_same_size(self):
        # TODO: move this function to misc
        """
        Cropping the images to the same size:
        Two condition exist:
            1- if the map information are valid and identical: we define the overlapping area based on geo-referencing
            2- if map information invalid or different: we define the overlapping area based we define overlapping area
                based on images size (pixel wise)
                """
        pass

    @abstractmethod
    def adjusting_cropped_images_according_2_grid_nongrid(self):
        pass

    @abstractmethod
    def write_blank_pixels(self):
        pass

    @abstractmethod
    def set_geo_referencing(self):
        pass

    @abstractmethod
    def set_corr_debug(self):
        pass

    @abstractmethod
    def run_correlation(self):
        pass

    @abstractmethod
    def run_corr_engine(self, base_array: np.ndarray, target_array: np.ndarray):
        pass

    @abstractmethod
    def corr_plot(self):
        pass

    @property
    def get_corr_config(self):
        return ConfigReader(CORRELATION.CONFIG_FILE)
