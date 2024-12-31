# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022

from abc import ABC
from typing import Optional


class BaseRasterInfo(ABC):
    def __init__(self, input_raster_path: str, band_number: Optional[int] = 1):
        self.input_raster_path = input_raster_path
        self.band_number = band_number
        pass

    # @property
    # def get_raster(self):
    #     return rasterio.open(self.get_raster_path)

    @property
    def get_raster_path(self) -> str:
        return self.input_raster_path

    # @property
    # def get_raster_as_array(self) -> np.ndarray:
    #     return self.get_raster.read(self.band_number)
    #
    # @property
    # def get_raster_width(self) -> int:
    #     return self.get_raster.width
    #
    # @property
    # def get_raster_height(self) -> int:
    #     return self.get_raster.height
