"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
from abc import abstractmethod, ABC
from typing import Dict, Optional, Type, List, Any

import numpy as np
from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import cGetSatMapGrid



class BaseInverseOrtho(ABC):
    def __init__(self,
                 input_l1a_path: str,
                 output_ortho_path: str,
                 output_trans_path: Optional[str],
                 ortho_params: Dict,
                 dem_path: Optional[str],
                 debug: bool = True):
        # self.ortho_grid = None
        self.ortho_geo_transform: List[float] = []
        self.input_l1a_path = input_l1a_path
        self.output_ortho_path = output_ortho_path
        self.output_trans_path = output_trans_path
        self.debug = debug
        self.ortho_params = ortho_params
        self.dem_path = dem_path
        self.ortho_grid: cGetSatMapGrid = Any

    @abstractmethod
    def orthorectify(self):
        pass

    @abstractmethod
    def _get_correction_model(self):
        """

        Returns:

        """
        pass

    @abstractmethod
    def set_ortho_geo_transform(self) -> List[float]:
        pass

    @abstractmethod
    def compute_tiles(self):
        """
        Compute the required number of tiles.
        Returns:

        """
        pass

    @abstractmethod
    def write_ortho_per_tile(self):
        pass

    @abstractmethod
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

        pass

    @abstractmethod
    def _set_ortho_grid(self) -> cGetSatMapGrid:
        pass

    @abstractmethod
    def DEM_interpolation(self, demInfo, demDims, tileCurrent, eastArr, northArr, modelData):
        """


        Args:
            demInfo:
            demDims:
            tileCurrent:
            eastArr:
            northArr:
            modelData:

        Returns:

        """
        pass

    @abstractmethod
    def compute_transformation_matrix(self, need_loop, init_data):
        """

        Args:
            need_loop:
            init_data:

        Returns:

        """
        pass


class BaseOrthoGrid(ABC):
    oBotRightNS = None
    oBotRightEW = None
    oUpLeftNS = None
    oRes = None
    oUpLeftEW = None
    gridEPSG = None

    def __init__(self, sat_model: Type['SatModel'], grid_epsg: int = None, gsd: float = None, ):
        self.sat_model = sat_model
        # modelType,
        # modelCorr = np.zeros((3, 3)),
        # rDEM = None,
        newRes = None,
        self.grid_epsg = grid_epsg
        self.grid_gsd = gsd


# TODo change SatModel to to another py file
class SatModel(ABC):
    def __init__(self):
        self.model_file_path = None
        self.model_type = None  # [RSM ,RFM]
        self.sat_sensor = None
        self.corr_model = np.zeros((3, 3))
