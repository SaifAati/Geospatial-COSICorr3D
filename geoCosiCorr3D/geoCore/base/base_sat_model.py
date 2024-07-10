from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from geoCosiCorr3D.geoCore.geoDEM import DEM


class BaseSatModel(ABC):
    def __init__(self, name: str,
                 dem_fn: Optional[str] = None,
                 corr_model: Optional[np.ndarray] = None, **kwargs):
        self._name = name
        self.corr_model = corr_model if not not None else np.zeros((3, 3))
        self.dem_fn = dem_fn

    @abstractmethod
    def i2g(self,
            col: Union[np.ndarray, List, float],
            lin: Union[np.ndarray, List, float],
            hgt: Optional[Union[List, float]] = None, **kwargs,
            ) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        pass

    @abstractmethod
    def g2i(self,
            lon: Union[np.ndarray, List, float],
            lat: Union[np.ndarray, List, float],
            hgt: Optional[Union[List, np.ndarray, float]] = None,
            **kwargs
            ) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        pass

    @abstractmethod
    def get_geotransform(self):
        pass

    @abstractmethod
    def get_footprint(self):
        pass

    @abstractmethod
    def get_gsd(self):
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


class SatModel(BaseSatModel):
    def __init__(self, name: str,
                 dem_fn: Optional[str] = None,
                 corr_model: Optional[np.ndarray] = None, **kwargs):
        super().__init__(name, dem_fn, corr_model, **kwargs)
        self._name = name
        self.corr_model = corr_model if not not None else np.zeros((3, 3))
        self.dem_fn = dem_fn
        self.set_elevation()

    def set_elevation(self):
        if self.dem_fn is None:
            self.dem = None
        else:
            self.dem = DEM(self.dem_fn)
