from typing import List, Tuple, Union

import numpy as np

from geoCosiCorr3D.geoCore.base.base_transform import BaseTransformer
import geoCosiCorr3D.geoCore.constants as C




class IdentityTransformer(BaseTransformer):

    def __init__(self):
        super().__init__(C.TransformationMethods.IDENTITY)
        self.geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def i2g(self,
            x: Union[List, np.ndarray, float],
            y: Union[List, np.ndarray, float],
            **kwargs) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        return x, y

    def g2i(self,
            lon: Union[List, np.ndarray, float],
            lat: Union[List, np.ndarray, float],
            **kwargs) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        return lon, lat

    @property
    def name(self):
        return super().name.upper()


class RsmTransformer(BaseTransformer):
    def __init__(self):
        super().__init__(C.TransformationMethods.RSM)


class RfmTransformer(BaseTransformer):
    def __init__(self):
        super().__init__(C.TransformationMethods.RFM)

    def i2g(self,
            x: Union[List, np.ndarray, float],
            y: Union[List, np.ndarray, float],
            **kwargs) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        return x, y

    def g2i(self,
            lon: Union[List, np.ndarray, float],
            lat: Union[List, np.ndarray, float],
            **kwargs) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        return lon, lat
