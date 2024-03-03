from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np


class TransformationError(Exception):
    """Custom exception for errors during transformation."""
    pass


class BaseTransformer(ABC):
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def i2g(self,
                      x: Union[np.ndarray, List, float],
                      y: Union[np.ndarray, List, float],
                      hgt: Optional[Union[List, float]] = None
                      ) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        pass

    @abstractmethod
    def g2i(self,
                      lon: Union[np.ndarray, List, float],
                      lat: Union[np.ndarray, List, float],
                      hgt: Optional[Union[List, float]] = None
                      ) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
        pass

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
