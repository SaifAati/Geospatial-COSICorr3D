from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class ImageSeriesAnalyzer(ABC):
    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def setup(self):
        pass

    @staticmethod
    def build_flatten_data_cube(data_cube: np.ndarray) -> np.ndarray:
        """
        Flattens a 3D data cube into a 2D array where each row corresponds to a flattened 2D slice of the cube.
        Args:
            data_cube: 3D array with shape (bands, height, width)

        Returns:
            2D array with shape (height * width, bands)

        """
        data_cube_shape = data_cube.shape
        return data_cube.reshape(data_cube_shape[0], -1).T

    @staticmethod
    def build_mask_any_nan(input_array: np.ndarray, mask_value: Optional[float] = None):
        """
        Builds a mask for the input array, masking any NaN values and optionally values outside a specified range.
        Args:
            input_array: input array to be masked.
            mask_value: The value range for masking. If specified, values outside the range [-mask_value, mask_value] will be masked.

        Returns:
            A mask array where valid values are marked as True and invalid values as False.

        """
        input_array = np.ma.masked_invalid(input_array)
        if mask_value is not None:
            input_array = np.ma.masked_outside(input_array, -mask_value, mask_value)
        return np.prod(~input_array.mask, axis=1)

    def mask_raster(self, mask, input_array):
        """
        Masks the input array based on the given mask.

        Parameters:
        mask (numpy.ndarray): The mask array.
        input_array (numpy.ndarray): The input array to be masked.

        Returns:
        numpy.ndarray: The masked array.
        """
        mask_fl = mask.flatten()
        index_list = np.where(mask_fl > 0)[0]
        shape = input_array.shape

        array_masked = input_array[index_list, :]
        if self.debug:
            print(
                f"Valid values: {len(index_list)}:{mask_fl.shape[0]} ===> invalid:{mask_fl.shape[0] - len(index_list)}")
            print(f"input_array dim:{shape}")
            print(f"masked array dim:{array_masked.shape}")

        return array_masked
