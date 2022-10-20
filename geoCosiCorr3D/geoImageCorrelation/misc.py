import numpy as np
from typing import Optional


# TODO: add type notation for rasterInfo class
def check_same_gsd(base_img_info, target_img_info, precision: Optional[float] = 1 / 1000) -> bool:
    """
    Check if the input images  have the same ground resolution (up to 1/1000 of the resolution to avoid precision error)
    Args:
        base_img_info:
        target_img_info:
        precision:

    Returns:

    """

    if (np.abs((base_img_info.pixel_width / target_img_info.pixel_width) - 1) > precision) or (
            np.abs((base_img_info.pixel_height / target_img_info.pixel_height) - 1) > precision):
        return False
    else:

        return True


def check_aligned_grids(base_img_info, target_img_info) -> bool:
    """
     Check that the input images are on geographically aligned grids (depends on origin and resolution)
    Args:
        base_img_info:
        target_img_info:

    Returns:

    """

    if decimal_mod(value=base_img_info.x_map_origin - target_img_info.x_map_origin,
                   param=base_img_info.pixel_width,
                   precision=base_img_info.pixel_width / 1000) != 0 or \
            decimal_mod(value=base_img_info.y_map_origin - target_img_info.y_map_origin,
                        param=np.abs(base_img_info.pixel_height),
                        precision=np.abs(base_img_info.pixel_height) / 1000) != 0:
        return False
    else:
        return True


def decimal_mod(value: float, param: float, precision: Optional[float] = 1e-5):
    """

    Args:
        value:
        param:
        precision:

    Returns:

    """
    result = value % param

    if (np.abs(result) < precision) or (param - np.abs(result) < precision):
        result = 0
    return result

