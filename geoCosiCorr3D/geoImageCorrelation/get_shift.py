"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
import cv2
import numpy as np

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoCore.core_correlation import (FreqCorrelator, SpatialCorrelator)


def apply_clahe(image: np.ndarray, dynamic_range=255.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=dynamic_range * 0.04, tileGridSize=(8, 8))

    image = image - np.min(image)
    image = (image * dynamic_range) / np.max(image)
    adjusted = clahe.apply(image.astype(np.uint8))
    return adjusted


def compute_shift(ref_patch: C.Patch, target_patch: C.Patch, clahe=False,
                  corr_method=C.CORRELATION.FREQUENCY_CORRELATOR):
    # TODO change this function to take as input equal arrays and correlation params
    # TODO Move this function to the correlation package or to the image registration package
    ## what we can do also: this function will be a class function of GCP patch that call a higher level function geoImageCorrealtion package

    ## TODO add the PhaseCorr-CV, PhaseCorr-SK, OpticalFlow
    ew, ns, snr, orthoSubsetRes, dx, dy = np.nan, np.nan, 0, 0, np.nan, np.nan
    # patch_info = geoRT.cRasterInfo(patch_path)
    # TODO apply CLAHE to the patches

    corr_wz = 4 * [int(ref_patch.data.shape[0] / 2)]
    if clahe:
        target_array = apply_clahe(target_patch.data)
        base_array = apply_clahe(ref_patch.data)
    else:
        target_array = target_patch.data
        base_array = ref_patch.data

    if corr_method == C.CORRELATION.FREQUENCY_CORRELATOR:
        ew_array, ns_array, snr_array = FreqCorrelator.run_correlator(base_array=base_array,
                                                                      target_array=target_array,
                                                                      window_size=corr_wz,
                                                                      step=[1, 1],
                                                                      iterations=4,
                                                                      mask_th=0.9)
        dx, dy, snr = ew_array[0, 0], ns_array[0, 0], snr_array[0, 0]

    elif corr_method == C.CORRELATION.SPATIAL_CORRELATOR:
        ew_array, ns_array, snr_array = SpatialCorrelator.run_correlator(base_array=base_array,
                                                                         target_array=target_array,
                                                                         window_size=corr_wz,
                                                                         step=[1, 1],
                                                                         search_range=[10, 10])

        dx, dy, snr = ew_array[0, 0], ns_array[0, 0], snr_array[0, 0]
    else:
        raise ValueError

    if ref_patch.gsd is not None and target_patch.gsd is not None and ref_patch.gsd == target_patch.gsd:
        ew = dx * ref_patch.gsd
        ns = dy * ref_patch.gsd
    else:
        ew = dx
        ns = dy

    return dx, dy, ew, ns, snr
