#!/usr/bin/env python3
"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""
import argparse

from geoCosiCorr3D.geoCore.constants import CORR_METHODS, SOFTWARE
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate


def main():
    parser = argparse.ArgumentParser(description="Run image correlation.")
    parser.add_argument("--base_image", type=str, required=True, help="Path to the base image.")
    parser.add_argument("--target_image", type=str, required=True, help="Path to the target image.")
    parser.add_argument("--base_band", type=int, default=1, help="Base image band.")
    parser.add_argument("--target_band", type=int, default=1, help="Target image band.")
    parser.add_argument("--output_path", type=str, default=SOFTWARE.WKDIR, help="Output correlation path.")
    parser.add_argument("--method", type=str,
                        choices=[CORR_METHODS.FREQUENCY_CORR.value, CORR_METHODS.SPATIAL_CORR.value], required=True,
                        help="Correlation method to use.")
    parser.add_argument("--window_size", type=int, nargs=4, default=[64, 64, 64, 64],
                        help="Window size. (Default [64])")
    parser.add_argument("--step", type=int, nargs=2, default=[8, 8], help="Step size. (Default [8,8])")
    parser.add_argument("--grid", action="store_true", help="Use grid.")
    parser.add_argument("--show", action="store_true", help="Show correlation. (Default False)")
    parser.add_argument("--pixel_based", action="store_true", help="Enable pixel-based correlation.")
    parser.add_argument("--vmin", type=float, default=-1, help="Minimum value for correlation plot. (Default -1)")
    parser.add_argument("--vmax", type=float, default=1, help="Maximum value for correlation plot.(Default 1)")

    # Specific arguments for frequency method
    freq_group = parser.add_argument_group("Frequency method arguments")
    freq_group.add_argument("--mask_th", type=float, default=0.95, help="Mask threshold (only for frequency method).")
    freq_group.add_argument("--nb_iters", type=int, default=4, help="Number of iterations (only for frequency method).")

    # Specific arguments for spatial method
    spatial_group = parser.add_argument_group("Spatial method arguments")
    spatial_group.add_argument("--search_range", type=int, nargs=2, help="Search range (only for spatial method).")

    args = parser.parse_args()

    corr_config = {}

    if args.method == CORR_METHODS.FREQUENCY_CORR.value:
        corr_config = {
            "correlator_name": CORR_METHODS.FREQUENCY_CORR.value,
            "correlator_params": {
                "window_size": args.window_size,
                "step": args.step,
                "grid": args.grid,
                "mask_th": args.mask_th,
                "nb_iters": args.nb_iters
            }
        }
    elif args.method == CORR_METHODS.SPATIAL_CORR.value:
        corr_config = {
            "correlator_name": CORR_METHODS.SPATIAL_CORR.value,
            "correlator_params": {
                "window_size": args.window_size,
                "step": args.step,
                "grid": args.grid,
                "search_range": args.search_range
            }
        }

    Correlate(base_image_path=args.base_image,
              target_image_path=args.target_image,
              base_band=args.base_band,
              target_band=args.target_band,
              output_corr_path=args.output_path,
              corr_config=corr_config,
              corr_show=args.show,
              pixel_based_correlation=args.pixel_based,
              vmin=args.vmin,
              vmax=args.vmax
              )


if __name__ == "__main__":
    geoCosiCorr3DLog("image_correlation")
    main()
