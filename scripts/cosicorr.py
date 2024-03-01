#!/usr/bin/env python3
"""
Move the Script to a Bin Directory: For the cosicorr command to be recognized globally,
 move it to a directory that's in your system's PATH.
 A common choice is /usr/local/bin/.
 You may rename cosicorr.py to cosicorr for convenience.

>> sudo mv cosicorr.py /usr/local/bin/cosicorr
"""
import argparse

import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate


def ortho_func(args):
    print("Executing ortho function with input:", args.input, "and output:", args.output)


def transform_func(args):
    print("Executing transform function with input:", args.input, "and output:", args.output)


def correlate_func(args):
    print(f'Executing correlation module :{args}')
    corr_config = {}

    if args.method == C.CORR_METHODS.FREQUENCY_CORR.value:
        corr_config = {
            "correlator_name": C.CORR_METHODS.FREQUENCY_CORR.value,
            "correlator_params": {
                "window_size": args.window_size,
                "step": args.step,
                "grid": args.grid,
                "mask_th": args.mask_th,
                "nb_iters": args.nb_iters
            }
        }
    elif args.method == C.CORR_METHODS.SPATIAL_CORR.value:
        corr_config = {
            "correlator_name": C.CORR_METHODS.SPATIAL_CORR.value,
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


def ortho_subparser(subparsers):
    ortho_parser = subparsers.add_parser('ortho', help='Orthorectification')
    ortho_parser.add_argument('--input', help='Input file for ortho', required=True)
    ortho_parser.add_argument('--output', help='Output file for ortho', required=True)
    ortho_parser.set_defaults(func=ortho_func)


def transform_subparser(subparsers):
    transform_parser = subparsers.add_parser('transform', help='Transfromation')
    transform_parser.add_argument('--input', help='Input file for transform', required=True)
    transform_parser.add_argument('--output', help='Output file for transform', required=True)
    transform_parser.set_defaults(func=transform_func)


def correlate_subparser(subparsers):
    correlate_parser = subparsers.add_parser('correlate', help='Correlation')
    correlate_parser.add_argument("base_image", type=str, help="Path to the base image.")
    correlate_parser.add_argument("target_image", type=str, help="Path to the target image.")
    correlate_parser.add_argument("--base_band", type=int, default=1, help="Base image band.")
    correlate_parser.add_argument("--target_band", type=int, default=1, help="Target image band.")
    correlate_parser.add_argument("--output_path", type=str, default=C.SOFTWARE.WKDIR, help="Output correlation path.")
    correlate_parser.add_argument("--method", type=str,
                                  choices=[C.CORR_METHODS.FREQUENCY_CORR.value, C.CORR_METHODS.SPATIAL_CORR.value],
                                  default=C.CORR_METHODS.FREQUENCY_CORR.value,
                                  help="Correlation method to use.")
    correlate_parser.add_argument("--window_size", type=int, nargs=4, default=[64, 64, 64, 64],
                                  help="Window size. (Default [64])")
    correlate_parser.add_argument("--step", type=int, nargs=2, default=[8, 8], help="Step size. (Default [8,8])")
    correlate_parser.add_argument("--grid", action="store_true", help="Use grid.")
    correlate_parser.add_argument("--show", action="store_true", help="Show correlation. (Default False)")
    correlate_parser.add_argument("--pixel_based", action="store_true", help="Enable pixel-based correlation.")
    correlate_parser.add_argument("--vmin", type=float, default=-1,
                                  help="Minimum value for correlation plot. (Default -1)")
    correlate_parser.add_argument("--vmax", type=float, default=1,
                                  help="Maximum value for correlation plot.(Default 1)")

    # Specific arguments for frequency method
    freq_group = correlate_parser.add_argument_group("Frequency method arguments")
    freq_group.add_argument("--mask_th", type=float, default=0.95, help="Mask threshold (only for frequency method).")
    freq_group.add_argument("--nb_iters", type=int, default=4, help="Number of iterations (only for frequency method).")

    # Specific arguments for spatial method
    spatial_group = correlate_parser.add_argument_group("Spatial method arguments")
    spatial_group.add_argument("--search_range", type=int, nargs=2, help="Search range (only for spatial method).")

    correlate_parser.set_defaults(func=correlate_func)


def cosicorr():
    parser = argparse.ArgumentParser(prog='cosicorr3d', description='GeoCosiCorr3D CLI',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(title='modules', dest='module', metavar='<module>')

    ortho_subparser(subparsers)
    transform_subparser(subparsers)
    correlate_subparser(subparsers)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    cosicorr()
