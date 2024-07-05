#!/usr/bin/env python3
"""
Move the Script to a Bin Directory: For the cosicorr command to be recognized globally,
move it to a directory that's in your system's PATH.
A common choice is /usr/local/bin/.
You may rename cosicorr.py to cosicorr for convenience.
>> sudo mv cosicorr.py /usr/local/bin/cosicorr
"""
import argparse
import os
from pathlib import Path

import geoCosiCorr3D.geoCore.constants as C
import numpy as np
from geoCosiCorr3D.geoCosiCorr3dLogger import GeoCosiCorr3DLog
from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate


def parse_list(value):
    return [float(item) for item in value.split(',')]


def ortho_func(args):
    from geoCosiCorr3D.geoOrthoResampling.geoOrtho import orthorectify

    if args.model_name == C.SATELLITE_MODELS.RFM:
        ortho_params = {"method": {"method_type": args.model_name,
                                   "metadata": args.rfm_fn,
                                   "corr_model": args.corr_model,
                                   },
                        "GSD": args.gsd,
                        "resampling_method": args.resampling_method}

        if os.path.isdir(args.o_ortho):
            o_ortho_path = os.path.join(
                args.o_ortho,
                f"ORTHO_{Path(args.input_img).stem}_{ortho_params['resampling_method']}_RFM_{ortho_params['GSD']}_GSD.tif")

        else:
            o_ortho_path = args.o_ortho
        GeoCosiCorr3DLog('Orthorectification', os.path.dirname(o_ortho_path))

        orthorectify(args.input_img, o_ortho_path, ortho_params, None, args.dem, args.debug)

    if args.model_name == C.SATELLITE_MODELS.RSM:
        if args.model_name == C.SATELLITE_MODELS.RSM:
            ortho_params = {"method": {"method_type": args.model_name,
                                       "metadata": args.rsm_fn,
                                       "corr_model": args.corr_model,
                                       "sensor": args.sat_id,
                                       },
                            "GSD": args.gsd,
                            "resampling_method": args.resampling_method}

            if os.path.isdir(args.o_ortho):
                o_ortho_path = os.path.join(
                    args.o_ortho,
                    f"ORTHO_{Path(args.input_img).stem}_{ortho_params['resampling_method']}_RSM_{ortho_params['GSD']}_GSD.tif")

            else:
                o_ortho_path = args.o_ortho
            GeoCosiCorr3DLog('Orthorectification', os.path.dirname(o_ortho_path))

            orthorectify(args.input_img, o_ortho_path, ortho_params, None, args.dem, args.refine,
                         args.gcps, args.ref_img, args.debug,args.show)


def transform_func(args):
    print("Transform function for model:", args)

    if args.model_name == C.SATELLITE_MODELS.RFM:
        from geoCosiCorr3D.geoRFM.RFM import RFM
        model_data = RFM(args.rfm_fn, dem_fn=args.dem_fn, debug=True)
        if not args.inv:
            res = np.asarray(model_data.i2g(args.x, args.y)).T
            print(f'lons:{res[:, 0]}')
            print(f'lat:{res[:, 1]}')
            print(f'alt:{res[:, 2]}')
        else:
            res = np.asarray(model_data.g2i(args.x, args.y, )).T
            print(f'cols:{res[:, 0]}')
            print(f'lines:{res[:, 1]}')
    if args.model_name == C.SATELLITE_MODELS.RSM:
        raise NotImplementedError


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
    ortho_parser = subparsers.add_parser('ortho', help='Orthorectification process')
    ortho_parser.add_argument('input_img', type=str, help='Input file for ortho')
    ortho_parser.add_argument('--o_ortho', type=str, default=os.getcwd(),
                              help='Output path for ortho. Defaults to the current working directory.')
    ortho_parser.add_argument('--corr_model', type=str, default=None, help='Correction model path (None)')
    ortho_parser.add_argument('--dem', type=str, default=None, help='DEM path (None)')
    ortho_parser.add_argument('--gsd', type=float, default=None, help='Output file for ortho (None)')
    ortho_parser.add_argument('--resampling_method', type=str, default=C.Resampling_Methods.SINC,
                              choices=C.GEOCOSICORR3D_RESAMLING_METHODS, help='Resampling method (SINC)')

    ortho_parser.add_argument("--debug", action="store_true")
    ortho_parser.add_argument("--show", action="store_true")

    ortho_parser.add_argument("--refine", action="store_true",
                              help="Refine model, this require GCPs or reference imagery to collect GCPs")
    ortho_parser.add_argument('--ref_img', type=str, default=None, help='Reference Ortho image (None)')
    ortho_parser.add_argument('--gcps', type=str, default=None, help='GCPs file (None)')

    model_subparsers = ortho_parser.add_subparsers(title='model', dest='model_name', metavar='<model_name>',
                                                   required=True)

    rfm_parser = model_subparsers.add_parser(C.SATELLITE_MODELS.RFM, help="RFM model specific arguments")
    rfm_parser.add_argument('rfm_fn', type=str, help="RFM file name (.tiff or .TXT)")
    rfm_parser.set_defaults(func=ortho_func)

    rsm_parser = model_subparsers.add_parser(C.SATELLITE_MODELS.RSM, help="RSM model specific arguments")
    rsm_parser.add_argument('sat_id', type=str, choices=C.GEOCOSICORR3D_SENSORS_LIST, help="Sat-name")
    rsm_parser.add_argument('rsm_fn', type=str, help="Specifies the path to the .xml DMP file. Additional formats "
                                                     "are supported in GeoCosiCorr3D.pro.")
    rsm_parser.set_defaults(func=ortho_func)


def transform_subparser(subparsers):
    transform_parser = subparsers.add_parser('transform', help='Transformation')

    transform_parser.add_argument('x', type=parse_list, help="list: x=cols and if with invert flag: lon")
    transform_parser.add_argument('y', type=parse_list, help="list: y=lines and if with invert flag: lat")
    transform_parser.add_argument("--inv", action="store_true", help="Transform form ground to image space.")
    transform_parser.add_argument('--dem_fn', type=str, default=None, help="DEM file name (None)")

    model_subparsers = transform_parser.add_subparsers(title='model', dest='model_name', metavar='<model_name>',
                                                       required=True)

    rfm_parser = model_subparsers.add_parser(C.SATELLITE_MODELS.RFM, help="RFM model specific arguments")
    rfm_parser.add_argument('rfm_fn', type=str, help="RFM file name (.tiff or .TXT)")
    rfm_parser.set_defaults(func=transform_func)

    # RSM model specific parser setup
    rsm_parser = model_subparsers.add_parser(C.SATELLITE_MODELS.RSM, help="RSM model specific arguments")
    rsm_parser.add_argument('sat_id', type=str, help="Sat-name")
    rsm_parser.add_argument('rsm_fn', type=str, help="Specifies the path to the .xml DMP file. Additional formats "
                                                     "are supported in GeoCosiCorr3D.pro"
                            )
    rsm_parser.set_defaults(func=transform_func)


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
