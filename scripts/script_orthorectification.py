"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import logging
import os
import shutil
import uuid
from pathlib import Path

import click
import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.geoCore.geoCosiCorrBaseCfg.BaseReadConfig as rcfg
from geoCosiCorr3D.geoCosiCorr3D_CLI.geoImageCorrelation_cli.cli_utils import \
    validatePath
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoOptimization.gcpOptimization import cGCPOptimization
from geoCosiCorr3D.geoOrthoResampling.geoOrtho import RFMOrtho, RSMOrtho
from geoCosiCorr3D.geoTiePoints.Tp2GCPs import TpsToGcps as tp2gcp


def single_ortho(ortho_inputs, ortho_config):
    if ortho_inputs.optimize:
        from geoCosiCorr3D.geoTiePoints.wf_features import features
        match_file = features(img1=ortho_inputs.ref_ortho_path,
                              img2=ortho_inputs.raw_img_path,
                              tp_params=ortho_config.feature_points_params,
                              output_folder=ortho_inputs.workspace_folder,
                              show=True)

        gcp = tp2gcp(in_tp_file=match_file,
                        base_img_path=ortho_inputs.raw_img_path,
                        ref_img_path=ortho_inputs.ref_ortho_path,
                        dem_path=ortho_inputs.dem_path,
                        debug=True)
        gcp()
        opt = cGCPOptimization(gcp_file_path=gcp.output_gcp_path,
                               raw_img_path=ortho_inputs.raw_img_path,
                               ref_ortho_path=ortho_inputs.ref_ortho_path,
                               sat_model_params=ortho_config.sat_model_params,
                               dem_path=ortho_inputs.dem_path,
                               opt_params=ortho_config.opt_params,
                               opt_gcp_file_path=os.path.join(ortho_inputs.workspace_folder,
                                                              Path(gcp.output_gcp_path).stem + "_opt.pts"),
                               corr_config=ortho_config.opt_corr_config,
                               debug=True,
                               svg_patches=True)
        opt()
        logging.info(f'correction model file:{opt.corr_model_file}')
        ortho_config.ortho_params['method']['corr_model'] = opt.corr_model_file

    if ortho_inputs.metadata_type == C.SATELLITE_MODELS.RSM:
        ortho = RSMOrtho(input_l1a_path=ortho_inputs.raw_img_path,
                 ortho_params=ortho_config.ortho_params,
                 output_ortho_path=ortho_inputs.output_ortho_path,
                 output_trans_path=ortho_inputs.output_trans_path,
                 dem_path=ortho_inputs.dem_path)
        ortho()
    if ortho_inputs.metadata_type == C.SATELLITE_MODELS.RFM:
        rfm_ortho= RFMOrtho(input_l1a_path=ortho_inputs.raw_img_path,
                 ortho_params=ortho_config.ortho_params,
                 output_ortho_path=ortho_inputs.output_ortho_path,
                 output_trans_path=ortho_inputs.output_trans_path,
                 dem_path=ortho_inputs.ref_ortho_path,
                 )
        rfm_ortho()


    return


def main_geoOrtho(input_file, config_file):
    input_dict = rcfg.parse_inputs(input_file)
    ortho_inputs = rcfg.IngestInputs(input_dict)
    wk_dir = ortho_inputs.workspace_folder
    geoCosiCorr3DLog(ortho_inputs.log_name, wk_dir)
    logging.info(f'ortho inputs:{ortho_inputs.__dict__}')
    config = rcfg.ConfigReader(config_file=config_file).get_config

    shutil.copyfile(input_file, os.path.join(ortho_inputs.workspace_folder, os.path.basename(input_file)))
    shutil.copyfile(config_file, os.path.join(ortho_inputs.workspace_folder, os.path.basename(config_file)))
    if isinstance(ortho_inputs.raw_img_path, list):
        raw_img_path_list, metadata_file_path_list = ortho_inputs.raw_img_path, ortho_inputs.metadata_file_path
        index = 1
        for raw_img_path, metadata_file in zip(raw_img_path_list, metadata_file_path_list):
            ortho_inputs.workspace_folder = os.path.join(wk_dir, f'{uuid.uuid1()}_{Path(raw_img_path).stem}')
            Path(ortho_inputs.workspace_folder).mkdir(parents=True, exist_ok=True)
            logging.info(f'output dir:{ortho_inputs.workspace_folder}')

            logging.info(
                f'[{index}]/[{len(raw_img_path_list)}]:{index / len(raw_img_path_list) * 100}%---> ORTHO:{raw_img_path}')
            ortho_inputs.raw_img_path = raw_img_path
            ortho_inputs.metadata_file_path = metadata_file

            ortho_inputs.output_ortho_path = os.path.join(ortho_inputs.workspace_folder,
                                                          f'{Path(raw_img_path).stem}_ORTHO.tif')
            ortho_inputs.output_trans_path = os.path.join(ortho_inputs.workspace_folder,
                                                          f'{Path(raw_img_path).stem}_TRX.tif')
            logging.info(f'ortho config:{config}')
            ortho_config = rcfg.IngestgeoOrthoWfConfig(ortho_inputs, config)

            logging.info(f'ortho_params:{ortho_config.ortho_params}')
            logging.info(f'sat_model_params:{ortho_config.sat_model_params}')
            logging.info(f'opt_params:{ortho_config.opt_params}')
            logging.info(f'opt_corr_config:{ortho_config.opt_corr_config}')
            logging.info(f'feature_points_params:{ortho_config.feature_points_params}')
            try:
                single_ortho(ortho_inputs, ortho_config)
            except:
                logging.error(f'ERROR: Incomplete ortho for {raw_img_path}')

                pass
            index += 1

    else:
        logging.info(f'ortho config:{config}')
        ortho_config = rcfg.IngestgeoOrthoWfConfig(ortho_inputs, config)

        logging.info(f'ortho_params:{ortho_config.ortho_params}')
        logging.info(f'sat_model_params:{ortho_config.sat_model_params}')
        logging.info(f'opt_params:{ortho_config.opt_params}')
        logging.info(f'opt_corr_config:{ortho_config.opt_corr_config}')
        logging.info(f'feature_points_params:{ortho_config.feature_points_params}')
        single_ortho(ortho_inputs, ortho_config)
    return


@click.group(context_settings=dict(help_option_names=["-help", "-h"]))
def cli():
    pass


@cli.command()
@click.argument('job_input_file', type=click.File('r'), callback=validatePath)
@click.argument('job_config_file', type=click.File('r'), callback=validatePath)
def main_geoOrtho_cli(job_input_file, job_config_file):
    main_geoOrtho(job_input_file, job_config_file)
    return


if __name__ == '__main__':
    main_geoOrtho_cli()
