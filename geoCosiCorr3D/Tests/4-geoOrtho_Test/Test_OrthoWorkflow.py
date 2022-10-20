"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

import logging
import os
from pathlib import Path
import geoCosiCorr3D.geoCore.geoCosiCorrBaseCfg.BaseReadConfig as rcfg
import geoCosiCorr3D.geoCore.constants as geoCosiCorr3D_db
from geoCosiCorr3D.geoOrthoResampling.geoOrtho import RFMOrtho, RSMOrtho
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.geoTiePoints.Tp2GCPs import TPsTOGCPS
from geoCosiCorr3D.geoOptimization.gcpOptimization_v2 import cGCPOptimization


def main(input_file, config_file):
    input_dict = rcfg.parse_inputs(input_file)
    input = rcfg.IngestInputs(input_dict)
    log = geoCosiCorr3DLog(input.log_name, input.workspace_folder)
    logging.info(f'ortho inputs:{input.__dict__}')
    config = rcfg.ConfigReader(config_file=config_file).get_config
    logging.info(config)
    ortho_config = rcfg.IngestgeoOrthoWfConfig(input, config)

    logging.info(f'ortho_params:{ortho_config.ortho_params}')
    logging.info(f'sat_model_params:{ortho_config.sat_model_params}')
    logging.info(f'opt_params:{ortho_config.opt_params}')
    logging.info(f'opt_corr_config:{ortho_config.opt_corr_config}')
    logging.info(f'feature_points_params:{ortho_config.feature_points_params}')

    if input.optimize == False:
        if input.metadata_type == geoCosiCorr3D_db.SATELLITE_MODELS.RSM:
            RSMOrtho(input_l1a_path=input.raw_img_path,
                     ortho_params=ortho_config.ortho_params,
                     output_ortho_path=input.output_ortho_path,
                     output_trans_path=input.output_trans_path,
                     dem_path=input.dem_path)
        if input.metadata_type == geoCosiCorr3D_db.SATELLITE_MODELS.RFM:
            RFMOrtho(input_l1a_path=input.raw_img_path,
                     ortho_params=ortho_config.ortho_params,
                     output_ortho_path=input.output_ortho_path,
                     output_trans_path=input.output_trans_path,
                     dem_path=input.ref_ortho_path,
                     )

    if input.optimize:
        from geoCosiCorr3D.geoWorkflow.wf_features import wf_features

        match_file = wf_features(img1=input.ref_ortho_path, img2=input.raw_img_path,
                                 tp_params=ortho_config.feature_points_params,
                                 output_folder=input.workspace_folder)

        gcp = TPsTOGCPS(in_tp_file=match_file,
                        base_img_path=input.raw_img_path,
                        ref_img_path=input.ref_ortho_path,
                        dem_path=input.dem_path,
                        debug=True)

        opt = cGCPOptimization(gcp_file_path=gcp.output_gcp_path,
                               raw_img_path=input.raw_img_path,
                               ref_ortho_path=input.ref_ortho_path,
                               sat_model_params=ortho_config.sat_model_params,
                               dem_path=input.dem_path,
                               opt_params=ortho_config.opt_params,
                               opt_gcp_file_path=os.path.join(input.workspace_folder,
                                                              Path(gcp.output_gcp_path).stem + "_opt.pts"),
                               corr_config=ortho_config.opt_corr_config,
                               debug=True,
                               svg_patches=False)
        logging.info(f'correction model file:{opt.corr_model_file}')
        ortho_config.ortho_params['method']['corr_model'] = opt.corr_model_file
        if input.metadata_type == geoCosiCorr3D_db.SATELLITE_MODELS.RSM:
            RSMOrtho(input_l1a_path=input.raw_img_path,
                     ortho_params=ortho_config.ortho_params,
                     output_ortho_path=input.output_ortho_path,
                     output_trans_path=input.output_trans_path,
                     dem_path=input.dem_path,

                     )
        if input.metadata_type == geoCosiCorr3D_db.SATELLITE_MODELS.RFM:
            RFMOrtho(input_l1a_path=input.raw_img_path,
                     ortho_params=ortho_config.ortho_params,
                     output_ortho_path=input.output_ortho_path,
                     output_trans_path=None,
                     dem_path=None,
                     )

    return


if __name__ == '__main__':
    # Parse inputs
    input_file = "/home/cosicorr/0-WorkSpace/3-PycharmProjects/geoCosiCorr3D/geoCosiCorr3D/Tests/4-geoOrtho_Test/geo_ortho_input.json"
    config_file = '/home/cosicorr/0-WorkSpace/3-PycharmProjects/geoCosiCorr3D/geoCosiCorr3D/Tests/4-geoOrtho_Test/geo_ortho_config.yaml'
    main(input_file, config_file=config_file)
