"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
import dataclasses
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas

import geoCosiCorr3D.geoCore.constants as C
import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarn
import geoCosiCorr3D.georoutines.file_cmd_routines as fileRT
import geoCosiCorr3D.georoutines.geo_utils as geoRT
from geoCosiCorr3D.geoCore.core_RSM import RSM
from geoCosiCorr3D.geoImageCorrelation.get_shift import compute_shift
from geoCosiCorr3D.geoOrthoResampling.GCPPatches import (OrthoPatch, RawInverseOrthoPatch)
from geoCosiCorr3D.utils.misc import write_patches

geoWarn.wrIgnoreNotGeoreferencedWarning()

MULTIPROCS_2LOOP_DEBUG = False


@dataclasses.dataclass
class CorrelationOptParams:
    METHOD: str = C.CORR_METHODS.FREQUENCY_CORR.value
    WZ: List[int] = dataclasses.field(default_factory=lambda: 4 * [64])


@dataclasses.dataclass
class OptParams:
    MAX_ITER: int = 3
    SNR_TH: float = 0.9
    SNR_WEIGHT: bool = True
    MEAN_ERR_TH: float = 1 / 20
    RESAMPLING_METHOD: str = C.ResamplingMethods.SINC.value


# TODO the goal here is to distinguish between the GCP/TP optimization and model refinement

class TpOptimizer:
    pass


class GcpOptimizer:
    def __init__(self,
                 gcp_fn: str,
                 ref_img_fn: str,
                 target_img_fn: str,
                 dem_fn: str,
                 sat_model_params: C.SatModelParams,
                 sat_corr_model: np.ndarray = np.zeros((3, 3)),
                 opt_gcp_fn: Optional[str] = None,
                 opt_params: OptParams = OptParams(),
                 corr_params: CorrelationOptParams = CorrelationOptParams(),
                 debug=False,
                 w_patches=False,
                 svg_patches=False):
        self.gcp_fn = gcp_fn
        self.ref_img_fn = ref_img_fn
        self.target_img_fn = target_img_fn
        self.dem_fn = dem_fn

        self.opt_gcp_fn = opt_gcp_fn
        self.corr_config = corr_params
        self.debug = debug
        self.svg_patches = svg_patches
        self.w_patches = w_patches

        self.opt_params = opt_params
        self.corr_params = corr_params
        self.sat_model_params = sat_model_params

        self.sat_corr_model = sat_corr_model

    def __call__(self):
        self._ingest()
        self._setup()

        logging.info(f'{self.__class__.__name__}: GCP patch generation ...')
        self.gcp_df_corr = self.gcp_df.copy()
        self.sat_corr_model = self._compute_sat_corr_model(self.gcp_df_corr)
        # TODO I need to keep track of all correction model in a json file with the corresponding Loop number

        ref_patches: List[C.Patch] = []
        for i in range(self.nb_gcps):
            ref_patch: OrthoPatch = OrthoPatch(ortho_info=self.ref_ortho_info,
                                               patch_sz=self.patch_sz,
                                               gcp_df=self.gcp_df_corr.iloc[i])
            ref_patches.append(ref_patch())

        # Generating raw_patches/target_patches
        target_l3b_patches: List[C.Patch] = []
        for i in range(self.nb_gcps):
            raw_patch: RawInverseOrthoPatch = RawInverseOrthoPatch(raw_img_info=self.target_img_info,
                                                                   sat_model=self.sat_model,
                                                                   corr_model=self.sat_corr_model,
                                                                   patch_sz=self.patch_sz,
                                                                   patch_epsg=self.ref_ortho_info.epsg_code,
                                                                   patch_res=np.abs(self.ref_ortho_info.pixel_width),
                                                                   patch_map_extent=ref_patches[i].ground_extent,
                                                                   dem_path=None,  # self.dem_fn,
                                                                   model_type=self.sat_model_params.SAT_MODEL,
                                                                   resampling_method=self.opt_params.RESAMPLING_METHOD,
                                                                   debug=self.debug,
                                                                   gcp_df=self.gcp_df_corr.iloc[i])

            target_l3b_patches.append(raw_patch())

        correspondences = self._get_correspondences(ref_patches, target_l3b_patches)
        if self.w_patches:
            write_patches(correspondences, self.patches_folder, prefix='loop1', png=True)

        # TODO make this loop to hande multithreading/cpu processing

        for ref_patch_, target_patch_ in correspondences:
            shift = compute_shift(ref_patch_, target_patch_, self.corr_params.METHOD)
            print(shift)

        # print(target_l3b_patches)
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots()
        # plt.imshow(target_l3b_patches[0].array, 'gray')
        # plt.show()
        # import sys
        # sys.exit()

    def _get_correspondences(self, ref_patches, target_patches) -> List[Tuple[C.Patch, C.Patch]]:
        correspondences = []
        for rp in ref_patches:
            patch_id = rp.id
            for tp in target_patches:
                if tp.id == patch_id:
                    correspondences.append((rp, tp))
                    break
        return correspondences

    def _ingest(self):
        self.gcp_df = pandas.read_csv(self.gcp_fn)
        logging.info(f'{self.__class__.__name__}: input GCPs:{self.gcp_df.shape}')
        print(self.gcp_df)
        self.nb_gcps = self.gcp_df.shape[0]

        logging.info(f'{self.__class__.__name__}: loading REF IMG: {self.ref_img_fn}')
        self.ref_ortho_info = geoRT.cRasterInfo(self.ref_img_fn)

        logging.info(f'{self.__class__.__name__}: loading Target IMG:{self.target_img_fn}')
        self.target_img_info = geoRT.cRasterInfo(self.target_img_fn)

        self.dem_info = None if self.dem_fn is None else geoRT.cRasterInfo(self.dem_fn)
        log_msg = 'No DEM provided' if self.dem_fn is None else f'{self.__class__.__name__}: loading DEM'
        logging.info(log_msg)

    def _set_sat_model(self):
        sat_model_name = self.sat_model_params.SAT_MODEL
        if sat_model_name not in C.GEOCOSICORR3D_SATELLITE_MODELS:
            msg = f'Sat_model: {sat_model_name} is not valid and not supported by {C.SOFTWARE.SOFTWARE_NAME} v{C.SOFTWARE.VERSION}'
            logging.error(msg)
            sys.exit(msg)
        else:
            logging.info(f'{self.__class__.__name__}:Sat_model:{sat_model_name}')

        if os.path.exists(self.sat_model_params.METADATA) == False:
            msg = f'Sat_model_metadata:{self.sat_model_params.METADATA} does not exist !'
            logging.error(msg)
            sys.exit(msg)

        if sat_model_name == C.SATELLITE_MODELS.RSM:
            if self.sat_model_params.SENSOR not in C.GEOCOSICORR3D_SENSORS_LIST:
                msg = f'sensor: {self.sat_model_params.SENSOR} is not valid and not supported by {C.SOFTWARE.SOFTWARE_NAME} v{C.SOFTWARE.VERSION}'
                logging.error(msg)
                sys.exit(msg)
            else:
                logging.info(f'{self.__class__.__name__}: {sat_model_name} sensor: {self.sat_model_params.SENSOR}')
                self.sat_model = RSM.build_RSM(metadata_file=self.sat_model_params.METADATA,
                                               sensor_name=self.sat_model_params.SENSOR,
                                               debug=self.debug)
                logging.info(f'{self.__class__.__name__}:Sat_model:{sat_model_name}')

        if sat_model_name == C.SATELLITE_MODELS.RFM:
            # TODO implement RFM
            raise NotImplementedError

    def _compute_sat_corr_model(self, gcp_df) -> np.ndarray:
        if self.sat_model_params.SAT_MODEL == C.SATELLITE_MODELS.RSM:
            from geoCosiCorr3D.geoOptimization.model_refinement.sat_model_refinement import RsmRefinement
            rsm_refinement = RsmRefinement(sat_model=self.sat_model, gcps=gcp_df, debug=self.debug)
            rsm_refinement.refine()
            return rsm_refinement.corr_model
        elif self.sat_model_params.SAT_MODEL == C.SATELLITE_MODELS.RFM:
            raise NotImplementedError
        else:
            raise ValueError(f'{self.__class__.__name__}:Sat_model:{self.sat_model_params.SAT_MODEL} is not supported')

    def _setup(self):

        logging.info(f'{self.__class__.__name__}: Sat params: {self.sat_model_params}')
        self._set_sat_model()
        logging.info(f'{self.__class__.__name__}: Opt params: {self.opt_params}')
        logging.info(f'{self.__class__.__name__}: Corr params: {self.corr_params}')
        self.set_patch_sz()

        self.opt_gcp_fn = os.path.join(
            self.opt_gcp_fn if self.opt_gcp_fn is not None and os.path.isdir(
                self.opt_gcp_fn) else os.path.dirname(self.gcp_fn),
            Path(self.gcp_fn).stem + "_opt.pts"
        )
        logging.info(f'{self.__class__.__name__}: Output OPT GCPs: {self.opt_gcp_fn}')

        if self.svg_patches:
            self.patches_folder = fileRT.CreateDirectory(directoryPath=os.path.dirname(self.opt_gcp_fn),
                                                         folderName=f"{self.sat_model_params.SAT_MODEL}_gcp_patches",
                                                         cal="y")
        else:
            self.patches_folder = None

    def set_patch_sz(self):
        corr_wz = self.corr_params.WZ
        margins = [int(corr_wz[0] // 2), int(corr_wz[1] // 2)]
        logging.info(f'{self.__class__.__name__}:Margins:{margins}')
        wz_col = corr_wz[0] + 2 * margins[0]
        wz_lin = corr_wz[0] + 2 * margins[1]
        self.patch_sz = [wz_col, wz_lin]
        logging.info(f"{self.__class__.__name__}:patch SZ:{self.patch_sz}")
        return


class IterativeGcpOptimizer:
    pass


if __name__ == '__main__':
    folder = '/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/NEW'
    gcp_fn = '/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/NEW/input_GCPs.csv'
    dmp_file = os.path.join(folder, 'SP-2.DIM')
    sat_model_params = C.SatModelParams(C.SATELLITE_MODELS.RSM, dmp_file, C.SENSOR.SPOT2)
    import logging

    # Set up logging to file
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # Set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Now you can use logging.info(), logging.warning(), etc.
    logging.info('This is an info message')
    fcp_opt = GcpOptimizer(gcp_fn=gcp_fn,
                           ref_img_fn='/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/basemap.TIF',
                           target_img_fn='/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/SP2.TIF',
                           dem_fn='/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/OPT/DEM.TIF',
                           sat_model_params=sat_model_params,
                           debug=False,
                           svg_patches=True,
                           w_patches=True)
    fcp_opt()
