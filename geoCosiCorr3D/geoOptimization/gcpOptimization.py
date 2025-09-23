"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import sys, pandas
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import geoCosiCorr3D.georoutines.geo_utils as geoRT
import geoCosiCorr3D.georoutines.file_cmd_routines as fileRT
import geoCosiCorr3D.geoErrorsWarning.geoWarnings as geoWarn
import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoCore.core_RSM import RSM
from geoCosiCorr3D.geoOptimization.RSM_Refinement import cRSMRefinement
from geoCosiCorr3D.geoCore.core_correlation import (FreqCorrelator, SpatialCorrelator)
from geoCosiCorr3D.geoOrthoResampling.GCPPatch import (GCPPatch, OrthoPatch, RawInverseOrthoPatch)
from geoCosiCorr3D.geoTiePoints.misc import opt_report

geoWarn.wrIgnoreNotGeoreferencedWarning()

MULTIPROCS_2LOOP_DEBUG = False


class cGCPOptimization:
    def __init__(self,
                 gcp_file_path: str,
                 raw_img_path: str,
                 ref_ortho_path: str,
                 sat_model_params: C.SatModelParams,
                 dem_path: Optional[str] = None,
                 opt_params: Dict = None,
                 opt_gcp_file_path: Optional[str] = None,
                 corr_config: Optional[Dict] = None,
                 debug: Optional[bool] = False,
                 svg_patches: Optional[bool] = False):

        self.gcp_file_path = gcp_file_path
        self.raw_img_path = raw_img_path
        self.ref_ortho_img = ref_ortho_path
        self.dem_path = dem_path
        self.sat_model_params = sat_model_params
        self.debug = debug
        self.opt_gcp_file = opt_gcp_file_path
        self.svg_patches = True  # svg_patches
        if opt_params is None:
            self.opt_params = {}
        else:
            self.opt_params = opt_params
        if corr_config is None:
            self.corr_params = {}
        else:
            self.corr_params = corr_config
        self.ingest()
        self.set_patch_sz()

    def __call__(self, *args, **kwargs):
        self.optimize()

    def ingest(self):
        self.gcp_df = pandas.read_csv(self.gcp_file_path)
        if 'Unnamed: 0' in self.gcp_df.columns:
            self.gcp_df.drop('Unnamed: 0', axis=1, inplace=True)
        logging.info(f'{self.__class__.__name__}: input GCPs:{self.gcp_df.shape}')
        self.nb_gcps = self.gcp_df.shape[0]
        logging.info(f'{self.__class__.__name__}: loading RAW IMG')
        self.raw_img_info = geoRT.cRasterInfo(self.raw_img_path)
        logging.info(f'{self.__class__.__name__}: loading REF IMG')
        self.ref_ortho_info = geoRT.cRasterInfo(self.ref_ortho_img)
        if self.dem_path is None:
            self.dem_info = None
            logging.warning('No DEM provided')
        else:
            logging.info(f'{self.__class__.__name__}: loading DEM ')
            self.dem_info = geoRT.cRasterInfo(self.dem_path)
        self.nb_loops = self.opt_params.get('nb_loops', 3)
        self.snr_th = self.opt_params.get('snr_th', 0.9)
        self.snr_weighting = self.opt_params.get('snr_weighting', True)
        self.mean_error_th = self.opt_params.get('mean_error_th', 1 / 20)
        self.resampling_method = self.opt_params.get('resampling_method', C.Resampling_Methods.SINC)
        logging.info(f'opt_params:nb_loops={self.nb_loops}, snr_th={self.snr_th} , snr_weighting={self.snr_weighting}, '
                     f'mean_error_th={self.mean_error_th}, resampling_method:{self.resampling_method}')

        self.sat_model_name = self.sat_model_params.SAT_MODEL
        if self.sat_model_name not in C.GEOCOSICORR3D_SATELLITE_MODELS:
            msg = f'Sat_model: {self.sat_model_name} is not valid and not supported by {C.SOFTWARE.SOFTWARE_NAME} v{C.SOFTWARE.VERSION}'
            logging.error(msg)
            sys.exit(msg)
        else:
            logging.info(f'Sat_model:{self.sat_model_name}')

        self.sat_model_metadata = self.sat_model_params.METADATA

        if os.path.exists(self.sat_model_metadata) == False:
            msg = f'Sat_model_metadata:{self.sat_model_metadata} does not exist !'
            logging.error(msg)
            sys.exit(msg)

        self.sensor = self.sat_model_params.SENSOR
        if self.sat_model_name == C.SATELLITE_MODELS.RSM:
            if self.sensor not in C.GEOCOSICORR3D_SENSORS_LIST:
                msg = f'sensor: {self.sensor} is not valid and not supported by {C.SOFTWARE.SOFTWARE_NAME} v{C.SOFTWARE.VERSION}'
                logging.error(msg)
                sys.exit(msg)
            else:
                logging.info(f'{self.__class__.__name__}: {self.sat_model_name} sensor: {self.sensor}')
                self.sat_model = RSM.build_RSM(metadata_file=self.sat_model_metadata, sensor_name=self.sensor,
                                               debug=self.debug)
                logging.info(f'{self.__class__.__name__}: {self.sat_model_name}')

        if self.sat_model_name == C.SATELLITE_MODELS.RFM:
            # TODO implement RFM
            raise NotImplementedError

        if self.opt_gcp_file is None:
            self.opt_gcp_file = os.path.join(os.path.dirname(self.gcp_file_path),
                                             Path(self.gcp_file_path).stem + "_opt.pts")
        elif os.path.isdir(self.opt_gcp_file):
            self.opt_gcp_file = os.path.join(self.opt_gcp_file, Path(self.gcp_file_path).stem + "_opt.pts")

        self.corr_method = self.corr_params.get('correlator_name', 'frequency')
        corr_params = self.corr_params.get('correlator_params', {})
        self.corr_wz = corr_params.get('window_size', [128, 128, 128, 128])
        logging.info(f'correlation params: {self.corr_method} || {self.corr_wz}')

        if self.svg_patches:
            self.patches_folder = fileRT.CreateDirectory(directoryPath=os.path.dirname(self.opt_gcp_file),
                                                         folderName=f"{self.raw_image_path}_gcp_patches", cal="y")
        else:
            self.patches_folder = None

        return

    def set_patch_sz(self):
        self.margins = [int(self.corr_wz[0] / 2), int(self.corr_wz[1] / 2)]
        logging.info(f'Margins:{self.margins}')
        self.full_winsz_X = self.corr_wz[0] + 2 * self.margins[0]
        self.full_winsz_Y = self.corr_wz[0] + 2 * self.margins[1]
        logging.info(f"fullwinszX:{self.full_winsz_X},fullwinszY:{self.full_winsz_Y}")
        self.patch_sz = [self.full_winsz_X, self.full_winsz_Y]
        return

    def set_dem(self):
        self.epsg = self.ref_ortho_info.epsg_code
        logging.info(f'epsg:{self.epsg}')
        if self.dem_info is not None:
            if self.dem_info.epsg_code != self.epsg:
                new_dem_path = geoRT.ReprojectRaster(input_raster_path=self.dem_path, o_prj=self.epsg, vrt=True,
                                                     output_raster_path=os.path.join(os.path.dirname(self.opt_gcp_file),
                                                                                     Path(
                                                                                         self.dem_path).stem + "_" + str(
                                                                                         self.epsg) + ".vrt"))
                logging.info(f'Convert input DEM from:{self.dem_info.epsg_code} ---> {self.epsg}')
                self.dem_path = new_dem_path
                self.dem_info = geoRT.cRasterInfo(new_dem_path)
        return

    def set_patch_dims(self):
        """
        Determining path subset dimensions coordinates surrounding each input GCP for : reference ortho, raw Image and DEM.
        For each GCP we define the necessary reference subset grid.
        Three subsets will be defined for the current GCP:
                - Reference image grid subset
                - Target image subset grid onto which orthorectify the target image.
                  If no ground displacement is accounted for between the Reference and the Target,
                  the Target and Reference ground grid coordinates are identical
                - DEM grid subset covering the Target image subset ground area
        Returns:

        """
        arg_list: List = []
        for i in range(self.nb_gcps):
            x_map_coord, y_map_coord, gcp_id = self.gcp_df_corr.get('x_map')[i], self.gcp_df_corr.get('y_map')[i], \
                self.gcp_df_corr.get("gcp_id")[i]
            arg = (
                x_map_coord, y_map_coord, gcp_id, [self.full_winsz_X, self.full_winsz_Y], self.ref_ortho_img,
                self.dem_info)
            arg_list.append(arg)
            # ref_ortho_path_dim_pix, patch_map_extent, dem_path_dim = self.set_patch_dim(*arg)
            # print(ref_ortho_path_dim_pix, patch_map_extent, dem_path_dim)

        logging.info(f'#CPU:{cpu_count()}')
        with Pool(processes=cpu_count()) as pool:
            res = pool.starmap(GCPPatch.set_patch_dim, arg_list)
        ref_ortho_patch_dims_pix = []
        patch_map_extents = []
        dem_patch_dims = []
        valid_gcps = []

        for res_ in res:
            ref_ortho_patch_dims_pix_, patch_map_extents_, dem_path_dims_, valid_gcp = res_[0], res_[1], res_[2], res_[
                3]
            ref_ortho_patch_dims_pix.append(list(ref_ortho_patch_dims_pix_.values()))
            patch_map_extents.append(list(patch_map_extents_.values()))
            dem_patch_dims.append(list(dem_path_dims_.values()))
            valid_gcps.append(valid_gcp)
        return np.asarray(ref_ortho_patch_dims_pix), np.asarray(patch_map_extents), \
            np.asarray(dem_patch_dims), valid_gcps

    @staticmethod
    def generate_patches(raw_img_info: geoRT.cRasterInfo, sat_model, corr_model, patch_sz,
                         ortho_info: geoRT.cRasterInfo,
                         patches_folder,
                         gcp: pandas.DataFrame,
                         dem_path,
                         sat_model_type=C.SATELLITE_MODELS.RSM,
                         loop_nb=None) -> Tuple:
        # ref_ortho_info = geoRT.cRasterInfo(ref_ortho_img)
        ref_ortho_patch: OrthoPatch = OrthoPatch(ortho_info=ortho_info,
                                                 patch_sz=patch_sz,
                                                 gcp=gcp)
        if ref_ortho_patch.patch_status == False:
            # TODO: implement exit
            pass
        # TODO add status to RAwInverseOrtho

        raw_ortho_patch: RawInverseOrthoPatch = RawInverseOrthoPatch(raw_img_info=raw_img_info,
                                                                     sat_model=sat_model,
                                                                     corr_model=corr_model,
                                                                     patch_sz=patch_sz,
                                                                     patch_epsg=ortho_info.epsg_code,
                                                                     patch_res=np.abs(ortho_info.pixel_width),
                                                                     patch_map_extent=ref_ortho_patch.ortho_patch_map_extent,
                                                                     dem_path=dem_path,
                                                                     model_type=sat_model_type,
                                                                     resampling_method=C.Resampling_Methods.SINC,
                                                                     debug=False)

        # TOdo: change the patch writing to optional
        patches_path = cGCPOptimization.write_patches(patches_folder=patches_folder,
                                                      ref_ortho_patch=ref_ortho_patch.get_patch(),
                                                      raw_ortho_patch=raw_ortho_patch.get_patch(),
                                                      gcp_id=gcp['gcp_id'],
                                                      loop_nb=loop_nb)

        return patches_path, gcp, raw_ortho_patch.raw_ortho_patch_map_grid.grid_res

    @staticmethod
    def compute_patch_shift(patch_path, patch_gsd, gcp, patches_folder,
                            corr_method=C.CORRELATION.FREQUENCY_CORRELATOR, debug=False, loop_nb=None):
        # TODO change this function to take as input equal arrays and correlation params
        # TODO Move this function to the correlation package or to the image registration package
        ## what we can do also: this function will be a class function of GCP patch that call a higher level function geoImageCorrealtion package

        ## TODO add the PhaseCorr-CV, PhaseCorr-SK, OpticalFlow
        ew, ns, snr, orthoSubsetRes, dx, dy = np.nan, np.nan, 0, 0, np.nan, np.nan
        patch_info = geoRT.cRasterInfo(patch_path)
        raw_patch = patch_info.image_as_array(2)
        ref_patch = patch_info.image_as_array(1)
        corr_wz = 4 * [int(ref_patch.shape[0] / 2)]
        if corr_method == C.CORRELATION.FREQUENCY_CORRELATOR:
            ew_array, ns_array, snr_array = FreqCorrelator.run_correlator(target_array=raw_patch,
                                                                          base_array=ref_patch,
                                                                          window_size=corr_wz,
                                                                          step=[1, 1],
                                                                          iterations=4,
                                                                          mask_th=0.9)
            dx, dy, snr = ew_array[0, 0], ns_array[0, 0], snr_array[0, 0]

        if corr_method == C.CORRELATION.SPATIAL_CORRELATOR:
            ew_array, ns_array, snr_array = SpatialCorrelator.run_correlator(base_array=ref_patch,
                                                                             target_array=raw_patch,
                                                                             window_size=corr_wz,
                                                                             step=[1, 1],
                                                                             search_range=[10, 10])

            dx, dy, snr = ew_array[0, 0], ns_array[0, 0], snr_array[0, 0]

        ref_ortho_res = patch_gsd
        ew = dx * ref_ortho_res
        ns = dy * ref_ortho_res
        if debug:
            cGCPOptimization.plot_gcp_patches_ground_space(ref_patch=ref_patch,
                                                           raw_ortho_patch=raw_patch,
                                                           gcp_id=gcp['gcp_id'], dx=dx, dy=dy, snr=snr,
                                                           saving_folder=patches_folder, loop_nb=loop_nb)

        return dx, dy, ew, ns, snr

    # def __OptGCP(self, i, loop):
    #
    #     logging.info(" #GCP[{:.3f},{:.3f},{:.3f}]:{}/{}--- Loop:{}".format(self.gcp_df_corr['lon'][i],
    #                                                                        self.gcp_df_corr['lat'][i],
    #                                                                        self.gcp_df_corr['alt'][i],
    #                                                                        i + 1,
    #                                                                        self.nb_gcps,
    #                                                                        loop))
    #
    #     ## Initialization --> in case the GCP is outside rOrtho boundaries
    #     ew, ns, snr, orthoSubsetRes, dx, dy = np.nan, np.nan, 0, 0, np.nan, np.nan
    #
    #     # if self.gcp_df_corr['valid'][i] == True:
    #
    #     if self.dem_info is not None:
    #         dem_path = self.dem_info.input_raster_path
    #     else:
    #         dem_path = None
    #     patches_path, gcp = self.generate_patches(raw_img_path=self.raw_img_path,
    #                                                                                 sat_model=self.sat_model,
    #                                                                                 corr_model=self.corr_model,
    #                                                                                 patch_sz=self.patch_sz,
    #                                                                                 ref_ortho_img=self.ref_ortho_img,
    #                                                                                 dem_path=dem_path,
    #                                                                                 gcp=self.gcp_df_corr.iloc[i],
    #                                                                                 sat_model_type=SATELLITE_MODELS.RSM,
    #                                                                                 patches_folder=self.patches_folder)
    #
    #     if self.corr_method == CORRELATION.FREQUENCY_CORRELATOR:
    #         ew_array, ns_array, snr_array = FreqCorrelator.run_correlator(target_array=raw_ortho_patch.get_patch(),
    #                                                                       base_array=ref_ortho_patch.get_patch(),
    #                                                                       window_size=self.corr_wz,
    #                                                                       step=[1, 1],
    #                                                                       iterations=4,
    #                                                                       mask_th=0.9)
    #         dx, dy, snr = ew_array[0, 0], ns_array[0, 0], snr_array[0, 0]
    #
    #     if self.corr_method == CORRELATION.SPATIAL_CORRELATOR:
    #         ew_array, ns_array, snr_array = SpatialCorrelator.run_correlator(base_array=ref_ortho_patch.get_patch(),
    #                                                                          target_array=raw_ortho_patch.get_patch(),
    #                                                                          window_size=self.corr_wz,
    #                                                                          step=[1, 1],
    #                                                                          search_range=[10, 10])
    #
    #         dx, dy, snr = ew_array[0, 0], ns_array[0, 0], snr_array[0, 0]
    #
    #     ref_ortho_res = raw_ortho_patch.raw_ortho_patch_map_grid.grid_res
    #     ew = dx * ref_ortho_res
    #     ns = dy * ref_ortho_res
    #     logging.info(
    #         'dx[pix]:{:.4f},dy[pix]:{:.4f},snr:{:.4f}, GSD:{}'.format(dx, dy, snr, ref_ortho_res))
    #
    #     if self.debug:
    #         self.plot_gcp_patches_ground_space(ref_patch=ref_ortho_patch.get_patch(),
    #                                            raw_ortho_patch=raw_ortho_patch.get_patch(),
    #                                            gcp_id=self.gcp_df_corr['gcp_id'][i], dx=dx, dy=dy, snr=snr,
    #                                            saving_folder=self.patches_folder)
    #     # else:
    #     #     msg = "gcpValidFlag:{} --> GCP #{} patch is outside rOrtho boundaries".format(self.gcp_df_corr['valid'][i],
    #     #                                                                                   i + 1)
    #     #     warnings.warn(msg)
    #
    #     return (dx, dy, ew, ns, snr)

    @staticmethod
    def compute_rsm_correction(sat_model, gcp_df: pandas.DataFrame, debug=False):
        rsm_refinement = cRSMRefinement(rsm_model=sat_model,
                                        gcps_df=gcp_df, debug=debug)
        corr_model = rsm_refinement()
        del rsm_refinement

        return corr_model

    def correct_gcp(self, snr, dx, dy, gcp_patch, ew, ns):
        # TODO
        ## ==>@SA check the validity of the correlation result ==> Future work

        # TODO
        ## Get the new elevation at the new GCP coordinates (if a DEM is used)
        ## ==> @SA: implement this function to interpolate for h for the new gcp coordinate
        ## For the instance we keep the same h -----> @SA: Future work
        ## Check if we are still inside the DEM boundaries. If yes, interpolate the DEM at the precise
        ## location. If not keep the previous altitude -----> @SA: Future work

        # TODO
        ## If the correlation failed, keep the previous GCP coordinates, set the SNR to 0
        ## GCP won't be accounted for the next loop
        ## inform the user with a message
        if snr != 0 and np.isnan(dx) == False:
            self.gcp_df_corr['x_map'] = np.where(self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id'],
                                                 self.gcp_df_corr['x_map'] - ew, self.gcp_df_corr['x_map'])
        if snr != 0 or np.isnan(dy) == False:
            self.gcp_df_corr['y_map'] = np.where(self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id'],
                                                 self.gcp_df_corr['y_map'] + ns, self.gcp_df_corr['y_map'])

        self.gcp_df_corr['weight'] = np.where(self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id'],
                                              snr, self.gcp_df_corr['weight'])

        res = geoRT.Convert.coord_map1_2_map2(
            X=self.gcp_df_corr[self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id']]['x_map'],
            Y=self.gcp_df_corr[self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id']]['y_map'],
            targetEPSG=4326,
            sourceEPSG=int(self.gcp_df_corr[self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id']]['epsg'].iloc[0]))

        self.gcp_df_corr['lon'] = np.where(self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id'],
                                           res[1][0], self.gcp_df_corr['lon'])
        self.gcp_df_corr['lat'] = np.where(self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id'],
                                           res[0][0], self.gcp_df_corr['lat'])
        if self.dem_info is not None:
            from geoCosiCorr3D.geoTiePoints.Tp2GCPs import TpsToGcps as tp2gcp
            alt = tp2gcp.get_gcp_alt(lon=res[1][0], lat=res[0][0], dem_path=self.dem_path)
            self.gcp_df_corr['alt'] = np.where(self.gcp_df_corr['gcp_id'] == gcp_patch[1]['gcp_id'],
                                               alt, self.gcp_df_corr['alt'])

        return

    def optimize(self):

        self.set_dem()
        self.gcp_df_corr = self.gcp_df.copy()
        if self.epsg != 4326:
            gcp_epsg = self.gcp_df.get('epsg', [None])[0]
            if gcp_epsg != self.epsg:
                logging.warning(f'Convert GCPs from {4326} --> {self.epsg} instead of {gcp_epsg}')
                x_map, y_map = geoRT.Convert.coord_map1_2_map2(X=self.gcp_df.get('lat'),
                                                               Y=self.gcp_df.get('lon'),
                                                               targetEPSG=self.epsg,
                                                               sourceEPSG=4326)
                self.gcp_df_corr['x_map'] = x_map
                self.gcp_df_corr['y_map'] = y_map
                self.gcp_df_corr['epsg'] = self.nb_gcps * [self.epsg]

        # gcp4Opt = np.copy(self.gcps)
        # self.ref_ortho_patch_dims = np.zeros((self.nb_gcps, 5))
        # self.dem_patch_dims = np.zeros((self.nb_gcps, 5))
        # self.raw_img_patch_X0Y0_dims = np.zeros((self.nb_gcps, 2))
        report_dict = {"GCP_ID": [], "Lon": [], "Lat": [], "Alt": [], "nbLoop": [], "dxPix": [], "dyPix": [],
                       "SNR": [], }
        # 'dx': [], 'dy': []}

        #####################
        # self.nbLoops = 1    #
        # self.nbGCPs = 5     #
        #####################
        # if self.visualize:
        #     fig, axs = plt.subplots(1, 2)  # , figsize=(7, 4), constrained_layout=True)

        # self.ref_ortho_patch_dims_pix, self.patch_map_extents, dem_path_dims, valid_gcps = self.set_patch_dims()
        # self.gcp_df_corr['valid'] = valid_gcps
        self.corr_model = np.zeros((3, 3))

        if self.dem_info is not None:
            dem_path = self.dem_info.input_raster_path
        else:
            dem_path = None

        for loop in range(self.nb_loops):
            logging.info(f"--------------------- #Loop:{loop}-----------------------")
            if self.sat_model_name == C.SATELLITE_MODELS.RSM:
                self.corr_model = self.compute_rsm_correction(self.sat_model, self.gcp_df_corr)
            if self.sat_model_name == C.SATELLITE_MODELS.RFM:
                # TODO
                continue
            self.corr_model_file = os.path.join(os.path.dirname(self.opt_gcp_file),
                                                Path(self.opt_gcp_file).stem + "loop_" + str(loop) + "_correction.txt")
            np.savetxt(self.corr_model_file, self.corr_model)
            logging.info(f'loop:{loop} --> Correction model :{self.corr_model}')

            dx_pixs: List = []
            dy_pixs: List = []
            logging.info(f'{self.__class__.__name__}: GCP patch generation ...')
            gcp_patches = generate_gcp_patches(self.nb_gcps, self.raw_img_info, self.sat_model, self.corr_model,
                                               self.patch_sz, self.ref_ortho_info, self.patches_folder,
                                               self.gcp_df_corr, dem_path, loop)

            for gcp_patch in gcp_patches:
                dx, dy, ew, ns, snr = self.compute_patch_shift(patch_path=gcp_patch[0],
                                                               patch_gsd=gcp_patch[2],
                                                               gcp=gcp_patch[1],
                                                               patches_folder=self.patches_folder,
                                                               corr_method=C.CORRELATION.FREQUENCY_CORRELATOR,
                                                               debug=self.svg_patches,
                                                               loop_nb=loop)

                logging.info(
                    'Loop:{} #GCP:{}--> dx[pix]:{:.4f},dy[pix]:{:.4f},snr:{:.4f}'.format(loop,
                                                                                         gcp_patch[1]['gcp_id'], dx,
                                                                                         dy, snr))

                report_dict["GCP_ID"].append(gcp_patch[1]['gcp_id'])
                report_dict["Lon"].append("%.10f" % (gcp_patch[1]['lon']))
                report_dict["Lat"].append("%.10f" % (gcp_patch[1]['lat']))
                report_dict["Alt"].append("%.3f" % (gcp_patch[1]['alt']))
                report_dict["nbLoop"].append("%d" % (loop))
                report_dict["dxPix"].append("%.3f" % (dx))
                report_dict["dyPix"].append("%.3f" % (dy))
                report_dict["SNR"].append("%.3f" % (snr))
                # report_dict["dx"].append("%.3f" % (ew))
                # report_dict["dy"].append("%.3f" % (ns))

                # NOTE Don't remove the following comment
                # for i in tqdm(range(self.nb_gcps), desc="Loop:" + str(loop) + " GCP opt"):
                #     dx, dy, ew, ns, snr = self.__OptGCP(i=i, loop=loop) # TODO: enable this loop for debug, manual comment need to be done.
                # sys.exit() # debug: endpoint
                dx_pixs.append(dx)
                dy_pixs.append(dy)

                self.correct_gcp(snr, dx, dy, gcp_patch, ew, ns)

            self.gcp_df_corr.to_csv(
                os.path.join(os.path.dirname(self.opt_gcp_file), f'{Path(self.opt_gcp_file).stem}_loop_{loop}.csv'))
            # TODO implement GCP selection based on snrTH and other criteria: mean or select the corr of the mean error loop if max loop reached
            # gcp4Opt = np.copy(self.gcps)
            # indexList = [index_ for index_, val_ in enumerate(gcp4Opt[:, 5]) if val_ < self.snrTh]
            # gcp4Opt = np.delete(gcp4Opt, indexList, 0)
            # indexList = [index_ for index_, val_ in enumerate(gcp4Opt[:, 0]) if np.isinf(val_)]
            # gcp4Opt = np.delete(gcp4Opt, indexList, 0)
            # indexList = [index_ for index_, val_ in enumerate(gcp4Opt[:, 1]) if np.isinf(val_)]
            # gcp4Opt = np.delete(gcp4Opt, indexList, 0)

            mean_dx_pix = np.nanmean(np.asarray(dx_pixs))
            mean_dy_pix = np.nanmean(np.asarray(dy_pixs))
            mean_err_xy_pix = np.sqrt(mean_dx_pix ** 2 + mean_dy_pix ** 2)
            std_dx_pix = np.nanstd(np.asarray(dx_pixs))
            std_dy_pix = np.nanstd(np.asarray(dy_pixs))
            xy_rmse = np.sqrt(std_dx_pix ** 2 + std_dy_pix ** 2)

            logging.info(f'{self.__class__.__name__}:: mean_err[pix]:{mean_err_xy_pix} --  RMSE[pix]:{xy_rmse}')

            # if self.debug and self.svg_patches:
            #     # TODO another plot will be the cumulative error plot, where we can compute the elbow thrshold and filter outliers
            #     ## similar to what I have implemented for RFM with Skysat/PlanetScope.
            #     self.plot_error_distribution(dx_pixs, dy_pixs, loop_nb=loop,
            #                                  saving_folder=os.path.dirname(self.opt_gcp_file))
            # if mean_err_xy_pix <= self.mean_error_th:
            #     break

        df_report = pandas.DataFrame.from_dict(report_dict)
        if self.debug:
            logging.info(f"opt_report:{df_report}")

        self.opt_report_path = os.path.join(os.path.dirname(self.opt_gcp_file),
                                            Path(self.opt_gcp_file).stem + ".opt_report.csv")
        df_report.to_csv(self.opt_report_path, index=False)
        self.opt_report_df = df_report

        opt_report(reportPath=self.opt_report_path, image_path= self.raw_img_path, snrTh=self.snr_th)
        return

    @staticmethod
    def plot_gcp_image_space(ref_ortho_info, gcp_df, patch_map_extents, raw_img_info, gcp_index):
        ####### Plotting Matching points in the image plan#####
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(ref_ortho_info.raster_array, cmap="gray")
        ax1.set_title("ref_img")
        xpix_, ypix_ = ref_ortho_info.Map2Pixel(x=gcp_df['x_map'][gcp_index], y=gcp_df['y_map'][gcp_index])
        ax1.scatter(xpix_, ypix_, marker="+", s=50)
        xMin = patch_map_extents[gcp_index, 0]
        xMax = patch_map_extents[gcp_index, 2]
        yMin = patch_map_extents[gcp_index, 3]
        yMax = patch_map_extents[gcp_index, 1]
        xBbox = [xMin, xMax, xMax, xMin, xMin]
        yBbox = [yMax, yMax, yMin, yMin, yMax]
        (xBboxPix, yBboxPix) = ref_ortho_info.Map2Pixel_Batch(X=xBbox, Y=yBbox)
        ax1.plot(xBboxPix, yBboxPix, label="patch_fp", c='g')

        ax2.imshow(raw_img_info.raster_array, cmap="gray")
        ax2.set_title("raw_img")
        ax2.scatter(gcp_df['xPix'][gcp_index], gcp_df['yPix'][gcp_index], marker="+", s=50)
        gcp_id = gcp_df['gcp_id'][gcp_index]
        fig.suptitle(f'GCP:{gcp_id} index:{gcp_index + 1}')
        plt.show()
        plt.clf()
        return

    @staticmethod
    def plot_gcp_patches_ground_space(ref_patch, raw_ortho_patch, gcp_id, dx, dy, snr, saving_folder=None,
                                      loop_nb=None):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(ref_patch, cmap="gray")
        ax1.set_title("ref_ortho_patch")

        ax2.imshow(raw_ortho_patch, cmap="gray")
        ax2.set_title("raw_ortho_patch")
        if loop_nb is not None:
            fig.suptitle("GCP_{}\n dx:{:.3f} , dy:{:.3f}, snr:{:.3f} \n Loop:{}".format(gcp_id, dx, dy, snr, loop_nb))

        else:
            fig.suptitle("GCP_{}\n dx:{:.3f} , dy:{:.3f}, snr:{:.3f}".format(gcp_id, dx, dy, snr))
        if saving_folder is not None:
            if loop_nb is not None:
                plt.savefig(os.path.join(saving_folder, f'{gcp_id}_loop_{loop_nb}.png'))
            else:
                plt.savefig(os.path.join(saving_folder, f'{gcp_id}.png'))

        plt.clf()
        plt.close(fig)
        return

    @staticmethod
    def plot_error_distribution(dx_pixs, dy_pixs, loop_nb=None, saving_folder=None):
        # TODO change distribution by another name as we are not computing a hist of the distribution
        mean_dx_pix = np.nanmean(np.asarray(dx_pixs))
        mean_dy_pix = np.nanmean(np.asarray(dy_pixs))
        # mean_err_xy_pix = np.sqrt(mean_dx_pix ** 2 + mean_dy_pix ** 2)
        # std_dx_pix = np.nanstd(np.asarray(dx_pixs))
        # std_dy_pix = np.nanstd(np.asarray(dy_pixs))
        # xy_rmse = np.sqrt(std_dx_pix ** 2 + std_dy_pix ** 2)

        fig = plt.figure()
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.scatter(dx_pixs, dy_pixs)
        ax.scatter(mean_dx_pix, mean_dy_pix, c='red', marker='d')
        plt.grid(True)

        if saving_folder is not None:
            if loop_nb is not None:
                plt.savefig(os.path.join(saving_folder, f'error_distribution_loop_{loop_nb}.png'))

        plt.clf()
        plt.close(fig)

        return

    @staticmethod
    def write_patches(patches_folder, ref_ortho_patch, raw_ortho_patch, gcp_id, loop_nb=None):
        # TODO add the geo-referencing for each patch, which should be the same (patch Map Grid).
        if loop_nb is not None:
            patches_raster_path = os.path.join(patches_folder, f'{gcp_id}_loop_{loop_nb}_patches.tif')
        else:
            patches_raster_path = os.path.join(patches_folder, f'{gcp_id}_patches.tif')
        geoRT.cRasterInfo.write_raster(output_raster_path=patches_raster_path,
                                       array_list=[ref_ortho_patch, raw_ortho_patch],
                                       descriptions=['ref_patch', 'raw_patch'])
        return patches_raster_path


def generate_gcp_patches(nb_gcps, raw_img_info, sat_model, corr_model, patch_sz,
                         ref_ortho_info,
                         patches_folder, gcp_df_corr, dem_path, loop):
    arg_list: List = []
    # ref_ortho_info = geoRT.cRasterInfo(ref_ortho_img)
    for i in range(nb_gcps):
        arg = [raw_img_info, sat_model, corr_model, patch_sz, ref_ortho_info,
               patches_folder, gcp_df_corr.iloc[i], dem_path, C.SATELLITE_MODELS.RSM, loop]
        arg_list.append(arg)
    # FIXME
    # pool = Pool()
    # gcp_patches_ = pool.map_async(cGCPOptimization.generate_patches_mp, arg_list)
    # gcp_patches_.wait()
    # gcp_patches = [value for value in gcp_patches_.get()]
    # pool.close()

    gcp_patches = []
    for index, arg in enumerate(arg_list):
        logging.info(f'___ Loop:{loop}  GCP:{index + 1}/{len(arg_list)} ___')
        gcp_patches.append(cGCPOptimization.generate_patches(*arg))

    return gcp_patches
