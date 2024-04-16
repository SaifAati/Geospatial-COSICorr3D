"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""
import logging
import os.path
import shutil
import warnings
from pathlib import Path
from typing import List, Optional

import geoCosiCorr3D.geoCore.constants as C
import numpy as np
import pandas
from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog
from geoCosiCorr3D.georoutines.file_cmd_routines import \
    get_files_based_on_extension
from tqdm import tqdm


class GeoCosiCorr3DPipeline:

    def __init__(self,
                 img_list: List,
                 event_date: str,
                 sensor: str,
                 dem_file: str,
                 ref_ortho: str,
                 config_file: Optional[str] = None,
                 workspace_dir: Optional[str] = None,
                 ):

        self.img_list = img_list
        self.event_date = event_date
        self.sensor = sensor
        self.dem_file = dem_file
        self.ref_ortho = ref_ortho
        self.workspace_dir = os.path.join(C.SOFTWARE.WKDIR,
                                          self.__class__.__name__) if workspace_dir is None else workspace_dir
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        geoCosiCorr3DLog(self.__class__.__name__, self.workspace_dir)
        self.config_file = config_file
        self._ingest()
        logging.info(f'{self.__class__.__name__}:: WD:{self.workspace_dir}')
        logging.info(f'{self.__class__.__name__}:: raw_imgs:{self.img_list} , {len(self.img_list)}')
        logging.info(f'{self.__class__.__name__}:: sensor:{self.sensor}')
        logging.info(f'{self.__class__.__name__}:: event_date:{self.event_date}')
        logging.info(f'{self.__class__.__name__}:: configuration:{self.config}')

    def _ingest(self):
        from geoCosiCorr3D.geoCore.geoCosiCorrBaseCfg.BaseReadConfig import \
            ConfigReader
        self.pre_post_pairs_file = os.path.join(self.workspace_dir, "PrePost_Pairs_overlap.csv")
        self.o_3DD_folder = os.path.join(self.workspace_dir, 'o3DDA')
        self.sets_path = os.path.join(self.workspace_dir, "Sets_3DDA.csv")
        self.data_file = os.path.join(self.workspace_dir, "Datafile.csv")
        self.rsm_folder = os.path.join(self.workspace_dir, "RSMs")
        self.corr_folder = os.path.join(self.workspace_dir, "Correlation")
        self.rsm_refinement_folder = os.path.join(self.workspace_dir, "RSM_Refinement")
        self.fp_folder = os.path.join(self.workspace_dir, "Footprints")
        self.matches_folder = os.path.join(self.workspace_dir, "Matches")
        self.ortho_folder = os.path.join(self.workspace_dir, "Orthos")
        self.trx_folder = os.path.join(self.workspace_dir, "Trxs")
        self.data_dic = {"Name": [], "Date": [], "Time": [], "Platform": [], "GSD": [], "ImgPath": [], "DIM": [],
                         "RSM": [], "Fp": [], "Tp": [], "MatchFile": [], "GCPs": [], 'RSM_Refinement': []}
        if self.config_file is None:
            self.config_file = os.path.join(C.SOFTWARE.PARENT_FOLDER,
                                            'geoCosiCorr3D/geoCore/geoCosiCorrBaseCfg/geo_ortho_config.yaml')
        shutil.copy(self.config_file, os.path.join(self.workspace_dir, os.path.basename(self.config_file)))
        logging.info(f'config file:{self.config_file}')
        self.config = ConfigReader(config_file=self.config_file).get_config

        return

    def parse_data_file(self, data_file):
        self.data = pandas.read_csv(data_file)
        self.data = self.data.sort_values('Name', ascending=False)
        self.data = self.data.drop_duplicates(subset='Name', keep='first')
        valid_index = self.data[~self.data['Name'].isnull()].index.tolist()
        self.valid_data = self.data.loc[valid_index]
        self.pre_event_df = self.valid_data[(self.valid_data['Date'] < self.event_date)]
        self.post_event_df = self.valid_data[(self.valid_data['Date'] >= self.event_date)]
        logging.info("{} ::PreEvent:{}  || PostEvent:{}".format(self.__class__.__name__, self.pre_event_df.shape[0],
                                                                self.post_event_df.shape[0]))
        return self.valid_data

    def update_data_file(self, data_file, updated_df):
        self.valid_data = updated_df
        self.data = pandas.read_csv(data_file)
        key_list = list(set(self.valid_data.columns) - set(self.data.columns))
        if len(key_list) > 0:
            for key in key_list:
                self.data[key] = self.data.shape[0] * [None]

        for index, row in updated_df.iterrows():
            indexList = self.data.index[self.data["Name"] == row["Name"]].to_list()
            self.data.iloc[indexList[0]] = row
        self.data_file = data_file
        self.data.to_csv(self.data_file, index=False, header=True)
        return

    def build_rsm_data_file(self):
        from geoCosiCorr3D.geoCore.core_RSM import RSM
        from geoCosiCorr3D.geoRSM.geoRSM_generation import geoRSM_generation
        Path(self.rsm_folder).mkdir(parents=True, exist_ok=True)

        ## Create RSM model List

        errorList = []
        for raw_img in tqdm(self.img_list, desc="Computing RSM model"):

            if self.sensor in C.GEOCOSICORR3D_SENSOR_SPOT_15:
                dmp_file = get_files_based_on_extension(os.path.dirname(raw_img), "*.DIM")[0]
            elif self.sensor in C.GEOCOSICORR3D_SENSOR_DG:
                dmp_file = get_files_based_on_extension(os.path.dirname(raw_img), "*.XML")[0]
            else:
                dmp_file = None
                logging.error(f'Enable to find RSM file for {raw_img}')
                continue
            logging.info(f'metadat file:{dmp_file}')
            try:
                shutil.copy(dmp_file, os.path.join(self.rsm_folder, os.path.basename(dmp_file)))
                dmp_file_orig = dmp_file
                dmp_file = os.path.join(self.rsm_folder, os.path.basename(dmp_file))
                rsm_model = geoRSM_generation(sensor_name=self.sensor, metadata_file=dmp_file, debug=True)

                name = rsm_model.date_time_obj.strftime("%Y-%m-%d-%H-%M-%S") + "-" + rsm_model.platform + "-" + str(
                    rsm_model.gsd)
                geoCosiCorr3D_rsm_file = RSM.write_rsm(output_rsm_file=os.path.join(self.rsm_folder, f'{name}.pkl'),
                                                       rsm_model=rsm_model)
                logging.info(f'{self.__class__.__name__}:: name:{name}')
                self.data_dic["ImgPath"].append(raw_img)
                self.data_dic["DIM"].append(dmp_file_orig)
                self.data_dic["Name"].append(name)
                self.data_dic["Date"].append(rsm_model.date)
                self.data_dic["Time"].append(rsm_model.time)
                self.data_dic["Platform"].append(rsm_model.platform)
                self.data_dic["GSD"].append(rsm_model.gsd)
                self.data_dic["RSM"].append(geoCosiCorr3D_rsm_file)
                self.data_dic["Fp"].append(None)
                self.data_dic["MatchFile"].append(None)
                self.data_dic["Tp"].append(None)
                self.data_dic["GCPs"].append(None)
                self.data_dic["RSM_Refinement"].append(None)

            except:
                msg = "UNABLE TO COMPUTE RSM FOR" + raw_img
                warnings.warn(msg)
                errorList.append(raw_img)
                logging.info(msg)
                logging.error(msg)

                continue
        # # print("___________________ WARNING: UNABLE TO COMPUTE RSM FOR THESE IMAGES: ___________________________")
        # # print(errorList)
        # # print("____________________ END WARNING: UNABLE TO COMPUTE RSM FOR THESE IMAGES: ______________________")
        df = pandas.DataFrame.from_dict(self.data_dic)

        df.to_csv(self.data_file, index=False, header=True)
        logging.info(f'{self.__class__.__name__}: data:{df}')
        return self.data_file

    def compute_footprint(self, data_file):
        from geoCosiCorr3D.geoCore.core_RSM import RSM
        dataDf = self.parse_data_file(data_file)

        Path(self.fp_folder).mkdir(parents=True, exist_ok=True)
        validIndexList = [index for index, row in dataDf.iterrows()]
        for index in tqdm(validIndexList, desc="Computing footprint"):
            rsm_model = RSM.read_geocosicorr3d_rsm_file(dataDf["RSM"][index])
            _, _, _, gdf_fp = RSM.compute_rsm_footprint(rsm_model=rsm_model, dem_file=self.dem_file)
            fp_path = os.path.join(self.fp_folder, dataDf["Name"][index] + ".geojson")
            gdf_fp.to_file(fp_path, driver='GeoJSON')
            dataDf.loc[index, "Fp"] = fp_path
            self.update_data_file(data_file=data_file, updated_df=dataDf)
        return

    def feature_detection(self, data_file):
        from geoCosiCorr3D.geoTiePoints.wf_features import features
        dataDf = self.parse_data_file(data_file)
        Path(self.matches_folder).mkdir(parents=True, exist_ok=True)
        validIndexList = [index for index, row in dataDf.iterrows()]
        for index in tqdm(validIndexList, desc="Tp detection and Matching"):
            raw_match_file = features(img1=self.ref_ortho,
                                      img2=dataDf["ImgPath"][index],
                                      tp_params=self.config['feature_points_params'],
                                      output_folder=self.matches_folder)
            dataDf.loc[index, "Tp"] = np.loadtxt(raw_match_file, comments=';').shape[0]
            match_file = os.path.join(self.matches_folder,
                                      f"{Path(self.ref_ortho).stem}_VS_{dataDf['Name'][index]}_matches.pts")
            os.rename(raw_match_file, match_file)
            dataDf.loc[index, "MatchFile"] = match_file

        self.update_data_file(data_file=data_file, updated_df=dataDf)
        return

    def gcp_generation(self, data_file):
        from geoCosiCorr3D.geoTiePoints.Tp2GCPs import TpsToGcps as tp2gcp
        def isNaN(string):
            return string != string

        self.data_file = data_file
        dataDf = self.parse_data_file(self.data_file)
        min_tps = self.config.get('min_tps', 20)
        validIndexList = [index for index, row in dataDf.iterrows()]
        for item, index in enumerate(validIndexList):
            logging.info(f'{self.__class__.__name__}:-- GCP generation--: [{item + 1}]/[{len(validIndexList)}]')
            match_file = dataDf.loc[index, "MatchFile"]
            if match_file is not None and isNaN(match_file) == False and dataDf.loc[index, "Tp"] > min_tps:
                gcp = tp2gcp(in_tp_file=match_file,
                                base_img_path=dataDf.loc[index, "ImgPath"],
                                ref_img_path=self.ref_ortho,
                                dem_path=self.dem_file,
                                debug=True)
                gcp()
                dataDf.loc[index, "GCPs"] = gcp.output_gcp_path
                logging.info(
                    f'{self.__class__.__name__}: GCP GENERATION: GCPs for:{dataDf.loc[index, "ImgPath"]} --> {gcp.output_gcp_path}')
                del gcp

            else:
                dataDf.loc[index, "GCPs"] = None
                logging.warning(
                    f'{self.__class__.__name__}: GCP GENERATION: UNABLE TO GENERATE GCPS for:{dataDf.loc[index, "ImgPath"]}')
        self.update_data_file(data_file=self.data_file, updated_df=dataDf)
        return

    def rsm_refinement(self, data_file, recompute=False):
        from geoCosiCorr3D.geoOptimization.gcpOptimization import \
            cGCPOptimization

        Path(self.rsm_refinement_folder).mkdir(parents=True, exist_ok=True)

        def isNaN(string):
            return string != string

        self.data_file = data_file
        dataDf = self.parse_data_file(self.data_file)
        dataDf['RSM_Refinement'] = dataDf.shape[0] * [None]
        validIndexList = [index for index, row in dataDf.iterrows()]
        for item, validIndex in enumerate(validIndexList):
            gcp_file = dataDf.loc[validIndex, "GCPs"]
            logging.info(
                f'{self.__class__.__name__}: -- RSM refinement --[{item + 1}]/[{len(validIndexList)}]:{gcp_file}')
            if isNaN(gcp_file) == False and str(gcp_file) is not None:
                # sat_model_params = {'sat_model': C.SATELLITE_MODELS.RSM, 'metadata': dataDf.loc[validIndex, "DIM"],
                #                     'sensor': self.sensor}
                sat_model_params = C.SatModelParams(C.SATELLITE_MODELS.RSM, dataDf.loc[validIndex, "DIM"], self.sensor)
                logging.info(
                    f"{self.__class__.__name__}:RSM refinement:{sat_model_params.METADATA}")
                rsm_refinement_dir = os.path.join(self.rsm_refinement_folder,
                                                  f"{dataDf.loc[validIndex, 'Name']}_{Path(sat_model_params.METADATA).stem}")
                Path(rsm_refinement_dir).mkdir(parents=True, exist_ok=True)
                opt = cGCPOptimization(gcp_file_path=gcp_file,
                                       raw_img_path=dataDf.loc[validIndex, "ImgPath"],
                                       ref_ortho_path=self.ref_ortho,
                                       sat_model_params=sat_model_params,
                                       dem_path=self.dem_file,
                                       opt_params=self.config['opt_params'],
                                       opt_gcp_file_path=os.path.join(rsm_refinement_dir,
                                                                      Path(gcp_file).stem + "_opt.pts"),
                                       corr_config=self.config['opt_corr_config'],
                                       debug=False,
                                       svg_patches=False)
                opt()
                dataDf.loc[validIndex, "RSM_Refinement"] = opt.opt_report_path
            else:
                msg = "RSM optimization and correction files exists for img :{}".format(dataDf.loc[validIndex, "Name"])
                warnings.warn(msg)
                logging.error(msg)

        self.update_data_file(self.data_file, dataDf)
        return

    def orthorectify(self, data_file, ortho_gsd):
        from geoCosiCorr3D.geoOrthoResampling.geoOrtho import RSMOrtho
        from geoCosiCorr3D.geoTiePoints.misc import parse_opt_report

        self.data_file = data_file

        dataDf = self.parse_data_file(self.data_file)
        dataDf['Orthos'] = dataDf.shape[0] * [None]
        dataDf['Trxs'] = dataDf.shape[0] * [None]
        Path(self.ortho_folder).mkdir(parents=True, exist_ok=True)
        Path(self.trx_folder).mkdir(parents=True, exist_ok=True)
        validIndexList = [index for index, row in dataDf.iterrows()]

        for item, validIndex in enumerate(validIndexList):
            _, _, loop_min_err = parse_opt_report(dataDf.loc[validIndex, "RSM_Refinement"])

            output_ortho_path = os.path.join(self.ortho_folder,
                                             f"{dataDf.loc[validIndex, 'Name']}_ORTHO_{ortho_gsd}.tif")
            output_trans_path = os.path.join(self.trx_folder, f"{dataDf.loc[validIndex, 'Name']}_TRX_{ortho_gsd}.tif")
            self.config['ortho_params']['method']['metadata'] = dataDf.loc[validIndex, "DIM"]
            self.config['ortho_params']['method']['sensor'] = self.sensor
            self.config['ortho_params']['method']['corr_model'] = \
                get_files_based_on_extension(os.path.dirname(dataDf.loc[validIndex, "RSM_Refinement"]),
                                             f"*_{loop_min_err}_correction.txt")[0]
            self.config['ortho_params']['GSD'] = ortho_gsd
            ortho = RSMOrtho(input_l1a_path=dataDf.loc[validIndex, "ImgPath"],
                     ortho_params=self.config['ortho_params'],
                     output_ortho_path=output_ortho_path,
                     output_trans_path=output_trans_path,
                     dem_path=self.dem_file)
            ortho()
            dataDf.loc[validIndex, "Orthos"] = output_ortho_path
            dataDf.loc[validIndex, "Trxs"] = output_trans_path
            # break

        self.update_data_file(self.data_file, dataDf)

        return

    def compute_pre_post_pairs(self, data_file, pre_post_overlap_th=80):
        pandas.options.mode.chained_assignment = None
        data = pandas.read_csv(data_file)
        data = data.sort_values('Name', ascending=False)
        data = data.drop_duplicates(subset='Orthos', keep='first')

        validData = data.loc[~data["Orthos"].isnull()]

        pre_event_df = validData[(validData['Date'] < self.event_date)]
        post_event_df = validData[(validData['Date'] >= self.event_date)]
        logging.info(f"Pre Event data:{pre_event_df['Name'].values}")
        logging.info(f"Post Event data:{post_event_df['Name'].values}")
        prePostDf = self.compute_pre_post_overlap(pre_event_df, post_event_df, pre_post_overlap_th)

        prePostDf.to_csv(self.pre_post_pairs_file, index=False, header=True)

        return

    @staticmethod
    def compute_pre_post_overlap(pre_event_df, post_event_df, overlap_th: Optional[int] = None):
        import geopandas
        import rasterio
        import shapely.geometry

        preList = [pair for index, pair in pre_event_df.iterrows()]
        postList = [pair for index, pair in post_event_df.iterrows()]

        pairs = []
        for pre_ in preList:
            for post_ in postList:
                pairs.append([pre_, post_])
        pairsDic = {"pre_i": [], "post_j": [], "Intersection": [], "Overlap": []}
        overlapList = []
        for idx, pair_ in enumerate(pairs):

            pre_ortho_raster = rasterio.open(pair_[0]["Orthos"])
            pre_ortho_fp = shapely.geometry.box(*pre_ortho_raster.bounds)
            pre_roi = geopandas.GeoSeries(pre_ortho_fp, crs=pre_ortho_raster.crs)
            post_ortho_raster = rasterio.open(pair_[1]["Orthos"])
            post_ortho_fp = shapely.geometry.box(*post_ortho_raster.bounds)
            post_roi = geopandas.GeoSeries(post_ortho_fp, crs=post_ortho_raster.crs)
            intersection = pre_roi.intersection(post_roi, align=False)
            if intersection.is_empty[0]:
                continue

            overlap = np.max([(intersection.area / pre_roi.area) * 100,
                              (intersection.area / post_roi.area) * 100])
            overlapList.append(overlap)
            logging.info(f'pair[{idx + 1}]/[{len(pairs)}] intersection :{intersection}, overlap:{overlap}%')
            pairsDic["pre_i"].append(pair_[0]["Orthos"])
            pairsDic["post_j"].append(pair_[1]["Orthos"])
            pairsDic["Intersection"].append(intersection[0])
            pairsDic["Overlap"].append(overlap)

        prePostDf = pandas.DataFrame.from_dict(pairsDic)

        if overlap_th is not None:
            prePostDf = prePostDf[prePostDf["Overlap"] >= overlap_th]
            logging.info("#Selected pairs:{} (with prePostOverlapTh={}) ".format(prePostDf.shape[0], overlap_th))

        return prePostDf

    def correlate(self, corr_mode='pre_post'):

        from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate

        Path(self.corr_folder).mkdir(parents=True, exist_ok=True)

        baseList = []
        targetList = []

        corr_config = self.config['corr_config']
        logging.info(f'Correlation config:{corr_config}')

        if corr_mode == "pre_post":
            data = pandas.read_csv(self.pre_post_pairs_file)
            baseList = data["pre_i"].tolist()
            targetList = data["post_j"].to_list()

            if corr_config['strategy'] == "full":
                baseList_temp = baseList.copy()
                baseList.extend(targetList)
                targetList.extend(baseList_temp)

        if corr_mode == "set":
            data = pandas.read_csv(self.sets_path)
            if corr_config['strategy'] == "full":
                from itertools import permutations
                for index, row in data.iterrows():
                    set = [row["pre_i"], row["pre_j"], row["post_i"], row["post_j"]]
                    perm = np.asarray(list(permutations(set, 2)))
                    baseList.extend(perm[:, 0])
                    targetList.extend(perm[:, 1])

        for i in tqdm(range(len(baseList)), desc="Batch correlation"):
            baseImg = baseList[i]
            targetImg = targetList[i]
            Correlate(base_image_path=baseImg,
                      target_image_path=targetImg,
                      base_band=1,
                      target_band=1,
                      output_corr_path=self.corr_folder,
                      corr_config=corr_config,
                      corr_show=True)

        return

    def generate_3DDA_sets(self, data_file, pairs_overlap_th: Optional[float] = None,
                           sets_overlap_th: Optional[float] = None):

        pandas.options.mode.chained_assignment = None
        data = pandas.read_csv(data_file)
        data = data.sort_values('Name', ascending=False)
        data = data.drop_duplicates(subset='Orthos', keep='first')

        validData = data.loc[~data["Orthos"].isnull()]

        preEventDF = validData[(validData['Date'] < self.event_date)]
        postEventDF = validData[(validData['Date'] >= self.event_date)]
        prePairsDf = self.compute_pairs(dataDf=preEventDF, overlap_th=pairs_overlap_th)
        prePairsDf.to_csv(os.path.join(self.workspace_dir, "Pre_Pairs_overlap.csv"), index=False, header=True)
        postPairsDf = self.compute_pairs(dataDf=postEventDF, overlap_th=pairs_overlap_th)
        postPairsDf.to_csv(os.path.join(self.workspace_dir, "Post_Pairs_overlap.csv"), index=False, header=True)
        setsDf = self.generate_sets(prePairsDf, postPairsDf, sets_overlap_th)
        setsDf.to_csv(self.sets_path, index=False, header=True)

        return

    @staticmethod
    def compute_pairs(dataDf, overlap_th: Optional[float] = None):
        from itertools import combinations

        import geopandas
        import rasterio
        import shapely.geometry
        fp_polygon = []

        for img_ in dataDf["Orthos"]:
            raster = rasterio.open(img_)
            raster_fp = shapely.geometry.box(*raster.bounds)
            fp_polygon.append(raster_fp)
            dataDf.loc[dataDf["Orthos"] == img_, "Ortho_fp"] = raster_fp
            dataDf.loc[dataDf["Orthos"] == img_, "Ortho_crs"] = str(raster.crs)

        pairs = np.array(list(combinations(dataDf["Orthos"].to_list(), 2)))
        pairsDic = {"img_i": pairs[:, 0],
                    "img_j": pairs[:, 1],
                    "fp_i": len(pairs[:, 1]) * [None],
                    "fp_j": len(pairs[:, 1]) * [None],
                    "Intersection": len(pairs[:, 1]) * [None],
                    "Overlap": len(pairs[:, 1]) * [None],
                    "crs": len(pairs[:, 1]) * [None],
                    }
        pairsDf = pandas.DataFrame.from_dict(pairsDic)
        pairList = [pair for index, pair in pairsDf.iterrows()]
        indexList = [index for index, pair in pairsDf.iterrows()]

        for index in tqdm(indexList, desc="Computing pairs Intersection"):
            # for index in indexList:

            pair = pairList[index]
            fp_i = geopandas.GeoSeries(dataDf.loc[dataDf["Orthos"] == pair["img_i"], "Ortho_fp"].to_list(),
                                       crs=dataDf.loc[dataDf["Orthos"] == pair["img_i"], "Ortho_crs"].values[0])
            fp_j = geopandas.GeoSeries(dataDf.loc[dataDf["Orthos"] == pair["img_j"], "Ortho_fp"].to_list(),
                                       crs=dataDf.loc[dataDf["Orthos"] == pair["img_j"], "Ortho_crs"].values[0])
            pairsDf["fp_i"][index] = fp_i[0]
            pairsDf["fp_j"][index] = fp_j[0]

            intersection = fp_i.intersection(fp_j, align=False)
            pairsDf["crs"][index] = dataDf.loc[dataDf["Orthos"] == pair["img_j"], "Ortho_crs"].values[0]
            if intersection.is_empty.values[0]:
                continue
            else:

                pairsDf["Intersection"][index] = intersection[0]
                overlap = np.max(
                    [(intersection.area / fp_i.area) * 100, (intersection.area / fp_j.area) * 100])
                pairsDf["Overlap"][index] = overlap

        if overlap_th is not None:
            old = pairsDf.shape[0]
            pairsDf = pairsDf[pairsDf["Overlap"] >= float(overlap_th)]
            print(old, "---->", pairsDf.shape[0])
        return pairsDf

    @staticmethod
    def generate_sets(pre_pairs_df: pandas.DataFrame, post_pairs_df: pandas.DataFrame,
                      sets_overlap_th: Optional[float] = None):

        import geopandas
        import shapely.wkt
        prePairList = [pair for index, pair in pre_pairs_df.iterrows()]
        postPairList = [pair for index, pair in post_pairs_df.iterrows()]
        sets = []
        for prePair_ in prePairList:
            for postPair_ in postPairList:
                sets.append([prePair_, postPair_])

        setsDic = {"pre_i": [], "pre_j": [], "post_i": [], "post_j": [], "Intersection": [], "Overlap": []}
        overlapList = []
        for idx, set_ in enumerate(sets):

            if isinstance(set_[0]["Intersection"], str):
                pre_ij_roi = geopandas.GeoSeries(shapely.wkt.loads(set_[0]["Intersection"]), crs=set_[0]['crs'])
            else:
                pre_ij_roi = geopandas.GeoSeries(set_[0]["Intersection"], crs=set_[0]['crs'])

            if isinstance(set_[1]["Intersection"], str):
                post_ij_roi = geopandas.GeoSeries(shapely.wkt.loads(set_[1]["Intersection"]), crs=set_[1]['crs'])
            else:
                post_ij_roi = geopandas.GeoSeries(set_[1]["Intersection"], crs=set_[1]['crs'])
            intersection = pre_ij_roi.intersection(post_ij_roi, align=False)
            if intersection.is_empty.values[0]:
                continue

            overlap = np.max(
                [(intersection.area / pre_ij_roi.area) * 100,
                 (intersection.area / post_ij_roi.area) * 100])
            # print(overlap)
            logging.info(f'SET [{idx + 1}]/[{len(sets)}] intersection overlap: {overlap} %')
            overlapList.append(overlap)

            setsDic["pre_i"].append(set_[0]["img_i"])
            setsDic["pre_j"].append(set_[0]["img_j"])
            setsDic["post_i"].append(set_[1]["img_i"])
            setsDic["post_j"].append(set_[1]["img_j"])
            setsDic["Intersection"].append(intersection[0])
            setsDic["Overlap"].append(overlap)

        setsDf = pandas.DataFrame.from_dict(setsDic)
        logging.info("#Sets:{} ".format(setsDf.shape[0]))
        if sets_overlap_th is not None:
            setsDf = setsDf[setsDf["Overlap"] >= sets_overlap_th]
            logging.info("#Selected sets:{} (with setOverlapTh={}) ".format(setsDf.shape[0], sets_overlap_th))

        return setsDf

    @staticmethod
    def plot_pairs_distribution(corrPair, oFolder=None, save=True):

        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.ticker import AutoMinorLocator
        fig = plt.figure()
        fontSize = 14
        gs = gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0, 0])  # row 0, col 0
        # sns.set()
        xLabel = "Overlap [%]"
        color = "#0000FF"
        # ax1.set_title('Displacement density distribution', fontsize=fontSize)
        overlapList = [float(val) for val in corrPair[:, 2]]

        sns.histplot(data=overlapList, bins=80, color=color, kde=False, ax=ax1, alpha=0.3, linewidth=1.3,
                     edgecolor=color)
        ax1.set_xlabel(xLabel, fontsize=fontSize)
        ax1.set_ylabel('# Pairs', fontsize=fontSize)

        ax1.tick_params(axis='x', labelsize=fontSize)
        ax1.tick_params(axis='y', labelsize=fontSize)
        # ax1.legend(fontsize=fontSize)
        # ax1.grid()
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(which='both', width=2, direction="in")
        ax1.grid(True, 'major', ls='solid', lw=0.5, color='gray')
        ax1.grid(True, 'minor', ls='solid', lw=0.1, color='gray')
        # print("# of correlation pairs: {}".format(len(corrPairList)))
        if save == True:
            fig.savefig(os.path.join(oFolder, "PairOverlapDistribution.svg"), dpi=600)
        # plt.show()

        return

    def compute_3DD(self, data_file):
        import datetime

        import pandas
        from geoCosiCorr3D.geo3DDA.main_geo3DDA import cCombination3DD
        from geoCosiCorr3D.geo3DDA.misc import generate_3DD_set_combination
        Path(self.o_3DD_folder).mkdir(parents=True, exist_ok=True)
        set_data = pandas.read_csv(self.sets_path)

        for set_idx, row in set_data.iterrows():
            logging.info(f"_________________SET:{set_idx + 1}/{set_data.shape[0]}_____________")
            set_dir = os.path.join(self.o_3DD_folder, f'3DDA_Set_{set_idx + 1}')
            Path(set_dir).mkdir(parents=True, exist_ok=True)
            set = [row["pre_i"], row["pre_j"], row["post_i"], row["post_j"]]
            set_comb = generate_3DD_set_combination([row["pre_i"], row["pre_j"]], [row["post_i"], row["post_j"]])
            logging.info(f'Set: [{set_idx + 1}] --> #3DDA Set combinations:{len(set_comb)}')

            for comb_idx, comb_ in enumerate(set_comb):
                set_comb_dir = os.path.join(set_dir, "Set_" + str(set_idx + 1) + "_Comb" + str(comb_idx + 1))
                if os.path.exists(os.path.join(set_comb_dir, "Set_" + str(set_idx + 1) + "_Comb" + str(
                        comb_idx + 1) + "_3DDA.tif")):
                    continue
                Path(set_comb_dir).mkdir(parents=True, exist_ok=True)

                comObj = cCombination3DD(comb_=comb_,
                                         data_df=pandas.read_csv(data_file),
                                         event_date=datetime.datetime.strptime(self.event_date, '%Y-%m-%d'),
                                         corr_dir=self.corr_folder,
                                         corr_config=self.config['corr_config'],
                                         dem_file=self.dem_file,
                                         tile_sz=128,
                                         num_cpus=40,
                                         o_set_comb_dir=set_comb_dir)
                del comObj
                # sys.exit()

        return

    # def PerformingThreeDD_old(self, refDEM,
    #                       setFile,
    #                       workspaceFolder,
    #                       corrFolder,
    #                       corrEngine,
    #                       dataFile,
    #                       tileSize,
    #                       eventDate,
    #                       recomputeRSM=False,
    #                       numCpus=40,
    #                       debug=False):
    #     import pandas
    #     import datetime
    #     from pro.geoThreeDD import ThreeDSetCombinations
    #     from pro.geoThreeDD.geoThreeDD import cCombination3DD
    #     oThreeDDFolder = fileRT.CreateDirectory(workspaceFolder, "o3DDA", "n")
    #     pairsData = pandas.read_csv(setFile)
    #     data = pandas.read_csv(dataFile)
    #     referenceDate = datetime.datetime.strptime(eventDate, '%Y-%m-%d')
    #     for setIndex, row in pairsData.iterrows():
    #         print("_________________SET:{}/{}_____________".format(setIndex + 1, pairsData.shape[0]))
    #         oSetFolder = fileRT.CreateDirectory(oThreeDDFolder, "3DDA_Set_" + str(setIndex + 1), "n")
    #         set = [row["pre_i"], row["pre_j"], row["post_i"], row["post_j"]]
    #
    #         totalComb = ThreeDSetCombinations(preOrthoList=[row["pre_i"], row["pre_j"]],
    #                                           postOrthoList=[row["post_i"], row["post_j"]])
    #
    #         for combIndex, comb_ in enumerate(totalComb):
    #             if os.path.exists(os.path.join(oSetFolder, "Set_" + str(setIndex + 1) + "_Comb" + str(combIndex + 1))):
    #                 oCombFolder = os.path.join(oSetFolder, "Set_" + str(setIndex + 1) + "_Comb" + str(combIndex + 1))
    #                 if os.path.exists(os.path.join(oCombFolder, "Set_" + str(setIndex + 1) + "_Comb" + str(
    #                         combIndex + 1) + "_3DDA.tif")):
    #                     print("*_3DDA.tif", " Exist !")
    #                     continue
    #             oCombFolder = fileRT.CreateDirectory(oSetFolder,
    #                                                  "Set_" + str(setIndex + 1) + "_Comb" + str(combIndex + 1), "n")
    #
    #             print(comb_)
    #             print(data)
    #             print(referenceDate)
    #             print(corrFolder)
    #             print(corrEngine)
    #             print(refDEM)
    #             print(tileSize)
    #             print(numCpus)
    #             print(oCombFolder)
    #             comObj = cCombination3DD(debug=debug,
    #                                      comb_=comb_,
    #                                      data=data,
    #                                      referenceDate=referenceDate,
    #                                      corrFolder=corrFolder,
    #                                      corrEngine=corrEngine,
    #                                      refDEM=refDEM,
    #                                      tileSize=tileSize,
    #                                      numCpus=numCpus,
    #                                      oCombFolder=oCombFolder)
    #             del comObj
    #             # sys.exit()
    #
    #     return
