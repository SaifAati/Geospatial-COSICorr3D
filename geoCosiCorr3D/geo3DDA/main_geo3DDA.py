"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2023
"""
import os
import pickle
import datetime
import multiprocessing
from typing import List, Optional
from osgeo import gdal
import glob
import pandas
from p_tqdm import p_map
from pathlib import Path
import numpy as np
from geoCosiCorr3D.georoutines.file_cmd_routines import get_files_based_on_extensions, CreateDirectory, CopyFile

from geoCosiCorr3D.georoutines.geo_utils import cRasterInfo, ReprojectRaster, crop_raster
from geoCosiCorr3D.geo3DDA.misc import ImgInfo, all_equal
from geoCosiCorr3D.geo3DDA.compute_3DD import cCompute3DD, fun_compute3DD
from geoCosiCorr3D.geo3DDA.misc import merge_3dd_tiles

TILE_FOLDER_SUFFIX = '-TILE'
TILE_FILE_SUFFIX = 'tile'


class cCombination3DD:
    def __init__(self, comb_, data_df: pandas.DataFrame, event_date: datetime.datetime, corr_dir, dem_file,
                 tile_sz, num_cpus, corr_config, o_set_comb_dir, debug=True, pmap=True, overwrite=True):
        cal = 'y' if overwrite else 'n'
        self.tile_sz = tile_sz
        if debug:
            print("Combination:", len(comb_))
        _3D_set = {"Base": [], "Pre1": [], "Post1": [], "Post2": []}
        self.corr_config = corr_config
        self.corr_dir = corr_dir
        self.dem_path = dem_file

        self.base_info = self.set_data_info(data_df, comb_[0])
        self.pre_event_info = self.set_data_info(data_df, comb_[1])
        self.post_event_1_info = self.set_data_info(data_df, comb_[2])
        self.post_event_2_info = self.set_data_info(data_df, comb_[3])

        if debug:
            self.repr_input_data_info()

        set_names = [self.base_info.imgName, self.pre_event_info.imgName, self.post_event_1_info.imgName,
                     self.post_event_2_info.imgName]
        ortho_list = [self.base_info.orthoPath, self.pre_event_info.orthoPath, self.post_event_1_info.orthoPath,
                      self.post_event_2_info.orthoPath]
        if debug:
            print("Set Names:", set_names)
            print("Ortho path", ortho_list)

        _3D_set["Base"].append(self.base_info.rsmFile)

        inversion_flag = self.set_inversion_flag(self.base_info, event_date)

        rsm_list = [self.base_info.rsmFile, self.pre_event_info.rsmFile, self.post_event_1_info.rsmFile,
                    self.post_event_2_info.rsmFile]

        trx_list = [self.base_info.warpRaster, self.pre_event_info.warpRaster, self.post_event_1_info.warpRaster,
                    self.post_event_2_info.warpRaster]
        self._3DD_map_epsg = cRasterInfo(self.base_info.orthoPath).epsg_code
        print(f'3DD EPSG:{self._3DD_map_epsg}')
        if debug:
            print("====================================")
            print("             Warp Files: ")
            print(self.base_info.warpRaster)
            print(self.pre_event_info.warpRaster)
            print(self.post_event_1_info.warpRaster)
            print(self.post_event_2_info.warpRaster)
            print("====================================")

        self.set_3DDA_processing_folders(o_set_comb_dir, cal)

        self.get_3DD_com_rsm(rsm_list, self.rsm_folder)
        self.overlap_roi_utm = self.get_overlap_area(trx_list)
        self.crop_rasters(raster_list=trx_list, o_folder=self.trx_folder)
        corr_list = self.get_corr_list()
        self.crop_rasters(raster_list=corr_list, o_folder=self.crop_corr_folder)
        dem_path_crp = self.set_dem()

        self.tiling(in_folders=[self.dem_folder, self.crop_corr_folder, self.trx_folder],
                    ref_img_path=glob.glob(f'{self.crop_corr_folder}/*.vrt')[0])
        # Update input data information, with cropped data
        self.update_3D_set_info()

        self.repr_input_data_info()
        corr_set = self.get_correlation_set()

        pre_corr_tiles = self._get_set_data_tiles(os.path.basename(corr_set['pre_corr']))
        event1CorrTileList = self._get_set_data_tiles(os.path.basename(corr_set['post_corr1']))
        event2CorrTileList = self._get_set_data_tiles(os.path.basename(corr_set['post_corr2']))

        warpMat1TileList = self._get_set_data_tiles(os.path.basename(self.base_info.warpRaster))

        warpMat2TileList = self._get_set_data_tiles(os.path.basename(self.pre_event_info.warpRaster))
        warpMat3TileList = self._get_set_data_tiles(os.path.basename(self.post_event_1_info.warpRaster))
        warpMat4TileList = self._get_set_data_tiles(os.path.basename(self.post_event_2_info.warpRaster))

        demTileList = self._get_set_data_tiles(os.path.basename(dem_path_crp))
        lenList = [len(pre_corr_tiles), len(event1CorrTileList), len(event2CorrTileList),
                   len(warpMat1TileList),
                   len(warpMat2TileList), len(warpMat3TileList), len(warpMat4TileList), len(demTileList)]

        if all_equal(lenList) == False:
            raise ValueError("Raster don't have the same tile number")
        totalNb_Tiles = lenList[0]
        totalProccNb = multiprocessing.cpu_count()
        if debug:
            print("TotalProccNb=", totalProccNb)
        _3DDA_tile_folder = CreateDirectory(self._3DDA_folder, "3DDTiles", cal="n")

        preEventCorr = pre_corr_tiles
        eventCorr1 = event1CorrTileList
        eventCorr2 = event2CorrTileList
        demPath = demTileList
        tr1 = warpMat1TileList
        tr2 = warpMat2TileList
        tr3 = warpMat3TileList
        tr4 = warpMat4TileList
        filename_ancillary_master_pre = totalNb_Tiles * [self.base_info.rsmFile]
        filename_ancillary_slave_pre = totalNb_Tiles * [self.pre_event_info.rsmFile]
        filename_ancillary_post_1 = totalNb_Tiles * [self.post_event_1_info.rsmFile]
        filename_ancillary_post_2 = totalNb_Tiles * [self.post_event_2_info.rsmFile]
        output = []
        for i in range(totalNb_Tiles):
            output.append(os.path.join(_3DDA_tile_folder, "3DDisp_" + str(i + 1) + ".tif"))
        inversionFlags = totalNb_Tiles * [inversion_flag]

        if pmap:
            p_map(fun_compute3DD, preEventCorr, eventCorr1, eventCorr2, demPath, tr1, tr2, tr3, tr4,
                  filename_ancillary_master_pre, filename_ancillary_slave_pre, filename_ancillary_post_1,
                  filename_ancillary_post_2, output, inversionFlags, num_cpus=num_cpus)
        else:
            nbTile_ = 0
            for preEventCorr_, eventCorr1_, eventCorr2_, demPath_, tr1_, tr2_, tr3_, tr4_, filename_ancillary_master_pre_, \
                    filename_ancillary_slave_pre_, filename_ancillary_post_1_, filename_ancillary_post_2_, output_, \
                    inversionFlags_ in zip(preEventCorr, eventCorr1, eventCorr2,
                                           demPath, tr1, tr2, tr3, tr4, filename_ancillary_master_pre,
                                           filename_ancillary_slave_pre, filename_ancillary_post_1,
                                           filename_ancillary_post_2, output, inversionFlags):
                print("--- Tile :{} \ {}".format(nbTile_, totalNb_Tiles))
                if os.path.exists(output_) == False:
                    cCompute3DD(preEventCorr_, eventCorr1_, eventCorr2_, demPath_, tr1_, tr2_, tr3_, tr4_,
                                filename_ancillary_master_pre_,
                                filename_ancillary_slave_pre_, filename_ancillary_post_1_,
                                filename_ancillary_post_2_,
                                output_,
                                inversionFlags_)
                else:
                    print("--- Tile :{}  exist".format(nbTile_))
                nbTile_ += 1

        merge_3dd_tiles(_3DDA_tile_folder, os.path.dirname(_3DDA_tile_folder))

    def _ingest(self):
        return

    def set_data_info(self, data_df, ortho_path):
        img_info = ImgInfo()
        img_info.imgName = data_df.loc[data_df["Orthos"] == ortho_path, "Name"].values[0]
        img_info.orthoPath = ortho_path
        img_info.imgFolder = data_df.loc[data_df["Orthos"] == ortho_path, "ImgPath"].values[0]
        img_info.rsmFile = data_df.loc[data_df['Orthos'] == ortho_path, "RSM"].values[0]
        img_info.date = data_df.loc[data_df['Orthos'] == ortho_path, "Date"].values[0]
        img_info.warpRaster = data_df.loc[data_df['Orthos'] == ortho_path, "Trxs"].values[0]

        return img_info

    def set_inversion_flag(self, base_info, event_date: datetime.datetime):
        with open(base_info.rsmFile, "rb") as input:
            obj_ = pickle.load(input)
        base_info.date = obj_.date_time_obj
        delta = event_date - base_info.date
        inversion_flag = 'True' if delta.days < 0 else 'False'
        return inversion_flag

    def set_3DDA_processing_folders(self, o_set_comb_dir, cal):
        self._3DDA_folder = CreateDirectory(o_set_comb_dir, os.path.basename(o_set_comb_dir) + "_3DDA", cal=cal)
        self.rsm_folder = CreateDirectory(self._3DDA_folder, "RSM_files", cal=cal)
        self.trx_folder = CreateDirectory(self._3DDA_folder, "Trx", cal=cal)
        self.crop_corr_folder = CreateDirectory(self._3DDA_folder, "Corr", cal=cal)
        self.dem_folder = CreateDirectory(self._3DDA_folder, "rDEM", cal=cal)
        self.tile_folder = CreateDirectory(self._3DDA_folder, "Tiles", cal=cal)
        return

    def get_3DD_com_rsm(self, rsm_list, rsm_folder):

        for rsm_ in rsm_list:
            CopyFile(rsm_, rsm_folder, True)
        return

    @staticmethod
    def get_overlap_area(raster_list):
        from geoCosiCorr3D.georoutines.geo_utils import compute_rasters_overlap
        overlap_area = compute_rasters_overlap(raster_list)
        if overlap_area is None:
            raise ValueError('Invalid overlap')
        print(f'overlap area:{overlap_area}')
        coord = np.asarray(list(overlap_area.exterior.coords))

        roi_coord_wind = [min(coord[:, 0]), max(coord[:, 1]), max(coord[:, 0]), min(coord[:, 1])]

        return roi_coord_wind

    def crop_rasters(self, raster_list, o_folder, ):

        for raster_path in raster_list:
            crop_raster(input_raster=raster_path, roi_coord_wind=self.overlap_roi_utm, output_dir=o_folder, vrt=True)

        # # "-projwin 447225.5722201146 3949058.1522487984 458500.62972338597 3939574.181140272"
        # # "445736.44193266885 3948226.6834077216 453940.68389572675 3938979.290404687"
        # # SubsetRasters(rasterList=imgList,
        # #               areaCoord=[445736.44193266885, 3948226.6834077216, 453940.68389572675 ,3938979.290404687])
        return

    def get_corr_list(self):
        corr_params = self.corr_config['correlator_params']
        corr_suffix = f"{self.corr_config['correlator_name']}_wz_{corr_params['window_size'][0]}_step_{corr_params['step'][0]}.tif"
        corr_prefix = f'{Path(self.base_info.orthoPath).stem}_VS_{Path(self.pre_event_info.orthoPath).stem}_'
        corr_base_pre = os.path.join(self.corr_dir, f'{corr_prefix}{corr_suffix}')
        corr_prefix = f'{Path(self.base_info.orthoPath).stem}_VS_{Path(self.post_event_1_info.orthoPath).stem}_'
        corr_base_post1 = os.path.join(self.corr_dir, f'{corr_prefix}{corr_suffix}')
        corr_prefix = f'{Path(self.base_info.orthoPath).stem}_VS_{Path(self.post_event_2_info.orthoPath).stem}_'
        corr_base_post2 = os.path.join(self.corr_dir, f'{corr_prefix}{corr_suffix}')
        corr_list = [corr_base_pre, corr_base_post1, corr_base_post2]

        corr_status_ = all(flag == True for flag in [os.path.exists(corr_path) for corr_path in corr_list])
        if corr_status_:
            return corr_list
        else:
            raise FileNotFoundError('Correlations not found')

    def set_dem(self):

        dem_info = cRasterInfo(self.dem_path)
        if dem_info.epsg_code != self._3DD_map_epsg:
            import warnings
            msg = "Reproject DEM from {}-->{}".format(dem_info.epsg_code, self._3DD_map_epsg)
            warnings.warn(msg)
            self.dem_path = ReprojectRaster(self.dem_path, self._3DD_map_epsg)
        # TODO: check is we need UInt16 instead of FLOAT32 ?
        return crop_raster(self.dem_path, self.overlap_roi_utm, self.dem_folder, raster_type=gdal.GDT_UInt16)

    def tiling(self, in_folders: List, ref_img_path,
               ext_file_filter: Optional[List] = None):
        from geoCosiCorr3D.geo3DDA.tiling import TilingRaster

        if ext_file_filter is None:
            ext_file_filter = ["*.tif", "*.vrt"]
        img_path_list = []
        for folder_ in in_folders:
            file_list = get_files_based_on_extensions(folder_, filter_list=ext_file_filter)
            img_path_list.extend(file_list)
        print(img_path_list, len(img_path_list))

        self.tmp_tiles_folder = CreateDirectory(directoryPath=self.tile_folder, folderName="temp_tiles")

        raster_info = cRasterInfo(ref_img_path)
        raster_shape = (raster_info.raster_height, raster_info.raster_width)
        tile = TilingRaster(tile_sz=self.tile_sz)
        tiles_extent = tile.get_raster_tiles_extent(raster_shape)
        map_tiles_extent = tile.get_map_tiles_extent(tiles_extent, raster_info)

        for img_ in img_path_list:
            tile.create_geo_tiles(in_raster_path=img_, o_folder=self.tmp_tiles_folder,
                                  map_tiles_extent=map_tiles_extent, tile_file_suffix=TILE_FILE_SUFFIX,
                                  tile_folder_suffix=TILE_FOLDER_SUFFIX)

        return

    def repr_input_data_info(self):
        print(self.base_info.__repr__())
        print(self.pre_event_info.__repr__())
        print(self.post_event_1_info.__repr__())
        print(self.post_event_2_info.__repr__())
        return

    def update_3D_set_info(self):
        crp_trx_list = [os.path.basename(item) for item in
                        get_files_based_on_extensions(self.trx_folder, ["*.tif", "*.vrt"])]

        for trx_ in crp_trx_list:
            if self.base_info.imgName in Path(trx_).stem:
                self.base_info.warpRaster = os.path.join(self.trx_folder, trx_)
            if self.pre_event_info.imgName in Path(trx_).stem:
                self.pre_event_info.warpRaster = os.path.join(self.trx_folder, trx_)
            if self.post_event_1_info.imgName in Path(trx_).stem:
                self.post_event_1_info.warpRaster = os.path.join(self.trx_folder, trx_)
            if self.post_event_2_info.imgName in Path(trx_).stem:
                self.post_event_2_info.warpRaster = os.path.join(self.trx_folder, trx_)

        rsm_list = get_files_based_on_extensions(self.rsm_folder, ["*.pkl"])
        for rsm_ in rsm_list:
            if self.base_info.imgName in Path(rsm_).stem:
                self.base_info.rsmFile = rsm_
            if self.pre_event_info.imgName in Path(rsm_).stem:
                self.pre_event_info.rsmFile = rsm_
            if self.post_event_1_info.imgName in Path(rsm_).stem:
                self.post_event_1_info.rsmFile = rsm_
            if self.post_event_2_info.imgName in Path(rsm_).stem:
                self.post_event_2_info.rsmFile = rsm_
        return

    def get_correlation_set(self):

        corr_set = {'pre_corr': None, 'post_corr1': None, 'post_corr2': None}
        for corr_ in get_files_based_on_extensions(self.crop_corr_folder, ["*.tif", "*.vrt"]):
            if self.pre_event_info.imgName in Path(corr_).stem.split("_VS_")[1]:
                corr_set['pre_corr'] = corr_
            if self.post_event_1_info.imgName in Path(corr_).stem.split("_VS_")[1]:
                corr_set['post_corr1'] = corr_
            if self.post_event_2_info.imgName in Path(corr_).stem.split("_VS_")[1]:
                corr_set['post_corr2'] = corr_

        if None in corr_set.values():
            raise ValueError('Enable to get correlation set')
        else:
            return corr_set

    def _get_set_data_tiles(self, arg):
        return get_files_based_on_extensions(
            os.path.join(self.tmp_tiles_folder, f'{Path(arg).stem}{TILE_FOLDER_SUFFIX}'), ["*.tif", "*.vrt"])


if __name__ == '__main__':
    merge_3dd_tiles(
        tiles_dir='/home/cosicorr/0-WorkSpace/3-PycharmProjects/GEO_COSI_CORR_3D_WD/GeoCosiCorr3DPipeline/o3DDA/3DDA_Set_1/Set_1_Comb1/Set_1_Comb1_3DDA/3DDTiles',
        o_dir='/home/cosicorr/0-WorkSpace/3-PycharmProjects/GEO_COSI_CORR_3D_WD/GeoCosiCorr3DPipeline/o3DDA/3DDA_Set_1/Set_1_Comb1/Set_1_Comb1_3DDA')
    # tile_path = '/home/cosicorr/0-WorkSpace/3-PycharmProjects/GEO_COSI_CORR_3D_WD/GeoCosiCorr3DPipeline/o3DDA/3DDA_Set_1/Set_1_Comb1/Set_1_Comb1_3DDA/3DDTiles/3DDisp_1.tif'
    # raster_info = cRasterInfo(tile_path)
    # import rasterio
    # with rasterio.open(tile_path) as src:
    #     print(src.descriptions)
