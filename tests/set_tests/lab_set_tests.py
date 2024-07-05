import os

import geoCosiCorr3D.geoCore.constants as C
import numpy as np

from geoCosiCorr3D.geoCosiCorr3dLogger import GeoCosiCorr3DLog

GeoCosiCorr3DLog("set_tests")

folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset/test_ortho_dataset")
test_raw_img_fn = os.path.join(folder, "RAW_SP2.TIF")
test_dem_fn = os.path.join(folder, "DEM.tif")
test_rfm_fn = os.path.join(folder, "SP2_RPC.txt")
dmpFile = os.path.join(folder, "SP2_METADATA.DIM")


def correlation():
    from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate
    folder = os.path.join(C.SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
    img1 = os.path.join(folder, "BASE_IMG.TIF")
    img2 = os.path.join(folder, "TARGET_IMG.TIF")
    corr_obj = Correlate(base_image_path=img1,
                         target_image_path=img2,
                         # output_corr_path=tmp_dir,
                         corr_config=C.TEST_CONFIG.FREQ_CORR_CONFIG,
                         corr_show=False)
    return


def ortho():
    from geoCosiCorr3D.geoOrthoResampling.geoOrtho import orthorectify

    output_ortho_path = "temp_rfm_ortho.tif"
    ortho_params = {
        "method":
            {"method_type": C.SATELLITE_MODELS.RFM, "metadata": test_rfm_fn, "corr_model": None,
             },
        "GSD": 20,
        "resampling_method": C.Resampling_Methods.SINC}
    orthorectify(input_l1a_path=test_raw_img_fn,
                 output_ortho_path=output_ortho_path,
                 ortho_params=ortho_params,
                 dem_path=test_dem_fn,
                 debug=True)

    ortho_params = {
        "method":
            {"method_type": C.SATELLITE_MODELS.RSM, "metadata": dmpFile, "corr_model": None,
             "sensor": C.SENSOR.SPOT1_5,
             },
        "GSD": 20,
        "resampling_method": C.Resampling_Methods.SINC}
    output_ortho_path = "temp_rsm_ortho.tif"
    orthorectify(input_l1a_path=test_raw_img_fn,
                 output_ortho_path=output_ortho_path,
                 ortho_params=ortho_params,
                 dem_path=test_dem_fn,
                 debug=True)
    return


def transformer():
    from geoCosiCorr3D.geoRSM.Pixel2GroundDirectModel import cPix2GroundDirectModel
    from geoCosiCorr3D.geoCore.core_RSM import RSM
    test_rfm_fn = '/home/saif/PycharmProjects/Geospatial-COSICorr3D/tests/test_dataset/test_ortho_dataset/test_psscene_basic_analytic_udm2/PSScene/20240222_084309_00_241c_1B_AnalyticMS_RPC.TXT'

    debug = True
    geoCoordList = []
    cols = [0, 5999, 0, 5999, 3000, 3001]
    lins = [0, 0, 5999, 5999, 3000, 3001]

    print('_____________________________________________________________________')
    # model_data = RSM.build_RSM(metadata_file=dmpFile, sensor_name=C.SENSOR.SPOT1_5, debug=debug)
    # for xVal, yVal in zip(cols, lins):
    #     pix2Ground_obj = cPix2GroundDirectModel(rsmModel=model_data,
    #                                             xPix=xVal,
    #                                             yPix=yVal,
    #                                             rsmCorrectionArray=None,
    #                                             demFile=test_dem_fn)
    #     geoCoordList.append(pix2Ground_obj.geoCoords)
    # coords = np.asarray(geoCoordList)
    # print(coords)
    print('_____________________________________________________________________')
    from geoCosiCorr3D.geoRFM.RFM import RFM
    import geoCosiCorr3D.georoutines.geo_utils as geoRT
    model_data = RFM(test_rfm_fn, dem_fn=test_dem_fn, debug=True)

    # # TODO use getGSD from RFM class

    lonBBox_RFM, latBBox_RFM, altBBox_RFM = model_data.i2g(col=cols,
                                                           lin=lins,
                                                           )
    rfm_coords = np.array([lonBBox_RFM, latBBox_RFM, altBBox_RFM]).T
    print('rfm_coords:\n', rfm_coords)
    # # print(lonBBox_RFM, latBBox_RFM, altBBox_RFM)
    # lons = [30.52895296, 31.44808898, 30.32253163, 31.22086155, 30.87121092, 30.87136232]
    # lats = [41.24090926, 41.05062958, 40.72244546, 40.53710197, 40.89057797, 40.89045288]
    # alts = [1691.49774714, 367.12418755, 30., 918.10812233, 550.42556207, 544.97850478]
    #
    # res = model_data.g2i(lons, lats, alts)
    # print(np.asarray(res).T)
    #
    # res_2 = model_data_2.g2i(lons, lats)#, alts)
    # print(np.asarray(res_2).T)

    # TODO add to unit/functional tests
    # img = '/home/cosicorr/0-WorkSpace/3-PycharmProjects/geoCosiCorr3D/geoCosiCorr3D/Tests/3-geoOrtho_Test/Sample/Sample1/SPOT2.TIF'
    # rfm = RFM(test_rfm_fn, debug=True)
    print(f'attitude range:{model_data.get_altitude_range()}')
    print(f'GSD:{model_data.get_gsd()}')
    print(f'geoTransform:{model_data.get_geotransform()}')
    print(f'fp:{model_data.get_footprint()}')
    # return


def rfm_transformer():
    from geoCosiCorr3D.geoRFM.RFM import RFM
    cols = [0, 5999, 0, 5999, 3000, 3001]
    lins = [0, 0, 5999, 5999, 3000, 3001]

    model_data = RFM(test_rfm_fn, dem_fn=test_dem_fn, debug=True)

    # # TODO use getGSD from RFM class

    lonBBox_RFM, latBBox_RFM, altBBox_RFM = model_data.i2g(col=cols,
                                                           lin=lins,
                                                           )
    rfm_coords = np.array([lonBBox_RFM, latBBox_RFM, altBBox_RFM]).T
    print('rfm_coords:\n', rfm_coords)

    res = model_data.g2i(rfm_coords[:, 0], rfm_coords[:, 1], rfm_coords[:, 2])
    print(np.asarray(res).T)

    res = model_data.g2i(rfm_coords[:, 0], rfm_coords[:, 1])
    print(np.asarray(res).T)

    # TODO add to unit/functional tests

    print(f'attitude range:{model_data.get_altitude_range()}')
    print(f'GSD:{model_data.get_gsd()}')
    print(f'geoTransform:{model_data.get_geotransform()}')
    print(f'fp:{model_data.get_footprint()}')
    return


def set_grid():
    from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import SatMapGrid
    import geoCosiCorr3D.georoutines.geo_utils as geoRT
    from geoCosiCorr3D.geoRFM.RFM import RFM
    from geoCosiCorr3D.geoCore.core_RSM import RSM

    o_res = 20
    model = RFM(test_rfm_fn, debug=True)
    # model = RSM.build_RSM(metadata_file=dmpFile, sensor_name=C.SENSOR.SPOT1_5,
    #                       debug=True)

    l1a_raster_info = geoRT.cRasterInfo(test_raw_img_fn)
    ortho_grid = SatMapGrid(raster_info=l1a_raster_info,
                            model_data=model,
                            model_type=C.SATELLITE_MODELS.RFM,
                            dem_fn=test_dem_fn,
                            new_res=o_res,
                            corr_model=np.zeros((3, 3)),
                            debug=True)
    print(ortho_grid.grid_fp())

    return


if __name__ == '__main__':
    # correlation()
    # ortho()
    rfm_transformer()
    # set_grid()
