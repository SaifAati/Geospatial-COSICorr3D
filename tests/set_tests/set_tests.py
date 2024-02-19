import os

import geoCosiCorr3D.geoCore.constants as const
import numpy as np

from geoCosiCorr3D.geoCosiCorr3dLogger import geoCosiCorr3DLog

geoCosiCorr3DLog("set_tests")

folder = os.path.join(const.SOFTWARE.PARENT_FOLDER, "tests/test_dataset/test_ortho_dataset")
test_raw_img_fn = os.path.join(folder, "RAW_SP2.TIF")
test_dem_fn = os.path.join(folder, "DEM.tif")
test_rfm_fn = os.path.join(folder, "SP2_RPC.txt")
dmpFile = os.path.join(folder, "SP2_METADATA.DIM")


def correlation():
    from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate
    folder = os.path.join(const.SOFTWARE.PARENT_FOLDER, "tests/test_dataset")
    img1 = os.path.join(folder, "BASE_IMG.TIF")
    img2 = os.path.join(folder, "TARGET_IMG.TIF")
    corr_obj = Correlate(base_image_path=img1,
                         target_image_path=img2,
                         # output_corr_path=tmp_dir,
                         corr_config=const.TEST_CONFIG.FREQ_CORR_CONFIG,
                         corr_show=False)
    return


def ortho():
    from geoCosiCorr3D.geoOrthoResampling.geoOrtho import orthorectify

    output_ortho_path = "temp_rfm_ortho.tif"
    ortho_params = {
        "method":
            {"method_type": const.SATELLITE_MODELS.RFM, "metadata": test_rfm_fn, "corr_model": None,
             },
        "GSD": 20,
        "resampling_method": const.Resampling_Methods.SINC}
    orthorectify(input_l1a_path=test_raw_img_fn,
                 output_ortho_path=output_ortho_path,
                 ortho_params=ortho_params,
                 dem_path=test_dem_fn,
                 debug=True)

    ortho_params = {
        "method":
            {"method_type": const.SATELLITE_MODELS.RSM, "metadata": dmpFile, "corr_model": None,
             "sensor": const.SENSOR.SPOT1_5,
             },
        "GSD": 20,
        "resampling_method": const.Resampling_Methods.SINC}
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

    debug = True
    geoCoordList = []
    cols = [0, 5999, 0, 5999, 3000, 3001]
    lins = [0, 0, 5999, 5999, 3000, 3001]
    model_data = RSM.build_RSM(metadata_file=dmpFile, sensor_name=const.SENSOR.SPOT1_5, debug=debug)
    for xVal, yVal in zip(cols, lins):
        pix2Ground_obj = cPix2GroundDirectModel(rsmModel=model_data,
                                                xPix=xVal,
                                                yPix=yVal,
                                                rsmCorrectionArray=None,
                                                demFile=test_dem_fn)
        geoCoordList.append(pix2Ground_obj.geoCoords)
    coords = np.asarray(geoCoordList)
    print(coords)

    from geoCosiCorr3D.geoRFM.RFM import RFM
    import geoCosiCorr3D.georoutines.geo_utils as geoRT
    model_data = RFM(test_rfm_fn)
    dem_info = geoRT.cRasterInfo(test_dem_fn)
    # TODO use getGSD from RFM class
    lonBBox_RFM, latBBox_RFM, altBBox_RFM = model_data.Img2Ground_RFM(col=cols,
                                                                      lin=lins,
                                                                      demInfo=dem_info,
                                                                      corrModel=None)
    rfm_coords = np.array([lonBBox_RFM, latBBox_RFM, altBBox_RFM]).T
    print('rfm_coords:\n', rfm_coords)
    return


def set_grid():
    from geoCosiCorr3D.geoOrthoResampling.geoOrthoGrid import SatMapGrid
    import geoCosiCorr3D.georoutines.geo_utils as geoRT
    from geoCosiCorr3D.geoRFM.RFM import RFM

    o_res = 20
    rfm_model = RFM(test_rfm_fn, debug=True)

    l1a_raster_info = geoRT.cRasterInfo(test_raw_img_fn)
    ortho_grid = SatMapGrid(raster_info=l1a_raster_info,
                            model_data=rfm_model,
                            model_type=const.SATELLITE_MODELS.RFM,
                            dem_fn=test_dem_fn,
                            new_res=o_res,
                            corr_model=np.zeros((3, 3)),
                            debug=True)
    # print(ortho_grid)
    print(ortho_grid.grid_fp())

    return


if __name__ == '__main__':
    # correlation()
    ortho()
    # transformer()
    # set_grid()
    pass
