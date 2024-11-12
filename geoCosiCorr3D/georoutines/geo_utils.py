"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""
import matplotlib.pyplot as plt
import rasterio
import rasterio.crs
import numpy as np
import warnings
import os
import pyproj
import logging
from shapely.geometry import Polygon
from osgeo import osr, gdal
from typing import Any, List, Optional, Union
from astropy.time import Time
from scipy.stats import norm
from scipy import stats

from pathlib import Path
from geoCosiCorr3D.geoCore.base.base_georoutines import BaseRasterInfo
from geoCosiCorr3D.geoCore.constants import WRITERASTER, SOFTWARE

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class cRasterInfo(BaseRasterInfo):
    def __init__(self, input_raster_path: str):
        super().__init__(input_raster_path)
        self.raster = rasterio.open(self.get_raster_path)
        self.raster_width, self.raster_height = self.raster.width, self.raster.height
        self.band_number = self.raster.count
        self.raster_type = self.raster.dtypes
        self.no_data = self.raster.nodata  # src.nodatavals
        self.bands_description = self.raster.descriptions
        try:
            self.valid_map_info = True
            self.proj = self.raster.crs.wkt
            # self.epsg_code = int(str(self.raster.crs).split(":")[1])
            self.epsg_code = self.raster.crs.to_epsg()
        except:
            self.valid_map_info = False
            self.proj = None
            self.epsg_code = None
        self.pixel_width, self.pixel_height = self.raster.res
        self.x_map_origin, self.y_map_origin = self.raster.transform * (0, 0)

        self.geo_transform_affine = self.raster.transform

        self.geo_transform = [self.geo_transform_affine[2],
                              self.geo_transform_affine[0],
                              self.geo_transform_affine[1],
                              self.geo_transform_affine[5],
                              self.geo_transform_affine[3],
                              self.geo_transform_affine[4]]
        self.rpcs = self.raster.tags(ns='RPC')
        self.bbox_map = self.raster.bounds
        # self.raster_array = self.raster.read()  # all bands
        self._raster_array = None  # Initialize a private attribute to None to return all bands
        # self.raster = None

    @property
    def raster_array(self):
        if self._raster_array is None:  # Check if the raster data has been loaded
            self._raster_array = self.raster.read()  # Read & store the data
        return self._raster_array

    def image_as_array_subset(self,
                              col_off_min: int,
                              col_off_max: int,
                              row_off_min: int,
                              row_off_max: int,
                              band_number: Optional[int] = 1):
        """

        Args:

        Returns:
        References: https://gdal.org/python/osgeo.gdal-pysrc.html#Band.ReadAsArray
        https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        """
        from rasterio.windows import Window

        # raster = self.raster  # = rasterio.open(self.get_raster_path)
        width = (col_off_max - col_off_min) + 1
        height = (row_off_max - row_off_min) + 1
        # Note: This operation is performed each time the method is called and will allocate memory
        # for the subset array that is being read
        # raster = None
        return self.raster.read(band_number,
                                window=Window(col_off=col_off_min,
                                              row_off=row_off_min,
                                              width=width,
                                              height=height))

    def image_as_array(self, band: Optional[int] = 1, read_masked=False):
        # raster = rasterio.open(self.get_raster_path)
        return self.raster.read(band, masked=read_masked)

    @staticmethod
    def write_raster(output_raster_path,
                     array_list: List[Any],
                     geo_transform: Union[List[float], rasterio.Affine] = None,
                     epsg_code=None,
                     dtype: str = "uint16",
                     descriptions: List[str] = None,
                     compress: str = WRITERASTER.COMPRESS,
                     no_data=None):
        """
        Notes:
            geo_transform = [x-origin, x-res,0, y-origin,0,-y-res,]
            geo_transform_affine = [x-res,0,x-origin] [0, -y-res,y-origin] [0 , 0 , 1]

        """
        meta = {'driver': WRITERASTER.DRIVER,
                'dtype': dtype,  # 'uint16',
                'nodata': no_data,
                'width': array_list[0].shape[1],
                'height': array_list[0].shape[0],
                'count': len(array_list),
                'compress': compress,
                # 'blockxsize': 256,
                # 'blockysize': 256,

                'BIGTIFF': 'YES'
                }
        if geo_transform is None:
            geo_transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        if isinstance(geo_transform, rasterio.Affine):
            meta['transform'] = geo_transform
        else:

            meta['transform'] = rasterio.Affine(geo_transform[1], geo_transform[2], geo_transform[0]
                                                , geo_transform[4], geo_transform[5], geo_transform[3])
        if epsg_code is not None:
            meta['crs'] = epsg_code  # CRS.from_epsg(32647),']

        with rasterio.open(output_raster_path, "w", **meta) as dest:
            for id in range(len(array_list)):
                dest.write(array_list[id], id + 1)
                if descriptions is not None and len(descriptions) == len(array_list):
                    dest.set_band_description(id + 1, descriptions[id])

            dest.update_tags(Author=SOFTWARE.AUTHOR,
                             Software=SOFTWARE.SOFTWARE_NAME,
                             GenerationTime=Time(Time.now(), format='isot', scale='utc'))
        return

    @staticmethod
    def normalize(array, type=255):
        """
        # Function to normalize the grid values
        Normalizes numpy arrays into scale 0.0 - 1.0
        Args:
            array:
            type 255 or 1

        Returns:

        """
        if type == 255:
            return (array * (255 / np.max(array))).astype(np.uint8)
        if type == 1:
            array_min, array_max = array.min(), array.max()
            return ((array - array_min) / (array_max - array_min))

    @staticmethod
    def create_rgb_natural_color_composite(red_n_array, green_n_array, blue_n_array):
        """
        # Create RGB natural color composite
        Args:
            red_n_array:
            green_n_array:
            blue_n_array:

        Returns:

        """
        # USING OPENCV cv2.merge([red_array, green_array, blue_Array])
        return np.dstack((red_n_array, green_n_array, blue_n_array))

    @staticmethod
    def raster_histogram_equalization(in_array, method: Optional[str] = "contrast_stretch"):
        """

        Args:
            in_array:
            method:

        Returns:
        Notes:
            https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
        """
        from skimage import exposure
        if method == "contrast_stretch":
            # Contrast stretching
            p2, p98 = np.percentile(in_array, (2, 98))
            return exposure.rescale_intensity(in_array, in_range=(p2, p98))
        if method == "equalization":
            # Equalization
            return exposure.equalize_hist(in_array)
        if method == "adaptative_equalization":
            # Adaptive Equalization
            return exposure.equalize_adapthist(in_array, clip_limit=0.3)

        return

    @staticmethod
    def generate_raster_preview(input_raster_path: str, scale: float = 1, output_fig_path: str = None,
                                band_nb_list: List = None, dpi: int = 400, normalization_type: int = 255):
        """

        Args:
            input_raster_path:
            scale:
            output_fig_path:
            band_nb_list:

        Returns:
        Notes:
            we don't check if all bands exists
            The normalization is hard coded to 1.

        """
        from skimage.transform import rescale
        logging.info('Raster preview generation')
        if band_nb_list is None:
            band_nb_list = [1]

        raster_info = cRasterInfo(input_raster_path)
        raster_array = raster_info.raster_array
        band_array_list = []
        logging.info(f'Recalling img --> factor:{scale}')
        for index, band in enumerate(band_nb_list):
            band_array = raster_array[band - 1, :, :]
            band_array = raster_info.normalize(band_array, type=normalization_type)
            band_array = raster_info.raster_histogram_equalization(band_array)
            band_array = rescale(band_array, scale)

            band_array_list.append(band_array)
        preview_img_array = np.dstack(tuple(band_array_list))

        preview_img_array_ = np.ma.masked_where(preview_img_array == 0, preview_img_array)
        logging.info(f'Scaled preview raster shape:{preview_img_array.shape}')
        if preview_img_array_.shape[2] < 3:
            plt.imshow(preview_img_array_[:, :, 0], cmap='gray')
        elif preview_img_array_.shape[2] > 3:
            plt.imshow(preview_img_array_[:, :, 0:3])
        else:
            plt.imshow(preview_img_array_, interpolation='none')
        plt.axis('off')
        if output_fig_path is None:
            output_fig_path = os.path.join(os.path.dirname(input_raster_path), f'{Path(input_raster_path).stem}.png')

        plt.savefig(output_fig_path, bbox_inches='tight', dpi=dpi)
        logging.info(f'preview saved: {output_fig_path}')
        return

    def Map2Pixel(self, x, y):
        """
        Convert coordinate from map space to image space
        Args:
            x: xMap coordinate : int or float
            y: yMap coordinate: int or float

        Returns: coordinate in image space : tuple in pix
            in case of WGS84: x=lon,y=lat --> x,y <--> col,lin
        """

        ## Apply inverse affine transformation
        rtnX = self.geo_transform[2]
        rtnY = self.geo_transform[4]
        ## Apply affine transformation
        mat = np.array([[self.pixel_width, rtnX], [rtnY, -self.pixel_height]])
        trans = np.array([[self.x_map_origin, self.y_map_origin]])

        temp = np.array([[x, y]]).T - trans.T
        res = np.dot(np.linalg.inv(mat), temp)
        xPx = res[0].item()
        yPx = res[1].item()

        if xPx < 0 or xPx > self.raster_width:
            warnings.warn("xPix outside the image dimension", DeprecationWarning, stacklevel=2)
        if yPx < 0 or yPx > self.raster_height:
            warnings.warn("yPix outside the image dimension", DeprecationWarning, stacklevel=2)
        # NOTE using RASTERIO
        # print(rasterio.transform.rowcol(info_rasterio.transform, map_coord[0], map_coord[1])) --> we dont have the sub-pixel info
        return (xPx, yPx)

    def Map2Pixel_Batch(self, X: List, Y: List):
        """
        Convert coordinate from map space to image space
        Args:
            X: list of xMap coordinate : list of int or float
            Y: list of yMap coordinate: list of int or float

        Returns:
            coordinate in image space (X_pix, Y_pix) : tuple in pix

        """
        X_pix = []
        Y_pix = []
        for x, y in zip(X, Y):
            xPix, yPix = self.Map2Pixel(x=x, y=y)
            X_pix.append(xPix)
            Y_pix.append(yPix)
        return (X_pix, Y_pix)

    def Pixel2Map(self, cols, rows):
        """
        Convert pixel coordinate to map coordinate,
        Notes:
            The top Left coordinate of the image with GDAl correspond to (0,0)pix
        Args:
            cols:  : int or float
            rows:  int or float

        Returns: xMap,yMap : tuple  (non integer coordinates)


        rows : int or sequence of ints
            Pixel rows.
        cols : int or sequence of ints
            Pixel columns.
        offset : str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner.
        """

        # rtnX = self.geo_transform[2]
        # rtnY = self.geo_transform[4]
        # ## Apply affine transformation
        # mat = np.array([[self.pixel_width, rtnX], [rtnY, self.pixel_height]])
        # trans = np.array([[self.x_origin, self.y_origin]])
        # res = np.dot(mat, np.array([[x, y]]).T) + trans.T
        # xMap = res[0].item()
        # yMap = res[1].item()
        raster = rasterio.open(self.input_raster_path)

        xMap, yMap = rasterio.transform.xy(raster.transform, rows, cols)
        raster = None
        return (xMap, yMap)

    def raster_dims(self):

        ds = rasterio.open(self.get_raster_path)
        bounds = ds.bounds
        raster_map_dims = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        x0, y0 = self.Map2Pixel(x=bounds.left, y=bounds.bottom)
        xf, yf = self.Map2Pixel(x=bounds.right, y=bounds.top)
        # imgDimsPix = {"x0Pix": int(x0), "xfPix": int(xf), "y0Pix": int(y0), "yfPix": int(yf)}
        raster_pix_dims = [int(x0), int(xf), int(y0), int(yf)]
        # return list(imgDimsMap.values()), list(imgDimsPix.values())
        return raster_pix_dims, raster_map_dims

    def __repr__(self):
        return f"Raster Path: {self.get_raster_path} \n" \
                f"Raster Dimensions: {self.raster_width} x {self.raster_height} \n" \
                f"Raster Bands: {self.band_number} \n" \
                f"Raster Type: {self.raster_type} \n" \
                f"Raster Projection: {self.proj} \n" \
                f"Raster EPSG: {self.epsg_code} \n" \
                f"Raster Pixel Size: {self.pixel_width} x {self.pixel_height} \n" \
                f"Raster Origin: {self.x_map_origin} x {self.y_map_origin} \n" \
                f"Raster GeoTransform: {self.geo_transform} \n" \
                f"Raster RPCs: {self.rpcs} \n" \
                f"Raster Bounding Box: {self.bbox_map} \n" \
                f"Raster NoData: {self.no_data} \n" \
                f"Raster Bands Description: {self.bands_description} \n"\
                "***********************************************************\n"



class cRasterInfoGDAL:
    def __init__(self, input_raster_path: str):
        self.raster = None
        self.input_raster_path = input_raster_path
        self.raster = gdal.Open(self.input_raster_path)
        self.geo_transform = self.raster.GetGeoTransform()
        self.raster_width, self.raster_height = self.raster.RasterXSize, self.raster.RasterYSize
        self.x_map_origin, self.y_map_origin = self.geo_transform[0], self.geo_transform[3]
        self.pixel_width, self.pixel_height = self.geo_transform[1], self.geo_transform[5]
        self.nb_bands = self.raster.RasterCount
        try:
            self.valid_map_info = True
            projection = self.raster.GetProjection()
            self.proj = osr.SpatialReference(wkt=self.raster.GetProjection())
            self.epsg_code = self.proj.GetAttrValue('AUTHORITY', 1)
        except:
            self.valid_map_info = False
            self.proj = None
            self.epsg_code = None

        # self.geo_transform_affine = self.raster.transform
        #
        self.rpcs = self.raster.GetMetadata('RPC')
        # self.bbox_map = self.raster.bounds
        self.raster_array = self.image_as_array()
        self.raster = None

    @staticmethod
    def image_as_array_subset(input_raster_path: str,
                              col_off_min: int,
                              col_off_max: int,
                              row_off_min: int,
                              row_off_max: int,
                              band_number: Optional[int] = 1):

        width = (col_off_max - col_off_min) + 1
        height = (row_off_max - row_off_min) + 1
        raster = gdal.Open(input_raster_path)
        array = np.array(raster.GetRasterBand(band_number).ReadAsArray(int(col_off_min),
                                                                       int(row_off_min),
                                                                       int(width),
                                                                       int(height)))
        raster = None
        return array

    def image_as_array(self, band: Optional[int] = 1):

        return np.array(self.raster.GetRasterBand(band).ReadAsArray())

    def __repr__(self):
        pass


def WriteRaster(oRasterPath,
                geoTransform,
                arrayList,
                epsg=4326,
                dtype=gdal.GDT_Float32,
                metaData: Optional[List] = None,
                resample_alg=gdal.GRA_Lanczos,
                descriptions=None,
                noData=None,
                progress=False,
                driver='GTiff'):
    """
    Returns:
    Notes:

        https://gdal.org/python/osgeo.gdalconst-module.html
        geoTransfrom it's an affine transformation
        geoTransform = originX, pixelWidth, rtx, originY,rty, pixelHeight
    """
    global outband
    driver = gdal.GetDriverByName(driver)
    rows, cols = np.shape(arrayList[0])
    outRaster = driver.Create(oRasterPath, cols, rows, len(arrayList), dtype,
                              options=["TILED=YES", "BIGTIFF=YES", "COMPRESS=LZW"])
    outRaster.SetGeoTransform((geoTransform[0], geoTransform[1], geoTransform[2], geoTransform[3], geoTransform[4],
                               geoTransform[5]))
    if epsg is not None:
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(epsg)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())

    outRaster.SetMetadataItem("Author", "SAIF AATI saif@caltech.edu")

    # Set the metadata
    metaData_ = []
    if metaData is not None:
        if isinstance(metaData, dict):
            for key, value in metaData.items():
                temp = [key, value]
                metaData_.append(temp)
        elif isinstance(metaData, list):
            metaData_ = metaData

        for mm in metaData_:
            if not isinstance(mm[1], dict):
                if not isinstance(mm[1], str):
                    str_ = str(mm[1])
                    outRaster.SetMetadataItem(mm[0], str_)
                else:
                    outRaster.SetMetadataItem(mm[0], mm[1])
    ## Write the data
    for i in range(len(arrayList)):
        outband = outRaster.GetRasterBand(i + 1)
        outband.WriteArray(arrayList[i], resample_alg=resample_alg)
        if noData != None:
            if progress:
                print("No data=", noData)
            outband.SetNoDataValue(noData)
        if descriptions is not None:
            outband.SetDescription(descriptions[i])
            # outBand.SetRasterCategoryNames(descriptions[i])
        if progress:
            print("Writing band number: ", i + 1, " ", i + 1, "/", len(arrayList))

    outband.FlushCache()
    outRaster = None
    return oRasterPath


def ComputeEpsg(lon, lat):
    """
    Compute the EPSG code of the UTM zone which contains
    the point with given longitude and latitude

    Args:
        lon : longitude
        lat : latitude

    Returns:
        EPSG code
    Notes:
        UTM zone number starts from 1 at longitude -180, and increments by 1 every 6 degrees of longitude

        EPSG = CONST + ZONE where CONST is
        - 32600 for positive latitudes
        - 32700 for negative latitudes
    """

    zone = int((lon + 180) // 6 + 1)
    const = 32600 if lat > 0 else 32700
    return const + zone


def ConvCoordMap1ToMap2(x, y, targetEPSG, z: Optional[float] = None, sourceEPSG=4326, display=False):
    """
    convert point coordinates from source to target system
    Args:
        x:  map coordinate (e.g lon,lat)
        y:
        targetEPSG: target coordinate system of the point; integer
        z:
        sourceEPSG:  source coordinate system of the point; integer (default geographic coordinate )
        display:

    Returns:
        point in target coordinate system; list =[xCoord,yCoord,zCoord]
    Notes:
        - If the transformation from WGS to UTM, x= lat, y=lon ==> coord =(easting(xMap) ,northing(yMap))

    """

    ## Set the source system
    source = osr.SpatialReference()  # instance from SpatialReference Class
    source.ImportFromEPSG(int(sourceEPSG))  # create a projection system based on EPSG code

    ## Set the target system
    target = osr.SpatialReference()
    target.ImportFromEPSG(int(targetEPSG))

    transform = osr.CoordinateTransformation(source,
                                             target)  # instance of the class Coordinate Transformation
    if z is None:
        coord = transform.TransformPoint(x, y)

    else:
        coord = transform.TransformPoint(x, y, z)
    if display:
        print("{}, {}, EPSG={} ----> Easting:{}  , Northing:{} , EPSG={}".format(x, y, sourceEPSG, coord[0], coord[1],
                                                                                 targetEPSG))
    return coord


def ConvCoordMap1ToMap2_Batch(X, Y, targetEPSG, Z: Optional[List] = None, sourceEPSG=4326):
    """
    Convert point coordinates from source to target system

    Args:
        X:map coordinate (e.g lon,lat)
        Y:map coordinate (e.g lon,lat)
        targetEPSG:target coordinate system of the point: integer
        Z:map coordinates
        sourceEPSG:source coordinate system of the point: integer (default geographic coordinate )

    Returns:         point in target coordinate system; list =[xCoord,yCoord,zCoord] or ([lats],[lons])

    Notes:
        - if the transformation from WGS to UTM, x= lat, y=lon ==> coord =(easting(xMap) ,northing(yMap))
        - if the transformation from UTM to WGS 84, x=easting, y=Notthing ==> lat, long

    """
    if Z is None:
        Z: List = []
    sourceEPSG_string = "epsg:" + str(sourceEPSG)
    targetEPSG_string = "epsg:" + str(targetEPSG)
    transformer = pyproj.Transformer.from_crs(sourceEPSG_string, targetEPSG_string)
    if len(Z) == 0:
        return transformer.transform(X, Y)

    else:
        return transformer.transform(X, Y, Z)


def ReprojectRaster(input_raster_path, o_prj, vrt: bool = True, output_raster_path: Optional[str] = None):
    """
    Reproject a raster to a new projection system


    """
    # TODO: use RASTERIO instead of gdal
    if output_raster_path is None:
        output_raster_path = os.path.join(os.path.dirname(input_raster_path),
                                          Path(input_raster_path).stem + "_" + str(o_prj) + ".tif")
        if vrt:
            output_raster_path = os.path.join(os.path.dirname(input_raster_path),
                                              Path(input_raster_path).stem + "_" + str(o_prj) + ".vrt")
    # print(oRasterPath)
    warpOptions = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs epsg:" + str(o_prj)))
    # gdal.Warp(oRasterPath, iRasterPath, dstSRS='EPSG:'+str(oPrj))#options=warpOptions)
    gdal.Warp(output_raster_path, input_raster_path, options=warpOptions)

    return output_raster_path


class Convert:
    @staticmethod
    def cartesian_2_geo(x, y, z):
        """

        Args:
            x:
            y:
            z:

        Returns:
        Notes:
            # ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        # # print(ecef)
        # lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        # # print(lla)
        # # transformer = pyproj.Transformer.from_crs(lla, ecef)
        # x, y, z = pyproj.transform(lla, ecef, lon, lat, alt, radians=False)
        # print( [x,y,z])
        https://pyproj4.github.io/pyproj/dev/api/proj.html
        """

        transproj = pyproj.Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, "EPSG:4326",
                                                always_xy=True)
        lon, lat, alt = transproj.transform(x, y, z, radians=False)
        return [lon, lat, alt]

    @staticmethod
    def cartesian_2_geo_batch(X, Y, Z):
        transproj = pyproj.Transformer.from_crs({"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, "EPSG:4326",
                                                always_xy=True)
        Lon, Lat, Alt = transproj.transform(X, Y, Z, radians=False)
        return Lon, Lat, Alt

    @staticmethod
    def geo_2_cartesian(Lon, Lat, Alt, method="pyprj"):
        """

        Args:
            Lon: list []
            Lat: list []
            Alt: list  []
            method: pyprj, custom

        Returns: X_cart, y_cart, Z_cart

        Notes:
            https://pyproj4.github.io/pyproj/dev/api/proj.html
        """
        # TODO: the conversion is performed using pyproj. House implementation could be used.
        ## (see IDL version: convert_geographic_to_cartesian)
        if method == "pyprj":
            transproj = pyproj.Transformer.from_crs("EPSG:4326",
                                                    {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
                                                    always_xy=True)
            X_cart, Y_cat, Z_cart = transproj.transform(Lon, Lat, Alt, radians=False)
            if len(Lon) == 1 and len(Lat) == 1 and len(Alt) == 1:
                return [X_cart[0], Y_cat[0], Z_cart[0]]
            return X_cart, Y_cat, Z_cart
        elif method == "custom":
            radLat = Lat * (np.pi / 180.0)
            radLon = Lon * (np.pi / 180.0)
            a = 6378137.0
            finv = 298.257223563
            f = 1 / finv
            e2 = 1 - (1 - f) * (1 - f)
            v = a / np.sqrt(1 - e2 * np.sin(radLat) * np.sin(radLat))

            X_cart = (v + Alt) * np.cos(radLat) * np.cos(radLon)
            Y_cart = (v + Alt) * np.cos(radLat) * np.sin(radLon)
            Z_cart = (v * (1 - e2) + Alt) * np.sin(radLat)
            if len(Lon) == 1 and len(Lat) == 1 and len(Alt) == 1:
                return [X_cart[0], Y_cart[0], Z_cart[0]]
            return X_cart, Y_cart, Z_cart
        else:
            import sys
            sys.exit("Error: Conversion from WG84 --> Cartesian ! ")

    @staticmethod
    def coord_map1_2_map2(X, Y, targetEPSG, Z: Optional[List] = None, sourceEPSG=4326):
        """
            Convert point coordinates from source to target system

            Args:
                X:map coordinate (e.g lon,lat)
                Y:map coordinate (e.g lon,lat)
                targetEPSG:target coordinate system of the point: integer
                Z:map coordinates
                sourceEPSG:source coordinate system of the point: integer (default geographic coordinate )

            Returns:         point in target coordinate system; list =[xCoord,yCoord,zCoord] or ([lats],[lons])

            Notes:
                - if the transformation from WGS to UTM, x= lat, y=lon ==> coord =(easting(xMap) ,northing(yMap))
                - if the transformation from UTM to WGS 84, x=easting, y=Notthing ==> lat, long

            """
        if Z is None:
            Z: List = []
        elif isinstance(Z, list) == False:
            Z = [Z]
        sourceEPSG_string = "epsg:" + str(sourceEPSG)
        targetEPSG_string = "epsg:" + str(targetEPSG)
        transformer = pyproj.Transformer.from_crs(sourceEPSG_string, targetEPSG_string)
        if len(Z) == 0:
            return transformer.transform(X, Y)

        else:
            return transformer.transform(X, Y, Z)

    @staticmethod
    def polygon(input_polygon: Polygon, source_epsg_code: int, target_epsg_code: int) -> Polygon:
        import geopandas
        shp_df = geopandas.GeoDataFrame({'geometry': [input_polygon]}, crs=rasterio.crs.CRS.from_epsg(source_epsg_code))
        shp_df = shp_df.to_crs(rasterio.crs.CRS.from_epsg(target_epsg_code))

        return shp_df['geometry'][0]


def multi_bands_form_multi_rasters(raster_list: List, output_path: str, no_data: Optional[float] = None,
                                   mask_vls: Optional[List] = None, band_idx=1, dtype='uint16') -> str:
    """
    Notes: we assume the input raster have the same resolution and projection system
    """

    array_list = []
    band_description = []
    info = cRasterInfo(raster_list[0])
    if no_data is None:
        no_data = info.no_data
    for index, img_ in enumerate(raster_list):
        raster_info = cRasterInfo(img_)
        array = raster_info.image_as_array(read_masked=True, band=band_idx)

        if mask_vls is not None:
            for mask_val in mask_vls:
                array = np.ma.masked_where(array == mask_val, array)
                array = array.filled(fill_value=no_data)
        array_list.append(array)
        band_description.append("Band" + str(index + 1) + "_" + Path(img_).stem)

    cRasterInfo.write_raster(output_raster_path=output_path, array_list=array_list, geo_transform=info.geo_transform,
                             epsg_code=info.epsg_code, descriptions=band_description, no_data=no_data, dtype=dtype)
    return output_path


class geoStat:
    def __init__(self, in_array: np.ndarray, display_values: Optional[bool] = False):
        sample = np.ma.masked_invalid(in_array)
        mask = np.ma.getmask(sample)

        # Remove mask and array to vector
        if isinstance(sample, np.ma.MaskedArray):  # check if the sample was masked using the class numpy.ma.MaskedArray
            sample = sample.compressed()  ## return all the non-masked values as 1-D array
        else:
            if sample.ndim > 1:  # if the dimension of the array more than 1 transform it to 1-D array
                sample = sample.flatten()
        self.sample = sample
        # Estimate initial sigma and RMSE
        (self.mu, self.sigma) = norm.fit(sample)
        self.sigma_ = '%.3f' % (self.sigma)
        temp = np.square(sample)
        temp = np.ma.masked_where(temp <= 0, temp)
        self.RMSE = '%.3f' % (np.ma.sqrt(np.ma.mean(temp)))

        self.max = '%.3f' % (np.nanmax(sample))
        self.min = '%.3f' % (np.nanmin(sample))
        self.std = '%.3f' % (np.nanstd(sample))
        self.mean = '%.3f' % (np.nanmean(sample))
        self.median = '%.3f' % (np.nanmedian(sample))
        self.mad = '%.3f' % (stats.median_abs_deviation(sample))
        self.nmad = '%.3f' % (1.4826 * stats.median_abs_deviation(sample))
        self.ce68 = stats.norm.interval(0.68, loc=self.mu, scale=self.sigma)
        self.ce90 = stats.norm.interval(0.9, loc=self.mu, scale=self.sigma)
        self.ce95 = stats.norm.interval(0.95, loc=self.mu, scale=self.sigma)
        self.ce99 = stats.norm.interval(0.99, loc=self.mu, scale=self.sigma)

        if display_values:
            print("mu, sigma", self.mu, self.sigma)
            print("RMSE=", self.RMSE)
            print("max=", self.max, "min=", self.min, "std=", self.std, "mean=", self.mean, "median", self.median,
                  "mad=",
                  self.mad, "nmad=", self.nmad)
            print("CE68=", self.ce68, "CE90=", self.ce90, "CE95=", self.ce95, "CE99=", self.ce99)

    def __repr__(self):
        return "mu:{} , sigm:{} , RMSE:{}, CE90:{}".format(self.mu, self.sigma, self.RMSE, self.ce90)


def crop_raster(input_raster, roi_coord_wind, output_dir: Optional[str] = None, vrt=False,
                raster_type=gdal.GDT_Float32):
    if output_dir is None:
        output_dir = os.path.dirname(input_raster)

    if vrt:
        format = "VRT"
        o_path = os.path.join(output_dir, f"{Path(input_raster).stem}.crop.vrt")
    else:
        format = "GTiff"
        o_path = os.path.join(output_dir, f"{Path(input_raster).stem}.crop.tif")
    params = gdal.TranslateOptions(projWin=roi_coord_wind, format=format, outputType=raster_type, noData=-32767)

    gdal.Translate(destName=o_path,
                   srcDS=gdal.Open(input_raster),
                   options=params)

    return o_path


def compute_rasters_overlap(rasters: List[str]):
    from shapely.geometry import box

    fp_extent = []
    for raster_path in rasters:
        raster = rasterio.open(raster_path)
        extent_geom = box(*raster.bounds)
        fp_extent.append(extent_geom)

    overlap_area = fp_extent[0].intersection(fp_extent[1]) if fp_extent[0].intersects(fp_extent[1]) else None
    index = 2

    while overlap_area is not None and index < len(rasters):
        overlap_area = overlap_area.intersection(fp_extent[index]) if overlap_area.intersects(
            fp_extent[index]) else None
        index += 1

    return overlap_area


def merge_tiles(in_tiles: List, o_file):
    from rasterio.merge import merge

    src_files_to_mosaic = []
    for fp in in_tiles:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)
    # print(mosaic.shape)
    #### Copy the metadata
    out_meta = src.meta.copy()
    crs = src.crs
    # print(crs)
    # Update the metadata
    out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans,
                     "crs": crs})  # "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
    print(in_tiles[0])
    rasterInfo = cRasterInfo(in_tiles[0])
    print(in_tiles[0])
    descriptions = rasterInfo.bands_description  # rasterTemp["BandInfo"]
    listArrays = []
    for id in range(mosaic.shape[0]):
        with rasterio.open(o_file, "w", **out_meta) as dest:
            # the .astype(rasterio.int16) forces dtype to int16
            dest.write_band(id + 1, mosaic[id, :, :])
            dest.set_band_description(id + 1, descriptions[id])
        listArrays.append(mosaic[id, :, :])

    WriteRaster(oRasterPath=o_file, descriptions=descriptions, arrayList=listArrays, epsg=rasterInfo.epsg_code,
                geoTransform=rasterInfo.geo_transform)
