#   This Python module is part of the PyRate software package.
#
#   Copyright 2017 Geoscience Australia
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
This Python module contains bindings for the GDAL library
"""
# pylint: disable=too-many-arguments,R0914
import logging

from osgeo import gdal, gdalnumeric, gdalconst
from PIL import Image, ImageDraw
import numpy as np

# from pyrate.prepifg import _crop_resample_setup

log = logging.getLogger("rootLogger")
gdal.SetCacheMax(2**15)


def world_to_pixel(geo_transform, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate;
    see: http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html

    :param list geo_transform: Affine transformation coefficients
    :param float x: longitude coordinate
    :param float y: latitude coordinate

    :return: col: pixel column number
    :rtype: int
    :return: line: pixel line number
    :rtype: int
    """
    ul_x = geo_transform[0]
    ul_y = geo_transform[3]
    xres = geo_transform[1]
    yres = geo_transform[5]
    col = int(np.round((x - ul_x) / xres))
    line = int(np.round((ul_y - y) / abs(yres))) # yres has negative size

    return col, line


def crop(input_file, extents, geo_trans=None, nodata=np.nan):
    """
    Adapted from http://karthur.org/2015/clipping-rasters-in-python.html

    Clips a raster (given as either a gdal.Dataset or as a numpy.array
    instance) to a polygon layer provided by a Shapefile (or other vector
    layer). If a numpy.array is given, a "GeoTransform" must be provided
    (via dataset.GetGeoTransform() in GDAL). Returns an array. Clip features
    must be a dissolved, single-part geometry (not multi-part). Modified from:

    http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    #clip-a-geotiff-with-shapefile

    Arguments:
        rast            A gdal.Dataset or a NumPy array
        features_path   The path to the clipping features
        geo_trans              An optional GDAL GeoTransform to use instead
        nodata          The NoData value; defaults to -9999.

    :param str input_file: input image file path
    :param list extents: extents for the cropped area
    :param list geo_trans: An optional GDAL GeoTransform to use instead
    :param int nodata: The NoData value; defaults to -9999

    :return: clip: cropped part of the image
    :rtype: ndarray
    :return: gt2: geotransform parameters for the cropped image
    :rtype: list
    """

    def image_to_array(i):
        """
        Converts a Python Imaging Library (PIL) array to a gdalnumeric image.
        """
        arr = gdalnumeric.fromstring(i.tobytes(), 'b')
        arr.shape = i.im.size[1], i.im.size[0]
        return arr

    raster = gdal.Open(input_file)
    # Can accept either a gdal.Dataset or numpy.array instance
    if not isinstance(raster, np.ndarray):
        if not geo_trans:
            geo_trans = raster.GetGeoTransform()
        raster = raster.ReadAsArray()
    else:
        if not geo_trans:
            raise ValueError('geo transform must be supplied')

    # Convert the layer extent to image pixel coordinates
    min_x, min_y, max_x, max_y = extents
    ul_x, ul_y = world_to_pixel(geo_trans, min_x, max_y)
    lr_x, lr_y = world_to_pixel(geo_trans, max_x, min_y)

    # Calculate the pixel size of the new image
    px_width = int(lr_x - ul_x)
    px_height = int(lr_y - ul_y)

    # If the clipping features extend out-of-bounds and ABOVE the raster...
    if geo_trans[3] < max_y:
        # In such a case... ul_y ends up being negative--can't have that!
        # iY = ul_y
        ul_y = 0

    # Multi-band image?
    try:
        clip = raster[:, ul_y:lr_y, ul_x:lr_x]

    except IndexError:
        clip = raster[ul_y:lr_y, ul_x:lr_x]

    # Create a new geomatrix for the image
    gt2 = list(geo_trans)
    gt2[0] = min_x
    gt2[3] = max_y

    # Map points to pixels for drawing the boundary on a blank 8-bit,
    #   black and white, mask image.
    points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_y, max_y)]
    pixels = []


    for point in points:
        pixels.append(world_to_pixel(gt2, point[0], point[1]))

    raster_poly = Image.new('L', size=(px_width, px_height), color=1)
    rasterize = ImageDraw.Draw(raster_poly)
    rasterize.polygon(pixels, 0)  # Fill with zeroes


    mask = image_to_array(raster_poly)

    # Clip the image using the mask
    try:
        clip = gdalnumeric.choose(mask, (clip, nodata))

    # If the clipping features extend out-of-bounds and BELOW the raster...
    except ValueError:
        # We have to cut the clipping features to the raster!
        rshp = list(mask.shape)
        if mask.shape[-2] != clip.shape[-2]:
            rshp[0] = clip.shape[-2]

        if mask.shape[-1] != clip.shape[-1]:
            rshp[1] = clip.shape[-1]

        mask.resize(*rshp, refcheck=False)

        clip = gdalnumeric.choose(mask, (clip, nodata))

    # AttributeError: 'numpy.ndarray' object has no attribute 'close'
    # raster.close()
    raster = None

    return clip, gt2


# def resample_nearest_neighbour(input_tif, extents, new_res, output_file):
#     """
#     Nearest neighbor resampling and cropping of an image.
#
#     :param str input_tif: input geotiff file path
#     :param list extents: new extents for cropping
#     :param float new_res: new resolution for resampling
#     :param str output_file: output geotiff file path
#
#     :return: dst: resampled image
#     :rtype: ndarray
#     """
#     dst, resampled_proj, src, _ = _crop_resample_setup(extents, input_tif,
#                                                        new_res, output_file)
#     # Do the work
#     gdal.ReprojectImage(src, dst, '', resampled_proj,
#                         gdalconst.GRA_NearestNeighbour)
#     return dst.ReadAsArray()


def _gdalwarp_width_and_height(max_x, max_y, min_x, min_y, geo_trans):
    """
    Modify pixel height and width
    """
    # modified image extents
    ul_x, ul_y = world_to_pixel(geo_trans, min_x, max_y)
    lr_x, lr_y = world_to_pixel(geo_trans, max_x, min_y)
    # Calculate the pixel size of the new image
    px_width = int(lr_x - ul_x)
    px_height = int(lr_y - ul_y)
    return px_height, px_width  # this is the same as `gdalwarp`


def _get_resampled_data_size(xscale, yscale, data):
    """convenience function mimicking the Legacy output size"""
    xscale = int(xscale)
    yscale = int(yscale)
    ysize, xsize = data.shape
    xres, yres = int(xsize / xscale), int(ysize / yscale)
    return xres, yres
