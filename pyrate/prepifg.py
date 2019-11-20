#   This Python module is part of the PyRate software package.
#
#   Copyright 2017 Geoscience Australia
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
This Python script applies optional multilooking and cropping to input
interferogram geotiff files.
"""
import logging
import os
import shutil
from os.path import splitext

import numexpr as ne
import numpy as np
from osgeo import gdal, gdalconst

import pyrate.constants
import pyrate.conv2tif
import pyrate.core
import pyrate.core.shared
from pyrate.core import shared, mpiops, prepifg_helper, gamma, roipac, ifgconstants as ifc
from pyrate.core.gdal_python import _gdalwarp_width_and_height, _get_resampled_data_size
from pyrate.constants import GAMMA, ROIPAC, LOW_FLOAT32, ALREADY_SAME_SIZE
import multiprocessing as mp

from pyrate.core.shared import output_tiff_filename, Ifg, DEM

log = logging.getLogger("rootLogger")


def main(params=None):
    """
    Main workflow function for preparing interferograms for PyRate.

    :param dict params: Parameters dictionary read in from the config file
    """

    base_ifg_paths = params["BASE_INTERFEROGRAM_PATHS"]

    # optional DEM conversion
    if params[pyrate.constants.DEM_FILE] is not None:
        base_ifg_paths.append(params[pyrate.constants.DEM_FILE])

    # create output dir
    shared.mkdir_p(params[pyrate.constants.OUT_DIR])

    # process_base_ifgs_paths = np.array_split(base_ifg_paths, mpiops.size)[mpiops.rank]
    gtiff_paths = [shared.output_tiff_filename(f, params[pyrate.constants.OBS_DIR]) for f in base_ifg_paths]

    log.info("Preparing interferograms by cropping/multilooking")

    for f in gtiff_paths:
        if not os.path.isfile(f):
            raise Exception("Can not find geotiff: " + str(f) + ". Ensure you have converted your interferograms to geotiffs.")

    ifgs = [dem_or_ifg(p) for p in gtiff_paths]
    xlooks, ylooks, crop = params["IFG_LKSX"], params["IFG_LKSY"], params["IFG_CROP_OPT"]

    user_exts = (params["IFG_XFIRST"], params["IFG_YFIRST"], params["IFG_XLAST"], params["IFG_YLAST"])
    exts = prepifg_helper.get_analysis_extent(crop, ifgs, xlooks, ylooks, user_exts=user_exts)
    log.debug("Extents (xmin, ymin, xmax, ymax): " + str(exts))
    thresh = params["NO_DATA_AVERAGING_THRESHOLD"]

    # Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # Running pools
    pool.map(_prepifg_multiprocessing, [(ifg_path, xlooks, ylooks, exts, thresh, crop, params) for ifg_path in gtiff_paths])

    # Closing pools
    pool.close()

    log.info("Finished prepifg")


def _prepifg_multiprocessing(parameters):
    """
    Multiprocessing wrapper for prepifg
    """
    path, xlooks, ylooks, exts, thresh, crop, params = parameters
    processor = params[pyrate.constants.PROCESSOR]  # roipac or gamma
    if processor == GAMMA:
        header = gamma.gamma_header(path, params)
    elif processor == ROIPAC:
        header = roipac.roipac_header(path, params)
    else:
        raise Exception('Processor must be ROI_PAC (0) or GAMMA (1)')
    # If we're performing coherence masking, find the coherence file for this IFG.
    if params[pyrate.constants.COH_MASK] and shared._is_interferogram(header):
        coherence_path = pyrate.core.shared.coherence_paths_for(path, params, tif=True)[0]
        coherence_thresh = params[pyrate.constants.COH_THRESH]
    else:
        coherence_path = None
        coherence_thresh = None

    prepare_ifg(path, xlooks, ylooks, exts, thresh, crop, out_path=params[pyrate.constants.OUT_DIR], header=header, coherence_path=coherence_path, coherence_thresh=coherence_thresh)


def prepare_ifg(raster_path, xlooks, ylooks, exts, thresh, crop_opt, write_to_disk=True, out_path=None, header=None, coherence_path=None, coherence_thresh=None):
    """
    Open, resample, crop and optionally save to disk an interferogram or DEM.
    Returns are only given if write_to_disk=False

    :param str raster_path: Input raster file path name
    :param int xlooks: Number of multi-looks in x; 5 is 5 times smaller,
        1 is no change
    :param int ylooks: Number of multi-looks in y
    :param tuple exts: Tuple of user defined georeferenced extents for
        new file: (xfirst, yfirst, xlast, ylast)cropping coordinates
    :param float thresh: see thresh in prepare_ifgs()
    :param int crop_opt: Crop option
    :param bool write_to_disk: Write new data to disk
    :param str out_path: Path for output file
    :param dict header: dictionary of metadata from header file

    :return: resampled_data: output cropped and resampled image
    :rtype: ndarray
    :return: out_ds: destination gdal dataset object
    :rtype: gdal.Dataset
    """
    do_multilook = xlooks > 1 or ylooks > 1
    # resolution=None completes faster for non-multilooked layers in gdalwarp
    resolution = [None, None]
    raster = dem_or_ifg(raster_path)
    if not raster.is_open:
        raster.open()
    if do_multilook:
        resolution = [xlooks * raster.x_step, ylooks * raster.y_step]
    if not do_multilook and crop_opt == ALREADY_SAME_SIZE:
        renamed_path = mlooked_path(raster.data_path, looks=xlooks, crop_out=crop_opt)
        shutil.copy(raster.data_path, renamed_path)
        # set metadata to indicated has been cropped and multilooked
        # copy file with mlooked path
        # Convenience dummy operation for when no multi-looking or cropping required
        ifg = dem_or_ifg(renamed_path)
        ifg.open()
        ifg.dataset.SetMetadataItem(ifc.DATA_TYPE, ifc.MULTILOOKED)
        data = ifg.dataset.ReadAsArray()
        return data, ifg.dataset

    return _warp(raster, xlooks, ylooks, exts, resolution, thresh, crop_opt, write_to_disk, out_path, header, coherence_path, coherence_thresh)


def _warp(ifg, x_looks, y_looks, extents, resolution, thresh, crop_out, write_to_disk=True, out_path=None, header=None, coherence_path=None, coherence_thresh=None):
    """
    Convenience function for calling GDAL functionality
    """
    if x_looks != y_looks:
        raise ValueError('X and Y looks mismatch')

    # cut, average, resample the final output layers
    op = output_tiff_filename(ifg.data_path, out_path)
    looks_path = mlooked_path(op, y_looks, crop_out)

    #     # Add missing/updated metadata to resampled ifg/DEM
    #     new_lyr = type(ifg)(looks_path)
    #     new_lyr.open(readonly=True)
    #     # for non-DEMs, phase bands need extra metadata & conversions
    #     if hasattr(new_lyr, "phase_band"):
    #         # TODO: LOS conversion to vertical/horizontal (projection)
    #         # TODO: push out to workflow
    #         #if params.has_key(REPROJECTION_FLAG):
    #         #    reproject()
    driver_type = 'GTiff' if write_to_disk else 'MEM'
    resampled_data, out_ds = crop_resample_average( input_tif=ifg.data_path, extents=extents, new_res=resolution, output_file=looks_path, thresh=thresh, out_driver_type=driver_type, hdr=header, coherence_path=coherence_path, coherence_thresh=coherence_thresh)
    if not write_to_disk:
        return resampled_data, out_ds


def mlooked_path(path, looks, crop_out):
    """
    Adds suffix to ifg path, for creating a new path for multilooked files.

    :param str path: original interferogram path
    :param int looks: number of range looks applied
    :param int crop_out: crop option applied

    :return: multilooked file name
    :rtype: str
    """
    base, ext = splitext(path)
    return "{base}_{looks}rlks_{crop_out}cr{ext}".format(base=base, looks=looks, crop_out=crop_out, ext=ext)


def crop_resample_average(input_tif, extents, new_res, output_file, thresh, out_driver_type='GTiff', match_pyrate=False, hdr=None, coherence_path=None, coherence_thresh=None):
    """
    Crop, resample, and average a geotiff image.

    :param str input_tif: Path to input geotiff to resample/crop
    :param tuple extents: Cropping extents (xfirst, yfirst, xlast, ylast)
    :param list new_res: [xres, yres] resolution of output image
    :param str output_file: Path to output resampled/cropped geotiff
    :param float thresh: NaN fraction threshold
    :param str out_driver_type: The output driver; `MEM` or `GTiff` (optional)
    :param bool match_pyrate: Match Legacy output (optional)
    :param dict hdr: dictionary of metadata

    :return: resampled_average: output cropped and resampled image
    :rtype: ndarray
    :return: out_ds: destination gdal dataset object
    :rtype: gdal.Dataset
    """
    dst_ds, _, _, _ = _crop_resample_setup(extents, input_tif, new_res, output_file, out_bands=2, dst_driver_type='MEM')

    # make a temporary copy of the dst_ds for PyRate style prepifg
    tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', dst_ds) if (match_pyrate and new_res[0]) else None
    src_ds, src_ds_mem = _setup_source(input_tif)

    if coherence_path and coherence_thresh:
        coherence_raster = dem_or_ifg(coherence_path)
        coherence_raster.open()
        coherence_ds = coherence_raster.dataset
        coherence_masking(src_ds_mem, coherence_ds, coherence_thresh)
    elif coherence_path and not coherence_thresh:
        raise ValueError(f"Coherence file provided without a coherence "
                         f"threshold. Please ensure you provide 'cohthresh' "
                         f"in your config if coherence masking is enabled.")

    resampled_average, src_ds_mem = gdal_average(dst_ds, src_ds, src_ds_mem, thresh)
    src_dtype = src_ds_mem.GetRasterBand(1).DataType
    src_gt = src_ds_mem.GetGeoTransform()

    # required to match Legacy output
    if tmp_ds:
        _alignment(input_tif, new_res, resampled_average, src_ds_mem,src_gt, tmp_ds)

    # grab metadata from existing geotiff
    gt = dst_ds.GetGeoTransform()
    wkt = dst_ds.GetProjection()

    # TEST HERE IF EXISTING FILE HAS PYRATE METADATA. IF NOT ADD HERE
    if not ifc.DATA_TYPE in dst_ds.GetMetadata() and hdr is not None:
        md = shared.collate_metadata(hdr)
    else:
        md = dst_ds.GetMetadata()

    # update metadata for output
    for k, v in md.items():
        if k == ifc.DATA_TYPE:
            # update data type metadata
            if v == ifc.ORIG and coherence_path:
                md.update({ifc.DATA_TYPE: ifc.COHERENCE})
            elif v == ifc.ORIG and not coherence_path:
                md.update({ifc.DATA_TYPE: ifc.MULTILOOKED})
            elif v == ifc.DEM:
                md.update({ifc.DATA_TYPE: ifc.MLOOKED_DEM})
            elif v == ifc.INCIDENCE:
                md.update({ifc.DATA_TYPE: ifc.MLOOKED_INC})
            elif v == ifc.COHERENCE and coherence_path:
                pass
            elif v == ifc.MULTILOOKED and coherence_path:
                md.update({ifc.DATA_TYPE:ifc.COHERENCE})
            elif v == ifc.MULTILOOKED and not coherence_path:
                pass
            else:
                raise TypeError('Data Type metadata not recognised')

    # In-memory GDAL driver doesn't support compression so turn it off.
    creation_opts = ['compress=packbits'] if out_driver_type != 'MEM' else []
    out_ds = pyrate.core.shared.gdal_dataset(output_file, dst_ds.RasterXSize, dst_ds.RasterYSize, driver=out_driver_type, bands=1, dtype=src_dtype, metadata=md, crs=wkt, geotransform=gt, creation_opts=creation_opts)
    shared.write_geotiff(resampled_average, out_ds, np.nan)

    return resampled_average, out_ds


def _crop_resample_setup(extents, input_tif, new_res, output_file, dst_driver_type='GTiff', out_bands=2):
    """
    Convenience function for crop/resample setup
    """
    # Source
    src_ds = gdal.Open(input_tif, gdalconst.GA_ReadOnly)
    src_proj = src_ds.GetProjection()

    # source metadata to be copied into the output
    meta_data = src_ds.GetMetadata()

    # get the image extents
    min_x, min_y, max_x, max_y = extents
    geo_transform = src_ds.GetGeoTransform()  # tuple of 6 numbers

    # Create a new geotransform for the image
    gt2 = list(geo_transform)
    gt2[0] = min_x
    gt2[3] = max_y
    # We want a section of source that matches this:
    resampled_proj = src_proj
    if new_res[0]:  # if new_res is not None, it can't be zero either
        resampled_geotrans = gt2[:1] + [new_res[0]] + gt2[2:-1] + [new_res[1]]
    else:
        resampled_geotrans = gt2

    px_height, px_width = _gdalwarp_width_and_height(max_x, max_y, min_x, min_y, resampled_geotrans)

    # Output / destination
    dst = gdal.GetDriverByName(dst_driver_type).Create(output_file, px_width, px_height, out_bands, gdalconst.GDT_Float32)
    dst.SetGeoTransform(resampled_geotrans)
    dst.SetProjection(resampled_proj)

    for k, v in meta_data.items():
        dst.SetMetadataItem(k, v)

    return dst, resampled_proj, src_ds, src_proj


def _setup_source(input_tif):
    """convenience setup function for gdal_average"""
    src_ds = gdal.Open(input_tif)
    data = src_ds.GetRasterBand(1).ReadAsArray()
    src_dtype = src_ds.GetRasterBand(1).DataType
    mem_driver = gdal.GetDriverByName('MEM')
    src_ds_mem = mem_driver.Create('', src_ds.RasterXSize, src_ds.RasterYSize, 2, src_dtype)
    src_ds_mem.GetRasterBand(1).WriteArray(data)
    src_ds_mem.GetRasterBand(1).SetNoDataValue(0)
    # if data==0, then 1, else 0
    nan_matrix = np.isclose(data, 0, atol=1e-6)
    src_ds_mem.GetRasterBand(2).WriteArray(nan_matrix)
    src_ds_mem.SetGeoTransform(src_ds.GetGeoTransform())
    return src_ds, src_ds_mem


def coherence_masking(src_ds, coherence_ds, coherence_thresh):
    """
    Perform coherence masking on raster in-place.

    Based on gdal_calc formula provided by Nahidul:
    gdal_calc.py -A 20151127-20151209_VV_8rlks_flat_eqa.cc.tif
     -B 20151127-20151209_VV_8rlks_eqa.unw.tif
     --outfile=test_v1.tif --calc="B*(A>=0.8)-999*(A<0.8)"
     --NoDataValue=-999

    Args:
        ds: The interferogram to mask as GDAL dataset.
        coherence_ds: The coherence GDAL dataset.
        coherence_thresh: The coherence threshold.
    """
    coherence_band = coherence_ds.GetRasterBand(1)
    src_band = src_ds.GetRasterBand(1)
    # ndv = src_band.GetNoDataValue()
    ndv = np.nan
    coherence = coherence_band.ReadAsArray()
    src = src_band.ReadAsArray()
    var = {'coh': coherence, 'src': src, 't': coherence_thresh, 'ndv': ndv}
    formula = 'where(coh>=t, src, ndv)'
    res = ne.evaluate(formula, local_dict=var)
    src_band.WriteArray(res)


def gdal_average(dst_ds, src_ds, src_ds_mem, thresh):
    """
    Perform subsampling of an image by averaging values

    :param gdal.Dataset dst_ds: Destination gdal dataset object
    :param str input_tif: Input geotif
    :param float thresh: NaN fraction threshold

    :return resampled_average: resampled image data
    :rtype: ndarray
    :return src_ds_mem: Modified in memory src_ds with nan_fraction in Band2. The nan_fraction
        is computed efficiently here in gdal in the same step as the that of
        the resampled average (band 1). This results is huge memory and
        computational efficiency
    :rtype: gdal.Dataset
    """
    src_ds_mem.GetRasterBand(2).SetNoDataValue(-100000)
    src_gt = src_ds.GetGeoTransform()
    src_ds_mem.SetGeoTransform(src_gt)
    gdal.ReprojectImage(src_ds_mem, dst_ds, '', '', gdal.GRA_Average)
    # dst_ds band2 average is our nan_fraction matrix
    nan_frac = dst_ds.GetRasterBand(2).ReadAsArray()
    resampled_average = dst_ds.GetRasterBand(1).ReadAsArray()
    resampled_average[nan_frac >= thresh] = np.nan
    return resampled_average, src_ds_mem


def _alignment(input_tif, new_res, resampled_average, src_ds_mem, src_gt, tmp_ds):
    """
    Correction step to match python multi-look/crop output to match that of
    Legacy data. Modifies the resampled_average array in place.
    """
    src_ds = gdal.Open(input_tif)
    data = src_ds.GetRasterBand(1).ReadAsArray()
    xlooks = ylooks = int(new_res[0] / src_gt[1])
    xres, yres = _get_resampled_data_size(xlooks, ylooks, data)
    nrows, ncols = resampled_average.shape
    # Legacy nearest neighbor resampling for the last
    # [yres:nrows, xres:ncols] cells without nan_conversion
    # turn off nan-conversion
    src_ds_mem.GetRasterBand(1).SetNoDataValue(LOW_FLOAT32)
    # nearest neighbor resapling
    gdal.ReprojectImage(src_ds_mem, tmp_ds, '', '', gdal.GRA_NearestNeighbour)
    # only take the [yres:nrows, xres:ncols] slice
    if nrows > yres or ncols > xres:
        resampled_nearest_neighbor = tmp_ds.GetRasterBand(1).ReadAsArray()
        resampled_average[yres - nrows:, xres - ncols:] = resampled_nearest_neighbor[yres - nrows:, xres - ncols:]


def dem_or_ifg(data_path):
    """
    Returns an Ifg or DEM class object from input geotiff file.

    :param str data_path: file path name

    :return: Interferogram or DEM object from input file
    :rtype: Ifg or DEM class object
    """
    ds = gdal.Open(data_path)
    md = ds.GetMetadata()
    if ifc.MASTER_DATE in md:  # ifg
        return Ifg(data_path)
    else:
        return DEM(data_path)