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
This Python script converts ROI_PAC or GAMMA format input interferograms 
into geotiff format files
"""

import os
import logging
import struct

import numpy as np
from osgeo import osr

import pyrate.constants
import pyrate.core.shared
from pyrate.constants import GAMMA, ROIPAC
from pyrate.core.prepifg_helper import PreprocessError
from pyrate.core import shared, mpiops, gamma, roipac, ifgconstants as ifc
import multiprocessing as mp

from pyrate.core.shared import _data_format, _is_interferogram, ROIPAC, _check_raw_data, _check_pixel_res_mismatch, \
    GeotiffException, _is_incidence, collate_metadata, gdal_dataset

log = logging.getLogger("rootLogger")


def main(params):
    """
    Prepare files for conversion.

    :param dict params: Parameters dictionary read in from the config file
    """

    base_ifg_paths = params["BASE_INTERFEROGRAM_PATHS"]

    if params[pyrate.constants.COH_MASK]:
        base_ifg_paths.extend(pyrate.core.shared.coherence_paths(params))

    if params[pyrate.constants.DEM_FILE] is not None:  # optional DEM conversion
        base_ifg_paths.append(params[pyrate.constants.DEM_FILE])

    # ifgs_paths = np.array_split(base_ifg_paths, mpiops.size)[mpiops.rank]
    log.info("Converting input interferograms to geotiff")

    # Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # Running pools
    destination_paths = pool.map(_geotiff_multiprocessing, [(ifg_path, params) for ifg_path in base_ifg_paths])

    # Closing pools
    pool.close()

    log.info("Finished conv2tif")
    return destination_paths



def _geotiff_multiprocessing(parameters):
    """
    Multiprocessing wrapper for full-res geotiff conversion
    """
    unw_path, params = parameters
    # TODO: Need a more robust method for identifying coherence files.
    if params[pyrate.constants.COH_FILE_DIR] and unw_path.endswith('.cc'):
        # If the user has provided a dir for coherence files, place 
        #  converted coherence files in that directory.
        dest = shared.output_tiff_filename(unw_path, params[pyrate.constants.COH_FILE_DIR])
    else:
        dest = shared.output_tiff_filename(unw_path, params[pyrate.constants.OBS_DIR])
    processor = params[pyrate.constants.PROCESSOR]  # roipac or gamma

    # Create full-res geotiff if not already on disk
    if not os.path.exists(dest):
        if processor == GAMMA:
            header = gamma.gamma_header(unw_path, params)
        elif processor == ROIPAC:
            header = roipac.roipac_header(unw_path, params)
        else:
            raise PreprocessError('Processor must be ROI_PAC (0) or GAMMA (1)')
        write_fullres_geotiff(header, unw_path, dest, nodata=params[pyrate.constants.NO_DATA_VALUE])
        return dest
    else:
        log.info("Full-res geotiff already exists")
        return None


def write_fullres_geotiff(header, data_path, dest, nodata):
    """
    Creates a copy of input image data (interferograms, DEM, incidence maps
    etc) in GeoTIFF format with PyRate metadata.

    :param dict header: Interferogram metadata dictionary
    :param str data_path: Input file
    :param str dest: Output destination file
    :param float nodata: No-data value

    :return: None, file saved to disk
    """
    ifg_proc = header[ifc.PYRATE_INSAR_PROCESSOR]
    ncols = header[ifc.PYRATE_NCOLS]
    nrows = header[ifc.PYRATE_NROWS]
    bytes_per_col, fmtstr = _data_format(ifg_proc, _is_interferogram(header), ncols)
    if _is_interferogram(header) and ifg_proc == ROIPAC:
        # roipac ifg has 2 bands
        _check_raw_data(bytes_per_col*2, data_path, ncols, nrows)
    else:
        _check_raw_data(bytes_per_col, data_path, ncols, nrows)

    _check_pixel_res_mismatch(header)

    # position and projection data
    gt = [header[ifc.PYRATE_LONG], header[ifc.PYRATE_X_STEP], 0, header[ifc.PYRATE_LAT], 0, header[ifc.PYRATE_Y_STEP]]
    srs = osr.SpatialReference()
    res = srs.SetWellKnownGeogCS(header[ifc.PYRATE_DATUM])
    if res:
        msg = 'Unrecognised projection: %s' % header[ifc.PYRATE_DATUM]
        raise GeotiffException(msg)

    wkt = srs.ExportToWkt()
    dtype = 'float32' if (_is_interferogram(header) or _is_incidence(header)) else 'int16'

    # get subset of metadata relevant to PyRate
    md = collate_metadata(header)

    # create GDAL object
    ds = gdal_dataset(dest, ncols, nrows, driver="GTiff", bands=1, dtype=dtype, metadata=md, crs=wkt, geotransform=gt, creation_opts=['compress=packbits'])

    # copy data from the binary file
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)

    row_bytes = ncols * bytes_per_col

    with open(data_path, 'rb') as f:
        for y in range(nrows):
            if ifg_proc == ROIPAC:
                if _is_interferogram(header):
                    f.seek(row_bytes, 1)  # skip interleaved band 1

            data = struct.unpack(fmtstr, f.read(row_bytes))

            band.WriteArray(np.array(data).reshape(1, ncols), yoff=y)

    ds = None  # manual close
    del ds


