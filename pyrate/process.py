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
# coding: utf-8
"""
This Python module runs the main PyRate processing workflow
"""
import logging
import os
from itertools import product
from os.path import join
import pickle as cp
from collections import OrderedDict
import numpy as np
import pyrate.constants
from pyrate.configuration import get_dest_paths
import pyrate.core.shared
from pyrate.core import shared, algorithm, orbital, ref_phs_est as rpe, ifgconstants as ifc, timeseries, mst, covariance as vcm_module,  linrate, refpixel
from pyrate.core.aps import _wrap_spatio_temporal_filter
from pyrate.core.shared import Ifg, PrereadIfg, Tile

log = logging.getLogger("rootLogger")


from pyrate.core.shared import output_tiff_filename
from pyrate.prepifg import mlooked_path


def get_tiles(ifg_path, rows, cols):
    """
    Break up the interferograms into smaller tiles based on user supplied
    rows and columns.

    :param list ifg_path: List of destination geotiff file names
    :param int rows: Number of rows to break each interferogram into
    :param int cols: Number of columns to break each interferogram into

    :return: tiles: List of shared.Tile instances
    :rtype: list
    """
    ifg = Ifg(ifg_path)
    ifg.open(readonly=True)
    tiles = create_tiles(ifg.shape, nrows=rows, ncols=cols)
    ifg.close()
    return tiles

def main(params):
    """
    Top level function to perform PyRate workflow on given interferograms

    :param dict params: Dictionary of configuration parameters

    :return: refpt: tuple of reference pixel x and y position
    :rtype: tuple
    :return: maxvar: array of maximum variance values of interferograms
    :rtype: ndarray
    :return: vcmt: Variance-covariance matrix array
    :rtype: ndarray
    """

    # cut, average, resample the final output layers
    ifg_paths = []
    gtiff_paths = [shared.output_tiff_filename(f, params[pyrate.constants.OBS_DIR]) for f in params["BASE_INTERFEROGRAM_PATHS"] ]
    for ifg in gtiff_paths:
        op = output_tiff_filename(ifg, params[pyrate.constants.OBS_DIR])
        looks_path = mlooked_path(op, params["IFG_LKSX"], params["IFG_CROP_OPT"])
        ifg_paths.append(looks_path)

    tiles = get_tiles(ifg_paths[0], params["SUB_ROW"], params["SUB_COL"])

    preread_ifgs = _create_ifg_dict(ifg_paths, params=params, tiles=tiles)
    # _mst_calc(ifg_paths, params, tiles, preread_ifgs)

    refpx, refpy = _ref_pixel_calc(ifg_paths, params)
    log.debug("refpx, refpy: " + str(refpx) + " " + str(refpy))

    # remove non ifg keys
    for k in ['gt', 'epochlist', 'md', 'wkt']:
        preread_ifgs.pop(k)

    if pyrate.constants.ORBITAL_FIT_METHOD:
        _orb_fit_calc(ifg_paths, params, preread_ifgs)

    _ref_phase_estimation(ifg_paths, params, refpx, refpy)

    _mst_calc(ifg_paths, params, tiles, preread_ifgs)

    # spatio-temporal aps filter
    _wrap_spatio_temporal_filter(ifg_paths, params, tiles, preread_ifgs)

    maxvar, vcmt = _maxvar_vcm_calc(ifg_paths, params, preread_ifgs)

    # save phase data tiles as numpy array for timeseries and linrate calc
    shared.save_numpy_phase(ifg_paths, tiles, params)

    _timeseries_calc(ifg_paths, params, vcmt, tiles, preread_ifgs)

    _linrate_calc(ifg_paths, params, vcmt, tiles, preread_ifgs)

    log.info('PyRate workflow completed')


def _join_dicts(dictionaries):
    """
    Function to concatenate dictionaries
    """
    if dictionaries is None:  # pragma: no cover
        return
    assembled_dict = {k: v for D in dictionaries for k, v in D.items()}
    return assembled_dict


def _create_ifg_dict(dest_tifs, params, tiles):
    """
    1. Convert ifg phase data into numpy binary files.
    2. Save the preread_ifgs dict with information about the ifgs that are
    later used for fast loading of Ifg files in IfgPart class

    :param list dest_tifs: List of destination tifs
    :param dict params: Config dictionary
    :param list tiles: List of all Tile instances

    :return: preread_ifgs: Dictionary containing information regarding
                interferograms that are used later in workflow
    :rtype: dict
    """
    ifgs_dict = {}
    shared.save_numpy_phase(dest_tifs, tiles, params)
    for d in dest_tifs:
        ifg = shared._prep_ifg(d, params)
        ifgs_dict[d] = PrereadIfg(path=d, nan_fraction=ifg.nan_fraction, master=ifg.master, slave=ifg.slave, time_span=ifg.time_span, nrows=ifg.nrows, ncols=ifg.ncols, metadata=ifg.meta_data)
        ifg.close()

    preread_ifgs_file = join(params[pyrate.constants.TMPDIR], 'preread_ifgs.pk')

    # add some extra information that's also useful later
    gt, md, wkt = shared.get_geotiff_header_info(dest_tifs[0])
    ifgs_dict['epochlist'] = algorithm.get_epochs(ifgs_dict)[0]
    ifgs_dict['gt'] = gt
    ifgs_dict['md'] = md
    ifgs_dict['wkt'] = wkt
    # dump ifgs_dict file for later use
    cp.dump(ifgs_dict, open(preread_ifgs_file, 'wb'))

    preread_ifgs = OrderedDict(sorted(cp.load(open(preread_ifgs_file, 'rb')).items()))
    log.debug('Finished converting phase_data to numpy.')
    return preread_ifgs


def _mst_calc(dest_tifs, params, process_tiles, preread_ifgs):
    """
    MPI wrapper function for MST calculation
    """

    for tile in process_tiles:
        log.info('Calculating minimum spanning tree matrix')
        mst_tile = mst.mst_multiprocessing(tile, dest_tifs, preread_ifgs)
        # locally save the mst_mat
        mst_file_process_n = join(params[pyrate.constants.TMPDIR], 'mst_mat_{}.npy'.format(tile.index))
        np.save(file=mst_file_process_n, arr=mst_tile)

    log.debug('Finished mst calculation for process')


def _ref_pixel_calc(ifg_paths, params):
    """
    Wrapper for reference pixel calculation
    """
    refx = params[pyrate.constants.REFX]
    refy = params[pyrate.constants.REFY]

    ifg = Ifg(ifg_paths[0])
    ifg.open(readonly=True)

    if refx == -1 or refy == -1:

        log.info('Searching for best reference pixel location')
        half_patch_size, thresh, process_grid = refpixel.ref_pixel_setup(ifg_paths, params)
        refpixel.save_ref_pixel_blocks(process_grid, half_patch_size, ifg_paths, params)
        mean_sds = refpixel._ref_pixel_mpi(process_grid, half_patch_size, ifg_paths, thresh, params)
        mean_sds = np.hstack(mean_sds)
        refy, refx = refpixel.find_min_mean(mean_sds, process_grid)
        log.info('Selected reference pixel coordinate: ({}, {})'.format(refx, refy))
    else:
        log.info('Reusing reference pixel from config file: ({}, {})'.format(refx, refy))
    ifg.close()
    return refx, refy


def _orb_fit_calc(ifg_paths, params, preread_ifgs=None):
    """
    MPI wrapper for orbital fit correction
    """
    log.info('Calculating orbital correction')

    if preread_ifgs:  # don't check except for mpi tests
        # perform some general error/sanity checks
        log.debug('Checking Orbital error correction status')
        if shared.check_correction_status(ifg_paths, ifc.PYRATE_ORBITAL_ERROR):
            log.debug('Finished Orbital error correction')
            return  # return if True condition returned

    # Here we do all the multilooking in one process, but in memory
    # can use multiple processes if we write data to disc during
    # remove_orbital_error step
    # A performance comparison should be made for saving multilooked
    # files on disc vs in memory single process multilooking
    orbital.remove_orbital_error(ifg_paths, params, preread_ifgs)
    log.debug('Finished Orbital error correction')


def _ref_phase_estimation(ifg_paths, params, refpx, refpy):
    """
    Wrapper for reference phase estimation.
    """
    log.info("Calculating reference phase")
    if len(ifg_paths) < 2:
        raise rpe.ReferencePhaseError(f"At least two interferograms required for reference phase "f"correction ({len(ifg_paths)} provided).")

    if shared.check_correction_status(ifg_paths, ifc.PYRATE_REF_PHASE):
        log.debug('Finished reference phase estimation')
        return

    if params[pyrate.constants.REF_EST_METHOD] == 1:
        ref_phs = rpe.est_ref_phase_method1(ifg_paths, params)
    elif params[pyrate.constants.REF_EST_METHOD] == 2:
        ref_phs = rpe.est_ref_phase_method2(ifg_paths, params, refpx, refpy)
    else:
        raise rpe.ReferencePhaseError("No such option, use '1' or '2'.")

    # Save reference phase numpy arrays to disk.
    ref_phs_file = os.path.join(params[pyrate.constants.TMPDIR], 'ref_phs.npy')

    collected_ref_phs = np.zeros(len(ifg_paths), dtype=np.float64)
    process_indices = range(len(ifg_paths))
    collected_ref_phs[process_indices] = ref_phs

    this_process_ref_phs = np.zeros(shape=len(process_indices), dtype=np.float64)
    collected_ref_phs[process_indices] = this_process_ref_phs
    np.save(file=ref_phs_file, arr=ref_phs)

    log.debug('Finished reference phase estimation')


def _linrate_calc(ifg_paths, params, vcmt, tiles, preread_ifgs):
    """
    MPI wrapper for linrate calculation
    """
    process_tiles = tiles
    log.info('Calculating rate map from stacking')
    output_dir = params[pyrate.constants.TMPDIR]
    for t in process_tiles:
        log.debug('Stacking of tile {}'.format(t.index))
        ifg_parts = [shared.IfgPart(p, t, preread_ifgs) for p in ifg_paths]
        mst_grid_n = np.load(os.path.join(output_dir, 'mst_mat_{}.npy'.format(t.index)))
        rate, error, samples = linrate.linear_rate(ifg_parts, params, vcmt, mst_grid_n)
        # declare file names
        np.save(file=os.path.join(output_dir, 'linrate_{}.npy'.format(t.index)), arr=rate)
        np.save(file=os.path.join(output_dir, 'linerror_{}.npy'.format(t.index)), arr=error)
        np.save(file=os.path.join(output_dir, 'linsamples_{}.npy'.format(t.index)), arr=samples)


def _maxvar_vcm_calc(ifg_paths, params, preread_ifgs):
    """
    MPI wrapper for maxvar and vcmt computation
    """
    log.info('Calculating the temporal variance-covariance matrix')

    # Get RDIst class object
    ifg = Ifg(ifg_paths[0])
    ifg.open()
    r_dist = vcm_module.RDist(ifg)()
    ifg.close()

    prcs_ifgs = ifg_paths
    process_maxvar = []

    for n, i in enumerate(prcs_ifgs):
        log.debug('Calculating maxvar for {} of process ifgs {} of total {}'.format(n+1, len(prcs_ifgs), len(ifg_paths)))
        process_maxvar.append(vcm_module.cvd(i, params, r_dist, calc_alpha=True, write_vals=True, save_acg=True)[0])

    maxvar = process_maxvar
    vcmt = vcm_module.get_vcmt(preread_ifgs, maxvar)
    return maxvar, vcmt


def _timeseries_calc(ifg_paths, params, vcmt, tiles, preread_ifgs):
    """
    MPI wrapper for time series calculation.
    """
    if params[pyrate.constants.TIME_SERIES_CAL] == 0:
        log.info('Time Series Calculation not required')
        return

    if params[pyrate.constants.TIME_SERIES_METHOD] == 1:
        log.info('Calculating time series using Laplacian Smoothing method')
    elif params[pyrate.constants.TIME_SERIES_METHOD] == 2:
        log.info('Calculating time series using SVD method')

    output_dir = params[pyrate.constants.TMPDIR]
    process_tiles = tiles
    for t in process_tiles:
        log.debug('Calculating time series for tile {}'.format(t.index))
        ifg_parts = [shared.IfgPart(p, t, preread_ifgs) for p in ifg_paths]
        mst_tile = np.load(os.path.join(output_dir, 'mst_mat_{}.npy'.format(t.index)))
        res = timeseries.time_series(ifg_parts, params, vcmt, mst_tile)
        tsincr, tscum, _ = res
        np.save(file=os.path.join(output_dir, 'tsincr_{}.npy'.format(t.index)), arr=tsincr)
        np.save(file=os.path.join(output_dir, 'tscuml_{}.npy'.format(t.index)), arr=tscum)


def get_tiles(ifg_path, rows, cols):
    """
    Break up the interferograms into smaller tiles based on user supplied
    rows and columns.

    :param list ifg_path: List of destination geotiff file names
    :param int rows: Number of rows to break each interferogram into
    :param int cols: Number of columns to break each interferogram into

    :return: tiles: List of shared.Tile instances
    :rtype: list
    """
    ifg = Ifg(ifg_path)
    ifg.open(readonly=True)
    tiles = create_tiles(ifg.shape, nrows=rows, ncols=cols)
    ifg.close()
    return tiles


def create_tiles(shape, nrows=2, ncols=2):
    """
    Return a list of tiles containing nrows x ncols with each tile preserving
    the physical layout of original array. The number of rows can be changed
    (increased) such that the resulting tiles with float32's do not exceed
    500MB in memory. When the array shape (rows, columns) are not divisible
    by (nrows, ncols) then some of the array dimensions can change according
    to numpy.array_split.

    :param tuple shape: Shape tuple (2-element) of interferogram.
    :param int nrows: Number of rows of tiles
    :param int ncols: Number of columns of tiles

    :return: List of Tile class instances.
    :rtype: list
    """

    if len(shape) != 2:
        raise ValueError('shape must be a length 2 tuple')

    no_y, no_x = shape

    if ncols > no_x or nrows > no_y:
        raise ValueError('nrows/cols must be greater than ifg dimensions')
    col_arr = np.array_split(range(no_x), ncols)
    row_arr = np.array_split(range(no_y), nrows)
    return [Tile(i, (r[0], c[0]), (r[-1]+1, c[-1]+1)) for i, (r, c) in enumerate(product(row_arr, col_arr))]