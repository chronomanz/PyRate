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
This Python module implements a reference phase estimation algorithm.
"""
import logging
import os

from joblib import Parallel, delayed
import numpy as np

from pyrate.core import ifgconstants as ifc, config as cf
from pyrate.core.shared import joblib_log_level, nanmedian, check_correction_status, Ifg
from pyrate.core import mpiops

log = logging.getLogger(__name__)

MASTER_PROCESS = 0

def estimate_ref_phase(ifg_paths, params, refpx, refpy, preread_ifgs=None):
    """
    Wrapper function for reference phase estimation and interferogram
    correction.

    :param list ifgs: List of interferogram objects
    :param dict params: Dictionary of configuration parameters
    :param int refpx: Reference pixel X found by ref pixel method
    :param int refpy: Reference pixel Y found by ref pixel method

    :return: ref_phs: Numpy array of reference phase values of size (nifgs, 1)
    :rtype: ndarray
    :return: ifgs: Reference phase data is removed interferograms in place
    """
    # Bren: remove the _check method that was in this module and replace with
    #   the one in shared.py. 
    
    # Bren: it might be nicer to move this wrapper function to process.py -
    #   do the correction check, branching based on ref_est_method and saving
    #   in process.py and just call est_ref_phase_method1 and 
    #   est_ref_phase_method2 in this module.

    # Check if IFGs already corrected.
    if preread_ifgs and mpiops.run_once(check_correction_status, preread_ifgs, 
                                        ifc.PYRATE_REF_PHASE):  
        log.info('Skipped: interferograms already corrected')
        return

    if params[cf.REF_EST_METHOD] == 1:
        ref_phs = est_ref_phase_method1(ifg_paths, params)

    elif params[cf.REF_EST_METHOD] == 2:
        ref_phs = est_ref_phase_method2(ifg_paths, params, refpx, refpy)
    else:
        raise ReferencePhaseError('No such option. Use refest=1 or 2')

    # Bren: This is pulled straight out of process._ref_phase_estimation.
    #   This should be refactored. Whether that's a function in this module
    #   or return the ifgs/ref_phs and save in process.py is undecided.

    # Save reference phase numpy arrays to disk.
    ref_phs_file = os.path.join(params[cf.TMPDIR], 'ref_phs.npy')
    if mpiops.rank == MASTER_PROCESS:
        collected_ref_phs = np.zeros(len(ifg_paths), dtype=np.float64)
        process_indices = mpiops.array_split(range(len(ifg_paths)))
        collected_ref_phs[process_indices] = ref_phs
        for r in range(1, mpiops.size):
            process_indices = mpiops.array_split(range(len(ifg_paths)), r)
            this_process_ref_phs = np.zeros(shape=len(process_indices),
                                            dtype=np.float64)
            mpiops.comm.Recv(this_process_ref_phs, source=r, tag=r)
            collected_ref_phs[process_indices] = this_process_ref_phs
        np.save(file=ref_phs_file, arr=ref_phs)
    else:
        mpiops.comm.Send(ref_phs, dest=MASTER_PROCESS, tag=mpiops.rank)
    log.info('Finished reference phase estimation')
    
    return ref_phs, ifg_paths

def est_ref_phase_method2(ifg_paths, params, refpx, refpy):
    """
    Reference phase estimation using method 2. Reference phase is the
    median calculated with a patch around the supplied reference pixel.

    :param list ifgs: List of interferogram objects
    :param dict params: Dictionary of configuration parameters
    :param int refpx: Reference pixel X found by ref pixel method
    :param int refpy: Reference pixel Y found by ref pixel method

    :return: ref_phs: Numpy array of reference phase values of size (nifgs, 1)
    :rtype: ndarray
    :return: ifgs: Reference phase data is removed interferograms in place
    """
    # Bren: see the comment I made in est_ref_phase_method2 - follow the 
    #   same pattern.
    half_chip_size = int(np.floor(params[cf.REF_CHIP_SIZE] / 2.0))
    chipsize = 2 * half_chip_size + 1
    thresh = chipsize * chipsize * params[cf.REF_MIN_FRAC]

    def _inner(ifg_paths):
        ifgs = [Ifg(ifg_path) for ifg_path in ifg_paths]
        for ifg in ifgs:
            if not ifg.is_open:
                ifg.open()

        phase_data = [i.phase_data for i in ifgs]
        if params[cf.PARALLEL]:
            ref_phs = Parallel(n_jobs=params[cf.PROCESSES],
                               verbose=joblib_log_level(cf.LOG_LEVEL))(
                delayed(_est_ref_phs_method2)(p, half_chip_size,
                                              refpx, refpy, thresh)
                for p in phase_data)

            for n, ifg in enumerate(ifgs):
                ifg.phase_data -= ref_phs[n]
        else:
            ref_phs = np.zeros(len(ifgs))
            for n, ifg in enumerate(ifgs):
                ref_phs[n] = \
                    _est_ref_phs_method2(phase_data[n], half_chip_size,
                                         refpx, refpy, thresh)
                ifg.phase_data -= ref_phs[n]

        for ifg in ifgs:
            _update_phase_metadata(ifg)
            ifg.close()
        return ref_phs
    
    process_ifgs_paths = mpiops.array_split(ifg_paths)
    ref_phs = _inner(process_ifgs_paths)
    return ref_phs   

def _est_ref_phs_method2(phase_data, half_chip_size, refpx, refpy, thresh):
    """
    Convenience function for ref phs estimate method 2 parallelisation
    """
    patch = phase_data[refpy - half_chip_size: refpy + half_chip_size + 1,
                       refpx - half_chip_size: refpx + half_chip_size + 1]
    patch = np.reshape(patch, newshape=(-1, 1), order='F')
    nanfrac = np.sum(~np.isnan(patch))
    if nanfrac < thresh:
        raise ReferencePhaseError('The data window at the reference pixel '
                                  'does not have enough valid observations. '
                                  'Actual = {}, Threshold = {}.'.format(
                                          nanfrac, thresh))
    ref_ph = nanmedian(patch)
    return ref_ph


def est_ref_phase_method1(ifg_paths, params):
    """
    Reference phase estimation using method 1. Reference phase is the
    median of the whole interferogram image.

    :param list ifgs: List of interferogram objects
    :param dict params: Dictionary of configuration parameters

    :return: ref_phs: Numpy array of reference phase values of size (nifgs, 1)
    :rtype: ndarray
    :return: ifgs: Reference phase data is removed interferograms in place
    """
    # Bren: we can preserve the majority of this method, the major changes
    #   are putting the body of the function in the _inner function and
    #   changing the params to take ifg_paths instead of ifg objects (MPI has to
    #   communicate via pickle - it's a lot easier to pickle and pass a 
    #   string to processes rather than a complex Ifg object) and then each
    #   procesor will open the Ifgs and operate on them.

    def _inner(ifg_paths):
        proc_ifgs = [Ifg(ifg_path) for ifg_path in ifg_paths]
        for ifg in proc_ifgs:
            if not ifg.is_open:
                ifg.open()

        ifg_phase_data_sum = np.zeros(proc_ifgs[0].shape, dtype=np.float64)
        phase_data = [i.phase_data for i in proc_ifgs]
        for ifg in proc_ifgs:
            ifg_phase_data_sum += ifg.phase_data

        comp = np.isnan(ifg_phase_data_sum)
        comp = np.ravel(comp, order='F')
        if params[cf.PARALLEL]:
            log.info("Calculating ref phase using multiprocessing")
            ref_phs = Parallel(n_jobs=params[cf.PROCESSES], 
                               verbose=joblib_log_level(cf.LOG_LEVEL))(
                delayed(_est_ref_phs_method1)(p, comp)
                for p in phase_data)
            for n, ifg in enumerate(proc_ifgs):
                ifg.phase_data -= ref_phs[n]
        else:
            log.info("Calculating ref phase")
            ref_phs = np.zeros(len(proc_ifgs))
            for n, ifg in enumerate(proc_ifgs):
                ref_phs[n] = _est_ref_phs_method1(ifg.phase_data, comp)
                ifg.phase_data -= ref_phs[n]

        for ifg in proc_ifgs:
            _update_phase_metadata(ifg)
            ifg.close()

        return ref_phs

    process_ifg_paths = mpiops.array_split(ifg_paths)
    ref_phs = _inner(process_ifg_paths)
    return ref_phs

def _est_ref_phs_method1(phase_data, comp):
    """
    Convenience function for ref phs estimate method 1 parallelisation
    """
    ifgv = np.ravel(phase_data, order='F')
    ifgv[comp == 1] = np.nan
    return nanmedian(ifgv)

def _update_phase_metadata(ifg):
    ifg.meta_data[ifc.PYRATE_REF_PHASE] = ifc.REF_PHASE_REMOVED
    ifg.write_modified_phase()

class ReferencePhaseError(Exception):
    """
    Generic class for errors in reference phase estimation.
    """
    pass
