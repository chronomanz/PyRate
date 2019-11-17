from collections import namedtuple

import numpy as np

__version__ = "0.4.0"
LOG_LEVEL = 'DEBUG'
SIXTEEN_DIGIT_EPOCH_PAIR = r'\d{8}-\d{8}'
TWELVE_DIGIT_EPOCH_PAIR = r'\d{6}-\d{6}'
EIGHT_DIGIT_EPOCH = r'\d{8}'
IFG_FILE_LIST = 'IFG_FILE_LIST'
PROCESSOR = 'PROCESSOR'
OBS_DIR = 'OBS_DIR'
OUT_DIR = 'OUT_DIR'
DEM_FILE = 'DEM_FILE'
DEM_HEADER_FILE = 'DEM_HEADER_FILE'
SLC_DIR = 'SLC_DIR'
SLC_FILE_LIST = 'SLC_FILE_LIST'
INPUT_IFG_PROJECTION = 'INPUT_IFG_PROJECTION'
NO_DATA_VALUE = 'NO_DATA_VALUE'
NO_DATA_AVERAGING_THRESHOLD = 'NO_DATA_AVERAGING_THRESHOLD'
NAN_CONVERSION = 'NAN_CONVERSION'
IFG_CROP_OPT = 'IFG_CROP_OPT'
IFG_LKSX = 'IFG_LKSX'
IFG_LKSY = 'IFG_LKSY'
IFG_XFIRST = 'IFG_XFIRST'
IFG_XLAST = 'IFG_XLAST'
IFG_YFIRST = 'IFG_YFIRST'
IFG_YLAST = 'IFG_YLAST'
REFX = 'REFX'
REFY = 'REFY'
REFNX = "REFNX"
REFNY = "REFNY"
REF_CHIP_SIZE = 'REF_CHIP_SIZE'
REF_MIN_FRAC = 'REF_MIN_FRAC'
REF_EST_METHOD = 'REF_EST_METHOD'
COH_MASK = 'COH_MASK'
COH_THRESH = 'COH_THRESH'
COH_FILE_DIR = 'COH_FILE_DIR'
COH_FILE_LIST = 'COH_FILE_LIST'
APS_CORRECTION = 'APS_CORRECTION'
APS_METHOD = 'APS_METHOD'
APS_INCIDENCE_MAP = 'APS_INCIDENCE_MAP'
APS_INCIDENCE_EXT = 'APS_INCIDENCE_EXT'
APS_ELEVATION_MAP = 'APS_ELEVATION_MAP'
APS_ELEVATION_EXT = 'APS_ELEVATION_EXT'
ORBITAL_FIT = 'ORBITAL_FIT'
ORBITAL_FIT_METHOD = 'ORBITAL_FIT_METHOD'
ORBITAL_FIT_DEGREE = 'ORBITAL_FIT_DEGREE'
ORBITAL_FIT_LOOKS_X = 'ORBITAL_FIT_LOOKS_X'
ORBITAL_FIT_LOOKS_Y = 'ORBITAL_FIT_LOOKS_Y'
LR_NSIG = 'LR_NSIG'
LR_PTHRESH = 'LR_PTHRESH'
LR_MAXSIG = 'LR_MAXSIG'
APSEST = 'APSEST'
TLPF_METHOD = 'TLPF_METHOD'
TLPF_CUTOFF = 'TLPF_CUTOFF'
TLPF_PTHR = 'TLPF_PTHR'
SLPF_METHOD = 'SLPF_METHOD'
SLPF_CUTOFF = 'SLPF_CUTOFF'
SLPF_ORDER = 'SLPF_ORDER'
SLPF_NANFILL = 'SLPF_NANFILL'
SLPF_NANFILL_METHOD = 'SLPF_NANFILL_METHOD'
TIME_SERIES_CAL = 'TIME_SERIES_CAL'
TIME_SERIES_METHOD = 'TIME_SERIES_METHOD'
TIME_SERIES_PTHRESH = 'TIME_SERIES_PTHRESH'
TIME_SERIES_SM_ORDER = 'TIME_SERIES_SM_ORDER'
TIME_SERIES_SM_FACTOR = 'TIME_SERIES_SM_FACTOR'
PARALLEL = 'PARALLEL'
PROCESSES = 'PROCESSES'
INDEPENDENT_METHOD = 1
NETWORK_METHOD = 2
PLANAR = 1
QUADRATIC = 2
PART_CUBIC = 3
ORB_METHOD_NAMES = {INDEPENDENT_METHOD: 'INDEPENDENT', NETWORK_METHOD: 'NETWORK'}
ORB_DEGREE_NAMES = {PLANAR: 'PLANAR',  QUADRATIC: 'QUADRATIC', PART_CUBIC: 'PART CUBIC'}
TMPDIR = 'TMPDIR'
GAMMA = 1
ROIPAC = 0
MASTER_PROCESS = 0
GDAL_WARP_MEMORY_LIMIT = 2**10
LOW_FLOAT32 = np.finfo(np.float32).min*1e-10
MINIMUM_CROP = 1
MAXIMUM_CROP = 2
CUSTOM_CROP = 3
ALREADY_SAME_SIZE = 4
CROP_OPTIONS = [MINIMUM_CROP, MAXIMUM_CROP, CUSTOM_CROP, ALREADY_SAME_SIZE]
GRID_TOL = 1e-6
CustomExts = namedtuple('CustExtents', ['xfirst', 'yfirst', 'xlast', 'ylast'])