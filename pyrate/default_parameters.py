PYRATE_DEFAULT_CONFIGRATION = [
    {
        "Name": "PROCESSOR",
        "DataType": int,
        "DefaultValue": None,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1]
    },
    {
        "Name": "NO_DATA_AVERAGING_THRESHOLD",
        "DataType": float,
        "DefaultValue": 0.0,
        "MinValue": 0,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "NO_DATA_VALUE",
        "DataType": float,
        "DefaultValue": 0.0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "NAN_CONVERSION",
        "DataType": int,
        "DefaultValue": 0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1]
    },
    {
        "Name": "PARALLEL",
        "DataType": int,
        "DefaultValue": 0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1, 2]
    },
    {
        "Name": "PROCESSES",
        "DataType": int,
        "DefaultValue": 8,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "COH_MASK",
        "DataType": int,
        "DefaultValue": 0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1]
    },
    {
        "Name": "COH_THRESH",
        "DataType": float,
        "DefaultValue": 0.1,
        "MinValue": 0.0,
        "MaxValue": 1.0,
        "PossibleValues": None
    },
    {
        "Name": "IFG_CROP_OPT",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": 1,
        "MaxValue": 4,
        "PossibleValues": [1, 2, 3, 4]
    },
    {
        "Name": "IFG_LKSX",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "IFG_LKSY",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "IFG_XFIRST",
        "DataType": float,
        "DefaultValue": None,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "IFG_XLAST",
        "DataType": float,
        "DefaultValue": None,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "IFG_YFIRST",
        "DataType": float,
        "DefaultValue": None,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "IFG_YLAST",
        "DataType": float,
        "DefaultValue": None,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "REFX",
        "DataType": float,
        "DefaultValue": -1,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "REFY",
        "DataType": float,
        "DefaultValue": -1,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "REFNX",
        "DataType": int,
        "DefaultValue": 10,
        "MinValue": 1,
        "MaxValue": 50,
        "PossibleValues": None
    },
    {
        "Name": "REFNY",
        "DataType": int,
        "DefaultValue": 10,
        "MinValue": 1,
        "MaxValue": 50,
        "PossibleValues": None
    },
    {
        "Name": "REF_CHIP_SIZE",
        "DataType": int,
        "DefaultValue": 21,
        "MinValue": 1,
        "MaxValue": 101,
        "PossibleValues": None,
        "Note": "Must be an odd number."
    },
    {
        "Name": "REF_MIN_FRAC",
        "DataType": float,
        "DefaultValue": 0.5,
        "MinValue": 0.0,
        "MaxValue": 1.0,
        "PossibleValues": None
    },
    {
        "Name": "REF_EST_METHOD",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1, 2]
    },
    {
        "Name": "ORBITAL_FIT",
        "DataType": int,
        "DefaultValue": 0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1]
    },
    {
        "Name": "ORBITAL_FIT_METHOD",
        "DataType": int,
        "DefaultValue": 2,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1,2]
    },
    {
        "Name": "ORBITAL_FIT_DEGREE",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1, 2, 3]
    },
    {
        "Name": "ORBITAL_FIT_LOOKS_X",
        "DataType": int,
        "DefaultValue": 10,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "ORBITAL_FIT_LOOKS_Y",
        "DataType": int,
        "DefaultValue": 10,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "APSEST",
        "DataType": int,
        "DefaultValue": 0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1]
    },
    {
        "Name": "SLPF_METHOD",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1, 2]
    },
    {
        "Name": "SLPF_CUTOFF",
        "DataType": float,
        "DefaultValue": 1.0,
        "MinValue": 0.001,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "SLPF_ORDER",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1, 2, 3]
    },
    {
        "Name": "SLPF_NANFILL",
        "DataType": int,
        "DefaultValue": 0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1]
    },
    {
        "Name": "SLPF_NANFILL_METHOD",
        "DataType": str,
        "DefaultValue": "cubic",
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": ["linear", "nearest", "cubic"]
    },
    {
        "Name": "TLPF_METHOD",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1, 2, 3]
    },
    {
        "Name": "TLPF_CUTOFF",
        "DataType": float,
        "DefaultValue": 1.0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "TLPF_PTHR",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "TIME_SERIES_CAL",
        "DataType": int,
        "DefaultValue": 0,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [0, 1]
    },
    {
        "Name": "TIME_SERIES_METHOD",
        "DataType": int,
        "DefaultValue": 2,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1, 2]
    },
    {
        "Name": "TIME_SERIES_SM_ORDER",
        "DataType": int,
        "DefaultValue": None,
        "MinValue": None,
        "MaxValue": None,
        "PossibleValues": [1, 2]
    },
    {
        "Name": "TIME_SERIES_SM_FACTOR",
        "DataType": float,
        "DefaultValue": -1.0,
        "MinValue": -5.0,
        "MaxValue": 0,
        "PossibleValues": None
    },
    {
        "Name": "TIME_SERIES_PTHRESH",
        "DataType": int,
        "DefaultValue": 3,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "LR_NSIG",
        "DataType": int,
        "DefaultValue": 2,
        "MinValue": 1,
        "MaxValue": 10,
        "PossibleValues": None
    },
    {
        "Name": "LR_PTHRESH",
        "DataType": int,
        "DefaultValue": 3,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "LR_MAXSIG",
        "DataType": int,
        "DefaultValue": 10,
        "MinValue": 0,
        "MaxValue": 1000,
        "PossibleValues": None
    },
    {
        "Name": "SUB_ROW",
        "DataType": int,
        "DefaultValue": 4,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    },
    {
        "Name": "SUB_COL",
        "DataType": int,
        "DefaultValue": 1,
        "MinValue": 1,
        "MaxValue": None,
        "PossibleValues": None
    }
]
