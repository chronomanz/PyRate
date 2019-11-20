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
This Python module contains utilities to parse PyRate configuration
files. It also includes numerous general constants relating to options
in configuration files. Examples of PyRate configuration files are
provided in the configs/ directory
"""
import os
import pathlib
from os.path import splitext

from pyrate.constants import OUT_DIR, TMPDIR
from pyrate.default_parameters import PYRATE_DEFAULT_CONFIGRATION


def get_config_params(config_file_path):
    """
    Reads the parameters file provided by the user and converts it into
    a dictionary.
    Args:
        path: Absolute path to the parameters file.
    Returns:
       A dictionary of parameters.
    """
    # TODO Fix this merge multiple steps for reading the input file

    params = _parse_conf_file(config_file_path)
    params[TMPDIR] = os.path.join(os.path.abspath(params[OUT_DIR]), 'tmpdir')

    return params


def set_parameter_value(data_type, input_value, default_value):
    if len(input_value) < 1:
        input_value = None
    if input_value is not None:
        return data_type(input_value)
    return default_value


def validate_parameter_value(input_name, input_value, min_value=None, max_value=None,possible_values=None):
    if min_value is not None:
        if input_value < min_value:
            print("Invalid value for "+input_name+" supplied: "+input_value+". Please provided a valid value greater than "+min_value+".")
            return False
    if max_value is not None:
        if input_value > max_value:
            print("Invalid value for "+input_name+" supplied: "+input_value+". Please provided a valid value less than "+max_value+".")
            return False
    if possible_values is not None:
        if input_value not in possible_values:
            print("Invalid value for " + input_name + " supplied: " + input_value + ". Please provided a valid value from with in: " + str(possible_values) + ".")
    return True


def _parse_conf_file(config_file_path):
    """
    Converts the parameters from their text form into a dictionary.
    Args:
        config_file_content: Parameters as text.
    Returns:
        A dictionary of parameters.
    kvpair = [(e[0].rstrip(":"), e[1]) for e in config_file_content if len(e) == 2] + [(e[0].rstrip(":"), None) for e in config_file_content if len(e) == 1]
    parameters = dict(kvpair)
    """

    config_file_content = ''
    with open(config_file_path, 'r') as inputFile:
        for line in inputFile:
            config_file_content += line
    config_file_content = [ln for ln in config_file_content.split('\n')]

    parameters = {}
    # TODO replace manual parsing with configparser module
    for line in config_file_content:
        line = line.strip()
        if len(line) > 1:
            if line[0] not in "#":
                if ":" in line:
                    config_key, config_value = line.split(":")
                    config_key = config_key.strip()
                    config_value = config_value.strip()
                    parameters[config_key] = config_value

    # handle control parameters
    for default_parameter in PYRATE_DEFAULT_CONFIGRATION:
        parameter_name = default_parameter["Name"]
        parameters[parameter_name] = set_parameter_value(default_parameter["DataType"], parameters[parameter_name], default_parameter["DefaultValue"])
        validate_parameter_value(parameter_name, parameters[parameter_name], default_parameter["MinValue"], default_parameter["MaxValue"], default_parameter["PossibleValues"])

    # handle file paths
    ROOT = pathlib.Path(config_file_path).parents[0]
    parameters["ROOT"] = ROOT
    parameters["BASE_INTERFEROGRAM_PATHS"] = []

    for path_str in pathlib.Path(ROOT / parameters["IFG_FILE_LIST"]).read_text().split('\n'):
        if len(path_str) > 1:
            path = pathlib.Path(path_str)
            # TODO replace the dependency of path being a str with pathlib module
            parameters["BASE_INTERFEROGRAM_PATHS"].append(str(ROOT / parameters["OBS_DIR"] / path))

    return parameters


def get_dest_paths(base_paths, crop, params, looks):
    """
    Determines the full path names for the destination multilooked files

    :param list base_paths: original interferogram paths
    :param int crop: Crop option applied
    :param dict params: Parameters dictionary
    :param int looks: number of range looks applied

    :return: full path names for destination files
    :rtype: list
    """
    dest_mlooked_ifgs = []

    for q in base_paths:
        tiff_file_name = os.path.basename(q).split('.')[0] + "_" + os.path.basename(q).split('.')[1] + ".tif"

        base, ext = splitext(tiff_file_name)
        new_tiff_file_name = "{base}_{looks}rlks_{crop}cr{ext}".format(base=base, looks=looks, crop=crop, ext=ext)

        dest_mlooked_ifgs.append(new_tiff_file_name)
    return [str(params["ROOT"]/os.path.join(params[OUT_DIR], p)) for p in dest_mlooked_ifgs]


def get_ifg_paths(params):
    """
    Read the configuration file, extract interferogram file list and determine
    input and output interferogram path names.

    :param str config_file: Configuration file path

    :return: ifgs_paths: List of unwrapped inteferograms
    :return: dest_paths: List of multi-looked and cropped geotifs
    :return: params: Dictionary corresponding to the config file
    :rtype: list
    :rtype: list
    :rtype: dict
    """

    xlks, _, crop = params["IFG_LKSX"], params["IFG_LKSY"], params["IFG_CROP_OPT"]

    base_unw_paths = params["BASE_INTERFEROGRAM_PATHS"]
    # dest_paths are tifs that have been coherence masked (if enabled),
    #  cropped and multilooked

    if "tif" in base_unw_paths[0].split(".")[1]:
        dest_paths = get_dest_paths(base_unw_paths, crop, params, xlks)
        for i, dest_path in enumerate(dest_paths):
            dest_paths[i] = dest_path.replace("_tif", "")
    else:
        dest_paths = get_dest_paths(base_unw_paths, params["IFG_CROP_OPT"], params, params["IFG_LKSX"])

    return base_unw_paths, dest_paths


if __name__ == "__main__":
    config_file_path = "C:/Users/sheec/Desktop/Projects\PyRate/input_parameters.conf"
    parameters = get_config_params(config_file_path)

    for parameter in parameters:
        print(parameter, ": ", parameters[parameter])
