#   This Python module is part of the PyRate software package.
#
#   Copyright 2019 Geoscience Australia
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
import configparser
import pathlib

class Configuration(object):
    def __init__(self, config_file):

        file_content = "[root]\n"
        with open(config_file) as f:
            for line in f.readlines():
                if ":" in line:
                    file_content = file_content + line

        config = configparser.RawConfigParser()
        config.read_string(file_content)

        # TODO atm accept relative path wrt input file
        #  get rid of this functionality this causes ambiguity
        self.root = pathlib.Path(config_file).parents[0]

        self.obs_dir = self.root/pathlib.Path(config["root"]["OBS_DIR"])

        self.dem_path = self.root / pathlib.Path(config["root"]["DEM_FILE"])
        self.dem_header_path = self.root/pathlib.Path(config["root"]["DEM_HEADER_FILE"])


        self.header_paths = []

        with open(self.root/config["root"]["SLC_FILE_LIST"], 'r') as f:
            for line in f.readlines():
                self.header_paths.append(self.root/line.strip())

        self.base_interferogram_paths = []

        for path_str in pathlib.Path(self.root/config["root"]["IFG_FILE_LIST"]).read_text().split('\n'):
            if len(path_str) > 1:
                path = pathlib.Path(path_str)
                # TODO slowly replace the dependency of path being a str with pathlib module
                self.base_interferogram_paths.append(str(self.obs_dir/path))

        self.destination_path = pathlib.Path(config["root"]["OBS_DIR"])
        self.destination_path.mkdir(parents=True, exist_ok=True)
        self.output_tiff_list = self.destination_path.joinpath('tiff_list.txt')

        # # cropping parameters
        # # IFG_LKSX, IFG_LKSY, IFG_CROP_OPT
        # self.xlooks = config["root"]["ifglksx"]
        # self.ylooks = config["root"]["ifglksy"]
        # self.crop = config["root"]["ifgcropopt"]
        #
        # self.ifgxfirst = config["root"]["ifgxfirst"]
        # self.ifgyfirst = config["root"]["ifgyfirst"]
        # self.ifgxlast = config["root"]["ifgxlast"]
        # self.ifgylast = config["root"]["ifgylast"]
        #
        # self.thresh = config["root"]["noDataAveragingThreshold"]
        # self.coherence_thresh = config["root"]["cohthresh"]
        #
        self.processor = config["root"]["processor"]

    def __str__(self):
        pprint_string = ""
        for key, value in self.__dict__.items():
            if type(value) is list:
                pprint_string = pprint_string + str(key) + ": [" "\n"
                for item in value:
                    pprint_string = pprint_string + "    " + str(item) + "\n"
                pprint_string = pprint_string + "]" "\n"
            else:
                pprint_string = pprint_string + str(key) + ": " + str(value) + "\n"
        return pprint_string


if __name__ == "__main__":
    config_file = "/Users/sheeced/Desktop/Projects/PyRate/input_parameters.conf"
    configration_settings = Configuration(config_file)
    print(configration_settings)
