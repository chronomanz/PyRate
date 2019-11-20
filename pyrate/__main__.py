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
This Python module defines executable run configuration for the PyRate software
"""
import os
import logging
import argparse
from argparse import RawTextHelpFormatter
from pyrate import conv2tif, prepifg, process, merge, configuration as cf
import time


def conv2tif_handler(config_file):
    """
    Convert interferograms to geotiff.
    """
    config_file = os.path.abspath(config_file)
    params = cf.get_config_params(config_file)
    conv2tif.main(params)


def prepifg_handler(config_file):
    """
    Perform multilooking and cropping on geotiffs.
    """
    config_file = os.path.abspath(config_file)
    params = cf.get_config_params(config_file)
    prepifg.main(params)


def process_handler(config_file):
    """
    Time series and linear rate computation.
    """
    config_file = os.path.abspath(config_file)
    params = cf.get_config_params(config_file)
    process.main(params)


def merge_handler(config_file):
    """
    Reassemble computed tiles and save as geotiffs.
    """
    config_file = os.path.abspath(config_file)
    params = cf.get_config_params(config_file)
    merge.main(params)


CLI_DESCRIPTION = """
PyRate workflow: 

    Step 1: conv2tif
    Step 2: prepifg
    Step 3: process
    Step 4: merge 

Refer to https://geoscienceaustralia.github.io/PyRate/usage.html for 
more details.
"""


def main():
    start_time = time.time()
    log.debug("Starting PyRate")
    log.info("Starting PyRate")

    parser = argparse.ArgumentParser(prog='pyrate', description=CLI_DESCRIPTION, add_help=True, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--verbosity', type=str, default='DEBUG', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Increase output verbosity")

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    # create the parser for the "conv2tif" command
    parser_conv2tif = subparsers.add_parser('conv2tif', help='Convert interferograms to geotiff.', add_help=True)
    parser_conv2tif.add_argument('-f', '--config_file', action="store", type=str, default=None, help="Pass configuration file", required=True)

    # create the parser for the "prepifg" command
    parser_prepifg = subparsers.add_parser('prepifg', help='Perform multilooking and cropping on geotiffs.', add_help=True)
    parser_prepifg.add_argument('-f', '--config_file', action="store", type=str, default=None, help="Pass configuration file", required=True)

    # create the parser for the "process" command
    parser_process = subparsers.add_parser('process', help='Main processing workflow including corrections, time series and stacking computation.', add_help=True)
    parser_process.add_argument('-f', '--config_file', action="store", type=str, default=None, help="Pass configuration file", required=True)

    # create the parser for the "merge" command
    parser_merge = subparsers.add_parser('merge', help="Reassemble computed tiles and save as geotiffs.", add_help=True)
    parser_merge.add_argument('-f', '--config_file', action="store", type=str, default=None, help="Pass configuration file", required=False)

    args = parser.parse_args()
    log.debug("Arguments supplied at command line: ")
    log.debug(args)

    if args.verbosity:
        log.info("Verbosity set to " + str(args.verbosity) + ".")
        log.setLevel(args.verbosity)

    if args.command == "conv2tif":
        conv2tif_handler(args.config_file)

    if args.command == "prepifg":
        prepifg_handler(args.config_file)

    if args.command == "process":
        process_handler(args.config_file)

    if args.command == "merge":
        merge_handler(args.config_file)

    log.debug("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":

    # create default logger
    log = logging.getLogger("rootLogger")
    stream = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream.setFormatter(formatter)
    log.addHandler(stream)

    main()
