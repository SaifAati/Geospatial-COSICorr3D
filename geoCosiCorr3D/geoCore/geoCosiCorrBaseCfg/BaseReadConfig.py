# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022

import yaml
import json
import click


class ConfigReader:

    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._read_config(config_file=self.config_file)

    @staticmethod
    def _read_config(config_file: str) -> None:
        with open(config_file) as fp:
            config = yaml.full_load(fp)
        return config

    @property
    def get_config(self):
        return self.config


def parse_inputs(input_file):
    import json

    f = open(input_file)
    input_dict = json.load(f)
    f.close()
    return input_dict


class IngestInputs(object):
    def __init__(self, dict):

        self.raw_img_path = None
        self.metadata_file_path = None
        self.metadata_type = None
        self.sensor = None
        self.workspace_folder = None
        self.output_ortho_path = None
        self.ortho_gsd = None
        self.output_trans_path = None
        self.dem_path = None
        self.ref_ortho_path = None
        self.log_name = None
        self.correction_model_file = None
        self.optimize = None
        for key in dict:
            value = dict[key]
            if value == '':
                value = None
            setattr(self, key, value)


class IngestgeoOrthoWfConfig:
    def __init__(self, input, config):
        config['ortho_params']['method']['metadata'] = input.metadata_file_path
        config['ortho_params']['method']['method_type'] = input.metadata_type
        config['ortho_params']['method']['sensor'] = input.sensor
        config['ortho_params']['method']['corr_model'] = input.correction_model_file

        config['sat_model_params']['sat_model'] = input.metadata_type
        config['sat_model_params']['metadata'] = input.metadata_file_path
        config['sat_model_params']['sensor'] = input.sensor

        if input.ortho_gsd is not None:
            config['ortho_params']['GSD'] = input.ortho_gsd

        self.ortho_params = config['ortho_params']
        self.opt_params = config['opt_params']
        self.sat_model_params = config['sat_model_params']
        self.opt_corr_config = config['opt_corr_config']
        self.feature_points_params = config['feature_points_params']


def load_config(argToConfig, config_path):
    """Loads a configuration file into a dictionary."""
    f = open(config_path, )
    config = json.load(f)
    f.close()

    options = {}
    for arg in argToConfig:
        isArg = arg[0] == '$'
        fullPath = argToConfig[arg].split('.')
        current = config
        for dir in fullPath:
            if dir not in current:
                if isArg: raise click.UsageError(f'required argument "{arg[1:]}" is missing from configuration file.')
                break
            current = current[dir]
        else:
            options[arg] = current
    return options
