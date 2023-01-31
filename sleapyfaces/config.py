
from config import config
from config.configuration_set import ConfigurationSet
import logging
from pathvalidate._filename import is_valid_filename
import pathlib

default_config = {
    "Files": {
        "FileTypes": {
            "ExperimentEvents": "csv",
            "sleap": "hdf5",
            "Video": "mp4",
            "ExperimentSetup": "json"
        },
        "FileNaming": {
            "ExperimentEvents": "*_events.csv",
            "sleap": "*.h5",
            "Video": "*.mp4",
            "ExperimentSetup": "*.json"
        },
        "FileGlob": {
            "ExperimentEvents": True,
            "sleap": True,
            "Video": True,
            "ExperimentSetup": True
        }
    },
    "Prefixes": {
        "Project": None,
        "Experiment": "week",
    },
    "TrialEvents": {
        "TrackedData": ["Speaker_on", "LED590_on"],
        "Reduced": [False, True],
        "start_buffer": 10000,
        "end_buffer": 13000
    },
    "ExperimentEvents" : "columns",
    "SLEAP": "datasets",
    "Video": "metadata",
    "ExperimentSetup": {
        "beh_metadata": ["trialArray", "ITIArray"]
    },
    "Logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "filename": None,
    }
}

def set_config(*args, **kwargs) -> ConfigurationSet:
    """Initializes the config file for the project

    Args:
        prefix (str): the prefix for the sleapyfaces config parameters in the environment variables
        separator (str): the separator used to separate the config parameters in the environment variables
        file (str): the path to the config file
        dict (dict): a dictionary of the config file
        interpolate (bool): whether to interpolate the config file

    Returns:
        configuration (ConfigurationSet): the hierarchical configuration set for the project
    """
    configs = []
    interpolate_config = False

    for arg in [*list(args), *list(kwargs.values())]:
        if is_valid_filename(arg):
            if pathlib.Path(arg).suffix in ["json", "JSON"]:
                configs.append(('json', arg, True))
            elif pathlib.Path(arg).suffix in ["yaml", "yml", "YAML", "YML"]:
                configs.append(('yaml', arg, True))
            elif pathlib.Path(arg).suffix in ["toml", "tml", "TOML", "TML"]:
                configs.append(('toml', arg, True))
            elif pathlib.Path(arg).suffix in ["ini", "INI"]:
                configs.append(('ini', arg, True))
            else:
                raise ValueError(f"Invalid config file type: {arg} \nMust be one of [json, yaml, yml, toml, tml, ini]")
        elif isinstance(arg, dict):
            configs.append(('dict', arg))
        elif isinstance(arg, str) and arg.isupper():
            configs.append(('env', arg, kwargs.get("config_separator", "__")))
        elif isinstance(arg, str) and not arg.isalnum():
            configs.append(('env', kwargs.get("config_prefix", "sleapyfaces"), arg))
        elif isinstance(arg, bool):
            interpolate_config = arg

    configs.append(('dict', default_config))
    configuration = config(*configs, interpolate=interpolate_config)

    logging.basicConfig(
        filename = configuration["Logging"]["filename"],
        format=configuration["Logging"]["format"],
        level=configuration["Logging"]["level"]
    )

    return configuration
