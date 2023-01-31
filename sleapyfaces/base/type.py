from sleapyfaces.utils.structs import CustomColumn
from sleapyfaces.utils.normalize import mean_center, pca, z_score
import os
from config import config
from config.configuration_set import ConfigurationSet
import logging
import itertools
from typing import Protocol
from pathvalidate._filename import is_valid_filename
import pandas as pd
import plotly.graph_objects as go
import pickle
from sleapyfaces.config import set_config

class objdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

class DataClassProtocol(Protocol):
    base: str
    name: str
    names: list[str]
    paths: list[str]
    fileStruct: dict[str, str]
    ExprEventsFile: tuple[str, bool] | str
    ExprSetupFile: tuple[str, bool] | str
    SLEAPFile: tuple[str, bool] | str
    VideoFile: tuple[str, bool] | str
    level: str
    tabs: str
    config: dict[str, any]
    data: dict[str, object]
    all_data: pd.DataFrame
    all_scores: pd.DataFrame
    custom_columns: list[CustomColumn]
    all_trials: list[pd.DataFrame]
    numeric_columns: list[str]

    def __init__(
        self,
        name: str,
        base: str,
        file_structure: dict[str, str] | bool = None,
        ExperimentEventsFile: tuple[str, bool] | str = None,
        ExperimentSetupFile: tuple[str, bool] | str = None,
        SLEAPFile: tuple[str, bool] | str = None,
        VideoFile: tuple[str, bool] | str = None,
        tabs: str = "",
        passed_config: dict[str, any] = None,
        prefix: str = None,
        *args,
        **kwargs) -> None:

        self.base = base
        self.name = name
        self.fileStruct = file_structure
        self.ExprEventsFile = ExperimentEventsFile
        self.ExprSetupFile = ExperimentSetupFile
        self.SLEAPFile = SLEAPFile
        self.VideoFile = VideoFile
        self.tabs = tabs
        self.config = passed_config

    def buildColumns(self, columns: list = None, values: list = None):
        pass
    def buildTrials(self, trials: list = None, values: list = None):
        pass
    def meanCenter(self, alldata: bool = False):
        pass
    def zScore(self, alldata: bool = False):
        pass
    def normalize(self, alldata: bool = False):
        pass
    def pca(self, alldata: bool = False):
        pass
    def visualize(self, alldata: bool = False):
        pass
    def save(self, filename: str | pd.HDFStore, title: str = None, all: bool = False):
        pass

class BaseType:
    """Base type for project/experiment/projects

    Args:
        name (str): the name of the project/experiment
        base (str): the base directory of the project/experiment
        file_structure (dict[str, str] | bool): the file structure of the project/experiment. Set to `False` to enable autodetection of subdirectories.
        ExperimentEventsFile (tuple[str, bool] | str): the naming convention for the experimental events files (e.g. ("*_events.csv", True) or ("DAQOutput.csv", False))
        ExperimentSetupFile (tuple[str, bool] | str): the naming convention for the experimental structure files (e.g. ("*_config.json", True) or ("BehMetadata.json", False))
        SLEAPFile (tuple[str, bool] | str): the naming convention for the SLEAP files (e.g. ("*_sleap.h5", True) or ("SLEAP.h5", False))
        VideoFile (tuple[str, bool] | str): the naming convention for the video files (e.g. ("*_video.mp4", True) or ("Video.mp4", False))
        prefix (str): the prefix to use for the project/experiment (e.g. "week") in the dataframes
        *args, **kwargs: Configuration args

		Configuration Args:
			FilePath (str): The full filepath to the configuration file or the relative path to the current working directory without a leading slash


    """

    def __init__(
        self,
        name: str,
        base: str,
        file_structure: dict[str, str] | bool = None,
        ExperimentEventsFile: tuple[str, bool] | str = None,
        ExperimentSetupFile: tuple[str, bool] | str = None,
        SLEAPFile: tuple[str, bool] | str = None,
        VideoFile: tuple[str, bool] | str = None,
        tabs: str = "",
        passed_config: dict[str, any] | ConfigurationSet = None,
        prefix: str = None,
        sublevel: str = "base",
        *args,
        **kwargs) -> None:

        self.base = base
        self.name = name
        self.fileStruct = file_structure
        self.ExprEventsFile = ExperimentEventsFile
        self.ExprSetupFile = ExperimentSetupFile
        self.SLEAPFile = SLEAPFile
        self.VideoFile = VideoFile
        self.tabs = tabs
        self.sublevel = sublevel
        self.prefix = prefix

        if passed_config is None:
            self.config = self._init_config(*args, **kwargs)
        else:
            self.config = passed_config

        self.fileTypes = self.config["Files"]["FileTypes"].as_attrdict()
        self.fileNaming = self.config["Files"]["FileNaming"].as_attrdict()
        self.fileGlob = self.config["Files"]["FileGlob"].as_attrdict()

        if self.prefix is None and self.sublevel is not None:
            if self.sublevel in self.config["Prefixes"]:
                self.prefix = self.config["Prefixes"][self.sublevel]

        if self.fileStruct is None:
            self.fileStruct = self._init_file_structure(self.base, self.prefix)

        if self.fileStruct is not None and self.fileStruct is not False:
            self.names: list[str] = list(self.fileStruct.keys())
            self.paths: list[str] = list(self.fileStruct.values())

        if self.ExprEventsFile is None and not isinstance(self.ExprEventsFile, tuple):
            self.ExprEventsFile = (self.fileNaming.ExperimentEvents, self.fileGlob.ExperimentEvents)
        elif isinstance(self.ExprEventsFile, str) and not isinstance(self.ExprEventsFile, tuple):
            if "ExperimentEvents" in self.fileGlob:
                self.ExprEventsFile = (self.ExprEventsFile, self.fileGlob.ExperimentEvents)
            elif is_valid_filename(self.ExprEventsFile):
                self.ExprEventsFile = (self.ExprEventsFile, False)
            else:
                self.ExprEventsFile = (self.ExprEventsFile, True)

        if self.ExprSetupFile is None and not isinstance(self.ExprSetupFile, tuple):
            self.ExprSetupFile = (self.fileNaming.ExperimentSetup, self.fileGlob.ExperimentSetup)
        elif isinstance(self.ExprSetupFile, str) and not isinstance(self.ExprSetupFile, tuple):
            if "ExperimentSetup" in self.fileGlob:
                self.ExprSetupFile = (self.ExprSetupFile, self.fileGlob.ExperimentSetup)
            elif is_valid_filename(self.ExprSetupFile):
                self.ExprSetupFile = (self.ExprSetupFile, False)
            else:
                self.ExprSetupFile = (self.ExprSetupFile, True)

        if self.SLEAPFile is None and not isinstance(self.SLEAPFile, tuple):
            self.SLEAPFile = (self.fileNaming.sleap, self.fileGlob.sleap)
        elif isinstance(self.SLEAPFile, str) and not isinstance(self.SLEAPFile, tuple):
            if "sleap" in self.fileGlob:
                self.SLEAPFile = (self.SLEAPFile, self.fileGlob.sleap)
            elif is_valid_filename(self.SLEAPFile):
                self.SLEAPFile = (self.SLEAPFile, False)
            else:
                self.SLEAPFile = (self.SLEAPFile, True)

        if self.VideoFile is None and not isinstance(self.VideoFile, tuple):
            self.VideoFile = (self.fileNaming.Video, self.fileGlob.Video)
        elif isinstance(self.VideoFile, str):
            if "Video" in self.fileGlob and not isinstance(self.VideoFile, tuple):
                self.VideoFile = (self.VideoFile, self.fileGlob.Video)
            elif is_valid_filename(self.VideoFile):
                self.VideoFile = (self.VideoFile, False)
            else:
                self.VideoFile = (self.VideoFile, True)

        self.data = {}
        self._init_data()

    def _init_config(self, *args, **kwargs) -> ConfigurationSet:
        configuration = set_config( *args, **kwargs)
        return configuration

    def _init_file_structure(self, base: str = None, prefix: str = None) -> dict[str, any]:
        iterator = {}
        subdirs = os.listdir(base)
        subdirs = [
            subdir for subdir in subdirs if os.path.isdir(os.path.join(base, subdir))
        ]
        subdirs.sort()
        if self.prefix is not None:
            for i, subdir in enumerate(subdirs):
                iterator[f"{self.prefix}_{i}"] = subdir
        elif prefix is not None:
            for i, subdir in enumerate(subdirs):
                iterator[f"{prefix}_{i}"] = subdir
        else:
            for i, subdir in enumerate(subdirs):
                iterator[f"{i}"] = subdir
        return iterator

    def _init_data(self) -> dict[str, DataClassProtocol]:
        data = {"test": DataClassProtocol()}
        self.data: dict[str, DataClassProtocol] = data
        self.all_data = pd.concat([data.all_data for data in self.data.values()], keys=self.data.keys())
        self.all_scores = pd.concat([data.all_scores for data in self.data.values()], keys=self.data.keys())
        self.numeric_columns = self.data[self.names[0]].numeric_columns

    def _rename_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index.names = []
        return df

    def buildColumns(self, columns: list = None, values: list = None):
        """Builds the custom columns for the project and builds the data for each experiment

        Args:
            columns (list[str]): the column titles
            values (list[any]): the data for each column

        Initializes attributes:
            custom_columns (list[CustomColumn]): list of custom columns

        Updates attributes:
            all_data (pd.DataFrame): the data for all experiments concatenated together
            all_scores (pd.DataFrame): the scores for all experiments and trials concatenated together
        """
        logging.info("Building columns...")
        if columns is None and not hasattr(self.data[self.names[0]], "custom_columns"):
            for data in self.data.values():
                data.buildColumns()
        elif len(columns) != len(values) and not isinstance(columns[0], CustomColumn):
            raise ValueError("Number of columns and values must be equal")
        elif isinstance(columns[0], CustomColumn):
            for data in self.data.values():
                self.custom_columns = columns
                data.buildColumns(self.custom_columns)
        else:
            logging.debug(f"{self.tabs}{[(col, val) for col, val in zip(columns, values)]}")
            self.custom_columns = [CustomColumn(col, val) for col, val in zip(columns, values)]
            for data in self.data.values():
                data.buildColumns(self.custom_columns)
        self.custom_columns = self.data[self.names[0]].custom_columns
        self.all_data = pd.concat([data.all_data for data in self.data.values()], keys=self.names)
        self.all_scores = pd.concat([data.all_scores for data in self.data.values()], keys=self.names)

    def buildTrials(
        self,
        TrackedData: list[str] = None,
        Reduced: list[bool] = None,
        start_buffer: int = None,
        end_buffer: int = None,
    ):
        """Parses the data from each experiment into its individual trials

        Args:
            TrackedData (list[str]): The title of the columns from the DAQ data to be tracked
            Reduced (list[bool]): The corresponding boolean for whether the DAQ data is to be reduced (`True`) or not (`False`)
            start_buffer (int, optional): The time in milliseconds before the trial start to capture. Defaults to 10000.
            end_buffer (int, optional): The time in milliseconds after the trial start to capture. Defaults to 13000.

        Initializes attributes:
            all_trials (list[pd.DataFrame]): the list of data frames containing the trial data for each trial for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the data for all experiments and trials concatenated together
            all_scores (pd.DataFrame): the scores for all experiments and trials concatenated together
        """
        if TrackedData is None:
            TrackedData = self.config["TrialEvents"]["TrackedData"]
        if Reduced is None:
            Reduced = self.config["TrialEvents"]["Reduced"]
        if start_buffer is None:
            start_buffer = self.config["TrialEvents"]["start_buffer"]
        if end_buffer is None:
            end_buffer = self.config["TrialEvents"]["end_buffer"]
        if len(TrackedData) != len(Reduced):
            raise ValueError("TrackedData and Reduced must be the same length")
        if TrackedData is None or Reduced is None or start_buffer is None or end_buffer is None:
            raise ValueError("TrackedData, Reduced, start_buffer, and end_buffer must all be specified either directly or in the config file")

        print(self.tabs, "Building trials for:", self.name)

        for data in self.data.values():
            data.buildTrials(TrackedData, Reduced, start_buffer, end_buffer)
        self.all_trials = [trial for data in self.data.values() for trial in data.all_trials]
        self.all_data = pd.concat([data.all_data for data in self.data.values()], keys=self.names)
        self.all_data = self._rename_index(self.all_data)
        self.all_scores = pd.concat([data.all_scores for data in self.data.values()], keys=self.names)

    def meanCenter(self, alldata: bool = False):
        """Recursively mean centers the data for each trial for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the mean centered data for all trials and experiments concatenated together
        """
        if alldata:
            self.all_data = mean_center(self.all_data, self.numeric_columns)
        else:
            for data in self.data.values():
                data.meanCenter()
            self.all_data = mean_center(
                pd.concat(
                    [
                        data.all_data for data in self.data.values()
                    ],
                    keys=self.names
                ),
                self.numeric_columns)
            self.all_data = self._rename_index(self.all_data)

    def zScore(self, alldata: bool = False):
        """Z scores the mean centered data for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the z-scored data for all experiments concatenated together
        """
        if alldata:
            self.all_data = z_score(self.all_data, self.numeric_columns)
        else:
            for data in self.data.values():
                data.zScore()
            self.all_data = z_score(
                pd.concat(
                    [
                        data.all_data for data in self.data.values()
                    ],
                    keys=self.names
                ),
                self.numeric_columns)
            self.all_data = self._rename_index(self.all_data)

    def normalize(self):
        """Runs the mean centering and z scoring functions

        Updates attributes:
            all_data (pd.DataFrame): the normaized data for all experiments concatenated together
        """

        logging.info(f"{self.tabs}Normalizing project {self.name if self.name is not None else ''}")

        for data in self.data.values():
                data.normalize()
        self.all_data = z_score(
                pd.concat(
                    [
                        data.all_data for data in self.data.values()
                    ],
                    keys=self.names
                ),
                self.numeric_columns)
        self.all_data = self._rename_index(self.all_data)

    def runPCA(self, data: pd.DataFrame = None, numeric_columns: list[str] | pd.Index = None):
        """Reduces `all_data` to 2 and 3 dimensions using PCA

        Initializes attributes:
            pcas (dict[str, pd.DataFrame]): a dictionary containing the 2 and 3 dimensional PCA data for each experiment (the keys are 'pca2d', 'pca3d')
        """
        if data is not None and numeric_columns is not None:
            self.pcas = objdict(pca(data, numeric_columns))
        elif data is not None:
            self.pcas = objdict(pca(data, self.numeric_columns))
        else:
            self.pcas = objdict(pca(self.all_data, self.numeric_columns))

    def visualize(self, dimensions: int, normalized: bool = False, color_column: str = "Trial", lines: bool = False, filename: str = None, *args, **kwargs) -> go.Figure:
        """Plots the trajectories from the PCA

        Args:
            filename (str, optional): The filename to save the plot to. Defaults to None.
        """
        if normalized:
            self.runPCA()
        elif not normalized:
            df: pd.DataFrame = pd.concat(self.all_trials, keys=range(len(self.all_trials)))
            df.index.rename(["Trial", "Trial_index"], inplace=True)
            self.runPCA(df.reset_index(), self.numeric_columns)

        self.pcas[f"pca{dimensions}d"][color_column] = self.pcas[f"pca{dimensions}d"][color_column].astype(str)

        if dimensions == 2:
            fig = go.Figure(
                data=[go.Scatter(
                    x=self.pcas[f"pca{dimensions}d"].loc[self.pcas[f"pca{dimensions}d"][color_column] == color]["principal component 1"],
                    y=self.pcas[f"pca{dimensions}d"].loc[self.pcas[f"pca{dimensions}d"][color_column] == color]["principal component 2"],
                    mode = "markers+lines" if lines else "markers",
					customdata=self.pcas[f"pca{dimensions}d"].loc[self.pcas[f"pca{dimensions}d"][color_column] == color],
                    marker=dict(size=4)) for color in self.pcas[f"pca{dimensions}d"][color_column].unique()]
            )
        elif dimensions == 3:
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=self.pcas[f"pca{dimensions}d"].loc[self.pcas[f"pca{dimensions}d"][color_column] == color]["principal component 1"],
                    y=self.pcas[f"pca{dimensions}d"].loc[self.pcas[f"pca{dimensions}d"][color_column] == color]["principal component 2"],
                    z=self.pcas[f"pca{dimensions}d"].loc[self.pcas[f"pca{dimensions}d"][color_column] == color]["principal component 3"],
                    mode = "markers+lines" if lines else "markers",
                    customdata=self.pcas[f"pca{dimensions}d"].loc[self.pcas[f"pca{dimensions}d"][color_column] == color],
                    marker=dict(size=4)) for color in self.pcas[f"pca{dimensions}d"][color_column].unique()]
            )
        else:
            raise ValueError("dimensions must be 2 or 3")

        if filename is not None:
            if filename.lower().endswith('.html'):
                fig.write_html(filename, *args, **kwargs)
            else:
                fig.write_image(filename, *args, **kwargs)

        return fig

    def save(self, filename: str | pd.HDFStore, title: str = None, all: bool = False):
        """Saves the project data to a HDF5 file or HDFStore or a pickle file

        Args:
            filename (str): The filename to save the data to
        """
        if all and isinstance(filename, str):
            if filename.lower().endswith(".h5"):
                with pd.HDFStore(filename) as store:
                    for data in self.data.values():
                        data.save(store, all=True)
                    if self.name is not None:
                        store.put(f"{self.name}/all_data", self.all_data)
                        store.put(f"{self.name}/all_scores", self.all_scores)
                    elif title is not None:
                        store.put(f"projects/{title}/all_data", self.all_data)
                        store.put(f"projects/{title}/all_scores", self.all_scores)
                    else:
                        store.put("all_data", self.all_data)
                        store.put("all_scores", self.all_scores)
            elif filename.lower().endswith(".pickle"):
                with open(filename, "wb") as file:
                    pickle.dump(self, file)
            else:
                raise ValueError("filename must be a HDF5 or pickle file")

        elif isinstance(filename, str):
            if filename.lower().endswith(".h5"):
                with pd.HDFStore(filename) as store:
                    if self.name is not None:
                        store.put(f"{self.name}/all_data", self.all_data)
                        store.put(f"{self.name}/all_scores", self.all_scores)
                    elif title is not None:
                        store.put(f"projects/{title}/all_data", self.all_data)
                        store.put(f"projects/{title}/all_scores", self.all_scores)
                    else:
                        store.put("all_data", self.all_data)
                        store.put("all_scores", self.all_scores)
            elif filename.lower().endswith(".pickle"):
                with open(filename, "wb") as file:
                    pickle.dump(self, file)
            else:
                raise ValueError("filename must be a HDF5 or pickle file")

        if isinstance(filename, pd.HDFStore):
            store = filename
            if self.name is not None:
                store.put(f"{self.name}/all_data", self.all_data)
                store.put(f"projects/{self.name}/all_scores", self.all_scores)
            elif title is not None:
                store.put(f"projects/{title}/all_data", self.all_data)
                store.put(f"projects/{title}/all_scores", self.all_scores)
            else:
                store.put("all_data", self.all_data)
                store.put("all_scores", self.all_scores)

    @property
    def quant_cols(self) -> list[str]:
        """Returns the quantitative columns of the data.

        Returns:
            list[str]: the columns from the data with the target quantitative data.
        """
        return self.numeric_columns

    @property
    def qual_cols(self) -> list[str]:
        """Returns the qualitative columns of the data.

        Returns:
            list[str]: the columns from the data with the qualitative (or rather non-target) data.
        """
        cols = self.all_data.reset_index().columns.to_list()
        cols = [i for i in cols if i not in self.quant_cols]
        return cols

    @property
    def cols(self) -> tuple[list[str], list[str]]:
        """Returns the target and non target columns of the data.

        Returns:
            tuple[list[str], list[str]]: a tuple of column lists, the first being the target columns and the second being the non-target columns.
        """
        return (self.quant_cols, self.qual_cols)

	def current_config(self) -> None:
		print(self.config)
