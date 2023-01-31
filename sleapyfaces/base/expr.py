import warnings
import logging

import numpy as np
import pandas as pd

from sleapyfaces.files import ExprMetadata, EventsData, SLEAPData, VideoMetadata
from sleapyfaces.utils import flatten_list, into_trial_format, reduce_daq
from sleapyfaces.utils.normalize import mean_center, z_score
from sleapyfaces.utils.structs import CustomColumn, FileConstructor, File
from sleapyfaces.base.type import BaseType
from config.configuration_set import ConfigurationSet


class Experiment(BaseType):
    """Class constructor for the Experiment object.

    Args:
        name (str): The name of the experiment.
        files (FileConstructor): The FileConstructor object containing the paths to the experiment files.

    Attributes:
        name (str): The name of the experiment.
        files (FileConstructor): The FileConstructor object containing the paths to the experiment files.
        sleap (SLEAPanalysis): The SLEAPanalysis object containing the SLEAP data.
        beh (BehMetadata): The BehMetadata object containing the behavior metadata.
        video (VideoMetadata): The VideoMetadata object containing the video metadata.
        daq (DAQData): The DAQData object containing the DAQ data.
        numeric_columns (list[str]): A list of the titles of the numeric columns in the SLEAP data.
    """
    def __init__(self,
            name: str,
            base: str,
            ExperimentEventsFile: tuple[str, bool] | str = None,
            ExperimentSetupFile: tuple[str, bool] | str = None,
            SLEAPFile: tuple[str, bool] | str = None,
            VideoFile: tuple[str, bool] | str = None,
            tabs: str = "",
            passed_config: dict[str, any] | ConfigurationSet = None,
            prefix: str = None,
            file_structure: dict | bool = False,
            sublevel: str = None,
         ):

        super().__init__(
            name=name,
            base=base,
            file_structure=file_structure,
            ExperimentEventsFile=ExperimentEventsFile,
            ExperimentSetupFile=ExperimentSetupFile,
            SLEAPFile=SLEAPFile,
            VideoFile=VideoFile,
            tabs=tabs,
            passed_config=passed_config,
            prefix=prefix,
            sublevel=sublevel
        )

    def _init_data(self):
        print(self.ExprEventsFile)
        self.files = FileConstructor(
            ExperimentEventsFile=File(self.base, *self.ExprEventsFile),
            ExperimentSetupFile=File(self.base, *self.ExprSetupFile),
            SLEAPFile=File(self.base, *self.SLEAPFile),
            VideoFile=File(self.base, *self.VideoFile),
        )
        logging.info("=========================================")
        logging.info(f"{self.tabs}Initializing Experiment...")
        logging.debug(f"{self.tabs}\tPath:{self.files.sleap.basepath}")
        logging.info("=========================================")

        self.sleap = SLEAPData(self.files.sleap.file, tabs=self.tabs + "\t")
        self.structure = ExprMetadata(self.files.setup.file, tabs=self.tabs + "\t")
        self.video = VideoMetadata(self.files.video.file, tabs=self.tabs + "\t")
        self.events = EventsData(self.files.events.file, tabs=self.tabs + "\t")
        self.numeric_columns = self.sleap.track_names

        self.all_data = self.sleap.tracks
        self.all_scores = self.sleap.scores
        self.numeric_columns = self.sleap.track_names

    def buildColumns(self, CustomColumns: list[CustomColumn] = None):
        """Builds the data for the experiment.

        Args:
            CustomColumns (list[CustomColumn]): A list of the CustomColumn objects to be added to the experiment.

        Raises:
            ValueError: If the columns cannot be appended to the SLEAP data.

        Returns:
            None

        Initializes attributes:
            sleap.tracks (pd.DataFrame): The SLEAP data.
            custom_columns (pd.DataFrame): The non-numeric columns.
        """
        logging.info(f"{self.tabs}Building columns for experiment: {self.name}")

        if CustomColumns is None:
            CustomColumns = []
        self.custom_columns = [0] * (len(self.sleap.tracks.index) + len(CustomColumns))
        col_names = [0] * (len(CustomColumns) + 2)
        for i, col in enumerate(CustomColumns):
            col_names[i] = col.ColumnTitle
            col.buildColumn(len(self.sleap.tracks.index))
            self.custom_columns[i] = col.Column
            self.custom_columns[i].reset_index(inplace=True)
        ms_per_frame = (self.video.fps**-1) * 1000
        for i in range(len(self.sleap.tracks.index)):
            self.custom_columns[(i + len(CustomColumns))] = pd.DataFrame(
                {"Timestamps": [i * ms_per_frame], "Frames": [i]},
                columns=["Timestamps", "Frames"],
            )
        col_names[len(CustomColumns)] = "Timestamps"
        col_names[len(CustomColumns) + 1] = "Frames"
        self.custom_columns[len(CustomColumns)] = pd.concat(
            self.custom_columns[len(CustomColumns) :], axis=0
        )
        self.custom_columns[len(CustomColumns)].reset_index(inplace=True)
        self.custom_columns: pd.DataFrame = pd.concat(
            self.custom_columns[: (len(CustomColumns) + 1)], axis=1
        )
        self.sleap.append(self.custom_columns.loc[:, col_names])
        self.all_data = self.sleap.tracks

    def append(self, item: pd.Series | pd.DataFrame):
        """Appends a column to the SLEAP data.

        Args:
            item (pd.Series | pd.DataFrame): A pandas series or dataframe to be appended to the SLEAP data.
        """
        if isinstance(item, pd.Series) or isinstance(item, pd.DataFrame):
            self.sleap.append(item)
            if isinstance(item, pd.Series):
                self.custom_columns.append(item.name)
            elif isinstance(item, pd.DataFrame):
                self.custom_columns.append(item.index)
        else:
            raise TypeError("The item to be appended must be a pandas series or dataframe.")

    def buildTrials(
        self,
        TrackedData: list[str],
        Reduced: list[bool],
        start_buffer: int = 10000,
        end_buffer: int = 13000,
    ):
        """Converts the data into trial by trial format.

        Args:
            TrackedData (list[str]): the list of columns from the DAQ data that signify the START of each trial.
            Reduced (list[bool]): a boolean list with the same length as the TrackedData list that signifies the columns from the tracked data with quick TTL pulses that occur during the trial.
                (e.g. the LED TTL pulse may signify the beginning of a trial, but during the trial the LED turns on and off, so the LED TTL column should be marked as True)
            start_buffer (int, optional): The time in miliseconds you want to capture before the trial starts. Defaults to 10000 (i.e. 10 seconds).
            end_buffer (int, optional): The time in miliseconds you want to capture after the trial starts. Defaults to 13000 (i.e. 13 seconds).

        Raises:
            ValueError: if the length of the TrackedData and Reduced lists are not equal.

        Initializes attributes:
                trials (pd.DataFrame): the dataframe with the data in trial by 	trial format, with a metaindex of trial number and frame number
                trialData (list[pd.DataFrame]): a list of the dataframes with the individual trial data.
        """
        logging.info(f"{self.tabs}Building trials for experiment: {self.name}")

        if len(Reduced) != len(TrackedData):
            raise ValueError(
                "The number of Reduced arguments must be equal to the number of TrackedData arguments. NOTE: If you do not want to reduce the data, pass in a list of False values."
            )

        start_indecies = [0] * len(TrackedData)
        end_indecies = [0] * len(TrackedData)
        timestamps = self.custom_columns.loc[:, "Timestamps"].to_numpy(dtype=np.float64)

        for data, reduce, i in zip(TrackedData, Reduced, range(len(TrackedData))):

            if reduce:
                times = pd.Series(self.events.cache.loc[:, data])
                times = times[times != 0]
                times = reduce_daq(times.to_list())
                times = np.array(times, dtype=np.float64)

            else:
                times = pd.Series(self.events.cache.loc[:, data])
                times = times[times != 0]
                times = times.to_numpy(dtype=np.float64, na_value=0)

            times = times[times != 0]

            start_indecies[i] = [0] * len(times)
            end_indecies[i] = [0] * len(times)

            for j, time in enumerate(times):
                start_indecies[i][j] = int(
                    np.absolute(timestamps - (time - start_buffer)).argmin()
                )
                end_indecies[i][j] = int(
                    (np.absolute(timestamps - (time + end_buffer)).argmin() + 1)
                )

        if type(start_indecies) is not list and type(start_indecies[0]) is not list:
            raise TypeError(
                "The start indecies are not in the correct format in the DAQ data."
            )

        start_indecies = flatten_list(start_indecies)
        end_indecies = flatten_list(end_indecies)

        if len(start_indecies) != len(end_indecies):
            raise ValueError(
                "The number of start indecies does not match the number of end indecies."
            )

        start_indecies = np.unique(np.array(start_indecies, dtype=np.int64))
        end_indecies = np.unique(np.array(end_indecies, dtype=np.int64))

        self.all_trials = into_trial_format(
            self.all_data,
            self.structure.cache.loc[:, "trialArray"],
            start_indecies,
            end_indecies,
        )
        self.trial_scores = into_trial_format(
            self.all_scores,
            self.structure.cache.loc[:, "trialArray"],
            start_indecies,
            end_indecies,
        )
        self.all_trials = [i for i in self.all_trials if type(i) is pd.DataFrame]
        self.trial_scores = [i for i in self.trial_scores if type(i) is pd.DataFrame]
        if len(self.all_trials) != len(self.trial_scores):
            warnings.warn(
                "The number of trial dataframes does not match the number of trial score dataframes.", RuntimeWarning
            )
        self.all_data = pd.concat(
            self.all_trials, axis=0, keys=[i for i in range(len(self.all_trials))]
        )
        self.all_data.index.names = ["Trial", "Trial_index"]
        self.all_scores = pd.concat(
            self.trial_scores, axis=0, keys=[i for i in range(len(self.trial_scores))]
        )

    def saveTrials(self, filename, *args):
        """Saves the trial data to a csv file.

        Args:
            filename (str): the file to save the HDF5 data to.
        """
        logging.debug(f"{self.tabs}Saving experiment:{self.name}")

        with pd.HDFStore(filename) as store:
            store.put(f"{self.name}/trials", self.all_data, format="table", data_columns=True)
            for i, trial in enumerate(self.all_trials):
                store.put(f"{self.name}/trialData/trial{i}", trial, format="table", data_columns=True)

    def meanCenter(self, alldata: bool = False):
        """Recursively mean centers the data for each trial for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the mean centered data for all trials and experiments concatenated together
        """
        if alldata:
            self.all_data = mean_center(self.all_data, self.numeric_columns)
        else:
            self.all_data = mean_center(
                            pd.concat(
                                [
                                    mean_center(
                                        trial, self.numeric_columns
                                    ) for trial in self.all_trials
                                ],
                                axis=0,
                                keys=range(len(self.all_trials)),
                            ),
                            self.numeric_columns
                        )
            self.all_data.index.names = ["Trial", "Trial_index"]

    def zScore(self, alldata: bool = False):
        """Z scores the mean centered data for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the z-scored data for all experiments concatenated together
        """
        if alldata:
            self.all_data = z_score(self.all_data, self.numeric_columns)
        else:
            self.all_data = z_score(
                            pd.concat(
                                [
                                    z_score(
                                        trial, self.numeric_columns
                                    ) for trial in self.all_trials
                                ],
                                axis=0,
                                keys=range(len(self.all_trials)),
                            ),
                            self.numeric_columns
                        )
            self.all_data.index.names = ["Trial", "Trial_index"]

    def normalize(self):
        """Runs the mean centering and z scoring functions

        Updates attributes:
            all_data (pd.DataFrame): the normaized data for all experiments concatenated together
        """
        logging.debug(f"{self.tabs}Normalizing experiment: {self.name}")

        self.all_data = z_score(
                            pd.concat(
                                [
                                    mean_center(
                                        trial, self.numeric_columns
                                    ) for trial in self.all_trials
                                ],
                                axis=0,
                                keys=range(len(self.all_trials)),
                            ),
                            self.numeric_columns
                        )
        self.all_data.index.names = ["Trial", "Trial_index"]
