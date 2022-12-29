from dataclasses import dataclass
from sleapyfaces.io import SLEAPanalysis, BehMetadata, VideoMetadata, DAQData
from sleapyfaces.structs import FileConstructor, CustomColumn

from sleapyfaces.utils import into_trial_format, reduce_daq, flatten_list

import pandas as pd
import numpy as np


class Experiment:
    def __init__(self, name: str, files: FileConstructor):
        self.name = name
        self.files = files
        self.sleap = SLEAPanalysis(self.files.sleap.file)
        self.beh = BehMetadata(self.files.beh.file)
        self.video = VideoMetadata(self.files.video.file)
        self.daq = DAQData(self.files.daq.file)
        self.numeric_columns = self.sleap.track_names

    def buildData(self, CustomColumns: list[CustomColumn]):
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
        self.custom_columns = pd.concat(
            self.custom_columns[: (len(CustomColumns) + 1)], axis=1
        )
        self.sleap.append(self.custom_columns.loc[:, col_names])

    def append(self, item: pd.Series | pd.DataFrame):
        self.sleap.append(item)

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
            DAQ (DAQData): the DAQ data object.
            Reduced (list[bool]): a boolean list with the same length as the TrackedData list that signifies the columns from the tracked data with quick TTL pulses that occour during the trial.
                (e.g. the LED TTL pulse may signify the beginning of a trial, but during the trial the LED turns on and off, so the LED TTL column should be marked as True)
            start_buffer (int, optional): The time in miliseconds you want to capture before the trial starts. Defaults to 10000 (i.e. 10 seconds).
            end_buffer (int, optional): The time in miliseconds you want to capture after the trial starts. Defaults to 13000 (i.e. 13 seconds).

        Raises:
            ValueError: if the length of the TrackedData and Reduced lists are not equal.

        Exposes the instance attribute:
                trials (pd.DataFrame): the dataframe with the data in trial by 	trial format, with a metaindex of trial number and frame number
        """

        if len(Reduced) != len(TrackedData):
            raise ValueError(
                "The number of Reduced arguments must be equal to the number of TrackedData arguments. NOTE: If you do not want to reduce the data, pass in a list of False values."
            )

        start_indecies = [0] * len(TrackedData)
        end_indecies = [0] * len(TrackedData)
        timestamps = self.custom_columns.loc[:, "Timestamps"].to_numpy(dtype=np.float64)

        for data, reduce, i in zip(TrackedData, Reduced, range(len(TrackedData))):

            if reduce:
                times = pd.Series(self.daq.cache.loc[:, data])
                times = times[times != 0]
                times = reduce_daq(times.to_list())
                times = np.array(times, dtype=np.float64)

            else:
                times = pd.Series(self.daq.cache.loc[:, data])
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

        self.trialData = into_trial_format(
            self.sleap.tracks,
            self.beh.cache.loc[:, "trialArray"],
            start_indecies,
            end_indecies,
        )
        self.trialData = [i for i in self.trialData if type(i) is pd.DataFrame]
        self.trials = pd.concat(
            self.trialData, axis=0, keys=[i for i in range(len(self.trialData))]
        )
