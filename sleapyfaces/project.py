import os
from sleapyfaces.structs import CustomColumn, File, FileConstructor
from sleapyfaces.experiment import Experiment
from sleapyfaces.normalize import mean_center, z_score, pca
from dataclasses import dataclass
import pandas as pd


class Project:
    """Base class for project

    Args:
        base (str): Base path of the project (e.g. "/specialk_cs/2p/raw/CSE009")
        iterator (dict[str, str]): Iterator for the project files, with keys as the label and values as the folder name (e.g. {"week 1": "20211105", "week 2": "20211112"})
        DAQFile (str): The naming convention for the DAQ files (e.g. "*_events.csv" or "DAQOutput.csv")
        ExprMetaFile (str): The naming convention for the experimental structure files (e.g. "*_config.json" or "BehMetadata.json")
        SLEAPFile (str): The naming convention for the SLEAP files (e.g. "*_sleap.h5" or "SLEAP.h5")
        VideoFile (str): The naming convention for the video files (e.g. "*.mp4" or "video.avi")
        glob (bool): Whether to use glob to find the files (e.g. True or False)
            NOTE: if glob is True, make sure to include the file extension in the naming convention

    """

    def __init__(
        self,
        DAQFile: str,
        BehFile: str,
        SLEAPFile: str,
        VideoFile: str,
        base: str,
        iterator: dict[str, str] = {},
        get_glob: bool = False,
    ):
        self.base = base
        self.DAQFile = DAQFile
        self.BehFile = BehFile
        self.SLEAPFile = SLEAPFile
        self.VideoFile = VideoFile
        self.get_glob = get_glob
        if len(iterator.keys()) == 0:
            weeks = os.listdir(self.base)
            weeks = [
                week for week in weeks if os.path.isdir(os.path.join(self.base, week))
            ]
            weeks.sort()
            for i, week in enumerate(weeks):
                iterator[f"week {i+1}"] = week
        self.iterator = iterator
        self.exprs = [0] * len(self.iterator.keys())
        self.files = [0] * len(self.iterator.keys())
        for i, name in enumerate(list(self.iterator.keys())):
            daq_file = File(
                os.path.join(self.base, self.iterator[name]),
                self.DAQFile,
                self.get_glob,
            )
            sleap_file = File(
                os.path.join(self.base, self.iterator[name]),
                self.SLEAPFile,
                self.get_glob,
            )
            beh_file = File(
                os.path.join(self.base, self.iterator[name]),
                self.BehFile,
                self.get_glob,
            )
            video_file = File(
                os.path.join(self.base, self.iterator[name]),
                self.VideoFile,
                self.get_glob,
            )
            self.files[i] = FileConstructor(daq_file, sleap_file, beh_file, video_file)
            self.exprs[i] = Experiment(name, self.files[i])

    def buildColumns(self, columns: list, values: list):
        self.custom_columns = [0] * len(columns)
        for i in range(len(self.custom_columns)):
            self.custom_columns[i] = CustomColumn(columns[i], values[i])
        exprs_list = [0] * len(self.exprs)
        names_list = [0] * len(self.exprs)
        for i in range(len(self.exprs)):
            self.exprs[i].buildData(self.custom_columns)
            exprs_list[i] = self.exprs[i].sleap.tracks
            names_list[i] = self.exprs[i].name
        self.all_data = pd.concat(exprs_list, keys=names_list)

    def buildTrials(
        self,
        TrackedData: list[str],
        Reduced: list[bool],
        start_buffer: int = 10000,
        end_buffer: int = 13000,
    ):
        for i in range(len(self.exprs)):
            self.exprs[i].buildTrials(TrackedData, Reduced, start_buffer, end_buffer)

    def meanCenter(self):
        mean_all = [0] * len(self.exprs)
        for i in range(len(self.exprs)):
            mean_all[i] = [0] * len(self.exprs[i].trialData)
            for j in range(len(self.exprs[i].trialData)):
                mean_all[i][j] = mean_center(
                    self.exprs[i].trialData[i], self.exprs[i].sleap.track_names
                )
            mean_all[i] = pd.concat(
                mean_all[i],
                axis=0,
                keys=range(len(mean_all[i])),
            )
            mean_all[i] = mean_center(mean_all[i], self.exprs[i].sleap.track_names)
        self.all_data = pd.concat(mean_all, keys=list(self.iterator.keys()))

    def zScore(self):
        self.all_data = z_score(self.all_data, self.exprs[0].sleap.track_names)

    def analyze(self):
        self.meanCenter()
        self.zScore()

    def visualize(self):
        self.pcas = pca(self.all_data, self.exprs[0].sleap.track_names)
