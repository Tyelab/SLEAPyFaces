import os
from sleapyfaces.structs import CustomColumn, File, FileConstructor
from sleapyfaces.experiment import Experiment
from sleapyfaces.normalize import mean_center, z_score, pca
import pandas as pd
import plotly.express as px


class Project:
    """Base class for project

    Args:
        base (str): Base path of the project (e.g. "/specialk_cs/2p/raw/CSE009")
        iterator (dict[str, str]): Iterator for the project files, with keys as the label and values as the folder name (e.g. {"week 1": "20211105", "week 2": "20211112"})
        DAQFile (tuple[str, bool]): a tuple with the first argument being the naming convention for the DAQ files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_events.csv", True) or ("DAQOutput.csv", False))
        ExprMetaFile (str): a tuple with the first argument being the naming convention for the experimental structure files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_config.json", True) or ("BehMetadata.json", False))
        SLEAPFile (str): a tuple with the first argument being the naming convention for the SLEAP files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_sleap.h5", True) or ("SLEAP.h5", False))
        VideoFile (str): a tuple with the first argument being the naming convention for the video files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*.mp4", True) or ("video.avi", False))
        glob (bool): Whether to use glob to find the files (e.g. True or False)
            NOTE: if glob is True, make sure to include the file extension in the naming convention

    """

    def __init__(
        self,
        DAQFile: tuple[str, bool],
        BehFile: tuple[str, bool],
        SLEAPFile: tuple[str, bool],
        VideoFile: tuple[str, bool],
        base: str,
        iterator: dict[str, str] = {},
        name: str = "",
    ):
        self.base = base
        self.name = name
        self.DAQFile = DAQFile
        self.BehFile = BehFile
        self.SLEAPFile = SLEAPFile
        self.VideoFile = VideoFile
        if len(iterator.keys()) == 0:
            weeks = os.listdir(self.base)
            weeks = [
                week for week in weeks if os.path.isdir(os.path.join(self.base, week))
            ]
            weeks.sort()
            for i, week in enumerate(weeks):
                iterator[f"week {i+1}"] = week
        self.files = iterator
        self.exprs = {}
        self.names = list(self.files.keys())
        for name in self.names:
            daq_file = File(
                os.path.join(self.base, self.files[name]),
                self.DAQFile[0],
                self.DAQFile[1],
            )
            sleap_file = File(
                os.path.join(self.base, self.files[name]),
                self.SLEAPFile[0],
                self.SLEAPFile[1],
            )
            beh_file = File(
                os.path.join(self.base, self.files[name]),
                self.BehFile[0],
                self.BehFile[1],
            )
            video_file = File(
                os.path.join(self.base, self.files[name]),
                self.VideoFile[0],
                self.VideoFile[1],
            )
            self.files[name] = FileConstructor(daq_file, sleap_file, beh_file, video_file)
            self.exprs[name] = Experiment(name, self.files[name])
        self.numeric_columns = self.exprs[self.names[0]].numeric_columns

    def buildColumns(self, columns: list, values: list):
        """Builds the custom columns for the project and builds the data for each experiment

        Args:
            columns (list[str]): the column titles
            values (list[any]): the data for each column

        Initializes attributes:
            custom_columns (list[CustomColumn]): list of custom columns
            all_data (pd.DataFrame): the data for all experiments concatenated together
        """
        self.custom_columns = [0] * len(columns)
        for i in range(len(self.custom_columns)):
            self.custom_columns[i] = CustomColumn(columns[i], values[i])
        exprs_list = [0] * len(self.names)
        names_list = [0] * len(self.names)
        for i, name in enumerate(self.names):
            self.exprs[name].buildData(self.custom_columns)
            exprs_list[i] = self.exprs[name].sleap.tracks
            names_list[i] = self.exprs[name].name
        self.all_data = pd.concat(exprs_list, keys=names_list)

    def buildTrials(
        self,
        TrackedData: list[str],
        Reduced: list[bool],
        start_buffer: int = 10000,
        end_buffer: int = 13000,
    ):
        """Parses the data from each experiment into its individual trials

        Args:
            TrackedData (list[str]): The title of the columns from the DAQ data to be tracked
            Reduced (list[bool]): The corresponding boolean for whether the DAQ data is to be reduced (`True`) or not (`False`)
            start_buffer (int, optional): The time in milliseconds before the trial start to capture. Defaults to 10000.
            end_buffer (int, optional): The time in milliseconds after the trial start to capture. Defaults to 13000.

        Initializes attributes:
            exprs[i].trials (pd.DataFrame): the data frame containing the concatenated trial data for each experiment
            exprs[i].trialData (list[pd.DataFrame]): the list of data frames containing the trial data for each trial for each experiment
        """
        for name in self.names:
            self.exprs[name].buildTrials(TrackedData, Reduced, start_buffer, end_buffer)

    def meanCenter(self):
        """Recursively mean centers the data for each trial for each experiment

        Initializes attributes:
            all_data (pd.DataFrame): the mean centered data for all trials and experiments concatenated together
        """
        mean_all = [0] * len(self.names)
        for i, name in enumerate(self.names):
            mean_all[i] = [0] * len(self.exprs[name].trialData)
            for j in range(len(self.exprs[name].trialData)):
                mean_all[i][j] = mean_center(
                    self.exprs[name].trialData[i], self.numeric_columns
                )
            mean_all[i] = pd.concat(
                mean_all[i],
                axis=0,
                keys=range(len(mean_all[i])),
            )
            mean_all[i] = mean_center(mean_all[i], self.numeric_columns)
        self.all_data = pd.concat(mean_all, keys=self.names)
        self.all_data = mean_center(self.all_data, self.numeric_columns)

    def zScore(self):
        """Z scores the mean centered data for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the z-scored data for all experiments concatenated together
        """
        self.all_data = z_score(self.all_data, self.numeric_columns)


    def analyze(self):
        """Runs the mean centering and z scoring functions
        """
        analyze_all = [0] * len(self.names)
        for i, name in enumerate(self.names):
            analyze_all[i] = [0] * len(self.exprs[name].trialData)
            for j in range(len(self.exprs[name].trialData)):
                analyze_all[i][j] = mean_center(
                    self.exprs[name].trialData[i], self.numeric_columns
                )
            analyze_all[i] = pd.concat(
                analyze_all[i],
                axis=0,
                keys=range(len(analyze_all[i])),
            )
            analyze_all[i] = z_score(analyze_all[i], self.numeric_columns)
        self.all_data = pd.concat(analyze_all, keys=self.names)
        self.all_data = z_score(self.all_data, self.numeric_columns)

    def runPCA(self):
        """Reduces `all_data` to 2 and 3 dimensions using PCA

        Initializes attributes:
            pcas (dict[str, pd.DataFrame]): a dictionary containing the 2 and 3 dimensional PCA data for each experiment (the keys are 'pca2d', 'pca3d')
        """
        self.pcas = pca(self.all_data, self.numeric_columns)

    def visualize(self, dimentions: int, filename=None, *args, **kwargs):
        """Plots the data from the PCA

        Args:
            filename (str, optional): The filename to save the plot to. Defaults to None.
        """
        if dimentions == 2:
            fig = px.scatter(self.pcas[f"pca{dimentions}d"], x="principal component 1", y="principal component 2", color="Mouse")
        elif dimentions == 3:
            fig = px.scatter_3d(self.pcas[f"pca{dimentions}d"], x="principal component 1", y="principal component 2", z="principal component 3", color="Mouse")
        else:
            raise ValueError("dimentions must be 2 or 3")
        if filename is not None:
            fig.write_image(filename, *args, **kwargs)
        fig.show()

    def save(self, filename: str, title: str = None):
        """Saves the project data to a HDF5 file

        Args:
            filename (str): The filename to save the data to
        """
        with pd.HDFStore(filename) as store:
            if self.name is not None:
                store.put(f"projects/{self.name}", self.all_data)
            elif title is not None:
                store.put(f"projects/{title}", self.all_data)
            else:
                store.put("all_data", self.all_data)

    def saveAll(self, filename: str, title: str = None):
        """Saves all the data to a HDF5 file

        Args:
            filename (str): The filename to save the data to
        """
        for name in self.names:
            self.exprs[name].saveTrials(filename)
        with pd.HDFStore(filename) as store:
            if self.name is not None:
                store.put(f"projects/{self.name}", self.all_data)
            elif title is not None:
                store.put(f"projects/{title}", self.all_data)
            else:
                store.put("projects/all_data", self.all_data)


class Projects:
    def __init__(self,
        DAQFile: tuple[str, bool],
        BehFile: tuple[str, bool],
        SLEAPFile: tuple[str, bool],
        VideoFile: tuple[str, bool],
        projects_base: dict[str, str],
        iterator: dict[str, str] = {},
        get_glob: bool = False) -> None:
        """A class to handle multiple projects"""
        self.names = list(projects_base.keys())
        proj = {}
        for key in list(projects_base.keys()):
            proj[key] = Project(
                DAQFile,
                BehFile,
                SLEAPFile,
                VideoFile,
                projects_base[key],
                iterator,
                get_glob,
                key
            )
        self.projects = proj
        self.numeric_columns = self.projects[self.names[0]].numeric_columns

    def buildColumns(self, columns: list, values: list):
        """Builds the custom columns for each project and builds the data for each experiment

        Args:
            columns (list[str]): the column titles
            values (list[any]): the data for each column

        Initializes attributes:
            projects.custom_columns (list[CustomColumn]): list of custom columns
            projects.all_data (pd.DataFrame): the data for all experiments concatenated together
        """
        for name in self.names:
            self.projects[name].buildColumns(columns, values)
        self.custom_columns = self.projects[self.names[0]].custom_columns
        self.all_data = len(self.names)
        for i, name in enumerate(self.names):
            self.all_data[i] = self.projects[name].all_data
        self.all_data = pd.concat(self.all_data, keys=self.names)


    def buildTrials(
        self,
        TrackedData: list[str],
        Reduced: list[bool],
        start_buffer: int = 10000,
        end_buffer: int = 13000,
    ):
        """Parses the data from each experiment into its individual trials

        Args:
            TrackedData (list[str]): The title of the columns from the DAQ data to be tracked
            Reduced (list[bool]): The corresponding boolean for whether the DAQ data is to be reduced (`True`) or not (`False`)
            start_buffer (int, optional): The time in milliseconds before the trial start to capture. Defaults to 10000.
            end_buffer (int, optional): The time in milliseconds after the trial start to capture. Defaults to 13000.

        Initializes attributes:
            projects[name].exprs[exprName].trials (pd.DataFrame): the data frame containing the concatenated trial data for each experiment
            projects[name].exprs[exprName].trialData (list[pd.DataFrame]): the list of data frames containing the trial data for each trial for each experiment
        """
        for name in self.names:
            self.projects[name].buildTrials(TrackedData, Reduced, start_buffer, end_buffer)

    def meanCenter(self):
        mean_all = [0] * len(self.names)
        for i, name in enumerate(self.names):
            self.projects[name].meanCenter()
            mean_all[i] = self.projects[name].all_data
        self.all_data = pd.concat(mean_all, keys=self.names)
        self.all_data = mean_center(self.all_data, self.numeric_columns)

    def zScore(self):
        z_score_all = [0] * len(self.names)
        for i, name in enumerate(self.names):
            self.projects[name].zScore()
            z_score_all[i] = self.projects[name].all_data
        self.all_data = pd.concat(z_score_all, keys=self.names)
        self.all_data = z_score(self.all_data, self.numeric_columns)

    def analyze(self):
        analyze_all = [0] * len(self.names)
        for i, name in enumerate(self.names):
            self.projects[name].analyze()
            analyze_all[i] = self.projects[name].all_data
        self.all_data = pd.concat(analyze_all, keys=self.names)
        self.all_data = z_score(self.all_data, self.numeric_columns)

    def runPCA(self):
        self.pcas = pca(self.all_data, self.numeric_columns)

    def visualize(self, dimentions: int, filename=None, *args, **kwargs):
        """Plots the data from the PCA

        Args:
            filename (str, optional): The filename to save the plot to. Defaults to None.
        """
        if dimentions == 2:
            fig = px.scatter(self.pcas[f"pca{dimentions}d"], x="principal component 1", y="principal component 2", color="Mouse")
        elif dimentions == 3:
            fig = px.scatter_3d(self.pcas[f"pca{dimentions}d"], x="principal component 1", y="principal component 2", z="principal component 3", color="Mouse")
        else:
            raise ValueError("dimentions must be 2 or 3")
        if filename is not None:
            fig.write_image(filename, *args, **kwargs)
        fig.show()

    def save(self, filename: str, title: str = None):
        """Saves projects data to a HDF5 file

        Args:
            filename (str): The filename to save the data to
        """
        with pd.HDFStore(filename) as store:
            if title is not None:
                store.put(f"{title}/all_data", self.all_data)
            else:
                store.put("all_data", self.all_data)

    def saveAll(self, filename: str, title: str = None):
        """Saves all the data to a HDF5 file

        Args:
            filename (str): The filename to save the data to
        """
        for name in self.names:
            self.projects[name].saveAll(filename)
        with pd.HDFStore(filename) as store:
            if title is not None:
                store.put(f"{title}/all_data", self.all_data)
            else:
                store.put("all_data", self.all_data)
