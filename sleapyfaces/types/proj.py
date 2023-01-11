import os
from sleapyfaces.utils.structs import CustomColumn, File, FileConstructor
from sleapyfaces.types.expr import Experiment
from sleapyfaces.utils.normalize import mean_center, z_score, pca
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Project:
    """Base class for project

    Args:
        DAQFile (tuple[str, bool]): a tuple with the first argument being the naming convention for the DAQ files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_events.csv", True) or ("DAQOutput.csv", False))
        ExprMetaFile (str): a tuple with the first argument being the naming convention for the experimental structure files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_config.json", True) or ("BehMetadata.json", False))
        SLEAPFile (str): a tuple with the first argument being the naming convention for the SLEAP files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_sleap.h5", True) or ("SLEAP.h5", False))
        VideoFile (str): a tuple with the first argument being the naming convention for the video files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*.mp4", True) or ("video.avi", False))
        base (str): Base path of the project (e.g. "/specialk_cs/2p/raw/CSE009")
        iterator (dict[str, str]): Iterator for the project files, with keys as the label and values as the folder name (e.g. {"week 1": "20211105", "week 2": "20211112"})
        glob (bool): Whether to use glob to find the files (e.g. True or False)
            NOTE: if glob is True, make sure to include the file extension in the naming convention
        name (str): Name of the project (e.g. "CSE009")

    """

    def __init__(
        self,
        DAQFile: tuple[str, bool],
        BehFile: tuple[str, bool],
        SLEAPFile: tuple[str, bool],
        VideoFile: tuple[str, bool],
        base: str,
        name: str = None,
        expr_prefix: str = "week",
        tabs: str = "") -> None:

        self.base = base
        self.name = name
        print()
        print("=========================================")
        print(tabs, "Initializing Project...", self.name)
        print(tabs + "\t", "Path:", self.base)
        print("=========================================")
        print()
        self.prefix = expr_prefix
        self.DAQFile = DAQFile
        self.BehFile = BehFile
        self.SLEAPFile = SLEAPFile
        self.VideoFile = VideoFile
        self.iterator = {}
        if len(self.iterator.keys()) == 0:
            self.weeks = os.listdir(self.base)
            self.weeks = [
                week for week in self.weeks if os.path.isdir(os.path.join(self.base, week))
            ]
            self.weeks.sort()
            for i, week in enumerate(self.weeks):
                self.iterator[f"{self.prefix}_{i+1}"] = week
        self.exprs: dict[str, Experiment] = {}
        self.files: list[str] = [os.path.join(self.base, path) for path in list(self.iterator.values())]
        self.names: list[str] = list(self.iterator.keys())
        self.tabs = tabs
        for name, file in self.iterator.items():
            reuse = os.path.join(self.base, file)
            self.exprs[name]: Experiment = Experiment(name, FileConstructor(File(
                reuse,
                self.DAQFile[0],
                self.DAQFile[1],
            ), File(
                reuse,
                self.SLEAPFile[0],
                self.SLEAPFile[1],
            ), File(
                reuse,
                self.BehFile[0],
                self.BehFile[1],
            ), File(
                reuse,
                self.VideoFile[0],
                self.VideoFile[1],
            )), tabs=tabs + "\t")
        self.numeric_columns: list[str] = self.exprs[name].numeric_columns
        self.all_data: pd.DataFrame = pd.concat([expr.data for expr in self.exprs.values()], keys=self.names)
        self.all_scores: pd.DataFrame = pd.concat([expr.scores for expr in self.exprs.values()], keys=self.names)

    def buildColumns(self, columns: list, values: list):
        """Builds the custom columns for the project and builds the data for each experiment

        Args:
            columns (list[str]): the column titles
            values (list[any]): the data for each column

        Initializes attributes:
            custom_columns (list[CustomColumn]): list of custom columns
            all_data (pd.DataFrame): the data for all experiments concatenated together
            all_scores (pd.DataFrame): the scores for all experiments and trials concatenated together
        """
        print(self.tabs, "Building custom columns:")
        print(self.tabs, [(col, val) for col, val in zip(columns, values)])
        self.custom_columns = [CustomColumn(col, val) for col, val in zip(columns, values)]
        for expr in self.exprs.values():
            expr.buildData(self.custom_columns)
        self.all_data = pd.concat([expr.data for expr in self.exprs.values()], keys=self.names)
        self.all_scores = pd.concat([expr.scores for expr in self.exprs.values()], keys=self.names)

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
            all_data (pd.DataFrame): the data for all experiments and trials concatenated together
            all_scores (pd.DataFrame): the scores for all experiments and trials concatenated together
        """
        print(self.tabs, "Building trials for project:", self.name)
        for expr in self.exprs.values():
            expr.buildTrials(TrackedData, Reduced, start_buffer, end_buffer)
        self.all_data = pd.concat([expr.data for expr in self.exprs.values()], keys=self.names)
        self.all_scores = pd.concat([expr.scores for expr in self.exprs.values()], keys=self.names)

    def meanCenter(self, alldata: bool = False):
        """Recursively mean centers the data for each trial for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the mean centered data for all trials and experiments concatenated together
        """
        if alldata:
            self.all_data = mean_center(self.all_data, self.numeric_columns)
        else:
            for expr in self.exprs.values():
                expr.meanCenter()
            self.all_data = mean_center(
                pd.concat(
                    [
                        expr.data for expr in self.exprs.values()
                    ],
                    keys=self.names
                ),
                self.numeric_columns)

    def zScore(self, alldata: bool = False):
        """Z scores the mean centered data for each experiment

        Updates attributes:
            all_data (pd.DataFrame): the z-scored data for all experiments concatenated together
        """
        if alldata:
            self.all_data = z_score(self.all_data, self.numeric_columns)
        else:
            for expr in self.exprs.values():
                expr.zScore()
            self.all_data = z_score(
                pd.concat(
                    [
                        expr.data for expr in self.exprs.values()
                    ],
                    keys=self.names
                ),
                self.numeric_columns)

    def normalize(self):
        """Runs the mean centering and z scoring functions

        Updates attributes:
            all_data (pd.DataFrame): the normaized data for all experiments concatenated together
        """
        print(self.tabs, "Normalizing project", self.name)
        for expr in self.exprs.values():
                expr.normalize()
        self.all_data = z_score(
                pd.concat(
                    [
                        expr.data for expr in self.exprs.values()
                    ],
                    keys=self.names
                ),
                self.numeric_columns)

    def runPCA(self):
        """Reduces `all_data` to 2 and 3 dimensions using PCA

        Initializes attributes:
            pcas (dict[str, pd.DataFrame]): a dictionary containing the 2 and 3 dimensional PCA data for each experiment (the keys are 'pca2d', 'pca3d')
        """
        self.pcas = pca(self.data, self.numeric_columns)

    def visualize(self, dimensions: int, filename: str = None, *args, **kwargs) -> go.Figure:
        """Plots the data from the PCA

        Args:
            filename (str, optional): The filename to save the plot to. Defaults to None.
        """
        if dimensions == 2:
            fig = px.scatter(self.pcas[f"pca{dimensions}d"], x="principal component 1", y="principal component 2", color="Mouse")
        elif dimensions == 3:
            fig = px.scatter_3d(self.pcas[f"pca{dimensions}d"], x="principal component 1", y="principal component 2", z="principal component 3", color="Mouse")
        else:
            raise ValueError("dimensions must be 2 or 3")
        if filename is not None:
            fig.write_image(filename, *args, **kwargs)
        return fig

    def save(self, filename: str | pd.HDFStore, title: str = None):
        """Saves the project data to a HDF5 file

        Args:
            filename (str): The filename to save the data to
        """
        with pd.HDFStore(filename) as store:
            if self.name is not None:
                store.put(f"projects/{self.name}/data", self.data)
                store.put(f"projects/{self.name}/scores", self.scores)
            elif title is not None:
                store.put(f"projects/{title}/data", self.data)
                store.put(f"projects/{title}/scores", self.scores)
            else:
                store.put("all_data", self.data)
                store.put("all_scores", self.scores)

    def saveAll(self, filename: str, title: str = None):
        """Saves all the data to a HDF5 file

        Args:
            filename (str): The filename to save the data to
        """
        print(self.tabs, "Saving project:", self.name)
        for expr in self.exprs.values():
            expr.saveTrials(filename)
        with pd.HDFStore(filename) as store:
            if self.name is not None:
                store.put(f"projects/{self.name}/data", self.data)
                store.put(f"projects/{self.name}/scores", self.scores)
            elif title is not None:
                store.put(f"projects/{title}/data", self.data)
                store.put(f"projects/{title}/scores", self.scores)
            else:
                store.put("projects/all_data", self.data)
                store.put("projects/all_scores", self.scores)

    @property
    def data(self) -> pd.DataFrame:
        if len(self.all_data.index.levshape) == 3:
            self.all_data.index.set_names(["Experiment", "Trial", "Trial_frame"], inplace=True)
        elif len(self.all_data.index.levshape) == 1:
            self.all_data.index.set_names(["Experiment"], inplace=True)

        return self.all_data

    @property
    def scores(self) -> pd.DataFrame:
        return self.all_scores

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
        cols = self.data.reset_index().columns.to_list()
        cols = [i for i in cols if i not in self.quant_cols]
        return cols

    @property
    def cols(self) -> tuple[list[str], list[str]]:
        """Returns the target and non target columns of the data.

        Returns:
            tuple[list[str], list[str]]: a tuple of column lists, the first being the target columns and the second being the non-target columns.
        """
        return (self.quant_cols, self.qual_cols)
