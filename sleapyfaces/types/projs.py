import os
from sleapyfaces.types.proj import Project
from sleapyfaces.utils.normalize import mean_center, z_score, pca
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class Projects:
    """Base class for multiple projects

    Args:
        DAQFile (tuple[str, bool]): a tuple with the first argument being the naming convention for the DAQ files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_events.csv", True) or ("DAQOutput.csv", False))
        ExprMetaFile (tuple[str, bool]): a tuple with the first argument being the naming convention for the experimental structure files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_config.json", True) or ("BehMetadata.json", False))
        SLEAPFile (tuple[str, bool]): a tuple with the first argument being the naming convention for the SLEAP files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*_sleap.h5", True) or ("SLEAP.h5", False))
        VideoFile (tuple[str, bool]): a tuple with the first argument being the naming convention for the video files and the second argument whether or not to find the file based on a globular expression passed in the first argument (e.g. ("*.mp4", True) or ("video.avi", False))
        base (str): the base folder for the project (e.g. "path/to/project")
        projects_base (dict[str, str]): an iterative dictionary with the keys as the project name and the values as the folder name (e.g. {"Resilient 1": "CSE009", "Control 1": "CSC008"})
        iterator (dict[str, str]): Iterator for the project files, with keys as the label and values as the folder name (e.g. {"week 1": "20211105", "week 2": "20211112"})
        glob (bool): Whether to use glob to find the files (e.g. True or False)
            NOTE: if glob is True, make sure to include the file extension in the naming convention
        name (str): Name of the project (e.g. "CSE009")
    """
    def __init__(self,
        DAQFile: tuple[str, bool],
        BehFile: tuple[str, bool],
        SLEAPFile: tuple[str, bool],
        VideoFile: tuple[str, bool],
        base: str,
        projects_base: dict[str, str],
        expr_prefix: str = "week",
        tabs: str = "") -> None:

        """A class to handle multiple projects"""

        self.base_project = projects_base
        self.base_path = base
        self.names = list(projects_base.keys())
        self.files = [os.path.join(self.base_path, path) for path in list(self.base_project.values())]
        self.projects: dict[str, Project] = {}
        print()
        print("=========================================")
        print()
        print(tabs, "Initializing Projects...")
        print()
        print("------------------------------------------")
        print(tabs+"\t", "Base path:", self.base_path)
        print(tabs+"\t", "Projects:", self.base_project)
        print()
        print(tabs+"\t", "Transforming to:")
        print("------------------------------------------")
        print(tabs+"\t\t", "Project keys (names):", self.names)
        print(tabs+"\t\t", "Project files (paths):", self.files)
        print("------------------------------------------")
        print()
        print("=========================================")
        print()
        for name, file in self.base_project.items():
            self.projects[name] = Project(
                DAQFile=DAQFile,
                BehFile=BehFile,
                SLEAPFile=SLEAPFile,
                VideoFile=VideoFile,
                base=os.path.join(self.base_path, file),
                name=name,
                expr_prefix=expr_prefix,
                tabs = tabs + "\t"
            )
        self.numeric_columns = self.projects[self.names[0]].numeric_columns
        self.all_data = pd.concat([project.all_data for project in self.projects.values()], keys=self.names)
        self.all_scores = pd.concat([project.all_scores for project in self.projects.values()], keys=self.names)

    def buildColumns(self, columns: list, values: list):
        """Builds the custom columns for each project and builds the data for each experiment

        Args:
            columns (list[str]): the column titles
            values (list[any]): the data for each column

        Updates attributes:
            projects.custom_columns (list[CustomColumn]): list of custom columns
            projects.all_data (pd.DataFrame): the data for all experiments concatenated together
        """
        for project in self.projects.values():
            project.buildColumns(columns, values)
        self.custom_columns = self.projects[self.names[0]].custom_columns
        self.all_data = pd.concat([project.all_data for project in self.projects.values()], keys=self.names)
        self.all_scores = pd.concat([project.all_scores for project in self.projects.values()], keys=self.names)


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
        for project in self.projects.values():
            project.buildTrials(TrackedData, Reduced, start_buffer, end_buffer)
        self.all_data = pd.concat([project.all_data for project in self.projects.values()], keys=self.names)
        self.all_scores = pd.concat([project.all_scores for project in self.projects.values()], keys=self.names)

    def meanCenter(self, alldata: bool = False):
        if alldata:
            self.all_data = mean_center(self.all_data, self.numeric_columns)
        else:
            for project in self.projects.values():
                project.meanCenter()
            self.all_data = mean_center(
                pd.concat(
                    [
                        project.all_data
                        for project in self.projects.values()
                    ], keys=self.names
                ), self.numeric_columns)

    def zScore(self, alldata: bool = False):
        if alldata:
            self.all_data = z_score(self.all_data, self.numeric_columns)
        else:
            for project in self.projects.values():
                project.zScore()
            self.all_data = z_score(
                pd.concat(
                    [
                        project.all_data
                        for project in self.projects.values()
                    ], keys=self.names
                ), self.numeric_columns)

    def normalize(self):
        for project in self.projects.values():
            project.normalize()
        self.all_data = z_score(
            pd.concat(
                    [
                        project.all_data
                        for project in self.projects.values()
                    ], keys=self.names
                ),
            self.numeric_columns
        )

    def runPCA(self):
        self.pcas = pca(self.all_data, self.numeric_columns)

    def visualize(self, dimensions: int, filename=None, *args, **kwargs) -> go.Figure:
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

    @property
    def data(self) -> pd.DataFrame:
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
        return self.quant_cols, self.qual_cols
