import os

import pandas as pd

from sleapyfaces.base.proj import Project
from sleapyfaces.base.type import BaseType

import logging


class Projects(BaseType):
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
        ExperimentEventsFile: tuple[str, bool] | str,
        ExperimentSetupFile: tuple[str, bool] | str,
        SLEAPFile: tuple[str, bool],
        VideoFile: tuple[str, bool],
        base: str,
        name: str,
        file_structure: dict[str, str],
        tabs: str = "",
        sublevel: str = "Project",
        *args,
        **kwargs) -> None:
        """A class to handle multiple projects"""

        super().__init__(
            ExperimentEventsFile=ExperimentEventsFile,
            ExperimentSetupFile=ExperimentSetupFile,
            SLEAPFile=SLEAPFile,
            VideoFile=VideoFile,
            file_structure=file_structure,
            base=base,
            name=name,
            tabs=tabs,
            sublevel=sublevel,
            *args,
            **kwargs
        )

    def _init_data(self):
        """Initializes the data for each project"""
        logging.info("=========================================")
        logging.info(self.tabs, "Initializing Projects...")
        logging.info("=========================================")
        logging.debug("=========================================")
        logging.debug(self.tabs, "Initializing Data...")
        logging.debug("------------------------------------------")
        logging.debug(self.tabs+"\t", "Base path:", self.base)
        logging.debug(self.tabs+"\t", "Projects:", self.fileStruct)
        logging.debug(self.tabs+"\t", "Transforming to:")
        logging.debug("------------------------------------------")
        logging.debug(self.tabs+"\t\t", "Project keys (names):", self.names)
        logging.debug(self.tabs+"\t\t", "Project files (paths):", self.paths)
        logging.debug("------------------------------------------")
        logging.debug("=========================================")
        for name, file in self.fileStruct.items():
            self.data[name] = Project(
                name=name,
                base=os.path.join(self.base, file),
                ExperimentEventsFile=self.ExprEventsFile,
                ExperimentSetupFile=self.ExprSetupFile,
                SLEAPFile=self.SLEAPFile,
                VideoFile=self.VideoFile,
                tabs=self.tabs + "\t",
                passed_config=self.config,
                sublevel="Experiment",
                prefix="week"
            )
        self.numeric_columns = self.data[self.names[0]].numeric_columns
        self.all_data = pd.concat([data.all_data for data in self.data.values()], keys=self.names)
        self.all_scores = pd.concat([data.all_scores for data in self.data.values()], keys=self.names)

    def _rename_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index.names = ["Project", "Experiment", "Trial", "Trial_index"]
        return df

    # def buildColumns(self, columns: list = None, values: list = None):
    #     """Builds the custom columns for each project and builds the data for each experiment

    #     Args:
    #         columns (list[str]): the column titles
    #         values (list[any]): the data for each column

    #     Updates attributes:
    #         projects.custom_columns (list[CustomColumn]): list of custom columns
    #         projects.all_data (pd.DataFrame): the data for all experiments concatenated together
    #     """
    #     self.custom_columns = []
    #     print(self.tabs, "Building columns...")
    #     if columns is None:
    #         if not hasattr(self.projects[self.names[0]], "custom_columns"):
    #             for project in self.projects.values():
    #                 project.buildColumns()
    #     elif len(columns) != len(values):
    #             raise ValueError("The number of columns and values must be equal")
    #     else:
    #         for project in self.projects.values():
    #             project.buildColumns(columns, values)
    #     self.custom_columns = self.projects[self.names[0]].custom_columns
    #     self.all_data = pd.concat([project.all_data for project in self.projects.values()], keys=self.names)
    #     self.all_scores = pd.concat([project.all_scores for project in self.projects.values()], keys=self.names)


    # def buildTrials(
    #     self,
    #     TrackedData: list[str],
    #     Reduced: list[bool],
    #     start_buffer: int = 10000,
    #     end_buffer: int = 13000,
    # ):
    #     """Parses the data from each experiment into its individual trials

    #     Args:
    #         TrackedData (list[str]): The title of the columns from the DAQ data to be tracked
    #         Reduced (list[bool]): The corresponding boolean for whether the DAQ data is to be reduced (`True`) or not (`False`)
    #         start_buffer (int, optional): The time in milliseconds before the trial start to capture. Defaults to 10000.
    #         end_buffer (int, optional): The time in milliseconds after the trial start to capture. Defaults to 13000.

    #     Initializes attributes:
    #         projects[name].exprs[exprName].trials (pd.DataFrame): the data frame containing the concatenated trial data for each experiment
    #         projects[name].exprs[exprName].trialData (list[pd.DataFrame]): the list of data frames containing the trial data for each trial for each experiment
    #     """
    #     print(self.tabs, "Building trials...")
    #     for project in self.projects.values():
    #         project.buildTrials(TrackedData, Reduced, start_buffer, end_buffer)
    #     self.allTrials = [trial for project in self.projects.values() for trial in project.allTrials]
    #     self.all_data = pd.concat([project.all_data for project in self.projects.values()], keys=self.names)
    #     self.all_scores = pd.concat([project.all_scores for project in self.projects.values()], keys=self.names)
    #     self.all_data.index.names = ["Project", "Experiment", "Trial", "Trial_index"]

    # def meanCenter(self, alldata: bool = False):
    #     if alldata:
    #         self.all_data = mean_center(self.all_data, self.numeric_columns)
    #     else:
    #         for project in self.projects.values():
    #             project.meanCenter()
    #         self.all_data = mean_center(
    #             pd.concat(
    #                 [
    #                     project.all_data
    #                     for project in self.projects.values()
    #                 ], keys=self.names
    #             ), self.numeric_columns)
    #         self.all_data.index.names = ["Project", "Experiment", "Trial", "Trial_index"]

    # def zScore(self, alldata: bool = False):
    #     if alldata:
    #         self.all_data = z_score(self.all_data, self.numeric_columns)
    #     else:
    #         for project in self.projects.values():
    #             project.zScore()
    #         self.all_data = z_score(
    #             pd.concat(
    #                 [
    #                     project.all_data
    #                     for project in self.projects.values()
    #                 ], keys=self.names
    #             ), self.numeric_columns)
    #         self.all_data.index.names = ["Project", "Experiment", "Trial", "Trial_index"]

    # def normalize(self):
    #     print(self.tabs, "Normalizing...")
    #     for project in self.projects.values():
    #         project.normalize()
    #     self.all_data = z_score(
    #         pd.concat(
    #                 [
    #                     project.all_data
    #                     for project in self.projects.values()
    #                 ], keys=self.names
    #             ),
    #         self.numeric_columns
    #     )
    #     self.all_data.index.names = ["Project", "Experiment", "Trial", "Trial_index"]

    # def runPCA(self, data: pd.DataFrame = None, numeric_columns: list[str] | pd.Index = None):
    #     """Reduces `all_data` to 2 and 3 dimensions using PCA

    #     Initializes attributes:
    #         pcas (dict[str, pd.DataFrame]): a dictionary containing the 2 and 3 dimensional PCA data for each experiment (the keys are 'pca2d', 'pca3d')
    #     """
    #     if data is not None and numeric_columns is not None:
    #         self.pcas = pca(data, numeric_columns)
    #     elif data is not None:
    #         self.pcas = pca(data, self.numeric_columns)
    #     else:
    #         self.pcas = pca(self.data, self.numeric_columns)

    # def visualize(self, dimensions: int, normalized: bool = False, color_column: str = "Trial", filename: str = None, *args, **kwargs) -> go.Figure:
    #     """Plots the data from the PCA

    #     Args:
    #         filename (str, optional): The filename to save the plot to. Defaults to None.
    #     """
    #     if normalized:
    #         self.runPCA()
    #     elif not normalized:
    #         self.runPCA(pd.concat(self.allTrials, keys=range(len(self.allTrials))), self.numeric_columns)
    #     if dimensions == 2:
    #         fig = px.scatter(self.pcas[f"pca{dimensions}d"], x="principal component 1", y="principal component 2", color=color_column)
    #     elif dimensions == 3:
    #         fig = px.scatter_3d(self.pcas[f"pca{dimensions}d"], x="principal component 1", y="principal component 2", z="principal component 3", color=color_column)
    #     else:
    #         raise ValueError("dimensions must be 2 or 3")
    #     if filename is not None:
    #         fig.write_image(filename, *args, **kwargs)
    #     return fig

    # def save(self, filename: str, title: str = None):
    #     """Saves projects data to a HDF5 file

    #     Args:
    #         filename (str): The filename to save the data to
    #     """
    #     with pd.HDFStore(filename) as store:
    #         if title is not None:
    #             store.put(f"{title}/all_data", self.all_data)
    #         else:
    #             store.put("all_data", self.all_data)

    # def saveAll(self, filename: str, title: str = None):
    #     """Saves all the data to a HDF5 file

    #     Args:
    #         filename (str): The filename to save the data to
    #     """
    #     print(self.tabs, "Saving...")
    #     for name in self.names:
    #         self.projects[name].saveAll(filename)
    #     with pd.HDFStore(filename) as store:
    #         if title is not None:
    #             store.put(f"{title}/all_data", self.all_data)
    #         else:
    #             store.put("all_data", self.all_data)

    # @property
    # def data(self) -> pd.DataFrame:
    #     return self.all_data

    # @property
    # def scores(self) -> pd.DataFrame:
    #     return self.all_scores

    # @property
    # def quant_cols(self) -> list[str]:
    #     """Returns the quantitative columns of the data.

    #     Returns:
    #         list[str]: the columns from the data with the target quantitative data.
    #     """
    #     return self.numeric_columns

    # @property
    # def qual_cols(self) -> list[str]:
    #     """Returns the qualitative columns of the data.

    #     Returns:
    #         list[str]: the columns from the data with the qualitative (or rather non-target) data.
    #     """
    #     cols = self.data.reset_index().columns.to_list()
    #     cols = [i for i in cols if i not in self.quant_cols]
    #     return cols

    # @property
    # def cols(self) -> tuple[list[str], list[str]]:
    #     """Returns the target and non target columns of the data.

    #     Returns:
    #         tuple[list[str], list[str]]: a tuple of column lists, the first being the target columns and the second being the non-target columns.
    #     """
    #     return self.quant_cols, self.qual_cols
