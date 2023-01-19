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
        logging.info(f"{self.tabs}Initializing Projects...")
        logging.info("=========================================")
        logging.debug("=========================================")
        logging.debug(f"{self.tabs}Initializing Data...")
        logging.debug("------------------------------------------")
        logging.debug(f"{self.tabs}\tBase path: {self.base}")
        logging.debug(f"{self.tabs}\tProjects: {self.fileStruct}")
        logging.debug(f"{self.tabs}\tTransforming to:")
        logging.debug("------------------------------------------")
        logging.debug(f"{self.tabs}\t\tProject keys (names): {self.names}")
        logging.debug(f"{self.tabs}\t\tProject files (paths): {self.paths}")
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
