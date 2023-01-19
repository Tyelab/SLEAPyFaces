import os
import logging

import pandas as pd

from sleapyfaces.base.expr import Experiment
from sleapyfaces.base.type import BaseType
from config.configuration_set import ConfigurationSet


class Project(BaseType):
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
        ExperimentEventsFile: tuple[str, bool] | str,
        ExperimentSetupFile: tuple[str, bool] | str,
        SLEAPFile: tuple[str, bool],
        VideoFile: tuple[str, bool],
        base: str,
        name: str,
        file_structure: dict[str, str] = None,
        tabs: str = "",
        sublevel: str = "Experiment",
        passed_config: ConfigurationSet = None,
        prefix: str = None,
        *args,
        **kwargs) -> None:

        super().__init__(
            ExperimentEventsFile=ExperimentEventsFile,
            ExperimentSetupFile=ExperimentSetupFile,
            SLEAPFile=SLEAPFile,
            VideoFile=VideoFile,
            file_structure=file_structure,
            passed_config=passed_config,
            base=base,
            name=name,
            tabs=tabs,
            sublevel=sublevel,
            prefix=prefix,
            *args,
            **kwargs
        )

    def _init_data(self):
        logging.info("=========================================")
        logging.info(f"{self.tabs}Initializing Project...{self.name if self.name is not None else ''}")
        logging.info(f"{self.tabs}\tPath:{self.base}")
        logging.info("=========================================")
        logging.debug("=========================================")
        logging.debug(f"{self.tabs}Initializing Data...")
        logging.debug("------------------------------------------")
        logging.debug(f"{self.tabs}\tBase path: {self.base}")
        logging.debug(f"{self.tabs}\tExperiments: {self.fileStruct}")
        logging.debug(f"{self.tabs}\tTransforming to:")
        logging.debug("------------------------------------------")
        logging.debug(f"{self.tabs}\t\tExperiment keys (names): {self.names}")
        logging.debug(f"{self.tabs}\t\tExperiment files (paths): {self.paths}")
        logging.debug("------------------------------------------")
        logging.debug("=========================================")
        for name, file in self.fileStruct.items():
            self.data[name]: Experiment = Experiment(
                name=name,
                base=os.path.join(self.base, file),
                file_structure=False,
                ExperimentEventsFile=self.ExprEventsFile,
                ExperimentSetupFile=self.ExprSetupFile,
                SLEAPFile=self.SLEAPFile,
                VideoFile=self.VideoFile,
                passed_config=self.config,
                sublevel=None,
                tabs=self.tabs+"\t"
            )
        self.numeric_columns: list[str] = self.data[self.names[0]].numeric_columns
        self.all_data: pd.DataFrame = pd.concat([data.all_data for data in self.data.values()], keys=self.names)
        self.all_scores: pd.DataFrame = pd.concat([data.all_scores for data in self.data.values()], keys=self.names)

    def _rename_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index.names = ["Experiment", "Trial", "Trial_index"]
        return df
