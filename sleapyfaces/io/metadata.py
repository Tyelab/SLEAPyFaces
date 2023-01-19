from dataclasses import dataclass
from os import PathLike
import pandas as pd
from io import FileIO
from sleapyfaces.utils.io import json_dumps
import json
import ffmpeg

@dataclass(slots=True)
class ExprMetadata:
    """
    Summary:
        Cache for experimental metadata data.

    Args:
        path (str of PathLike[str]): Path to the file containing the JSON data.
        MetaDataKey (str, optional): Key for the metadata in the JSON data. Defaults to "beh_metadata" based on bruker_control.
        TrialArrayKey (str, optional): Key for the trial array in the JSON data. Defaults to "trialArray" based on bruker_control.
        ITIArrayKey (str, optional): Key for the ITI array in the JSON data. Defaults to "ITIArray" based on bruker_control.

        Bruker Control Repository:
            Link: https://github.com/Tyelab/bruker_control
            Author: Jeremy Delahanty

    Attributes:
        cache (pd.DataFrame): Pandas DataFrame containing the JSON data.
        columns (List): List of column names in the cache.

    Methods:
        saveData: saves the data to a json file"""

    path: str | PathLike[str]
    MetaDataKey: str
    TrialArrayKey: str
    ITIArrayKey: str
    cache: pd.DataFrame
    columns: list[str]

    def __init__(
        self,
        path: str | PathLike[str],
        MetaDataKey="beh_metadata",
        TrialArrayKey="trialArray",
        ITIArrayKey="ITIArray",
        tabs: str = "",
    ):
        self.path = path
        self.MetaDataKey = MetaDataKey
        self.TrialArrayKey = TrialArrayKey
        self.ITIArrayKey = ITIArrayKey

        with open(self.path, "r") as json_file:
            json_file = json.load(json_file)
            trialArray = json_file.get(self.MetaDataKey)[self.TrialArrayKey]
            ITIArray = json_file.get(self.MetaDataKey)[self.ITIArrayKey]
        self.cache = pd.DataFrame(
            {self.TrialArrayKey: trialArray, self.ITIArrayKey: ITIArray},
            columns=[self.TrialArrayKey, self.ITIArrayKey],
        )
        self.columns = self.cache.columns.to_list()
        print(tabs, "Experimental metadata loaded.")
        print(tabs + "\t", f"Columns: {self.columns}")
        print()

    def saveData(self, filename: str | PathLike[str] | FileIO) -> None:
        """Saves the DAQ data to a csv file.

        Args:
            filename (str | PathLike[str] | FileIO): The name and path of the file to save the data to.
        """
        if (
            filename.endswith(".csv")
            or filename.endswith(".CSV")
            or isinstance(filename, FileIO)
        ):
            self.cache.to_csv(filename, index=True)
        else:
            self.cache.to_csv(f"{filename}.csv", index=True)


@dataclass(slots=True)
class VideoMetadata:
    """
    Summary:
        class for caching the video metadata.

    Args:
        path (str of PathLike[str]): Path to the directory containing the video data.

    Attributes:
        cache (dict): Dictionary containing the video metadata from ffmpeg.
        fps (float): Frames per second of the video data.

    Methods:
        saveData: saves the data to a json file
    """

    path: str | PathLike[str]
    cache: dict
    fps: float

    def __init__(self, path: str | PathLike[str], tabs: str = ""):
        self.path = path
        self.cache = ffmpeg.probe(f"{self.path}")["streams"][
            (int(ffmpeg.probe(f"{self.path}")["format"]["nb_streams"]) - 1)
        ]
        self.fps = float(eval(self.cache.get("avg_frame_rate")))
        print(tabs, "Video metadata loaded.")
        print(tabs + "\t", f"Average FPS: {self.fps}")

    def saveData(self, filename: str | PathLike[str] | FileIO) -> None:
        """Saves the video metadata to a json file.

        Args:
            filename (str | PathLike[str] | FileIO): the name and path of the file to save the data to.
        """
        if (
            filename.endswith(".json")
            or filename.endswith(".JSON")
            or isinstance(filename, FileIO)
        ):
            json_dumps(self.cache, filename)
        else:
            json_dumps(self.cache, f"{filename}.csv")
