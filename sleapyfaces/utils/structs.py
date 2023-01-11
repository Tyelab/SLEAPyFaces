from dataclasses import dataclass
from os import PathLike
import os
import glob
import pandas as pd


@dataclass(slots=True)
class File:
    """A structured file object that contains the base path and filename of a file.

    Class Attributes:
        file_str: the location of the file.

    Instance Attributes:
        file: the location of the file.
        filename: the name of the file.
        basepath: the base path of the file.
        iPath: the ith base path of the file path. (i.e. iPath(1) returns the second to last path in the file path.)
    """

    get_glob: bool
    file: str | PathLike[str]
    basepath: str | PathLike[str]
    filename: str

    def __init__(self, basepath: str | PathLike[str], filename: str, get_glob=False):
        self.basepath = basepath
        self.filename = filename
        self.get_glob = get_glob

        self.file = glob.glob(os.path.join(self.basepath, self.filename)) if self.get_glob else os.path.join(self.basepath, self.filename)
        self.file = self.file[0] if type(self.file) is list else self.file

    def iPath(self, i: int) -> str:
        """Returns the ith path in the file path."""
        return "/".join(self.file.split("/")[:-i])


@dataclass(slots=True)
class FileConstructor:

    """Takes in the base paths and filenames of the experimental data and returns them as a structured object.

    Args:
        DAQFile (File): The location of the DAQ data file.
        SLEAPFile (File): The location of the SLEAP analysis file.
        BehFile (File): The location of the behavioral metadata file.
        VideoFile (File): The location of the video file.

    Attributes:
        daq (File): The location of the DAQ file as a structured File object.
        sleap (File): The location of the SLEAP analysis file as a structured File object.
        beh (File): The location of the behavioral metadata file as a structured File object.
        video (File): The location of the video file as a structured File object.
    """

    daq: File
    sleap: File
    beh: File
    video: File

    def __init__(
        self, daq_file: File, sleap_file: File, beh_file: File, video_file: File
    ) -> None:
        self.daq = daq_file
        self.sleap = sleap_file
        self.beh = beh_file
        self.video = video_file


@dataclass(slots=True)
class CustomColumn:
    """Builds an annotation column for the base dataframe.

    Attrs:
        ColumnTitle (str): The title of the column.
        ColumnData (str | int | float | bool): The data to be added to the column.
        Column (pd.Series): The column to be added to the base dataframe.
    """

    ColumnTitle: str
    ColumnData: str | int | float | bool
    Column: pd.Series

    def __init__(self, ColumnTitle: str, ColumnData: str | int | float | bool):
        self.ColumnTitle = ColumnTitle
        self.ColumnData = ColumnData

    def buildColumn(self, length: int) -> None:
        """Initializes a column of a given length.

        Args:
            length (int): The length of the column to be built.

        Initializes Attributes:
            Column (pd.DataFrame): The initialized column at a given length.
        """
        self.Column = [self.ColumnData] * length
        self.Column = pd.DataFrame(self.Column, columns=[self.ColumnTitle])
