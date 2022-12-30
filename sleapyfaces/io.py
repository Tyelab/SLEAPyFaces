from dataclasses import dataclass
from os import PathLike
import pandas as pd
import numpy as np
from io import FileIO
from sleapyfaces.utils import (
    fill_missing,
    json_dumps,
    save_dict_to_hdf5,
    save_dt_to_hdf5,
    tracks_deconstructor,
)
import json
import ffmpeg
import h5py as h5


@dataclass(slots=True)
class DAQData:
    """
    Summary:
        Cache for DAQ data.

    Attrs:
        path (Text or PathLike[Text]): Path to the directory containing the DAQ data.
        cache (pd.DataFrame): Pandas DataFrame containing the DAQ data.
        columns (List): List of column names in the cache.

    Methods:
        append: Append a column to the cache.
        save_data: Save the cache to a csv file.
    """

    path: str | PathLike[str]
    cache: pd.DataFrame
    columns: list

    def __init__(self, path: str | PathLike[str]):
        self.path = path
        self.cache = pd.read_csv(self.path)
        self.columns = self.cache.columns.to_list()[1:]

    def append(self, name: str, value: list) -> None:
        """takes in a list with a name and appends it to the cache as a column

        Args:
            name (str): The column name.
            value (list): The column data.

        Raises:
            ValueError: If the length of the list does not match the length of the cached data.
        """
        if len(list) == len(self.cache.iloc[:, 0]):
            self.cache = pd.concat(
                [self.cache, pd.DataFrame(value, columns=[name])], axis=1
            )
        elif len(list) == len(self.cache.iloc[0, :]):
            self.cache.columns = value
        else:
            raise ValueError("Length of list does not match length of cached data.")

    def saveData(self, filename: str | PathLike[str] | FileIO) -> None:
        """saves the cached data to a csv file

        Args:
            filename (Text | PathLike[Text] | BufferedWriter): the name of the file to save the data to
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
class SLEAPanalysis:
    """
    Summary:
        a class for reading and storing SLEAP analysis files

    Args:
        path (Text | PathLike[Text]): path to the directory containing the SLEAP analysis file

    Attributes:
        data (Dict): dictionary of all the data from the SLEAP analysis file
        track_names (List): list of the track names from the SLEAP analysis file
        tracks (pd.DataFrame): a pandas DataFrame containing the tracks from the SLEAP analysis file
                (with missing frames filled in using a linear interpolation method)

    Methods:
        getDatasets: gets the datasets from the SLEAP analysis file
        getTracks: gets the tracks from the SLEAP analysis file
        getTrackNames: gets the track names from the SLEAP analysis file
        append: appends a column to the tracks DataFrame
        saveData: saves the data to a json file
    """

    path: str | PathLike[str]
    data: dict[str, np.ndarray | pd.DataFrame | list]
    tracks: pd.DataFrame
    track_names: list

    def __init__(self, path: str | PathLike[str]):
        self.path = path
        self.getDatasets()
        self.getTracks()
        self.getTrackNames()

    def getDatasets(
        self,
    ) -> None:
        """gets the datasets from the SLEAP analysis file

        Initializes Attributes:
            data (Dict): dictionary of all the data from the SLEAP analysis file
        """
        self.data = {}
        with h5.File(f"{self.path}", "r") as f:
            datasets = list(f.keys())
            for dataset in datasets:
                if len(f[dataset].shape) == 0:
                    continue
                elif dataset == "tracks":
                    self.data[dataset] = fill_missing(f[dataset][:].T)
                elif "name" in dataset:
                    self.data[dataset] = [n.decode() for n in f[dataset][:].flatten()]
                else:
                    self.data[dataset] = f[dataset][:].T

    def getTracks(self) -> None:
        """gets the tracks from the SLEAP analysis file

        Initializes Attributes:
            tracks (pd.DataFrame): a pandas DataFrame containing the tracks from the SLEAP analysis file
            (with missing frames filled in using a linear interpolation method)
        """
        if len(self.data.values()) == 0:
            raise ValueError("No data has been loaded.")
        else:
            self.tracks = tracks_deconstructor(
                self.data["tracks"], self.data["node_names"]
            )

    def getTrackNames(self) -> None:
        """gets the track names from the SLEAP analysis file

        Initializes Attributes:
            track_names (List): list of the track names from the SLEAP analysis file
        """
        self.track_names = [0] * (len(self.data["node_names"]) * 2)
        for name, i in zip(
            self.data["node_names"], range(0, (len(self.data["node_names"]) * 2), 2)
        ):
            self.track_names[i] = f"{name.replace(' ', '_')}_x"
            self.track_names[i + 1] = f"{name.replace(' ', '_')}_y"

    def append(self, item: pd.Series | pd.DataFrame) -> None:
        """Appends a column to the tracks DataFrame

        Args:
            item (pd.Series | pd.DataFrame): The column to append to the tracks DataFrame

        Raises:
            ValueError: if the length of the column does not match the length of the tracks data columns
            (i.e. if the column is not the same length as the number of frames)

        Updates Attributes:
            tracks (pd.DataFrame): a pandas DataFrame containing the tracks from the SLEAP analysis file
        """
        if len(item.index) == len(self.tracks.index):
            self.tracks = pd.concat([self.tracks, item], axis=1)
        else:
            raise ValueError("Length of list does not match length of cached data.")

    def saveData(self, filename: str | PathLike[str], path="SLEAP") -> None:
        """saves the SLEAP analysis data to an HDF5 file

        Args:
            filename (Text | PathLike[Text]): the name of the file to save the data to
            path (str, optional): the internal HDF5 path to save the data to. Defaults to "SLEAP".
        """
        if filename.endswith(".h5") or filename.endswith(".hdf5"):
            with h5.File(filename) as f:
                save_dict_to_hdf5(f, path, self.datasets)
            with pd.HDFStore(filename, mode="a") as store:
                save_dt_to_hdf5(store, self.tracks, f"{path}/tracks")


@dataclass(slots=True)
class BehMetadata:
    """
    Summary:
        Cache for JSON data.

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

    def __init__(self, path: str | PathLike[str]):
        self.path = path
        self.cache = ffmpeg.probe(f"{self.path}")["streams"][
            (int(ffmpeg.probe(f"{self.path}")["format"]["nb_streams"]) - 1)
        ]
        self.fps = float(eval(self.cache.get("avg_frame_rate")))

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
