from dataclasses import dataclass
from os import PathLike
import pandas as pd
import numpy as np
from sleapyfaces.utils.io import (
    save_dict_to_hdf5,
    save_dt_to_hdf5,
)
from sleapyfaces.utils.reform import (
    fill_missing,
    tracks_deconstructor
)
import h5py as h5

@dataclass(slots=True)
class SLEAPData:
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
    scores: pd.DataFrame
    track_names: list[str]
    nodes: list[str]

    def __init__(self, path: str | PathLike[str], tabs: str = ""):
        self.path = path
        self.getDatasets()
        self.getTracks()
        self.getTrackNames()
        print(tabs, "SLEAP analysis loaded.")
        print(tabs + "\t", f"Tracks: {self.track_names}")

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
            self.nodes = [name.replace(" ", "_") for name in self.data["node_names"]]
            self.tracks = tracks_deconstructor(
                self.data["tracks"], self.data["node_names"]
            )
            self.scores = pd.DataFrame(np.squeeze(self.data.get('point_scores')), columns=self.nodes)

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
