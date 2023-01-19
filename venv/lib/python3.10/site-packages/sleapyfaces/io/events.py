from dataclasses import dataclass
from os import PathLike
import pandas as pd
from io import FileIO

@dataclass(slots=True)
class EventsData:
    """
    Summary:
        Cache for DAQ events data.

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

    def __init__(self, path: str | PathLike[str], tabs: str = ""):
        self.path = path
        self.cache = pd.read_csv(self.path)
        self.columns = self.cache.columns.to_list()[1:]
        print(tabs, "DAQ data loaded.")
        print(tabs + "\t", f"Columns: {self.columns}")

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
