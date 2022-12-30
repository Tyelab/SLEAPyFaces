
# API Reference

----

## Table of Contents

- [API Reference](#api-reference)
	- [Table of Contents](#table-of-contents)
	- [`sleapyfaces.project.Project`](#sleapyfacesprojectproject)
		- [`sleapyfaces.project.Project.buildColumns`](#sleapyfacesprojectprojectbuildcolumns)
		- [`sleapyfaces.project.Project.buildTrials`](#sleapyfacesprojectprojectbuildtrials)
		- [`sleapyfaces.project.Project.meanCenter`](#sleapyfacesprojectprojectmeancenter)
		- [`sleapyfaces.project.Project.zScore`](#sleapyfacesprojectprojectzscore)
		- [`sleapyfaces.project.Project.analyze`](#sleapyfacesprojectprojectanalyze)
		- [`sleapyfaces.project.Project.visualize`](#sleapyfacesprojectprojectvisualize)
	- [`sleapyfaces.experiment.Experiment`](#sleapyfacesexperimentexperiment)
		- [`sleapyfaces.experiment.Experiment.buildData`](#sleapyfacesexperimentexperimentbuilddata)
		- [`sleapyfaces.experiment.Experiment.append`](#sleapyfacesexperimentexperimentappend)
		- [`sleapyfaces.experiment.Experiment.buildTrials`](#sleapyfacesexperimentexperimentbuildtrials)
	- [`sleapyfaces.io.DAQData`](#sleapyfacesiodaqdata)
		- [`sleapyfaces.io.DAQData.append`](#sleapyfacesiodaqdataappend)
		- [`sleapyfaces.io.DAQData.saveData`](#sleapyfacesiodaqdatasavedata)
	- [`sleapyfaces.io.SLEAPanalysis`](#sleapyfacesiosleapanalysis)
		- [`sleapyfaces.io.SLEAPanalysis.getDatasets`](#sleapyfacesiosleapanalysisgetdatasets)
		- [`sleapyfaces.io.SLEAPanalysis.getTracks`](#sleapyfacesiosleapanalysisgettracks)
		- [`sleapyfaces.io.SLEAPanalysis.getTrackNames`](#sleapyfacesiosleapanalysisgettracknames)
		- [`sleapyfaces.io.SLEAPanalysis.append`](#sleapyfacesiosleapanalysisappend)
		- [`sleapyfaces.io.SLEAPanalysis.saveData`](#sleapyfacesiosleapanalysissavedata)
	- [`sleapyfaces.io.BehMetadata`](#sleapyfacesiobehmetadata)
		- [`sleapyfaces.io.BehMetadata.saveData`](#sleapyfacesiobehmetadatasavedata)
	- [`sleapyfaces.io.VideoMetadata`](#sleapyfacesiovideometadata)
		- [`sleapyfaces.io.VideoMetadata.saveData`](#sleapyfacesiovideometadatasavedata)
	- [`sleapyfaces.normalize.mean_center`](#sleapyfacesnormalizemean_center)
	- [`sleapyfaces.normalize.z_score`](#sleapyfacesnormalizez_score)
	- [`sleapyfaces.normalize.pca`](#sleapyfacesnormalizepca)
	- [`sleapyfaces.structs.File`](#sleapyfacesstructsfile)
	- [`sleapyfaces.structs.FileConstructor`](#sleapyfacesstructsfileconstructor)
	- [`sleapyfaces.structs.CustomColumn`](#sleapyfacesstructscustomcolumn)
		- [`sleapyfaces.structs.CustomColumn.buildColumn`](#sleapyfacesstructscustomcolumnbuildcolumn)


## `sleapyfaces.project.Project`

```python
proj = sleapyfaces.project.Project(
    DAQFile: str,
    BehFile: str,
    SLEAPFile: str,
    VideoFile: str,
    base: str,
    iterator: dict[str, str] | None, # Optional
    get_glob: bool = False, # Optional
    )
```

- `Project` is the main class for the `sleapyfaces` package. It is used to initialize the project and iterate over the project structure.
  - Args:
    - `DAQFile` (string): The naming convention for the DAQ files (e.g. "*_events.csv" or "DAQOutput.csv")
    - `ExprMetaFile` (string): The naming convention for the experimental structure files (e.g. "*_config.json" or "BehMetadata.json")
    - `SLEAPFile` (string): The naming convention for the SLEAP files (e.g. "*_sleap.h5" or "SLEAP.h5")
    - `VideoFile` (string): The naming convention for the video files (e.g. "*.mp4" or "video.avi")
    - `base` (string): Base path of the project (e.g. "/specialk_cs/2p/raw/CSE009")
    - `iterator` (dictionary[string, string], optional): Iterator for the project files, with keys as the label and values as the folder name (e.g. {"week 1": "20211105", "week 2": "20211112"})
    - `get_glob` (boolean, default = `False`): Whether to use glob to find the files (e.g. True or False). Default is False.
      - NOTE: if glob is True, make sure to at least include the file extension with an asterisk in the naming convention
      - e.g. `DAQFile="*.csv"` or `SLEAPFile="*.h5"`
  - Attributes:
    - `exprs` (list[`Experiment`]): List of the experiment objects
    - `files` (list[`FileConstructor`]): `FileConstructor` object


### `sleapyfaces.project.Project.buildColumns`

```python
proj.buildColumns(
    columns: list[str],
    values: list[str],
    )
```

- Builds the custom columns for the project and builds the data for each experiment
  - Args:
    - `columns` (list[string]): the column titles
    - `values` (list[any]): the data for each column
  - Initializes attributes:
    - `custom_columns` (list[`CustomColumn`]): list of custom columns
    - `all_data` (`pd.DataFrame`): the data for all experiments concatenated together


### `sleapyfaces.project.Project.buildTrials`

```python
proj.buildTrials(
    TrackedData: list[str],
    Reduced: list[bool],
    start_buffer: int = 10000, # Optional
    end_buffer: int = 13000, # Optional
    )
```

- Parses the data from each experiment into its individual trials
  - Args:
    - `TrackedData` (list[string]): The title of the columns from the DAQ data to be tracked
    - `Reduced` (list[boolean]): The corresponding boolean for whether the DAQ data is to be reduced (`True`) or not (`False`)
    - `start_buffer` (integer, optional): The time in milliseconds before the trial start to capture. Defaults to 10000.
    - `end_buffer` (integer, optional): The time in milliseconds after the trial start to capture. Defaults to 13000.
  - Initializes attributes:
    - `exprs[i].trials` (`pd.DataFrame`): the data frame containing the concatenated trial data for each experiment
    - `exprs[i].trialData` (list[`pd.DataFrame`]): the list of data frames containing the trial data for each trial for each experiment


### `sleapyfaces.project.Project.meanCenter`

```python
proj.meanCenter()
```

- Recursively mean centers the data for each trial for each experiment
  - Initializes attributes:
    - `all_data` (`pd.DataFrame`): the mean centered data for all trials and experiments concatenated together


### `sleapyfaces.project.Project.zScore`

```python
proj.zScore()
```

- Z scores the mean centered data for each experiment
  - Updates attributes:
    - `all_data` (`pd.DataFrame`): the z-scored data for all experiments concatenated together


### `sleapyfaces.project.Project.analyze`

```python
proj.analyze()
```

- Runs the mean centering and z scoring functions


### `sleapyfaces.project.Project.visualize`

```python
proj.visualize()
```

- Reduces `all_data` to 2 and 3 dimensions using PCA
  - Initializes attributes:
    - `pcas` (dictionary[string, `pd.DataFrame`]): a dictionary containing the 2 and 3 dimensional PCA data for each experiment
    - the dictionary keys are `'pca2d'`, `'pca3d'`

----

## `sleapyfaces.experiment.Experiment`

```python
expr = sleapyfaces.experiment.Experiment(
    name: str,
    files: sleapyfaces.structs.FileConstructor,
    )
```

- Class constructor for the Experiment object.
  - Args:
    - `name` (string): The name of the experiment.
    - `files` (`FileConstructor`): The `FileConstructor` object containing the paths to the experiment files.
  - Attributes:
    - `name` (string): The name of the experiment.
    - `files` (`FileConstructor`): The `FileConstructor` object containing the paths to the experiment files.
    - `sleap` (`SLEAPanalysis`): The `SLEAPanalysis` object containing the SLEAP data.
    - `beh` (`BehMetadata`): The `BehMetadata` object containing the behavior metadata.
    - `video` (`VideoMetadata`): The `VideoMetadata` object containing the video metadata.
    - `daq` (`DAQData`): The `DAQData` object containing the DAQ data.
    - `numeric_columns` (list[string]): A list of the titles of the numeric columns in the SLEAP data.


### `sleapyfaces.experiment.Experiment.buildData`

```python
expr.buildData(
    CustomColumns: list[sleapyfaces.structs.CustomColumn]
    )
```

- Builds the data for the experiment.
  - Args:
    - `CustomColumns` (list[`CustomColumn`]): A list of the `CustomColumn` objects to be added to the experiment.
  - Initializes attributes:
    - `sleap.tracks` (`pd.DataFrame`): The SLEAP data.
    - `custom_columns` (`pd.DataFrame`): The non-numeric columns.


### `sleapyfaces.experiment.Experiment.append`

```python
expr.append(
    item: pd.Series | pd.DataFrame
    )
```

- Appends a column to the SLEAP data.
  - Args:
    - `item` (`pd.Series` or `pd.DataFrame`): A pandas series or dataframe to be appended to the SLEAP data.
  - Updates attributes:
    - `sleap.tracks` (`pd.DataFrame`): The SLEAP data.
    - `custom_columns` (`pd.DataFrame`): The non-numeric columns.


### `sleapyfaces.experiment.Experiment.buildTrials`

```python
expr.buildTrials(
    TrackedData: list[str],
    Reduced: list[bool],
    start_buffer: int = 10000, # Optional
    end_buffer: int = 13000, # Optional
    )
```

- Converts the data into trial by trial format.
  -  Args:
     -  `TrackedData` (list[string]): the list of columns from the DAQ data that signify the START of each trial.
     -  `Reduced` (list[boolean]): a boolean list with the same length as the TrackedData list that signifies the columns from the tracked data with quick TTL pulses that occur during the trial.
        -  (e.g. the LED TTL pulse may signify the beginning of a trial, but during the trial the LED turns on and off, so the LED TTL column should be marked as True)
     -  `start_buffer` (integer, optional): The time in miliseconds you want to capture before the trial starts. Defaults to 10000 (i.e. 10 seconds).
     -  `end_buffer` (integer, optional): The time in miliseconds you want to capture after the trial starts. Defaults to 13000 (i.e. 13 seconds).
  -  Initializes attributes:
     -  `trials` (`pd.DataFrame`): the dataframe with the data in trial by 	trial format, with a metaindex of trial number and frame number
     -  `trialData` (list[`pd.DataFrame`]): a list of the dataframes with the individual trial data.

----

## `sleapyfaces.io.DAQData`

```python
daq = sleapyfaces.io.DAQData(
    path: str | PathLike[string]
    )
```
- Cache for DAQ data.
  - Args:
    - `path` (string or PathLike[string]): Path to the directory containing the DAQ data.
  - Attrs:
    - `cache` (`pd.DataFrame`): Pandas DataFrame containing the DAQ data.
    - `columns` (list[string]): List of column names in the cache.


### `sleapyfaces.io.DAQData.append`

```python
daq.append(
    name: str,
    value: list[any]
    )
```
- Takes in a list with a name and appends it to the cache as a column
  - Args:
    - `name` (string): The column name.
    - `value` (list[any]): The column data.
  - Updates attributes:
    - `cache` (`pd.DataFrame`): Pandas DataFrame containing the DAQ data.
    - `columns` (list[string]): List of column names in the cache.


### `sleapyfaces.io.DAQData.saveData`

```python
daq.saveData(
    filename: str | PathLike[str] | FileIO,
    )
```
- saves the cached data to a csv file
  - Args:
    - `filename` (Text | PathLike[Text] | BufferedWriter): the path and name of the file to save the data to

----

## `sleapyfaces.io.SLEAPanalysis`

```python
sleap = sleapyfaces.io.SLEAPanalysis(
    path: str | PathLike[string]
    )
```
- a class for reading and storing `SLEAP` analysis files
  - Args:
    - `path` (string or PathLike[string]): path to the directory containing the SLEAP analysis file
  - Attributes:
    - `data` (dictionary): dictionary of all the data from the `SLEAP` analysis file
    - `track_names` (list): list of the track names from the `SLEAP` analysis file
    - `tracks` (`pd.DataFrame`): a pandas DataFrame containing the tracks from the `SLEAP` analysis file (with missing frames filled in using a linear interpolation method)

### `sleapyfaces.io.SLEAPanalysis.getDatasets`

```python
sleap.getDatasets()
```
- gets the datasets from the `SLEAP` analysis file
  - Initializes Attributes:
    - `data` (dictionary): dictionary of all the data from the `SLEAP` analysis file

### `sleapyfaces.io.SLEAPanalysis.getTracks`

```python
sleap.getTracks()
```
- gets the tracks from the `SLEAP` analysis file
  - Initializes Attributes:
    - `tracks` (`pd.DataFrame`): a pandas DataFrame containing the tracks from the `SLEAP` analysis file (with missing frames filled in using a linear interpolation method)

### `sleapyfaces.io.SLEAPanalysis.getTrackNames`

```python
sleap.getTrackNames()
```
- gets the track names from the `SLEAP` analysis file
  - Initializes Attributes:
    - `track_names` (List): list of the track names from the `SLEAP` analysis file

### `sleapyfaces.io.SLEAPanalysis.append`

```python
sleap.append(
    item: pd.Series | pd.DataFrame
)
```
- Appends a column to the tracks DataFrame
  - Args:
    - `item` (`pd.Series` or `pd.DataFrame`): The column to append to the tracks DataFrame
  - Updates Attributes:
    - `tracks` (`pd.DataFrame`): a pandas DataFrame containing the tracks from the `SLEAP` analysis file

### `sleapyfaces.io.SLEAPanalysis.saveData`

```python
sleap.saveData(
    filename: str | PathLike[str],
    path: str="SLEAP" # Optional
)
```
- saves the modified `SLEAP` analysis data to an HDF5 file
  - Args:
    - `filename` (string or PathLike[string]): the path and name of the file to save the data to
    - path (string, optional): the internal HDF5 path to save the data to. Defaults to "SLEAP".

----

## `sleapyfaces.io.BehMetadata`

```python
beh = sleapyfaces.io.BehMetadata(
    path: str | PathLike[string],
    MetaDataKey: str="beh_metadata", # Optional
    TrialArrayKey: str="trialArray", # Optional
    ITIArrayKey: str="ITIArray", # Optional
    )
```
- Cache for JSON data.
  - Args:
    - `path` (string of PathLike[string]): Path to the file containing the JSON data.
    - `MetaDataKey` (string, optional): Key for the metadata in the JSON data. Defaults to `"beh_metadata"` based on `bruker_control`.
    - `TrialArrayKey` (string, optional): Key for the trial array in the JSON data. Defaults to `"trialArray"` based on `bruker_control`.
    - `ITIArrayKey` (string, optional): Key for the inter-trial interval array in the JSON data. Defaults to `"ITIArray"` based on `bruker_control`.
      - Bruker Control Repository:
        - Link: (https://github.com/Tyelab/bruker_control)[https://github.com/Tyelab/bruker_control]
        - Author: Jeremy Delahanty
  - Attributes:
    - `cache` (`pd.DataFrame`): Pandas DataFrame containing the JSON data.
    - `columns` (list): List of column names in the cache.

### `sleapyfaces.io.BehMetadata.saveData`

```python
beh.saveData(
    filename: str | PathLike[str] | FileIO,
    )
```
- Saves the DAQ data to a *CSV* file.
  - Args:
    - `filename` (string or PathLike[string] or FileIO): The name and path of the file to save the data to.

----

## `sleapyfaces.io.VideoMetadata`

```python
video = sleapyfaces.io.VideoMetadata(
    path: str | PathLike[string],
    )
```
- A class for caching the video metadata.
  - Args:
    - `path` (string of PathLike[string]): Path to the directory containing the video metadata.
  - Attributes:
    - `cache` (dictionary): Dictionary containing the video metadata from ffmpeg.
    - `fps` (float): The precise frames per second of the video.

### `sleapyfaces.io.VideoMetadata.saveData`

```python
video.saveData(
    filename: str | PathLike[str] | FileIO,
    )
```
- Saves the video metadata to a json file.
  - Args:
    - `filename` (string or PathLike[string] or FileIO): the name and path of the file to save the data to.

----

## `sleapyfaces.normalize.mean_center`

```python
sleapyfaces.normalize.mean_center(
    data: pd.DataFrame,
    track_names: list[str]
    ) -> pd.DataFrame
```

- Mean centers the data for each track.
  - Args:
    - `data` (`pd.DataFrame`): The data to mean center.
    - `track_names` (list[str]): The names of the tracks to mean center.
  - Returns:
    - `pd.DataFrame`: The mean centered data.

----

## `sleapyfaces.normalize.z_score`

```python
sleapyfaces.normalize.z_score(
    data: pd.DataFrame,
    track_names: list[str]
    ) -> pd.DataFrame
```

- Z-score the data.
  - Args:
    - `data` (`pd.DataFrame`): The data to z-score.
    - `track_names` (list[str]): The names of the tracks to z-score.
  - Returns:
    - `pd.DataFrame`: The z-scored data.

----

## `sleapyfaces.normalize.pca`

```python
sleapyfaces.normalize.pca(
    data: pd.DataFrame,
    track_names: list[str]
    ) -> dict[str, pd.DataFrame]
```

- Runs 2D and 3D PCA dimensionality reduction on the data.
  - Args:
    - `data` (`pd.DataFrame`): The data to be reduced.
    - `track_names` (list[str]): The names of the tracks to be reduced.
  - Returns:
    - dictionary[string, `pd.DataFrame`]: The reduced data with keys "pca2d" and "pca3d".
      -i.e. `{"pca2d": pd.DataFrame, "pca3d": pd.DataFrame}`

----

## `sleapyfaces.structs.File`

```python
file = sleapyfaces.structs.File(
    basepath: str | PathLike[str],
    filename: str,
    get_glob: bool=False, # Optional
    )
```
- A structured file object that contains the base path and filename of a file.
  - Args:
    - `basepath` (string): the path to the file.
    - `filename` (string): the name of the file, or a glob pattern if `get_glob` is `True`.
    - `get_glob` (bool, optional): whether or not to get the glob pattern. Defaults to `False`.
  - Attributes:
    - `file` (string): the complete filepath
    - `iPath(i)` (string): the path to the `i`th file in the glob pattern.

----

## `sleapyfaces.structs.FileConstructor`

```python
files = sleapyfaces.structs.FileConstructor(
    daq: File,
    sleap: File,
    beh: File,
    video: File
    )
```
- Takes in the base paths and filenames of the experimental data and returns them as a structured object.
  - Args/Attributes:
    - `daq` (`File`): The location of the DAQ data file.
    - `sleap` (`File`): The location of the SLEAP analysis file.
    - `beh` (`File`): The location of the behavioral metadata file.
    - `video` (`File`): The location of the video file.

----

## `sleapyfaces.structs.CustomColumn`

```python
column = sleapyfaces.structs.CustomColumn(
    ColumnTitle: str,
    ColumnData: str | int | float | bool
    )
```
- Takes in the base paths and filenames of the experimental data and returns them as a structured object.
  - Args:
    - `ColumnTitle` (string): The title of the column.
    - `ColumnData` (string or integer or float or boolean): The data to be added to the column.

### `sleapyfaces.structs.CustomColumn.buildColumn`

```python
column.buildColumn(
    length: int,
    )
```
- Initializes a column of a given length.
  - Args:
    - `length` (integer): The length of the column to be built.
  - Initializes Attributes:
    - `Column` (`pd.DataFrame`): The initialized column at a given length.
