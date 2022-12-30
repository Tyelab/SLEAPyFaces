# SLEAPyFaces

*A package for extracting facial expressions from SLEAP analyses with sensible assumptions.*

Based on [these](https://github.com/annie444/Facial-Expression-Analyses) scripts

[![PyPI - Version](https://img.shields.io/pypi/v/sleapyfaces.svg)](https://pypi.org/project/sleapyfaces)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sleapyfaces.svg)](https://pypi.org/project/sleapyfaces)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

----

## Table of Contents

- [SLEAPyFaces](#sleapyfaces)
	- [Table of Contents](#table-of-contents)
	- [Description](#description)
		- [Citing:](#citing)
		- [License](#license)
	- [Getting Started:](#getting-started)
		- [Installation](#installation)
			- [Using pip:](#using-pip)
			- [Using conda (coming soon):](#using-conda-coming-soon)
		- [Basic Usage Step by Step](#basic-usage-step-by-step)
			- [Importing the package](#importing-the-package)
			- [Creating a project](#creating-a-project)
			- [Building the data](#building-the-data)
			- [Building the trials](#building-the-trials)
			- [Normalizing the data](#normalizing-the-data)
		- [Complete Example](#complete-example)
	- [Project Structure](#project-structure)
		- [Data Files Formats](#data-files-formats)
			- [DAQ file](#daq-file)
			- [Behavioral metadata file](#behavioral-metadata-file)
			- [SLEAP file](#sleap-file)
			- [Video file](#video-file)
		- [Project Directory Structure](#project-directory-structure)
	- [API Documentation](#api-documentation)
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
	- [Changelog:](#changelog)
		- [Version 1.0.0 (2022-12-27)](#version-100-2022-12-27)
		- [Version 1.0.1](#version-101)
		- [Version 1.1.0 (in progress)](#version-110-in-progress)

----

## Description

Sleapyfaces is a data analysis package for extracting facial expressions of mice from SLEAP analyses. It is designed to work with the SLEAP software package, which provides a graphical user interface for annotating animal behavioral videos. More information on SLEAP is available at [https://sleap.ai](https://sleap.ai). This package also depends on many assumptions about the data format and structure of the SLEAP analyses. It is not intended to be a general tool for extracting facial expressions from SLEAP analyses, but rather a tool for extracting facial expressions from the specific data format and structure used in the lab of Dr. Kay Tye at The Salk Institute for Biological Studies.

### Citing:

If you use SLEAPyFaces in your research, this does not fall under *standard software* according to the *Publication Manual* for the APA, MLA, AMA, Turabian, IEEE, Vancouver style, Harvard style, or Chicago style guides. **Please cite the following:**

> A. Ehler,  J. Delahantey, A. Coley, D. LeDuke, L. Keyes, T.D. Pereira, and K. Tye. SLEAPyFaces: A package for extracting facial expressions from SLEAP analyses with sensible assumptions. SLEAPyFaces python package, v1.0.1, 2023. Retrieved from [https://github.com/annie444/sleapyfaces/](https://github.com/annie444/sleapyfaces/)

**BibTeX:**
```bibtex
 @misc{Ehler2023sleapyfaces,
    title={SLEAPyFaces: A package for extracting facial expressions from SLEAP analyses with sensible assumptions.},
    author={
        Ehler, Analetta and
        Delahantey, Jeremey and
        Coley, Austin and
        LeDuke, Deryn and
        Keyes, Laurel and
        Pereira, Talmo D and
        Tye, Kay},
    url={https://github.com/annie444/sleapyfaces/},
    journal={SLEAPyFaces python package},
    publisher={GitHub repository},
    volume={v1.0.1},
    year={2023},
    month={Dec},
    day={27}
}
```

**Please also cite the original SLEAP paper:**

>T.D. Pereira, N. Tabris, A. Matsliah, D. M. Turner, J. Li, S. Ravindranath, E. S. Papadoyannis, E. Normand, D. S. Deutsch, Z. Y. Wang, G. C. McKenzie-Smith, C. C. Mitelut, M. D. Castro, J. D’Uva, M. Kislin, D. H. Sanes, S. D. Kocher, S. S-H, A. L. Falkner, J. W. Shaevitz, and M. Murthy. Sleap: A deep learning system for multi-animal pose tracking. Nature Methods, 19(4), 2022

**BibTeX:**
```bibtex
@ARTICLE{Pereira2022sleap,
   title={SLEAP: A deep learning system for multi-animal pose tracking},
   author={Pereira, Talmo D and
      Tabris, Nathaniel and
      Matsliah, Arie and
      Turner, David M and
      Li, Junyu and
      Ravindranath, Shruthi and
      Papadoyannis, Eleni S and
      Normand, Edna and
      Deutsch, David S and
      Wang, Z. Yan and
      McKenzie-Smith, Grace C and
      Mitelut, Catalin C and
      Castro, Marielisa Diez and
      D'Uva, John and
      Kislin, Mikhail and
      Sanes, Dan H and
      Kocher, Sarah D and
      Samuel S-H and
      Falkner, Annegret L and
      Shaevitz, Joshua W and
      Murthy, Mala},
   journal={Nature Methods},
   volume={19},
   number={4},
   year={2022},
   publisher={Nature Publishing Group}
   }
}
```

### License

`sleapyfaces` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license. The full license text is available in the [LICENSE](LICENSE) file.

----
## Getting Started:
----

### Installation

#### Using pip:

```console
pip install sleapyfaces
```

#### Using conda (coming soon):

```console
conda install -c conda-forge sleapyfaces
```
-----

### Basic Usage Step by Step

#### Importing the package

For **basic** usage, the package can be imported with the following:

```python
from sleapyfaces.project import Project
```

Alternatively, the package can be imported with the following:

```python
import sleapyfaces as sf
```

#### Creating a project

The project object is the basis for the data analysis. It serves as a container for all of the data produced from a series of experiments, and allows a root object to be created for the data analysis. The project object is initialized with a series of file or file glob patterns, which are used to find the files associated with each experiment. The project object also requires a base path, which is the root directory for the project. The project object can be initialized with the optional `get_glob` argument set to `True`, which will automatically find the files associated with each experiment. Alternatively, the `get_glob` argument can be set to `False` (default), which will allow the user to manually set the files associated with each experiment as the file arguments. Additionally, there is an `iterators` argument, which is a dictionary of iterators that can be used to iterate over the project structure. The `iterators` argument is optional, and if not provided, the project object will assume that the project follows a specific structure.

**Without iterators + with glob:**

```python
project = Project(
    DAQFile="*.csv",
    BehFile="*.json",
    SLEAPFile="*.h5",
    VideoFile="*.mp4",
    base="/base/path/to/project",
    get_glob= True
    )
```

**With iterators + without glob:**

```python
project = Project(
    DAQFile="DAQOutput.csv",
    BehFile="ExperimentMetadata.json",
    SLEAPFile="SLEAPAnalysis.h5",
    VideoFile="FacialVideo.mp4",
    base="/base/path/to/project",
    iterators={
        "Label 1": "/relative/path/to/week1",
        "Label 2": "/relative/path/to/week2",
        "Label 3": "/relative/path/to/week3",
        "Label 4": "/relative/path/to/week4",
        ...
        }
    }
    )
```

> The `iterators` paths are relative to the `base` path. If no iterators are provided, the project object will assume that the project follows the default structure outlined below. The default labels then are `"week 1", "week 2", "week 3", "week 4", ...`.

#### Building the data

The project object has a `buildColumns` method, which is used to build the columns of the data. The `buildColumns` method takes two arguments: `columns` and `values`. The `columns` argument is a list of column titles, and the `values` argument is a list of values for each column. The `buildColumns` method will build the columns of the data, and will also build the iterators for each column. The `buildColumns` method can be called multiple times, and each time it is called, it will add a new column to the data.

```python
project.buildColumns(
    columns=["Mouse"],
    values=["CSE008"]
    )
```

#### Building the trials

The project object has a `buildTrials` method, which is used to build the trials of the data. The `buildTrials` method takes four arguments: `TrackedData`, `Reduced`, `start_buffer`, and `end_buffer`. The `TrackedData` argument is a list of the tracked data columns from the DAQ file that will be used to build the based on the DAQ timestamps. The `Reduced` argument is a list of boolean values, which indicate whether the tracked data should be reduced to a single value for each trial. This is used in this example for the `"LED590_on"` column, as the LED turns on and off multiple times during the experiment. To keep the `trialData` from intitializing a new trial for each timestamp in the `"LED590_on"` column, reduced is set to true. The `start_buffer` argument is the *number of milliseconds before* the **trial start time** that should be included in the trial. The `end_buffer` argument is the *number of milliseconds after* the **trial start time** that should be included in the trial. The `buildTrials` method will build the trials of the data, and will also build the iterators for each trial. The `buildTrials` method can be called multiple times, and each time it is called, it will rebuild the trials of the data. This is useful if you want to build the trials based on different tracked data columns, or if you want to change the start and end buffers.

```python
project.buildTrials(
    TrackedData=["Speaker_on", "LED590_on"],
    Reduced=[False, True],
    start_buffer=10000,
    end_buffer=13000
    )
```

#### Normalizing the data

So far there are two normalization methods available:

- Mean Centering

The `meanCenter` method will iterate recursively to the smallest data frame in the project, and then mean-center its way up the project tree. More precisely, the `meanCenter` method first mean centers each trial from each experiment. Then, it will `meanCenter` each experiment in the project. Finally, it will `meanCenter` across all the experiments in the project. This method will mean-center the data in place, and will not return anything. However, it does expose a `all_data` attribute, which is a `pd.DataFrame` of all the data in the project.

```python
project.meanCenter()
```

- Z-Scoring

The `zScore` method assumes the data has already been mean-centered, and will further z-score the data across all the experiments in the project.

The `zScore` method will iterate recursively to the smallest data frame in the project, and then z-score its way up the project tree. This method will z-score the data in place, and will not return anything.

```python
project.zScore()
```

**For convenience, you can also normalize the data in one step:**

```python
project.analyze()
```

----

### Complete Example

**This implementation of the data objects is roughly 10x faster than the scripted version of the data analysis. The full example is shown below:**

```python
from sleapyfaces.project import Project

project = Project(
    DAQFile="*.csv",
    BehFile="*.json",
    SLEAPFile="*.h5",
    VideoFile="*.mp4",
    base="/base/path/to/project",
    get_glob= True
    )

project.buildColumns(
    columns=["Mouse"],
    values=["CSE008"]
    )

project.buildTrials(
    TrackedData=["Speaker_on", "LED590_on"],
    Reduced=[False, True],
    start_buffer=10000,
    end_buffer=13000
    )

project.analyze()

project.visualize()

data = project.pcas["pca3d"]

```

----
## Project Structure
----

### Data Files Formats

#### DAQ file

This file is the output of the DAQ collection, which is used to record the TTL pulses from all of the stimuli used in the experiment. The DAQ file is usually a CSV file, which contains the timestamps for each TTL pulse, and a column for each TTL pulse being tracked. These timestamps are all relative to the experiment start time and are in milliseconds. The TTL pulses are named according to the following convention:

- `Stimuli_on` - The TTL pulse for when the stimuli starts
- `Stimuli_on` - The TTL pulse for when the stimuli ends

#### Behavioral metadata file

This file is the output of the experimental parameters output from [`bruker_control`](https://github.com/Tyelab/bruker_control) as a JSON file. This file contains the experimental parameters for the experiment, including the trial types used, the number of trials, and the inter-trial interval. `SLEAPyFaces` assumed that the file follows the following structure:

```json
{
    "beh_metadata": {
        ...
        "any_keys": "any_values",
        ...
        "trialArray": [1, 1, 1, 0, 1, 0, 1, 1, ...],
        "ITIArray": [16848, 23012, 25678, 19107, ...],
        ...
        "any_keys": "any_values",
        ...
    }
}
```

Note that this is the default output from `bruker_control` and does not need to be modified.

#### SLEAP file

The `SLEAP` file is the output of the `sleap-convert` command from the `SLEAP` software package, which is used to track the facial expressions of the mice. The `SLEAP` file is a HDF5 file, which contains the tracking data for each frame of the video using a common skeleton across all experiments. The `SLEAP` file also contains the metadata for the experiment, which includes the tracking accuracy, the number of frames in the video, and the number of mice in the video. The `SLEAP` analysis file can be obtained from the following command:

```console
sleap-convert --format analysis -o /path/to/output.h5 /path/to/input.slp
```

**or**

```console
sleap-convert --format h5 -o /path/to/output.h5 /path/to/input.slp
```

Further documentation on this command is available on [this `SLEAP` documentation page](https://sleap.ai/guides/cli.html#sleap-convert).


#### Video file

The video file is the raw video file that was used to generate the facial trackings. The video file is used to extract the frame rate of the video, which is used to accurately calculate the timestamps and append them to the `SLEAP` data. The video file can be any video format that is supported by `ffmpeg`, which includes `.mp4`, `.avi`, `.mov`, `.mkv`, and many more.

----

### Project Directory Structure

```
/base/path/to/project
    │
    └───20231208
    │   │   DAQOutput.csv
    │   │   ExperimentMetadata.json
    |   |   SLEAPAnalysis.h5
    |   |   FacialVideo.mp4
    │
    └───20231215
    |   │   DAQOutput.csv
    │   │   ExperimentMetadata.json
    |   |   SLEAPAnalysis.h5
    |   |   FacialVideo.mp4
    |
    └─── ...
```

Note that the files do not have to be named as shown above, but the file extensions must be correct for the DAQ file as `*.csv`, the metadata file as `*.json`, and the `SLEAP` analysis file as `*.h5`. The video file can be any video format that is supported by `ffmpeg`.

By default, the sub-directories can be named anything, as long as the date proceeds any other naming. The dates must be in the format `YYYYMMDD`, so the experiments can be sorted by date.

> e.g. `20231208_Exp1` or `20231208_Exp2` are valid file names, but `Exp1_20231208` is not

However, this functionality can be disabled by specifying the sub-directory paths and their respective labels in the `iterators` argument of the `Project` object. For example, if the project directory structure is as follows:

```
/base/path/to/project
    │
    └───Day1
    |   |
    |   └───Exp1
    │   │   |   DAQOutput.csv
    │   │   |   ExperimentMetadata.json
    |   |   |   SLEAPAnalysis.h5
    |   |   |   FacialVideo.mp4
    |   |
    |   └───Exp2
    │   │   |   DAQOutput.csv
    │   │   |   ExperimentMetadata.json
    |   |   |   SLEAPAnalysis.h5
    |   |   |   FacialVideo.mp4
    │
    └───Day2
    |   |
    |   └───Exp1
    │   │   |   DAQOutput.csv
    │   │   |   ExperimentMetadata.json
    |   |   |   SLEAPAnalysis.h5
    |   |   |   FacialVideo.mp4
    |   |
    |   └───Exp2
    │   │   |   DAQOutput.csv
    │   │   |   ExperimentMetadata.json
    |   |   |   SLEAPAnalysis.h5
    |   |   |   FacialVideo.mp4
    |
    └─── ...
```

Then the `Project` object can be initialized as follows:

```python
proj = Project(
    DAQFile="DAQOutput.csv",
    BehFile="ExperimentMetadata.json",
    SLEAPFile="SLEAPAnalysis.h5",
    VideoFile="FacialVideo.mp4",
    base="/base/path/to/project",
    iterators={
        "Day1-Exp1": "/Day1/Exp1",
        "Day1-Exp2": "/Day1/Exp2",
        "Day2-Exp1": "/Day2/Exp1",
        "Day2-Exp2": "/Day2/Exp2",
        ...
        }
    }
    )
```

----
## API Documentation
----

### `sleapyfaces.project.Project`

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


#### `sleapyfaces.project.Project.buildColumns`

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


#### `sleapyfaces.project.Project.buildTrials`

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


#### `sleapyfaces.project.Project.meanCenter`

```python
proj.meanCenter()
```

- Recursively mean centers the data for each trial for each experiment
  - Initializes attributes:
    - `all_data` (`pd.DataFrame`): the mean centered data for all trials and experiments concatenated together


#### `sleapyfaces.project.Project.zScore`

```python
proj.zScore()
```

- Z scores the mean centered data for each experiment
  - Updates attributes:
    - `all_data` (`pd.DataFrame`): the z-scored data for all experiments concatenated together


#### `sleapyfaces.project.Project.analyze`

```python
proj.analyze()
```

- Runs the mean centering and z scoring functions


#### `sleapyfaces.project.Project.visualize`

```python
proj.visualize()
```

- Reduces `all_data` to 2 and 3 dimensions using PCA
  - Initializes attributes:
    - `pcas` (dictionary[string, `pd.DataFrame`]): a dictionary containing the 2 and 3 dimensional PCA data for each experiment
    - the dictionary keys are `'pca2d'`, `'pca3d'`

----

### `sleapyfaces.experiment.Experiment`

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


#### `sleapyfaces.experiment.Experiment.buildData`

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


#### `sleapyfaces.experiment.Experiment.append`

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


#### `sleapyfaces.experiment.Experiment.buildTrials`

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

### `sleapyfaces.io.DAQData`

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


#### `sleapyfaces.io.DAQData.append`

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


#### `sleapyfaces.io.DAQData.saveData`

```python
daq.saveData(
    filename: str | PathLike[str] | FileIO,
    )
```
- saves the cached data to a csv file
  - Args:
    - `filename` (Text | PathLike[Text] | BufferedWriter): the path and name of the file to save the data to

----

### `sleapyfaces.io.SLEAPanalysis`

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

#### `sleapyfaces.io.SLEAPanalysis.getDatasets`

```python
sleap.getDatasets()
```
- gets the datasets from the `SLEAP` analysis file
  - Initializes Attributes:
    - `data` (dictionary): dictionary of all the data from the `SLEAP` analysis file

#### `sleapyfaces.io.SLEAPanalysis.getTracks`

```python
sleap.getTracks()
```
- gets the tracks from the `SLEAP` analysis file
  - Initializes Attributes:
    - `tracks` (`pd.DataFrame`): a pandas DataFrame containing the tracks from the `SLEAP` analysis file (with missing frames filled in using a linear interpolation method)

#### `sleapyfaces.io.SLEAPanalysis.getTrackNames`

```python
sleap.getTrackNames()
```
- gets the track names from the `SLEAP` analysis file
  - Initializes Attributes:
    - `track_names` (List): list of the track names from the `SLEAP` analysis file

#### `sleapyfaces.io.SLEAPanalysis.append`

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

#### `sleapyfaces.io.SLEAPanalysis.saveData`

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

### `sleapyfaces.io.BehMetadata`

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

#### `sleapyfaces.io.BehMetadata.saveData`

```python
beh.saveData(
    filename: str | PathLike[str] | FileIO,
    )
```
- Saves the DAQ data to a *CSV* file.
  - Args:
    - `filename` (string or PathLike[string] or FileIO): The name and path of the file to save the data to.

----

### `sleapyfaces.io.VideoMetadata`

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

#### `sleapyfaces.io.VideoMetadata.saveData`

```python
video.saveData(
    filename: str | PathLike[str] | FileIO,
    )
```
- Saves the video metadata to a json file.
  - Args:
    - `filename` (string or PathLike[string] or FileIO): the name and path of the file to save the data to.

----

### `sleapyfaces.normalize.mean_center`

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

### `sleapyfaces.normalize.z_score`

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

### `sleapyfaces.normalize.pca`

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

### `sleapyfaces.structs.File`

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

### `sleapyfaces.structs.FileConstructor`

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

### `sleapyfaces.structs.CustomColumn`

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

#### `sleapyfaces.structs.CustomColumn.buildColumn`

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

----

## Changelog:

### Version 1.0.0 (2022-12-27)
1. iterate (repeatedly) over each mouse and each week (each mouse and each experiment)
    - [x] get project files (experimental) structure
    - [x] initialize an iterator over the project structure
2. get daq data from CSV file
    - [x] read CSV files
    - [x] save each column from CSV file
        * Note: CSV columns are of differing lengths
3. get “beh_metadata” from json metadata
    - [x] read JSON file
    - [x] grab the values for key “beh_metadata”
        - [x] get the values of sub key “trialArray”
        - [x] get the values of sub-key “ITIArray”
4. get video metadata from *.mp4 file (with ffmpeg.probe)
    - [x] read in the *.mp4 metadata
    - [x] select the correct video stream
    - [x] get the average frames per second
5. get SLEAP data from *.h5 file
    - [x] open h5 file
    - [x] get transposed values of key “tracks” (tracking_locations)
    - [x] fill missing locations (linear regress. fit)
    - [x] get transposed values of key “edge_inds”
    - [x] get values of key “edge_names”
    - [x] get transposed values of “instance_scores”
    - [x] get transposed values of “point_scores”
    - [x] get values of “track_occupancy”
    - [x] get transposed values of “tracking_scores”
    - [x] get decoded values of “node_names” (make sure there's no encoding issues)
6. deconstruct SLEAP points into x and y points (across all frames)
    - [x] iterate over each node
    - [x] breakup the 4D array “tracks” into 1D array for x and y values respectively
        * Note: [frame, node, x/y, color] for greyscale the color dimension is 1D (i.e. essentially the 4D array is 3D because the color dimension is constant)
    - [x] iterate over each frame
    - [x] assign mouse, week, frame #, and timestamp (using average frames per second)
7. Split data into individual trials by trial type using the Speaker and LED data from the CSV daq data
    - [x] initialize trial iterators for the consistently documented points from the daq CSV
    - [x] iterate over each trial in “trialArray”
    - [x] get the index of 10sec before and 13sec after trial start
    - [x] for each feature, grab the start and end indices
    - [x] store data from each trial in a pd.dataframe
    - [x] concatenate all pd.dataframes together for each video
    - [x] concatenate the pd.dataframes from each video together for each mouse (base expr split)
8. Prepare the data
    - [x] (opt.) mean center across all points for a single trial
    - [x] mean center across all trials for a single experiment
    - [x] mean center across all experiments for a single mouse
    - [x] mean center across all mice
    - [x] (opt.) z-score mean-centered data
9. Analyze the data
    - [x] Perform 2D and 3D PCAs on all data (raw, centered, by trial, by week, by mouse, etc…)
    - [x] apply gaussian kernel to PCA outputs
10. Save the data
    - [x] write everything to HDF5 file(s)
### Version 1.0.1
  - [x] add exhaustive documentation
  - [x] add inline documentation
  - [x] strengthen type hints
### Version 1.1.0 (in progress)
  - [ ] add support for multiple mice
  - [ ] add clustering/prediction algorithm(s)
  - [ ] add velocity, acceleration, and jerk calculations
  - [ ] add save option for all data
  - [ ] add plotting functions
  - [ ] clustering features
      - [ ] distance to a point
          - [ ] vector to a point (theta, magnitude) or (angle, distance)
      - [ ] velocity/acceleration
      - [ ] distance to centroid
      - [ ] distance between given points
