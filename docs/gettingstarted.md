# Getting Started

----

## Table of Contents

- [Getting Started](#getting-started)
	- [Table of Contents](#table-of-contents)
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

## Installation

### Using pip:

```console
pip install sleapyfaces
```

### Using conda (coming soon):

```console
conda install -c conda-forge sleapyfaces
```
-----

## Basic Usage Step by Step

### Importing the package

For **basic** usage, the package can be imported with the following:

```python
from sleapyfaces.project import Project
```

Alternatively, the package can be imported with the following:

```python
import sleapyfaces as sf
```

### Creating a project

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

### Building the data

The project object has a `buildColumns` method, which is used to build the columns of the data. The `buildColumns` method takes two arguments: `columns` and `values`. The `columns` argument is a list of column titles, and the `values` argument is a list of values for each column. The `buildColumns` method will build the columns of the data, and will also build the iterators for each column. The `buildColumns` method can be called multiple times, and each time it is called, it will add a new column to the data.

```python
project.buildColumns(
    columns=["Mouse"],
    values=["CSE008"]
    )
```

### Building the trials

The project object has a `buildTrials` method, which is used to build the trials of the data. The `buildTrials` method takes four arguments: `TrackedData`, `Reduced`, `start_buffer`, and `end_buffer`. The `TrackedData` argument is a list of the tracked data columns from the DAQ file that will be used to build the based on the DAQ timestamps. The `Reduced` argument is a list of boolean values, which indicate whether the tracked data should be reduced to a single value for each trial. This is used in this example for the `"LED590_on"` column, as the LED turns on and off multiple times during the experiment. To keep the `trialData` from intitializing a new trial for each timestamp in the `"LED590_on"` column, reduced is set to true. The `start_buffer` argument is the *number of milliseconds before* the **trial start time** that should be included in the trial. The `end_buffer` argument is the *number of milliseconds after* the **trial start time** that should be included in the trial. The `buildTrials` method will build the trials of the data, and will also build the iterators for each trial. The `buildTrials` method can be called multiple times, and each time it is called, it will rebuild the trials of the data. This is useful if you want to build the trials based on different tracked data columns, or if you want to change the start and end buffers.

```python
project.buildTrials(
    TrackedData=["Speaker_on", "LED590_on"],
    Reduced=[False, True],
    start_buffer=10000,
    end_buffer=13000
    )
```

### Normalizing the data

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

## Complete Example

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
