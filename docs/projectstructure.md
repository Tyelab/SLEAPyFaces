# Project Structure

----

## Table of Contents

- [Project Structure](#project-structure)
	- [Table of Contents](#table-of-contents)
	- [Data Files Formats](#data-files-formats)
		- [DAQ file](#daq-file)
		- [Behavioral metadata file](#behavioral-metadata-file)
		- [SLEAP file](#sleap-file)
		- [Video file](#video-file)
	- [Project Directory Structure](#project-directory-structure)

## Data Files Formats

### DAQ file

This file is the output of the DAQ collection, which is used to record the TTL pulses from all of the stimuli used in the experiment. The DAQ file is usually a CSV file, which contains the timestamps for each TTL pulse, and a column for each TTL pulse being tracked. These timestamps are all relative to the experiment start time and are in milliseconds. The TTL pulses are named according to the following convention:

- `Stimuli_on` - The TTL pulse for when the stimuli starts
- `Stimuli_on` - The TTL pulse for when the stimuli ends

### Behavioral metadata file

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

### SLEAP file

The `SLEAP` file is the output of the `sleap-convert` command from the `SLEAP` software package, which is used to track the facial expressions of the mice. The `SLEAP` file is a HDF5 file, which contains the tracking data for each frame of the video using a common skeleton across all experiments. The `SLEAP` file also contains the metadata for the experiment, which includes the tracking accuracy, the number of frames in the video, and the number of mice in the video. The `SLEAP` analysis file can be obtained from the following command:

```console
sleap-convert --format analysis -o /path/to/output.h5 /path/to/input.slp
```

**or**

```console
sleap-convert --format h5 -o /path/to/output.h5 /path/to/input.slp
```

Further documentation on this command is available on [this `SLEAP` documentation page](https://sleap.ai/guides/cli.html#sleap-convert).


### Video file

The video file is the raw video file that was used to generate the facial trackings. The video file is used to extract the frame rate of the video, which is used to accurately calculate the timestamps and append them to the `SLEAP` data. The video file can be any video format that is supported by `ffmpeg`, which includes `.mp4`, `.avi`, `.mov`, `.mkv`, and many more.

----

## Project Directory Structure

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
