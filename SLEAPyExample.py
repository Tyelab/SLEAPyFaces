# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: sleapyfaces
#     language: python
#     name: python3
# ---

# %% [markdown]
# SLEAPyFaces Example
# ===================
# %% [markdown]
This notebook is an example of how to use the SLEAPyFaces package to analyze SLEAP data. It is intended to be run in a Jupyter notebook, but can be run in a Python script as well. The notebook is available on GitHub at [annie444/SLEAPyFaces](https://github.com/annie444/SLEAPyFaces)

# %% [markdown]
There are three main base types in SLEAPyFaces: 
1. `Experiment` - a single experiment, which contains a single SLEAP file, a single video file, and a single events file. 
2. `Project` - a collection of experiments, which can be used to compare multiple experiments.
3. `Projects` - a collection of projects, which can be used to compare multiple projects. (and their inherited experiments)

# %% [markdown]
## Setup 
# %% [markdown]
For a single `Experiment` analysis, you will need to have the following files in the same directory: 
1. A SLEAP file (`.h5`), which contains the SLEAP tracking data. 
2. A video file (`.mp4`), which contains the video data. 
3. An events file (`.csv`), which contains the events data. 
4. An experiment setup file (`.json`), which contains the experiment setup/metadata. 

# %%
from sleapyfaces.base import Experiment

expr = Experiment(
    name="SLEAPyExampleExperiment", # first we give our experiment a name 
    base="/Volumes/specialk_cs/2p/raw/CSE011/20211105/", # then we assign the base directory, in this case it is on `specialk_cs/2p/raw/CSE011/20211105`
    ExperimentEventsFile=("*_events.csv", True), # then we assign the events file, which is a csv file
    ExperimentSetupFile=("*.json", True), # then we assign the experiment setup file, which is a json file 
    SLEAPFile=("*.h5", True), # then we assign the SLEAP file, which is a h5 file 
    VideoFile=("*.mp4", True), # then we assign the video file, which is a mp4 file 
) 

# For any of the above files, you can also just pass the filename (not in parentases). 
# However, byt passing a tuple with the 'naming convention' and a boolean, you can use wildcards `*` to find the file based on a consistent naming scheme. 
# This is less important for individual experiments, but is very useful for `Project` and `Projects` analysis. 

# %% [markdown]
For a single `Project` analysis, you will need to have multiple folders containing all the necessary `Experiment` files.
# %%
from sleapyfaces.base import Project

proj = Project(
    ExperimentEventsFile=("*_events.csv", True), # we assign the events file (which is a csv file)
    ExperimentSetupFile=("*.json", True), # we assign the experiment setup file (which is a json file)
    SLEAPFile=("*.h5", True), # we assign the SLEAP file (which is a h5 file)
    VideoFile=("*.mp4", True), # we assign the video file (which is a mp4 file)
    base="/Volumes/specialk_cs/2p/raw/CSE011/", # we assign the base directory, in this case it is on `specialk_cs/2p/raw/CSE011` 
    # NOTE THIS IS DIFFERENT FROM THE ABOVE EXAMPLE
    name="SLEAPyExampleProject", # we give our project a name
)

# %% [markdown]
For a single `Projects` analysis, you will need to have multiple folders containing all the necessary `Project` files. 
However, the `Projects` class will automatically search the direct subdirectories for `Experiment` files.
This results in an `Experiment` for each subdirectory. If the subdirectoy names contain numbers, they will also automatically be sorted and labeled in ascending order. 
# %%
from sleapyfaces import Projects

projs = Projects(
    ExperimentEventsFile=("*_events.csv", True), # we assign the events file (which is a csv file)
    ExperimentSetupFile=("*.json", True), # we assign the experiment setup file (which is a json file)
    SLEAPFile=("*.h5", True), # we assign the SLEAP file (which is a h5 file)
    VideoFile=("*.mp4", True), # we assign the video file (which is a mp4 file)
    base="/Volumes/specialk_cs/2p/raw/", # we assign the base directory, in this case it is on `specialk_cs/2p/raw`
    file_structure={
        "CSE011": "CSE011",
        "CSE014": "CSE014",
        "CSE016": "CSE016",
        "CSC011": "CSC011",
        "CSE020": "CSE020",
    }) # we assign the file structure, which is a dictionary of the subdirectories and their names

# %% [markdown]
## Data Manipulation
# %% [markdown]
> Moving forward I am going to use all three objects to show how they can be used in different scenarios. 

So far we've only imported all of the data and stored them in pandas dataframes. Now we have to do some data manipulation to get the data into the correct format for analysis. 
The first thing in this process is adding the correct annotation columns to the data and initializing it all together. 
# %%

expr.buildColumns() # `.buildColumns()` is a method on all base obejcets, and it can be empty. There is no need to add annotations if there aren't any to add, but this is still a necessary method to call as it initializes the data into a main dataframe.

proj.buildColumns(["Mouse_Type"], ["Resilient"]) # If you choose to annotate the data, the `.buildColumns()` method takes a list of column names and a list of their corresponding values. The values will be added to the column names for each experiment in the project and can be referenced as needed. 

# for the `Projects` object we're wanting to label different projects with a different `Mouse_Type`
for name, project in projs.data.items(): # so we can loop through all of the project data items, which give the name (from the `file_structure`) and the `Project` object for the corresponding project 
    if name in ["CSE011", "CSE014", "CSE016"]: # This allows us to run a fast comparison of the project names to the names we want to label 
        project.buildColumns(["Mouse_Type"], ["Resilient"]) # and then we can build the columns for each project, accessing the `Project` object directly only once per project 
    elif name in ["CSE020"]:
        project.buildColumns(["Mouse_Type"], ["Susceptible"])
    else:
        project.buildColumns(["Mouse_Type"], ["Control"])

# %% [markdown]
> Now that we have build our data, it's now time to align the video data with the event data and the setup data to extract the idividual trials. 
> While aligning timeseries data is a computationally expensive operation, it is necessary to get the data into the correct format for analysis. 
> In lieu of this, the following command utilized `multiprocessing` which is known to be tempermental, so don't mess with the code when executing the following lines. 
# %% [markdown]
From here on out I'm just going to use the `Experiemnt` object, as the next steps are the same across all objects. This is to minimize the amount of time lost to computations in this example. 
However, all of the information here in can be extrapolated to the `Project` and `Projects` objects.
# %% 

# In order to build the trials, we have to pass in two lists. 

# The first list is the column names from the `ExperimentEventsFile` that dictate the start of a trial from the events (or DAQ) data. 
cols = ["Speaker_on", "LED_on"]

# The second is a list of boolean values that let the code know if there are multiple events for each column in the trial period. 
# e.g. For Austin's data, the LED will blink repeatedly during a trial, but we don't want to initialize each blink as a new trial, so we set that list to True for trial reduction. 
# Conversely, the speaker will only turn on once per trial, so we set that list to False for trial reduction. 
reduce = [False, True]

# Now we can build the trials. 
expr.buildTrials(cols, reduce)

# NOTE: These lists can be passed directly, however, for the sake of this example they were seperated out. 

# %% [markdown]
## Normalization 
# %% [markdown]
> Now that we have our data in the correct format, we can start to analyzing it.
# %%

expr.normalize() # This will normalize the data for each trial. 

# %% [markdown]
## Visualizing the Data 
# %%

expr.visualize(dimensions=3, normalized=True, filename=".ignore/3DExample.html") # This will visualize the data in 3D and output the visualization as an interactive plot. 

# %% [markdown]

