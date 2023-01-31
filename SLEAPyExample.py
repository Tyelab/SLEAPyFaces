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

# %%
from sleapyfaces.base import Experiment

expr = Experiment(
    name="SLEAPyExample",
    base="/Volumes/specialk_cs/2p/raw/CSE011/20211105/",
    ExperimentEventsFile=("*_events.csv", True),
    ExperimentSetupFile=("*.json", True),
    SLEAPFile=("*.h5", True),
    VideoFile=("*.mp4", True),
)

# %%
