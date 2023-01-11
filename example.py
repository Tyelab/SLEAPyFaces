from sleapyfaces.project import Project
from sleapyfaces.utils.normalize import mean_center
import plotly.express as px
import pandas as pd

proj = Project(
    DAQFile="*.csv",
    BehFile="*.json",
    SLEAPFile="*.h5",
    VideoFile="*.mp4",
    base="/base/path/to/project",
    get_glob= True
    )
proj.buildColumns(
    columns=["Mouse"],
    values=["CSE008"])
proj.buildTrials(
    TrackedData=["Speaker_on", "LED590_on"],
    Reduced=[False, True],
    start_buffer=10000,
    end_buffer=13000)
proj.analyze()
proj.visualize()

data = proj.pcas["pca3d"]
fig = px.scatter_3d(data, x="principal component 1", y="principal component 2", z="principal component 3", color="Mouse")
fig.show()
