#%%
from sleapyfaces.project import Project
from sleapyfaces.normalize import mean_center
import plotly.express as px
import pandas as pd

#%%
proj = Project("*.csv", "*.json", "*.h5", "*.mp4", "/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data", get_glob= True)
proj.buildColumns(["Mouse"], ["CSE008"])
proj.buildTrials(["Speaker_on", "LED590_on"], [False, True])
proj.analyze()
proj.visualize()

#%%
data = proj.pcas["pca3d"]

fig = px.scatter_3d(data, x="principal component 1", y="principal component 2", z="principal component 3", color="Mouse")

fig.show()
