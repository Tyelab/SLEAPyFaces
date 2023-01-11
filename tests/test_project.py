from sleapyfaces.project import Project
from sleapyfaces.clustering import ClusterData
import pandas as pd

noGlob = Project(
    base="/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data",
    DAQFile=("DAQOutput.csv", False),
    BehFile=("BehMetadata.json", False),
    SLEAPFile=("SLEAP.h5", False),
    VideoFile=("video.mp4", False)
)

withGlob = Project(
    base="/Users/annieehler/Projects/Jupyter_Notebooks/SLEAPyFaces/tests/data",
    iterator={"week 1": "20211105", "week 2": "20211112"},
    DAQFile=("*.csv", True),
    BehFile=("*.json", True),
    SLEAPFile=("*.h5", True),
    VideoFile=("*.mp4", True),
)

def test_project():

    assert noGlob.exprs["week 1"].sleap.tracks.equals(withGlob.exprs["week 1"].sleap.tracks)
    assert noGlob.exprs["week 1"].sleap.path == noGlob.exprs["week 1"].files.sleap.file
    assert withGlob.exprs["week 1"].sleap.path == withGlob.exprs["week 1"].files.sleap.file
    assert noGlob.exprs["week 1"].sleap.path == withGlob.exprs["week 1"].sleap.path

    noGlob.buildColumns(["Mouse"], ["CSE008"])
    withGlob.buildColumns(["Mouse"], ["CSE008"])

    assert noGlob.all_data.equals(withGlob.all_data)

    noGlob.buildTrials(["Speaker_on", "LED590_on"], [False, True])
    withGlob.buildTrials(["Speaker_on", "LED590_on"], [False, True])

    assert len(noGlob.exprs["week 1"].trialData) == len(withGlob.exprs["week 1"].trialData)
    assert noGlob.exprs["week 1"].trials.equals(withGlob.exprs["week 1"].trials)

    noGlob.meanCenter()
    noGlob.zScore()
    noGlob.runPCA()

    withGlob.normalize()
    withGlob.runPCA()

    assert type(noGlob.pcas["pca2d"]) is pd.DataFrame


def test_clustering():
    withGlob.buildColumns(["Mouse"], ["CSE008"])
    withGlob.buildTrials(["Speaker_on", "LED590_on"], [False, True])
    withGlob.normalize()
    cd = ClusterData(withGlob.all_data, withGlob.numeric_columns)
    assert len(withGlob.all_data.columns) == len(cd.data.columns)
